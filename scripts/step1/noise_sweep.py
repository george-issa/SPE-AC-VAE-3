"""Noise-convergence study: rank-scan recovery vs noise level on the three
ladder probes. Expectation: counts lock in, then parameters converge toward
the unique zero-noise pole set."""
import sys, json, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.losses import spectral_w1, spectral_l2
from spevae.data.noise import NoiseModel, sample_correlated_noise

beta, ntau = 8.0, 64
def ladder(e0, Om, m, g, d):
    e = [e0 + k*Om for k in range(m)]; a = [d**k for k in range(m)]; s = sum(a)
    return e, [g]*m, [v/s for v in a]
e1,g1,a1 = ladder(0.5,1.0,3,0.30,0.55)
e2,g2,a2 = ladder(-0.4,0.8,4,0.25,0.60)
em,gm,am = ladder(1.2,1.0,2,0.30,0.45)
e3 = [-v for v in reversed(em)]+em; g3 = list(reversed(gm))+gm
a3 = [v/2 for v in reversed(am)]+[v/2 for v in am]
CASES = [("3-rung", e1,g1,a1), ("4-rung", e2,g2,a2), ("mirrored", e3,g3,a3)]
NOISES = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0,1.0,ntau+1)
model = load_model("out/microsweep/id_ladder_full.pt")
out_rows = []
for name, eps, gam, w in CASES:
    truth = PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                    a=torch.tensor([w]), b=torch.zeros(1,len(eps)))
    Gc = PoleGreens(quad)(truth, x)
    At = truth.spectral(quad.nodes)
    for nz in NOISES:
        n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=nz, xi=0.1),
                                              torch.Generator().manual_seed(7))
        G = Gc + n
        ev, V = torch.linalg.eigh(R)
        isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
        # The encoder's noise conditioning saw scale >= 3e-5 in training;
        # below that we hand it its nearest in-distribution sigma.  The
        # scan itself always uses the true sigma.
        sig_net = sigma.clamp_min(3e-5 * float(Gc.abs().max()))
        with torch.no_grad():
            out = model(G, x, beta, sigma=sig_net, deterministic=True)
        sel, tr = rank_scan(out.poles.detach().to_physical(beta), beta, x[None],
                            G, sigma, inv_sqrt_R=isr)
        As = sel.to_dimensionless(beta).spectral(quad.nodes)
        w1 = float(spectral_w1(As, At, quad, beta=beta))
        l2 = float(spectral_l2(As, At, quad, relative=True))
        # position error of matched poles (greedy nearest, weight >= 2%)
        pe = sel.eps[0][sel.a[0] > 0.02].tolist()
        derr = max((min(abs(p - t) for p in pe) for t in eps), default=float("nan")) \
            if pe else float("nan")
        row = dict(case=name, noise=nz, n_sel=tr.selected_n, n_true=len(eps),
                   chi2=tr.chi2_selected, w1=w1, l2=l2, max_pos_err=derr)
        out_rows.append(row)
        print(json.dumps(row), flush=True)
json.dump(out_rows, open("out/microsweep/noise_sweep.json", "w"))
print("DONE")
