"""Template campaign sweep: full menu at every noise level; persist everything."""
import sys, json, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.physics.templates import TemplateFit, fit_ladder, ladder_starts, select_model
from spevae.losses import spectral_w1
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
NOISES = [1e-2]
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0,1.0,ntau+1)
model = load_model("out/microsweep/id_ladder_full.pt")
import json as _j
store = _j.load(open("out/microsweep/template_sweep.json"))
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
        sig_net = sigma.clamp_min(3e-5 * float(Gc.abs().max()))
        with torch.no_grad():
            out = model(G, x, beta, sigma=sig_net, deterministic=True)
        amort = out.poles.detach().to_physical(beta)
        st = ladder_starts(amort); st_m = ladder_starts(amort, mirrored=True)
        menu = []
        for m in (2, 3, 4, 5):
            menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m, starts=st))
        for m in (1, 2, 3):
            menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m,
                                   mirrored=True, starts=st_m, steps=4000))
        free, tr = rank_scan(amort, beta, x[None], G, sigma, inv_sqrt_R=isr)
        menu.append(TemplateFit(name="free", m=tr.selected_n,
                                k_params=4*tr.selected_n, chi2=tr.chi2_selected,
                                poles=free, params={}))
        winner = select_model(menu, L_eff=ntau+1)
        As = winner.poles.to_dimensionless(beta).spectral(quad.nodes)
        w1w = float(spectral_w1(As, At, quad, beta=beta))
        store["results"].append(dict(
            case=name, noise=nz, n_true=len(eps),
            winner=dict(name=winner.name, m=winner.m, k=winner.k_params,
                        chi2=winner.chi2, w1=w1w, params=winner.params,
                        A=(As[0]*beta).tolist(),
                        pole_eps=winner.poles.eps.reshape(-1).tolist(),
                        pole_a=winner.poles.a.reshape(-1).tolist()),
            truth_A=(At[0]*beta).tolist(),
            menu=[dict(name=c.name, m=c.m, k=c.k_params, chi2=c.chi2)
                  for c in menu]))
        print(f"{name} {nz:.0e}: {winner.name}({winner.m}) chi2 {winner.chi2:.3f} "
              f"W1 {w1w:.5f}", flush=True)
json.dump(store, open("out/microsweep/template_sweep.json", "w"))
print("DONE")
