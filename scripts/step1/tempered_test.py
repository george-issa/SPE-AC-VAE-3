"""Tempered scan on the three cases: w in {0, 4, 16, 64}, ladder to 16."""
import sys, json, torch
sys.path.insert(0, "src")
import numpy as np
from spevae import load_model
from spevae.refine import (z_descend, refine_latent, RefineConfig,
                           prior_count_belief)
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise
from spevae.data.readers import read_gissa_cube

beta = 8.0
WS = [0.0, 4.0, 16.0, 64.0]
quad = SpectralQuadrature.for_beta(beta)
model = load_model("out/step1/s1_metric.pt")
ART = {"nodes": (quad.nodes/beta).tolist(), "cases": []}

def run(name, poles, A_anchor, cc, G, x, sigma, isr, truth):
    def w1_of(ps):
        if truth is None: return None
        return float(spectral_w1(
            ps.to_dimensionless(beta).spectral(quad.nodes.to(torch.float64)).float(),
            truth.spectral(quad.nodes), quad, beta=beta))
    print(f"[{name}] c(n): " + " ".join(f"{v:.1f}" for v in cc), flush=True)
    rows = []
    for w in WS:
        sel, tr = rank_scan(poles, beta, x[None], G.reshape(1,-1),
                            sigma.reshape(1,-1), inv_sqrt_R=isr,
                            anchor_A=A_anchor, anchor_weight=max(w, 1e-9),
                            pre_refit_steps=3000, initial_reduce="drop",
                            n_max=16, capacity_c=cc)
        r = dict(weight=w, n=tr.selected_n, n_LR=tr.n_LR,
                 chi2=tr.chi2_selected, w1=w1_of(sel),
                 ns=tr.ns, chi2s=tr.chi2s,
                 A=(sel.to_dimensionless(beta).spectral(
                     quad.nodes.to(torch.float64))[0]*beta).tolist())
        rows.append(r)
        print(f"[{name}] w={w:>4}: n={r['n']} (LR {r['n_LR']}) "
              f"chi2 {r['chi2']:.3f} w1 {r['w1']}", flush=True)
    ART["cases"].append(dict(name=name, rows=rows,
                             c=cc,
                             truth_A=None if truth is None else
                             (truth.spectral(quad.nodes)[0]*beta).tolist()))

def mk(eps, gam, a, t):
    a_, t_ = np.array(a), np.array(t)
    b = a_ * (t_ - (a_ * t_).sum() / a_.sum())
    return PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                   a=torch.tensor([a]), b=torch.tensor([b.tolist()]))
ks = np.linspace(-1, 1, 5)
for name, truth in [("fano band", mk((0.3+ks*1.2).tolist(), [0.5]*5, [0.2]*5,
                                     np.clip(0.7*ks, -0.45, 0.45).tolist())),
                    ("asym triplet", mk([0.4,1.1,2.6],[0.35,0.25,0.45],
                                        [0.5,0.3,0.2],[0.35,-0.4,-0.25]))]:
    x = torch.linspace(0.0, 1.0, 65)
    Gc = PoleGreens(quad)(truth, x)
    n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=1e-4, xi=0.1),
                                          torch.Generator().manual_seed(7))
    G = Gc + n
    ev, V = torch.linalg.eigh(R)
    isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    zr = z_descend(model, G, x, beta, sigma=sigma, inv_sqrt_R=isr, steps=1500)
    A_anchor = zr.poles.to_dimensionless(beta).spectral(quad.nodes.to(torch.float64))
    cc = prior_count_belief(model, G, x, beta, sigma=sigma)
    run(name, zr.poles, A_anchor, cc, G, x, sigma, isr, truth)

CUBE = "/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/gIssa/for_george_ac.jld2"
mg = read_gissa_cube(CUBE, beta=8.0, omega=1.0, n=1.0, lam=0.325)
cov = mg.covariance(1e-10, method="structured")
sl = mg.statistical_slices
rr = refine_latent(model, mg, RefineConfig(cov_method="structured",
                                           min_eigenvalue=1e-10, steps=3000))
A_anchor = rr.poles.to_dimensionless(beta).spectral(quad.nodes.to(torch.float64))
cc = prior_count_belief(model, mg.mean, mg.x, beta, sigma=mg.sigma_encoder)
run("Holstein", rr.poles, A_anchor, cc, mg.mean[sl], mg.x[sl],
    mg.sigma_mean[sl], cov["inv_sqrt_R"], None)
json.dump(ART, open("out/step1/tempered.json", "w"))
print("TEMPERED_DONE")
