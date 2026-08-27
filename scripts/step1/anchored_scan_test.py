"""Anchored rank scan (chi2 + wL2 refits, prune init, pre-refit) vs the old
scan on fano, asym triplet, Holstein."""
import sys, json, torch
sys.path.insert(0, "src")
import numpy as np
from spevae import load_model
from spevae.refine import z_descend, refine_latent, RefineConfig
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise
from spevae.data.readers import read_gissa_cube

beta = 8.0
quad = SpectralQuadrature.for_beta(beta)
model = load_model("out/step1/s1_metric.pt")
ART = {"nodes": (quad.nodes/beta).tolist(), "cases": []}

def run(name, zr_poles, zr_z_A, G, x, sigma, isr, truth, w):
    def w1_of(ps):
        if truth is None: return None
        return float(spectral_w1(
            ps.to_dimensionless(beta).spectral(quad.nodes.to(torch.float64)).float(),
            truth.spectral(quad.nodes), quad, beta=beta))
    old_free, old_tr = rank_scan(zr_poles, beta, x[None], G.reshape(1,-1),
                                 sigma.reshape(1,-1), inv_sqrt_R=isr)
    new_free, new_tr = rank_scan(zr_poles, beta, x[None], G.reshape(1,-1),
                                 sigma.reshape(1,-1), inv_sqrt_R=isr,
                                 anchor_A=zr_z_A, anchor_weight=w,
                                 pre_refit_steps=3000, initial_reduce="drop")
    print(f"[{name}] old: n={old_tr.selected_n} chi2 {old_tr.chi2_selected:.3f} "
          f"W1 {w1_of(old_free)} | anchored(w={w}): n={new_tr.selected_n} "
          f"chi2 {new_tr.chi2_selected:.3f} W1 {w1_of(new_free)}", flush=True)
    print(f"   old curve: {['%.3g' % c for c in old_tr.chi2s]}")
    print(f"   new curve: {['%.3g' % c for c in new_tr.chi2s]}", flush=True)
    ART["cases"].append(dict(
        name=name, truth_A=None if truth is None else
        (truth.spectral(quad.nodes)[0]*beta).tolist(),
        old=dict(n=old_tr.selected_n, chi2=old_tr.chi2_selected,
                 w1=w1_of(old_free), ns=old_tr.ns, chi2s=old_tr.chi2s,
                 A=(old_free.to_dimensionless(beta).spectral(
                     quad.nodes.to(torch.float64))[0]*beta).tolist()),
        new=dict(n=new_tr.selected_n, chi2=new_tr.chi2_selected,
                 w1=w1_of(new_free), ns=new_tr.ns, chi2s=new_tr.chi2s,
                 A=(new_free.to_dimensionless(beta).spectral(
                     quad.nodes.to(torch.float64))[0]*beta).tolist())))

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
    run(name, zr.poles, A_anchor, G, x, sigma, isr, truth, w=0.5)

CUBE = "/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/gIssa/for_george_ac.jld2"
mg = read_gissa_cube(CUBE, beta=8.0, omega=1.0, n=1.0, lam=0.325)
cov = mg.covariance(1e-10, method="structured")
sl = mg.statistical_slices
rr = refine_latent(model, mg, RefineConfig(cov_method="structured",
                                           min_eigenvalue=1e-10, steps=3000))
A_anchor = rr.poles.to_dimensionless(beta).spectral(quad.nodes.to(torch.float64))
run("Holstein", rr.poles, A_anchor, mg.mean[sl], mg.x[sl], mg.sigma_mean[sl],
    cov["inv_sqrt_R"], None, w=1.0)
json.dump(ART, open("out/step1/anchored_scan.json", "w"))
print("ANCHORED_DONE")
