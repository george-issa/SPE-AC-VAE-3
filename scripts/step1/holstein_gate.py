import sys, json, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.consolidate import rank_scan
from spevae.data.readers import read_gissa_cube
beta = 8.0
art = json.load(open("out/step1/prior_start.json"))
h = next(c for c in art["cases"] if c["name"] == "Holstein")
# rebuild the w=2 prior-started pole set? not stored as poles -- rerun cheaply
# instead: seed the scan from the stored A? cannot. Recompute via the script's
# path would cost minutes; rather run the scan from the z-descent poles is the
# old path. Simplest honest gate: scan the w=2 result -- rerun that one refit.
from spevae.refine import refine_latent, RefineConfig, manifold_refit
model = load_model("out/step1/s1_metric.pt")
CUBE = "/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/gIssa/for_george_ac.jld2"
mg = read_gissa_cube(CUBE, beta=8.0, omega=1.0, n=1.0, lam=0.325)
cov = mg.covariance(1e-10, method="structured")
sl = mg.statistical_slices
rr = refine_latent(model, mg, RefineConfig(cov_method="structured",
                                           min_eigenvalue=1e-10, steps=3000))
with torch.no_grad():
    _, _, norm = model.encode(mg.mean.unsqueeze(0).float(), mg.x.float(), beta,
                              sigma=mg.sigma_encoder.unsqueeze(0).float())
pf, info = manifold_refit(model, rr.poles, rr.z, norm, mg.mean[sl], mg.x[sl],
                          beta, mg.sigma_mean[sl], inv_sqrt_R=cov["inv_sqrt_R"],
                          weight=2.0, rounds=4, fit_steps=1000, proj_steps=300)
print(f"prior-started w=2: chi2 {info['chi2']:.3f}")
free, tr = rank_scan(pf, beta, mg.x[sl][None], mg.mean[sl].unsqueeze(0),
                     mg.sigma_mean[sl].unsqueeze(0), inv_sqrt_R=cov["inv_sqrt_R"])
print(f"count-gated scan from that basin: {tr}")
quadA = None
from spevae.physics.forward import SpectralQuadrature
quad = SpectralQuadrature.for_beta(beta)
json.dump(dict(scan=dict(ns=tr.ns, chi2s=tr.chi2s, selected_n=tr.selected_n,
                         chi2_selected=tr.chi2_selected),
               A=(free.to_dimensionless(beta).spectral(
                   quad.nodes.to(torch.float64))[0]*beta).tolist()),
          open("out/step1/holstein_gated.json", "w"))
print("GATE_DONE")
