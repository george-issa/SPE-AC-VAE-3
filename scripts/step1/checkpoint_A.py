"""Real-data checkpoint A: beta=8 Holstein cell, full pipeline, arm-C model."""
import sys, json, time, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.data.readers import read_gissa_cube
from spevae.physics.forward import SpectralQuadrature
from spevae.physics.consolidate import rank_scan
from spevae.refine import RefineConfig, refine_latent
from spevae.physics.templates import (TemplateFit, fit_ladder, ladder_starts,
                                      fit_peaks, peak_starts, fit_edged_band,
                                      fit_mirrored_peaks, select_model)

CUBE = "/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/gIssa/for_george_ac.jld2"
DOS = "/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/gIssa/george_DOS.jld2"
beta, omega, nfill, lam = 8.0, 1.0, 1.0, 0.325
try:
    mg = read_gissa_cube(CUBE, beta=beta, omega=omega, n=nfill, lam=lam,
                         dos_path=DOS)
except Exception as e:
    print("dos attach failed:", e)
    mg = read_gissa_cube(CUBE, beta=beta, omega=omega, n=nfill, lam=lam)
print(mg.summary())
cov = mg.covariance(min_eigenvalue=1e-10, method="structured")
print(f"whitener: structured, rho {cov['rho']:.3f}, xi {cov['xi']:.3f}")
# restrict everything to the statistical slices up front: the whitener
# lives on that subset, and the kernel happily evaluates on any tau-subset
sl = mg.statistical_slices
G = mg.mean[sl].unsqueeze(0)
sigma = mg.sigma_mean[sl].unsqueeze(0)
isr = cov["inv_sqrt_R"]
x = mg.x[sl]
slices = None
model = load_model(sys.argv[1])
t0 = time.perf_counter()
rr = refine_latent(model, mg, RefineConfig(cov_method="structured",
                                           min_eigenvalue=1e-10, steps=3000))
print(f"z-refine (MAP): chi2 {rr.chi2_init:.1f} -> {rr.chi2:.2f} "
      f"(target {rr.chi2_target:.3f}, {rr.steps_run} steps)")
# boundary proof: release the prior anchor and give the optimiser room --
# the resulting chi2 is the decoder MANIFOLD's floor on this data, i.e.
# step-1 training's expressiveness boundary, free of regularisation excuses
rb = refine_latent(model, mg, RefineConfig(cov_method="structured",
                                           min_eigenvalue=1e-10, steps=10000,
                                           prior_weight=0.0, patience=3000))
print(f"z-refine (lam=0, 10k): manifold floor chi2 {rb.chi2:.2f} "
      f"(ratio {rb.chi2/rb.chi2_target:.1f}x target)")
amort = rr.poles
free, tr = rank_scan(amort, beta, x[None], G, sigma, inv_sqrt_R=isr,
                     slices=slices)
st = ladder_starts(amort); st_m = ladder_starts(amort, mirrored=True)
menu = []
for m in (2, 3, 4, 5):
    menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m, starts=st))
for m in (1, 2, 3):
    menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m,
                           mirrored=True, starts=st_m, steps=4000))
for np_ in (2, 3, 4):
    st_p = peak_starts(amort, np_)
    if np_ in tr.fits_by_n:
        st_p = st_p + [sorted(tr.fits_by_n[np_].eps.reshape(-1).tolist())]
    menu.append(fit_peaks(beta, x, G, sigma, inv_sqrt_R=isr, n=np_, starts=st_p))
    mp_starts = [sorted(abs(v) for v in st_p[0])]
    if 2 * np_ in tr.fits_by_n:
        fe = tr.fits_by_n[2 * np_].eps.reshape(-1)
        mp_starts.append(sorted(float(v) for v in fe[fe > 0])[:np_] or [1.0] * np_)
    menu.append(fit_mirrored_peaks(beta, x, G, sigma, inv_sqrt_R=isr, n=np_,
                                   starts=[st for st in mp_starts
                                           if len(st) == np_]))
menu.append(fit_edged_band(beta, x, G, sigma, inv_sqrt_R=isr, m=5,
                           starts=[(0.0, 1.0), (0.0, 2.0)]))
menu.append(TemplateFit(name="free", m=tr.selected_n,
                        k_params=4*tr.selected_n - 2,
                        chi2=tr.chi2_selected, poles=free, params={}))
winner = select_model(menu, L_eff=int(x.numel()))
dt = time.perf_counter() - t0
quad = SpectralQuadrature.for_beta(beta)
As = winner.poles.to_dimensionless(beta).spectral(quad.nodes.double())
print(f"\nchi2_target {cov['chi2_target']:.3f}")
print(f"winner: {winner}   wall {dt:.0f}s")
print("menu:")
for c in sorted(menu, key=lambda c: (c["k"] if isinstance(c, dict) else c.k_params, c.chi2)):
    print(f"  {c.name}({c.m}) k={c.k_params} chi2 {c.chi2:.4g}")
print(f"scan: {tr}")
ties = [c for c in menu
        if c is not winner and c.k_params == winner.k_params
        and abs(c.chi2 - winner.chi2) * x.numel() < 5.0]
A_zd = rr.poles.to_dimensionless(beta).spectral(quad.nodes.double())
A_free = free.to_dimensionless(beta).spectral(quad.nodes.double())
art = dict(label=mg.label, chi2_target=float(cov["chi2_target"]),
           manifold_floor=float(rb.chi2),
           zd_chi2=float(rr.chi2),
           A_zdescent=(A_zd[0]*beta).tolist(),
           A_free=(A_free[0]*beta).tolist(),
           ties=[dict(name=c.name, m=c.m, k=c.k_params, chi2=c.chi2,
                      A=(c.poles.to_dimensionless(beta).spectral(
                          quad.nodes.double())[0]*beta).tolist())
                 for c in ties],
           nodes=(quad.nodes/beta).tolist(), A=(As[0]*beta).tolist(),
           winner=dict(name=winner.name, m=winner.m, k=winner.k_params,
                       chi2=winner.chi2, params=winner.params,
                       pole_eps=winner.poles.eps.reshape(-1).tolist(),
                       pole_gam=winner.poles.gamma.reshape(-1).tolist(),
                       pole_a=winner.poles.a.reshape(-1).tolist()),
           menu=[dict(name=c.name, m=c.m, k=c.k_params, chi2=c.chi2) for c in menu],
           scan=dict(ns=tr.ns, chi2s=tr.chi2s, selected_n=tr.selected_n))
if mg.reference is not None:
    art["reference"] = dict(omega=mg.reference[0].tolist(),
                            A=mg.reference[1].tolist())
json.dump(art, open(sys.argv[2], "w"))
print("CHECKPOINT_DONE")
