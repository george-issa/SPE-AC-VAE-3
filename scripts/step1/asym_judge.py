"""b-channel probes: asymmetric triplet + fano band through the extended menu."""
import sys, json, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan, refit_poles_to_greens
from spevae.physics.templates import (TemplateFit, fit_ladder, ladder_starts,
                                      fit_peaks, peak_starts, fit_edged_band,
                                      select_model)
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise
from spevae.refine import z_descend
import numpy as np

beta, ntau, nz = 8.0, 64, 1e-4
def mk(eps, gam, a, t):
    a_, t_ = np.array(a), np.array(t)
    b = a_ * (t_ - (a_ * t_).sum() / a_.sum())
    return (torch.tensor([eps]) * beta, torch.tensor([gam]) * beta,
            torch.tensor([a]), torch.tensor([b.tolist()]))
ks = np.linspace(-1, 1, 5)
PROBES = [
    ("asym triplet", *mk([0.4, 1.1, 2.6], [0.35, 0.25, 0.45], [0.5, 0.3, 0.2],
                         [0.35, -0.4, -0.25]), 3),
    ("fano band", *mk((0.3 + ks * 1.2).tolist(), [0.5]*5, [0.2]*5,
                      np.clip(0.7 * ks, -0.45, 0.45).tolist()), 5),
]
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0, 1.0, ntau + 1)
model = load_model("out/step1/s1_metric.pt")
ART = {"nodes": (quad.nodes / beta).tolist(), "probes": []}
for name, te, tg, ta, tb, n_true in PROBES:
    truth = PoleSet(eps=te, gamma=tg, a=ta, b=tb)
    Gc = PoleGreens(quad)(truth, x)
    n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=nz, xi=0.1),
                                          torch.Generator().manual_seed(7))
    G = Gc + n
    ev, V = torch.linalg.eigh(R)
    isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    zr = z_descend(model, G, x, beta, sigma=sigma, inv_sqrt_R=isr, steps=800)
    amort = zr.poles
    print(f"  z-descent: chi2 {zr.chi2_init:.1f} -> {zr.chi2:.2f} "
          f"({zr.steps_run} steps)")
    free, tr = rank_scan(amort, beta, x[None], G, sigma, inv_sqrt_R=isr)
    st = ladder_starts(amort)
    menu = []
    for m in (2, 3, 4):
        menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m, starts=st))
    for np_ in (2, 3, 4):
        st_p = peak_starts(amort, np_)
        if np_ in tr.fits_by_n:
            st_p = st_p + [sorted(tr.fits_by_n[np_].eps.reshape(-1).tolist())]
        menu.append(fit_peaks(beta, x, G, sigma, inv_sqrt_R=isr, n=np_, starts=st_p))
        menu.append(fit_peaks(beta, x, G, sigma, inv_sqrt_R=isr, n=np_,
                              starts=st_p, asym=True, steps=6000))
    menu.append(fit_edged_band(beta, x, G, sigma, inv_sqrt_R=isr, m=5,
                               starts=[(0.0, 1.0), (0.3, 1.5)]))
    # effective parameters: the sum rule and sum-zero b each remove one
    menu.append(TemplateFit(name="free", m=tr.selected_n,
                            k_params=4*tr.selected_n - 2,
                            chi2=tr.chi2_selected, poles=free, params={}))

    def repolish(c):
        if c.name in ("ladder", "mirrored_ladder"):
            return fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=c.m,
                              mirrored=(c.name == "mirrored_ladder"),
                              starts=[(c.params["eps0"], c.params["Omega"])],
                              steps=8000, lr=0.01)
        if c.name in ("peaks", "peaks_asym"):
            st_ = [sorted(c.params[f"eps{i}"] for i in range(c.m))]
            ts = None
            if c.name == "peaks_asym":
                aa = c.poles.a.reshape(-1)
                bb2 = c.poles.b.reshape(-1)
                ts = [(bb2 / aa.clamp_min(1e-12)).clamp(-0.45, 0.45).tolist()]
            return fit_peaks(beta, x, G, sigma, inv_sqrt_R=isr, n=c.m,
                             starts=st_, asym=(c.name == "peaks_asym"),
                             steps=12000, lr=0.01, t_starts=ts)
        if c.name == "edged_band":
            return fit_edged_band(beta, x, G, sigma, inv_sqrt_R=isr, m=c.m,
                                  starts=[(c.params["centre"],
                                           c.params["half_width"])],
                                  steps=6000, lr=0.01)
        if c.name == "free":
            fitted, _, c2 = refit_poles_to_greens(c.poles, beta, x[None], G,
                                                  sigma, inv_sqrt_R=isr,
                                                  steps=2000, lr=0.01)
            return TemplateFit(name="free", m=c.m, k_params=c.k_params,
                               chi2=float(c2), poles=fitted, params={})
        return c

    winner = select_model(menu, L_eff=ntau + 1, repolish=repolish)
    As = winner.poles.to_dimensionless(beta).spectral(quad.nodes)
    At = truth.spectral(quad.nodes)
    w1 = float(spectral_w1(As, At, quad, beta=beta))
    print(f"[{name:<13}] winner {winner.name}({winner.m}) k={winner.k_params} "
          f"chi2 {winner.chi2:.3f} W1 {w1:.4f} | scan n={tr.selected_n} (true {n_true})")
    for c in sorted(menu, key=lambda c: c.chi2)[:5]:
        print(f"    {c.name}({c.m}) k={c.k_params} chi2 {c.chi2:.4g}")
    ART["probes"].append(dict(
        name=name, n_true=n_true, truth_eps=(te[0]/beta).tolist(),
        truth_A=(At[0]*beta).tolist(), truth_b=(tb[0]).tolist(),
        free_A=(free.to_dimensionless(beta).spectral(quad.nodes)[0]*beta).tolist(),
        zd_A=(zr.poles.to_dimensionless(beta).spectral(quad.nodes)[0]*beta).tolist(),
        ties=[dict(name=c.name, m=c.m, k=c.k_params, chi2=c.chi2,
                   A=(c.poles.to_dimensionless(beta).spectral(quad.nodes)[0]*beta).tolist())
              for c in menu if c is not winner
              and c.k_params == winner.k_params
              and abs(c.chi2 - winner.chi2) * (ntau+1) < 5.0],
        winner=dict(name=winner.name, m=winner.m, k=winner.k_params,
                    chi2=winner.chi2, w1=w1, params=winner.params,
                    A=(As[0]*beta).tolist(),
                    pole_b=winner.poles.b.reshape(-1).tolist(),
                    pole_a=winner.poles.a.reshape(-1).tolist()),
        menu=[dict(name=c.name, m=c.m, k=c.k_params, chi2=c.chi2) for c in menu],
        scan=dict(ns=tr.ns, chi2s=tr.chi2s, selected_n=tr.selected_n)))
json.dump(ART, open("out/step1/asym_probes_zd.json", "w"))
print("ASYM_DONE")
