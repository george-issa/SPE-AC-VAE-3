"""Step-1 judge: frozen streams + full-pipeline probes for one checkpoint."""
import sys, json, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.physics.templates import (TemplateFit, fit_ladder, ladder_starts,
                                      fit_peaks, peak_starts, fit_edged_band,
                                      select_model)
from spevae.losses import spectral_w1
from spevae.data.pole_prior import PolePriorGenerator
from spevae.data.synthetic import SyntheticGenerator
from spevae.training import evaluate
from spevae.data.noise import NoiseModel, sample_correlated_noise

ckpt = sys.argv[1]
model = load_model(ckpt)
ART = {"ckpt": ckpt, "probes": []}
beta, ntau, nz = 8.0, 64, 1e-4
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0, 1.0, ntau + 1)

m_pole = evaluate(model, PolePriorGenerator(beta_range=(8.0, 8.0)), n_batches=16)
m_leg = evaluate(model, SyntheticGenerator(beta_range=(8.0, 8.0)), n_batches=16)
print(f"[stream pole ] W1 {m_pole['w1_omega']:.4f}  l2 {m_pole['l2_rel']:.4f}  "
      f"chi2geo {m_pole['chi2_geo']:.3g}  ripple {m_pole['ripple']:.4f}")
print(f"[canary legacy] W1 {m_leg['w1_omega']:.4f}  l2 {m_leg['l2_rel']:.4f}  "
      f"chi2geo {m_leg['chi2_geo']:.3g}", flush=True)
ART["stream"] = {k: float(v) for k, v in m_pole.items() if isinstance(v, float)}
ART["canary"] = {k: float(v) for k, v in m_leg.items() if isinstance(v, float)}
ART["nodes"] = (quad.nodes / beta).tolist()

def ladder_t(e0, Om, m, g, d):
    e = [e0 + k*Om for k in range(m)]; a = [d**k for k in range(m)]; s = sum(a)
    return e, [g]*m, [v/s for v in a]
e1,g1,a1 = ladder_t(0.5,1.0,3,0.30,0.55)
e2,g2,a2 = ladder_t(-0.4,0.8,4,0.25,0.60)
em,gm,am = ladder_t(1.2,1.0,2,0.30,0.45)
e3 = [-v for v in reversed(em)]+em; g3 = list(reversed(gm))+gm
a3 = [v/2 for v in reversed(am)]+[v/2 for v in am]
import torch as T
ksb = T.linspace(-1, 1, 5)
eb_eps = (0.3 + ksb * 1.4).tolist()
eb_gam = [0.12] + [0.45]*3 + [0.12]
eb_w = [0.2]*5
PROBES = [("3-rung ladder", e1,g1,a1, 3), ("4-rung ladder", e2,g2,a2, 4),
          ("mirrored ladder", e3,g3,a3, 4),
          ("edged band", eb_eps, eb_gam, eb_w, 5),
          ("irregular triplet", [0.4,1.1,2.6],[0.35,0.25,0.45],[0.5,0.3,0.2], 3)]
for name, eps, gam, w, n_true in PROBES:
    truth = PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                    a=torch.tensor([w]), b=torch.zeros(1,len(eps)))
    Gc = PoleGreens(quad)(truth, x)
    n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=nz, xi=0.1),
                                          torch.Generator().manual_seed(7))
    G = Gc + n
    ev, V = torch.linalg.eigh(R)
    isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    with torch.no_grad():
        out = model(G, x, beta, sigma=sigma, deterministic=True)
    amort = out.poles.detach().to_physical(beta)
    st = ladder_starts(amort); st_m = ladder_starts(amort, mirrored=True)
    menu = []
    for m in (2, 3, 4, 5):
        menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m, starts=st))
    for m in (1, 2, 3):
        menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m,
                               mirrored=True, starts=st_m, steps=4000))
    free, tr = rank_scan(amort, beta, x[None], G, sigma, inv_sqrt_R=isr)
    for np_ in (2, 3, 4):
        st_p = peak_starts(amort, np_)
        if np_ in tr.fits_by_n:
            st_p = st_p + [sorted(tr.fits_by_n[np_].eps.reshape(-1).tolist())]
        menu.append(fit_peaks(beta, x, G, sigma, inv_sqrt_R=isr, n=np_,
                              starts=st_p))
    menu.append(fit_edged_band(beta, x, G, sigma, inv_sqrt_R=isr, m=5,
                               starts=[(0.0, 1.0), (0.5, 1.5)]))
    menu.append(TemplateFit(name="free", m=tr.selected_n, k_params=4*tr.selected_n,
                            chi2=tr.chi2_selected, poles=free, params={}))
    winner = select_model(menu, L_eff=ntau+1)
    As = winner.poles.to_dimensionless(beta).spectral(quad.nodes)
    w1 = float(spectral_w1(As, truth.spectral(quad.nodes), quad, beta=beta))
    amort_w1 = float(spectral_w1(out.poles.spectral(quad.nodes),
                                 truth.spectral(quad.nodes), quad, beta=beta))
    print(f"[{name:<17}] amort W1 {amort_w1:.4f} | winner {winner.name}({winner.m}) "
          f"chi2 {winner.chi2:.3f} W1 {w1:.4f} | free scan n={tr.selected_n} "
          f"(true {n_true})", flush=True)
    ART["probes"].append(dict(
        name=name, n_true=n_true, truth_eps=eps,
        truth_A=(truth.spectral(quad.nodes)[0] * beta).tolist(),
        amort_w1=amort_w1,
        winner=dict(name=winner.name, m=winner.m, k=winner.k_params,
                    chi2=winner.chi2, w1=w1, A=(As[0] * beta).tolist()),
        menu=[dict(name=c.name, m=c.m, k=c.k_params, chi2=c.chi2) for c in menu],
        scan=dict(ns=tr.ns, chi2s=tr.chi2s, selected_n=tr.selected_n)))
json.dump(ART, open(sys.argv[2], "w"))
print("JUDGE_DONE")
