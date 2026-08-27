"""Full pipeline on an unseen, deliberately non-ladder function; timed."""
import sys, json, time, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.physics.templates import TemplateFit, fit_ladder, ladder_starts, select_model
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise

beta, ntau, nz = 8.0, 64, 1e-4
# irregular triplet: unequal spacings AND widths AND weights -- no ladder fits this
eps, gam, w = [0.4, 1.1, 2.6], [0.35, 0.25, 0.45], [0.5, 0.3, 0.2]
truth = PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                a=torch.tensor([w]), b=torch.zeros(1, 3))
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0, 1.0, ntau+1)
Gc = PoleGreens(quad)(truth, x)
n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=nz, xi=0.1),
                                      torch.Generator().manual_seed(11))
G = Gc + n
ev, V = torch.linalg.eigh(R)
isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T

model = load_model("out/microsweep/id_ladder_full.pt")
t0 = time.perf_counter()
with torch.no_grad():
    out = model(G, x, beta, sigma=sigma, deterministic=True)
t_amort = time.perf_counter() - t0
amort = out.poles.detach().to_physical(beta)

t0 = time.perf_counter()
free, tr = rank_scan(amort, beta, x[None], G, sigma, inv_sqrt_R=isr)
t_scan = time.perf_counter() - t0

t0 = time.perf_counter()
st = ladder_starts(amort); st_m = ladder_starts(amort, mirrored=True)
menu = []
for m in (2, 3, 4, 5):
    menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m, starts=st))
for m in (1, 2, 3):
    menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m,
                           mirrored=True, starts=st_m, steps=4000))
t_tpl = time.perf_counter() - t0
menu.append(TemplateFit(name="free", m=tr.selected_n, k_params=4*tr.selected_n,
                        chi2=tr.chi2_selected, poles=free, params={}))
winner = select_model(menu, L_eff=ntau+1)
As = winner.poles.to_dimensionless(beta).spectral(quad.nodes)
w1 = float(spectral_w1(As, truth.spectral(quad.nodes), quad, beta=beta))
print(f"amortized {t_amort*1e3:.0f} ms | rank scan {t_scan:.1f} s | "
      f"templates {t_tpl:.1f} s | total {t_amort+t_scan+t_tpl:.1f} s")
print(f"winner: {winner}  W1 {w1:.4f}")
print(f"rank scan: {tr}")
for i in torch.argsort(winner.poles.eps[0]):
    if float(winner.poles.a[0, i]) > 1e-4:
        print(f"  eps {float(winner.poles.eps[0,i]):+.4f}  "
              f"gam {float(winner.poles.gamma[0,i]):.4f}  "
              f"a {float(winner.poles.a[0,i]):.4f}")
print(f"truth: {list(zip(eps, gam, w))}")
json.dump(dict(case="irregular triplet", noise=nz, n_true=3,
               truth_A=(truth.spectral(quad.nodes)[0]*beta).tolist(),
               winner=dict(name=winner.name, m=winner.m, k=winner.k_params,
                           chi2=winner.chi2, w1=w1, params=winner.params,
                           A=(As[0]*beta).tolist(),
                           pole_eps=winner.poles.eps.reshape(-1).tolist(),
                           pole_a=winner.poles.a.reshape(-1).tolist()),
               menu=[dict(name=c.name, m=c.m, k=c.k_params, chi2=c.chi2)
                     for c in menu]),
          open("out/microsweep/unseen_test.json", "w"))
print("saved")
