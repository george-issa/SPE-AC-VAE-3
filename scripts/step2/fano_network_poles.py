"""What does the NETWORK itself output for the in-distribution fano probe?

Same probe and noise as asym_judge.py (5 poles, noise 1e-4, seed 7).  For
s1_metric and M1-60k: the amortized pole set (posterior mean -> decoder) and
the z-descent pole set (3000 steps), against the exact label; plus the
network's own count belief (participation ratio) and posterior-sample spread.
Output: out/step2/m1judge/fano_network_poles.{json,png}
"""
import json, sys
sys.path.insert(0, "src")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spevae import load_model
from spevae.physics.poles import PoleSet, effective_pole_count
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.data.noise import NoiseModel, sample_correlated_noise
from spevae.losses import spectral_w1
from spevae.refine import z_descend
torch.set_num_threads(4)
BETA = 8.0
quad = SpectralQuadrature.for_beta(BETA)
x = torch.linspace(0, 1, 65)
ks = np.linspace(-1, 1, 5)
eps_t = 0.3 + ks * 1.2; a_t = np.full(5, 0.2); t_t = np.clip(0.7 * ks, -0.45, 0.45)
b_t = a_t * (t_t - (a_t * t_t).sum())
truth = PoleSet(torch.tensor([eps_t * BETA]).float(), torch.full((1, 5), 0.5 * BETA),
                torch.tensor([a_t]).float(), torch.tensor([b_t]).float())
Gc = PoleGreens(quad)(truth, x)
n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=1e-4, xi=0.1), torch.Generator().manual_seed(7))
G = Gc + n
ev, V = torch.linalg.eigh(R); isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
At = truth.spectral(quad.nodes)
w = torch.linspace(-4, 4.5, 1200)
rep = {"label": dict(eps=eps_t.tolist(), gamma=[0.5] * 5, a=a_t.tolist(), b=b_t.tolist())}

def summ(ps_phys, name):
    e, g, a, b = (ps_phys.eps[0].numpy(), ps_phys.gamma[0].numpy(), ps_phys.a[0].numpy(), ps_phys.b[0].numpy())
    o = np.argsort(e)
    neff = float(effective_pole_count(ps_phys.a)[0])
    big = int((a > 0.02).sum())
    w1 = float(spectral_w1(ps_phys.to_dimensionless(BETA).spectral(quad.nodes), At, quad, beta=BETA))
    print(f"  {name:10s} W1 {w1:.4f}  effective poles {neff:.1f}  poles with a>0.02: {big}  "
          f"widths {g.min():.3f}..{g.max():.3f}  |b|/a max {np.max(np.abs(b) / np.maximum(a, 1e-9)):.2f}")
    return dict(eps=e[o].tolist(), gamma=g[o].tolist(), a=a[o].tolist(), b=b[o].tolist(), w1=w1, neff=neff,
                A=ps_phys.spectral(w)[0].tolist())

for tag, ck in (("s1_metric", "out/step1/s1_metric.pt"), ("M1_60k", "out/step2/map_M1_60k.pt")):
    f = load_model(ck).eval()
    print(f"== {tag}")
    with torch.no_grad():
        out = f(G, x, BETA, sigma=sigma, deterministic=True)
        mu, logvar, norm = f.encode(G, x, BETA, sigma=sigma)
        samp = [f.decode(f.reparameterise(mu, logvar), norm, BETA).to_physical(BETA) for _ in range(64)]
    r = {"amortized": summ(out.poles.to_physical(BETA), "amortized")}
    zr = z_descend(f, G[0], x, BETA, sigma=sigma[0], inv_sqrt_R=isr, steps=3000)
    r["z_descent"] = summ(zr.poles, "z-descent")
    ne = np.array([float(effective_pole_count(s.a)[0]) for s in samp])
    w1s = np.array([float(spectral_w1(s.to_dimensionless(BETA).spectral(quad.nodes), At, quad, beta=BETA)) for s in samp])
    r["posterior_samples"] = dict(neff_mean=float(ne.mean()), neff_sd=float(ne.std()), w1_mean=float(w1s.mean()),
                                  w1_best=float(w1s.min()), A_sd=torch.stack([s.spectral(w)[0] for s in samp]).std(0).tolist())
    print(f"  posterior  64 samples: effective poles {ne.mean():.1f} ± {ne.std():.1f}, W1 mean {w1s.mean():.4f} best {w1s.min():.4f}")
    rep[tag] = r
rep["omega"] = w.tolist(); rep["A_truth"] = truth.to_physical(BETA).spectral(w)[0].tolist()
json.dump(rep, open("out/step2/m1judge/fano_network_poles.json", "w"))

# ---- figure: stems of pole weight at position, label vs network -------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"s1_metric": "#eb6834", "M1_60k": "#2a78d6"}
fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.6), constrained_layout=True, sharex=True)
for i, tag in enumerate(("s1_metric", "M1_60k")):
    for j, stage in enumerate(("amortized", "z_descent")):
        ax = axes[i, j]; r = rep[tag][stage]
        ax.plot(rep["omega"], rep["A_truth"], color=INK, lw=2.0, label="truth spectrum")
        ax.plot(rep["omega"], r["A"], color=COL[tag], lw=1.4, ls="--", label="network spectrum")
        ax2 = ax.twinx()
        ax2.vlines(eps_t, 0, a_t, color=INK, lw=3.0, alpha=0.35)
        ax2.scatter(eps_t, a_t, s=46, marker="D", color=INK, label="label poles (exact)", zorder=4)
        ax2.vlines(r["eps"], 0, r["a"], color=COL[tag], lw=1.4)
        ax2.scatter(r["eps"], r["a"], s=[max(10, 260 * g) for g in r["gamma"]], color=COL[tag],
                    edgecolors="white", linewidths=0.8, zorder=5, label="network poles (size ∝ width)")
        ax2.set_ylim(0, 0.42); ax.set_ylim(0, 0.42)
        ax2.set_ylabel("pole weight a", color=INK2); ax2.tick_params(colors=INK2)
        ax.set_title(f"{tag} — {stage.replace('_', '-')}:  {sum(np.array(r['a']) > 0.02)} poles with a>0.02, "
                     f"effective {r['neff']:.1f};  W$_1$ {r['w1']:.3f}", loc="left", fontsize=8.5)
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        if j == 0: ax.set_ylabel(r"$A(\omega)$")
        if i == 1: ax.set_xlabel(r"$\omega$")
        if i == 0 and j == 0:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left")
fig.suptitle("Fano test band (in the training distribution): the exact 5-pole label vs the poles the network emits "
             "(noise 1e-4, beta 8)", fontsize=10, x=0.01, ha="left")
fig.savefig("out/step2/m1judge/fano_network_poles.png", dpi=180, facecolor="white")
print("figure saved")
