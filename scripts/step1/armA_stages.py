"""Arm-A stage overlay: truth vs amortized vs 16-pole refined vs menu winner."""
import json, sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import refit_poles_to_greens
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#3A3F45"})
art = json.load(open("out/step1/judge_A.json"))
beta, ntau, nz = 8.0, 64, 1e-4
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0, 1.0, ntau + 1)
om = art["nodes"]
model = load_model("out/step1/s1_fixes.pt")
fig, axes = plt.subplots(1, 5, figsize=(15.2, 3.3), constrained_layout=True)
for ax, p in zip(axes, art["probes"]):
    # rebuild the probe's data exactly (same seed protocol as the judge)
    n_p = len(p["truth_eps"])
    # reconstruct truth pole set from the judge run inputs
    TRUTHS = {"3-rung ladder": ([0.5,1.5,2.5],[0.3]*3,None,0.55),
              "4-rung ladder": ([-0.4,0.4,1.2,2.0],[0.25]*4,None,0.60),
              "mirrored ladder": ([-2.2,-1.2,1.2,2.2],[0.3]*4,
                                  [0.155,0.345,0.345,0.155],None),
              "edged band": (p["truth_eps"], [0.12]+[0.45]*3+[0.12], [0.2]*5, None),
              "irregular triplet": ([0.4,1.1,2.6],[0.35,0.25,0.45],[0.5,0.3,0.2],None)}
    eps, gam, w, decay = TRUTHS[p["name"]]
    if w is None:
        a = [decay**k for k in range(len(eps))]; s0 = sum(a); w = [v/s0 for v in a]
    truth = PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                    a=torch.tensor([w]), b=torch.zeros(1, len(eps)))
    Gc = PoleGreens(quad)(truth, x)
    n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=nz, xi=0.1),
                                          torch.Generator().manual_seed(7))
    G = Gc + n
    ev, V = torch.linalg.eigh(R)
    isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    with torch.no_grad():
        out = model(G, x, beta, sigma=sigma, deterministic=True)
        A_am = out.poles.spectral(quad.nodes.to(out.poles.eps.dtype))
    pref, _, c_ref = refit_poles_to_greens(out.poles.detach().to_physical(beta),
                                           beta, x[None], G, sigma,
                                           inv_sqrt_R=isr, steps=1500, lr=0.02)
    A_rf = pref.to_dimensionless(beta).spectral(quad.nodes)
    At = truth.spectral(quad.nodes)
    w1a = float(spectral_w1(A_am, At, quad, beta=beta))
    w1r = float(spectral_w1(A_rf, At, quad, beta=beta))
    ax.plot(om, p["truth_A"], color="#1D242C", lw=1.7, zorder=2)
    ax.plot(om, (A_am[0]*beta).numpy(), color="#9AA0A7", lw=1.0, ls=":", zorder=3)
    ax.plot(om, (A_rf[0]*beta).numpy(), color="#0E7C86", lw=1.0, ls="-.", zorder=4)
    ax.plot(om, p["winner"]["A"], color="#B8451F", lw=1.2, ls="--", zorder=5)
    ax.scatter(eps, [0.008]*len(eps), s=60, marker="*", facecolors="none",
               edgecolors="#111", linewidths=0.8, zorder=6, clip_on=False)
    ax.set_xlim(-4.4, 4.4); ax.set_title(p["name"], pad=4)
    ax.text(0.02, 0.97,
            f"amort  {w1a:.4f}\nrefine {w1r:.4f}  $\\chi^2$ {float(c_ref):.2f}\n"
            f"winner {p['winner']['w1']:.4f}  ({p['winner']['name'][:4]}({p['winner']['m']}))",
            transform=ax.transAxes, va="top", fontsize=6.4, family="monospace")
    if ax is axes[0]: ax.set_ylabel("$A(\\omega)$")
    ax.set_xlabel("$\\omega$")
handles = [Line2D([], [], color="#1D242C", lw=1.7, label="truth"),
           Line2D([], [], color="#9AA0A7", lw=1.0, ls=":", label="amortized"),
           Line2D([], [], color="#0E7C86", lw=1.0, ls="-.", label="refined (16 poles)"),
           Line2D([], [], color="#B8451F", lw=1.2, ls="--", label="menu winner")]
axes[2].legend(handles=handles, fontsize=6.5, frameon=False, loc="upper right")
fig.suptitle("Arm A, stage by stage: proposal $\\to$ 16-pole $\\chi^2$ refinement "
             "$\\to$ selected hypothesis (W$_1$ per stage annotated)", fontsize=10)
fig.savefig("out/step1/armA_stages.png", dpi=195)
print("figure saved")
