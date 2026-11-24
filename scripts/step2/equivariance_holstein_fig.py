"""Holstein decomposition figure: how much of the group-averaging gain is symmetry?

Three versions of the same trained decoder, fitted to the Holstein cell
(beta 8, lambda 0.325) by latent descent:
  plain              one decode                      (Lz latent, 16 poles)
  2-branch control   1/2 dec(z1) + 1/2 dec(z2)        (2 Lz latent, 32 poles)
  mirror-averaged    1/2 dec(z1) + 1/2 M dec(z2)      (2 Lz latent, 32 poles)
The control and the mirror-averaged model have identical capacity, so the
step between them is what the PH symmetry contributes.
Reads out/step2/equivariance/{tag}_holstein.json, {tag}_control_holstein.{json|log}
and out/step2/m1judge/M1_60k_dual.json.
Output: out/step2/equivariance/holstein_floor_decomposition.png
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("out/step2/equivariance")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"M1_60k": "#2a78d6", "s1_metric": "#eb6834"}        # categorical slots 1, 2
GRAY = "#8f8e89"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#8f8e89", "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
TAGS = ["M1_60k", "s1_metric"]
LABEL = {"M1_60k": "M1-60k (dispersive map)", "s1_metric": "s1_metric (step-1 map)"}


def control_seeds(tag):
    js = D / f"{tag}_control_holstein.json"
    if js.exists():
        r = json.load(open(js))
        return ([r[k]["floor"] for k in sorted(r) if k.startswith("seed")],
                [r[k]["chi2_map"] for k in sorted(r) if k.startswith("seed")])
    fl, mp = [], []
    for line in open(D / f"{tag}_control_holstein.log"):
        m = re.search(r"MAP ([\d.]+)\s+floor ([\d.]+)", line)
        if m:
            mp.append(float(m.group(1)))
            fl.append(float(m.group(2)))
    return fl, mp


H = {t: json.load(open(D / f"{t}_holstein.json")) for t in TAGS}
CS = {t: control_seeds(t) for t in TAGS}
target = H["M1_60k"]["f"]["target"]

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.15, 1.0, 1.45]})
X = [0, 1, 2]
XL = ["plain\n1 decode\n16 poles", "2-branch control\n(no mirror)\n32 poles",
      "mirror-averaged F\n(equivariant)\n32 poles"]


def slope_panel(ax, key, ckey, title, annotate):
    for t in TAGS:
        c = CS[t][ckey]
        ys = [H[t]["f"][key], float(np.median(c)), H[t]["F"][key]]
        ax.plot(X, ys, color=COL[t], lw=2.0, zorder=3, label=LABEL[t])
        ax.scatter(X, ys, s=64, color=COL[t], edgecolors="white", linewidths=2.0, zorder=4)
        ax.scatter([1] * len(c), c, s=14, color=COL[t], alpha=0.55, zorder=4,
                   edgecolors="none")
        for xx, yy in zip(X, ys):
            ax.annotate(f"{yy:.3g}", (xx, yy), xytext=(7, 6), textcoords="offset points",
                        ha="left", va="bottom", fontsize=8, color=INK)
        if annotate:
            for (x0, y0), (x1, y1), what in (((0, ys[0]), (1, ys[1]), "capacity"),
                                              ((1, ys[1]), (2, ys[2]), "symmetry")):
                fac = y0 / y1
                txt = f"÷{fac:.1f} {what}" if fac >= 1.05 else f"×{1 / fac:.2f} ({what}: none)"
                below = t == "M1_60k"
                flat = what == "symmetry" and not below
                ax.annotate(txt, ((x0 + x1) / 2, np.sqrt(y0 * y1)),
                            xytext=(-8, -10) if below else ((0, -12) if flat else (8, 10)),
                            textcoords="offset points",
                            ha="right" if below else ("center" if flat else "left"),
                            va="top" if (below or flat) else "bottom", fontsize=7.5,
                            color=INK2, style="italic")
    ax.set_yscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels(XL, fontsize=7.5)
    ax.set_xlim(-0.35, 2.55)
    ax.grid(axis="y", color=GRID, lw=0.6, which="both")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", pad=6)


ax = axes[0]
ax.axhspan(target - 0.061, target + 0.061, color="#d9d8d3", alpha=0.7, lw=0, zorder=1)
ax.axhline(target, color=INK2, lw=0.8, ls="--", zorder=2)
ax.text(2.5, target * 0.72, f"noise floor {target:.3f} ± 0.061\n(a perfect spectrum scores here)",
        ha="right", va="top", fontsize=7, color=INK2)
slope_panel(ax, "floor", 0, "(a) Best fit the decoder can reach (latent prior off)", True)
ax.set_ylabel(r"Holstein $\chi^2$ per dof")
ax.set_ylim(0.08, 60)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

ax = axes[1]
slope_panel(ax, "chi2_map", 1, "(b) Same, latent prior on (MAP)", True)
ax.set_ylim(4, 60)

# (c) spectra
ax = axes[2]
d = json.load(open("out/step2/m1judge/M1_60k_dual.json"))
c = next(q for q in d["cases"] if q["name"] == "Holstein")
om = np.array(d["nodes"])
o = np.argsort(om)
keep = np.abs(om[o]) < 6
ax.plot(om[o][keep], np.array(c["full"]["A"])[o][keep], color=INK, lw=1.0,
        label=f"free-pole fit, off the manifold ($\\chi^2$ {c['full']['chi2']:.3f})", zorder=5)
w = np.array(H["M1_60k"]["f"]["omega"])
ax.plot(w, H["M1_60k"]["f"]["A_floor"], color=GRAY, lw=1.6,
        label=f"plain decoder floor ($\\chi^2$ {H['M1_60k']['f']['floor']:.2f})", zorder=3)
ax.plot(w, H["M1_60k"]["F"]["A_floor"], color=COL["M1_60k"], lw=2.0,
        label=f"mirror-averaged floor ($\\chi^2$ {H['M1_60k']['F']['floor']:.3f})", zorder=4)
ax.axhline(0, color=GRAY, lw=0.5)
ax.set_xlim(-6, 6)
ax.set_ylim(-0.03, 0.58)
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$A(\omega)$")
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("(c) M1-60k: the spectra behind the two floors", loc="left", pad=6)
ax.legend(frameon=False, fontsize=7.5, loc="upper center")

n_seeds = len(CS["M1_60k"][0])
fig.suptitle("Holstein cell (beta 8, lambda 0.325): how much of the group-averaging gain is "
             f"extra capacity, and how much is the PH symmetry?   Control: {n_seeds} jitter "
             "seeds (faint dots), median (large dot)", fontsize=10, x=0.01, ha="left")
out = D / "holstein_floor_decomposition.png"
fig.savefig(out, dpi=200, facecolor="white")
print(f"saved {out}")
