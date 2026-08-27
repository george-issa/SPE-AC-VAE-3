"""Unseen-function result: spectrum panel + menu heatmaps for all four cases."""
import json, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
store = json.load(open("out/microsweep/template_sweep.json"))
unseen = json.load(open("out/microsweep/unseen_test.json"))
om = store["nodes"]

# ---- figure 1: the unseen spectrum ----
fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
w = unseen["winner"]
ax.plot(om, unseen["truth_A"], color="#1D242C", lw=1.9, label="truth", zorder=2)
ax.plot(om, w["A"], color="#B8451F", lw=1.4, ls="--",
        label=f"pipeline: free({w['m']})", zorder=3)
tp = [0.4, 1.1, 2.6]
ax.scatter(tp, [0.01]*3, s=140, marker="*", facecolors="none",
           edgecolors="#111", linewidths=1.2, zorder=5, clip_on=False)
pe = [e for e, a in zip(w["pole_eps"], w["pole_a"]) if a > 1e-3]
Aw = np.interp(pe, om, w["A"])
ax.vlines(pe, 0, Aw, color="#B8451F", lw=0.9, alpha=0.65, zorder=1)
ax.scatter(pe, Aw, s=26, color="#B8451F", edgecolors="#5A2410",
           linewidths=0.5, zorder=4)
ax.axhline(0, color="#999", lw=0.4)
ax.set_xlim(-4.2, 4.6); ax.set_xlabel("$\\omega$"); ax.set_ylabel("$A(\\omega)$")
ax.legend(fontsize=8, frameon=False)
ax.text(0.02, 0.97, f"unseen irregular triplet · $\\sigma$ = 1e-4\n"
        f"W$_1$ {w['w1']:.4f} · $\\chi^2$ {w['chi2']:.2f} · 52 s",
        transform=ax.transAxes, va="top", fontsize=8, family="monospace")
ax.set_title("Unseen non-ladder function: free-pole fallback, exact count", pad=6)
fig.savefig("out/microsweep/unseen_spectrum.png", dpi=200)

# ---- figure 2: menu heatmaps ----
PANELS = [next(r for r in store["results"] if r["case"] == c and r["noise"] == 1e-4)
          for c in ["3-rung", "4-rung", "mirrored"]] + [unseen]
TITLES = ["3-rung ladder", "4-rung ladder", "mirrored ladder", "irregular triplet"]
TYPES = ["ladder", "mirrored_ladder", "free"]
TLAB = ["ladder", "mirrored", "free"]
MS = [1, 2, 3, 4, 5]
fig2, axes = plt.subplots(1, 4, figsize=(13.4, 3.0), constrained_layout=True)
for ax, r, title in zip(axes, PANELS, TITLES):
    grid = np.full((3, 5), np.nan)
    kann = np.full((3, 5), "", dtype=object)
    for c in r["menu"]:
        col = min(c["m"], 5) - 1
        rowi = TYPES.index(c["name"])
        grid[rowi][col] = math.log10(c["chi2"])
        kann[rowi][col] = f"k={c['k']}"
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=-0.2, vmax=6, aspect="auto")
    wn = r["winner"]
    wc = min(wn["m"], 5) - 1; wr = TYPES.index(wn["name"])
    ax.add_patch(plt.Rectangle((wc-0.48, wr-0.48), 0.96, 0.96, fill=False,
                               edgecolor="#B8451F", lw=2.2))
    for i in range(3):
        for j in range(5):
            if not np.isnan(grid[i][j]):
                ax.text(j, i, f"{10**grid[i][j]:.3g}\n{kann[i][j]}", ha="center",
                        va="center", fontsize=6,
                        color="#1D242C" if grid[i][j] < 2.6 else "#F5F5F5")
    ax.set_xticks(range(5), [str(m) for m in MS])
    ax.set_yticks(range(3), TLAB)
    ax.set_title(title, pad=5); ax.set_xlabel("multiplicity $m$ (rungs / pairs / poles)")
cb = fig2.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
cb.set_label("$\\log_{10}\\chi^2$", fontsize=8)
fig2.suptitle("Menu $\\chi^2$ heatmaps ($\\sigma$ = 1e-4): type vs multiplicity, "
              "winner outlined; cell labels: $\\chi^2$ and parameter cost", fontsize=10)
fig2.savefig("out/microsweep/menu_heatmap.png", dpi=200)
print("saved both")
