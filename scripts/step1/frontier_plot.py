"""chi^2 vs parameter count/type with the selection frontier."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, "src")
from spevae.physics.templates import chi2_quantile_95
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
store = json.load(open("out/microsweep/template_sweep.json"))
unseen = json.load(open("out/microsweep/unseen_test.json"))
L = 65
PANELS = [next(r for r in store["results"] if r["case"] == c and r["noise"] == 1e-4)
          for c in ["3-rung", "4-rung", "mirrored"]] + [unseen]
TITLES = ["3-rung ladder", "4-rung ladder", "mirrored ladder",
          "irregular triplet (unseen, non-ladder)"]
MSTYLE = {"ladder": ("o", "#0E7C86"), "mirrored_ladder": ("^", "#5A6572"),
          "free": ("s", "#1D242C")}
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.8), constrained_layout=True)
for ax, r, title in zip(axes, PANELS, TITLES):
    w = r["winner"]
    # dodge same-k points horizontally
    seen = {}
    for c in sorted(r["menu"], key=lambda c: (c["k"], c["chi2"])):
        kx = c["k"] + 0.35 * seen.get(c["k"], 0)
        seen[c["k"]] = seen.get(c["k"], 0) + 1
        mk, colr = MSTYLE[c["name"]]
        hit = (c["name"] == w["name"] and c["m"] == w["m"]
               and abs(c["chi2"] - w["chi2"]) < 1e-9)
        ax.scatter([kx], [c["chi2"]], s=34, marker=mk, color=colr, zorder=3)
        ax.annotate(str(c["m"]), (kx, c["chi2"]), textcoords="offset points",
                    xytext=(0, 5), fontsize=6, ha="center", color="#5A6572")
        if hit:
            ax.scatter([kx], [c["chi2"]], s=150, marker="o", facecolors="none",
                       edgecolors="#B8451F", linewidths=1.6, zorder=4)
    # selection frontier through the winner
    ks = np.arange(2, 22)
    front = np.array([w["chi2"] + (chi2_quantile_95(w["k"] - k) / L if k < w["k"]
                      else -chi2_quantile_95(k - w["k"]) / L) for k in ks])
    ax.plot(ks, np.clip(front, 1e-3, None), "--", color="#B8451F", lw=1.0,
            alpha=0.7, zorder=2)
    ax.fill_between(ks, 1e-3, np.clip(front, 1e-3, None), color="#B8451F",
                    alpha=0.05, zorder=1)
    ax.set_yscale("log"); ax.set_xlim(2.5, 21)
    ax.set_ylim(0.4, None)
    ax.axhline(1.0, color="#999", lw=0.5, ls=":")
    ax.set_title(title, pad=5); ax.set_xlabel("parameters $k$")
axes[0].set_ylabel("$\\chi^2$ per dof")
handles = [Line2D([], [], marker="o", ls="", color="#0E7C86", ms=5, label="ladder(m)"),
           Line2D([], [], marker="^", ls="", color="#5A6572", ms=5, label="mirrored(m)"),
           Line2D([], [], marker="s", ls="", color="#1D242C", ms=5, label="free(n)"),
           Line2D([], [], marker="o", ls="", mfc="none", mec="#B8451F", ms=9, label="winner"),
           Line2D([], [], color="#B8451F", ls="--", lw=1, alpha=0.7,
                  label="selection frontier")]
axes[3].legend(handles=handles, fontsize=6.5, frameon=False, loc="center right")
fig.suptitle("Model menus at $\\sigma$ = 1e-4: $\\chi^2$ vs complexity, winner circled — "
             "a candidate wins only by dipping below the frontier", fontsize=10)
fig.savefig(sys.argv[1], dpi=200)
print("figure ->", sys.argv[1])
