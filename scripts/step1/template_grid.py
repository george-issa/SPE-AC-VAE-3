"""Template reconstructions at every noise level, menu-chi^2 insets."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#3A3F45"})
store = json.load(open("out/microsweep/template_sweep.json"))
om = store["nodes"]
CASES = ["3-rung", "4-rung", "mirrored"]
TRUTH_POLES = {"3-rung": [0.5, 1.5, 2.5], "4-rung": [-0.4, 0.4, 1.2, 2.0],
               "mirrored": [-2.2, -1.2, 1.2, 2.2]}
NOISES = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
MSTYLE = {"ladder": ("o", "#0E7C86"), "mirrored_ladder": ("^", "#5A6572"),
          "free": ("s", "#1D242C")}
fig, axes = plt.subplots(3, 5, figsize=(14.2, 8.0), constrained_layout=True,
                         sharex=True, sharey="row")
for row, case in enumerate(CASES):
    for col, nz in enumerate(NOISES):
        r = next(q for q in store["results"]
                 if q["case"] == case and q["noise"] == nz)
        w = r["winner"]
        ax = axes[row][col]
        ax.plot(om, r["truth_A"], color="#1D242C", lw=1.4, zorder=2)
        ax.plot(om, w["A"], color="#B8451F", lw=1.0, ls="--", zorder=3)
        ax.scatter(TRUTH_POLES[case], [0.006]*len(TRUTH_POLES[case]), s=55,
                   marker="*", facecolors="none", edgecolors="#111",
                   linewidths=0.8, zorder=5, clip_on=False)
        ax.set_xlim(-4.4, 4.4)
        n_poles = 2 * w["m"] if w["name"] == "mirrored_ladder" else w["m"]
        ok = "✓" if n_poles == r["n_true"] else "✗"
        short = {"ladder": "lad", "mirrored_ladder": "mir", "free": "free"}[w["name"]]
        ax.text(0.02, 0.97, f"{short}({w['m']}) {ok}\nW$_1$ {w['w1']:.4f}",
                transform=ax.transAxes, va="top", fontsize=6.5, family="monospace")
        if row == 0: ax.set_title(f"$\\sigma$ = {nz:.0e}", pad=4)
        if col == 0: ax.set_ylabel(f"{case}\n$A(\\omega)$")
        if row == 2: ax.set_xlabel("$\\omega$")
fig.suptitle("Template-menu reconstructions at every noise level (dashed = winner, "
             "solid = truth, stars = true poles)",
             fontsize=10)
fig.savefig(sys.argv[1], dpi=185)
print("figure ->", sys.argv[1])
