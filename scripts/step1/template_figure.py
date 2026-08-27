"""Template campaign figure: reconstruction | menu chi^2 | noise convergence."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
store = json.load(open("out/microsweep/template_sweep.json"))
free_rows = json.load(open("out/microsweep/noise_sweep.json"))
om = store["nodes"]
CASES = ["3-rung", "4-rung", "mirrored"]
TRUTH_POLES = {"3-rung": [0.5, 1.5, 2.5], "4-rung": [-0.4, 0.4, 1.2, 2.0],
               "mirrored": [-2.2, -1.2, 1.2, 2.2]}
SHOW_NOISE = 1e-4
fig, axes = plt.subplots(3, 3, figsize=(11.8, 8.4), constrained_layout=True)
for row, case in enumerate(CASES):
    rows = [r for r in store["results"] if r["case"] == case]
    r0 = next(r for r in rows if r["noise"] == SHOW_NOISE)
    # (1) reconstruction at 1e-4
    ax = axes[row][0]
    ax.plot(om, r0["truth_A"], color="#1D242C", lw=1.7, zorder=2)
    ax.plot(om, r0["winner"]["A"], color="#B8451F", lw=1.2, ls="--", zorder=3)
    ax.scatter(TRUTH_POLES[case], [0.008]*len(TRUTH_POLES[case]), s=90, marker="*",
               facecolors="none", edgecolors="#111", linewidths=1.0, zorder=5,
               clip_on=False)
    pe = [e for e, a in zip(r0["winner"]["pole_eps"], r0["winner"]["pole_a"]) if a > 1e-4]
    ax.vlines(pe, 0, [max(r0["winner"]["A"])*0.06]*len(pe), color="#B8451F",
              lw=1.0, zorder=4)
    ax.set_xlim(-4.4, 4.4)
    w = r0["winner"]
    ax.text(0.02, 0.97, f"{w['name']}({w['m']})\nW$_1$ {w['w1']:.4f}\n$\\chi^2$ {w['chi2']:.2f}",
            transform=ax.transAxes, va="top", fontsize=7.5, family="monospace")
    ax.set_ylabel(f"{case}\n$A(\\omega)$")
    if row == 0: ax.set_title(f"winning reconstruction ($\\sigma$ = 1e-4)", pad=5)
    if row == 2: ax.set_xlabel("$\\omega$")
    # (2) menu chi^2 chart
    ax = axes[row][1]
    menu = sorted(r0["menu"], key=lambda c: (c["k"], c["chi2"]))
    labels = [f"{c['name'][:3]}({c['m']})" if c["name"] != "free"
              else f"free({c['m']})" for c in menu]
    chis = [c["chi2"] for c in menu]
    ks = [c["k"] for c in menu]
    colors = ["#B8451F" if (c["name"] == w["name"] and c["m"] == w["m"])
              else "#0E7C86" for c in menu]
    ax.bar(range(len(menu)), chis, color=colors, width=0.62)
    for i, k in enumerate(ks):
        ax.text(i, chis[i]*1.15, f"k={k}", ha="center", fontsize=6, color="#5A6572")
    ax.set_yscale("log"); ax.axhline(1.0, color="#999", lw=0.6, ls=":")
    ax.set_xticks(range(len(menu)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    if row == 0: ax.set_title("model menu $\\chi^2$ (winner in orange)", pad=5)
    if row == 0: ax.set_ylabel("$\\chi^2$ per dof")
    # (3) noise convergence: template winner vs free poles
    ax = axes[row][2]
    nz = [r["noise"] for r in rows]
    ax.plot(nz, [r["winner"]["w1"] for r in rows], "-o", color="#B8451F",
            ms=4, lw=1.3, label="template menu")
    fr = [r for r in free_rows if r["case"] == case]
    ax.plot([r["noise"] for r in fr], [r["w1"] for r in fr], "-o",
            color="#0E7C86", ms=4, lw=1.3, label="free poles (rank scan)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.grid(alpha=0.2, lw=0.4)
    if row == 0: ax.set_title("noise convergence, W$_1$", pad=5)
    if row == 0: ax.legend(fontsize=7, frameon=False)
    if row == 2: ax.set_xlabel("noise $\\sigma$")
fig.suptitle("The template campaign: structure recovered through declared hypotheses — "
             "the log wall crossed where structure exists", fontsize=10)
fig.savefig(sys.argv[1], dpi=195)
print("figure ->", sys.argv[1])
