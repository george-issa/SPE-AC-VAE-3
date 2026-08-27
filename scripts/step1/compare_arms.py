"""Arm-comparison figure: probe overlays (truth vs each arm's winner) + streams.
Usage: compare_arms.py out.png labelA=judge_A.json labelB=judge_B.json [...]"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#3A3F45"})
outp = sys.argv[1]
arms = []
for spec in sys.argv[2:]:
    label, path = spec.split("=", 1)
    arms.append((label, json.load(open(path))))
COLORS = ["#B8451F", "#0E7C86", "#8A6410", "#5A6572"]
n_probes = len(arms[0][1]["probes"])
fig, axes = plt.subplots(2, n_probes, figsize=(2.9 * n_probes, 6.0),
                         constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.6, 1.0]})
om = arms[0][1]["nodes"]
for j in range(n_probes):
    ax = axes[0][j]
    p0 = arms[0][1]["probes"][j]
    ax.plot(om, p0["truth_A"], color="#1D242C", lw=1.6, zorder=2, label="truth")
    for (label, art), color in zip(arms, COLORS):
        p = art["probes"][j]
        w = p["winner"]
        ax.plot(om, w["A"], color=color, lw=1.1, ls="--", zorder=3,
                label=f"{label}: {w['name'][:4]}({w['m']})")
    ax.scatter(p0["truth_eps"], [0.008] * len(p0["truth_eps"]), s=55, marker="*",
               facecolors="none", edgecolors="#111", linewidths=0.8,
               zorder=5, clip_on=False)
    ax.set_xlim(-4.4, 4.4); ax.set_title(p0["name"], pad=4)
    txt = "\n".join(f"{lb}: W$_1$ {a['probes'][j]['winner']['w1']:.4f}"
                    for lb, a in arms)
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=6.5,
            family="monospace")
    if j == 0:
        ax.set_ylabel("$A(\\omega)$")
        ax.legend(fontsize=6, frameon=False, loc="center left")
    # scan curves below
    ax2 = axes[1][j]
    for (label, art), color in zip(arms, COLORS):
        sc = art["probes"][j]["scan"]
        ax2.plot(sc["ns"], sc["chi2s"], "-o", color=color, ms=3, lw=1.0)
        ax2.scatter([sc["selected_n"]],
                    [sc["chi2s"][sc["ns"].index(sc["selected_n"])]],
                    s=60, facecolors="none", edgecolors=color, linewidths=1.3)
    ax2.axvline(p0["n_true"], color="#1D242C", lw=0.7, ls="--", alpha=0.5)
    ax2.set_yscale("log"); ax2.set_xlabel("$n$")
    if j == 0: ax2.set_ylabel("scan $\\chi^2(n)$")
metrics = "  |  ".join(
    f"{lb}: stream W1 {a['stream']['w1_omega']:.3f} l2 {a['stream']['l2_rel']:.3f} "
    f"rip {a['stream']['ripple']:.3f} · canary W1 {a['canary']['w1_omega']:.3f}"
    for lb, a in arms)
fig.suptitle(f"Arm comparison — probe winners (top), free-scan curves (bottom)\n{metrics}",
             fontsize=9)
fig.savefig(outp, dpi=195)
print("figure ->", outp)
