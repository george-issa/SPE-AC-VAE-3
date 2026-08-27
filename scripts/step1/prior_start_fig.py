import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
art = json.load(open("out/step1/prior_start.json"))
om = art["nodes"]
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), constrained_layout=True)
for ax, case in zip(axes, art["cases"]):
    if case.get("truth_A"):
        ax.plot(om, case["truth_A"], color="#1D242C", lw=1.8, label="truth")
    ax.plot(om, case["zd"]["A"], color="#8A6410", lw=0.9, alpha=0.5,
            label="z-descent (start)")
    ax.plot(om, case["free"]["A"], color="#0E7C86", lw=1.0, ls="-.",
            label=f"free({case['free']['n']}) scan")
    fs = next(s for s in case["free_started"] if s["weight"] == 2.0)
    ax.plot(om, fs["A"], color="#9AA0A7", lw=1.0, ls=":",
            label=f"free-started w=2 ($\\chi^2$ {fs['chi2']:.2f})")
    ps = next(s for s in case["sweeps"] if s["weight"] == 2.0)
    ax.plot(om, ps["A"], color="#B8451F", lw=1.5, ls="--",
            label=f"prior-started w=2 ($\\chi^2$ {ps['chi2']:.2f})")
    lim = 6 if case["name"] == "Holstein" else 4.4
    ax.set_xlim(-lim, lim + (0.4 if case["name"] != "Holstein" else 0))
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_title(case["name"], pad=5)
    ax.set_xlabel("$\\omega$")
    ax.legend(fontsize=6.2, frameon=False, loc="upper right")
    if case.get("truth_A"):
        rows = [f"free          W$_1$ {case['free']['w1']:.4f}",
                f"free-started  W$_1$ {fs['w1']:.4f}",
                f"prior-started W$_1$ {ps['w1']:.4f}"]
        ax.text(0.02, 0.97, "\n".join(rows), transform=ax.transAxes,
                va="top", fontsize=6.4, family="monospace")
axes[0].set_ylabel("$A(\\omega)$")
fig.suptitle("Direction of approach decides the basin: prior-started (orange) vs "
             "free-started (grey) manifold refits, same objective and budget", fontsize=10)
fig.savefig("out/step1/prior_start.png", dpi=200)
print("figure saved")
