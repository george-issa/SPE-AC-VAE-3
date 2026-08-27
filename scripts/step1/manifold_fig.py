import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
art = json.load(open("out/step1/manifold_test.json"))
om = art["nodes"]
W_COLORS = {0.5: "#D9A845", 2.0: "#B8451F", 8.0: "#7A2E1F"}
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8), constrained_layout=True)
for ax, case in zip(axes, art["cases"]):
    if case.get("truth_A"):
        ax.plot(om, case["truth_A"], color="#1D242C", lw=1.8, label="truth")
    ax.plot(om, case["zd"]["A"], color="#8A6410", lw=0.9, alpha=0.55,
            label=f"z-descent ($\\chi^2$ {case['zd']['chi2']:.1f})")
    ax.plot(om, case["free"]["A"], color="#0E7C86", lw=1.1, ls="-.",
            label=f"free({case['free']['n']}) ($\\chi^2$ {case['free']['chi2']:.2f})")
    for sw in case["sweeps"]:
        ax.plot(om, sw["A"], color=W_COLORS[sw["weight"]], lw=1.2, ls="--",
                label=f"w={sw['weight']:g} ($\\chi^2$ {sw['chi2']:.2f})")
    ax.set_xlim(-6 if case["name"] == "Holstein" else -4.2, 6 if case["name"] == "Holstein" else 4.6)
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_title(case["name"], pad=5)
    ax.set_xlabel("$\\omega$")
    ax.legend(fontsize=6.2, frameon=False, loc="upper right")
    if case.get("truth_A"):
        rows = [f"free   W$_1$ {case['free']['w1']:.4f}"] + \
               [f"w={sw['weight']:<3g} W$_1$ {sw['w1']:.4f}" for sw in case["sweeps"]]
        ax.text(0.02, 0.97, "\n".join(rows), transform=ax.transAxes,
                va="top", fontsize=6.4, family="monospace")
    else:
        ax.text(0.02, 0.97, f"target {art['holstein_target']:.3f}",
                transform=ax.transAxes, va="top", fontsize=6.4, family="monospace")
axes[0].set_ylabel("$A(\\omega)$")
fig.suptitle("Manifold-regularized refits (dashed, by weight) vs the free scan and z-descent — "
             "the harshness dial, measured on all three cases", fontsize=10)
fig.savefig("out/step1/manifold_test.png", dpi=200)
print("figure saved")
