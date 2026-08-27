import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
art = json.load(open("out/step1/anchored_scan.json"))
om = art["nodes"]
fig, axes = plt.subplots(2, 3, figsize=(13.4, 6.4), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.5, 1.0]})
for j, case in enumerate(art["cases"]):
    ax = axes[0][j]
    if case.get("truth_A"):
        ax.plot(om, case["truth_A"], color="#1D242C", lw=1.8, label="truth")
    ax.plot(om, case["old"]["A"], color="#0E7C86", lw=1.1, ls="-.",
            label=f"old scan: n={case['old']['n']}, $\\chi^2$ {case['old']['chi2']:.2f}")
    ax.plot(om, case["new"]["A"], color="#B8451F", lw=1.4, ls="--",
            label=f"anchored: n={case['new']['n']}, $\\chi^2$ {case['new']['chi2']:.2f}")
    lim = 6 if case["name"] == "Holstein" else 4.4
    ax.set_xlim(-lim, lim)
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_title(case["name"], pad=5)
    if case["old"]["w1"]:
        ax.text(0.02, 0.97, f"W$_1$ old {case['old']['w1']:.4f}\n"
                f"W$_1$ new {case['new']['w1']:.4f}",
                transform=ax.transAxes, va="top", fontsize=7, family="monospace")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    if j == 0: ax.set_ylabel("$A(\\omega)$")
    ax2 = axes[1][j]
    ax2.plot(case["old"]["ns"], case["old"]["chi2s"], "-o", color="#0E7C86",
             ms=3.5, lw=1.1, label="old")
    ax2.plot(case["new"]["ns"], case["new"]["chi2s"], "-o", color="#B8451F",
             ms=3.5, lw=1.1, label="anchored")
    for key, color in [("old", "#0E7C86"), ("new", "#B8451F")]:
        c = case[key]
        ax2.scatter([c["n"]], [c["chi2s"][c["ns"].index(c["n"])]], s=70,
                    facecolors="none", edgecolors=color, linewidths=1.4)
    ax2.set_yscale("log"); ax2.set_xlabel("pole count $n$")
    if j == 0:
        ax2.set_ylabel("scan $\\chi^2(n)$")
        ax2.legend(fontsize=7, frameon=False)
fig.suptitle("The anchored rank scan ($\\chi^2 + wL2$ refits, prune init, pre-refit) vs the old scan — "
             "same seeds, same LR rule on pure $\\chi^2$", fontsize=10)
fig.savefig("out/step1/anchored_scan.png", dpi=200)
print("figure saved")
