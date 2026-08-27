import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
art = json.load(open("out/step1/tempered.json"))
om = art["nodes"]
WC = {0.0: "#0E7C86", 4.0: "#D9A845", 16.0: "#B8451F", 64.0: "#5A2410"}
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), constrained_layout=True)
for ax, case in zip(axes, art["cases"]):
    if case.get("truth_A"):
        ax.plot(om, case["truth_A"], color="#1D242C", lw=2.0, label="truth", zorder=5)
    for r in case["rows"]:
        lbl = f"w={r['weight']:g}: n={r['n']}"
        if r.get("w1") is not None:
            lbl += f", W$_1$ {r['w1']:.4f}"
        else:
            lbl += f", $\\chi^2$ {r['chi2']:.3f}"
        ax.plot(om, r["A"], color=WC[r["weight"]], lw=1.1, ls="--", label=lbl)
    lim = 6 if case["name"] == "Holstein" else 4.4
    ax.set_xlim(-lim, lim)
    ax.axhline(0, color="#999", lw=0.4)
    cmin = min(range(len(case["c"])), key=lambda i: case["c"][i]) + 1
    ax.set_title(f"{case['name']} — prior count belief peaks at n = {cmin}", pad=5)
    ax.set_xlabel("$\\omega$")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
axes[0].set_ylabel("$A(\\omega)$")
fig.suptitle("The tempered family: selection climbs from the evidence count toward the prior's count as $w$ grows",
             fontsize=10)
fig.savefig("out/step1/tempered.png", dpi=200)
print("figure saved")
