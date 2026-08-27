"""Anchored-scan w-sweep: reconstruction overlays for every weight."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
art = json.load(open("out/step1/anchored_wsweep.json"))
om = art["nodes"]
ws = art["weights"]
cmap = cm.get_cmap("copper_r")
wcolors = {w: cmap(0.25 + 0.7 * i / (len(ws) - 1)) for i, w in enumerate(ws)}
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9), constrained_layout=True)
for ax, case in zip(axes, art["cases"]):
    if case.get("truth_A"):
        ax.plot(om, case["truth_A"], color="#1D242C", lw=2.0, label="truth",
                zorder=5)
    for r in case["rows"]:
        lbl = f"w={r['weight']:g}"
        if r.get("w1") is not None:
            lbl += f" (W$_1$ {r['w1']:.4f})"
        else:
            lbl += f" ($\\chi^2$ {r['chi2']:.3f})"
        ax.plot(om, r["A"], color=wcolors[r["weight"]], lw=1.0, ls="--",
                label=lbl, zorder=3)
    lim = 6 if case["name"] == "Holstein" else 4.4
    ax.set_xlim(-lim, lim)
    ax.axhline(0, color="#999", lw=0.4)
    n0 = case["rows"][0]["n"]
    ax.set_title(f"{case['name']} — n = {n0} at every w", pad=5)
    ax.set_xlabel("$\\omega$")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
axes[0].set_ylabel("$A(\\omega)$")
fig.suptitle("Anchored-scan reconstructions across the weight sweep "
             "(darker = larger $w$): the curves nearly coincide — the plateau, drawn",
             fontsize=10)
fig.savefig("out/step1/anchored_wsweep_recon.png", dpi=200)
print("figure saved")
