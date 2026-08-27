import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
a16 = json.load(open("out/step1/anchored16.json"))
aw = json.load(open("out/step1/anchored_wsweep.json"))
om = a16["nodes"]
W8 = {"fano band": 0.5, "asym triplet": 0.5, "Holstein": 1.0}
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.8), constrained_layout=True)
for ax, c16 in zip(axes, a16["cases"]):
    name = c16["name"]
    c8 = next(q for q in aw["cases"] if q["name"] == name)
    r8 = next(r for r in c8["rows"] if r["weight"] == W8[name])
    if c16.get("truth_A"):
        ax.plot(om, c16["truth_A"], color="#1D242C", lw=1.9, label="truth")
    l8 = f"n_max=8: n={r8['n']}"
    l16 = f"n_max=16: n={c16['new']['n']}"
    if r8.get("w1"):
        l8 += f", W$_1$ {r8['w1']:.4f}"
        l16 += f", W$_1$ {c16['new']['w1']:.4f}"
    else:
        l8 += f", $\\chi^2$ {r8['chi2']:.3f}"
        l16 += f", $\\chi^2$ {c16['new']['chi2']:.3f}"
    ax.plot(om, r8["A"], color="#0E7C86", lw=1.3, ls="--", label=l8)
    ax.plot(om, c16["new"]["A"], color="#B8451F", lw=1.1, ls="-.", label=l16)
    lim = 6 if name == "Holstein" else 4.4
    ax.set_xlim(-lim, lim)
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_title(name, pad=5); ax.set_xlabel("$\\omega$")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
axes[0].set_ylabel("$A(\\omega)$")
fig.suptitle("Ladder length at matched weight: the short ladder (unfitted 16$\\to$8 reduction) preserves the "
             "arrangement better than the long fitted descent", fontsize=10)
fig.savefig("out/step1/ladder_length.png", dpi=200)
print("figure saved")
