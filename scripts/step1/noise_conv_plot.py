import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
rows = json.load(open("out/microsweep/noise_sweep.json"))
CASES = [("3-rung", "#0E7C86"), ("4-rung", "#B8451F"), ("mirrored", "#5A6572")]
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), constrained_layout=True)
for case, color in CASES:
    rr = [r for r in rows if r["case"] == case]
    nz = [r["noise"] for r in rr]
    axes[0].plot(nz, [r["n_sel"] for r in rr], "-o", color=color, ms=4, lw=1.2, label=case)
    axes[0].axhline(rr[0]["n_true"], color=color, lw=0.7, ls=":", alpha=0.6)
    axes[1].plot(nz, [r["w1"] for r in rr], "-o", color=color, ms=4, lw=1.2)
    axes[2].plot(nz, [r["max_pos_err"] for r in rr], "-o", color=color, ms=4, lw=1.2)
for ax, title, ylab in [(axes[0], "selected pole count", "$n_{sel}$"),
                        (axes[1], "spectral error", "W$_1$"),
                        (axes[2], "worst pole-position error", "max $|\\Delta\\varepsilon|$")]:
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("noise level $\\sigma$"); ax.set_title(title, pad=5)
    ax.set_ylabel(ylab); ax.grid(alpha=0.2, lw=0.4)
axes[1].set_yscale("log"); axes[2].set_yscale("log")
axes[0].set_yticks([2,3,4,5])
axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle("Noise convergence of the full pipeline (cleaner data $\\rightarrow$): counts staircase up fast; "
             "spectra and positions hit the logarithmic wall", fontsize=10)
fig.savefig(sys.argv[1], dpi=200)
print("figure ->", sys.argv[1])
