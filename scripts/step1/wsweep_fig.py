import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
art = json.load(open("out/step1/wsweep.json"))
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), constrained_layout=True)
# (1) probes: W1 vs w
ax = axes[0]
for case, color in [("fano band", "#0E7C86"), ("asym triplet", "#5A6572")]:
    c = next(q for q in art["cases"] if q["name"] == case)
    ws = [s["weight"] for s in c["sweeps"]]
    ax.plot(ws, [s["w1"] for s in c["sweeps"]], "-o", color=color, ms=4,
            lw=1.2, label=case)
    ax.axhline(c["free"]["w1"], color=color, lw=0.7, ls=":", alpha=0.7)
ax.set_xscale("log"); ax.set_xlabel("weight $w$"); ax.set_ylabel("W$_1$ to truth")
ax.set_title("probes: W$_1$ plateau (dotted = free scan)", pad=5)
ax.legend(fontsize=7, frameon=False)
# (2) Holstein: refit chi2 + gated chi2 vs w, with calibration band
h = next(q for q in art["cases"] if q["name"] == "Holstein")
ws = [s["weight"] for s in h["sweeps"]]
ax = axes[1]
ax.axhspan(0.175-0.061, 0.175+0.061, color="#22703B", alpha=0.10)
ax.axhline(0.175, color="#22703B", lw=0.8, ls="--", alpha=0.7)
ax.plot(ws, [s["chi2"] for s in h["sweeps"]], "-o", color="#B8451F", ms=4,
        lw=1.2, label="refit (16 poles, ungated)")
ax.plot(ws, [s["gated_chi2"] for s in h["sweeps"]], "-s", color="#0E7C86",
        ms=4, lw=1.2, label="count-gated")
for wv, s in zip(ws, h["sweeps"]):
    ax.annotate(f"n={s['gated_n']}", (wv, s["gated_chi2"]),
                textcoords="offset points", xytext=(0, 7), fontsize=6.5,
                ha="center", color="#0E7C86")
ax.set_xscale("log"); ax.set_xlabel("weight $w$"); ax.set_ylabel("$\\chi^2$")
ax.set_title("Holstein: $\\chi^2$ vs $w$ (band = perfect fit $\\pm1\\sigma$)", pad=5)
ax.legend(fontsize=7, frameon=False, loc="upper left")
# (3) Holstein: manifold distance vs w
ax = axes[2]
ax.plot(ws, [s["penalty"] for s in h["sweeps"]], "-o", color="#8A6410", ms=4, lw=1.2)
ax.set_xscale("log"); ax.set_xlabel("weight $w$")
ax.set_ylabel("manifold distance (L2rel)")
ax.set_title("Holstein: how far the data pulls off-manifold", pad=5)
fig.suptitle("Prior-started refit, fine weight sweep: in-vocabulary data never leaves the manifold; "
             "real data trades fidelity for noise absorption", fontsize=10)
fig.savefig("out/step1/wsweep.png", dpi=200)
print("figure saved")
