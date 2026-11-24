import json, sys
CUSTOM = len(sys.argv) > 1   # banked default keeps its step-1 wording
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
art = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "out/step1/anchored_wsweep.json"))
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)
ax = axes[0]
for case, color in [("fano band", "#0E7C86"), ("asym triplet", "#5A6572")]:
    c = next(q for q in art["cases"] if q["name"] == case)
    ws = [r["weight"] for r in c["rows"]]
    ax.plot(ws, [r["w1"] for r in c["rows"]], "-o", color=color, ms=4, lw=1.2,
            label=f"{case} (n={c['rows'][0]['n']} at every w)")
ax.set_xscale("log"); ax.set_xlabel("anchor weight $w$")
ax.set_ylabel("W$_1$ to truth")
ax.set_title("probes: broad plateau, counts w-invariant" if not CUSTOM else
             f"probes ({art.get('ckpt', '')})", pad=5)
ax.legend(fontsize=7, frameon=False)
ax = axes[1]
h = next(q for q in art["cases"] if q["name"] == "Holstein")
ws = [r["weight"] for r in h["rows"]]
ns = [r["n"] for r in h["rows"]]
ax.axhspan(0.175-0.061, 0.175+0.061, color="#22703B", alpha=0.10)
ax.axhline(0.175, color="#22703B", lw=0.8, ls="--", alpha=0.7,
           label="perfect-fit band")
ax.plot(ws, [r["chi2"] for r in h["rows"]], "-o", color="#B8451F", ms=4, lw=1.2,
        label=(f"certified fit (n={ns[0]} at every w)" if len(set(ns)) == 1 else
               "certified fit (n = " + "/".join(map(str, ns)) + " across w)"))
ax.set_xscale("log"); ax.set_xlabel("anchor weight $w$")
ax.set_ylabel("$\\chi^2$")
ax.set_title("Holstein: calibration improves with $w$" if not CUSTOM else
             "Holstein: certified fit vs anchor weight", pad=5)
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.suptitle("Anchored-scan weight sweep: no sharp optimum — counts invariant, probes plateaued, "
             "real-data calibration favours $w \\approx 1$–$2$" if not CUSTOM else
             f"Anchored-scan weight sweep — {art.get('ckpt', '')}", fontsize=10)
fig.savefig(sys.argv[2] if len(sys.argv) > 2 else "out/step1/anchored_wsweep.png", dpi=200)
print("figure saved")
