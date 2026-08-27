import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
art = json.load(open("out/step1/dual_output.json"))
om = np.array(art["nodes"])
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), constrained_layout=True)
for ax, case in zip(axes, art["cases"]):
    if case.get("truth_A"):
        ax.plot(om, case["truth_A"], color="#1D242C", lw=2.0, label="truth", zorder=4)
    f = case["full"]
    lbl_f = f"full rendering ($\\chi^2$ {f['chi2']:.2f}"
    lbl_f += f", W$_1$ {f['w1']:.4f})" if f.get("w1") else ")"
    ax.plot(om, f["A"], color="#8A6410", lw=1.4, label=lbl_f, zorder=3)
    m = case["minimal"]
    lbl_m = f"minimal: n={m['n']} ($\\chi^2$ {m['chi2']:.2f}"
    lbl_m += f", W$_1$ {m['w1']:.4f})" if m.get("w1") else ")"
    ax.plot(om, m["A"], color="#B8451F", lw=1.2, ls="--", label=lbl_m, zorder=5)
    pe = [e for e, a in zip(m["pole_eps"], m["pole_a"]) if a > 1e-4]
    srt = np.argsort(om)
    Am = np.interp(pe, om[srt], np.array(m["A"])[srt])
    ax.vlines(pe, 0, Am, color="#B8451F", lw=0.8, alpha=0.6, zorder=2)
    ax.scatter(pe, Am, s=20, color="#B8451F", edgecolors="#5A2410",
               linewidths=0.4, zorder=6)
    lim = 6 if case["name"] == "Holstein" else 4.4
    ax.set_xlim(-lim, lim)
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_title(case["name"], pad=5)
    ax.set_xlabel("$\\omega$")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
axes[0].set_ylabel("$A(\\omega)$")
fig.suptitle("The standing protocol's dual output: the LR-certified minimal skeleton (dashed, stems) "
             "and the prior-started full rendering (gold) — one scan, both answers", fontsize=10)
fig.savefig("out/step1/dual_output.png", dpi=200)
print("figure saved")
