import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
art = json.load(open("out/step1/asym_probes_zd.json"))
om = art["nodes"]
n = len(art["probes"])
fig, axes = plt.subplots(2, n, figsize=(5.6 * n, 6.2), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.5, 1.0]})
for j, p in enumerate(art["probes"]):
    ax = axes[0][j]
    w = p["winner"]
    ax.plot(om, p["truth_A"], color="#1D242C", lw=1.7, label="truth (b ≠ 0)")
    if "zd_A" in p:
        ax.plot(om, p["zd_A"], color="#8A6410", lw=1.0, alpha=0.8,
                label="z-descent")
    if "free_A" in p:
        ax.plot(om, p["free_A"], color="#0E7C86", lw=1.0, ls="-.",
                label="free(n) (rank scan)")
    ax.plot(om, w["A"], color="#B8451F", lw=1.3, ls="--",
            label=f"winner: {w['name']}({w['m']})")
    for t in p.get("ties", []):
        ax.plot(om, t["A"], color="#B8451F", lw=1.0, ls=":", alpha=0.7,
                label=f"tie: {t['name']}({t['m']})")
    ax.scatter(p["truth_eps"], [0.006]*len(p["truth_eps"]), s=90, marker="*",
               facecolors="none", edgecolors="#111", linewidths=1.0,
               zorder=5, clip_on=False)
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_xlim(-4.2, 4.6); ax.set_title(p["name"], pad=5)
    ax.text(0.02, 0.97, f"W$_1$ {w['w1']:.4f}\n$\\chi^2$ {w['chi2']:.2f}\n"
            f"k = {w['k']}", transform=ax.transAxes, va="top",
            fontsize=7.5, family="monospace")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    if j == 0: ax.set_ylabel("$A(\\omega)$")
    ax2 = axes[1][j]
    menu = sorted(p["menu"], key=lambda c: (c["k"], c["chi2"]))
    labels = [f"{c['name'].replace('peaks_asym','asym').replace('mirrored_ladder','mir')[:6]}({c['m']})"
              for c in menu]
    colors = ["#B8451F" if (c["name"] == w["name"] and c["m"] == w["m"]
              and abs(c["chi2"] - w["chi2"]) < 1e-9) else "#0E7C86" for c in menu]
    ax2.bar(range(len(menu)), [c["chi2"] for c in menu], color=colors, width=0.62)
    for i, c in enumerate(menu):
        ax2.text(i, c["chi2"]*1.12, f"k={c['k']}", ha="center", fontsize=5.5,
                 color="#5A6572")
    ax2.set_yscale("log"); ax2.axhline(1.0, color="#999", lw=0.5, ls=":")
    ax2.set_xticks(range(len(menu)))
    ax2.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
    if j == 0: ax2.set_ylabel("$\\chi^2$ per dof")
fig.suptitle("The b channel probed: asymmetric truths, extended menu "
             "(winner orange; k annotated)", fontsize=10)
fig.savefig("out/step1/asym_probes_final.png", dpi=200)
print("figure saved")
