import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
art = json.load(open("out/step1/latent_landscape.json"))
om = art["nodes"]
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.4, 1.0]})
for j, p in enumerate(art["probes"]):
    ax = axes[0][j]
    ax.plot(om, p["truth_A"], color="#1D242C", lw=1.7, label="truth")
    ax.plot(om, p["A_zdesc"], color="#B8451F", lw=1.3, ls="--",
            label="z-descent decode")
    ax.set_xlim(-4.2, 4.6); ax.set_title(p["name"], pad=5)
    ax.axhline(0, color="#999", lw=0.4)
    ax.text(0.02, 0.97,
            f"mean:      $\\chi^2$ {p['mean'][0]:7.0f}  W$_1$ {p['mean'][1]:.3f}\n"
            f"z-descent: $\\chi^2$ {p['zdesc_mean'][0]:7.1f}  W$_1$ {p['zdesc_mean'][1]:.3f}",
            transform=ax.transAxes, va="top", fontsize=7, family="monospace")
    if j == 0: ax.set_ylabel("$A(\\omega)$")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax2 = axes[1][j]
    ch = [r[0] for r in p["samples"]]; w1 = [r[1] for r in p["samples"]]
    ax2.scatter(ch, w1, s=14, color="#0E7C86", alpha=0.6, label="posterior samples")
    ax2.scatter([p["mean"][0]], [p["mean"][1]], s=70, marker="s",
                color="#1D242C", label="posterior mean", zorder=4)
    ax2.scatter([p["zdesc_mean"][0]], [p["zdesc_mean"][1]], s=90, marker="*",
                color="#B8451F", label="z-descent", zorder=5)
    ax2.set_xscale("log"); ax2.set_xlabel("$\\chi^2$ (per dof, unwhitened)")
    if j == 0: ax2.set_ylabel("W$_1$ to truth")
    if j == 0: ax2.legend(fontsize=7, frameon=False)
fig.suptitle("The latent landscape, measured: the posterior mean is far from the "
             "manifold's best fit; $\\chi^2$ descent in $z$ finds it", fontsize=10)
fig.savefig("out/step1/latent_landscape.png", dpi=200)
print("figure saved")
