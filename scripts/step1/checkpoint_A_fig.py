import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
a = json.load(open("out/step1/checkpoint_final.json"))
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
ax = axes[0]
if "A_zdescent" in a:
    ax.plot(a["nodes"], a["A_zdescent"], color="#8A6410", lw=1.1, ls="-",
            alpha=0.8, label="z-descent (prior-informed)")
if "A_free" in a:
    ax.plot(a["nodes"], a["A_free"], color="#0E7C86", lw=1.0, ls="-.",
            label=f"free(n) scan")
ax.plot(a["nodes"], a["A"], color="#B8451F", lw=1.4, ls="--",
        label=f"winner: {a['winner']['name']}({a['winner']['m']})")
for t in a.get("ties", []):
    ax.plot(a["nodes"], t["A"], color="#B8451F", lw=1.0, ls=":",
            alpha=0.7, label=f"tie: {t['name']}({t['m']})")
if "reference" in a:
    ax.plot(a["reference"]["omega"], a["reference"]["A"], color="#5A6572",
            lw=1.2, ls=":", label="SmoQyDEAC (reference)")
pe = [e for e, w in zip(a["winner"]["pole_eps"], a["winner"]["pole_a"]) if w > 1e-4]
ax.vlines(pe, 0, max(a["A"]) * 0.06, color="#B8451F", lw=0.8, alpha=0.7)
ax.axhline(0, color="#999", lw=0.4)
ax.set_xlim(-6, 6); ax.set_xlabel("$\\omega$"); ax.set_ylabel("$A(\\omega)$")
ax.legend(fontsize=7.5, frameon=False)
ax.set_title(a["label"], pad=5)
txt = (f"$\\chi^2$ {a['winner']['chi2']:.2f}  target {a['chi2_target']:.2f}\n"
       f"manifold floor {a.get('manifold_floor', float('nan')):.1f}")
ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7.5,
        family="monospace")
ax = axes[1]
menu = sorted(a["menu"], key=lambda c: (c["k"], c["chi2"]))
labels = [f"{c['name'][:4]}({c['m']})" for c in menu]
colors = ["#B8451F" if (c["name"] == a["winner"]["name"] and c["m"] == a["winner"]["m"])
          else "#0E7C86" for c in menu]
ax.bar(range(len(menu)), [c["chi2"] for c in menu], color=colors, width=0.62)
ax.axhline(a["chi2_target"], color="#999", lw=0.7, ls=":")
ax.set_yscale("log"); ax.set_xticks(range(len(menu)))
ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
ax.set_title("model menu $\\chi^2$", pad=5); ax.set_ylabel("$\\chi^2$ per dof")
ax = axes[2]
sc = a["scan"]
ax.plot(sc["ns"], sc["chi2s"], "-o", color="#0E7C86", ms=3.5, lw=1.1)
ax.scatter([sc["selected_n"]], [sc["chi2s"][sc["ns"].index(sc["selected_n"])]],
           s=80, facecolors="none", edgecolors="#B8451F", linewidths=1.4)
ax.axhline(a["winner"]["chi2"], color="#B8451F", lw=0.7, ls="--", alpha=0.7)
ax.text(5.5, a["winner"]["chi2"] * 1.4, "menu winner", fontsize=6.5, color="#B8451F")
ax.set_yscale("log"); ax.set_xlabel("pole count $n$")
ax.set_title("free-pole scan $\\chi^2(n)$", pad=5)
fig.suptitle("Real-data checkpoint A — half-filled Holstein, $\\beta$ = 8 (arm-C model)",
             fontsize=10)
fig.savefig("out/step1/checkpoint_final.png", dpi=200)
print("figure saved")
