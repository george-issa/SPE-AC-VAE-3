"""Concentration screen figure (6k, M1 recipe): structure gained vs accuracy paid.

Row 1: (a) network effective pole count vs label count on the training prior;
       (b-e) small multiples: training-prior stream W1, step-1 stream W1,
       legacy canary W1, geo-mean W1 of the 7 pipeline probes, Holstein chi2.
Row 2: the fano probe (5-pole label) -- network poles after z-descent, per arm.
Reads out/step2/conc/<tag>_{count,judge,asym,dual}.json.
Output: out/step2/conc/conc_screen.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("out/step2/conc")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [("ctrl_s1", "control, seed 1", "#8f8e89", "o"),
        ("ctrl_s2", "control, seed 2", "#52514e", "o"),
        ("conc01", "concentration 0.1", "#2a78d6", "s"),
        ("conc03", "concentration 0.3", "#eb6834", "D")]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#8f8e89", "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def J(tag, part):
    return json.load(open(D / f"{tag}_{part}.json"))


def style(ax):
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


R = {}
for tag, *_ in ARMS:
    c, j, a, d = J(tag, "count"), J(tag, "judge"), J(tag, "asym"), J(tag, "dual")
    probes = [p["winner"]["w1"] for p in j["probes"]] + [p["winner"]["w1"] for p in a["probes"]]
    hol = next(q for q in d["cases"] if q["name"] == "Holstein")
    R[tag] = dict(count=c["count"], fano=c["fano"],
                  metrics={"training-prior stream W$_1$\n(one-shot)": c["count"]["w1"],
                           "step-1 stream W$_1$\n(one-shot)": j["stream"]["w1_omega"],
                           "legacy canary W$_1$\n(one-shot)": j["canary"]["w1_omega"],
                           "7 probes, pipeline\n(geo-mean W$_1$)": float(np.exp(np.mean(np.log(probes)))),
                           "Holstein minimal $\\chi^2$\n(target 0.175)": hol["minimal"]["chi2"]},
                  hol_n=hol["minimal"]["n"])

fig = plt.figure(figsize=(15.5, 8.0), constrained_layout=True)
gs = fig.add_gridspec(2, 7, height_ratios=[1.15, 1.0], width_ratios=[2.2, 1, 1, 1, 1, 1, 0.01])

ax = fig.add_subplot(gs[0, 0])
ax.plot([1, 12.5], [1, 12.5], color=INK2, lw=0.8, ls="--", label="perfect: network = label")
for tag, lab, col, mk in ARMS:
    b = R[tag]["count"]["bins"]
    xs = [v["label"] for v in b.values()]
    ys = [v["net"] for v in b.values()]
    sd = [v["net_sd"] for v in b.values()]
    ax.errorbar(xs, ys, yerr=sd, color=col, marker=mk, ms=5, lw=1.4, capsize=2,
                label=f"{lab} (corr {R[tag]['count']['corr']:.2f})")
ax.set_xlabel("label's effective pole count (binned)")
ax.set_ylabel("network's effective pole count")
ax.set_xlim(0.5, 12.5); ax.set_ylim(0, 17)
ax.set_title("(a) Does the network's pole count follow the label's?", loc="left")
ax.legend(frameon=False, fontsize=7, loc="lower right")
style(ax)

names = list(R["ctrl_s1"]["metrics"])
for k, name in enumerate(names):
    ax = fig.add_subplot(gs[0, 1 + k])
    for i, (tag, lab, col, mk) in enumerate(ARMS):
        v = R[tag]["metrics"][name]
        ax.scatter(v, len(ARMS) - 1 - i, color=col, marker=mk, s=40, zorder=3)
        txt = f"{v:.3f}" + (f" (n={R[tag]['hol_n']})" if "Holstein" in name else "")
        ax.annotate(txt, (v, len(ARMS) - 1 - i), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=6.8, color=INK2)
    c1, c2 = R["ctrl_s1"]["metrics"][name], R["ctrl_s2"]["metrics"][name]
    ax.axvspan(min(c1, c2), max(c1, c2), color="#8f8e89", alpha=0.18, lw=0,
               label="spread of the two control seeds")
    if "Holstein" in name:
        ax.axvline(0.175, color="#22703B", lw=0.9, ls="--", label="target 0.175")
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([a[1] for a in ARMS][::-1] if k == 0 else [], fontsize=7)
    ax.set_ylim(-0.6, len(ARMS) - 0.3)
    vals = [R[t]["metrics"][name] for t, *_ in ARMS]
    pad = 0.25 * (max(vals) - min(vals) + 1e-9)
    ax.set_xlim(min(vals) - pad, max(vals) + pad)
    ax.set_title(("(b) " if k == 0 else "") + name, fontsize=7.8)
    if k == 0:
        ax.legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0, -0.32))
    ax.tick_params(axis="x", labelsize=6.5)
    style(ax)

row2 = gs[1, :].subgridspec(1, 4)
for i, (tag, lab, col, mk) in enumerate(ARMS):
    ax = fig.add_subplot(row2[0, i])
    fz = R[tag]["fano"]
    w = np.array(fz["omega"])
    ax.plot(w, fz["A_truth"], color=INK, lw=1.8, label="truth")
    z = fz["z_descent"]
    ax.plot(w, z["A"], color=col, lw=1.2, ls="--", label="network (z-descent)")
    ax2 = ax.twinx()
    ax2.vlines(fz["label"]["eps"], 0, fz["label"]["a"], color=INK, lw=3, alpha=0.3)
    ax2.scatter(fz["label"]["eps"], fz["label"]["a"], marker="D", s=30, color=INK, zorder=4)
    ax2.vlines(z["eps"], 0, z["a"], color=col, lw=1.2)
    ax2.scatter(z["eps"], z["a"], s=[max(8, 220 * g) for g in z["gamma"]], color=col,
                edgecolors="white", linewidths=0.6, zorder=5)
    ax.set_ylim(0, 0.45); ax2.set_ylim(0, 0.45)
    ax2.tick_params(labelsize=6.5, colors=INK2)
    ax.set_xlim(-3, 3.5)
    ax.set_title(f"(c) fano probe — {lab}\n{z['neff']:.1f} effective poles, W$_1$ {z['w1']:.3f}",
                 loc="left", fontsize=8)
    ax.set_xlabel(r"$\omega$")
    if i == 0:
        ax.set_ylabel(r"$A(\omega)$; stems: pole weight")
    style(ax)

fig.suptitle("Concentration screen (M1 recipe, 6k steps): one-label-one-pole pressure at weight 0.1 and 0.3 "
             "vs two control seeds.  (b): lower is better; gray band = seed-to-seed spread.  "
             "(c): label = 5 poles (diamonds)", fontsize=10, x=0.01, ha="left")
fig.savefig(D / "conc_screen.png", dpi=180, facecolor="white")
print("figure saved")
