"""Pole-type reconstructions: full-pipeline winners of s1_metric and M1-60k vs truth.

Seven pole-closed probes at noise 1e-4 (beta 8): the five step-1 probes
(judge_arm.py: amortized proposal -> rank scan -> template menu) and the two
b-channel probes (asym_judge.py: z-descent -> rank scan -> extended menu).
Each probe: spectrum overlay (top) and reconstruction minus truth (bottom).
Reads out/step2/m1judge/{tag}_{judge,asym}.json.
Output: out/step2/m1judge/pole_reconstructions.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("out/step2/m1judge")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"s1_metric": "#eb6834", "M1_60k": "#2a78d6"}
NAME = {"s1_metric": "s1_metric", "M1_60k": "M1-60k"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#8f8e89", "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def load(tag):
    probes = {}
    for part in ("judge", "asym"):
        art = json.load(open(D / f"{tag}_{part}.json"))
        om = np.array(art["nodes"])
        for p in art["probes"]:
            w = p["winner"]
            probes[p["name"]] = dict(om=om, truth=np.array(p["truth_A"]), A=np.array(w["A"]),
                                     label=f"{w['name']}({w['m']})", w1=w["w1"], chi2=w["chi2"],
                                     n_true=p["n_true"], eps=p.get("truth_eps"))
    return probes


R = {t: load(t) for t in COL}
ORDER = ["3-rung ladder", "4-rung ladder", "mirrored ladder", "edged band",
         "irregular triplet", "asym triplet", "fano band"]

fig = plt.figure(figsize=(15.5, 9.2), constrained_layout=True)
gs = fig.add_gridspec(4, 4, height_ratios=[3, 1.1, 3, 1.1])
for i, name in enumerate(ORDER):
    r, c = 2 * (i // 4), i % 4
    ax = fig.add_subplot(gs[r, c])
    axr = fig.add_subplot(gs[r + 1, c], sharex=ax)
    ref = R["s1_metric"][name]
    o = np.argsort(ref["om"])
    om = ref["om"][o]
    keep = np.abs(om) < 4.5
    ax.plot(om[keep], ref["truth"][o][keep], color=INK, lw=2.4, label="truth", zorder=2)
    lines = []
    for t in COL:
        p = R[t][name]
        oo = np.argsort(p["om"])
        omt = p["om"][oo]
        kk = np.abs(omt) < 4.5
        ax.plot(omt[kk], p["A"][oo][kk], color=COL[t], lw=1.3, ls="--", zorder=3,
                label=f"{NAME[t]}: {p['label']}")
        axr.plot(omt[kk], (p["A"] - p["truth"])[oo][kk], color=COL[t], lw=1.0)
        lines.append(f"{NAME[t]} {p['label']}, W$_1$ {p['w1']:.4f}")
    if ref["eps"] is not None:
        ax.scatter(ref["eps"], [0] * len(ref["eps"]), marker="^", s=22, color=INK,
                   zorder=5, clip_on=False)
    ax.set_title(f"{name}  (true n = {ref['n_true']})\n" + "\n".join(lines),
                 loc="left", pad=4, fontsize=8)
    ax.set_ylim(bottom=-0.02)
    axr.axhline(0, color=INK2, lw=0.5)
    axr.set_xlabel(r"$\omega$")
    if c == 0:
        ax.set_ylabel(r"$A(\omega)$")
        axr.set_ylabel("recon − truth")
    for a in (ax, axr):
        a.grid(color=GRID, lw=0.5)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)

# 8th slot: W1 summary
ax = fig.add_subplot(gs[2:, 3])
y = np.arange(len(ORDER))[::-1]
for t, dy, mk in (("s1_metric", 0.12, "o"), ("M1_60k", -0.12, "s")):
    w = [R[t][n]["w1"] for n in ORDER]
    ax.scatter(w, y + dy, color=COL[t], marker=mk, s=42, zorder=3, label=NAME[t],
               edgecolors="white", linewidths=1.2)
    for wi, yi in zip(w, y + dy):
        ax.annotate(f"{wi:.4f}", (wi, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=6.8, color=INK2)
ax.axvline(np.pi / 8, color=INK2, lw=0.8, ls=":")
ax.text(np.pi / 8 * 0.92, -0.75, "thermal\nresolution\nπ/β", ha="right", fontsize=6.8, color=INK2)
ax.set_xscale("log")
ax.set_xlim(1e-4, 1.0)
ax.set_yticks(y)
ax.set_yticklabels(ORDER, fontsize=7.5)
ax.set_ylim(-1, len(ORDER))
ax.set_xlabel("W$_1$ to truth (physical units, log)")
ax.set_title("Accuracy per probe", loc="left", pad=4)
ax.grid(axis="x", color=GRID, lw=0.5, which="both")
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=INK, lw=2.4, label="truth"),
                    Line2D([], [], color=COL["s1_metric"], lw=1.3, ls="--", label="s1_metric"),
                    Line2D([], [], color=COL["M1_60k"], lw=1.3, ls="--", label="M1-60k")],
           loc="upper right", ncol=3, frameon=False, fontsize=8.5)
fig.suptitle("Pole-type spectra at noise 1e-4, beta 8: full-pipeline reconstructions "
             "(template menu winner) vs. truth  —  triangles mark the true pole positions",
             fontsize=10, x=0.01, ha="left")
out = D / "pole_reconstructions.png"
fig.savefig(out, dpi=190, facecolor="white")
print(f"saved {out}")
