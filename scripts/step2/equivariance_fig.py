"""Figure for the PH group-averaging test (equivariance_test.py + equivariance_control.py).

Panels: (a) Holstein manifold floor -- plain f, capacity control (2-branch
mixture, no mirror, 3 seeds), mirror-averaged F -- both checkpoints, against the
calibrated perfect-fit band; (b) z-descent W1 per probe, f vs control vs F;
(c) Holstein floor spectra f vs F; (d) one probe's reconstructions.
Usage: equivariance_fig.py   ->  out/step2/equivariance/equivariance.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("out/step2/equivariance")
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
C = {"f": "#5A6572", "ctrl": "#C9A227", "F": "#0E7C86"}
TAGS = ["s1_metric", "M1_60k"]
H = {t: json.load(open(D / f"{t}_holstein.json")) for t in TAGS}
HC = {t: json.load(open(D / f"{t}_control_holstein.json")) for t in TAGS}
P = {t: json.load(open(D / f"{t}_probes.json")) for t in TAGS}
PC = {t: json.load(open(D / f"{t}_control_probes.json")) for t in TAGS}

fig = plt.figure(figsize=(14.5, 8.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.9, 1.2])

# (a) Holstein floor
ax = fig.add_subplot(gs[0, 0])
tgt = H["M1_60k"]["f"]["target"]
ax.axhspan(tgt - 0.061, tgt + 0.061, color="#22703B", alpha=0.12)
ax.axhline(tgt, color="#22703B", lw=0.8, ls="--", label="perfect-fit band")
for i, t in enumerate(TAGS):
    ax.bar(i - 0.27, H[t]["f"]["floor"], 0.25, color=C["f"], label="plain f" if i == 0 else None)
    cs = [HC[t][f"seed{s}"]["floor"] for s in range(3)]
    ax.bar(i, np.median(cs), 0.25, color=C["ctrl"],
           label="2-branch control (no mirror)" if i == 0 else None)
    ax.scatter([i] * 3, cs, s=10, color="#5A4A10", zorder=5)
    ax.bar(i + 0.27, H[t]["F"]["floor"], 0.25, color=C["F"],
           label="mirror-averaged F" if i == 0 else None)
    for dx, v in ((-0.27, H[t]["f"]["floor"]), (0.27, H[t]["F"]["floor"])):
        ax.text(i + dx, v * 1.12, f"{v:.3g}", ha="center", fontsize=6.5)
ax.set_yscale("log"); ax.set_xticks([0, 1]); ax.set_xticklabels(TAGS)
ax.set_ylabel(r"manifold floor $\chi^2$ (anchor released)")
ax.set_title("(a) Holstein: decoder-manifold floor", pad=5)
ax.legend(fontsize=6.5, frameon=False, loc="upper right")

# (b) probe z-descent W1
ax = fig.add_subplot(gs[0, 1:])
keys = [k for k in P["M1_60k"] if "@" in k]
xs = np.arange(len(keys))
for j, t in enumerate(TAGS):
    off = -0.2 if j == 0 else 0.2
    mk = "o" if j == 0 else "s"
    wf = [P[t][k]["zd"]["w1_f"] for k in keys]
    wc = [PC[t][k]["w1"] for k in keys]
    wF = [P[t][k]["zd"]["w1_F"] for k in keys]
    for v, col, lab in ((wf, C["f"], "f"), (wc, C["ctrl"], "control"), (wF, C["F"], "F")):
        ax.scatter(xs + off, v, s=16, marker=mk, color=col, zorder=4,
                   label=f"{t} {lab}")
    for xx, a_, b_ in zip(xs + off, wf, wF):
        ax.plot([xx, xx], [a_, b_], color="#999", lw=0.6, zorder=2)
ax.set_yscale("log"); ax.set_xticks(xs)
ax.set_xticklabels([k.replace("@", "\n@") for k in keys], rotation=70, fontsize=6)
ax.set_ylabel("W$_1$ to truth (z-descent 3000)")
ax.set_title("(b) probes: z-descent on the plain manifold (grey), the 2-branch "
             "control (gold), and the mirror-averaged manifold (teal); "
             "circles s1_metric, squares M1-60k", pad=5)
ax.legend(fontsize=6, frameon=False, ncol=6, loc="upper left")

# (c) Holstein floor spectra
ax = fig.add_subplot(gs[1, 0:2])
for t, ls in (("s1_metric", ":"), ("M1_60k", "-")):
    w = np.array(H[t]["f"]["omega"])
    ax.plot(w, H[t]["f"]["A_floor"], color=C["f"], lw=1.1, ls=ls,
            label=f"{t} f floor ($\\chi^2$ {H[t]['f']['floor']:.3g})")
    ax.plot(w, H[t]["F"]["A_floor"], color=C["F"], lw=1.3, ls=ls,
            label=f"{t} F floor ($\\chi^2$ {H[t]['F']['floor']:.3g})")
ax.set_xlim(-6, 6); ax.axhline(0, color="#999", lw=0.4)
ax.set_xlabel(r"$\omega$"); ax.set_ylabel(r"$A(\omega)$")
ax.set_title("(c) Holstein (beta 8, lambda 0.325): best decoder-manifold spectra", pad=5)
ax.legend(fontsize=6.5, frameon=False)

# (d) one probe
ax = fig.add_subplot(gs[1, 2])
k = "edged band@0.0001"
om = np.array(P["M1_60k"]["nodes"])
o = np.argsort(om)
A = P["M1_60k"][k]["A"]
ax.plot(om[o], np.array(A["truth"])[o], color="#1D242C", lw=1.8, label="truth")
ax.plot(om[o], np.array(A["zd_f"])[o], color=C["f"], lw=1.1,
        label=f"f z-descent (W$_1$ {P['M1_60k'][k]['zd']['w1_f']:.3f})")
ax.plot(om[o], np.array(A["zd_F"])[o], color=C["F"], lw=1.3, ls="--",
        label=f"F z-descent (W$_1$ {P['M1_60k'][k]['zd']['w1_F']:.3f})")
ax.set_xlim(-4, 4.5); ax.axhline(0, color="#999", lw=0.4)
ax.set_xlabel(r"$\omega$")
ax.set_title(f"(d) M1-60k, {k}\n(same-size control, no mirror: W$_1$ "
             f"{PC['M1_60k'][k]['w1']:.3f} -- the gain here is capacity)", pad=5, fontsize=8)
ax.legend(fontsize=6.5, frameon=False)

fig.suptitle("PH group averaging (arXiv:2609.27132) at inference on the banked checkpoints: "
             "F(G) = 1/2 f(G) U 1/2 M f(G(beta - tau)), exactly equivariant, no retraining",
             fontsize=10)
fig.savefig(D / "equivariance.png", dpi=190)
print("figure saved")
