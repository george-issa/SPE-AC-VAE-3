"""The three error laws, read off the sweep data. Guide lines, no fits:
quadratic (detection evidence ~ 1/sigma^2), linear (identified-parameter
error ~ sigma), logarithmic (free-pole arrangement resolution ~ 1/ln)."""
import json, sys, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
store = json.load(open("out/microsweep/template_sweep.json"))
free_rows = json.load(open("out/microsweep/noise_sweep.json"))
CASES = {"3-rung": ("ladder", 3, [0.5, 1.5, 2.5]),
         "4-rung": ("ladder", 4, [-0.4, 0.4, 1.2, 2.0]),
         "mirrored": ("mirrored_ladder", 2, [-2.2, -1.2, 1.2, 2.2])}
COLORS = {"3-rung": "#0E7C86", "4-rung": "#B8451F", "mirrored": "#5A6572"}
L = 65
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), constrained_layout=True)

for case, (fam, m_true, tpoles) in CASES.items():
    rows = sorted([r for r in store["results"] if r["case"] == case],
                  key=lambda r: -r["noise"])
    nz = [r["noise"] for r in rows]
    # (A) detection evidence: true-m template vs (m_true - 1)
    ev = []
    for r in rows:
        c = {(c["name"], c["m"]): c["chi2"] for c in r["menu"]}
        lo = c.get((fam, m_true - 1)); hi = c.get((fam, m_true))
        ev.append((lo - hi) * L if lo and hi else float("nan"))
    axes[0].plot(nz, ev, "-o", color=COLORS[case], ms=4, lw=1.2, label=case)
    # (B) linear: winner W1 (identified-structure error)
    axes[1].plot(nz, [r["winner"]["w1"] for r in rows], "-o",
                 color=COLORS[case], ms=4, lw=1.2)
    # (C) faint-pole position error: template winner vs free poles
    perr = []
    for r in rows:
        pe = [e for e, a in zip(r["winner"]["pole_eps"], r["winner"]["pole_a"])
              if a > 1e-3]
        perr.append(max(min(abs(p - t) for p in pe) for t in tpoles)
                    if pe else float("nan"))
    axes[2].plot(nz, perr, "-o", color=COLORS[case], ms=4, lw=1.2)
    fr = sorted([r for r in free_rows if r["case"] == case], key=lambda r: -r["noise"])
    axes[2].plot([r["noise"] for r in fr], [r["max_pos_err"] for r in fr], "--s",
                 color=COLORS[case], ms=3.5, lw=0.9, alpha=0.55)

# guide lines (anchored by eye, slopes exact)
import numpy as np
s = np.array([1e-2, 1e-5])
axes[0].plot(s, 9.9e3 * (1e-4 / s) ** 2, ":", color="#999", lw=1.2)
axes[0].text(2.5e-4, 3e5, "slope $-2$\n(evidence $\\propto \\sigma^{-2}$)",
             fontsize=7, color="#777")
axes[0].axhline(9.49, color="#B8451F", lw=0.7, ls="--", alpha=0.6)
axes[0].text(2e-3, 13, "$\\chi^2_4$ threshold", fontsize=6.5, color="#B8451F")
axes[1].plot(s, 2e-4 * (s / 1e-4), ":", color="#999", lw=1.2)
axes[1].text(1e-3, 1.5e-3, "slope $+1$\n(error $\\propto \\sigma$)",
             fontsize=7, color="#777")
sl = np.geomspace(1e-2, 1e-5, 50)
axes[2].plot(sl, 3.2 / np.log(1.0 / sl), ":", color="#999", lw=1.2)
axes[2].text(1.2e-3, 0.55, "$\\propto 1/\\ln(1/\\sigma)$", fontsize=7, color="#777")

for ax, title, ylab in [(axes[0], "detection: rung-count evidence", "$\\Delta\\chi^2_{tot}$"),
                        (axes[1], "identification: template W$_1$", "W$_1$"),
                        (axes[2], "resolution: faint-pole position error",
                         "max $|\\Delta\\varepsilon|$")]:
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("noise $\\sigma$"); ax.set_title(title, pad=5)
    ax.set_ylabel(ylab); ax.grid(alpha=0.2, lw=0.4)
axes[0].legend(fontsize=7, frameon=False, loc="lower left")
from matplotlib.lines import Line2D
axes[2].legend(handles=[Line2D([], [], color="#666", ls="-", marker="o", ms=3.5,
                               label="template (linear law)"),
                        Line2D([], [], color="#666", ls="--", marker="s", ms=3,
                               alpha=0.55, label="free poles (log wall)")],
               fontsize=7, frameon=False, loc="lower left")
fig.suptitle("The three error laws, read off the data (guide lines, no fits): "
             "quadratic detection, linear identification, logarithmic resolution",
             fontsize=10)
fig.savefig(sys.argv[1], dpi=200)
print("figure ->", sys.argv[1])
