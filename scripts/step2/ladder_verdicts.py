"""Discrete-vs-continuum verdicts on the banked rank-scan ladders.

Applies physics.consolidate.ladder_verdict to the dual-output chi2(n)
traces (fano band, asym triplet, Holstein) and renders them next to two
noiseless label-bank ladders that exemplify the exact dichotomy (a
pole-closed crash vs a smooth family's geometric decay).  Artifacts:
out/step2/ladder_verdicts.{json,png}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spevae.data.label_bank import LabelBank
from spevae.physics import ladder_verdict

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                     "axes.labelsize": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})

OUT = Path("out/step2")
FLOORS = {"fano band": 1.0, "asym triplet": 1.0, "Holstein": 0.175}

d = json.load(open("out/step1/dual_output.json"))
cases = {c["name"]: c["scan"] for c in d["cases"]}

report = {}
fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.0), constrained_layout=True)
for ax, name in zip(axes, ("asym triplet", "fano band", "Holstein")):
    s = cases[name]
    floor = FLOORS[name]
    v = ladder_verdict(s["ns"], s["chi2s"], floor_target=floor)
    report[name] = {"ns": s["ns"], "chi2s": s["chi2s"], "floor": floor,
                    "verdict": v.verdict, "n_arrival": v.n_arrival,
                    "r_arrival": v.r_arrival, "shelf": v.shelf,
                    "cliff_share": v.cliff_share}
    ax.semilogy(s["ns"], s["chi2s"], "o-", color="#4269d0", ms=4, lw=1.4)
    ax.axhspan(floor / 2, 2 * floor, color="0.5", alpha=0.18, lw=0)
    ax.axvline(v.n_arrival, color="0.3", ls=":", lw=1)
    ax.set_title(f"{name}\n{v.verdict} (arrive n={v.n_arrival} "
                 f"x{v.r_arrival:.1f}, shelf {v.shelf})", fontsize=8.5)
    ax.set_xlabel("pole count n")
    print(f"{name:14s}: {v}")
axes[0].set_ylabel(r"whitened $\chi^2$ per dof")

# noiseless dichotomy references, both built with the PURE data ladder
# (no smoothness budget -- the budget shapes the bank ladders near 1e-6,
# which is protocol, not physics) and classified AT the quietest training
# noise: "what a perfect experiment at the best realistic precision
# concludes".
import math

from spevae.data.label_bank import (GAMMA_MIN, SmoothTruth, _aaa_seed_sets,
                                    _polish, _rel_resid, _seed_cfg,
                                    _split_heaviest)
from spevae.physics import PoleSet

NOISE_FLOOR = 3e-5
beta = 8.0
om = (2.0 * np.arange(64) + 1.0) * math.pi / beta
gfloor = 1.05 * GAMMA_MIN / beta


def pure_ladder(F, n_max=10):
    seeds = _aaa_seed_sets(om, F)
    rungs, prev = [], None
    for n in range(1, n_max + 1):
        if n in seeds:
            cfg0 = _seed_cfg(*seeds[n][:2], gfloor)
        elif prev is not None and prev[0].size == n - 1:
            cfg0 = _split_heaviest(prev, gfloor)
        else:
            continue
        cfg = _polish(om, F, cfg0, gfloor)
        rungs.append((n, _rel_resid(om, F, *cfg)))
        prev = cfg
    return rungs


semi = SmoothTruth("semicircle", beta, {"centre": 0.0, "radius": 2.0,
                                        "sigma_res": GAMMA_MIN / beta,
                                        "W": 2.0})
lad = pure_ladder(semi.matsubara()[1])
v_geo = ladder_verdict([n for n, _ in lad], [r for _, r in lad],
                       floor_target=NOISE_FLOOR)
ax = axes[3]
ax.semilogy([n for n, _ in lad], [r for _, r in lad], "o-",
            color="#3ca951", ms=4, lw=1.4)
ax.axhline(NOISE_FLOOR, color="0.4", ls=":", lw=1)
ax.set_title(f"reference: semicircle, pure ladder\n(noiseless: "
             f"{v_geo.verdict} at the 3e-5 floor)", fontsize=8.5)
ax.set_xlabel("rank n")
ax.set_ylabel("Matsubara residual")

# a pole-closed noiseless crash: the exact 3-pole truth
truth = PoleSet(eps=torch.tensor([[-2.0, 0.4, 2.4]]),
                gamma=torch.tensor([[0.5, 0.3, 0.7]]),
                a=torch.tensor([[0.3, 0.45, 0.25]]),
                b=torch.tensor([[0.06, -0.02, -0.04]]))
zp = torch.complex(truth.eps, -truth.gamma)[0].numpy()
rp = torch.complex(truth.a, truth.b)[0].numpy()
F = (rp[None, :] / (1j * om[:, None] - zp[None, :])).sum(1)
crash = pure_ladder(F, n_max=6)
ax = axes[4]
ax.semilogy([n for n, _ in crash], [r for _, r in crash], "o-",
            color="#ff725c", ms=4, lw=1.4)
ax.axhline(NOISE_FLOOR, color="0.4", ls=":", lw=1)
v_crash = ladder_verdict([n for n, _ in crash], [r for _, r in crash],
                         floor_target=NOISE_FLOOR)
ax.set_title(f"reference: exact 3-pole truth\n(noiseless crash: "
             f"{v_crash.verdict} at n={v_crash.n_arrival})", fontsize=8.5)
ax.set_xlabel("rank n")
report["_references"] = {
    "noise_floor": NOISE_FLOOR,
    "semicircle_ladder": lad, "semicircle_verdict": v_geo.verdict,
    "three_pole_crash": crash, "crash_verdict": v_crash.verdict}

fig.suptitle("Rank-ladder shape verdicts: pole-closed vs continuum "
             "(banked dual-output scans + noiseless dichotomy references)",
             fontsize=10)
fig.savefig(OUT / "ladder_verdicts.png", dpi=150)
with open(OUT / "ladder_verdicts.json", "w") as fh:
    json.dump(report, fh, indent=1)
print(f"-> {OUT}/ladder_verdicts.png, .json")
