"""Does P_max = 12 still capacity-match at large beta?  (Design number for
the beta-extension arm; the bank's n_true pile-up at the cap for beta > 13
is the warning this script turns into a measurement.)

For each family and beta, extract labels with the ladder allowed to run to
n = 20 and measure (i) the rank the truth actually selects, (ii) the cost
of capping at 12: ladder residual at rung 12 over the uncapped label
residual.  Artifacts: out/step2/capacity_vs_beta.{json,png}.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch

from spevae.data.label_bank import (SMOOTH_FAMILIES, extract_smooth_label,
                                    sample_smooth_truth)

BETAS = (8.0, 12.0, 16.0, 20.0)
N_DRAWS = 8
N_MAX = 20
OUT = Path("out/step2")

FAMILY_COLOR = {"gauss_mixture": "#4269d0", "gapped_pair": "#efb118",
                "semicircle": "#3ca951", "box": "#ff725c",
                "pseudogap": "#9c6b4e"}


def one(args):
    fam, beta, params = args
    from spevae.data.label_bank import SmoothTruth
    lab = extract_smooth_label(SmoothTruth(fam, beta, params), P_max=N_MAX)
    res = dict(lab.ladder)
    res12 = res.get(12, min(r for n, r in lab.ladder if n <= 12))
    return {"family": fam, "beta": beta, "n": lab.n, "resid": lab.resid,
            "resid12": res12, "cap_cost": res12 / max(lab.resid, 1e-300),
            "ladder": lab.ladder}


def main():
    rng = torch.Generator().manual_seed(11)
    jobs = []
    for beta in BETAS:
        for fam in SMOOTH_FAMILIES:
            for _ in range(N_DRAWS):
                tr = sample_smooth_truth(fam, beta, rng)
                jobs.append((fam, beta, tr.params))

    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, jobs, chunksize=2))

    # report
    report = {"n_max": N_MAX, "betas": list(BETAS), "n_draws": N_DRAWS,
              "cells": {}}
    for beta in BETAS:
        for fam in SMOOTH_FAMILIES:
            sel = [r for r in rows if r["family"] == fam and r["beta"] == beta]
            ns = [r["n"] for r in sel]
            cc = [r["cap_cost"] for r in sel]
            report["cells"][f"{fam}@{beta:g}"] = {
                "n_med": float(np.median(ns)), "n_max": int(max(ns)),
                "frac_over_12": float(np.mean([n > 12 for n in ns])),
                "cap_cost_med": float(np.median(cc)),
                "cap_cost_max": float(np.max(cc)),
                "resid12_max": float(np.max([r["resid12"] for r in sel]))}
    over = {b: float(np.mean([r["n"] > 12 for r in rows if r["beta"] == b]))
            for b in BETAS}
    cost = {b: float(np.median([r["cap_cost"] for r in rows if r["beta"] == b]))
            for b in BETAS}
    report["frac_over_12_by_beta"] = over
    report["cap_cost_med_by_beta"] = cost
    with open(OUT / "capacity_vs_beta.json", "w") as fh:
        json.dump(report, fh, indent=1)

    print(f"{'beta':>6s} {'frac n>12':>10s} {'med cap cost':>13s}")
    for b in BETAS:
        print(f"{b:6.0f} {over[b]:10.0%} {cost[b]:13.2f}")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                         "axes.labelsize": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.5})
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), constrained_layout=True)
    rng2 = np.random.default_rng(0)
    for fam in SMOOTH_FAMILIES:
        sel = [r for r in rows if r["family"] == fam]
        x = np.array([r["beta"] for r in sel]) + rng2.uniform(-0.5, 0.5, len(sel))
        axes[0].plot(x, [r["n"] for r in sel], "o", ms=3.5, alpha=0.6,
                     color=FAMILY_COLOR[fam], label=fam)
        axes[1].semilogy([b + 0.0 for b in BETAS],
                         [np.median([r["cap_cost"] for r in sel
                                     if r["beta"] == b]) for b in BETAS],
                         "o-", color=FAMILY_COLOR[fam], ms=4, lw=1.3)
        axes[2].semilogy(BETAS,
                         [np.max([r["resid12"] for r in sel
                                  if r["beta"] == b]) for b in BETAS],
                         "o-", color=FAMILY_COLOR[fam], ms=4, lw=1.3)
    axes[0].axhline(12, color="0.2", ls=":", lw=1.2)
    axes[0].text(BETAS[0], 12.3, " P_max = 12 (current cap)", fontsize=7.5,
                 color="0.25")
    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel("selected label rank (cap 20)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel(r"$\beta$")
    axes[1].set_ylabel("median resid(12) / resid(n*)")
    axes[1].axhline(1.0, color="0.4", ls=":", lw=1)
    axes[2].set_xlabel(r"$\beta$")
    axes[2].set_ylabel("worst ladder resid at n = 12")
    axes[2].axhspan(3e-5, 3e-2, color="0.5", alpha=0.15, lw=0)
    axes[2].text(BETAS[0], 1e-3, "training noise", fontsize=7.5, color="0.35")
    fig.suptitle(f"Capacity vs beta: uncapped label ranks and the cost of "
                 f"P_max = 12 ({len(rows)} labels, ladder to n = {N_MAX})",
                 fontsize=10)
    fig.savefig(OUT / "capacity_vs_beta.png", dpi=150)
    print(f"-> {OUT}/capacity_vs_beta.png, .json")


if __name__ == "__main__":
    main()
