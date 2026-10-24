"""Build the step-2 smooth-family label bank, judge it, render the figures.

Step-2 build order item (1).  Produces, under out/step2/:
  label_bank.pt           -- the bank (labels + params + ladders + meta)
  label_bank_judge.npz    -- per-entry judge arrays (resid, G gap, W1, ...)
  label_bank_report.json  -- the pinned per-family numbers
  label_bank_overlays.png -- truth vs label spectra per family
  label_bank_ladders.png  -- rank ladders: residual vs n per family
  label_bank_stats.png    -- n, residual, G gap, W1, width, dip distributions

Judge metrics per entry, none of them pointwise-A (the kernel has a null
space; G-space and transport metrics are the meaningful ones):
  resid  -- relative Matsubara residual of the label (from the extraction)
  g_gap  -- max |G_label - G_truth| through the standard forward map
  w1     -- physical-units W1 between label and truth spectra on [-25, 25]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch

from spevae.data.label_bank import (DIP_TOL, GAMMA_MIN, SMOOTH_FAMILIES,
                                    LabelBank, build_label_bank)
from spevae.physics.forward import PoleGreens, SpectralQuadrature
from spevae.physics.poles import PoleSet, spectral_from_poles

FAMILY_COLOR = {                     # fixed categorical order, never cycled
    "gauss_mixture": "#4269d0",
    "gapped_pair": "#efb118",
    "semicircle": "#3ca951",
    "box": "#ff725c",
    "pseudogap": "#9c6b4e",
}


def judge_entry(args):
    """(family, beta, params, label arrays) -> (g_gap, w1, bsat)."""
    from spevae.data.label_bank import SmoothTruth
    family, beta, params, eps, gam, a, b = args
    truth = SmoothTruth(family, beta, params)
    quad = SpectralQuadrature(nu_c=beta * 20.0, h=GAMMA_MIN / 3.5, n_tail=32,
                              dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    Ad = torch.tensor(truth.A(quad.nodes.numpy() / beta)) / beta
    K = PoleGreens(quad).kernel_matrix(x).double()
    G_t = (Ad * quad.weights.double()) @ K
    te, tg, ta, tb = (torch.tensor(v).unsqueeze(0) for v in (eps, gam, a, b))
    G_l = PoleGreens(quad)(PoleSet(eps=te * beta, gamma=tg * beta,
                                   a=ta, b=tb), x)[0]
    g_gap = float((G_l - G_t).abs().max())

    w = torch.linspace(-25.0, 25.0, 4001, dtype=torch.float64)
    At = torch.tensor(truth.A(w.numpy()))
    Al = spectral_from_poles(te, tg, ta, tb, w)[0]
    dw = float(w[1] - w[0])
    w1 = float((torch.cumsum(Al - At, 0) * dw).abs().sum() * dw)
    bsat = float(np.max(np.abs(b) / np.maximum(a, 1e-30))) if a.size else 0.0
    return g_gap, w1, bsat


def quantiles(v):
    v = np.asarray(v, float)
    return {"median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)),
            "max": float(v.max()), "min": float(v.min())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-family", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--beta-range", type=float, nargs=2, default=(2.0, 20.0))
    ap.add_argument("--p-max", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/step2"))
    ap.add_argument("--bank", type=Path, default=None,
                    help="judge an existing bank instead of building")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.bank is not None:
        bank = LabelBank.load(args.bank)
        print(f"loaded {args.bank}: {len(bank)} entries")
    else:
        t0 = time.time()

        def prog(done, total):
            if done % 50 == 0 or done == total:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  extracted {done}/{total} "
                      f"({rate:.1f}/s, eta {(total-done)/max(rate,1e-9):.0f}s)",
                      flush=True)

        bank = build_label_bank(n_per_family=args.n_per_family,
                                seed=args.seed, beta_range=tuple(args.beta_range),
                                P_max=args.p_max, workers=args.workers,
                                progress=prog)
        bank.save(args.out / "label_bank.pt")
        print(f"built {len(bank)} labels in {time.time()-t0:.0f}s "
              f"-> {args.out/'label_bank.pt'}")

    # ---- judge pass -------------------------------------------------------- #
    t0 = time.time()
    jobs = []
    for i in range(len(bank)):
        m = int(bank.n_true[i])
        jobs.append((bank.family[i], float(bank.beta[i]), bank.params[i],
                     bank.eps[i, :m].double().numpy(),
                     bank.gamma[i, :m].double().numpy(),
                     bank.a[i, :m].double().numpy(),
                     bank.b[i, :m].double().numpy()))
    if args.workers > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            judged = list(ex.map(judge_entry, jobs, chunksize=8))
    else:
        judged = [judge_entry(j) for j in jobs]
    g_gap = np.array([j[0] for j in judged])
    w1 = np.array([j[1] for j in judged])
    bsat = np.array([j[2] for j in judged])
    resid = bank.resid.numpy()
    n_true = bank.n_true.numpy()
    beta = bank.beta.numpy()
    negw = bank.neg_weight.numpy()
    min_gd = np.array([
        float((bank.gamma[i, :int(bank.n_true[i])].min()) * bank.beta[i])
        for i in range(len(bank))])
    fam_idx = np.array([SMOOTH_FAMILIES.index(f) for f in bank.family])
    np.savez(args.out / "label_bank_judge.npz", resid=resid, g_gap=g_gap,
             w1=w1, bsat=bsat, n_true=n_true, beta=beta, neg_weight=negw,
             min_gamma_dimless=min_gd, family_index=fam_idx,
             families=np.array(SMOOTH_FAMILIES))
    print(f"judged in {time.time()-t0:.0f}s")

    # ---- report ------------------------------------------------------------ #
    report = {"n_entries": len(bank), "meta": {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in bank.meta.items() if k != "protocol"},
        "protocol": bank.meta.get("protocol", {}), "per_family": {}}
    for f in SMOOTH_FAMILIES:
        m = fam_idx == SMOOTH_FAMILIES.index(f)
        report["per_family"][f] = {
            "n_true": quantiles(n_true[m]),
            "resid": quantiles(resid[m]),
            "g_gap": quantiles(g_gap[m]),
            "w1": quantiles(w1[m]),
            "neg_weight_max": float(negw[m].max()),
            "frac_below_noise_floor": float((g_gap[m] < 3e-5).mean()),
            "min_gamma_dimless": float(min_gd[m].min()),
            "b_saturation_p95": float(np.quantile(bsat[m], 0.95)),
        }
    with open(args.out / "label_bank_report.json", "w") as fh:
        json.dump(report, fh, indent=1)

    print(f"\n{'family':14s} {'n med(max)':>11s} {'resid med/p95':>16s} "
          f"{'Ggap med/p95':>16s} {'W1 med':>8s} {'<3e-5':>6s}")
    for f in SMOOTH_FAMILIES:
        r = report["per_family"][f]
        print(f"{f:14s} {r['n_true']['median']:6.0f}({r['n_true']['max']:.0f}) "
              f"  {r['resid']['median']:.1e}/{r['resid']['p95']:.1e} "
              f" {r['g_gap']['median']:.1e}/{r['g_gap']['p95']:.1e} "
              f"{r['w1']['median']:8.4f} {r['frac_below_noise_floor']:6.0%}")

    # ---- figures ----------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                         "axes.labelsize": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.5})

    # overlays: per family, [median resid, p95-ish, worst] entries
    fig, axes = plt.subplots(len(SMOOTH_FAMILIES), 3,
                             figsize=(11.5, 2.3 * len(SMOOTH_FAMILIES)),
                             constrained_layout=True)
    for fi, f in enumerate(SMOOTH_FAMILIES):
        idx = np.where(fam_idx == fi)[0]
        order = idx[np.argsort(resid[idx])]
        picks = [order[len(order) // 2], order[int(len(order) * 0.95)],
                 order[-1]] if len(order) >= 3 else list(order)
        for ci, i in enumerate(picks):
            ax = axes[fi, ci] if axes.ndim == 2 else axes[ci]
            tr = bank.truth(int(i))
            m = int(bank.n_true[i])
            ps = bank.label(int(i))
            hiW = float(tr.params.get("W", 10.0))
            w = np.linspace(-1.6 * hiW - 2, 1.6 * hiW + 2, 1200)
            At = tr.A(w)
            Al = ps.spectral(torch.tensor(w).float())[0].numpy()
            ax.plot(w, At, color=FAMILY_COLOR[f], lw=1.6, label="truth")
            ax.plot(w, Al, color="0.15", lw=1.0, ls="--", label="label")
            ax.plot(bank.eps[i, :m], np.zeros(m), "|", color="0.15", ms=7)
            ax.set_title(f"{f}  $\\beta$={float(bank.beta[i]):.1f}  n={m}  "
                         f"resid {float(resid[i]):.0e}", fontsize=8)
            if fi == 0 and ci == 0:
                ax.legend(frameon=False, fontsize=7.5)
            if fi == len(SMOOTH_FAMILIES) - 1:
                ax.set_xlabel(r"$\omega$")
    fig.suptitle("Label bank: truth vs canonical MPM label "
                 "(median / p95 / worst residual per family)", fontsize=10)
    fig.savefig(args.out / "label_bank_overlays.png", dpi=150)
    plt.close(fig)

    # ladders
    fig, axes = plt.subplots(1, len(SMOOTH_FAMILIES),
                             figsize=(2.9 * len(SMOOTH_FAMILIES), 3.1),
                             constrained_layout=True, sharey=True)
    for fi, f in enumerate(SMOOTH_FAMILIES):
        ax = axes[fi]
        idx = np.where(fam_idx == fi)[0]
        grid = np.full((len(idx), bank.P_max), np.nan)
        for k, i in enumerate(idx):
            for n, r in bank.ladders[i]:
                grid[k, n - 1] = r
            ax.plot(np.arange(1, bank.P_max + 1), grid[k],
                    color=FAMILY_COLOR[f], alpha=0.12, lw=0.7)
        med = np.nanmedian(grid, axis=0)
        ax.plot(np.arange(1, bank.P_max + 1), med, color="0.1", lw=1.8)
        ax.axhspan(3e-5, 3e-2, color="0.5", alpha=0.12, lw=0)
        ax.set_yscale("log")
        ax.set_title(f)
        ax.set_xlabel("rank n")
        if fi == 0:
            ax.set_ylabel("Matsubara residual")
            ax.text(bank.P_max, 1e-3, "training noise range", ha="right",
                    fontsize=7, color="0.35")
    fig.suptitle("Rank ladders: residual vs pole count "
                 "(thin: entries, bold: median) -- rank grows with SNR",
                 fontsize=10)
    fig.savefig(args.out / "label_bank_ladders.png", dpi=150)
    plt.close(fig)

    # distributions
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.6), constrained_layout=True)
    panels = [
        ("n_true", n_true, None, np.arange(0.5, bank.P_max + 1.5, 1), None),
        ("log10 Matsubara residual", np.log10(np.maximum(resid, 1e-16)), None,
         24, None),
        ("log10 max|dG| (G-space gap)", np.log10(np.maximum(g_gap, 1e-12)),
         math.log10(3e-5), 24, "quietest training noise"),
        ("W1(label, truth), physical", w1, None, 24, None),
        ("min dimensionless width", min_gd, GAMMA_MIN, 24, "width floor"),
        ("negative weight", np.log10(np.maximum(negw, 1e-9)),
         math.log10(DIP_TOL), 24, "dip bound"),
    ]
    for ax, (title, v, guide, bins, gname) in zip(axes.ravel(), panels):
        for fi, f in enumerate(SMOOTH_FAMILIES):
            ax.hist(v[fam_idx == fi], bins=bins, histtype="step",
                    color=FAMILY_COLOR[f], lw=1.3, label=f)
        if guide is not None:
            ax.axvline(guide, color="0.2", ls=":", lw=1.2)
            ax.text(guide, ax.get_ylim()[1] * 0.95, f" {gname}", fontsize=7,
                    va="top", color="0.25")
        ax.set_title(title)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Label bank distributions ({len(bank)} entries, seed "
                 f"{bank.meta.get('seed')})", fontsize=10)
    fig.savefig(args.out / "label_bank_stats.png", dpi=150)
    plt.close(fig)
    print(f"figures -> {args.out}/label_bank_{{overlays,ladders,stats}}.png")


if __name__ == "__main__":
    main()
