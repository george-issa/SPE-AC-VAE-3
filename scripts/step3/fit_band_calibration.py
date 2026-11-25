"""Fit and test the band table of a checkpoint from a finished calibration run.

The flow's samples are over-dispersed, so their plain quantiles make bands
that are too wide.  From held-out truths of the training prior this script
measures, for every truth and window, the central level of the sample
ensemble at which the truth is just covered, and takes the level to quote
for a target coverage as the matching quantile -- pooled (`--strata none`),
per band of relative error bar (`noise`), or per band of error bar and beta
(`noise+beta`).

The table is fitted on the even-numbered truths of every cell and tested on
the odd ones; all three variants are printed.  The table that is written is
then refitted on all truths, with the split's test results stored inside it.

    python scripts/step3/fit_band_calibration.py out/step3/cal_F3_60k_s1_big out/step3/F3_60k_s1.pt [--strata noise] [--arm posterior]

`--unflagged-only` fits and tests on the truths whose report does not flag
narrow lines: the report withholds the table when it does (and states both
readings of the lines instead), so those truths are not what the table
corrects.  Use it for a model whose prior holds exact line spectra.

With `--cv` (and any number of further runs through `--also DIR`) the test is
leave-one-repetition-out instead: each repetition of the design (288 truths
with their own pairing of noise, beta and dtau) is scored with a table fitted
on all the others.  Read it with care: pooled over repetitions, coverage in
a stratum the table itself has is nominal almost by construction.  The lines
that test something are the partitions the table does NOT have (beta, for a
table by error bar) and the spread over repetitions.  `--evaluate` fits nothing and scores the table already
beside the checkpoint on the given run.

Writes `<flow.pt>.bands.json` (picked up by `load_flow`) and
`<cal dir>/band_calibration.{json,png}`.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spevae.posterior import BAND_TARGETS, band_levels, fit_band_table

NOISE_EDGES = (1e-4, 1e-3, 1e-2)
BETA_EDGES = (2.5, 10.0)


def load(cals, arm, unflagged_only=False):
    """Per truth: the central level that just covers each window weight, its error bar, beta,
    parity and repetition.  With `unflagged_only`, truths whose report flags narrow lines are
    left out: the report withholds the table from those, so they are not what it corrects."""
    rows = []
    for cal in cals:
        for p in sorted(Path(cal).glob("calibration.shard*.json")):
            rows += json.load(open(p))
    rows.sort(key=lambda r: (r["noise"], r.get("rep", 0)))
    rng = np.random.default_rng(17)
    need, noise, beta, half, fold = [], [], [], [], []
    for r in rows:
        wt = np.array(r["w_true"])
        for m, a in enumerate(r["arms"][arm]):
            w = np.array(a["w"])
            pit = ((w < wt[m]).sum(0) + rng.random(wt.shape[1])) / (len(w) + 1)
            if unflagged_only and a.get("line_shape"):
                continue                                  # after the draw: the other truths keep their ranks
            need.append(np.abs(2.0 * pit - 1.0))
            noise.append(r["rel_noise"][m] if "rel_noise" in r else r["noise"])
            beta.append(r["beta"])
            half.append(m % 2)
            fold.append(r.get("rep", 0))
    return np.array(need), np.array(noise), np.array(beta), np.array(half), np.array(fold)


def hits_of(table, need, noise, beta):
    """Per target: (n_truths, n_stats) booleans, truth inside the band the table gives."""
    lev = [band_levels(table, n, b) for n, b in zip(noise, beta)]
    return {t: need < np.array([l[t] for l in lev])[:, None] for t in table["targets"]}


def coverage(hits, sel):
    return {t: (float(h[sel].mean()), float(h[sel].mean(1).std() / np.sqrt(max(sel.sum(), 1))))
            for t, h in hits.items()}


def report(name, table, need, noise, beta, test, hits=None, fold=None):
    hits = hits_of(table, need, noise, beta) if hits is None else hits
    res = {"all": coverage(hits, test), "noise": {}, "beta": {}}
    if fold is not None:
        res["repetition"] = {str(k): {"n": int((test & (fold == k)).sum()), **coverage(hits, test & (fold == k))}
                             for k in sorted(set(fold[test].tolist()))}
    ne = (0.0,) + NOISE_EDGES + (np.inf,)
    for lo, hi in zip(ne[:-1], ne[1:]):
        sel = test & (noise >= lo) & (noise < hi)
        if sel.sum():
            res["noise"][f"{lo:g}-{hi:g}"] = {"n": int(sel.sum()), **coverage(hits, sel)}
    for b in sorted(set(beta.tolist())):
        sel = test & (beta == b)
        res["beta"][f"{b:g}"] = {"n": int(sel.sum()), **coverage(hits, sel)}
    z = [(v[t][0] - t) / max(v[t][1], 1e-3) for grp in ("noise", "beta") for v in res[grp].values()
         for t in (0.5, 0.9)]
    worst = max(abs(v[t][0] - t) for grp in ("noise", "beta") for v in res[grp].values() for t in (0.5, 0.9))
    res["rms_z"], res["max_abs_z"], res["worst_gap"] = float(np.sqrt(np.mean(np.square(z)))), float(np.max(np.abs(z))), float(worst)
    a = res["all"]
    print(f"{name}: held-out coverage " + "  ".join(f"{t:.2f}->{a[t][0]:.3f}±{a[t][1]:.3f}" for t in hits)
          + f"   strata: rms z {res['rms_z']:.2f}, max |z| {res['max_abs_z']:.2f}, worst gap {worst:.3f}")
    for grp in ("noise", "beta"):
        for k, v in res[grp].items():
            print(f"    {grp:5s} {k:>12s} n={v['n']:4d}: 50% {v[0.5][0]:.3f}±{v[0.5][1]:.3f}   90% {v[0.9][0]:.3f}±{v[0.9][1]:.3f}")
    if "repetition" in res:
        # a repetition scored by a table fitted on the others: the test that is not nominal by construction
        v = list(res["repetition"].values())
        for t in (0.5, 0.9):
            c = np.array([x[t][0] for x in v])
            e = np.array([x[t][1] for x in v])
            res[f"repetition_range_{t}"] = [float(c.min()), float(c.max())]
            print(f"    by repetition ({len(v)} x {v[0]['n']} truths), {int(t * 100)}% band: "
                  + " ".join(f"{x:.3f}" for x in c) + f"   range {c.min():.3f}-{c.max():.3f}, "
                  f"rms z {np.sqrt(np.mean(((c - t) / e) ** 2)):.2f}")
    return res


def jsonable(res):
    if isinstance(res, dict):
        return {str(k): jsonable(v) for k, v in res.items()}
    if isinstance(res, tuple):
        return list(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cal")
    ap.add_argument("flow")
    ap.add_argument("--strata", choices=("none", "noise", "noise+beta"), default="none",
                    help="strata of the table that is written")
    ap.add_argument("--arm", default="posterior")
    ap.add_argument("--min-truths", type=int, default=100)
    ap.add_argument("--evaluate", action="store_true",
                    help="do not fit: test the table already beside the checkpoint on this run")
    ap.add_argument("--also", action="append", default=[], help="further calibration runs to pool")
    ap.add_argument("--cv", action="store_true", help="test by leave-one-repetition-out")
    ap.add_argument("--unflagged-only", action="store_true",
                    help="fit and test on the truths whose report does not flag narrow lines (the ones "
                         "the table is applied to); needs runs made with CAL_FLOOR=auto")
    args = ap.parse_args()
    cal = Path(args.cal)
    need, noise, beta, half, fold = load([cal] + args.also, args.arm, args.unflagged_only)
    if args.evaluate:
        table = json.load(open(args.flow + ".bands.json"))
        every = np.ones(len(need), bool)
        ident = {"targets": list(BAND_TARGETS), "noise_edges": [], "beta_edges": [],
                 "levels": [[list(BAND_TARGETS)]]}
        print(f"{len(need)} truths, none of them used to fit the table ({table['kind']}, from {table['source']})")
        res = {"plain": jsonable(report("plain quantiles     ", ident, need, noise, beta, every)),
               "table": jsonable(report("table as shipped    ", table, need, noise, beta, every))}
        with open(cal / "band_table_test.json", "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"-> {cal}/band_table_test.json")
        return
    ident = {"targets": list(BAND_TARGETS), "noise_edges": [], "beta_edges": [],
             "levels": [[list(BAND_TARGETS)]]}
    variants = (("pooled table        ", ((), ())), ("table by error bar  ", (NOISE_EDGES, ())),
                ("by error bar & beta ", (NOISE_EDGES, BETA_EDGES)))
    if args.cv:
        folds = sorted(set(fold.tolist()))
        test = np.ones(len(need), bool)
        print(f"{len(need)} truths in {len(folds)} repetitions, leave-one-repetition-out, arm '{args.arm}'")
        before = report("plain quantiles     ", ident, need, noise, beta, test)
        out = []
        for name, ed in variants:
            hits = {t: np.zeros(need.shape, bool) for t in BAND_TARGETS}
            for k in folds:
                tr, te = fold != k, fold == k
                tab = fit_band_table(need[tr], noise[tr], beta[tr], *ed, min_truths=args.min_truths)
                for t, h in hits_of(tab, need[te], noise[te], beta[te]).items():
                    hits[t][te] = h
            out.append(report(name, None, need, noise, beta, test, hits=hits, fold=fold))
        r_pool, r_noise, r_str = out
    else:
        fit, test = half == 0, half == 1
        print(f"{len(need)} truths ({fit.sum()} to fit, {test.sum()} to test), arm '{args.arm}'")
        before = report("plain quantiles     ", ident, need, noise, beta, test)
        half_min = max(args.min_truths // 2, 20)
        r_pool, r_noise, r_str = (
            report(name, fit_band_table(need[fit], noise[fit], beta[fit], *ed, min_truths=half_min),
                   need, noise, beta, test) for name, ed in variants)
    edges = {"none": ((), ()), "noise": (NOISE_EDGES, ()), "noise+beta": (NOISE_EDGES, BETA_EDGES)}[args.strata]
    final = fit_band_table(need, noise, beta, *edges, min_truths=args.min_truths)
    chosen = {"none": r_pool, "noise": r_noise, "noise+beta": r_str}[args.strata]
    kind = {"none": "pooled", "noise": "per-error-bar", "noise+beta": "per-error-bar-and-beta"}[args.strata]
    final.update(kind=kind, arm=args.arm, truths_total=int(len(need)),
                 fitted_on=("truths whose report does not flag narrow lines" if args.unflagged_only
                            else "all truths"),
                 source=", ".join([str(cal)] + args.also),
                 test_protocol="leave-one-repetition-out" if args.cv else "fitted on even truths, tested on odd",
                 held_out_test=jsonable(chosen), held_out_plain=jsonable(before))
    with open(args.flow + ".bands.json", "w") as fh:
        json.dump(final, fh, indent=1)
    with open(cal / "band_calibration.json", "w") as fh:
        json.dump({"plain": jsonable(before), "pooled": jsonable(r_pool), "by_noise": jsonable(r_noise),
                   "by_noise_and_beta": jsonable(r_str), "table": final}, fh, indent=1)
    print("levels written (" + final["kind"] + "): pooled " + ", ".join(
        f"{t:.2f}->{v:.3f}" for t, v in zip(final["targets"], final["pooled"])))

    f, axes = plt.subplots(1, 3, figsize=(16, 4.0), constrained_layout=True)
    ax = axes[0]
    ts = np.linspace(0.02, 0.98, 49)
    ax.plot(ts, [np.quantile(need, t) for t in ts], color="#2a78d6", lw=1.6, label="level to quote (all truths)")
    ax.plot([0, 1], [0, 1], color="#52514e", ls="--", lw=0.9, label="calibrated samples")
    for t, v in zip(final["targets"], final["pooled"]):
        ax.plot([t], [v], "o", color="#0b0b0b", ms=4)
        ax.annotate(f"{v:.2f}", (t, v), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("coverage wanted")
    ax.set_ylabel("central level of the sample ensemble")
    ax.set_title("which central band of the samples holds the truth that often", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    for ax, grp, xl in ((axes[1], "noise", "relative error bar"), (axes[2], "beta", "beta")):
        keys = list(before[grp])
        x = np.arange(len(keys))
        for t, mk in ((0.5, "o"), (0.9, "s")):
            ax.errorbar(x - 0.08, [before[grp][k][t][0] for k in keys], yerr=[before[grp][k][t][1] for k in keys],
                        color="#eb6834", marker=mk, ms=4, lw=1.1, capsize=2,
                        label="plain sample quantiles" if t == 0.5 else None)
            ax.errorbar(x + 0.08, [chosen[grp][k][t][0] for k in keys], yerr=[chosen[grp][k][t][1] for k in keys],
                        color="#0f8a8a", marker=mk, ms=4, lw=1.1, capsize=2,
                        label=f"with the {final['kind']} band table" if t == 0.5 else None)
            ax.axhline(t, color="#52514e", ls="--", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=8)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel(xl)
        ax.set_title(f"held-out coverage of the 50% and 90% bands, by {xl}", fontsize=9.5)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    for ax in axes:
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    a = chosen["all"]
    how = "each repetition scored by a table fitted on the others" if args.cv else "fitted on half, tested on the other half"
    f.suptitle(f"Band calibration ({len(need)} held-out truths, {how}): "
               f"50% / 90% bands hold the truth {a[0.5][0]:.2f} / {a[0.9][0]:.2f} of the time "
               f"(plain quantiles: {before['all'][0.5][0]:.2f} / {before['all'][0.9][0]:.2f})", fontsize=10.5)
    f.savefig(cal / "band_calibration.png", dpi=150)
    print(f"-> {args.flow}.bands.json, {cal}/band_calibration.json + .png")


if __name__ == "__main__":
    main()
