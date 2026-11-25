"""Fit and test the band table of a checkpoint from a finished calibration run.

The flow's samples are over-dispersed, so their plain quantiles make bands
that are too wide.  From held-out truths of the training prior this script
measures, for every truth and window, the central level of the sample
ensemble at which the truth is just covered, and takes the level to quote
for a target coverage as the matching quantile -- pooled, or per stratum of
relative error bar and beta (`--strata`).

The table is fitted on the even-numbered truths of every cell and tested on
the odd ones; both variants are printed.  The table that is written is then
refitted on all truths, with the split's test results stored inside it.

    python scripts/step3/fit_band_calibration.py out/step3/cal_F3_60k_s1_big out/step3/F3_60k_s1.pt [--strata] [--arm posterior]

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


def load(cal, arm):
    rows = []
    for p in sorted(cal.glob("calibration.shard*.json")):
        rows += json.load(open(p))
    rows.sort(key=lambda r: (r["noise"], r.get("rep", 0)))
    rng = np.random.default_rng(17)
    need, noise, beta, half = [], [], [], []
    for r in rows:
        wt = np.array(r["w_true"])
        for m, a in enumerate(r["arms"][arm]):
            w = np.array(a["w"])
            pit = ((w < wt[m]).sum(0) + rng.random(wt.shape[1])) / (len(w) + 1)
            need.append(np.abs(2.0 * pit - 1.0))
            noise.append(r["rel_noise"][m] if "rel_noise" in r else r["noise"])
            beta.append(r["beta"])
            half.append(m % 2)
    return np.array(need), np.array(noise), np.array(beta), np.array(half)


def coverage(table, need, noise, beta, sel):
    out = {}
    for t in table["targets"]:
        lev = np.array([band_levels(table, n, b)[t] for n, b in zip(noise[sel], beta[sel])])
        hit = need[sel] < lev[:, None]
        out[t] = (float(hit.mean()), float(hit.mean(1).std() / np.sqrt(max(sel.sum(), 1))))
    return out


def report(name, table, need, noise, beta, test):
    res = {"all": coverage(table, need, noise, beta, test), "noise": {}, "beta": {}}
    ne = (0.0,) + NOISE_EDGES + (np.inf,)
    for lo, hi in zip(ne[:-1], ne[1:]):
        sel = test & (noise >= lo) & (noise < hi)
        if sel.sum():
            res["noise"][f"{lo:g}-{hi:g}"] = {"n": int(sel.sum()), **coverage(table, need, noise, beta, sel)}
    for b in sorted(set(beta.tolist())):
        sel = test & (beta == b)
        res["beta"][f"{b:g}"] = {"n": int(sel.sum()), **coverage(table, need, noise, beta, sel)}
    z = [(v[t][0] - t) / max(v[t][1], 1e-3) for grp in ("noise", "beta") for v in res[grp].values()
         for t in (0.5, 0.9)]
    worst = max(abs(v[t][0] - t) for grp in ("noise", "beta") for v in res[grp].values() for t in (0.5, 0.9))
    res["rms_z"], res["max_abs_z"], res["worst_gap"] = float(np.sqrt(np.mean(np.square(z)))), float(np.max(np.abs(z))), float(worst)
    a = res["all"]
    print(f"{name}: held-out coverage " + "  ".join(f"{t:.2f}->{a[t][0]:.3f}±{a[t][1]:.3f}" for t in table["targets"])
          + f"   strata: rms z {res['rms_z']:.2f}, max |z| {res['max_abs_z']:.2f}, worst gap {worst:.3f}")
    for grp in ("noise", "beta"):
        for k, v in res[grp].items():
            print(f"    {grp:5s} {k:>12s} n={v['n']:4d}: 50% {v[0.5][0]:.3f}±{v[0.5][1]:.3f}   90% {v[0.9][0]:.3f}±{v[0.9][1]:.3f}")
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
    ap.add_argument("--strata", action="store_true", help="per stratum of error bar and beta, not pooled")
    ap.add_argument("--arm", default="posterior")
    ap.add_argument("--min-truths", type=int, default=100)
    args = ap.parse_args()
    cal = Path(args.cal)
    need, noise, beta, half = load(cal, args.arm)
    fit, test = half == 0, half == 1
    ident = {"targets": list(BAND_TARGETS), "noise_edges": [], "beta_edges": [],
             "levels": [[list(BAND_TARGETS)]]}
    print(f"{len(need)} truths ({fit.sum()} to fit, {test.sum()} to test), arm '{args.arm}'")
    before = report("plain quantiles     ", ident, need, noise, beta, test)
    t_pool = fit_band_table(need[fit], noise[fit], beta[fit])
    r_pool = report("pooled table        ", t_pool, need, noise, beta, test)
    t_str = fit_band_table(need[fit], noise[fit], beta[fit], NOISE_EDGES, BETA_EDGES,
                           min_truths=max(args.min_truths // 2, 20))
    r_str = report("stratified table    ", t_str, need, noise, beta, test)
    final = (fit_band_table(need, noise, beta, NOISE_EDGES, BETA_EDGES, min_truths=args.min_truths)
             if args.strata else fit_band_table(need, noise, beta))
    final.update(kind="stratified" if args.strata else "pooled", arm=args.arm, truths_total=int(len(need)),
                 source=str(cal), held_out_test=jsonable(r_str if args.strata else r_pool),
                 held_out_plain=jsonable(before))
    with open(args.flow + ".bands.json", "w") as fh:
        json.dump(final, fh, indent=1)
    with open(cal / "band_calibration.json", "w") as fh:
        json.dump({"plain": jsonable(before), "pooled": jsonable(r_pool), "stratified": jsonable(r_str),
                   "table": final}, fh, indent=1)
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
    chosen = r_str if args.strata else r_pool
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
    f.suptitle(f"Band calibration ({len(need)} held-out truths, fitted on half, tested on the other half): "
               f"50% / 90% bands hold the truth {a[0.5][0]:.2f} / {a[0.9][0]:.2f} of the time "
               f"(plain quantiles: {before['all'][0.5][0]:.2f} / {before['all'][0.9][0]:.2f})", fontsize=10.5)
    f.savefig(cal / "band_calibration.png", dpi=150)
    print(f"-> {args.flow}.bands.json, {cal}/band_calibration.json + .png")


if __name__ == "__main__":
    main()
