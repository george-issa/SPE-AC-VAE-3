"""How far must the width floor be lowered at small error bars?  No retraining.

`floor_factor="auto"` divides the trained width floor by 4 when the samples
pile up at it.  At error bars of 1e-5 a quarter of the floor can still be
wider than the data allow (`edges_summary.py`: half the samples rejected on
11 of 24 exact line spectra, MaxEnt the lower on 22).  Here the posterior
report is run on the same line spectra and noise draws with the floor
divided by a given factor instead, and scored the same way.

    python scripts/step3/floor_depth_probe.py run OUT FLOW BETA NOISE FACTOR [K N]
    python scripts/step3/floor_depth_probe.py fig OUT REFERENCE.json

`run` writes OUT/depth_b<beta>_n<noise>_f<factor>[.shard<K>of<N>].json (K of N
splits the 24 spectra over processes); `fig` compares every factor found in
OUT with the `auto` rows of REFERENCE (a `floor_option_judge.py` result at the
same beta and noise) and with MaxEnt, and writes OUT/floor_depth.{json,png}.
"""
import json
import math
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/step3")
sys.path.insert(0, "scripts/eval")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import exact_models_judge as J
import floor_option_judge as F

COLS = {4.0: "#2a78d6", 16.0: "#0b7a75", 64.0: "#eb6834", 256.0: "#7a3fb0"}
ME, INK = "#7d7c78", "#0b0b0b"


def run(out, flow_path, beta, noise, factor, k, n):
    from spevae.evaluation import cases as C
    from spevae.models.flow import load_flow
    from spevae.posterior import DEFAULT_WINDOWS, continue_posterior
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    flow = load_flow(flow_path).eval()
    items = [it for it in J.make_truths(beta) + F.extra_discrete(beta, 2) if it["kind"] == "discrete"]
    gphys = flow.coords.gamma_min / beta
    rows = []
    for i, item in enumerate(items):
        if i % n != k:
            continue
        inst = C.draw(item["truth"], noise, J.SEED)
        truth_w = F.true_windows(item, DEFAULT_WINDOWS)
        t0 = time.time()
        rep = continue_posterior(flow, inst.G, inst.truth.x, beta, inst.sigma, inst.isr, n_samples=F.S,
                                 omega=J.WG, bounds=False, floor_factor=factor,
                                 generator=torch.Generator().manual_seed(9))
        row = {"index": i, "family": item["family"], "params": item["params"],
               "fit": F.score(rep, item, DEFAULT_WINDOWS, truth_w, gphys), "seconds": time.time() - t0,
               "near_edge": [m for _, m in truth_w]}
        rows.append(row)
        print(f"{i:2d} {item['family']:16s} floor/{factor:g}: W1 {row['fit']['w1']:.4f} accepted "
              f"{row['fit']['accepted']:.0%} chi2 {row['fit']['chi2_median']:.2f} at the floor "
              f"{row['fit']['weight_at_width_floor']:.2f} | {row['seconds']:.0f}s", flush=True)
    tag = f"depth_b{beta:g}_n{noise:g}_f{factor:g}" + (f".shard{k}of{n}" if n > 1 else "")
    with open(out / f"{tag}.json", "w") as fh:
        json.dump({"flow": flow_path, "beta": beta, "noise": noise, "factor": factor, "rows": rows}, fh,
                  indent=1, default=float)
    print(f"-> {out}/{tag}.json")


def ratio_ci(d):
    from scipy import stats
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return [float(np.exp(d.mean())), float(np.exp(d.mean() - h)), float(np.exp(d.mean() + h))]


def gm(v):
    return float(np.exp(np.mean(np.log(v))))


def fig(out, reference):
    out = Path(out)
    ref = json.load(open(reference))
    dis = [x for x in ref["rows"] if x["kind"] == "discrete"]
    arms = {4.0: {i: {"fit": x["auto"], "seconds": x["seconds_auto"], "kind": "discrete", "fired": x["fired"]}
                  for i, x in enumerate(dis)}}
    for p in sorted(out.glob("depth_b*_f*.json")):
        r = json.load(open(p))
        if (r["beta"], r["noise"]) != (ref["beta"], ref["noise"]):
            continue
        for x in r["rows"]:
            assert x["family"] == dis[x["index"]]["family"] and x["params"] == dis[x["index"]]["params"]
            arms.setdefault(float(r["factor"]), {})[x["index"]] = {"fit": x["fit"], "seconds": x["seconds"]}
    fired = [i for i, x in enumerate(dis) if x["fired"]]       # the spectra the report lowers the floor on
    res = {"flow": ref["flow"], "beta": ref["beta"], "noise": ref["noise"], "spectra_flagged": len(fired),
           "spectra": len(dis), "factors": {}}
    for f_, rows in sorted(arms.items()):
        idx = [i for i in fired if i in rows]
        if len(idx) < 2:
            continue
        w = [rows[i]["fit"]["w1"] for i in idx]
        me = [dis[i]["maxent"]["w1"] for i in idx]
        base = [arms[4.0][i]["fit"]["w1"] for i in idx]
        cov = F.coverage([{"kind": "discrete", "near_edge": dis[i]["near_edge"], "x": rows[i]["fit"]} for i in idx],
                         "x", "reported", skip_edge=False)
        res["factors"][f"{f_:g}"] = {
            "n": len(idx), "w1": gm(w), "w1_maxent": gm(me),
            "over_maxent": ratio_ci(np.log(w) - np.log(me)),
            "over_factor_4": ratio_ci(np.log(w) - np.log(base)) if f_ != 4.0 else [1.0, 1.0, 1.0],
            "lower_than_maxent": int(sum(a < b for a, b in zip(w, me))),
            "accepted_median": float(np.median([rows[i]["fit"]["accepted"] for i in idx])),
            "accepted_below_half": int(sum(rows[i]["fit"]["accepted"] < 0.5 for i in idx)),
            "weight_at_floor_median": float(np.median([rows[i]["fit"]["weight_at_width_floor"] for i in idx])),
            "still_at_floor": int(sum(rows[i]["fit"]["weight_at_width_floor"] >= 0.1 for i in idx)),
            "seconds_median": float(np.median([rows[i]["seconds"] for i in idx])),
            "stated_intervals": cov}
    with open(out / "floor_depth.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    fs = [float(k) for k in res["factors"]]
    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(18, 5.0), constrained_layout=True)
    order = sorted(fired, key=lambda i: dis[i]["maxent"]["w1"])
    for j, i in enumerate(order):
        a0.plot([j], [dis[i]["maxent"]["w1"]], "D", color=ME, ms=6, mfc="white")
        for f_ in fs:
            if i in arms[f_]:
                a0.plot([j], [arms[f_][i]["fit"]["w1"]], "o", color=COLS.get(f_, INK), ms=6, mec="white", mew=0.5)
    a0.set_yscale("log")
    a0.set_xticks(range(len(order)), [dis[i]["family"].replace("_", " ") for i in order], fontsize=7, rotation=90)
    a0.set_ylabel(r"W$_1$ to the exact answer")
    for f_ in fs:
        a0.plot([], [], "o", color=COLS.get(f_, INK), label=f"width floor / {f_:g}")
    a0.plot([], [], "D", color=ME, mfc="white", label="MaxEnt")
    a0.legend(fontsize=8.5, frameon=False)
    a0.set_title("error per line spectrum (sorted by MaxEnt's)", fontsize=9.5)
    for j, f_ in enumerate(fs):
        v = res["factors"][f"{f_:g}"]
        m, lo, hi = v["over_maxent"]
        a1.errorbar([j], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=COLS.get(f_, INK), ms=9, capsize=4, lw=1.5)
        a1.annotate(f"lower on {v['lower_than_maxent']}/{v['n']}\naccepted {v['accepted_median']:.0%}\n"
                    f"{v['seconds_median']:.0f} s per spectrum", (j, hi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
        c = v["stated_intervals"]
        a2.errorbar([j - 0.1], [c["c50"]], yerr=[1.96 * c["se50"]], fmt="o", color=COLS.get(f_, INK), ms=8,
                    capsize=3, lw=1.3)
        a2.errorbar([j + 0.1], [c["c90"]], yerr=[1.96 * c["se90"]], fmt="o", color=COLS.get(f_, INK), ms=8,
                    mfc="white", capsize=3, lw=1.3)
    a1.axhline(1.0, color=INK, lw=0.8, ls=":")
    a1.set_yscale("log")
    a1.margins(y=0.35)
    a1.set_ylabel(r"W$_1$ of the report / W$_1$ of MaxEnt (geometric mean, 95% interval)")
    a1.set_title("against MaxEnt on the same data (below 1 = better)", fontsize=9.5)
    for lv in (0.5, 0.9):
        a2.axhline(lv, color=INK, lw=0.8, ls="--")
    a2.set_ylim(0, 1.03)
    a2.set_ylabel("share of windows with the exact weight inside")
    a2.plot([], [], "o", color=INK, label="stated 50% interval")
    a2.plot([], [], "o", color=INK, mfc="white", label="stated 90% interval")
    a2.legend(fontsize=8.5, frameon=False, loc="lower right")
    a2.set_title("exact window weights inside the stated intervals (dashes: nominal)", fontsize=9.5)
    for ax in (a1, a2):
        ax.set_xticks(range(len(fs)), [f"floor / {f_:g}" for f_ in fs])
    for ax in (a0, a1, a2):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle(f"How far to lower the width floor: {Path(res['flow']).stem} on the {len(fired)} exact line spectra "
               f"the report flags, beta = {res['beta']:g}, relative error bar {res['noise']:g}", fontsize=10.5)
    f.savefig(out / "floor_depth.png", dpi=150)

    fmt = lambda t: "{:.3f} [{:.3f}, {:.3f}]".format(*t)
    for k, v in res["factors"].items():
        c = v["stated_intervals"]
        print(f"floor/{k:4s} n {v['n']:2d}  W1 {v['w1']:.4f} (MaxEnt {v['w1_maxent']:.4f})  /MaxEnt {fmt(v['over_maxent'])}"
              f"  /floor4 {fmt(v['over_factor_4'])}  lower on {v['lower_than_maxent']}  accepted median "
              f"{v['accepted_median']:.0%}, <50% on {v['accepted_below_half']}  still at floor on {v['still_at_floor']}"
              f"  {v['seconds_median']:.0f}s  stated {c['c50']:.3f}±{c['se50']:.3f} / {c['c90']:.3f}±{c['se90']:.3f}")
    print(f"-> {out}/floor_depth.{{json,png}}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        torch.set_num_threads(2)
        run(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]),
            int(sys.argv[7]) if len(sys.argv) > 7 else 0, int(sys.argv[8]) if len(sys.argv) > 8 else 1)
    elif sys.argv[1] == "fig":
        fig(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
