"""The range the data alone allow, recomputed for the spectra of a floor-option judge result.

`posterior.prior_free_bounds` is run with the current code on every spectrum
of REF (a `floor_option_judge.py` result: same spectra, same noise draws) and
compared with the brackets stored there: is each one solvable, does it contain
the exact weight, does it contain the stored bracket, how much wider is it.
Used to judge the change that lets the measure sit beyond the frequency grid
(TRAINING.md (xxviii)).

    python scripts/step3/bracket_check.py run OUT REF.json [K N]     # spectra K mod N
    python scripts/step3/bracket_check.py fig OUT

`run` writes OUT/bracket_<name of REF>[.shardKofN].json (JUDGE_DTAU as in
`exact_models_judge.py`); `fig` pools every result in OUT into
OUT/bracket_check.{json,png}.
"""
import json
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

import exact_models_judge as J
import floor_option_judge as F

TOL = 1e-5            # the linear programs are solved to about 1e-7; brackets are compared to this


def run(out, ref_path, k, n):
    from spevae.evaluation import cases as C
    from spevae.posterior import DEFAULT_WINDOWS, prior_free_bounds
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ref = json.load(open(ref_path))
    beta, noise = ref["beta"], ref["noise"]
    assert abs(ref.get("dtau", 0.125) - J.DTAU) < 1e-12, "set JUDGE_DTAU to the dtau of REF"
    items = J.make_truths(beta) + F.extra_discrete(beta, 2)
    rows = []
    for i, item in enumerate(items):
        if i % n != k:
            continue
        y = ref["rows"][i]
        assert y["family"] == item["family"] and y["params"] == item["params"]
        inst = C.draw(item["truth"], noise, J.SEED)
        t0 = time.time()
        bnd = prior_free_bounds(inst.G[0], inst.truth.x, beta, inst.sigma[0], inst.isr,
                                norm=float(inst.G[0, 0] + inst.G[0, -1]))
        rows.append({"index": i, "family": item["family"], "kind": item["kind"],
                     "truth": [t for t, _ in F.true_windows(item, DEFAULT_WINDOWS)],
                     "new": [[b["lo"], b["hi"]] if b["feasible"] else None for b in bnd],
                     "old": y["auto"].get("bracket"), "seconds": time.time() - t0})
        r = rows[-1]
        print(f"{i:2d} {item['family']:16s} solvable {sum(b is not None for b in r['new'])}/7 (stored: "
              f"{'7/7' if r['old'] else 'none'}) | exact weight inside "
              f"{sum(b is not None and b[0] - TOL <= t <= b[1] + TOL for b, t in zip(r['new'], r['truth']))}/7 "
              f"| {r['seconds']:.0f}s", flush=True)
    tag = f"bracket_{Path(ref_path).stem}" + (f".shard{k}of{n}" if n > 1 else "")
    with open(out / f"{tag}.json", "w") as fh:
        json.dump({"reference": str(ref_path), "beta": beta, "noise": noise, "dtau": J.DTAU, "rows": rows}, fh,
                  indent=1, default=float)
    print(f"-> {out}/{tag}.json")


def fig(out):
    out = Path(out)
    by = {}
    for p in sorted(out.glob("bracket_floor_option_*.json")):
        r = json.load(open(p))
        by.setdefault((r["beta"], r["noise"], r["dtau"]), []).extend(r["rows"])
    res, pts = {"settings": []}, []
    for (beta, noise, dtau), rows in sorted(by.items()):
        s = {"beta": beta, "noise": noise, "dtau": dtau, "spectra": len(rows), "windows": 7 * len(rows)}
        for name, key in (("stored", "old"), ("recomputed", "new")):
            solv = ins = 0
            for r in rows:
                for j, t in enumerate(r["truth"]):
                    b = r[key][j] if r[key] else None
                    if b is None or b[0] != b[0] or b[1] != b[1]:      # no solution for this window
                        continue
                    solv += 1
                    ins += b[0] - TOL <= t <= b[1] + TOL
            s[name] = {"solvable": int(solv), "exact_inside": int(ins)}
        both = [(o, w) for r in rows if r["old"] for o, w in zip(r["old"], r["new"])
                if w is not None and o[0] == o[0] and o[1] == o[1]]
        if both:
            o, w = np.array([a for a, _ in both]), np.array([b for _, b in both])
            s["windows_compared"] = len(both)
            s["contains_stored"] = bool(((w[:, 0] <= o[:, 0] + TOL) & (w[:, 1] >= o[:, 1] - TOL)).all())
            s["mean_width"] = {"stored": float((o[:, 1] - o[:, 0]).mean()), "recomputed": float((w[:, 1] - w[:, 0]).mean())}
            s["largest_widening_of_one_window"] = float(((w[:, 1] - w[:, 0]) - (o[:, 1] - o[:, 0])).max())
            pts.append((f"beta {beta:g}, err {noise:g}, dtau {dtau:g}", o[:, 1] - o[:, 0], w[:, 1] - w[:, 0]))
        res["settings"].append(s)
    with open(out / "bracket_check.json", "w") as fh:
        json.dump(res, fh, indent=1)

    f, (a0, a1) = plt.subplots(1, 2, figsize=(14.5, 5.0), constrained_layout=True)
    names = [f"beta {s['beta']:g}\nerr {s['noise']:g}\ndtau {s['dtau']:g}" for s in res["settings"]]
    x = np.arange(len(names))
    for dx, key, col in ((-0.2, "stored", "#7d7c78"), (0.2, "recomputed", "#0b7a75")):
        a0.bar(x + dx, [s[key]["exact_inside"] / s["windows"] for s in res["settings"]], width=0.38, color=col,
               label=f"{key}: windows with a bracket that holds the exact weight")
    a0.set_xticks(x, names, fontsize=8)
    a0.set_ylim(0, 1.12)
    a0.set_ylabel("share of windows")
    a0.legend(fontsize=8.5, frameon=False, loc="upper left")
    a0.set_title("solvable and containing the exact weight", fontsize=9.5)
    for name, o, w in pts:
        a1.plot(o, w, "o", ms=3, alpha=0.6, label=name)
    lim = max([max(o.max(), w.max()) for _, o, w in pts] or [1.0])
    a1.plot([0, lim], [0, lim], color="#0b0b0b", lw=0.8, ls=":")
    a1.set_xlabel("width of the stored bracket (measure on |omega| <= 20 only)")
    a1.set_ylabel("width of the recomputed bracket")
    a1.legend(fontsize=7.5, frameon=False)
    a1.set_title("one point per window, where both exist", fontsize=9.5)
    for ax in (a0, a1):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle("The range the data alone allow, with the measure free to sit beyond the frequency grid, against "
               "the stored ranges", fontsize=10.5)
    f.savefig(out / "bracket_check.png", dpi=150)
    for s in res["settings"]:
        print(f"beta {s['beta']:g} noise {s['noise']:g} dtau {s['dtau']:g}: {s['windows']} windows | stored: solvable "
              f"{s['stored']['solvable']}, exact inside {s['stored']['exact_inside']} | recomputed: solvable "
              f"{s['recomputed']['solvable']}, exact inside {s['recomputed']['exact_inside']}"
              + (f" | contains stored {s['contains_stored']}, mean width {s['mean_width']['stored']:.4f} -> "
                 f"{s['mean_width']['recomputed']:.4f}, largest widening {s['largest_widening_of_one_window']:.4f}"
                 if "mean_width" in s else ""))
    print(f"-> {out}/bracket_check.{{json,png}}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0,
            int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    elif sys.argv[1] == "fig":
        fig(sys.argv[2])
    else:
        raise SystemExit(__doc__)
