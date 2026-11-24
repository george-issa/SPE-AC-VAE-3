r"""Aggregation with honest error bars, and paired comparisons.

Two sources of spread are kept apart:

- **noise draws** -- every synthetic case is scored on several seeds;
- **training runs** -- a model tag may carry several checkpoints trained
  with the same recipe and different seeds (tags ``"M1#0"``, ``"M1#1"`` …
  group as ``"M1"``).

Point estimates are means; intervals are percentile bootstraps that
resample training runs first and noise draws within each run second
(hierarchical), so a single lucky checkpoint cannot narrow the interval.

Comparisons between two models are **paired**: both saw the same noise
draws, so the per-draw log-ratio of their errors removes the draw-to-draw
variance that dominates the marginal intervals.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = ["group_of", "summarize", "compare_models", "compare_pairs"]

N_BOOT = 2000


def group_of(tag: str) -> str:
    return tag.split("#", 1)[0]


def _hier_boot(by_run: dict, rng, stat=np.mean, n_boot=N_BOOT) -> tuple[float, float]:
    runs = [np.asarray(v, float) for v in by_run.values() if len(v)]
    if not runs:
        return float("nan"), float("nan")
    if len(runs) == 1 and len(runs[0]) == 1:
        return float(runs[0][0]), float(runs[0][0])
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(runs), len(runs))
        vals = [runs[j][rng.integers(0, len(runs[j]), len(runs[j]))] for j in pick]
        out[i] = stat(np.concatenate(vals))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def summarize(records: list[dict], metrics=("w1", "chi2", "consistency_w1", "neg_weight"),
              seed: int = 0) -> list[dict]:
    """One row per (model group, method, case, noise) with mean, CI, seed-7 value."""
    rng = np.random.default_rng(seed)
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    legacy = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("kind") != "synthetic":
            continue
        key = (group_of(r["model"]), r["method"], r["case"], r["noise"])
        for m in metrics:
            v = r.get(m)
            if v is None or not np.isfinite(v):
                continue
            if m == "consistency_w1" and r["seed"] % 2 == 0:
                continue                    # one value per pair, not two
            cells[key][m][r["model"]].append(float(v))
            if r["seed"] == 7:
                legacy[key][m].append(float(v))
    rows = []
    for key in sorted(cells):
        g, method, case, noise = key
        row = {"model": g, "method": method, "case": case, "noise": noise}
        for m, by_run in cells[key].items():
            allv = np.concatenate([np.asarray(v) for v in by_run.values()])
            lo, hi = _hier_boot(by_run, rng)
            row[m] = {"mean": float(allv.mean()), "median": float(np.median(allv)),
                      "lo": lo, "hi": hi, "n": int(allv.size), "runs": len(by_run),
                      "seed7": (float(np.mean(legacy[key][m]))
                                if legacy[key].get(m) else None)}
            if len(by_run) > 1:
                row[m]["run_means"] = [float(np.mean(v)) for v in by_run.values()]
        rows.append(row)
    return rows


def compare_models(records: list[dict], a: str, b: str, method: str, metric: str = "w1",
                   seed: int = 0) -> list[dict]:
    """Paired comparison of model groups ``a`` vs ``b`` under one method."""
    return compare_pairs(records, (a, method), (b, method), metric, seed)


def compare_pairs(records: list[dict], a: tuple, b: tuple, metric: str = "w1",
                  seed: int = 0) -> list[dict]:
    r"""Paired comparison of ``a = (group, method)`` vs ``b = (group, method)``.

    Model-free references (MaxEnt, DEAC) have group ``"-"``.

    Per (case, noise): the geometric-mean ratio :math:`b/a` of the metric over
    matched (training run, noise seed) draws, a paired bootstrap 95% interval,
    and the fraction of draws on which ``b`` is better.  A ratio below 1 means
    ``b`` has the smaller error.  Draws are matched by noise seed; when the two
    groups have different numbers of training runs, runs are paired by index
    modulo the smaller count.
    """
    rng = np.random.default_rng(seed)
    vals = defaultdict(lambda: defaultdict(dict))
    for r in records:
        if r.get("kind") != "synthetic" or r.get(metric) is None:
            continue
        key = (group_of(r["model"]), r["method"])
        if key not in (a, b):
            continue
        run = r["model"].split("#", 1)[1] if "#" in r["model"] else "0"
        vals[(r["case"], r["noise"])][key][(run, r["seed"])] = float(r[metric])
    rows = []
    for (case, noise), d in sorted(vals.items()):
        if a not in d or b not in d:
            continue
        ra = sorted({k[0] for k in d[a]})
        rb = sorted({k[0] for k in d[b]})
        pairs = []
        for i in range(max(len(ra), len(rb))):
            for (run, s), va in d[a].items():
                if run != ra[i % len(ra)]:
                    continue
                vb = d[b].get((rb[i % len(rb)], s))
                if vb is not None and va > 0 and vb > 0:
                    pairs.append(np.log(vb / va))
        if not pairs:
            continue
        lr = np.asarray(pairs)
        boot = np.array([lr[rng.integers(0, lr.size, lr.size)].mean() for _ in range(N_BOOT)])
        rows.append({"case": case, "noise": noise, "n": int(lr.size),
                     "ratio": float(np.exp(lr.mean())),
                     "lo": float(np.exp(np.percentile(boot, 2.5))),
                     "hi": float(np.exp(np.percentile(boot, 97.5))),
                     "b_better": float((lr < 0).mean())})
    return rows
