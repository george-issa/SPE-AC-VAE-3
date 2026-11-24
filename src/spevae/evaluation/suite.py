r"""Tiers, tasks and the parallel runner.

A *task* is one (model, method, case, noise level, seed pair).  Tasks are
independent and carry explicit seeds, and every worker pins torch to the
same thread count, so results do not depend on scheduling or worker count.
Seeds come in pairs so the paired-noise consistency (how much a method's
answer moves between two noise draws of the same truth) is measured on
every pair.

Two tiers:

- ``fast`` (minutes): streams, the amortized pass on every case, z-descent
  800 on the three step-2 screen probes, MaxEnt on every case, and the
  Holstein cell (amortized, MAP, scan pipeline; no floor).
- ``full``: adds z-descent 800 and 3000 on every case, the scan
  certified pipeline on every case (first seed pair only: ~1 min per
  run), the Holstein manifold floor, and DEAC on the Holstein cell.
- ``lm``: the least-squares latent descent (``zdlm``) on every case and the
  Holstein cell, plus the floor record -- to set next to a ``full`` card.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import torch

from . import cases as C
from . import methods as M

__all__ = ["TIERS", "Tier", "run_suite", "SUITE_VERSION"]

SUITE_VERSION = "1.0"
NOISES = (1e-4, 1e-3)
THREADS = 2
SCREEN = ("asym triplet", "fano band", "semicircle")


@dataclass(frozen=True)
class Tier:
    name: str
    synthetic: tuple                # ((method, cases | "all", n_seeds), ...)
    real: tuple                     # methods on the Holstein cell
    streams: bool = True
    floor: bool = False
    noises: tuple = NOISES


J8 = ("3-rung ladder", "4-rung ladder", "edged band", "irregular triplet", "asym triplet",
      "fano band", "lorentz pair", "gauss pair", "gapped pair", "semicircle", "box",
      "pseudogap")

TIERS = {
    "fast": Tier("fast",
                 synthetic=(("amortized", "all", 8), ("zd800", SCREEN, 4),
                            ("maxent", "all", 8)),
                 real=("amortized", "zd3000", "scan_minimal", "maxent", "deac_shipped")),
    "full": Tier("full",
                 synthetic=(("amortized", "all", 8), ("zd800", "all", 8),
                            ("zd3000", "all", 8), ("scan_minimal", "all", 2),
                            ("maxent", "all", 8)),
                 real=("amortized", "zd3000", "scan_minimal", "maxent", "deac", "deac_shipped"),
                 floor=True),
    # 6k screens: fast tier + z-descent 3000 everywhere + the Holstein floor
    "screen": Tier("screen",
                   synthetic=(("amortized", "all", 8), ("zd800", SCREEN, 4),
                              ("zd3000", "all", 2), ("maxent", "all", 8)),
                   real=("amortized", "zd3000", "scan_minimal", "maxent", "deac_shipped"),
                   floor=True),
    # the least-squares latent descent next to nothing else: pair its rows with a
    # banked `full` scorecard of the same models (same cases, same eight draws)
    "lm": Tier("lm", synthetic=(("zdlm", "all", 8),), real=("zdlm",), streams=False,
               floor=True),
    # the banked judges' exact protocols on the banked draws (acceptance.py)
    "accept": Tier("accept",
                   synthetic=(("amortized", J8, 1), ("zd800", SCREEN, 2), ("zd3000", J8, 1),
                              ("scan_minimal", ("fano band", "asym triplet"), 1)),
                   real=("amortized", "zd3000", "scan_minimal"), floor=True),
}


# --------------------------------------------------------------- worker --- #
_MODELS: dict = {}
_TRUTHS: dict = {}


def _init_worker(threads: int):
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def _model(path):
    if path is None:
        return None
    if path not in _MODELS:
        from .. import load_model
        _MODELS[path] = load_model(path)
    return _MODELS[path]


def _truth(name):
    if name not in _TRUTHS:
        _TRUTHS[name] = C.realize(name)
    return _TRUTHS[name]


def _run_synthetic(task):
    tag, path, method, case, noise, pair = task
    meth = M.make_method(method, _model(path))
    recs, As = [], []
    for seed in pair:
        inst = C.draw(_truth(case), noise, seed)
        out = meth.synthetic(inst)
        for name, o in [(method, out)] + list(out.alternates.items()):
            rec, A = M.score_synthetic(o, inst)
            rec.update(kind="synthetic", model=tag, method=name, case=case,
                       noise=noise, seed=seed)
            if name != method:
                rec["runtime"] = out.runtime
            if seed == C.LEGACY_SEED:
                rec["A_plot"] = M.spectrum_on_grid(o, inst.truth.beta).tolist()
            recs.append(rec)
            As.append((name, A))
    from ..losses import spectral_w1
    tr = _truth(case)
    for name in {n for n, _ in As}:
        mine = [A for n, A in As if n == name]
        if len(mine) == 2:
            cons = float(spectral_w1(mine[0], mine[1], tr.quad, beta=tr.beta))
            for r in recs:
                if r["method"] == name:
                    r["consistency_w1"] = cons
    return recs


def _run_real(task):
    tag, path, method = task
    mg = C.holstein_cell()
    out = M.make_method(method, _model(path)).real(mg)
    quad = M.SpectralQuadrature.for_beta(mg.beta)
    recs = []
    for name, o in [(method, out)] + list(out.alternates.items()):
        rec = M.score_real(o, mg, quad)
        rec.update(kind="real", case="holstein", model=tag, method=name,
                   A_plot=M.spectrum_on_grid(o, mg.beta).tolist())
        if name != method:
            rec["runtime"] = out.runtime
        recs.append(rec)
    return recs


def _run_floor(task):
    tag, path = task
    from ..refine import RefineConfig, refine_latent
    mg = C.holstein_cell()
    model = _model(path)
    t0 = time.perf_counter()
    rr = refine_latent(model, mg, RefineConfig(cov_method="structured",
                                               min_eigenvalue=1e-10, steps=3000))
    rb = refine_latent(model, mg, RefineConfig(cov_method="structured", min_eigenvalue=1e-10,
                                               steps=10000, prior_weight=0.0, patience=3000))
    # The two descents above are the banked protocol (Adam, fixed step caps) and are
    # kept so earlier scorecards stay reproducible -- but on 2026-11-24 every
    # reference checkpoint was found still descending at the 10,000-step cap, so
    # "floor" is an optimiser-limited number.  The least-squares solver gives the
    # converged ones, with the reason each fit ended.
    lm = {}
    for key, pw in (("floor_lm", 0.0), ("map_lm", 1.0)):
        r = refine_latent(model, mg, RefineConfig(cov_method="structured", min_eigenvalue=1e-10,
                                                  solver="lm", prior_weight=pw, lm_iters=300))
        lm[key] = float(r.chi2)
        lm[key + "_ratio"] = float(r.chi2 / r.chi2_target)
        lm[key + "_iters"] = int(r.steps_run)
        lm[key + "_status"] = ("at target" if r.converged else
                               "stationary" if r.stationary else "budget")
    return [{"kind": "floor", "case": "holstein", "model": tag,
             "chi2_init": float(rr.chi2_init), "chi2_map": float(rr.chi2),
             "floor": float(rb.chi2), "target": float(rb.chi2_target),
             "ratio": float(rb.chi2 / rb.chi2_target),
             "map_converged": bool(rr.converged), "map_steps_run": int(rr.steps_run),
             "floor_converged": bool(rb.converged), "floor_steps_run": int(rb.steps_run),
             **lm, "runtime": time.perf_counter() - t0}]


def _run_streams(task):
    tag, path = task
    from ..data.pole_prior import PolePriorGenerator
    from ..data.synthetic import SyntheticGenerator
    from ..training import evaluate
    model = _model(path)
    out = {"kind": "streams", "model": tag}
    for name, gen in (("pole", PolePriorGenerator(beta_range=(8.0, 8.0))),
                      ("canary", SyntheticGenerator(beta_range=(8.0, 8.0)))):
        r = evaluate(model, gen, n_batches=16)
        out[name] = {k: float(v) for k, v in r.items() if isinstance(v, (float, int))}
    return [out]


_RUNNERS = {"synthetic": _run_synthetic, "real": _run_real,
            "floor": _run_floor, "streams": _run_streams}


def _dispatch(item):
    kind, task = item
    return _RUNNERS[kind](task)


# --------------------------------------------------------------- driver --- #
def _needs_model(method: str) -> bool:
    return method in ("amortized", "scan_minimal") or method.startswith("zd")


def _tasks(tier: Tier, models: list[tuple[str, str]], cases_sel=None):
    """Expand a tier into tasks.  ``models`` is [(tag, path), ...]."""
    items = []
    for method, sel, n_seeds in tier.synthetic:
        names = list(C.CASES) if sel == "all" else list(sel)
        if cases_sel:
            names = [n for n in names if n in cases_sel]
        seeds = C.seeds_for(n_seeds)
        pairs = [tuple(seeds[i:i + 2]) for i in range(0, len(seeds), 2)]
        targets = models if _needs_model(method) else [("-", None)]
        for tag, path in targets:
            for case in names:
                for noise in tier.noises:
                    for pair in pairs:
                        items.append(("synthetic", (tag, path, method, case, noise, pair)))
    for method in tier.real:
        targets = models if _needs_model(method) else [("-", None)]
        for tag, path in targets:
            items.append(("real", (tag, path, method)))
    for tag, path in models:
        if tier.streams:
            items.append(("streams", (tag, path)))
        if tier.floor:
            items.append(("floor", (tag, path)))
    # slow items first so the pool never idles at the end
    order = {"floor": 0, "real": 1, "synthetic": 2, "streams": 3}
    items.sort(key=lambda it: order[it[0]])
    return items


def _git_state(root="."):
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:                                   # not a checkout
        return {"commit": None, "dirty": None}


def _file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_suite(models: list[tuple[str, str]], tier: str = "fast", workers: int | None = None,
              cases_sel=None, skip=(), progress=print) -> dict:
    """Run a tier on the given models and return the raw scorecard dict."""
    t = TIERS[tier]
    # repeated tags are training-seed replicates of one recipe: M1, M1 -> M1#0, M1#1
    counts = {}
    for tag, _ in models:
        counts[tag] = counts.get(tag, 0) + 1
    seen = {}
    runs = []
    for tag, path in models:
        k = seen.get(tag, 0)
        seen[tag] = k + 1
        runs.append((f"{tag}#{k}" if counts[tag] > 1 else tag, path))
    models = runs
    if skip:
        t = Tier(t.name, tuple(s for s in t.synthetic if s[0] not in skip),
                 () if "real" in skip else tuple(m for m in t.real if m not in skip),
                 t.streams and "streams" not in skip,
                 t.floor and "floor" not in skip, t.noises)
    items = _tasks(t, models, cases_sel)
    workers = workers or max(1, (os.cpu_count() or 2) // THREADS)
    t0 = time.time()
    records, failures = [], []
    progress(f"spevae eval {tier}: {len(items)} tasks on {workers} workers x {THREADS} threads")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(THREADS,)) as ex:
        futs = {ex.submit(_dispatch, it): it for it in items}
        for k, fut in enumerate(as_completed(futs), 1):
            it = futs[fut]
            try:
                records.extend(fut.result())
            except Exception as e:                      # keep going; report it
                failures.append({"task": repr(it), "error": f"{type(e).__name__}: {e}"})
            if k % max(1, len(items) // 10) == 0 or k == len(items):
                progress(f"  {k}/{len(items)} tasks  ({time.time() - t0:.0f}s)")
    model_info = {}
    from .. import load_model
    for tag, path in models:
        m = load_model(path)
        model_info.setdefault(tag, []).append(
            {"path": os.path.abspath(path), "sha256": _file_sha256(path),
             "params": int(m.num_parameters()),
             "config": {k: v for k, v in dict(getattr(m, "config", {})).items()
                        if isinstance(v, (int, float, str, bool, type(None)))}})
    return {
        "suite": {"version": SUITE_VERSION, "tier": tier, "threads": THREADS,
                  "workers": workers, "noises": list(t.noises),
                  "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "wall_seconds": time.time() - t0, "torch": torch.__version__,
                  "python": platform.python_version(), "machine": platform.machine(),
                  **_git_state()},
        "models": model_info,
        "cases": {n: {"tags": list(c.tags), "n_true": c.n_true} for n, c in C.CASES.items()},
        "plot_omega": M.PLOT_OMEGA.tolist(),
        "records": records,
        "failures": failures,
    }
