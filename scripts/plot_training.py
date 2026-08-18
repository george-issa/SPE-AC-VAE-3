#!/usr/bin/env python
"""Plot training curves from run logs.

    python scripts/plot_training.py out/f_*.log --out out/curves.png
    python scripts/plot_training.py out/gap*.log --floors gapA=92.9,gapB=4.5
"""
from __future__ import annotations

import argparse
import glob
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

from spevae.plotting import parse_training_log, plot_training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--labels", default=None, help="comma-separated")
    ap.add_argument("--floors", default=None,
                    help="label=value,... ansatz chi2 floors")
    ap.add_argument("--title", default=None)
    ap.add_argument("--smooth", type=int, default=15)
    ap.add_argument("--out", default="out/curves.png")
    args = ap.parse_args()

    paths = [p for g in args.logs for p in sorted(glob.glob(g))]
    if not paths:
        raise SystemExit("no logs matched")
    labels = args.labels.split(",") if args.labels else None
    floors = None
    if args.floors:
        floors = {k: float(v) for k, v in
                  (kv.split("=") for kv in args.floors.split(","))}

    plot_training(paths, labels=labels, floors=floors, smooth=args.smooth,
                  title=args.title, path=args.out)

    for p, lab in zip(paths, labels or [None] * len(paths)):
        r = parse_training_log(p, lab)
        tail = (f"{r.wall_time:.0f}s" if r.finished
                else f"running, step {int(r.step[-1]) if r.step.size else 0}")
        best = f"chi2 {r.evals['chi2'][-1]:.4g}" if "chi2" in r.evals else "-"
        print(f"{r.label:14s} {tail:>22s}   last eval {best}")
    print(f"\nfigure -> {args.out}")


if __name__ == "__main__":
    main()
