#!/usr/bin/env python
"""Overlay the held-out eval trajectories of pole-count comparison runs.

Parses the `[eval @ N] k=v ...` lines the `spevae pretrain` CLI emits.  All
runs share the frozen eval stream, so the curves are directly comparable
point by point -- the whole reason the figure is meaningful.

    python scripts/plot_pole_count.py out/pretrain-fast.log:P=16 \
        out/pretrain-fast-p24.log:P=24 out/pretrain-fast-p8.log:P=8 \
        --out out/pole_count_training.png
"""

from __future__ import annotations

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_EVAL = re.compile(r"\[eval @ (\d+)\] (.*)")
_KV = re.compile(r"([A-Za-z0-9_/]+)=([-+.\deE]+)")

COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def parse(path: str) -> dict[str, list]:
    out: dict[str, list] = {"step": []}
    with open(path, errors="replace") as f:
        for line in f:
            m = _EVAL.search(line)
            if not m:
                continue
            out["step"].append(int(m.group(1)))
            for k, v in _KV.findall(m.group(2)):
                out.setdefault(k, []).append(float(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", metavar="LOG:LABEL")
    ap.add_argument("--out", default="pole_count_training.png")
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        path, _, label = spec.partition(":")
        runs.append((label or path, parse(path)))

    panels = [("w1_omega", r"$W_1$ [physical $\omega$]"),
              ("l2_rel", r"relative $L^2$ on $A$"),
              ("chi2_geo", r"$\chi^2$ (geometric mean)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8))
    for ax, (key, ylabel) in zip(axes, panels):
        for i, (label, d) in enumerate(runs):
            if key not in d:
                continue
            n = min(len(d["step"]), len(d[key]))
            ax.plot(d["step"][:n], d[key][:n], "-o", ms=3.5, lw=1.3,
                    color=COLOURS[i % len(COLOURS)], label=label)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if key == "chi2_geo":
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Pole count at fixed seed, preset, and frozen eval stream "
                 "(curves comparable point by point)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
