#!/usr/bin/env python
"""Judge the p_pole micro sweep and draw the verdict figures.

Every checkpoint is scored on two frozen micro streams (the same narrowing
the sweep trained at: beta = 8, dtau = 0.125): the legacy family stream,
where the interference cost shows, and the pole-labeled stream, where the
pole-placement advantage shows.  Seeds are plotted as individual points --
the spread IS the error bar; nothing is hidden in a mean.

    python scripts/microsweep_report.py out/microsweep --figure out/microsweep/sweep.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spevae import load_model  # noqa: E402
from spevae.data.pole_prior import PolePriorGenerator  # noqa: E402
from spevae.data.synthetic import SyntheticGenerator  # noqa: E402
from spevae.training import evaluate  # noqa: E402

ARM_ORDER = ["p00", "p25", "p50", "p75", "cur"]
ARM_LABEL = {"p00": "0.00", "p25": "0.25", "p50": "0.50", "p75": "0.75",
             "cur": "0.9→0.1"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--batches", type=int, default=16)
    args = ap.parse_args()

    legacy = SyntheticGenerator(beta_range=(8.0, 8.0), dtau_choices=(0.125,))
    poleg = PolePriorGenerator(beta_range=(8.0, 8.0), dtau_choices=(0.125,))

    rows: dict[str, dict[int, dict]] = {}
    for path in sorted(glob.glob(os.path.join(args.sweep_dir, "ms_*.pt"))):
        m = re.match(r"ms_(\w+)_s(\d+)\.pt$", os.path.basename(path))
        if not m or m.group(1) not in ARM_ORDER:
            continue
        arm, seed = m.group(1), int(m.group(2))
        model = load_model(path)
        ml = evaluate(model, legacy, n_batches=args.batches)
        mp = evaluate(model, poleg, n_batches=args.batches)
        rows.setdefault(arm, {})[seed] = {
            "legacy_w1": ml["w1_omega"], "legacy_l2": ml["l2_rel"],
            "legacy_rip": ml["ripple"], "pole_w1": mp["w1_omega"],
            "pole_l2": mp["l2_rel"], "pole_rip": mp["ripple"],
        }

    hdr = (f"{'arm':<9}{'seeds':>5}{'legacy W1':>15}{'legacy l2':>15}"
           f"{'pole W1':>15}{'pole rip':>15}")
    print(hdr)
    print("-" * len(hdr))
    for arm in ARM_ORDER:
        if arm not in rows:
            continue
        vs = rows[arm].values()

        def rng(key):
            xs = [v[key] for v in vs]
            mid = sum(xs) / len(xs)
            return f"{mid:.3f} ±{(max(xs) - min(xs)) / 2:.3f}"

        print(f"{ARM_LABEL[arm]:<9}{len(rows[arm]):>5}{rng('legacy_w1'):>15}"
              f"{rng('legacy_l2'):>15}{rng('pole_w1'):>15}{rng('pole_rip'):>15}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True)
        for ax, key, title in [
                (axes[0], "legacy_w1", "family coverage (legacy stream W$_1$)"),
                (axes[1], "pole_w1", "pole placement (pole stream W$_1$)")]:
            for i, arm in enumerate(ARM_ORDER):
                if arm not in rows:
                    continue
                xs = [v[key] for v in rows[arm].values()]
                ax.scatter([i] * len(xs), xs, s=28, alpha=0.75, zorder=3,
                           color="#B8451F" if arm == "cur" else "#0E7C86")
                ax.plot([i - 0.22, i + 0.22],
                        [sum(xs) / len(xs)] * 2, color="#1D242C", lw=1.6, zorder=4)
            ax.set_title(title, fontsize=10)
            ax.set_xticks(range(len(ARM_ORDER)))
            ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], fontsize=9)
            ax.set_xlabel("p_pole")
            ax.grid(axis="y", alpha=0.25, lw=0.5)
        axes[0].set_ylabel("W$_1$ (physical $\\omega$)")
        fig.suptitle("p_pole micro sweep · β = 8 fixed · 1500 steps "
                     "· 3 seeds (bars = seed means)", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"\nfigure -> {args.figure}")


if __name__ == "__main__":
    main()
