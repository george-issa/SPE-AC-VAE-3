"""Overlay several runs of the step-3 calibration judge.

    python scripts/step3/compare_calibrations.py out/step3/temperature_scan "title" \
        "T = 1" out/step3/cal_F3_6k_lam10  "T = 0.9" out/step3/cal_F3_6k_lam10_T0.9 ...

Writes `<out>.json` (the pooled numbers side by side) and `<out>.png`
(coverage of the 50% and 90% intervals, data-fit rank and window-weight error
of the reported posterior against the noise level).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#0f8a8a", "#8a4fd6", "#7d7c78", "#c9a227"]


def main():
    out, title = Path(sys.argv[1]), sys.argv[2]
    runs = [(sys.argv[i], json.load(open(Path(sys.argv[i + 1]) / "calibration.json")))
            for i in range(3, len(sys.argv), 2)]
    keys = ("cov50", "cov50_se", "cov90", "cov90_se", "ks_max", "ks_crit_5pct_bonferroni7", "fit_rank_mean",
            "weight_error_median", "truths")
    res = {lab: {arm: {k: d["pooled"][arm][k] for k in keys} for arm in ("raw", "posterior")} for lab, d in runs}
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    f, axes = plt.subplots(1, 4, figsize=(18, 4.0), constrained_layout=True)
    for ax, key, name, nominal in zip(axes, ("cov50", "cov90", "fit_rank", "weight_error"),
                                      ("50% interval coverage", "90% interval coverage",
                                       "data-fit rank (0.5 = calibrated)",
                                       "window-weight error of the posterior mean (median)"),
                                      (0.5, 0.9, 0.5, None)):
        for (lab, d), col in zip(runs, COLORS):
            lv = d["levels"]
            q = d["pooled"]["posterior"]
            tag = {"cov50": f"{q['cov50']:.2f}", "cov90": f"{q['cov90']:.2f}", "fit_rank": f"{q['fit_rank_mean']:.2f}",
                   "weight_error": f"{q['weight_error_median']:.3f}"}[key]
            ax.plot([l["noise"] for l in lv], [l["posterior"][key] for l in lv], color=col, marker="o", ms=4,
                    lw=1.5, label=f"{lab}  (pooled {tag})")
        ax.set_xscale("log")
        if nominal is None:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.02)
            ax.axhline(nominal, color="#52514e", ls="--", lw=0.9)
        if key == "fit_rank":
            ax.axhspan(0.45, 0.55, color="#e4e3df", lw=0)
        ax.set_xlabel("relative noise level (beta cycles 1 ... 20)")
        ax.set_title(name, fontsize=9.5)
        ax.legend(fontsize=7.5, frameon=False, loc="lower left" if nominal is not None else "best")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    n = runs[0][1]["pooled"]["posterior"]["truths"]
    f.suptitle(f"{title}  (reported posterior, {n} held-out truths of the step-3 prior per run)", fontsize=10.5)
    f.savefig(out.with_suffix(".png"), dpi=150)
    print(f"-> {out.with_suffix('.json')} + .png")


if __name__ == "__main__":
    main()
