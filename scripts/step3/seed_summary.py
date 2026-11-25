"""Seed-to-seed spread of the full-length step-3 runs.

Collects, for each named run, the calibration judge (288 held-out truths,
plain sample quantiles) and the known-answer judge at its five settings, and
plots the seeds side by side.

    python scripts/step3/seed_summary.py out/step3/seeds F3_60k_s1 F3_60k_s2 F3_60k_s3
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SETTINGS = (("8", "0.0001"), ("8", "0.001"), ("8", "0.01"), ("2", "0.001"), ("16", "0.001"))
FLOW, VAE, ME = "#2a78d6", "#eb6834", "#7d7c78"


def main():
    out, names = Path(sys.argv[1]), sys.argv[2:]
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"runs": {}, "settings": [f"beta {b}, noise {n}" for b, n in SETTINGS]}
    for name in names:
        r = {"known": {}}
        p = Path(f"out/step3/cal_{name}/calibration.json")
        if p.exists():
            q = json.load(open(p))["pooled"]
            r["calibration"] = {"raw": {k: q["raw"][k] for k in ("cov50", "cov90", "weight_error_median")},
                                "posterior": {k: q["posterior"][k] for k in
                                              ("cov50", "cov90", "fit_rank_mean", "weight_error_median")}}
        for b, n in SETTINGS:
            p = Path(f"out/eval/exact_models_{name}/exact_models_b{b}_n{n}_lam10.json")
            if p.exists():
                s = json.load(open(p))["summary"]
                r["known"][f"beta {b}, noise {n}"] = {
                    kind: {"w1": s[kind]["w1_geomean"], "lowest": s[kind]["wins"]["flow"], "n": s[kind]["n"]}
                    for kind in ("continuum", "discrete")}
        res["runs"][name] = r
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for key in res["settings"]:
        for kind in ("continuum", "discrete"):
            v = [r["known"][key][kind]["w1"]["flow"] for r in res["runs"].values() if key in r["known"]]
            if v:
                ref = next(r["known"][key][kind]["w1"] for r in res["runs"].values() if key in r["known"])
                print(f"{key:22s} {kind:9s}: flow " + " ".join(f"{x:.3f}" for x in v)
                      + f"   mean {np.mean(v):.3f}, spread {(max(v) - min(v)) / np.mean(v):.0%}"
                      + f"   | VAE {ref['vae']:.3f}  MaxEnt {ref['maxent']:.3f}")
    for name, r in res["runs"].items():
        if "calibration" in r:
            c = r["calibration"]["posterior"]
            print(f"{name}: plain coverage {c['cov50']:.3f} / {c['cov90']:.3f}, fit rank {c['fit_rank_mean']:.2f}, "
                  f"error {c['weight_error_median']:.4f} (raw {r['calibration']['raw']['weight_error_median']:.3f})")

    f, axes = plt.subplots(1, 3, figsize=(17, 4.0), constrained_layout=True)
    x = np.arange(len(SETTINGS))
    for ax, kind, title in ((axes[0], "continuum", "exact lattice spectra"),
                            (axes[1], "discrete", "exactly diagonalised (line) spectra")):
        ref = None
        for j, (name, r) in enumerate(res["runs"].items()):
            ys = [r["known"].get(k, {}).get(kind, {}).get("w1", {}).get("flow", np.nan) for k in res["settings"]]
            ax.plot(x + (j - (len(names) - 1) / 2) * 0.12, ys, "o", color=FLOW, ms=6,
                    label="flow, one point per seed" if j == 0 else None)
            if ref is None and len(r["known"]) == len(SETTINGS):
                ref = [r["known"][k][kind]["w1"] for k in res["settings"]]
        if ref:
            ax.plot(x, [v["vae"] for v in ref], "s", color=VAE, ms=5, label="VAE latent MAP")
            ax.plot(x, [v["maxent"] for v in ref], "D", color=ME, ms=5, label="MaxEnt")
        ax.set_xticks(x)
        ax.set_xticklabels([f"beta {b}\nnoise {float(n):g}" for b, n in SETTINGS], fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"W$_1$ to the exact answer (geometric mean): {title}", fontsize=9.5)
        ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    lab = [("cov50", "50% coverage", 0.5), ("cov90", "90% coverage", 0.9), ("fit_rank_mean", "data-fit rank", 0.5)]
    for j, (name, r) in enumerate(res["runs"].items()):
        if "calibration" not in r:
            continue
        ax.plot(np.arange(3) + (j - (len(names) - 1) / 2) * 0.12, [r["calibration"]["posterior"][k] for k, _, _ in lab],
                "o", color=FLOW, ms=6)
    for i, (_, _, nom) in enumerate(lab):
        ax.plot([i - 0.3, i + 0.3], [nom, nom], color="#52514e", ls="--", lw=0.9)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([l for _, l, _ in lab])
    ax.set_ylim(0, 1.02)
    ax.set_title("held-out prior, plain sample quantiles (dashes: nominal)", fontsize=9.5)
    for ax in axes:
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle(f"Seed-to-seed spread of the full-length step-3 model ({len(names)} seeds: {', '.join(names)})",
               fontsize=10.5)
    f.savefig(out.with_suffix(".png"), dpi=150)
    print(f"-> {out.with_suffix('.json')} + .png")


if __name__ == "__main__":
    main()
