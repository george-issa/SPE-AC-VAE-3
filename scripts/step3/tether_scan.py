"""How firmly should a sample be held at the flow's proposal while it is fitted?

Collects the known-answer judge and the calibration judge at several tether
strengths (`lam` of the refinement) into one figure and one JSON.

    python scripts/step3/tether_scan.py out/step3/tether_scan \
        "new set, 6k steps" out/eval/exact_models_F3_6k out/step3/cal_F3_6k \
        ["old set, 6k steps" out/eval/exact_models_F6k -]

Each model is a (label, known-answer directory, calibration directory or "-")
triple; calibration directories are `<dir>` for tether 1 and `<dir>_lam<t>`
otherwise, known-answer files `exact_models_b8_n0.001[_lam<t>].json`.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TETHERS = (1, 10, 100)
SHADE = {1: "#9ec5f2", 10: "#2a78d6", 100: "#0f3d75"}
INK, ME, VAE = "#0b0b0b", "#7d7c78", "#eb6834"


def main():
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    models = [sys.argv[i:i + 3] for i in range(2, len(sys.argv), 3)]
    res = {"tethers": TETHERS, "models": {}}
    for label, kdir, cdir in models:
        m = {"known": {}, "calibration": {}}
        for t in TETHERS:
            p = Path(kdir) / ("exact_models_b8_n0.001" + ("" if t == 1 else f"_lam{t}") + ".json")
            if p.exists():
                s = json.load(open(p))["summary"]
                m["known"][t] = {k: {"w1": s[k]["w1_geomean"], "accepted": s[k]["flow_frac_accepted_min_median"][1],
                                     "chi2": s[k]["chi2_median"]["flow"]} for k in ("continuum", "discrete")}
            p = Path(cdir + ("" if t == 1 else f"_lam{t}")) / "calibration.json"
            if cdir != "-" and p.exists():
                d = json.load(open(p))
                q = d["pooled"]["posterior"]
                m["calibration"][t] = {
                    "pooled": {k: q[k] for k in ("cov50", "cov90", "ks_max", "fit_rank_mean", "weight_error_median")},
                    "levels": [{"noise": l["noise"], "beta": l["beta"], "accepted": l["accepted"],
                                **{k: l["posterior"][k] for k in ("cov50", "cov90", "fit_rank", "weight_error")}}
                               for l in d["levels"]]}
        res["models"][label] = m
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(res, fh, indent=1)

    f, axes = plt.subplots(1, 4, figsize=(18, 4.0), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(TETHERS))
    styles = [("-", "o"), ("--", "s"), (":", "^")]
    ref = None
    for (label, m), (ls, mk) in zip(res["models"].items(), styles):
        for kind, col, name in (("continuum", "#2a78d6", "lattice"), ("discrete", "#0f8a8a", "exactly diagonalised")):
            ts = [t for t in TETHERS if t in m["known"]]
            ax.plot([TETHERS.index(t) for t in ts], [m["known"][t][kind]["w1"]["flow"] for t in ts],
                    color=col, ls=ls, marker=mk, lw=1.5, ms=5, label=f"flow, {name}: {label}")
        ref = ref or m["known"].get(1)
    if ref:
        for kind, y0 in (("continuum", None), ("discrete", None)):
            ax.axhline(ref[kind]["w1"]["maxent"], color=ME, lw=1.0, ls="-" if kind == "continuum" else "--")
            ax.text(0.5, ref[kind]["w1"]["maxent"], f"MaxEnt, {'lattice' if kind == 'continuum' else 'exactly diagonalised'}",
                    color=ME, fontsize=7.5, va="bottom", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in TETHERS])
    ax.set_xlabel("tether strength")
    ax.set_ylim(0, None)
    ax.set_title("known answers: W$_1$ to the exact spectrum\n(geometric mean; beta 8, noise 1e-3)", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    first = next((m for m in res["models"].values() if m["calibration"]), None)
    for ax, key, title, nominal in zip(
            axes[1:], ("weight_error", "fit_rank", "cov50"),
            ("held-out prior: window-weight error of the posterior mean", "held-out prior: data-fit rank (0.5 = calibrated)",
             "held-out prior: 50% interval coverage"), (None, 0.5, 0.5)):
        if first is None:
            ax.axis("off")
            continue
        for t in TETHERS:
            if t not in first["calibration"]:
                continue
            lv = first["calibration"][t]["levels"]
            ax.plot([l["noise"] for l in lv], [l[key] for l in lv], color=SHADE[t], marker="o", ms=4, lw=1.5,
                    label=f"tether {t}")
        ax.set_xscale("log")
        if key == "weight_error":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1)
            ax.axhline(nominal, color="#52514e", ls="--", lw=0.9)
        if key == "fit_rank":
            ax.axhspan(0.45, 0.55, color="#e4e3df", lw=0)
        ax.set_xlabel("relative noise level (beta cycles 1 ... 20)")
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=7.5, frameon=False)
    for ax in axes:
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle("Tether strength of the refinement: 10 lowers the error and keeps the misfit calibrated; "
               "100 stops the samples short of the data; none of them changes the coverage", fontsize=10.5)
    f.savefig(out.with_suffix(".png"), dpi=150)
    print(f"-> {out.with_suffix('.json')} + .png")


if __name__ == "__main__":
    main()
