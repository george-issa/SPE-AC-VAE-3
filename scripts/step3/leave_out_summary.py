"""Does the flow need to have seen a family to continue it?  The leave-out arm, scored.

Compares a model trained with some exactly solvable families withheld to the
full model (and to the other training seeds of the full model, which give the
size of a difference that means nothing) on the known-answer judge: for every
spectrum, W1 of the leave-out model over W1 of the full model, split by
whether the leave-out model ever saw that spectrum's family.

    python scripts/step3/leave_out_summary.py out/step3/leave_out F3_60k_lo F3_60k_s1 [F3_60k_s2 F3_60k_s3]

The left-out families are read from the leave-out checkpoint
(out/step3/<name>.pt, recipe["left_out"]).  Writes <out>.{json,png}.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SETTINGS = (("8", "0.0001"), ("8", "0.001"), ("8", "0.01"), ("2", "0.001"), ("16", "0.001"))
HELD, KEPT, SEED, ME, INK = "#b3261e", "#2a78d6", "#c9c8c3", "#7d7c78", "#0b0b0b"


def load(name):
    out = {}
    for b, n in SETTINGS:
        p = Path(f"out/eval/exact_models_{name}/exact_models_b{b}_n{n}_lam10.json")
        if p.exists():
            out[(b, n)] = json.load(open(p))["rows"]
    return out


def ratio_ci(d):
    """exp(mean), with a 95% t interval, of log-ratios d."""
    from scipy import stats
    d = np.asarray(d, float)
    if len(d) < 2:
        return [float(np.exp(d.mean()))] * 3
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return [float(np.exp(d.mean())), float(np.exp(d.mean() - h)), float(np.exp(d.mean() + h))]


def main():
    from spevae.models.flow import load_flow
    out, lo_name, full_name, seeds = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
    out.parent.mkdir(parents=True, exist_ok=True)
    left = set(load_flow(f"out/step3/{lo_name}.pt").meta["recipe"].get("left_out") or [])
    if not left:
        raise SystemExit(f"{lo_name} records no left-out families")
    lo, full, others = load(lo_name), load(full_name), {s: load(s) for s in seeds}
    rec = []                                         # one record per (setting, spectrum)
    for key in lo:
        for k, (a, b) in enumerate(zip(lo[key], full[key])):
            assert a["family"] == b["family"] and a["params"] == b["params"], (key, k)
            r = {"beta": key[0], "noise": key[1], "family": a["family"], "kind": a["kind"],
                 "held_out": a["family"] in left, "lo": a["w1"]["flow"], "full": b["w1"]["flow"],
                 "maxent": a["w1"]["maxent"],
                 "seeds": [o[key][k]["w1"]["flow"] for o in others.values() if key in o]}
            rec.append(r)
    lg = lambda r: math.log(r["lo"] / r["full"])
    res = {"leave_out": lo_name, "full": full_name, "seeds": seeds, "left_out": sorted(left), "groups": {}}
    for kind in ("continuum", "discrete"):
        for held in (True, False):
            sel = [r for r in rec if r["kind"] == kind and r["held_out"] == held]
            if not sel:
                continue
            g = {"n": len(sel), "lo_over_full": ratio_ci([lg(r) for r in sel]),
                 "lo_over_maxent": ratio_ci([math.log(r["lo"] / r["maxent"]) for r in sel]),
                 "full_over_maxent": ratio_ci([math.log(r["full"] / r["maxent"]) for r in sel]),
                 "lo_below_maxent": sum(r["lo"] < r["maxent"] for r in sel),
                 "full_below_maxent": sum(r["full"] < r["maxent"] for r in sel)}
            sd = [math.log(s / r["full"]) for r in sel for s in r["seeds"]]
            if sd:
                g["other_seed_over_full"] = ratio_ci(sd)
                g["other_seed_abs_log_ratio_median"] = float(np.median(np.abs(sd)))
            g["by_setting"] = {f"beta {b}, noise {n}": ratio_ci([lg(r) for r in sel if (r["beta"], r["noise"]) == (b, n)])
                               for b, n in SETTINGS if any((r["beta"], r["noise"]) == (b, n) for r in sel)}
            res["groups"][f"{kind}, {'left out' if held else 'kept'}"] = g
    res["by_family"] = {}
    for fam in dict.fromkeys(r["family"] for r in rec):
        sel = [r for r in rec if r["family"] == fam]
        res["by_family"][fam] = {"held_out": sel[0]["held_out"], "kind": sel[0]["kind"], "n": len(sel),
                                 "lo_over_full": ratio_ci([lg(r) for r in sel]),
                                 "lo_over_maxent": ratio_ci([math.log(r["lo"] / r["maxent"]) for r in sel]),
                                 "full_over_maxent": ratio_ci([math.log(r["full"] / r["maxent"]) for r in sel])}
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(res, fh, indent=1)

    fams = list(res["by_family"])
    f, (a0, a1) = plt.subplots(1, 2, figsize=(17, 5.2), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.5, 1]})
    for i, fam in enumerate(fams):
        sel = [r for r in rec if r["family"] == fam]
        col = HELD if sel[0]["held_out"] else KEPT
        jit = np.linspace(-0.3, 0.3, len(sel))
        for r, dx in zip(sel, jit):
            for s in r["seeds"]:
                a0.plot([i + dx], [s / r["full"]], "o", color=SEED, ms=3.5, zorder=1)
        a0.plot(i + jit, [r["lo"] / r["full"] for r in sel], "o", color=col, ms=5, mec="white", mew=0.5, zorder=3)
        g, lo_, hi_ = res["by_family"][fam]["lo_over_full"]
        a0.plot([i - 0.38, i + 0.38], [g, g], color=col, lw=2.0, zorder=4)
    a0.axhline(1.0, color=INK, lw=0.8, ls=":")
    a0.set_yscale("log")
    a0.set_xticks(range(len(fams)), [f.replace("_", "\n") for f in fams], fontsize=7)
    a0.set_ylabel(r"W$_1$ of the leave-out model / W$_1$ of the full model")
    a0.plot([], [], "o", color=HELD, label="family never seen by the leave-out model")
    a0.plot([], [], "o", color=KEPT, label="family it was trained on")
    if seeds:
        a0.plot([], [], "o", color=SEED, ms=4, label="another training seed of the full model / full model")
    a0.legend(fontsize=8, frameon=False)
    a0.set_title("one point per spectrum and setting (bar: geometric mean of the family); above 1 = worse than the full model",
                 fontsize=9.5)
    names = list(res["groups"])
    for j, nm in enumerate(names):
        g = res["groups"][nm]
        col = HELD if "left out" in nm else KEPT
        for key, dx, c, mk in (("lo_over_maxent", -0.15, col, "o"), ("full_over_maxent", 0.15, "#5b5a56", "s")):
            m, lo_, hi_ = g[key]
            a1.errorbar([j + dx], [m], yerr=[[m - lo_], [hi_ - m]], fmt=mk, color=c, ms=7, capsize=3, lw=1.2)
    a1.axhline(1.0, color=INK, lw=0.8, ls=":")
    a1.set_yscale("log")
    a1.set_xticks(range(len(names)), [n.replace("continuum", "lattice spectra").replace("discrete", "line spectra")
                                      .replace(", ", ",\n") for n in names], fontsize=8.5)
    a1.set_ylabel(r"W$_1$ / W$_1$ of MaxEnt (geometric mean, 95% interval)")
    a1.plot([], [], "o", color=INK, label="leave-out model")
    a1.plot([], [], "s", color="#5b5a56", label="full model")
    a1.legend(fontsize=8, frameon=False)
    a1.set_title("against MaxEnt on the same data (below 1 = better than MaxEnt)", fontsize=9.5)
    for ax in (a0, a1):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle(f"Leave-families-out arm ({lo_name} against {full_name}): known answers at "
               f"{len(lo)} settings; never seen: {', '.join(sorted(left))}", fontsize=10.5)
    f.savefig(out.with_suffix(".png"), dpi=150)
    fmt = lambda t: "{:.3f} [{:.3f}, {:.3f}]".format(*t)
    for nm, g in res["groups"].items():
        print(f"{nm:22s} n={g['n']:3d}  leave-out/full {fmt(g['lo_over_full'])}"
              + (f"  (other seed/full {fmt(g['other_seed_over_full'])})" if "other_seed_over_full" in g else "")
              + f"  | vs MaxEnt: leave-out {fmt(g['lo_over_maxent'])}, full {fmt(g['full_over_maxent'])}"
              f"  | below MaxEnt {g['lo_below_maxent']}/{g['n']} (full {g['full_below_maxent']})")
    for fam, g in res["by_family"].items():
        print(f"  {fam:16s} {'LEFT OUT' if g['held_out'] else '        '} leave-out/full {fmt(g['lo_over_full'])}"
              f"  vs MaxEnt {fmt(g['lo_over_maxent'])}")
    print(f"-> {out}.json + .png")


if __name__ == "__main__":
    main()
