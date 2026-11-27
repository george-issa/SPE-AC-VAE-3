"""Zero-width lines in the posterior report, judged on known answers.

Until TRAINING.md (xxxi) a pole the flow pinned at its width floor was fitted
as a Lorentzian of a quarter of that floor.  A Lorentzian of any width has
tails: at error bars of 1e-5 the fit of a sharp line then fails for half the
samples and loses to MaxEnt by 1.6 times ((xxviii)), and at any error bar the
tails are most of the reported error on a line spectrum.  The report can now
fit those poles as zero-width lines, and does in two cases: when peaks of the
minimum width do not reach the data (the data ask for it), and when the user
states that the spectrum is discrete.  Here it is run both ways, as a user
runs it, on the spectra of `floor_option_judge.py` -- 20 exact lattice
spectra and 24 exactly diagonalised line spectra per setting, same noise
draws and sample seeds -- and scored against the stored results of the
report before the change and of MaxEnt on the same data.

    python scripts/step3/line_fit_judge.py run OUT FLOW BETA NOISE [K N] [stated]
    python scripts/step3/line_fit_judge.py fig OUT REFERENCE_DIR [REFERENCE_DIR ...] [always=DIR]

`run` writes OUT/line_fit_b<beta>_n<noise>[_dt<dtau>][.shard<K>of<N>].json:
the report at its defaults on all 44 spectra (K, N: spectra K mod N).  With
`stated` it writes OUT/line_fit_stated_...: the 24 line spectra with
`discrete=True` and the range the data alone allow, as `spevae continue
--discrete` runs.  JUDGE_DTAU as in `exact_models_judge.py`.  `fig` pairs
every result in OUT with the `floor_option_judge.py` result of the same
setting found in the REFERENCE_DIRs and writes OUT/line_fit.{json,png};
`always=DIR` adds the results of a run in which lines were fitted whenever
they fit (`posterior.PEAKS_FIT_BELOW` above 1), the arm that was not shipped.
"""
import json
import os
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
import torch

import exact_models_judge as J
import floor_option_judge as F

OLD, NEW, STATED, ALWAYS, INK = "#9ec5f4", "#2a78d6", "#0b7a75", "#c9c8c3", "#0b0b0b"
KEYS = ("line_fit", "weight_on_lines", "accepted_fraction_as_lines", "accepted_fraction_lowered_floor",
        "samples_restarted", "samples_fitting_at_trained_floor", "samples_mixed", "window_interval_floor")
# JUDGE_BOUNDS=0 leaves out the range the data alone allow in a `stated` run (it takes minutes per
# spectrum at beta = 20 with error bars of 1e-5); the discrete reading is then not cut to it
BOUNDS = os.environ.get("JUDGE_BOUNDS", "1") != "0"
ARMS = (("old", "before", OLD), ("default", "the report at its defaults", NEW),
        ("stated", "stated to be discrete", STATED))


def run(out, flow_path, beta, noise, k, n, stated):
    from spevae.evaluation import cases as C
    from spevae.models.flow import load_flow
    from spevae.posterior import DEFAULT_WINDOWS, continue_posterior
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    flow = load_flow(flow_path).eval()
    items = J.make_truths(beta) + F.extra_discrete(beta, 2)
    gphys = flow.coords.gamma_min / beta
    rows = []
    for i, item in enumerate(items):
        if i % n != k or (stated and item["kind"] != "discrete"):
            continue
        inst = C.draw(item["truth"], noise, J.SEED)
        truth_w = F.true_windows(item, DEFAULT_WINDOWS)
        t0 = time.time()
        rep = continue_posterior(flow, inst.G, inst.truth.x, beta, inst.sigma, inst.isr, n_samples=F.S,
                                 omega=J.WG, bounds=stated and BOUNDS, discrete=stated,
                                 generator=torch.Generator().manual_seed(9))
        fit = F.score(rep, item, DEFAULT_WINDOWS, truth_w, gphys)
        fit.update({key: rep.diagnostics[key] for key in KEYS if key in rep.diagnostics})
        fit["readings"] = {key: [list(w[key]) for w in rep.windows]
                           for key in ("as_fitted", "sharp_lines", "minimum_width_peaks", "discrete_lines")
                           if key in rep.windows[0]}
        rows.append({"index": i, "family": item["family"], "kind": item["kind"], "params": item["params"],
                     "near_edge": [m for _, m in truth_w], "fit": fit, "seconds": time.time() - t0})
        q, t = np.array(fit["reported"]), np.array(fit["truth"])
        print(f"{i:2d} {item['kind']:9s} {item['family']:16s} W1 {fit['w1']:.4f} accepted {fit['accepted']:.0%} "
              f"on lines {fit.get('weight_on_lines', 0.0):.2f} | exact weight inside the stated 50% / 90% interval "
              f"in {int(((q[:, 1] <= t) & (t <= q[:, 2])).sum())} / {int(((q[:, 0] <= t) & (t <= q[:, 3])).sum())}"
              f" of {len(t)} windows | {rows[-1]['seconds']:.0f}s", flush=True)
    tag = ("line_fit_" + ("stated_" if stated else "") + f"b{beta:g}_n{noise:g}"
           + ("" if J.DTAU == 0.125 else f"_dt{J.DTAU:g}") + (f".shard{k}of{n}" if n > 1 else ""))
    with open(out / f"{tag}.json", "w") as fh:
        json.dump({"flow": flow_path, "beta": beta, "noise": noise, "dtau": J.DTAU, "stated": bool(stated),
                   "bounds": bool(stated and BOUNDS), "rows": rows}, fh, indent=1, default=float)
    print(f"-> {out}/{tag}.json")


def load(folder, prefix):
    """{(beta, noise, dtau): {spectrum index: row}} of every result `prefix`b*.json in `folder`."""
    by = {}
    for p in sorted(Path(folder).glob(prefix + "b*_n*.json")):
        r = json.load(open(p))
        by.setdefault((r["beta"], r["noise"], r.get("dtau", 0.125)), {}).update({x["index"]: x for x in r["rows"]})
    return by


def arm_stats(rows, arm):
    """One arm on one group of spectra; MaxEnt is stored for every line spectrum and for the
    lattice spectra of the edge settings."""
    rows = [r for r in rows if r.get(arm) is not None]
    if len(rows) < 2:
        return None
    w, old = (np.array([r[a]["w1"] for r in rows]) for a in (arm, "old"))
    has = np.array([r["maxent"] is not None for r in rows])
    me = np.array([r["maxent"]["w1"] for r in rows if r["maxent"] is not None])
    q = np.concatenate([np.asarray(r[arm]["reported"]) for r in rows])
    out = {"spectra": len(rows), "w1": F.gmean(w), "over_before": F.ratio_ci(w, old),
           "accepted_median": float(np.median([r[arm]["accepted"] for r in rows])),
           "accepted_below_half": int(sum(r[arm]["accepted"] < 0.5 for r in rows)),
           "with_lines": int(sum(bool(r[arm].get("line_fit")) for r in rows)),
           "width90_median": float(np.median(q[:, 3] - q[:, 0])),
           "stated": F.coverage(rows, arm, "reported", skip_edge=False),
           "stated_no_line_at_an_edge": F.coverage(rows, arm, "reported", skip_edge=True)}
    if has.sum() >= 2:
        out.update(with_maxent=int(has.sum()), w1_maxent=F.gmean(me), over_maxent=F.ratio_ci(w[has], me),
                   lower_than_maxent=int((w[has] < me).sum()))
    return out


def fig(out, args):
    out = Path(out)
    always_dir = next((a.split("=", 1)[1] for a in args if a.startswith("always=")), None)
    refdirs = [Path(a) for a in args if not a.startswith("always=")]
    default, stated = load(out, "line_fit_"), load(out, "line_fit_stated_")
    always = load(always_dir, "line_fit_") if always_dir else {}
    arms = ARMS + ((("always", "lines whenever they fit (not shipped)", ALWAYS),) if always else ())
    pairs = []
    for st in sorted(set(default) | set(stated)):
        beta, noise, dtau = st
        name = f"floor_option_b{beta:g}_n{noise:g}" + ("" if dtau == 0.125 else f"_dt{dtau:g}") + ".json"
        ref = next((json.load(open(d / name))["rows"] for d in refdirs if (d / name).exists()), None)
        if ref is None:
            raise SystemExit(f"no {name} in {refdirs}")
        for i, y in enumerate(ref):
            got = {a: src.get(st, {}).get(i) for a, src in (("default", default), ("stated", stated),
                                                            ("always", always))}
            if got["default"] is None and got["stated"] is None:
                continue
            for x in got.values():
                assert x is None or (x["family"] == y["family"] and x["params"] == y["params"])
            pairs.append({"setting": st, "kind": y["kind"], "family": y["family"], "fired": bool(y["fired"]),
                          "near_edge": y["near_edge"], "maxent": y.get("maxent"), "old": y["auto"],
                          **{a: (x["fit"] if x else None) for a, x in got.items()}})
    line = lambda r: r["kind"] == "discrete"
    groups = (("line spectra, flagged", lambda r: line(r) and r["fired"]),
              ("line spectra, not flagged", lambda r: line(r) and not r["fired"]),
              ("line spectra, all", line),
              ("lattice spectra", lambda r: not line(r)))
    settings = sorted({r["setting"] for r in pairs})
    res = {"reference": [str(d) for d in refdirs], "always": always_dir, "pooled": {}, "settings": []}
    for name, sel in groups:
        rows = [r for r in pairs if sel(r)]
        res["pooled"][name] = {a: arm_stats(rows, a) for a, _, _ in arms}
    for st in settings:
        s = {"beta": st[0], "noise": st[1], "dtau": st[2]}
        for name, sel in groups:
            rows = [r for r in pairs if r["setting"] == st and sel(r)]
            s[name] = {a: arm_stats(rows, a) for a, _, _ in arms}
        # a spectrum fitted without lines takes the same route as before: its numbers must not have moved
        same = [r for r in pairs if r["setting"] == st and r["default"] is not None
                and not r["default"].get("line_fit")]
        s["default_without_lines"] = {"spectra": len(same), "within_1_percent": int(sum(
            abs(r["default"]["w1"] - r["old"]["w1"]) <= 1e-2 * r["old"]["w1"] for r in same))}
        res["settings"].append(s)
    with open(out / "line_fit.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    names = [f"beta {b:g}\nerr {n:g}" + ("" if d == 0.125 else f"\ndtau {d:g}") for b, n, d in settings]
    x = np.arange(len(settings))
    off = np.linspace(-0.27, 0.27, len(arms))
    f, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    (a0, a1), (a2, a3) = axes
    for dx, (arm, lab, col) in zip(off, arms):
        v = [s["line spectra, all"][arm] for s in res["settings"]]
        ok = [i for i, t in enumerate(v) if t and "over_maxent" in t]
        if not ok:
            continue
        m, lo, hi = (np.array([v[i]["over_maxent"][j] for i in ok]) for j in range(3))
        a0.errorbar(x[ok] + dx, m, yerr=[m - lo, hi - m], fmt="o", color=col, ms=7, capsize=3, lw=1.4, label=lab)
    a0.axhline(1.0, color=INK, lw=0.8, ls=":")
    a0.set_yscale("log")
    a0.set_xticks(x, names, fontsize=7.5)
    a0.set_ylabel(r"W$_1$ of the report / W$_1$ of MaxEnt (geometric mean, 95% interval)")
    a0.set_title("exact line spectra (24 per setting): error against MaxEnt on the same data; below 1 = better",
                 fontsize=9.5)
    a0.legend(fontsize=8.5, frameon=False)
    fams = sorted({r["family"] for r in pairs if line(r)})
    cmap = plt.get_cmap("tab10")
    for k, fam in enumerate(fams):
        rows = [r for r in pairs if r["family"] == fam and r["fired"] and r["maxent"] and r["stated"]]
        a1.plot([r["old"]["w1"] / r["maxent"]["w1"] for r in rows],
                [r["stated"]["w1"] / r["maxent"]["w1"] for r in rows],
                "o", ms=4.5, alpha=0.75, color=cmap(k), label=fam.replace("_", " "))
    lim = (0.02, 6.0)
    a1.plot(lim, lim, color=INK, lw=0.8, ls=":")
    a1.axhline(1.0, color=INK, lw=0.6)
    a1.axvline(1.0, color=INK, lw=0.6)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlim(*lim)
    a1.set_ylim(*lim)
    a1.set_xlabel(r"before: W$_1$ / W$_1$ of MaxEnt")
    a1.set_ylabel(r"stated to be discrete: W$_1$ / W$_1$ of MaxEnt")
    a1.legend(fontsize=8, frameon=False, loc="upper left")
    a1.set_title("one point per flagged line spectrum, all settings (below the diagonal = improved)", fontsize=9.5)
    for dx, (arm, lab, col) in zip(off, arms):
        v = [s["line spectra, all"][arm] for s in res["settings"]]
        ok = [i for i, t in enumerate(v) if t]
        if not ok:
            continue
        c = [v[i]["stated"] for i in ok]
        a2.errorbar(x[ok] + dx - 0.035, [t["c50"] for t in c], yerr=[1.96 * t["se50"] for t in c], fmt="o",
                    color=col, ms=6, capsize=2, lw=1.1)
        a2.errorbar(x[ok] + dx + 0.035, [t["c90"] for t in c], yerr=[1.96 * t["se90"] for t in c], fmt="o",
                    color=col, ms=6, mfc="white", capsize=2, lw=1.1)
        a2.plot([], [], "o", color=col, label=lab)
    for lv in (0.5, 0.9):
        a2.axhline(lv, color=INK, lw=0.8, ls="--")
    a2.set_ylim(0, 1.04)
    a2.set_xticks(x, names, fontsize=7.5)
    a2.set_ylabel("share of windows with the exact weight inside")
    a2.plot([], [], "o", color=INK, label="stated 50% interval")
    a2.plot([], [], "o", color=INK, mfc="white", label="stated 90% interval")
    a2.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    a2.set_title("all 24 line spectra per setting: exact window weights inside the stated intervals "
                 "(dashes: nominal)", fontsize=9.5)
    a3.axis("off")
    fmt = lambda t: "{:.2f} [{:.2f}, {:.2f}]".format(*t)
    cells, labels = [], []
    for name, v in res["pooled"].items():
        for arm, lab, _ in arms:
            t = v[arm]
            if not t:
                continue
            labels.append(f"{name}: {lab} ({t['spectra']})")
            c = t["stated"]
            cells.append([fmt(t["over_maxent"]) if "over_maxent" in t else "-",
                          f"{t['lower_than_maxent']} of {t['with_maxent']}" if "over_maxent" in t else "-",
                          fmt(t["over_before"]), f"{c['c50']:.2f} / {c['c90']:.2f}", f"{t['width90_median']:.3f}",
                          f"{t['with_lines']}", f"{t['accepted_below_half']}"])
    tab = a3.table(cellText=cells, rowLabels=labels, loc="center right", cellLoc="center",
                   colLabels=["W1 / MaxEnt", "lower than\nMaxEnt", "W1 / W1 before", "inside stated\n50% / 90%",
                              "median 90%\nwidth", "fitted with\nlines", "under half the\nsamples fit"])
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    tab.scale(1.0, 1.55)
    tab.auto_set_column_width(list(range(7)))
    a3.set_title("all settings pooled", fontsize=9.5)
    for ax in (a0, a1, a2):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle("Narrow poles fitted as zero-width lines when the data ask for it or the spectrum is stated to be "
               "discrete: the posterior report on exactly solvable models", fontsize=11)
    f.savefig(out / "line_fit.png", dpi=150)

    for name, v in res["pooled"].items():
        for arm, lab, _ in arms:
            t = v[arm]
            if not t:
                continue
            c = t["stated"]
            print(f"{name:26s} {lab:38s} n {t['spectra']:3d}  "
                  + (f"W1/MaxEnt {fmt(t['over_maxent'])} lower on {t['lower_than_maxent']:3d} of {t['with_maxent']:3d}  "
                     if "over_maxent" in t else " " * 52)
                  + f"/before {fmt(t['over_before'])}  stated {c['c50']:.3f}±{c['se50']:.3f}/{c['c90']:.3f}±{c['se90']:.3f}"
                  f"  width {t['width90_median']:.3f}  with lines {t['with_lines']:3d}  accepted<half "
                  f"{t['accepted_below_half']}")
    for s in res["settings"]:
        v, u = s["line spectra, all"], s["default_without_lines"]
        print(f"beta {s['beta']:g} noise {s['noise']:g} dtau {s['dtau']:g}: lines W1/MaxEnt "
              + " | ".join(f"{lab}: {fmt(v[arm]['over_maxent'])} ({v[arm]['lower_than_maxent']}/{v[arm]['spectra']}), "
                           f"{v[arm]['stated']['c50']:.2f}/{v[arm]['stated']['c90']:.2f}, lines on "
                           f"{v[arm]['with_lines']}" for arm, lab, _ in arms if v[arm])
              + f" | default without lines unchanged {u['within_1_percent']}/{u['spectra']}")
    print(f"-> {out}/line_fit.{{json,png}}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        torch.set_num_threads(3)
        rest = [a for a in sys.argv[6:] if a != "stated"]
        run(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]),
            int(rest[0]) if rest else 0, int(rest[1]) if len(rest) > 1 else 1, "stated" in sys.argv[6:])
    elif sys.argv[1] == "fig":
        fig(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)
