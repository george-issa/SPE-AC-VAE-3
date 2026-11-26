"""`continue_posterior(discrete=True)` on known answers: do the intervals hold on line spectra?

A spectrum of many close lines is indistinguishable in the data from a
continuous one, the prior favours the continuous one, and the report's window
intervals are then too narrow for the lines, with nothing to flag
(TRAINING.md (xxii), (xxvi): 0.34 / 0.62 at nominal 0.50 / 0.90 on the line
spectra the report does not flag).  `discrete=True` is the user's statement
that the spectrum is a set of lines; the intervals then span both readings of
every pole, as fitted and as one sharp line within its half-width.  Here the
report is run with that statement, as a user runs it, on the exact line spectra
of the floor-option judge (same spectra, noise draws and sample seeds, so the
fits are the same and only the intervals differ) and scored against the report
without it.  The reading was chosen on the five settings of the known-answer
judge (`discrete_rules.py`); the edge-of-range settings are held out.

    python scripts/step3/discrete_option_judge.py run OUT FLOW BETA NOISE [kind=discrete]
    python scripts/step3/discrete_option_judge.py fig OUT REFERENCE_DIR [REFERENCE_DIR ...]

`run` writes OUT/discrete_b<beta>_n<noise>[_dt<dtau>][_continuum].json and the
fitted poles beside it (JUDGE_DTAU as in `exact_models_judge.py`);
`kind=continuum` runs the exact lattice spectra instead (what a wrong statement
costs).  `fig` pairs every result in OUT with the `floor_option_judge.py`
result of the same setting found in the REFERENCE_DIRs and writes
OUT/discrete_option.{json,png}.
"""
import json
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

OFF, ON, INK = "#2a78d6", "#0b7a75", "#0b0b0b"


def run(out, flow_path, beta, noise, kind):
    from spevae.evaluation import cases as C
    from spevae.models.flow import load_flow
    from spevae.posterior import DEFAULT_WINDOWS, continue_posterior
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    flow = load_flow(flow_path).eval()
    items = J.make_truths(beta) + F.extra_discrete(beta, 2)
    gphys = flow.coords.gamma_min / beta
    rows, poles = [], {}
    for i, item in enumerate(items):
        if item["kind"] != kind:
            continue
        inst = C.draw(item["truth"], noise, J.SEED)
        truth_w = F.true_windows(item, DEFAULT_WINDOWS)
        t0 = time.time()
        rep = continue_posterior(flow, inst.G, inst.truth.x, beta, inst.sigma, inst.isr, n_samples=F.S,
                                 omega=J.WG, bounds=True, discrete=True,
                                 generator=torch.Generator().manual_seed(9))
        fit = F.score(rep, item, DEFAULT_WINDOWS, truth_w, gphys)
        fit["as_fitted"] = [list(w["as_fitted"]) for w in rep.windows]
        fit["discrete_lines"] = [list(w["discrete_lines"]) for w in rep.windows]
        # the accepted samples' poles (physical units), so that other readings can be scored without refitting
        poles[f"p{i}"] = np.stack([getattr(rep.poles, k).double().numpy() for k in ("eps", "gamma", "a", "b")])
        rows.append({"index": i, "family": item["family"], "kind": kind, "params": item["params"],
                     "near_edge": [m for _, m in truth_w], "fit": fit, "seconds": time.time() - t0})
        q = np.array(fit["reported"])
        t = np.array(fit["truth"])
        print(f"{i:2d} {item['family']:16s} W1 {fit['w1']:.4f} | exact weight inside the stated 90% interval in "
              f"{int(((q[:, 0] <= t) & (t <= q[:, 3])).sum())} of {len(t)} windows, median width "
              f"{np.median(q[:, 3] - q[:, 0]):.3f} | {rows[-1]['seconds']:.0f}s", flush=True)
    tag = (f"discrete_b{beta:g}_n{noise:g}" + ("" if J.DTAU == 0.125 else f"_dt{J.DTAU:g}")
           + ("" if kind == "discrete" else f"_{kind}"))
    np.savez_compressed(out / f"{tag}.poles.npz", **poles)
    with open(out / f"{tag}.json", "w") as fh:
        json.dump({"flow": flow_path, "beta": beta, "noise": noise, "dtau": J.DTAU, "kind": kind, "rows": rows},
                  fh, indent=1, default=float)
    print(f"-> {out}/{tag}.json")


def width(rows, arm):
    q = np.concatenate([np.asarray(r[arm]["reported"]) for r in rows])
    return {"w90_median": float(np.median(q[:, 3] - q[:, 0])), "w50_median": float(np.median(q[:, 2] - q[:, 1]))}


def share(rows, arm):
    """Median over windows of the stated 90% interval's width over the range the data alone allow."""
    v = []
    for r in rows:
        if "bracket" not in r["ref"]:
            continue
        q, b = np.asarray(r[arm]["reported"]), np.asarray(r["ref"]["bracket"])
        v += list((q[:, 3] - q[:, 0]) / np.maximum(b[:, 1] - b[:, 0], 1e-12))
    return float(np.median(v)) if v else None


TUNE = {(8.0, 1e-4, 0.125), (8.0, 1e-3, 0.125), (8.0, 1e-2, 0.125), (2.0, 1e-3, 0.125), (16.0, 1e-3, 0.125)}


def fig(out, refdirs):
    out, refdirs = Path(out), [Path(d) for d in refdirs]
    pairs = []                              # one row per spectrum: ref = the report without the statement
    for p in sorted(out.glob("discrete_b*_n*.json")):
        r = json.load(open(p))
        dtau = r.get("dtau", 0.125)
        name = f"floor_option_b{r['beta']:g}_n{r['noise']:g}" + ("" if dtau == 0.125 else f"_dt{dtau:g}") + ".json"
        ref = next((json.load(open(d / name))["rows"] for d in refdirs if (d / name).exists()), None)
        if ref is None:
            raise SystemExit(f"no {name} in {refdirs}")
        for x in r["rows"]:
            y = ref[x["index"]]
            assert y["family"] == x["family"] and y["params"] == x["params"]
            # the same fit up to the order of floating-point sums (thread count): 274 of 284 agree to
            # 1e-6, the others to 0.5%
            assert abs(y["auto"]["w1"] - x["fit"]["w1"]) < 1e-2 * y["auto"]["w1"], "the fits differ"
            pairs.append({"kind": x["kind"], "beta": r["beta"], "noise": r["noise"], "dtau": dtau,
                          "tune": (r["beta"], r["noise"], dtau) in TUNE, "fired": y["fired"],
                          "near_edge": x["near_edge"], "ref": y["auto"], "on": x["fit"]})
    line = lambda r: r["kind"] == "discrete"
    groups = (("lines not flagged, tuning settings", lambda r: line(r) and r["tune"] and not r["fired"]),
              ("lines not flagged, held-out settings", lambda r: line(r) and not r["tune"] and not r["fired"]),
              ("lines flagged, tuning settings", lambda r: line(r) and r["tune"] and r["fired"]),
              ("lines flagged, held-out settings", lambda r: line(r) and not r["tune"] and r["fired"]),
              ("all line spectra", line),
              ("lattice spectra (statement wrong)", lambda r: r["kind"] == "continuum"))
    res = {"reference": [str(d) for d in refdirs], "groups": {}}
    for name, sel in groups:
        rows = [r for r in pairs if sel(r)]
        if not rows:
            continue
        res["groups"][name] = {
            "spectra": len(rows),
            "without": {**F.coverage(rows, "ref", "reported", skip_edge=False), **width(rows, "ref"),
                        "share_of_data_alone": share(rows, "ref")},
            "discrete": {**F.coverage(rows, "on", "reported", skip_edge=False), **width(rows, "on"),
                         "share_of_data_alone": share(rows, "on")}}
    by = {}
    for r in pairs:
        if r["kind"] == "discrete":
            by.setdefault(f"beta {r['beta']:g}, noise {r['noise']:g}, dtau {r['dtau']:g}"
                          + ("" if r["tune"] else " (held out)"), []).append(r)
    res["line_spectra_by_setting"] = {k: {"spectra": len(v), "without": F.coverage(v, "ref", "reported", skip_edge=False),
                                          "discrete": F.coverage(v, "on", "reported", skip_edge=False)}
                                      for k, v in by.items()}
    with open(out / "discrete_option.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    G = res["groups"]
    f, (a0, a1) = plt.subplots(1, 2, figsize=(19.5, 5.4), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.5, 1.0]})
    for j, (name, g) in enumerate(G.items()):
        for k, (arm, col) in enumerate((("without", OFF), ("discrete", ON))):
            c = g[arm]
            for key, se, dx, fill in (("c50", "se50", -0.2, True), ("c90", "se90", 0.2, False)):
                a0.errorbar([j + (k - 0.5) * 0.14 + dx], [c[key]], yerr=[1.96 * c[se]], fmt="o", color=col, ms=7.5,
                            mfc=col if fill else "white", capsize=3, lw=1.3)
            a1.bar([j + (k - 0.5) * 0.36], [c["share_of_data_alone"] or 0.0], width=0.34, color=col)
    for lv in (0.5, 0.9):
        a0.axhline(lv, color=INK, lw=0.8, ls="--")
    a0.set_ylim(0, 1.03)
    ticks = [n.replace(", ", ",\n").replace(" (statement", "\n(statement") + f"\n({g['spectra']})"
             for n, g in G.items()]
    a0.set_xticks(range(len(G)), ticks, fontsize=8.5)
    a0.set_ylabel("share of windows with the exact weight inside")
    a0.plot([], [], "o", color=OFF, label="report as it is")
    a0.plot([], [], "o", color=ON, label="with discrete=True")
    a0.plot([], [], "o", color=INK, label="stated 50% interval")
    a0.plot([], [], "o", color=INK, mfc="white", label="stated 90% interval")
    a0.legend(fontsize=8.5, frameon=False, loc="lower right", ncol=2)
    a0.set_title("exact window weights inside the stated intervals (dashes: nominal)", fontsize=9.5)
    a1.set_xticks(range(len(G)), ticks, fontsize=7.5)
    a1.set_ylim(0, 1.0)
    a1.set_ylabel("median share")
    a1.set_title("what the wider intervals cost: width of the stated 90% interval\nas a share of the range the "
                 "data alone allow", fontsize=9.5)
    for ax in (a0, a1):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle("Stating that the spectrum is discrete (continue_posterior(discrete=True)): the same fits, "
               "window intervals spanning both readings of every pole\n(tuning settings: the five the reading was "
               "chosen on; held-out settings: the six edge-of-range settings; flagged: narrow lines flagged by the "
               "report itself)", fontsize=10.5)
    f.savefig(out / "discrete_option.png", dpi=150)

    for name, g in G.items():
        for arm in ("without", "discrete"):
            c = g[arm]
            print(f"{name:40s} {arm:8s} {c['c50']:.3f}±{c['se50']:.3f} / {c['c90']:.3f}±{c['se90']:.3f}   median "
                  f"width 50% {c['w50_median']:.3f}, 90% {c['w90_median']:.3f}, share of the data-alone range "
                  f"{c['share_of_data_alone'] if c['share_of_data_alone'] is None else round(c['share_of_data_alone'], 2)}"
                  f"   ({g['spectra']} spectra)")
    for k, v in res["line_spectra_by_setting"].items():
        print(f"   lines, {k:46s} without {v['without']['c50']:.2f} / {v['without']['c90']:.2f}   discrete "
              f"{v['discrete']['c50']:.2f} / {v['discrete']['c90']:.2f}   ({v['spectra']})")
    print(f"-> {out}/discrete_option.{{json,png}}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        torch.set_num_threads(2)
        run(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]),
            sys.argv[6] if len(sys.argv) > 6 else "discrete")
    elif sys.argv[1] == "fig":
        fig(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)
