"""Did training on exact line spectra (pole labels of exact line data) help?  Two models, scored.

NEW was trained with exact line spectra in its label bank (`run_lines.sh`),
OLD with broadened lines only (`run_full.sh`).  Both were judged on the same
known answers with the same noise draws, so every comparison is paired:

  - the known-answer judge (samples as trained, `exact_models_judge.py`);
  - the posterior report as a user gets it (`floor_option_judge.py`): W1, how
    many line spectra are flagged as narrow, and how often the exact weight of
    a frequency window lies inside the central 50% / 90% of the samples and
    inside the intervals the report states.  OLD carries a band table and NEW
    does not, so the like-for-like comparison is the central quantiles.

    python scripts/step3/lines_retrain_summary.py OUT NEW OLD NEW_REPORT_DIR OLD_REPORT_DIR

NEW / OLD are checkpoint names (out/eval/exact_models_<name>/); the two
directories hold the floor-option judge's output.  Writes OUT.{json,png}.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/step3")
sys.path.insert(0, "scripts/eval")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import floor_option_judge as F

SETTINGS = (("8", "0.0001"), ("8", "0.001"), ("8", "0.01"), ("2", "0.001"), ("16", "0.001"))
NEW, OLD, ME, INK = "#0b7a75", "#2a78d6", "#7d7c78", "#0b0b0b"
GROUPS = ("lines, flagged by both", "lines, flagged by neither", "lines, flagged by one", "lattice")


def ratio_ci(d):
    from scipy import stats
    d = np.asarray(d, float)
    if len(d) < 2:
        return [float(np.exp(d.mean()))] * 3 if len(d) else [float("nan")] * 3
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return [float(np.exp(d.mean())), float(np.exp(d.mean() - h)), float(np.exp(d.mean() + h))]


def gm(v):
    return float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")


def paired(a_rows, b_rows):
    assert len(a_rows) == len(b_rows)
    for a, b in zip(a_rows, b_rows):
        assert a["family"] == b["family"] and a["params"] == b["params"]
    return list(zip(a_rows, b_rows))


def main():
    out, new, old, dnew, dold = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"new": new, "old": old, "as_trained": {}, "report": {}}

    # 1. the known-answer judge, samples as trained
    pool = {"continuum": [], "discrete": []}
    for b, n in SETTINGS:
        A = json.load(open(f"out/eval/exact_models_{new}/exact_models_b{b}_n{n}_lam10.json"))["rows"]
        B = json.load(open(f"out/eval/exact_models_{old}/exact_models_b{b}_n{n}_lam10.json"))["rows"]
        s = {}
        for kind in ("continuum", "discrete"):
            pr = [(x, y) for x, y in paired(A, B) if x["kind"] == kind]
            d = [math.log(x["w1"]["flow"] / y["w1"]["flow"]) for x, y in pr]
            pool[kind] += d
            s[kind] = {"n": len(pr), "new": gm([x["w1"]["flow"] for x, _ in pr]),
                       "old": gm([y["w1"]["flow"] for _, y in pr]), "maxent": gm([x["w1"]["maxent"] for x, _ in pr]),
                       "new_over_old": ratio_ci(d),
                       "new_below_maxent": sum(x["w1"]["flow"] < x["w1"]["maxent"] for x, _ in pr),
                       "old_below_maxent": sum(y["w1"]["flow"] < y["w1"]["maxent"] for _, y in pr)}
        res["as_trained"][f"beta {b}, noise {n}"] = s
    res["as_trained"]["pooled new over old"] = {k: ratio_ci(v) for k, v in pool.items()}

    # 2. the posterior report as stated
    rows = {g: [] for g in GROUPS}                    # (new row, old row) by group
    per_setting, dl_all = [], []
    for b, n in SETTINGS:
        fn = f"floor_option_b{float(b):g}_n{float(n):g}.json"
        if not ((dnew / fn).exists() and (dold / fn).exists()):
            continue
        A, B = json.load(open(dnew / fn))["rows"], json.load(open(dold / fn))["rows"]
        pr = paired(A, B)
        dis = [(x, y) for x, y in pr if x["kind"] == "discrete"]
        lat = [(x, y) for x, y in pr if x["kind"] == "continuum"]
        for x, y in dis:
            g = ("lines, flagged by both" if x["fired"] and y["fired"] else
                 "lines, flagged by neither" if not (x["fired"] or y["fired"]) else "lines, flagged by one")
            rows[g].append((x, y))
        rows["lattice"] += lat
        dl = [math.log(x["auto"]["w1"] / y["auto"]["w1"]) for x, y in dis]
        dl_all += dl
        per_setting.append({
            "beta": float(b), "noise": float(n), "lines": len(dis),
            "w1_lines": {"new": gm([x["auto"]["w1"] for x, _ in dis]), "old": gm([y["auto"]["w1"] for _, y in dis]),
                         "maxent": gm([x["maxent"]["w1"] for x, _ in dis])},
            "new_over_old_lines": ratio_ci(dl),
            "new_over_maxent_lines": ratio_ci([math.log(x["auto"]["w1"] / x["maxent"]["w1"]) for x, _ in dis]),
            "old_over_maxent_lines": ratio_ci([math.log(y["auto"]["w1"] / y["maxent"]["w1"]) for _, y in dis]),
            "new_over_old_lattice": ratio_ci([math.log(x["auto"]["w1"] / y["auto"]["w1"]) for x, y in lat]),
            "flagged_lines": {"new": sum(x["fired"] for x, _ in dis), "old": sum(y["fired"] for _, y in dis)},
            "flagged_lattice": {"new": sum(x["fired"] for x, _ in lat), "old": sum(y["fired"] for _, y in lat)},
            "per_spectrum_new_over_old_lines": [float(np.exp(v)) for v in dl]})
    res["report"]["settings"] = per_setting
    res["report"]["pooled_new_over_old_lines"] = ratio_ci(dl_all)
    cov = {}
    for g, pr in rows.items():
        if not pr:
            continue
        cov[g] = {"spectra": len(pr),
                  "w1_new_over_old": ratio_ci([math.log(x["auto"]["w1"] / y["auto"]["w1"]) for x, y in pr])}
        for name, k in (("new", 0), ("old", 1)):
            sel = [p[k] for p in pr]
            cov[g][name] = {w: F.coverage(sel, "auto", w, skip_edge=False) for w in ("plain", "reported")}
    every = [p for g in GROUPS[:3] for p in rows[g]]
    cov["all line spectra"] = {"spectra": len(every), **{
        name: {w: F.coverage([p[k] for p in every], "auto", w, skip_edge=False) for w in ("plain", "reported")}
        for name, k in (("new", 0), ("old", 1))}}
    res["report"]["flagged"] = {
        "lines": {"new": sum(p[0]["fired"] for p in every), "old": sum(p[1]["fired"] for p in every), "of": len(every)},
        "lattice": {"new": sum(p[0]["fired"] for p in rows["lattice"]),
                    "old": sum(p[1]["fired"] for p in rows["lattice"]), "of": len(rows["lattice"])},
        "lines_new_only": sum(p[0]["fired"] and not p[1]["fired"] for p in every),
        "lines_old_only": sum(p[1]["fired"] and not p[0]["fired"] for p in every)}
    # every line spectrum the model's own report does not flag: the case with no warning
    for name, k in (("new", 0), ("old", 1)):
        sel = [p[k] for g in GROUPS[:3] for p in rows[g] if not p[k]["fired"]]
        cov[f"lines not flagged by {name}"] = {"spectra": len(sel), **{
            w: F.coverage(sel, "auto", w, skip_edge=False) for w in ("plain", "reported")}}
    res["report"]["coverage"] = cov
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(18, 5.0), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.0, 1.1, 1.3]})
    name = lambda s: f"beta {s['beta']:g}\nnoise {s['noise']:g}"
    for i, s in enumerate(per_setting):
        for k, col, dx, mk in (("old", OLD, -0.2, "o"), ("new", NEW, 0.0, "o"), ("maxent", ME, 0.2, "D")):
            a0.plot([i + dx], [s["w1_lines"][k]], mk, color=col, ms=9, mec="white", mew=0.8)
        r = np.array(s["per_spectrum_new_over_old_lines"])
        a1.plot(i + np.linspace(-0.3, 0.3, len(r)), r, "o", color=NEW, ms=4.5, mec="white", mew=0.4)
        m, lo, hi = s["new_over_old_lines"]
        a1.fill_between([i - 0.36, i + 0.36], [lo, lo], [hi, hi], color="#c9c8c3", alpha=0.6, lw=0)
        a1.plot([i - 0.36, i + 0.36], [m, m], color=INK, lw=1.8)
    a0.set_yscale("log")
    a0.set_xticks(range(len(per_setting)), [name(s) for s in per_setting], fontsize=8.5)
    a0.set_ylabel(r"W$_1$ to the exact answer (geometric mean)")
    for lab, col, mk in ((f"{old}: lines taught as broadened lines", OLD, "o"),
                         (f"{new}: exact line spectra in the label bank", NEW, "o"), ("MaxEnt", ME, "D")):
        a0.plot([], [], mk, color=col, ms=8, label=lab)
    a0.legend(fontsize=8, frameon=False)
    a0.set_title("exact line spectra through the report: error", fontsize=9.5)
    a1.axhline(1.0, color=INK, lw=0.8, ls=":")
    a1.set_xticks(range(len(per_setting)), [name(s) for s in per_setting], fontsize=8.5)
    a1.set_ylabel(r"W$_1$ new / W$_1$ old, per line spectrum")
    p = res["report"]["pooled_new_over_old_lines"]
    a1.set_title(f"the same spectra, one point each (below 1 = the retrain is better)\n"
                 f"all settings: {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]; bar = geometric mean, 95% interval",
                 fontsize=9.5)
    shown = [g for g in ("all line spectra", "lines, flagged by one", "lines, flagged by both",
                         "lines, flagged by neither", "lattice") if g in cov]
    label = {"all line spectra": "all line\nspectra", "lines, flagged by one": "lines flagged by\none model only",
             "lines, flagged by both": "lines flagged\nby both", "lines, flagged by neither": "lines flagged\nby neither",
             "lattice": "lattice\nspectra"}
    for j, g in enumerate(shown):
        for k, (nm, col) in enumerate((("old", OLD), ("new", NEW))):
            c = cov[g][nm]["reported"]
            for key, se, dx, fill in (("c50", "se50", -0.2, True), ("c90", "se90", 0.2, False)):
                a2.errorbar([j + (k - 0.5) * 0.14 + dx], [c[key]], yerr=[1.96 * c[se]], fmt="o", color=col, ms=7,
                            mfc=col if fill else "white", capsize=3, lw=1.2)
    for lv in (0.5, 0.9):
        a2.axhline(lv, color=INK, lw=0.8, ls="--")
    a2.set_ylim(0, 1.02)
    a2.set_xticks(range(len(shown)), [f"{label[g]}\n({cov[g]['spectra']})" for g in shown], fontsize=8.5)
    a2.set_ylabel("share of windows with the exact weight inside")
    a2.plot([], [], "o", color=OLD, label=old)
    a2.plot([], [], "o", color=NEW, label=new)
    a2.plot([], [], "o", color=INK, label="stated 50% interval")
    a2.plot([], [], "o", color=INK, mfc="white", label="stated 90% interval")
    a2.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    fl = res["report"]["flagged"]
    a2.set_title("exact window weights inside the intervals the report states\n"
                 f"(dashes: nominal; {new} has no band table yet)", fontsize=9.5)
    for ax in (a0, a1, a2):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle(f"Exact line spectra as training targets ({new}) against broadened lines only ({old}): "
               f"the posterior report on known answers, same noise draws\nnarrow lines flagged on "
               f"{fl['lines']['old']} -> {fl['lines']['new']} of {fl['lines']['of']} line spectra and on "
               f"{fl['lattice']['old']} -> {fl['lattice']['new']} of {fl['lattice']['of']} lattice spectra",
               fontsize=10.5)
    f.savefig(out.with_suffix(".png"), dpi=150)

    fmt = lambda t: "{:.3f} [{:.3f}, {:.3f}]".format(*t)
    print("as trained, new/old:", {k: fmt(v) for k, v in res["as_trained"]["pooled new over old"].items()})
    for s in per_setting:
        print(f"report  beta {s['beta']:g} noise {s['noise']:g}: lines old {s['w1_lines']['old']:.3f} new "
              f"{s['w1_lines']['new']:.3f} MaxEnt {s['w1_lines']['maxent']:.3f} | new/old {fmt(s['new_over_old_lines'])} "
              f"| new/MaxEnt {fmt(s['new_over_maxent_lines'])} | flagged {s['flagged_lines']} | lattice new/old "
              f"{fmt(s['new_over_old_lattice'])}")
    print("report, lines pooled new/old:", fmt(res["report"]["pooled_new_over_old_lines"]))
    print("flagged:", res["report"]["flagged"])
    for g, c in cov.items():
        for nm in ("new", "old"):
            for w in ("plain", "reported"):
                v = (c.get(nm) or {}).get(w) if nm in c else (c.get(w) if nm == "new" else None)
                if v:
                    print(f"coverage {g:28s} {nm if nm in c else '':3s} {w:8s} {v['c50']:.3f}±{v['se50']:.3f} / "
                          f"{v['c90']:.3f}±{v['se90']:.3f}  ({c['spectra']} spectra)")
    print(f"-> {out}.json + .png")


if __name__ == "__main__":
    main()
