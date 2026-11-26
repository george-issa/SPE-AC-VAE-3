"""Training seeds through the posterior report: do they give the same report?

Each run is the floor-option judge's output for one checkpoint
(`floor_option_judge.py`: the report on 20 exact lattice and 24 exact line
spectra per setting, same spectra and noise draws for every checkpoint).
Compared per run:

  - W1 of the posterior mean per setting, lattice and line spectra;
  - how many line spectra the report flags as narrow, and on how many the
    runs agree;
  - how often the exact weight of a frequency window lies inside the 50% /
    90% interval the report states: lattice spectra, line spectra the report
    flags, line spectra it does not;
  - per spectrum, W1 of every run over W1 of the first (paired).

    python scripts/step3/seed_report_summary.py OUT NAME=DIR NAME=DIR [...]

Compare runs that all carry a band table or all do not: the stated intervals
of unflagged spectra come from the table.  Writes OUT.{json,png}.
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

SETTINGS = ((8, 0.0001), (8, 0.001), (8, 0.01), (2, 0.001), (16, 0.001))
COLS = ("#2a78d6", "#0b7a75", "#eb6834", "#7a3fb0")
ME, INK = "#7d7c78", "#0b0b0b"
GROUPS = ("lattice", "lines flagged", "lines not flagged", "all line spectra")


def ratio_ci(d):
    from scipy import stats
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return [float(np.exp(d.mean())), float(np.exp(d.mean() - h)), float(np.exp(d.mean() + h))]


def gm(v):
    return float(np.exp(np.mean(np.log(v))))


def main():
    out = Path(sys.argv[1])
    runs = [a.split("=", 1) for a in sys.argv[2:]]
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {}                                           # name -> setting -> rows
    for name, d in runs:
        data[name] = {}
        for b, n in SETTINGS:
            p = Path(d) / f"floor_option_b{b:g}_n{n:g}.json"
            if p.exists():
                data[name][(b, n)] = json.load(open(p))["rows"]
    names = [n for n, _ in runs]
    shared = [s for s in SETTINGS if all(s in data[n] for n in names)]
    for s in shared:                                    # the same spectra in the same order
        for n in names[1:]:
            assert [(x["family"], x["params"]) for x in data[n][s]] == \
                   [(x["family"], x["params"]) for x in data[names[0]][s]]
    res = {"runs": dict(runs), "settings": [f"beta {b:g}, noise {n:g}" for b, n in shared], "per_run": {},
           "paired_over_first": {}}
    maxent = {}
    for s in shared:
        for kind in ("continuum", "discrete"):
            v = next(([x["maxent"]["w1"] for x in data[n][s] if x["kind"] == kind] for n in names
                      if all(x.get("maxent") for x in data[n][s] if x["kind"] == kind)), None)
            maxent[(s, kind)] = gm(v) if v else None
    for name in names:
        every = [x for s in shared for x in data[name][s]]
        lat = [x for x in every if x["kind"] == "continuum"]
        lin = [x for x in every if x["kind"] == "discrete"]
        sets = {"lattice": lat, "lines flagged": [x for x in lin if x["fired"]],
                "lines not flagged": [x for x in lin if not x["fired"]], "all line spectra": lin}
        res["per_run"][name] = {
            "w1": {f"beta {b:g}, noise {n:g}": {
                kind: gm([x["auto"]["w1"] for x in data[name][(b, n)] if x["kind"] == kind])
                for kind in ("continuum", "discrete")} for b, n in shared},
            "flagged": {"lines": int(sum(x["fired"] for x in lin)), "of_lines": len(lin),
                        "lattice": int(sum(x["fired"] for x in lat)), "of_lattice": len(lat)},
            "accepted_below_half": int(sum(x["auto"]["accepted"] < 0.5 for x in every)),
            "stated_intervals": {g: F.coverage(rows, "auto", "reported", skip_edge=False)
                                 for g, rows in sets.items() if rows},
            "central_quantiles": {g: F.coverage(rows, "auto", "plain", skip_edge=False)
                                  for g, rows in sets.items() if rows}}
    first = names[0]
    per_spectrum = {}
    for name in names[1:]:
        r = {}
        for kind in ("continuum", "discrete"):
            d = [math.log(x["auto"]["w1"] / y["auto"]["w1"]) for s in shared
                 for x, y in zip(data[name][s], data[first][s]) if x["kind"] == kind]
            r[kind] = {"ratio": ratio_ci(d), "largest_factor": float(np.exp(np.abs(d).max())),
                       "median_factor": float(np.exp(np.median(np.abs(d))))}
            per_spectrum[(name, kind)] = np.exp(d)
        pairs = [(x["fired"], y["fired"]) for s in shared for x, y in zip(data[name][s], data[first][s])
                 if x["kind"] == "discrete"]
        r["flags_on_lines"] = {"both": sum(a and b for a, b in pairs), "neither": sum(not (a or b) for a, b in pairs),
                               f"only {name}": sum(a and not b for a, b in pairs),
                               f"only {first}": sum(b and not a for a, b in pairs)}
        res["paired_over_first"][name] = r
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(18.5, 5.2), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.1, 1.0, 1.3]})
    x = np.arange(len(shared))
    for k, name in enumerate(names):
        dx = (k - (len(names) - 1) / 2) * 0.13
        for kind, mk in (("continuum", "o"), ("discrete", "s")):
            a0.plot(x + dx, [res["per_run"][name]["w1"][s][kind] for s in res["settings"]], mk, color=COLS[k],
                    ms=8, mec="white", mew=0.8)
        a0.plot([], [], "o", color=COLS[k], label=name)
    for kind, mk in (("continuum", "o"), ("discrete", "s")):
        v = [maxent[(s, kind)] for s in shared]
        if all(v):
            a0.plot(x + 0.3, v, mk, color=ME, ms=7, mfc="white")
    a0.plot([], [], "o", color=ME, mfc="white", label="MaxEnt")
    a0.plot([], [], "o", color=INK, label="circles: lattice spectra")
    a0.plot([], [], "s", color=INK, label="squares: line spectra")
    a0.set_yscale("log")
    a0.set_ylabel(r"W$_1$ to the exact answer (geometric mean)")
    a0.set_xticks(x, [f"beta {b:g}\nerr {n:g}" for b, n in shared], fontsize=8.5)
    a0.legend(fontsize=8, frameon=False, ncol=2)
    a0.set_title("error through the report, per setting", fontsize=9.5)
    for k, name in enumerate(names[1:], start=1):
        for j, (kind, lab) in enumerate((("continuum", "lattice spectra"), ("discrete", "line spectra"))):
            r = per_spectrum[(name, kind)]
            pos = j + (k - 1) * 0.8 / max(len(names) - 1, 1) - 0.4 + 0.4 / max(len(names) - 1, 1)
            a1.plot(pos + np.linspace(-0.3, 0.3, len(r)), r, "o", color=COLS[k], ms=3.8, mec="white", mew=0.3)
            m, lo, hi = res["paired_over_first"][name][kind]["ratio"]
            a1.fill_between([pos - 0.34, pos + 0.34], [lo, lo], [hi, hi], color="#c9c8c3", alpha=0.6, lw=0)
            a1.plot([pos - 0.34, pos + 0.34], [m, m], color=INK, lw=1.8)
    a1.axhline(1.0, color=INK, lw=0.8, ls=":")
    tick = lambda kind: "\n".join("{}: {:.3f} [{:.3f}, {:.3f}]".format(n, *res["paired_over_first"][n][kind]["ratio"])
                                  for n in names[1:])
    a1.set_xticks([0, 1], [f"lattice spectra\n{tick('continuum')}", f"line spectra\n{tick('discrete')}"], fontsize=8.5)
    a1.set_ylabel(rf"W$_1$ of the other seed / W$_1$ of {first}, per spectrum")
    a1.set_title("the same spectrum and noise draw under two trainings\n(one point each; bar = geometric mean, "
                 "95% interval)", fontsize=9.5)
    shown = [g for g in GROUPS if all(g in res["per_run"][n]["stated_intervals"] for n in names)]
    for j, g in enumerate(shown):
        for k, name in enumerate(names):
            c = res["per_run"][name]["stated_intervals"][g]
            off = (k - (len(names) - 1) / 2) * 0.12
            a2.errorbar([j + off - 0.2], [c["c50"]], yerr=[1.96 * c["se50"]], fmt="o", color=COLS[k], ms=7,
                        capsize=3, lw=1.2)
            a2.errorbar([j + off + 0.2], [c["c90"]], yerr=[1.96 * c["se90"]], fmt="o", color=COLS[k], ms=7,
                        mfc="white", capsize=3, lw=1.2)
    for lv in (0.5, 0.9):
        a2.axhline(lv, color=INK, lw=0.8, ls="--")
    a2.set_ylim(0, 1.03)
    counts = lambda g: " / ".join(str(res["per_run"][n]["stated_intervals"][g]["truths"]) for n in names)
    a2.set_xticks(range(len(shown)), [f"{g}\n({counts(g)} spectra)" for g in shown], fontsize=8.5)
    a2.set_ylabel("share of windows with the exact weight inside")
    for k, name in enumerate(names):
        a2.plot([], [], "o", color=COLS[k], label=name)
    a2.plot([], [], "o", color=INK, label="stated 50% interval")
    a2.plot([], [], "o", color=INK, mfc="white", label="stated 90% interval")
    a2.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    a2.set_title("exact window weights inside the intervals the report states\n(dashes: nominal)", fontsize=9.5)
    for ax in (a0, a1, a2):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fl = "; ".join(f"{n}: {res['per_run'][n]['flagged']['lines']} of {res['per_run'][n]['flagged']['of_lines']} line, "
                   f"{res['per_run'][n]['flagged']['lattice']} of {res['per_run'][n]['flagged']['of_lattice']} lattice"
                   for n in names)
    f.suptitle("Training seeds through the posterior report, on the same known answers and noise draws\n"
               f"spectra flagged as narrow lines - {fl}", fontsize=10.5)
    f.savefig(out.with_suffix(".png"), dpi=150)

    fmt = lambda t: "{:.3f} [{:.3f}, {:.3f}]".format(*t)
    for s in res["settings"]:
        print(f"{s:22s} lattice " + " / ".join(f"{res['per_run'][n]['w1'][s]['continuum']:.3f}" for n in names)
              + "   lines " + " / ".join(f"{res['per_run'][n]['w1'][s]['discrete']:.3f}" for n in names))
    for n in names:
        r = res["per_run"][n]
        print(f"{n}: flagged {r['flagged']}, accepted<50% on {r['accepted_below_half']}")
        for g, c in r["stated_intervals"].items():
            q = r["central_quantiles"][g]
            print(f"   stated {g:18s} {c['c50']:.3f}±{c['se50']:.3f} / {c['c90']:.3f}±{c['se90']:.3f}   central "
                  f"quantiles {q['c50']:.3f} / {q['c90']:.3f}   ({c['truths']} spectra)")
    for n, r in res["paired_over_first"].items():
        print(f"{n} over {first}: lattice {fmt(r['continuum']['ratio'])} (median factor "
              f"{r['continuum']['median_factor']:.3f}, largest {r['continuum']['largest_factor']:.2f}); lines "
              f"{fmt(r['discrete']['ratio'])} (median factor {r['discrete']['median_factor']:.3f}, largest "
              f"{r['discrete']['largest_factor']:.2f}); flags on lines {r['flags_on_lines']}")
    print(f"-> {out}.json + .png")


if __name__ == "__main__":
    main()
