"""Known answers at the edges of the trained range, scored (`run_edges.sh`).

For every setting found in OUT (one floor_option_b*_n*[_dt*].json each, the
posterior report on 20 exact lattice spectra and 24 exact line spectra):

  - W1 of the posterior mean over W1 of MaxEnt on the same data, per spectrum
    (geometric mean, 95% t interval), and on how many spectra the report is
    the lower;
  - how often the exact weight of a frequency window lies inside the 50% /
    90% interval the report states: lattice spectra, line spectra the report
    flags as narrow, line spectra it does not;
  - whether the exact weight lies inside the bracket the data alone allow;
  - how many spectra had fewer than half of the samples accepted.

    python scripts/step3/edges_summary.py out/step3/edges_F3L_60k_s1

Writes OUT/edges.{json,png}.
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

LAT, LIN, UNF, INK = "#2a78d6", "#0b7a75", "#b3261e", "#0b0b0b"


def ratio_ci(d):
    from scipy import stats
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return [float(np.exp(d.mean())), float(np.exp(d.mean() - h)), float(np.exp(d.mean() + h))]


def gm(v):
    return float(np.exp(np.mean(np.log(v))))


def main():
    out = Path(sys.argv[1])
    runs = sorted((json.load(open(p)) for p in out.glob("floor_option_b*_n*.json")),
                  key=lambda r: (r["beta"], -r["noise"], r.get("dtau", 0.125)))
    if not runs:
        raise SystemExit(f"no floor_option_b*_n*.json in {out}")
    res = {"flow": runs[0]["flow"], "settings": []}
    for r in runs:
        s = {"beta": r["beta"], "noise": r["noise"], "dtau": r.get("dtau", 0.125),
             "n_tau": int(round(r["beta"] / r.get("dtau", 0.125))) + 1}
        for kind, name in (("continuum", "lattice"), ("discrete", "lines")):
            sel = [x for x in r["rows"] if x["kind"] == kind]
            d = [math.log(x["auto"]["w1"] / x["maxent"]["w1"]) for x in sel]
            s[name] = {"n": len(sel), "w1": gm([x["auto"]["w1"] for x in sel]),
                       "w1_maxent": gm([x["maxent"]["w1"] for x in sel]), "over_maxent": ratio_ci(d),
                       "lower_than_maxent": int(sum(v < 0 for v in d)),
                       "flagged": int(sum(x["fired"] for x in sel)),
                       "accepted_below_half": int(sum(x["auto"]["accepted"] < 0.5 for x in sel)),
                       "accepted_min": float(min(x["auto"]["accepted"] for x in sel))}
        groups = (("lattice", [x for x in r["rows"] if x["kind"] == "continuum"]),
                  ("lines flagged", [x for x in r["rows"] if x["kind"] == "discrete" and x["fired"]]),
                  ("lines not flagged", [x for x in r["rows"] if x["kind"] == "discrete" and not x["fired"]]))
        s["stated_intervals"] = {g: F.coverage(rows, "auto", "reported", skip_edge=False) for g, rows in groups if rows}
        s["bracket"] = {k: F.bracket_holds([x for x in r["rows"] if x["kind"] == kind])
                        for k, kind in (("lattice", "continuum"), ("lines", "discrete"))}
        res["settings"].append(s)
    with open(out / "edges.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    S = res["settings"]
    name = lambda s: f"beta {s['beta']:g}\nerr {s['noise']:g}\ndtau {s['dtau']:g}\n({s['n_tau']} pts)"
    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(18.5, 5.2), constrained_layout=True)
    for i, s in enumerate(S):
        for key, col, dx in (("lattice", LAT, -0.12), ("lines", LIN, 0.12)):
            m, lo, hi = s[key]["over_maxent"]
            a0.errorbar([i + dx], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=7, capsize=3, lw=1.3)
            a0.annotate(f"{s[key]['lower_than_maxent']}/{s[key]['n']}", (i + dx, hi), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=7, color=col)
        for ax, groups in ((a1, (("lattice", LAT, 0.0),)),
                           (a2, (("lines flagged", LIN, -0.1), ("lines not flagged", UNF, 0.1)))):
            for g, col, dx in groups:
                c = s["stated_intervals"].get(g)
                if not c:
                    continue
                ax.errorbar([i + dx - 0.07], [c["c50"]], yerr=[1.96 * c["se50"]], fmt="o", color=col, ms=6.5,
                            capsize=3, lw=1.2)
                ax.errorbar([i + dx + 0.07], [c["c90"]], yerr=[1.96 * c["se90"]], fmt="o", color=col, ms=6.5,
                            mfc="white", capsize=3, lw=1.2)
                ax.annotate(str(c["truths"]), (i + dx, 0.03), ha="center", fontsize=7, color=col)
    a0.axhline(1.0, color=INK, lw=0.8, ls=":")
    a0.set_yscale("log")
    a0.set_ylabel(r"W$_1$ of the report / W$_1$ of MaxEnt (geometric mean, 95% interval)")
    a0.plot([], [], "o", color=LAT, label="exact lattice spectra")
    a0.plot([], [], "o", color=LIN, label="exact line spectra")
    a0.legend(fontsize=8.5, frameon=False)
    a0.set_title("error against MaxEnt on the same data (below 1 = better)\nlabels: spectra where the report is the lower",
                 fontsize=9.5)
    for ax, ttl in ((a1, "lattice spectra"), (a2, "line spectra: flagged as narrow (green), not flagged (red)")):
        for lv in (0.5, 0.9):
            ax.axhline(lv, color=INK, lw=0.8, ls="--")
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("share of windows with the exact weight inside")
        ax.plot([], [], "o", color=INK, label="stated 50% interval")
        ax.plot([], [], "o", color=INK, mfc="white", label="stated 90% interval")
        ax.legend(fontsize=8.5, frameon=False, loc="lower right")
        ax.set_title(f"{ttl}\nexact window weights inside the stated intervals (dashes: nominal; bottom: spectra)",
                     fontsize=9.5)
    for ax in (a0, a1, a2):
        ax.set_xticks(range(len(S)), [name(s) for s in S], fontsize=8)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle(f"Known answers at the edges of the trained range ({Path(res['flow']).stem}): the posterior report "
               "on 20 exact lattice and 24 exact line spectra per setting", fontsize=10.5)
    f.savefig(out / "edges.png", dpi=150)

    fmt = lambda t: "{:.3f} [{:.3f}, {:.3f}]".format(*t)
    for s in S:
        print(f"beta {s['beta']:g} noise {s['noise']:g} dtau {s['dtau']:g} ({s['n_tau']} points)")
        for key in ("lattice", "lines"):
            v = s[key]
            print(f"   {key:8s} W1 {v['w1']:.3f} MaxEnt {v['w1_maxent']:.3f}  ratio {fmt(v['over_maxent'])}  lower on "
                  f"{v['lower_than_maxent']}/{v['n']}  flagged {v['flagged']}  accepted<50% on {v['accepted_below_half']} "
                  f"(min {v['accepted_min']:.0%})  bracket {s['bracket'][key]}")
        for g, c in s["stated_intervals"].items():
            print(f"   stated intervals, {g:18s} {c['c50']:.3f}±{c['se50']:.3f} / {c['c90']:.3f}±{c['se90']:.3f}  "
                  f"({c['truths']} spectra)")
    print(f"-> {out}/edges.{{json,png}}")


if __name__ == "__main__":
    main()
