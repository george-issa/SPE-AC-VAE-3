"""Narrow lines in the posterior report, judged on known answers.

When the accepted samples put a tenth or more of their weight on poles at the
trained width floor, `continue_posterior` (a) fits the same proposals again
with the floor divided by 4 (`floor_factor="auto"`, its default), and
(b) states window intervals that span both readings of the narrow poles, as
fitted and as sharp lines, without the band table.  This runs the report
that way and with the floor as trained (same data, same seed) on the exactly
solvable models of the known-answer judge, plus further draws of the exactly
diagonalised (line) families, and scores

  - W1 of the posterior mean to the exact cumulative weight, next to MaxEnt
    on the same data;
  - how often the exact weight of a frequency window lies inside the
    intervals the report states, and inside the central 50% / 90% of the
    samples as fitted (by the randomised rank of the exact weight among the
    samples -- the statistic of `calibration.py`, unbiased at any sample
    count), also split by how far the nearest exact line is from a window
    edge;
  - whether the exact weight lies inside the bracket the data alone allow.

    python scripts/step3/floor_option_judge.py OUT FLOW BETA NOISE [extra=2]
    python scripts/step3/floor_option_judge.py fig OUT

The first form writes OUT/floor_option_b<beta>_n<noise>.json for one setting;
`fig` pools every setting found in OUT into floor_option.{json,png}.
"""
import json
import math
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/eval")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import exact_models_judge as J

S = 64
EDGE = 0.1            # a line this close to a window edge makes that window's exact weight a coin toss
STD, AUTO, ME, INK = "#9ec5f4", "#2a78d6", J.ME, J.INK


def extra_discrete(beta, per_family):
    """Further draws of the exactly diagonalised families (not in the known-answer judge)."""
    from spevae.data import exact_models as X
    from spevae.evaluation.cases import Truth
    from spevae.physics import SpectralQuadrature
    quad = SpectralQuadrature.for_beta(beta)
    x = torch.linspace(0.0, 1.0, int(round(beta / J.DTAU)) + 1)
    rng = torch.Generator().manual_seed(2027)
    out = []
    for fam in X.EXACT_DISCRETE:
        for _ in range(per_family):
            sp = X.sample_discrete_spectrum(fam, beta, rng)
            Gc = torch.tensor(sp.greens(x.double().numpy()), dtype=torch.float32).unsqueeze(0)
            out.append({"family": fam, "kind": "discrete", "params": sp.params, "spectrum": sp,
                        "cdf": sp.cdf(J.WG), "extra": True,
                        "truth": Truth(None, beta, x, quad, torch.zeros(1, quad.nodes.numel()), Gc, None)})
    return out


def true_windows(item, windows):
    """Exact weight per window, and whether a line sits within EDGE of one of its edges."""
    out = []
    if item["kind"] == "discrete":
        sp = item["spectrum"]
        e, w = sp.energies, sp.weights / sp.weights.sum()
        for lo, hi in windows:
            near = any(bool((w[np.abs(e - b) < EDGE]).sum() > 2e-3) for b in (lo, hi) if math.isfinite(b))
            out.append((float(w[(e >= lo) & (e < hi)].sum()), near))
    else:
        c = item["cdf"] / item["cdf"][-1]
        at = lambda b: 0.0 if b == -math.inf else 1.0 if b == math.inf else float(np.interp(b, J.WG, c))
        out = [(at(hi) - at(lo), False) for lo, hi in windows]
    return out


def sample_windows(poles, windows, lines_below=None):
    """Window weights of every accepted sample, (samples, windows); poles in physical units.
    With ``lines_below``, poles at most that wide are read as sharp lines (all their weight at
    their position) instead of Lorentzians."""
    eps, gam, a, b = (getattr(poles, k).double() for k in ("eps", "gamma", "a", "b"))
    line = (gam <= lines_below) if lines_below is not None else torch.zeros_like(gam, dtype=torch.bool)

    def cdf(w):
        if math.isinf(w):
            return torch.zeros(eps.shape[0], dtype=torch.float64) if w < 0 else a.sum(-1)
        u = w - eps
        absorptive = torch.where(line, (u > 0).double(), torch.atan(u / gam) / math.pi + 0.5)
        return (a * absorptive - b * torch.log(u * u + gam * gam) / (2.0 * math.pi)).sum(-1)
    tot = a.sum(-1)
    return torch.stack([(cdf(hi) - cdf(lo)) / tot for lo, hi in windows], -1).numpy()


def score(rep, item, windows, truth_w, gmin_trained):
    cdf = np.cumsum(rep.band["mean"]) * J.DW
    ww = sample_windows(rep.poles, windows)
    # the same samples with every pole at or below 1.25 trained floors read as a sharp line
    wl = sample_windows(rep.poles, windows, lines_below=1.25 * gmin_trained)
    lev = rep.diagnostics["band_levels"]
    d = rep.diagnostics
    out = {"w1": J.w1_cdf(cdf, item["cdf"]), "accepted": d["accepted_fraction"],
           "chi2_median": float(np.median(rep.chi2)), "floor_factor": d["floor_factor"],
           "weight_at_width_floor": d["weight_at_width_floor"],
           "weight_below_trained_width_floor": d.get("weight_below_trained_width_floor", 0.0),
           "n_poles_median": d["n_poles_median"], "w": np.round(ww, 6).tolist(),
           "w_lines": np.round(wl, 6).tolist(),
           "levels": {"0.9": lev[0.9], "0.5": lev[0.5]},
           "reported": [[w["q05"], w["q25"], w["q75"], w["q95"]] for w in rep.windows],
           "line_shape_readings": bool(d.get("line_shape_readings")),
           "band_table_withheld": bool(d.get("band_table_withheld")),
           "n_warnings": len(rep.warnings), "truth": [t for t, _ in truth_w]}
    if rep.windows and "lo" in rep.windows[0] and rep.windows[0]["lo"] == rep.windows[0]["lo"]:
        out["bracket"] = [[w["lo"], w["hi"]] for w in rep.windows]
    return out


def run(out, flow_path, beta, noise, extra):
    from spevae.evaluation import cases as C
    from spevae.evaluation import methods as M
    from spevae.models.flow import load_flow
    from spevae.posterior import DEFAULT_WINDOWS, continue_posterior
    flow = load_flow(flow_path).eval()
    items = J.make_truths(beta) + extra_discrete(beta, extra)
    gphys = flow.coords.gamma_min / beta               # trained width floor, physical units
    rows = []
    for k, item in enumerate(items):
        inst = C.draw(item["truth"], noise, J.SEED)
        tr = inst.truth
        truth_w = true_windows(item, DEFAULT_WINDOWS)
        t0 = time.time()
        call = lambda ff, bounds=False: continue_posterior(
            flow, inst.G, tr.x, beta, inst.sigma, inst.isr, n_samples=S, omega=J.WG, bounds=bounds,
            floor_factor=ff, generator=torch.Generator().manual_seed(9))
        rep = call("auto", True)
        t_auto = time.time() - t0
        fired = rep.diagnostics["floor_factor"] != 1.0
        row = {"family": item["family"], "kind": item["kind"], "params": item["params"],
               "extra": bool(item.get("extra")), "fired": fired, "near_edge": [n for _, n in truth_w],
               "auto": score(rep, item, DEFAULT_WINDOWS, truth_w, gphys), "seconds_auto": t_auto}
        row["standard"] = score(call(1.0), item, DEFAULT_WINDOWS, truth_w, gphys) if fired else row["auto"]
        me = M.make_method("maxent").synthetic(inst)          # lattice spectra too since 2026-11-25
        A_me = np.interp(J.WG, me.omega, me.A, left=0.0, right=0.0)
        row["maxent"] = {"w1": J.w1_cdf(np.cumsum(A_me) * J.DW, item["cdf"])}
        rows.append(row)
        print(f"{item['kind']:9s} {item['family']:16s} {'FIRED' if fired else '     '} "
              f"W1 as trained {row['standard']['w1']:.4f} auto {row['auto']['w1']:.4f}"
              + (f" MaxEnt {row['maxent']['w1']:.4f}" if "maxent" in row else "")
              + f" | accepted {row['standard']['accepted']:.0%} -> {row['auto']['accepted']:.0%}"
              f" | {t_auto:.0f}s", flush=True)
    res = {"flow": flow_path, "beta": beta, "noise": noise, "dtau": J.DTAU, "samples": S,
           "windows": [list(w) for w in DEFAULT_WINDOWS], "rows": rows}
    tag = "" if J.DTAU == 0.125 else f"_dt{J.DTAU:g}"          # JUDGE_DTAU, see exact_models_judge.py
    path = Path(out) / f"floor_option_b{beta:g}_n{noise:g}{tag}.json"
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"-> {path}")


def gmean(v):
    return float(np.exp(np.mean(np.log(v))))


def ratio_ci(a, b):
    """Geometric-mean ratio a/b with a paired 95% t interval on the log-ratios."""
    from scipy import stats
    d = np.log(np.asarray(a)) - np.log(np.asarray(b))
    if len(d) < 2 or float(d.std()) == 0.0:
        return float(np.exp(d.mean())), float(np.exp(d.mean())), float(np.exp(d.mean()))
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return float(np.exp(d.mean())), float(np.exp(d.mean() - h)), float(np.exp(d.mean() + h))


def coverage(rows, arm, which, skip_edge=True, seed=3):
    """Share of (spectrum, window) pairs whose exact weight is inside the central 50% / 90% of the
    samples (`plain`) or inside the band table's levels (`table`), by randomised rank as in
    calibration.py; the standard error treats each spectrum as one cluster."""
    rng = np.random.default_rng(seed)
    per = []
    for r in rows:
        s = r[arm]
        w, t = np.asarray(s["w"]), np.asarray(s["truth"])
        ok = ~np.asarray(r["near_edge"]) if skip_edge else np.ones(len(t), bool)
        if not ok.any():
            continue
        if which == "reported":                      # the intervals as the report states them
            q = np.asarray(s["reported"])
            per.append((((q[:, 1] <= t) & (t <= q[:, 2]))[ok].sum(), ((q[:, 0] <= t) & (t <= q[:, 3]))[ok].sum(),
                        ok.sum()))
            continue
        u = ((w < t).sum(0) + rng.random(len(t))) / (len(w) + 1)
        l50, l90 = (0.5, 0.9) if which == "plain" else (s["levels"]["0.5"], s["levels"]["0.9"])
        dist = np.abs(2.0 * u - 1.0)[ok]
        per.append(((dist < l50).sum(), (dist < l90).sum(), ok.sum()))
    if not per:
        return None
    p = np.array(per, float)
    tot = p[:, 2].sum()
    c50, c90 = p[:, 0].sum() / tot, p[:, 1].sum() / tot
    m = len(p)
    se = lambda h, c: float(math.sqrt(m / max(m - 1, 1) * (((h - c * p[:, 2]) ** 2).sum())) / tot)
    return {"c50": float(c50), "c90": float(c90), "se50": se(p[:, 0], c50), "se90": se(p[:, 1], c90),
            "truths": m, "pairs": int(tot)}


EDGE_CLASSES = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, math.inf))


def by_edge_distance(runs, extra=2, seed=3):
    """Line spectra: coverage per window, grouped by the distance from the nearest exact line
    (weight above 0.002) to an edge of that window.  The spectra are regenerated from their seeds."""
    rng = np.random.default_rng(seed)
    acc = {}
    for r in runs:
        items = J.make_truths(r["beta"]) + extra_discrete(r["beta"], extra)
        if len(items) != len(r["rows"]):
            return None
        for row, it in zip(r["rows"], items):
            if row["kind"] != "discrete":
                continue
            sp = it["spectrum"]
            e = sp.energies[sp.weights / sp.weights.sum() > 2e-3]
            s = row["auto"]
            w, t, q = np.asarray(s["w"]), np.asarray(s["truth"]), np.asarray(s["reported"])
            u = ((w < t).sum(0) + rng.random(len(t))) / (len(w) + 1)
            need = np.abs(2.0 * u - 1.0)
            for k, (lo, hi) in enumerate(r["windows"]):
                d = min([abs(x - b) for x in e for b in (lo, hi) if math.isfinite(b)] or [math.inf])
                c = next(i for i, (a, b) in enumerate(EDGE_CLASSES) if a <= d < b)
                for key in ((c, "flagged" if row["fired"] else "not flagged"), (c, "all")):
                    acc.setdefault(key, []).append((need[k] < 0.5, need[k] < 0.9, q[k, 1] <= t[k] <= q[k, 2],
                                                    q[k, 0] <= t[k] <= q[k, 3], q[k, 3] - q[k, 0]))
    out = {}
    for (c, grp), v in sorted(acc.items()):
        v = np.array(v, float)
        a, b = EDGE_CLASSES[c]
        out[f"{grp} | nearest line {a:g}-{b:g} from an edge".replace("-inf", " or more")] = {
            "windows": len(v), "as_fitted": [float(v[:, 0].mean()), float(v[:, 1].mean())],
            "reported": [float(v[:, 2].mean()), float(v[:, 3].mean())], "width90_reported": float(np.median(v[:, 4]))}
    return out


def bracket_holds(rows, tol=1e-6):
    """Share of windows whose exact weight is inside the data-only bracket (auto arm)."""
    hit = n = 0
    for r in rows:
        for t_, b in zip(r["auto"]["truth"], r["auto"].get("bracket") or []):
            n += 1
            hit += b[0] - tol <= t_ <= b[1] + tol
    return {"inside": int(hit), "windows": int(n)} if n else None


def fig(out):
    out = Path(out)
    runs = sorted((json.load(open(p)) for p in out.glob("floor_option_b*_n*.json")),
                  key=lambda r: (r["beta"], -r["noise"]))
    if not runs:
        raise SystemExit(f"no floor_option_b*_n*.json in {out}")
    name = lambda r: f"beta {r['beta']:g}\nnoise {r['noise']:g}"
    summary = {"flow": runs[0]["flow"], "settings": []}
    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17.5, 5.0), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]})
    lines_all, lat_all = [], []
    for i, r in enumerate(runs):
        dis = [x for x in r["rows"] if x["kind"] == "discrete"]
        lat = [x for x in r["rows"] if x["kind"] == "continuum"]
        lines_all += dis
        lat_all += lat
        fired = [x for x in dis if x["fired"]]
        w = {k: [x[k]["w1"] for x in dis] for k in ("standard", "auto", "maxent")}
        s = {"beta": r["beta"], "noise": r["noise"], "lines": len(dis), "lines_fired": len(fired),
             "lattice": len(lat), "lattice_fired": sum(x["fired"] for x in lat),
             "w1_lines": {k: gmean(v) for k, v in w.items()},
             "auto_over_standard": ratio_ci(w["auto"], w["standard"]),
             "standard_over_maxent": ratio_ci(w["standard"], w["maxent"]),
             "auto_over_maxent": ratio_ci(w["auto"], w["maxent"]),
             "accepted_lines": {k: float(np.mean([x[k]["accepted"] for x in dis])) for k in ("standard", "auto")},
             "seconds_auto_fired": float(np.mean([x["seconds_auto"] for x in fired])) if fired else None,
             "seconds_not_fired": float(np.mean([x["seconds_auto"] for x in r["rows"] if not x["fired"]]))
             if any(not x["fired"] for x in r["rows"]) else None}
        if fired:
            s["auto_over_standard_fired"] = ratio_ci([x["auto"]["w1"] for x in fired],
                                                     [x["standard"]["w1"] for x in fired])
        s["coverage"] = {f"{grp} | {which}": coverage(rws, "auto", which, skip_edge=False)
                         for grp, rws in (("lines flagged", fired), ("lines not flagged", [x for x in dis if not x["fired"]]),
                                          ("lattice", lat))
                         for which in ("plain", "reported") if rws}
        summary["settings"].append(s)
        for k, col, dx, mk in (("standard", STD, -0.2, "o"), ("auto", AUTO, 0.0, "o"), ("maxent", ME, 0.2, "D")):
            a0.plot([i + dx], [s["w1_lines"][k]], mk, color=col, ms=9, mec="white", mew=0.8, zorder=3)
        rt = np.array([x["auto"]["w1"] / x["standard"]["w1"] for x in dis])
        fr = np.array([x["fired"] for x in dis])
        jit = np.linspace(-0.28, 0.28, len(dis))
        a1.plot(i + jit[fr], rt[fr], "o", color=AUTO, ms=5.5, alpha=0.85, mec="white", mew=0.5)
        a1.plot(i + jit[~fr], rt[~fr], "o", color="#c9c8c3", ms=4.5)
        g, lo, hi = s["auto_over_standard"]
        a1.plot([i - 0.36, i + 0.36], [g, g], color=INK, lw=1.6)
        a1.fill_between([i - 0.36, i + 0.36], [lo, lo], [hi, hi], color=INK, alpha=0.12, lw=0)
    for ax in (a0, a1):
        ax.set_xticks(range(len(runs)), [name(r) for r in runs], fontsize=8.5)
    a0.set_yscale("log")
    a0.set_ylabel(r"W$_1$ to the exact answer (geometric mean)")
    for lab, col, mk in (("flow, floor as trained", STD, "o"), ("flow, floor lowered where lines are flagged", AUTO, "o"),
                         ("MaxEnt", ME, "D")):
        a0.plot([], [], mk, color=col, ms=8, label=lab)
    a0.legend(fontsize=8, frameon=False, loc="lower left")
    a0.set_title("exactly diagonalised (line) spectra: error", fontsize=10)
    a1.axhline(1.0, color=INK, lw=0.8, ls=":")
    a1.set_ylabel(r"W$_1$ with the option / W$_1$ as trained, per spectrum")
    a1.plot([], [], "o", color=AUTO, ms=6, label="narrow lines flagged, fitted again")
    a1.plot([], [], "o", color="#c9c8c3", ms=5, label="not flagged (same fit)")
    a1.plot([], [], color=INK, lw=1.6, label="geometric mean, 95% interval")
    a1.legend(fontsize=8, frameon=False)
    a1.set_title("the same spectra, one point each (below 1 = better)", fontsize=10)

    flagged = [x for x in lines_all if x["fired"]]
    groups = (("line spectra,\nnarrow lines flagged", flagged),
              ("line spectra,\nnot flagged", [x for x in lines_all if not x["fired"]]),
              ("lattice spectra\n(never flagged)" if not any(x["fired"] for x in lat_all) else "lattice spectra",
               lat_all))
    summary["coverage"] = {}
    for j, (lab, rws) in enumerate(groups):
        for which, dx, fill in (("plain", -0.17, "white"), ("reported", 0.17, None)):
            c = coverage(rws, "auto", which, skip_edge=False) if rws else None
            summary["coverage"][f"{lab.replace(chr(10), ' ')} | {which}"] = c
            if c is None:
                continue
            for lvl, col in (("90", AUTO), ("50", "#0b3d91")):
                a2.errorbar([j + dx], [c["c" + lvl]], yerr=[1.96 * c["se" + lvl]], fmt="o", color=col, ms=7,
                            mfc=fill or col, capsize=3, lw=1.2)
    # for the record: other scorings of the same samples
    summary["coverage_other"] = {
        f"{lab.replace(chr(10), ' ')} | {arm} | {which} | {'no line within %g of an edge' % EDGE if skip else 'every window'}":
            coverage(rws, arm, which, skip_edge=skip)
        for lab, rws in groups if rws
        for arm in (("standard", "auto") if rws is flagged else ("auto",))
        for which in ("plain", "table", "reported") for skip in (False, True)}
    summary["bracket"] = {"line spectra": bracket_holds(lines_all), "lattice spectra": bracket_holds(lat_all)}
    summary["by_edge_distance"] = by_edge_distance(runs)
    for lvl in (0.5, 0.9):
        a2.axhline(lvl, color=INK, lw=0.9, ls="--")
    a2.set_xticks(range(len(groups)), [g_[0] for g_ in groups], fontsize=8.5)
    a2.set_ylim(0.0, 1.02)
    a2.set_ylabel("share of windows with the exact\nweight inside the interval")
    a2.errorbar([], [], yerr=[], fmt="o", color=AUTO, mfc="white", label="90%: central quantiles of the samples as fitted")
    a2.errorbar([], [], yerr=[], fmt="o", color=AUTO, label="90%: the interval the report states")
    a2.errorbar([], [], yerr=[], fmt="o", color="#0b3d91", mfc="white", label="50%: samples as fitted")
    a2.errorbar([], [], yerr=[], fmt="o", color="#0b3d91", label="50%: as reported")
    a2.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    a2.set_title("exact window weights inside the 50% / 90% intervals\n(all settings and windows; dashes: nominal)",
                 fontsize=10)
    for ax in (a0, a1, a2):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle(f"Narrow lines in the report: the fit's width floor lowered when they are flagged, and window "
               f"intervals spanning both line shapes ({Path(runs[0]['flow']).stem}; {len(lines_all)} line spectra, "
               f"{len(lat_all)} lattice spectra)", fontsize=11)
    f.savefig(out / "floor_option.png", dpi=150)
    with open(out / "floor_option.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    for s in summary["settings"]:
        w = s["w1_lines"]
        print(f"beta {s['beta']:g} noise {s['noise']:g}: fired on {s['lines_fired']}/{s['lines']} line, "
              f"{s['lattice_fired']}/{s['lattice']} lattice | W1 lines as trained {w['standard']:.3f} "
              f"auto {w['auto']:.3f} MaxEnt {w['maxent']:.3f} | auto/as trained "
              "{:.3f} [{:.3f}, {:.3f}]".format(*s["auto_over_standard"])
              + " | auto/MaxEnt {:.3f} [{:.3f}, {:.3f}]".format(*s["auto_over_maxent"])
              + " | as trained/MaxEnt {:.3f} [{:.3f}, {:.3f}]".format(*s["standard_over_maxent"]))
    show = lambda k, c: print(f"{k:96s} 50%: {c['c50']:.3f}±{c['se50']:.3f}  90%: {c['c90']:.3f}±{c['se90']:.3f}"
                              f"  ({c['truths']} spectra, {c['pairs']} windows)")
    for k, c in list(summary["coverage"].items()) + list(summary["coverage_other"].items()):
        if c:
            show(k, c)
    for k, b in summary["bracket"].items():
        if b:
            print(f"data-only bracket holds the exact weight, {k}: {b['inside']} of {b['windows']} windows")
    for k, v in (summary["by_edge_distance"] or {}).items():
        print(f"{k:58s} {v['windows']:4d} windows  as fitted {v['as_fitted'][0]:.2f} / {v['as_fitted'][1]:.2f}"
              f"  reported {v['reported'][0]:.2f} / {v['reported'][1]:.2f}  (90% width {v['width90_reported']:.3f})")
    for s in summary["settings"]:
        print(f"-- beta {s['beta']:g} noise {s['noise']:g}")
        for k, c in s["coverage"].items():
            if c:
                show("   " + k, c)
    print(f"-> {out}/floor_option.{{json,png}}")


def main():
    if sys.argv[1] == "fig":
        return fig(sys.argv[2])
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    run(out, sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 2)


if __name__ == "__main__":
    main()
