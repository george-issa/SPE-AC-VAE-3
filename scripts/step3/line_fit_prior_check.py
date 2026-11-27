"""Zero-width lines on held-out draws of the training prior: what the change does where the
narrow features are NOT lines.

`line_fit_judge.py` scores the line fit on exact line spectra, where a line is
the right reading.  The prior also holds spectra whose narrowest features are
peaks of the model's minimum width (pole mixtures, band edges), and the report
flags those the same way.  Here two runs of `calibration.py` on the same
held-out truths, noise draws and seeds are compared: OLD (narrow poles fitted
as Lorentzians of a quarter of the floor) and NEW (as zero-width lines), by
kind of truth:

  - how often the exact weight of a frequency window lies inside the stated
    50% / 90% intervals (cluster standard error per truth), and inside the
    interval of the line reading alone;
  - the error of the posterior-mean window weights (mean absolute, per truth),
    NEW over OLD;
  - that a truth the report does not flag is reported exactly as before.

    python scripts/step3/line_fit_prior_check.py OLD_DIR NEW_DIR OUT     # -> OUT.{json,png}
"""
import glob
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OLD, NEW, INK = "#9ec5f4", "#0b7a75", "#0b0b0b"


def load(d):
    cells = sum((json.load(open(p)) for p in sorted(glob.glob(d + "/calibration.shard*.json"))), [])
    return {(round(c["noise"], 12), c["beta"], c["rep"]): c for c in cells}


def coverage(rows, key):
    """Share of windows with the exact weight inside the 50% / 90% interval `key` of each row."""
    per = []
    for r in rows:
        q, t = np.asarray(r[key]), np.asarray(r["truth"])
        per.append((((q[:, 1] <= t) & (t <= q[:, 2])).sum(), ((q[:, 0] <= t) & (t <= q[:, 3])).sum(), len(t)))
    p = np.array(per, float)
    tot, m = p[:, 2].sum(), len(p)
    c50, c90 = p[:, 0].sum() / tot, p[:, 1].sum() / tot
    se = lambda h, c: float(math.sqrt(m / max(m - 1, 1) * (((h - c * p[:, 2]) ** 2).sum())) / tot)
    return {"c50": float(c50), "c90": float(c90), "se50": se(p[:, 0], c50), "se90": se(p[:, 1], c90), "truths": m}


def ratio_ci(d):
    from scipy import stats
    d = np.asarray(d, float)
    if len(d) < 2 or float(d.std()) == 0.0:
        return [float(np.exp(d.mean()))] * 3
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return [float(np.exp(d.mean())), float(np.exp(d.mean() - h)), float(np.exp(d.mean() + h))]


def main():
    old, new, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    rows = []
    for k in sorted(set(old) & set(new)):
        co, cn = old[k], new[k]
        assert np.allclose(co["w_true"], cn["w_true"]), "the two runs do not hold the same truths"
        for m, fam in enumerate(co["family"]):
            a, b = co["arms"]["posterior"][m], cn["arms"]["posterior"][m]
            t = np.asarray(co["w_true"][m])
            err = lambda arm: float(np.abs(np.asarray(arm["w"]).mean(0) - t).mean())
            row = {"noise": co["rel_noise"][m], "beta": co["beta"], "family": fam, "truth": t,
                   "kind": ("exact line spectra" if fam.split(":")[0] in ("lines", "exact") else
                            "pole mixtures" if fam.startswith("pole_mixture") else "smooth and lattice spectra"),
                   "flagged": a["floor_factor"] != 1.0, "line_fit": bool(b.get("line_fit")),
                   "on_lines": b.get("on_lines", 0.0), "old": a["reported"], "new": b["reported"],
                   "err_old": err(a), "err_new": err(b), "accepted_old": a["accepted"], "accepted_new": b["accepted"],
                   # the same samples?  (the stated intervals can differ by a band table added since)
                   "same_samples": np.asarray(a["w"]).shape == np.asarray(b["w"]).shape
                   and bool(np.allclose(a["w"], b["w"], atol=1e-5))}
            if b.get("as_fitted"):                       # the line reading alone: 90% only
                q = np.asarray(b["as_fitted"])
                row["lines_alone_90"] = float(((q[:, 0] <= t) & (t <= q[:, 1])).mean())
            rows.append(row)
    res = {"old": sys.argv[1], "new": sys.argv[2], "truths": len(rows),
           "flagged": int(sum(r["flagged"] for r in rows)), "groups": {}}
    same = [r for r in rows if not r["flagged"]]
    res["not_flagged"] = {"truths": len(same), "samples_identical": int(sum(r["same_samples"] for r in same))}
    kinds = ("exact line spectra", "pole mixtures", "smooth and lattice spectra")
    for name, sel in [(f"flagged, {k}", (lambda r, k=k: r["flagged"] and r["kind"] == k)) for k in kinds] + [
            ("flagged, not line spectra", lambda r: r["flagged"] and r["kind"] != "exact line spectra"),
            ("flagged, all", lambda r: r["flagged"]), ("all truths", lambda r: True)]:
        g = [r for r in rows if sel(r)]
        if len(g) < 2:
            continue
        v = {"truths": len(g), "line_fit": int(sum(r["line_fit"] for r in g)),
             "stated_old": coverage(g, "old"), "stated_new": coverage(g, "new"),
             "window_error_old": float(np.mean([r["err_old"] for r in g])),
             "window_error_new": float(np.mean([r["err_new"] for r in g])),
             "window_error_new_over_old": ratio_ci([math.log(max(r["err_new"], 1e-12) / max(r["err_old"], 1e-12))
                                                    for r in g]),
             "width90_old": float(np.median(np.concatenate([np.asarray(r["old"])[:, 3] - np.asarray(r["old"])[:, 0]
                                                            for r in g]))),
             "width90_new": float(np.median(np.concatenate([np.asarray(r["new"])[:, 3] - np.asarray(r["new"])[:, 0]
                                                            for r in g])))}
        la = [r["lines_alone_90"] for r in g if "lines_alone_90" in r and r["line_fit"]]
        if la:
            v["line_reading_alone_90"] = float(np.mean(la))
        res["groups"][name] = v
    with open(out + ".json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    shown = [n for n in res["groups"] if n.startswith("flagged")]
    f, (a0, a1) = plt.subplots(1, 2, figsize=(15.5, 5.2), constrained_layout=True)
    x = np.arange(len(shown))
    for dx, arm, col in ((-0.2, "old", OLD), (0.2, "new", NEW)):
        c = [res["groups"][n][f"stated_{arm}"] for n in shown]
        a0.errorbar(x + dx - 0.06, [v["c50"] for v in c], yerr=[1.96 * v["se50"] for v in c], fmt="o", color=col,
                    ms=8, capsize=3, lw=1.3)
        a0.errorbar(x + dx + 0.06, [v["c90"] for v in c], yerr=[1.96 * v["se90"] for v in c], fmt="o", color=col,
                    ms=8, mfc="white", capsize=3, lw=1.3)
    for lv in (0.5, 0.9):
        a0.axhline(lv, color=INK, lw=0.8, ls="--")
    a0.set_ylim(0, 1.04)
    lab = [f"{n.replace('flagged, ', '')}\n({res['groups'][n]['truths']} truths)" for n in shown]
    a0.set_xticks(x, lab, fontsize=8.5)
    a0.set_ylabel("share of windows with the exact weight inside")
    a0.plot([], [], "o", color=OLD, label="narrow poles as Lorentzians of a quarter of the floor (before)")
    a0.plot([], [], "o", color=NEW, label="narrow poles as zero-width lines")
    a0.plot([], [], "o", color=INK, label="stated 50% interval")
    a0.plot([], [], "o", color=INK, mfc="white", label="stated 90% interval")
    a0.legend(fontsize=8.5, frameon=False, loc="lower left")
    a0.set_title("flagged held-out draws of the prior: exact window weights inside the stated intervals",
                 fontsize=9.5)
    for i, n in enumerate(shown):
        m, lo, hi = res["groups"][n]["window_error_new_over_old"]
        a1.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=NEW, ms=9, capsize=4, lw=1.5)
        a1.annotate(f"{res['groups'][n]['window_error_old']:.4f} -> {res['groups'][n]['window_error_new']:.4f}",
                    (i, hi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    a1.axhline(1.0, color=INK, lw=0.8, ls=":")
    a1.set_yscale("log")
    a1.margins(y=0.3)
    a1.set_xticks(x, lab, fontsize=8.5)
    a1.set_ylabel("error of the mean window weights, now / before\n(geometric mean over truths, 95% interval)")
    a1.set_title("error of the posterior-mean window weights (labels: mean absolute error before -> now)",
                 fontsize=9.5)
    for ax in (a0, a1):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    u = res["not_flagged"]
    f.suptitle(f"Zero-width lines on {res['truths']} held-out draws of the training prior ({res['flagged']} flagged; "
               f"of the {u['truths']} not flagged, {u['samples_identical']} have the same samples as before)",
               fontsize=10.5)
    f.savefig(out + ".png", dpi=150)

    fmt = lambda t: "{:.2f} [{:.2f}, {:.2f}]".format(*t)
    print(f"{res['truths']} truths, {res['flagged']} flagged; not flagged identical: "
          f"{u['samples_identical']} of {u['truths']}")
    for n, v in res["groups"].items():
        co, cn = v["stated_old"], v["stated_new"]
        print(f"{n:38s} n {v['truths']:3d} (line fit {v['line_fit']:3d})  stated {co['c50']:.2f}±{co['se50']:.2f}/"
              f"{co['c90']:.2f}±{co['se90']:.2f} -> {cn['c50']:.2f}±{cn['se50']:.2f}/{cn['c90']:.2f}±{cn['se90']:.2f}"
              f"  width {v['width90_old']:.3f} -> {v['width90_new']:.3f}  window error {v['window_error_old']:.4f} -> "
              f"{v['window_error_new']:.4f}, ratio {fmt(v['window_error_new_over_old'])}"
              + (f"  line reading alone 90%: {v['line_reading_alone_90']:.2f}" if "line_reading_alone_90" in v else ""))
    print(f"-> {out}.{{json,png}}")


if __name__ == "__main__":
    main()
