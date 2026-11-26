"""Which reading of the poles gives honest window intervals on line spectra?  Scored offline.

`discrete_option_judge.py run` stores the fitted poles of every exact line
spectrum.  Here the window intervals are formed from those poles under several
readings and scored against the exact window weights, without refitting:

  as fitted      every pole the smooth feature the fit made (no band table)
  lines          every pole as `n` sharp lines of equal weight, their positions
                 drawn from a distribution of the pole's width about its centre:
                 `cauchy` (the pole's own Lorentzian shape, tails included),
                 `u1` / `u2` (uniform within one / two half-widths)

The stated interval of a reading is the union of the as-fitted interval and
the lines interval; `clip` also cuts it to the range the data alone allow
(taken from the report judged without the statement).  The five settings of
the known-answer judge choose the reading; the edge-of-range settings are held
out and score the choice.

    python scripts/step3/discrete_rules.py OUT REF_DIR [REF_DIR ...]

OUT holds discrete_b*.json + .poles.npz; every REF_DIR is searched for the
`floor_option_judge.py` result of the same setting (flag, data-alone range).
Writes OUT/discrete_rules.{json,png}.
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

from spevae.posterior import DEFAULT_WINDOWS, central_interval

TUNE = {(8.0, 1e-4, 0.125), (8.0, 1e-3, 0.125), (8.0, 1e-2, 0.125), (2.0, 1e-3, 0.125), (16.0, 1e-3, 0.125)}
DRAWS = 16
RULES = [("as fitted", None, 0, False)] + [
    (f"{d}, {n} line{'s' if n > 1 else ''}{', clipped' if c else ''}", d, n, c)
    for d in ("cauchy", "u2", "u1") for n in (1, 2, 3) for c in (False, True)]
INK = "#0b0b0b"


def window_weights(p, beta, dist, n, rng):
    """(draws * samples, windows) window weights of the pole sets `p` = (eps, gamma, a, b) (physical
    units, each (samples, poles)); `dist` None = as fitted (one draw)."""
    eps, gam, a, b = (v * beta if k < 2 else v for k, v in enumerate(p))
    tot = a.sum(-1)
    if dist is None:
        pos = None
    else:
        shape = (DRAWS, n) + eps.shape
        t = (np.tan(math.pi * (rng.random(shape) - 0.5)) if dist == "cauchy"
             else (2.0 * rng.random(shape) - 1.0) * (1.0 if dist == "u1" else 2.0))
        pos = eps + gam * t                                        # (draws, n, samples, poles)

    def cdf(w):
        if math.isinf(w):
            return np.zeros_like(tot) if w < 0 else tot
        u = w * beta - eps
        disp = b * np.log(u * u + gam * gam) / (2.0 * math.pi)
        if pos is None:
            return (a * (np.arctan(u / gam) / math.pi + 0.5) - disp).sum(-1)
        return (a * (w * beta > pos).mean(1) - disp).sum(-1)       # (draws, samples)

    out = np.stack([(cdf(hi) - cdf(lo)) / tot for lo, hi in DEFAULT_WINDOWS], -1)
    return out.reshape(-1, len(DEFAULT_WINDOWS))


def coverage(rows, key):
    """Share of windows with the exact weight inside the 50% / 90% interval of reading `key`,
    cluster standard error per spectrum; median widths and share of the data-alone range."""
    per, w90, sh = [], [], []
    for r in rows:
        q, t = r["q"][key], r["truth"]
        per.append((((q[:, 1] <= t) & (t <= q[:, 2])).sum(), ((q[:, 0] <= t) & (t <= q[:, 3])).sum(), len(t)))
        w90 += list(q[:, 3] - q[:, 0])
        if r["bracket"] is not None:
            sh += list((q[:, 3] - q[:, 0]) / np.maximum(r["bracket"][:, 1] - r["bracket"][:, 0], 1e-12))
    p = np.array(per, float)
    tot, m = p[:, 2].sum(), len(p)
    c50, c90 = p[:, 0].sum() / tot, p[:, 1].sum() / tot
    se = lambda h, c: float(math.sqrt(m / max(m - 1, 1) * (((h - c * p[:, 2]) ** 2).sum())) / tot)
    return {"c50": float(c50), "c90": float(c90), "se50": se(p[:, 0], c50), "se90": se(p[:, 1], c90),
            "spectra": m, "width90_median": float(np.median(w90)),
            "share_of_data_alone": float(np.median(sh)) if sh else None}


def main():
    out, refs = Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]]
    rng = np.random.default_rng(11)
    rows = []
    for p in sorted(out.glob("discrete_b*.json")):
        r = json.load(open(p))
        if r["kind"] != "discrete":
            continue
        poles = np.load(str(p)[:-len(".json")] + ".poles.npz")
        dt = r.get("dtau", 0.125)
        name = f"floor_option_b{r['beta']:g}_n{r['noise']:g}" + ("" if dt == 0.125 else f"_dt{dt:g}") + ".json"
        ref = next((json.load(open(d / name))["rows"] for d in refs if (d / name).exists()), None)
        if ref is None:
            raise SystemExit(f"no {name} in {refs}")
        for x in r["rows"]:
            y = ref[x["index"]]
            assert y["family"] == x["family"] and y["params"] == x["params"]
            pp = poles[f"p{x['index']}"]
            row = {"setting": (r["beta"], r["noise"], dt), "tune": (r["beta"], r["noise"], dt) in TUNE,
                   "fired": bool(y["fired"]), "truth": np.asarray(x["fit"]["truth"]),
                   "bracket": np.asarray(y["auto"]["bracket"]) if "bracket" in y["auto"] else None,
                   "q": {"report as it is": np.asarray(y["auto"]["reported"])}}
            wf = window_weights(pp, r["beta"], None, 0, rng)
            fit = np.array([[*central_interval(wf[:, j], 0.9), *central_interval(wf[:, j], 0.5)]
                            for j in range(wf.shape[1])])[:, [0, 2, 3, 1]]          # q05, q25, q75, q95
            for label, dist, n, clip in RULES:
                if dist is None:
                    q = fit
                else:
                    wl = window_weights(pp, r["beta"], dist, n, rng)
                    ln = np.array([[*central_interval(wl[:, j], 0.9), *central_interval(wl[:, j], 0.5)]
                                   for j in range(wl.shape[1])])[:, [0, 2, 3, 1]]
                    q = np.column_stack([np.minimum(fit[:, :2], ln[:, :2]), np.maximum(fit[:, 2:], ln[:, 2:])])
                    if clip and row["bracket"] is not None:
                        lo, hi = row["bracket"][:, :1], row["bracket"][:, 1:]
                        q = np.minimum(np.maximum(q, lo), hi)
                row["q"][label] = q
            rows.append(row)
    labels = ["report as it is"] + [l for l, *_ in RULES]
    groups = (("tuning settings, not flagged", lambda r: r["tune"] and not r["fired"]),
              ("tuning settings, flagged", lambda r: r["tune"] and r["fired"]),
              ("tuning settings, all", lambda r: r["tune"]),
              ("held-out settings, not flagged", lambda r: not r["tune"] and not r["fired"]),
              ("held-out settings, flagged", lambda r: not r["tune"] and r["fired"]),
              ("held-out settings, all", lambda r: not r["tune"]))
    res = {"draws": DRAWS, "rules": {}}
    for label in labels:
        res["rules"][label] = {g: coverage([r for r in rows if sel(r)], label) for g, sel in groups
                               if any(sel(r) for r in rows)}
    held = {}
    for r in rows:
        if not r["tune"]:
            held.setdefault("beta {:g}, noise {:g}, dtau {:g}".format(*r["setting"]), []).append(r)
    res["held_out_by_setting"] = {label: {k: coverage(v, label) for k, v in held.items()} for label in labels}
    with open(out / "discrete_rules.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    shown = [g for g, _ in groups if g in res["rules"]["as fitted"]]
    f, axes = plt.subplots(1, len(shown), figsize=(4.3 * len(shown), 7.6), constrained_layout=True, sharey=True)
    for ax, g in zip(np.atleast_1d(axes), shown):
        for i, label in enumerate(labels):
            c = res["rules"][label][g]
            col = "#b3261e" if label == "report as it is" else "#7d7c78" if label == "as fitted" else (
                "#2a78d6" if "cauchy" in label else "#0b7a75" if "u2" in label else "#eb6834")
            ax.errorbar([c["c50"]], [i - 0.12], xerr=[1.96 * c["se50"]], fmt="o", color=col, ms=5, capsize=2, lw=1)
            ax.errorbar([c["c90"]], [i + 0.12], xerr=[1.96 * c["se90"]], fmt="o", color=col, ms=5, mfc="white",
                        capsize=2, lw=1)
            ax.annotate(f"{c['width90_median']:.2f}", (1.04, i), fontsize=7, va="center", color=col,
                        annotation_clip=False)
        for lv in (0.5, 0.9):
            ax.axvline(lv, color=INK, lw=0.8, ls="--")
        ax.set_xlim(0, 1.02)
        ax.set_title(f"{g}\n({res['rules']['as fitted'][g]['spectra']} spectra; right: median 90% width)",
                     fontsize=8.5)
        ax.set_xlabel("share of windows with the exact weight inside\n(filled 50%, open 90%; dashes nominal)",
                      fontsize=8)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    np.atleast_1d(axes)[0].set_yticks(range(len(labels)), labels, fontsize=8)
    np.atleast_1d(axes)[0].invert_yaxis()
    f.suptitle("Window intervals on exact line spectra under different readings of the fitted poles "
               "(union with the as-fitted interval; same fits throughout)", fontsize=10.5)
    f.savefig(out / "discrete_rules.png", dpi=150)

    for g in shown:
        print(f"== {g} ({res['rules']['as fitted'][g]['spectra']} spectra)")
        for label in labels:
            c = res["rules"][label][g]
            sh = c["share_of_data_alone"]
            print(f"   {label:28s} {c['c50']:.3f}±{c['se50']:.3f} / {c['c90']:.3f}±{c['se90']:.3f}   width90 "
                  f"{c['width90_median']:.3f}   share {sh if sh is None else round(sh, 2)}")
    print(f"-> {out}/discrete_rules.{{json,png}}")


if __name__ == "__main__":
    main()
