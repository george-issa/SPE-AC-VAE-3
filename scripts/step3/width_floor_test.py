"""Does a lower width floor in the FIT alone help on line spectra?  No retraining.

The flow proposes pole sets whose widths sit at or above its floor
(gamma >= 0.1 pi / beta).  Here the same proposals are refined twice against
the same data and the same noise redraws:

  standard   the floor the model was trained with
  lowered    the floor divided by `factor` (default 4), on a quadrature
             refined by the same factor (`spevae.models.flow.LoweredFloor`)

and, for the lowered floor, once more with a weaker tether (1 instead of 10),
so that a null result cannot be the tether holding the widths at the proposal.
Scored as in the known-answer judge (W1 of the posterior mean to the exact
cumulative weight); MaxEnt's score on the same draw is read from that judge's
output.

    python scripts/step3/width_floor_test.py out/step3/width_floor out/step3/F3_60k_s1.pt \
        out/eval/exact_models_F3_60k_s1 [factor=4] [quick]

Writes width_floor_test.{json,png}.  `quick` runs the first line spectrum of
the first setting only and prints; `fig DIR` re-renders the figure from the
stored results.
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

# (beta, noise, kind, arms)
SETTINGS = ((8.0, 1e-4, "discrete", ("standard", "lowered", "lowered_weak")),
            (2.0, 1e-3, "discrete", ("standard", "lowered", "lowered_weak")),
            (8.0, 1e-3, "continuum", ("standard", "lowered")),
            (8.0, 1e-4, "continuum", ("standard", "lowered")))
ARM = {"standard": (1.0, 10.0), "lowered": (None, 10.0), "lowered_weak": (None, 1.0)}   # (factor, tether)
COL = {"standard": "#9ec5f4", "lowered": "#2a78d6", "lowered_weak": "#0b3d91", "maxent": J.ME}
LABEL = {"standard": "flow, floor as trained", "lowered": "flow, floor lowered in the fit",
         "lowered_weak": "same, weaker tether (1)", "maxent": "MaxEnt"}


def run_truth(flow, item, beta, noise, arms, factor):
    from spevae.evaluation import cases as C
    from spevae.models.flow import LoweredFloor, refine_samples
    from spevae.posterior import weight_at_width_floor
    inst = C.draw(item["truth"], noise, J.SEED)
    tr = inst.truth
    L = tr.x.numel()
    g = torch.Generator().manual_seed(9)
    z, norm = flow.sample_coords(inst.G, tr.x, beta, sigma=inst.sigma, n_samples=J.S, generator=g,
                                 inv_sqrt_R=inst.isr)
    state = g.get_state()
    low = LoweredFloor(flow, factor)
    out = {}
    for arm in arms:
        fac, lam = ARM[arm]
        g.set_state(state)                                   # same noise redraws in every arm
        t0 = time.time()
        args = (norm, beta, tr.x, inst.G.expand(J.S, -1), inst.sigma.expand(J.S, -1), inst.isr, lam)
        kw = dict(perturb=True, generator=g, anneal_from=1e-2)
        if fac == 1.0:
            ps, c2 = refine_samples(flow, z, *args, **kw)
        else:
            ps, c2 = refine_samples(low, low.recode(z), *args, **kw)
        acc = (c2 <= 1.0 + 3.0 * math.sqrt(2.0 / L)).numpy()
        none_fit = not acc.any()
        if none_fit:
            acc[:] = True
        A = J.a_phys(ps, beta)[acc]
        at = torch.as_tensor(acc)
        tot = ps.a[at].sum(-1).clamp_min(1e-30)
        out[arm] = {"w1": J.w1_cdf(np.cumsum(A.mean(0)) * J.DW, item["cdf"]),
                    "accepted": float(acc.mean()), "no_sample_fits": bool(none_fit),
                    "chi2_median": float(np.median(c2.numpy()[acc])),
                    "chi2_median_all": float(np.median(c2.numpy())),
                    "share_at_trained_floor": weight_at_width_floor(ps.a[at], ps.gamma[at],
                                                                    flow.coords.gamma_min),
                    "share_below_trained_floor": float(
                        ((ps.a[at] * (ps.gamma[at] < 0.95 * flow.coords.gamma_min)).sum(-1) / tot).median()),
                    "negative_weight": float((np.minimum(A.mean(0), 0.0).sum() * J.DW)),
                    "seconds": time.time() - t0}
    return out


def gm(v):
    return float(np.exp(np.mean(np.log(v))))


def render(res, out):
    f, axes = plt.subplots(1, len(res), figsize=(5.2 * len(res), 4.6), constrained_layout=True)
    for ax, s in zip(np.atleast_1d(axes), res):
        xs = np.arange(len(s["rows"]))
        for a in s["arms"]:
            ax.plot(xs, [r[a]["w1"] for r in s["rows"]], "o", ms=5, color=COL[a], label=LABEL[a])
        ax.plot(xs, [r["maxent_w1"] for r in s["rows"]], "_", ms=11, mew=1.8, color=COL["maxent"],
                label=LABEL["maxent"])
        ax.set_yscale("log")
        ax.set_xticks(xs, [r["family"].replace("hubbard_", "hub_").replace("anderson_", "and_")
                           for r in s["rows"]], rotation=90, fontsize=7)
        g_ = s["w1_geomean"]
        what = "exact line spectra" if s["kind"] == "discrete" else "exact lattice spectra"
        ax.set_title(f"{what}, beta = {s['beta']:g}, error bar {s['noise']:g}\ngeometric mean: " +
                     ", ".join(f"{g_[a]:.3f}" for a in s["arms"]) + f"; MaxEnt {g_['maxent']:.3f}", fontsize=9)
        ax.set_ylabel("W$_1$ to the exact answer")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    f.legend(h, l, loc="outside lower center", ncol=4, fontsize=8.5, frameon=False)
    f.suptitle(f"Width floor lowered {res[0]['factor']:g}x in the fit only (same proposals, same data, same noise "
               "redraws; no retraining)", fontsize=10.5)
    f.savefig(out / "width_floor_test.png", dpi=150)


def main():
    if sys.argv[1] == "fig":                       # re-render from the stored results
        out = Path(sys.argv[2])
        return render(json.load(open(out / "width_floor_test.json")), out)
    from spevae.models.flow import load_flow
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    flow = load_flow(sys.argv[2]).eval()
    judged = Path(sys.argv[3])
    factor = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "quick" else 4.0
    quick = sys.argv[-1] == "quick"
    torch.set_num_threads(4)
    res = []
    for beta, noise, kind, arms in SETTINGS[:1] if quick else SETTINGS:
        rec = json.load(open(judged / f"exact_models_b{beta:g}_n{noise:g}_lam10.json"))
        ref = [r for r in rec["rows"] if r["kind"] == kind]
        items = [t for t in J.make_truths(beta) if t["kind"] == kind]
        rows = []
        for k, (item, r0) in enumerate(zip(items[:1] if quick else items, ref)):
            r = run_truth(flow, item, beta, noise, arms, factor)
            r["family"], r["maxent_w1"], r["judged_flow_w1"] = item["family"], r0["w1"]["maxent"], r0["w1"]["flow"]
            rows.append(r)
            print(f"b{beta:g} n{noise:g} {kind:9s} {item['family']:16s} MaxEnt {r['maxent_w1']:.3f} | " + " | ".join(
                f"{a} W1 {r[a]['w1']:.3f} acc {r[a]['accepted']:.0%} chi2 {r[a]['chi2_median_all']:.2f} "
                f"below {r[a]['share_below_trained_floor']:.2f} ({r[a]['seconds']:.0f}s)" for a in arms),
                flush=True)
        summ = {a: gm([r[a]["w1"] for r in rows]) for a in arms}
        summ["maxent"] = gm([r["maxent_w1"] for r in rows])
        res.append({"beta": beta, "noise": noise, "kind": kind, "arms": list(arms), "factor": factor,
                    "w1_geomean": summ, "rows": rows})
        print(f"== b{beta:g} n{noise:g} {kind}: geometric-mean W1 " +
              ", ".join(f"{k} {v:.3f}" for k, v in summ.items()), flush=True)
        if not quick:
            with open(out / "width_floor_test.json", "w") as fh:
                json.dump(res, fh, indent=1, default=float)
    if quick:
        return
    render(res, out)
    print(f"-> {out}/width_floor_test.{{json,png}}")


if __name__ == "__main__":
    main()
