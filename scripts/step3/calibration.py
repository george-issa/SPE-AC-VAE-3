"""Calibration of the reported posterior on the step-3 prior, across noise and beta.

Truths come from the step-3 training distribution -- pole motifs, smooth and
exact lattice spectra from the bank, exactly diagonalised models -- held out
by seed, one batch per noise level with beta cycled over the grid.  Each
truth goes through `posterior.continue_posterior` exactly as a user's data
would (refine + redraw, raw samples that fit kept, births, accept / reject),
and through the raw flow for comparison.

Judged on the spectral weight in the report's seven frequency windows:

  * PIT coverage of the truth (randomised ranks; uniform when calibrated),
  * the data-fit rank: where the truth's chi^2 falls among the samples',
  * the window-weight error of the posterior mean.

    python scripts/step3/calibration.py out/step3/F3_6k.pt out/step3/bank/label_bank_trusted.pt \
        out/step3/cal_F3_6k [shard nshards [tether [base_scale [reps [count.pt]]]]]
    python scripts/step3/calibration.py fig out/step3/cal_F3_6k
    python scripts/step3/calibration.py recount SRC_DIR DST_DIR count.pt bank.pt

`tether` is `lam` of the refinement: how firmly each sample is held at the
flow's own proposal while it is fitted (default 10, the package default).
`reps` repeats the noise grid with beta and dtau shifted against it, so that
every noise level is seen at `reps` temperatures (default 1: 288 truths;
4 gives 1152, enough to judge coverage per noise band and per beta).
`CAL_FIRST_REP=k` in the environment starts at repetition k: fresh truths,
for testing a band table on a run it was not fitted on.
`count.pt` is a pole-count classifier (`scripts/step3/train_count.py`); with
it the reported samples are resampled to its count posterior.
`CAL_P_SMOOTH`, `CAL_P_EXACT` set the mixture of the prior the truths are
drawn from (default 0.35 / 0.15, the step-3 recipe); with a bank that holds
exact line spectra those are drawn with the bank's share.
`CAL_FLOOR` is handed to the report as `floor_factor` (default `auto`, the
report's own default; `CAL_FLOOR=1` is the width floor as trained, which is
what every run before 2026-11-25 used); each truth records the factor its
fit used and the intervals as the report states them.
"""
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import torch

NOISES = np.exp(np.linspace(math.log(1e-5), math.log(3e-2), 12)).tolist()
BETAS = (1.0, 2.0, 4.0, 8.0, 12.0, 20.0)
DTAUS = (0.05, 0.125, 0.25, 0.1, 0.02, 0.2, 0.5)
M, S = int(os.environ.get("CAL_TRUTHS", 24)), int(os.environ.get("CAL_SAMPLES", 32))
# mixture of the prior the truths are drawn from: the step-3 recipe unless the checkpoint's differs
P_SMOOTH, P_EXACT = float(os.environ.get("CAL_P_SMOOTH", 0.35)), float(os.environ.get("CAL_P_EXACT", 0.15))
FLOOR = os.environ.get("CAL_FLOOR", "auto")
FLOOR = FLOOR if FLOOR == "auto" else float(FLOOR)
ARMS = ("raw", "posterior", "posterior_count")
BANDS = ((1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1.0))      # noise bands of the shipping gate
COL = {"raw": "#2a78d6", "posterior": "#eb6834", "posterior_count": "#0f8a8a"}
MUTED, GRID = "#52514e", "#d9d8d4"


def window_weights_poles(ps, beta, windows):
    """Exact window weights of dimensionless pole sets, (N, n_windows)."""
    eps, gam, a, b = (getattr(ps, k).double() for k in ("eps", "gamma", "a", "b"))

    def cdf(w):
        if math.isinf(w):
            return torch.zeros(eps.shape[0], dtype=torch.float64) if w < 0 else a.sum(-1)
        u = w * beta - eps
        return (a * (torch.atan(u / gam) / math.pi + 0.5)
                - b * torch.log(u * u + gam * gam) / (2.0 * math.pi)).sum(-1)

    tot = a.sum(-1)
    return torch.stack([(cdf(hi) - cdf(lo)) / tot for lo, hi in windows], -1).numpy()


def cell_batch(bank, j):
    """Cell ``j`` of the design: ``(noise, beta, repetition, batch, true window weights)``.

    Repetition 0 (``j < 12``) is the original design; later repetitions shift
    beta and dtau against the noise grid.  Deterministic in ``j``.
    """
    from spevae.data.pole_prior import PolePriorGenerator
    from spevae.posterior import DEFAULT_WINDOWS
    r_, i_ = divmod(j, len(NOISES))
    nz = NOISES[i_]
    beta = BETAS[(i_ + r_) % len(BETAS)]
    gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive", smooth_bank=bank,
                             p_smooth=P_SMOOTH, p_exact=P_EXACT, beta_choices=(beta,),
                             noise_scale_range=(nz, nz),
                             dtau_choices=(DTAUS[(i_ + 2 * r_) % len(DTAUS)],))
    with gen.fixed_scalars(31000 + j):
        b = gen.batch(M, torch.Generator().manual_seed(32000 + j))
    nodes = (b.quad.nodes.double() / beta).numpy()
    wq = b.quad.weights.double().numpy()
    A_true = b.A_true.double().numpy()                            # exact, dimensionless, on nodes
    tot = (A_true * wq).sum(-1)
    w_true = np.stack([(A_true * wq * ((nodes >= lo) & (nodes < hi))).sum(-1) / tot
                       for lo, hi in DEFAULT_WINDOWS], -1)        # (M, 7)
    return nz, beta, r_, b, w_true


def count_arm(count_model, b, m, beta, last, j):
    """The posterior arm's samples of truth ``m``, resampled to the classifier's counts."""
    from spevae.models.count import resample_by_count
    p_n = count_model.probs(b.G[m], b.x, beta, sigma=b.sigma[m])[0]
    take, cd = resample_by_count(p_n, torch.tensor(last["n"]),
                                 generator=torch.Generator().manual_seed(1300 + 97 * j + m))
    t = take.numpy()
    return {"w": np.asarray(last["w"])[t].tolist(), "chi2": np.asarray(last["chi2"])[t].tolist(),
            "ess": cd["count_ess"], "covered": cd["count_mass_covered"],
            "p_true": float(p_n[int(b.poles_true["n_true"][m]) - 1])}


def recount(src, dst, count_path, bank):
    """Re-score a finished run with another count classifier: same samples, new weights."""
    from spevae.models.count import load_count
    torch.set_num_threads(4)                       # as in run(): bit-identical classifier outputs
    count_model = load_count(count_path)
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("calibration.shard*.json")):
        rows = json.load(open(p))
        for r in rows:
            j = r.get("rep", 0) * len(NOISES) + min(range(len(NOISES)), key=lambda k: abs(NOISES[k] - r["noise"]))
            nz, beta, _, b, w_true = cell_batch(bank, j)
            if abs(nz - r["noise"]) > 1e-12 or np.abs(w_true - np.array(r["w_true"])).max() > 1e-9:
                raise SystemExit(f"{p}: cell {j} does not regenerate the stored truths")
            if "n" not in r["arms"]["posterior"][0]:
                raise SystemExit(f"{p}: this run did not store the samples' pole counts")
            r["arms"]["posterior_count"] = [count_arm(count_model, b, m, beta, a, j)
                                            for m, a in enumerate(r["arms"]["posterior"])]
        with open(dst / p.name, "w") as fh:
            json.dump(rows, fh)
    print(f"{src} re-scored with {count_path} -> {dst}")


def run(flow_path, bank, out, shard, nshards, lam=10.0, base_scale=None, reps=1, count_path=None):
    from spevae.models.flow import load_flow
    from spevae.physics import PoleGreens
    from spevae.physics.poles import PoleSet
    from spevae.posterior import DEFAULT_WINDOWS, continue_posterior

    torch.set_num_threads(4)
    flow = load_flow(flow_path).eval()
    if base_scale is not None:
        flow.base_scale = float(base_scale)
    count_model = None
    if count_path:
        from spevae.models.count import load_count
        count_model = load_count(count_path)
    rows = []
    path = out / f"calibration.shard{shard}.json"
    first = int(os.environ.get("CAL_FIRST_REP", 0))        # skip repetitions already run elsewhere
    for j in range(first * len(NOISES), (first + int(reps)) * len(NOISES)):
        if j % nshards != shard:
            continue
        nz, beta, r_, b, w_true = cell_batch(bank, j)
        i = j                                      # seeds below are per cell
        L = b.x.numel()
        d = ((b.G_clean - b.G) / b.sigma) @ b.inv_sqrt_R.T
        c_true = (d * d).mean(-1).numpy()
        fwd = PoleGreens(b.quad)
        t0 = time.time()
        rec = {"noise": nz, "beta": beta, "n_tau": int(L), "rep": r_, "reps": int(reps), "samples": S,
               "family": list(b.family), "w_true": w_true.tolist(),
               "n_true": [int(v) for v in b.poles_true["n_true"]],
               # the stratum variable of the band table, exactly as continue_posterior computes it
               "rel_noise": (b.sigma / b.G.abs().amax(-1, keepdim=True)).median(-1).values.tolist(),
               "chi2_truth": c_true.tolist(), "arms": {"raw": [], "posterior": []}}
        for m in range(M):
            g = torch.Generator().manual_seed(500 + 97 * i + m)
            z, norm = flow.sample_coords(b.G[m:m + 1], b.x, beta, sigma=b.sigma[m:m + 1], n_samples=S,
                                         generator=g, inv_sqrt_R=b.inv_sqrt_R)
            raw = flow.coords.decode(z, norm=norm, nu_max=flow.nu_max(beta))
            dr = ((fwd(raw, b.x) - b.G[m:m + 1]) / b.sigma[m:m + 1]) @ b.inv_sqrt_R.T
            rec["arms"]["raw"].append({"w": window_weights_poles(raw, beta, DEFAULT_WINDOWS).tolist(),
                                       "chi2": (dr * dr).mean(-1).tolist()})
            rep = continue_posterior(flow, b.G[m], b.x, beta, b.sigma[m], b.inv_sqrt_R, n_samples=S,
                                     bounds=False, lam=lam, floor_factor=FLOOR,
                                     generator=torch.Generator().manual_seed(900 + 97 * i + m))
            dim = PoleSet(rep.poles.eps * beta, rep.poles.gamma * beta, rep.poles.a, rep.poles.b)
            rec["arms"]["posterior"].append({
                "w": window_weights_poles(dim, beta, DEFAULT_WINDOWS).tolist(), "chi2": rep.chi2.tolist(),
                "accepted": rep.n_accepted, "kept_raw": rep.n_kept_raw,
                "born": rep.diagnostics.get("samples_given_a_pole", 0), "warnings": len(rep.warnings),
                "floor_factor": rep.diagnostics["floor_factor"],
                # the intervals exactly as the report states them: (q05, q25, q75, q95) per window
                "reported": [[w_["q05"], w_["q25"], w_["q75"], w_["q95"]] for w_ in rep.windows],
                "line_shape": bool(rep.diagnostics.get("line_shape_readings")),
                # narrow poles fitted as zero-width lines (TRAINING.md (xxxi)): the share of the
                # weight on them, and the window intervals of the line reading alone
                "line_fit": bool(rep.diagnostics.get("line_fit")),
                "on_lines": rep.diagnostics.get("weight_on_lines", 0.0),
                "as_fitted": [list(w_["as_fitted"]) for w_ in rep.windows] if "as_fitted" in rep.windows[0] else None,
                "at_floor": rep.diagnostics.get("weight_at_trained_width_floor",
                                                rep.diagnostics["weight_at_width_floor"])})
            rec["arms"]["posterior"][-1]["n"] = (rep.poles.a != 0).sum(-1).tolist()
            if count_model is not None:               # the same samples, counts matched to the classifier
                rec["arms"].setdefault("posterior_count", []).append(
                    count_arm(count_model, b, m, beta, rec["arms"]["posterior"][-1], j))
        rec["seconds"] = time.time() - t0
        rows.append(rec)
        acc = np.mean([a["accepted"] for a in rec["arms"]["posterior"]]) / S
        print(f"noise {nz:7.1e} beta {beta:4.0f} L {L:3d}: accepted {acc:.2f}  ({rec['seconds']:.0f}s)", flush=True)
        with open(path, "w") as fh:
            json.dump(rows, fh)


def _stats(pit, fit, err):
    in50, in90 = (pit > .25) & (pit < .75), (pit > .05) & (pit < .95)
    n = len(pit)
    se = lambda v: float(v.mean(1).std() / math.sqrt(n))
    return {"truths": n, "cov50": float(in50.mean()), "cov50_se": se(in50),
            "cov90": float(in90.mean()), "cov90_se": se(in90),
            "fit_rank": float(fit.mean()), "weight_error": float(np.median(err))}


def summarise(out):
    rows = []
    for p in sorted(out.glob("calibration.shard*.json")):
        rows += json.load(open(p))
    rows.sort(key=lambda r: (r["noise"], r.get("rep", 0)))
    reps = max(r.get("reps", 1) for r in rows) if rows else 1
    if len(rows) != reps * len(NOISES):          # shards write cell by cell: refuse a partial run
        raise SystemExit(f"{out}: {len(rows)} of {reps * len(NOISES)} cells present -- still running?")
    rng = np.random.default_rng(3)
    arms = [a for a in ARMS if all(a in r["arms"] for r in rows)]
    n_samp = rows[0].get("samples", 32)
    cells = []
    for r in rows:
        wt, ct = np.array(r["w_true"]), np.array(r["chi2_truth"])
        c = {"noise": r["noise"], "beta": r["beta"], "n_tau": r["n_tau"],
             "kind": [f.split(":")[0] for f in r["family"]]}
        for arm in arms:
            pit, fit, err = [], [], []
            for m, a in enumerate(r["arms"][arm]):
                w, ch = np.array(a["w"]), np.array(a["chi2"])
                n = len(w)
                pit.append(((w < wt[m]).sum(0) + rng.random(wt.shape[1])) / (n + 1))
                fit.append(((ch < ct[m]).sum() + rng.random()) / (n + 1))
                err.append(np.abs(w.mean(0) - wt[m]).sum())
            c[arm] = (np.array(pit), np.array(fit), np.array(err))
        c["accepted"] = float(np.mean([a["accepted"] for a in r["arms"]["posterior"]]) / n_samp)
        c["born"] = float(np.mean([a["born"] for a in r["arms"]["posterior"]]) / n_samp)
        if "posterior_count" in arms:
            pc = r["arms"]["posterior_count"]
            c["count"] = {k: float(np.mean([a[k] for a in pc])) for k in ("ess", "covered", "p_true")}
        cells.append(c)

    def group(sel):
        g = {arm: _stats(*(np.concatenate([c[arm][k] for c in sel]) for k in range(3)))
             for arm in arms}
        if "count" in sel[0]:
            g["count"] = {k: float(np.mean([c["count"][k] for c in sel])) for k in ("ess", "covered", "p_true")}
        g["accepted"] = float(np.mean([c["accepted"] for c in sel]))
        g["born"] = float(np.mean([c["born"] for c in sel]))
        return g

    res = {"design": {"reps": reps, "truths_per_cell": M, "samples": n_samp}, "levels": [], "cells": [],
           "by_beta": {}, "by_noise_band": {}}
    for nz in sorted({c["noise"] for c in cells}):
        sel = [c for c in cells if c["noise"] == nz]
        one = len(sel) == 1
        res["levels"].append({"noise": nz, "beta": sel[0]["beta"] if one else None,
                              "betas": [c["beta"] for c in sel],
                              "n_tau": sel[0]["n_tau"] if one else None, **group(sel)})
    for c in cells:
        res["cells"].append({"noise": c["noise"], "beta": c["beta"], "n_tau": c["n_tau"], **group([c])})
    for b in sorted({c["beta"] for c in cells}):
        res["by_beta"][f"{b:g}"] = group([c for c in cells if c["beta"] == b])
    for lo, hi in BANDS:
        sel = [c for c in cells if lo <= c["noise"] < hi]
        if sel:
            res["by_noise_band"][f"{lo:g}-{hi:g}"] = group(sel)
    res["pooled"] = {}
    for arm in arms:
        pit, fit, err = (np.concatenate([c[arm][k] for c in cells]) for k in range(3))
        kind = np.array(sum((c["kind"] for c in cells), []))
        n = len(pit)
        ks = max(float(np.max(np.abs(np.sort(pit[:, j]) - (np.arange(1, n + 1) - .5) / n)) + .5 / n)
                 for j in range(pit.shape[1]))
        in50, in90 = (pit > .25) & (pit < .75), (pit > .05) & (pit < .95)
        res["pooled"][arm] = {
            "truths": n, "cov50": float(in50.mean()), "cov50_se": float(in50.mean(1).std() / math.sqrt(n)),
            "cov90": float(in90.mean()), "cov90_se": float(in90.mean(1).std() / math.sqrt(n)),
            "ks_max": ks, "ks_crit_5pct_bonferroni7": 1.63 / math.sqrt(n) * 0.98,
            "fit_rank_mean": float(fit.mean()), "weight_error_median": float(np.median(err)),
            "by_kind": {k: {"n": int((kind == k).sum()), "cov50": float(in50[kind == k].mean()),
                            "cov90": float(in90[kind == k].mean()),
                            "weight_error": float(np.median(err[kind == k]))}
                        for k in sorted(set(kind))}}
    with open(out / "calibration.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def fig(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = summarise(out)
    lv = res["levels"]
    x = [l["noise"] for l in lv]
    f, axes = plt.subplots(1, 5, figsize=(21.5, 3.9), constrained_layout=True)
    for arm in [a for a in ARMS if a in res["pooled"]]:
        lab = {"raw": "raw flow", "posterior": "reported posterior",
               "posterior_count": "reported posterior, pole counts matched to the classifier"}[arm]
        axes[0].plot(x, [l[arm]["cov50"] for l in lv], "o-", color=COL[arm], lw=1.2, ms=4, label=lab)
        axes[1].plot(x, [l[arm]["cov90"] for l in lv], "o-", color=COL[arm], lw=1.2, ms=4)
        axes[2].plot(x, [l[arm]["fit_rank"] for l in lv], "o-", color=COL[arm], lw=1.2, ms=4)
        axes[3].plot(x, [l[arm]["weight_error"] for l in lv], "o-", color=COL[arm], lw=1.2, ms=4)
    for ax, nom, t in ((axes[0], 0.5, "50% interval coverage"), (axes[1], 0.9, "90% interval coverage"),
                       (axes[2], 0.5, "data-fit rank (0.5 = calibrated)")):
        ax.axhline(nom, color=MUTED, ls="--", lw=0.9)
        ax.set_ylim(0, 1.02)
        ax.set_title(t, fontsize=9.5)
    axes[3].set_yscale("log")
    axes[3].set_title("window-weight error of the posterior mean (median)", fontsize=9.5)
    bb = res["by_beta"]
    xb = [float(k) for k in bb]
    shown = [a for a in ("posterior", "posterior_count") if a in res["pooled"]]
    for key, mk, nom in (("cov50", "o", 0.5), ("cov90", "s", 0.9)):
        for arm in shown:
            axes[4].errorbar(xb, [bb[k][arm][key] for k in bb], yerr=[bb[k][arm][key + "_se"] for k in bb],
                             color=COL[arm], marker=mk, ms=4, lw=1.2, capsize=2,
                             label=f"{int(nom * 100)}% interval" if arm == shown[0] else None)
        axes[4].axhline(nom, color=MUTED, ls="--", lw=0.9)
    axes[4].set_ylim(0, 1.02)
    axes[4].set_xscale("log")
    axes[4].set_xlabel("beta")
    axes[4].set_title("reported posterior: coverage against beta", fontsize=9.5)
    axes[4].legend(fontsize=8, frameon=False, loc="lower left")
    for ax in axes:
        if ax is not axes[4]:
            ax.set_xscale("log")
            ax.set_xlabel("relative noise level (beta cycles 1, 2, 4, 8, 12, 20)")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    p = res["pooled"]["posterior"]
    extra = ""
    if "posterior_count" in res["pooled"]:
        pc = res["pooled"]["posterior_count"]
        extra = f"; with pole counts matched {pc['cov50']:.2f} / {pc['cov90']:.2f}"
    f.suptitle(f"Calibration on the step-3 prior ({p['truths']} held-out truths, {res['design']['samples']} samples each): reported "
               f"posterior covers {p['cov50']:.2f} / {p['cov90']:.2f} (nominal 0.50 / 0.90), data-fit rank "
               f"{p['fit_rank_mean']:.2f}{extra}", fontsize=10.5)
    f.savefig(out / "calibration.png", dpi=150)
    print(json.dumps(res["pooled"], indent=1))
    for name in ("by_noise_band", "by_beta"):
        for k, g in res[name].items():
            for arm in shown:
                q = g[arm]
                tail = (f"accepted {g['accepted']:.2f}" if arm == "posterior" else
                        f"ESS {g['count']['ess']:.0f}, count mass covered {g['count']['covered']:.2f}, "
                        f"p(true count) {g['count']['p_true']:.2f}")
                print(f"{name:14s} {k:>12s} {arm:15s}: {q['truths']:4d} truths  coverage {q['cov50']:.3f}±"
                      f"{q['cov50_se']:.3f} / {q['cov90']:.3f}±{q['cov90_se']:.3f}  fit rank {q['fit_rank']:.2f}  "
                      f"error {q['weight_error']:.4f}  {tail}")
    print(f"-> {out}/calibration.json + .png")


if __name__ == "__main__":
    if sys.argv[1] == "fig":
        fig(Path(sys.argv[2]))
    elif sys.argv[1] == "recount":
        recount(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        o = Path(sys.argv[3])
        o.mkdir(parents=True, exist_ok=True)
        run(sys.argv[1], sys.argv[2], o, int(sys.argv[4]) if len(sys.argv) > 4 else 0,
            int(sys.argv[5]) if len(sys.argv) > 5 else 1,
            float(sys.argv[6]) if len(sys.argv) > 6 else 10.0,
            float(sys.argv[7]) if len(sys.argv) > 7 else None,
            int(sys.argv[8]) if len(sys.argv) > 8 else 1,
            sys.argv[9] if len(sys.argv) > 9 else None)
