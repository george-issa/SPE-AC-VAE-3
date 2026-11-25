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
        out/step3/cal_F3_6k [shard nshards [tether [base_scale [reps]]]]
    python scripts/step3/calibration.py fig out/step3/cal_F3_6k

`tether` is `lam` of the refinement: how firmly each sample is held at the
flow's own proposal while it is fitted (default 10, the package default).
`reps` repeats the noise grid with beta and dtau shifted against it, so that
every noise level is seen at `reps` temperatures (default 1: 288 truths;
4 gives 1152, enough to judge coverage per noise band and per beta).
"""
import json
import math
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
M, S = 24, 32
BANDS = ((1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1.0))      # noise bands of the shipping gate
COL = {"raw": "#2a78d6", "posterior": "#eb6834"}
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


def run(flow_path, bank, out, shard, nshards, lam=10.0, base_scale=None, reps=1):
    from spevae.data.pole_prior import PolePriorGenerator
    from spevae.models.flow import load_flow
    from spevae.physics import PoleGreens
    from spevae.physics.poles import PoleSet
    from spevae.posterior import DEFAULT_WINDOWS, continue_posterior

    torch.set_num_threads(4)
    flow = load_flow(flow_path).eval()
    if base_scale is not None:
        flow.base_scale = float(base_scale)
    rows = []
    path = out / f"calibration.shard{shard}.json"
    for j in range(int(reps) * len(NOISES)):
        if j % nshards != shard:
            continue
        r_, i_ = divmod(j, len(NOISES))            # repetition 0 is the original 12-cell design
        nz = NOISES[i_]
        beta = BETAS[(i_ + r_) % len(BETAS)]
        i = j                                      # seeds below are per cell
        gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive", smooth_bank=bank,
                                 p_smooth=0.35, p_exact=0.15, beta_choices=(beta,),
                                 noise_scale_range=(nz, nz),
                                 dtau_choices=(DTAUS[(i_ + 2 * r_) % len(DTAUS)],))
        with gen.fixed_scalars(31000 + i):
            b = gen.batch(M, torch.Generator().manual_seed(32000 + i))
        L = b.x.numel()
        nodes = (b.quad.nodes.double() / beta).numpy()
        wq = b.quad.weights.double().numpy()
        A_true = b.A_true.double().numpy()                        # exact, dimensionless, on nodes
        tot = (A_true * wq).sum(-1)
        w_true = np.stack([(A_true * wq * ((nodes >= lo) & (nodes < hi))).sum(-1) / tot
                           for lo, hi in DEFAULT_WINDOWS], -1)    # (M, 7)
        d = ((b.G_clean - b.G) / b.sigma) @ b.inv_sqrt_R.T
        c_true = (d * d).mean(-1).numpy()
        fwd = PoleGreens(b.quad)
        t0 = time.time()
        rec = {"noise": nz, "beta": beta, "n_tau": int(L), "rep": r_, "reps": int(reps),
               "family": list(b.family), "w_true": w_true.tolist(),
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
                                     bounds=False, lam=lam,
                                     generator=torch.Generator().manual_seed(900 + 97 * i + m))
            dim = PoleSet(rep.poles.eps * beta, rep.poles.gamma * beta, rep.poles.a, rep.poles.b)
            rec["arms"]["posterior"].append({
                "w": window_weights_poles(dim, beta, DEFAULT_WINDOWS).tolist(), "chi2": rep.chi2.tolist(),
                "accepted": rep.n_accepted, "kept_raw": rep.n_kept_raw,
                "born": rep.diagnostics.get("samples_given_a_pole", 0), "warnings": len(rep.warnings)})
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
    cells = []
    for r in rows:
        wt, ct = np.array(r["w_true"]), np.array(r["chi2_truth"])
        c = {"noise": r["noise"], "beta": r["beta"], "n_tau": r["n_tau"],
             "kind": [f.split(":")[0] for f in r["family"]]}
        for arm in ("raw", "posterior"):
            pit, fit, err = [], [], []
            for m, a in enumerate(r["arms"][arm]):
                w, ch = np.array(a["w"]), np.array(a["chi2"])
                n = len(w)
                pit.append(((w < wt[m]).sum(0) + rng.random(wt.shape[1])) / (n + 1))
                fit.append(((ch < ct[m]).sum() + rng.random()) / (n + 1))
                err.append(np.abs(w.mean(0) - wt[m]).sum())
            c[arm] = (np.array(pit), np.array(fit), np.array(err))
        c["accepted"] = float(np.mean([a["accepted"] for a in r["arms"]["posterior"]]) / S)
        c["born"] = float(np.mean([a["born"] for a in r["arms"]["posterior"]]) / S)
        cells.append(c)

    def group(sel):
        g = {arm: _stats(*(np.concatenate([c[arm][k] for c in sel]) for k in range(3)))
             for arm in ("raw", "posterior")}
        g["accepted"] = float(np.mean([c["accepted"] for c in sel]))
        g["born"] = float(np.mean([c["born"] for c in sel]))
        return g

    res = {"design": {"reps": reps, "truths_per_cell": M, "samples": S}, "levels": [], "cells": [],
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
    for arm in ("raw", "posterior"):
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
    for arm in ("raw", "posterior"):
        lab = "raw flow" if arm == "raw" else "reported posterior"
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
    for key, mk, nom in (("cov50", "o", 0.5), ("cov90", "s", 0.9)):
        axes[4].errorbar(xb, [bb[k]["posterior"][key] for k in bb],
                         yerr=[bb[k]["posterior"][key + "_se"] for k in bb], color=COL["posterior"],
                         marker=mk, ms=4, lw=1.2, capsize=2, label=f"{int(nom * 100)}% interval")
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
    f.suptitle(f"Calibration on the step-3 prior ({p['truths']} held-out truths, {S} samples each): reported "
               f"posterior covers {p['cov50']:.2f} / {p['cov90']:.2f} (nominal 0.50 / 0.90), data-fit rank "
               f"{p['fit_rank_mean']:.2f}", fontsize=10.5)
    f.savefig(out / "calibration.png", dpi=150)
    print(json.dumps(res["pooled"], indent=1))
    for name in ("by_noise_band", "by_beta"):
        for k, g in res[name].items():
            q = g["posterior"]
            print(f"{name:14s} {k:>12s}: {q['truths']:4d} truths  coverage {q['cov50']:.3f}±{q['cov50_se']:.3f} / "
                  f"{q['cov90']:.3f}±{q['cov90_se']:.3f}  fit rank {q['fit_rank']:.2f}  accepted {g['accepted']:.2f}")
    print(f"-> {out}/calibration.json + .png")


if __name__ == "__main__":
    if sys.argv[1] == "fig":
        fig(Path(sys.argv[2]))
    else:
        o = Path(sys.argv[3])
        o.mkdir(parents=True, exist_ok=True)
        run(sys.argv[1], sys.argv[2], o, int(sys.argv[4]) if len(sys.argv) > 4 else 0,
            int(sys.argv[5]) if len(sys.argv) > 5 else 1,
            float(sys.argv[6]) if len(sys.argv) > 6 else 10.0,
            float(sys.argv[7]) if len(sys.argv) > 7 else None,
            int(sys.argv[8]) if len(sys.argv) > 8 else 1)
