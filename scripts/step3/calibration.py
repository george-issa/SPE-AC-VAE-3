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
        out/step3/cal_F3_6k [shard nshards]
    python scripts/step3/calibration.py fig out/step3/cal_F3_6k
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


def run(flow_path, bank, out, shard, nshards):
    from spevae.data.pole_prior import PolePriorGenerator
    from spevae.models.flow import load_flow
    from spevae.physics import PoleGreens
    from spevae.physics.poles import PoleSet
    from spevae.posterior import DEFAULT_WINDOWS, continue_posterior

    torch.set_num_threads(4)
    flow = load_flow(flow_path).eval()
    rows = []
    path = out / f"calibration.shard{shard}.json"
    for i, nz in enumerate(NOISES):
        if i % nshards != shard:
            continue
        beta = BETAS[i % len(BETAS)]
        gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive", smooth_bank=bank,
                                 p_smooth=0.35, p_exact=0.15, beta_choices=(beta,),
                                 noise_scale_range=(nz, nz), dtau_choices=(DTAUS[i % len(DTAUS)],))
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
        rec = {"noise": nz, "beta": beta, "n_tau": int(L), "family": list(b.family), "w_true": w_true.tolist(),
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
                                     bounds=False, generator=torch.Generator().manual_seed(900 + 97 * i + m))
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


def summarise(out):
    rows = []
    for p in sorted(out.glob("calibration.shard*.json")):
        rows += json.load(open(p))
    rows.sort(key=lambda r: r["noise"])
    rng = np.random.default_rng(3)
    res = {"levels": []}
    pooled = {a: {"pit": [], "fit": [], "err": [], "kind": []} for a in ("raw", "posterior")}
    for r in rows:
        wt, ct = np.array(r["w_true"]), np.array(r["chi2_truth"])
        lev = {"noise": r["noise"], "beta": r["beta"], "n_tau": r["n_tau"]}
        for arm in ("raw", "posterior"):
            pit, fit, err = [], [], []
            for m, a in enumerate(r["arms"][arm]):
                w, c = np.array(a["w"]), np.array(a["chi2"])
                n = len(w)
                pit.append(((w < wt[m]).sum(0) + rng.random(wt.shape[1])) / (n + 1))
                fit.append(((c < ct[m]).sum() + rng.random()) / (n + 1))
                err.append(np.abs(w.mean(0) - wt[m]).sum())
            pit, fit, err = np.array(pit), np.array(fit), np.array(err)
            lev[arm] = {"cov50": float(np.mean((pit > .25) & (pit < .75))),
                        "cov90": float(np.mean((pit > .05) & (pit < .95))),
                        "fit_rank": float(fit.mean()), "weight_error": float(np.median(err))}
            pooled[arm]["pit"].append(pit); pooled[arm]["fit"].append(fit); pooled[arm]["err"].append(err)
            pooled[arm]["kind"] += [f.split(":")[0] for f in r["family"]]
        lev["accepted"] = float(np.mean([a["accepted"] for a in r["arms"]["posterior"]]) / S)
        lev["born"] = float(np.mean([a["born"] for a in r["arms"]["posterior"]]) / S)
        res["levels"].append(lev)
    res["pooled"] = {}
    for arm in ("raw", "posterior"):
        pit = np.concatenate(pooled[arm]["pit"]); fit = np.concatenate(pooled[arm]["fit"])
        err = np.concatenate(pooled[arm]["err"]); kind = np.array(pooled[arm]["kind"])
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
    f, axes = plt.subplots(1, 4, figsize=(17, 3.9), constrained_layout=True)
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
    for ax in axes:
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
    print(f"-> {out}/calibration.json + .png")


if __name__ == "__main__":
    if sys.argv[1] == "fig":
        fig(Path(sys.argv[2]))
    else:
        o = Path(sys.argv[3])
        o.mkdir(parents=True, exist_ok=True)
        run(sys.argv[1], sys.argv[2], o, int(sys.argv[4]) if len(sys.argv) > 4 else 0,
            int(sys.argv[5]) if len(sys.argv) > 5 else 1)
