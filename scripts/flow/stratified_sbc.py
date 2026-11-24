"""Calibration of the refined flow, stratified by noise level.

`judge_str.py final` draws its "full training distribution" set as three
batches, and the prior generator draws ONE noise model and ONE grid per batch
-- so that test visits three noise settings.  Here every noise level gets its
own batch of held-out prior draws (grid spacing cycled, correlation length /
nugget / kernel drawn per level).

Coverage is read from randomised PIT values u = (rank + U) / (S + 1), which are
exactly uniform for a calibrated posterior at any sample count S -- unlike the
quantile-interval estimator of judge_str.metrics, whose calibrated value is
(S-1)/(S+1) times nominal (0.44 / 0.81 at S = 16, 0.47 / 0.85 at S = 32).

Two designs:

  small   9 noise levels x 16 truths x 16 samples.  Arms: raw flow / str (the
          Adam refinement banked 2026-11-23) / fission.  The 2026-11-24 review.
  big     40 noise levels x 25 truths x 32 samples = 1000 truths.  Arms: raw /
          lm (refine + redraw with the least-squares solver, the default since
          2026-11-24) / gated (a raw sample that already fits the measured data
          within noise is kept as drawn; the rest are refined) / accept (`lm`,
          then samples whose chi^2 against the measured data is implausible for
          a posterior draw -- above 1 + 3 sqrt(2/L) per point -- are rejected;
          they are overwhelmingly the ones the flow gave too few poles).

The `fission` arm conditions the flow on a deliberately noisier copy of the
data and refines against the independent remainder (xi ~ N(0, C), G1 = G + c xi,
G2 = G - xi / c, independent given the truth), so the data enter exactly once.
Measured: no better than `str` anywhere.

    python scripts/flow/stratified_sbc.py run out/flow/F_60k.pt out/flow/review_20261124 big 0 3
    python scripts/flow/stratified_sbc.py fig out/flow/review_20261124 big
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/flow")

import numpy as np
import torch

BETA = 8.0
DTAUS = [0.125, 0.05, 0.25, 0.1, 0.2]
SIGMA_COARSE = 3e-2
COL = {"raw": "#2a78d6", "str": "#1baf7a", "fission": "#b0489c", "lm": "#eb6834",
       "gated": "#7a5cc7", "accept": "#0f8a8a"}
LAB = {"raw": "raw flow", "str": "refine + redraw (Adam)", "fission": "fission + refine",
       "lm": "refine + redraw (least squares)", "gated": "refine only what does not fit",
       "accept": "refine + redraw + accept/reject"}
INK, MUTED, GRID = "#1f1e1c", "#52514e", "#d9d8d4"

# the `small` design; set_design("big") switches these
NOISES = [3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.5e-2, 3e-2]
M, S = 16, 16
ARMS = ("raw", "str", "fission")
PREFIX = "stratified_sbc"
SEED0 = 7000                    # generator scalars SEED0 + i, batch seed SEED0 + 100 + i
NBINS = 9                       # noise bins in the figure


def set_design(name):
    global NOISES, M, S, ARMS, PREFIX, SEED0, NBINS
    if name == "big":
        NOISES = np.exp(np.linspace(math.log(3e-5), math.log(3e-2), 40)).tolist()
        M, S = 25, 32
        ARMS = ("raw", "lm", "gated", "accept")
        PREFIX = "stratified_lm"
        SEED0 = 9000
        NBINS = 8
    elif name != "small":
        raise SystemExit(f"unknown design {name!r} (small | big)")


def batch_for(i):
    from spevae.data.pole_prior import PolePriorGenerator
    nz = NOISES[i]
    gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive",
                             beta_range=(BETA, BETA), dtau_choices=(DTAUS[i % len(DTAUS)],),
                             noise_scale_range=(nz, nz))
    with gen.fixed_scalars(SEED0 + i):
        return gen.batch(M, torch.Generator().manual_seed(SEED0 + 100 + i))


def run(flow_path, out, shard, nshards):
    import judge_flow as J
    import judge_str as JS
    from spevae.models.constraints import spectral_norm_from_greens
    from spevae.models.flow import load_flow, refine_samples
    from spevae.physics.poles import PoleSet

    torch.set_num_threads(4)
    flow = load_flow(flow_path).eval()
    path = out / f"{PREFIX}.shard{shard}.json"
    rows = json.load(open(path)) if path.exists() else []      # resume: finished rows are kept
    done = {(r["noise"], r["arm"]) for r in rows}
    for i, nz in enumerate(NOISES):
        if i % nshards != shard or all((nz, a) in done for a in ARMS):
            continue
        refined = None                                         # `gated` reuses `lm`'s solve
        b = batch_for(i)
        truth = PoleSet(b.poles_true["eps"], b.poles_true["gamma"], b.poles_true["a"],
                        b.poles_true["b"])
        At = J.A_phys(truth)
        st = J.stats_of(At)
        Gr, sr = b.G.repeat_interleave(S, 0), b.sigma.repeat_interleave(S, 0)
        L = b.x.numel()
        for arm in ARMS:
            if (nz, arm) in done:
                continue
            t0 = time.time()
            extra = {}
            g = torch.Generator().manual_seed(40 + i)
            c2f = (SIGMA_COARSE / nz) ** 2 - 1.0
            if arm == "fission" and c2f > 1e-9:
                c = math.sqrt(c2f)
                R_half = torch.linalg.inv(b.inv_sqrt_R)
                xi = (torch.randn(b.G.shape, generator=g) @ R_half.T) * b.sigma
                G1, s1 = b.G + c * xi, b.sigma * math.sqrt(1.0 + c2f)
                G2, s2 = b.G - xi / c, b.sigma * math.sqrt(1.0 + 1.0 / c2f)
                z, _ = flow.sample_coords(G1, b.x, BETA, sigma=s1, n_samples=S, generator=g)
                # the scale is a nuisance read off the original data, as in `str`
                norm = spectral_norm_from_greens(b.G, True).repeat_interleave(S, 0)
                ps, _ = refine_samples(flow, z, norm, BETA, b.x, G2.repeat_interleave(S, 0),
                                       s2.repeat_interleave(S, 0), b.inv_sqrt_R, 1.0,
                                       perturb=True, generator=g, **JS.REFINE)
            else:
                z, norm = flow.sample_coords(b.G, b.x, BETA, sigma=b.sigma, n_samples=S,
                                             generator=g)
                raw = flow.coords.decode(z, norm=norm, nu_max=flow.nu_max(BETA))
                if arm in ("str", "lm", "gated", "accept"):
                    if arm == "str" or refined is None:
                        ps, _ = refine_samples(flow, z, norm, BETA, b.x, Gr, sr, b.inv_sqrt_R,
                                               1.0, perturb=True, generator=g,
                                               **(JS.REFINE if arm == "str" else JS.REFINE_LM))
                        if arm != "str":
                            refined = ps
                    ps = ps if arm == "str" else refined
                    if arm == "gated":
                        # a raw sample whose chi^2 against the measured data is inside the
                        # noise band (mean 1, sd sqrt(2/L) per point; +2 sd) is kept as drawn
                        fits = J.chi2(raw, b.quad, b.x, Gr, sr, b.inv_sqrt_R) \
                            <= 1.0 + 2.0 * math.sqrt(2.0 / L)
                        ps = PoleSet(*(torch.where(fits.unsqueeze(-1),
                                                   getattr(raw, k).to(getattr(ps, k).dtype),
                                                   getattr(ps, k))
                                       for k in ("eps", "gamma", "a", "b")))
                        extra["frac_raw_kept"] = float(fits.float().mean())
                else:
                    ps = raw
            m, _ = JS.metrics(ps, b, S, truth)
            A = J.A_phys(ps)
            ss = J.stats_of(A).reshape(M, S, -1)
            c2 = J.chi2(ps, b.quad, b.x, Gr, sr, b.inv_sqrt_R).reshape(M, S)
            acc = np.ones((M, S), bool)
            if arm == "accept":
                acc = (c2 <= 1.0 + 3.0 * math.sqrt(2.0 / L)).numpy()
                acc[~acc.any(1)] = True                                   # nothing fits: keep all, flagged
                extra.update(accepted=acc.tolist(), frac_accepted=float(acc.mean()),
                             truths_with_no_accepted_sample=int(
                                 (c2 > 1.0 + 3.0 * math.sqrt(2.0 / L)).all(1).sum()))
            n_acc = acc.sum(1)[:, None]                                   # (M, 1)
            less = ((ss < st[:, None, :]) & acc[:, :, None]).sum(1)
            ties = ((ss == st[:, None, :]) & acc[:, :, None]).sum(1)
            U = np.random.default_rng(900 + i).random(less.shape)
            pit = (less + U * (ties + 1)) / (n_acc + 1)                   # (M, 9), uniform if calibrated
            Am = (A.reshape(M, S, -1).numpy() * acc[:, :, None]).sum(1) / n_acc
            w1 = [J.w1(Am[k], At[k].numpy()) for k in range(M)]
            n_pres = (ps.a.reshape(M, S, -1) > 0).sum(-1)
            row = dict(m, arm=arm, noise=nz, dtau=DTAUS[i % len(DTAUS)], n_tau=int(L),
                       xi=b.meta["noise"].xi, nugget=b.meta["noise"].nugget,
                       correlation=b.meta["correlation"], pit=pit.tolist(), w1_per_truth=w1,
                       chi2_per_sample=c2.tolist(), n_present=n_pres.tolist(),
                       n_true=b.poles_true["n_true"].tolist(), seconds=time.time() - t0, **extra)
            rows.append(row)
            print(f"noise {nz:7.1e} L={L:3d} {arm:8s} chi2 {m['chi2_median']:9.3g}  "
                  f"PIT cov50 {np.mean((pit > .25) & (pit < .75)):.2f} cov90 "
                  f"{np.mean((pit > .05) & (pit < .95)):.2f}  W1 {np.median(w1):.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            with open(path, "w") as fh:
                json.dump(rows, fh)


def truth_chi2():
    """chi^2 per point of each TRUTH against its own noisy data, per noise level
    (the batches are rebuilt from their seeds; no sampling)."""
    import judge_flow as J
    from spevae.physics.poles import PoleSet
    out = {}
    for i, nz in enumerate(NOISES):
        b = batch_for(i)
        truth = PoleSet(b.poles_true["eps"], b.poles_true["gamma"], b.poles_true["a"],
                        b.poles_true["b"])
        out[nz] = J.chi2(truth, b.quad, b.x, b.G, b.sigma, b.inv_sqrt_R).numpy()
    return out


def fig(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for p in sorted(out.glob(f"{PREFIX}.shard*.json")):
        rows += json.load(open(p))
    noises = sorted({r["noise"] for r in rows})
    get = lambda arm, nz: next(r for r in rows if r["arm"] == arm and r["noise"] == nz)
    ARMS = tuple(a for a in globals()["ARMS"]
                 if all(any(r["arm"] == a and r["noise"] == nz for r in rows) for nz in noises))
    hist_arms = [a for a in ("raw", "str", "lm", "accept", "fission", "gated") if a in ARMS][:3]

    def c2_kept(arm, nz):
        """chi^2 of the samples an arm actually reports (accepted ones for `accept`)."""
        r = get(arm, nz)
        c2 = np.array(r["chi2_per_sample"])
        return c2[np.array(r["accepted"], bool)] if "accepted" in r else c2.ravel()
    bins = np.array_split(np.arange(len(noises)), NBINS)             # noise levels per figure bin
    xb = [float(np.exp(np.mean(np.log([noises[k] for k in b])))) for b in bins]
    # data-fit rank (Modrak et al. 2023): for a calibrated posterior the truth and the samples
    # are exchangeable given the data, so the truth's chi^2 against the data must rank uniformly
    # among the samples' chi^2.  A posterior that ignores the data's precision (samples fit far
    # worse than the truth) or fits the noise (samples fit better) fails here even when
    # parameter-space coverage looks fine.
    tc = truth_chi2()
    rng = np.random.default_rng(5)
    fit_pit = {}
    for arm in ARMS:
        u = []
        for nz in noises:
            c2 = np.array(get(arm, nz)["chi2_per_sample"])               # (M, S)
            acc = np.array(get(arm, nz).get("accepted", np.ones_like(c2, bool)), bool)
            less = ((c2 < tc[nz][:, None]) & acc).sum(1)
            u.append((less + rng.random(len(less))) / (acc.sum(1) + 1))
        fit_pit[arm] = np.array(u)                                       # (levels, M)

    summ = {"design": {"M": M, "S": S, "levels": len(noises), "noise_range": [noises[0], noises[-1]],
                       "truths": M * len(noises), "sigma_coarse": SIGMA_COARSE,
                       "estimator_nominal_at_S": [0.5 * (S - 1) / (S + 1), 0.9 * (S - 1) / (S + 1)]},
            "per_bin": [], "pooled": {}}
    fig_, axes = plt.subplots(2, 4, figsize=(17, 7.2), constrained_layout=True)
    ax50, ax90, axc, axw = axes[0]
    for a_i, arm in enumerate(ARMS):
        off = 1.0 + 0.06 * (a_i - (len(ARMS) - 1) / 2)
        c50, c90, e50, e90, chi, w1 = [], [], [], [], [], []
        for b, x0 in zip(bins, xb):
            pit = np.concatenate([np.array(get(arm, noises[k])["pit"]) for k in b], 0)   # (truths, 9)
            in50, in90 = (pit > .25) & (pit < .75), (pit > .05) & (pit < .95)
            # error bars from per-truth means, so correlated statistics of one truth count once
            c50.append(in50.mean()); e50.append(in50.mean(1).std() / math.sqrt(len(pit)))
            c90.append(in90.mean()); e90.append(in90.mean(1).std() / math.sqrt(len(pit)))
            chi.append(float(np.median(np.concatenate([c2_kept(arm, noises[k]) for k in b]))))
            w1.append(float(np.median(np.concatenate(
                [get(arm, noises[k])["w1_per_truth"] for k in b]))))
            summ["per_bin"].append({"arm": arm, "noise": x0, "truths": int(len(pit)),
                                    "pit_cov50": float(c50[-1]), "pit_cov90": float(c90[-1]),
                                    "chi2_median": chi[-1], "w1_median": w1[-1]})
        xs = np.array(xb) * off
        ax50.errorbar(xs, c50, yerr=e50, fmt="o-", color=COL[arm], lw=1.2, ms=4, capsize=2, label=LAB[arm])
        ax90.errorbar(xs, c90, yerr=e90, fmt="o-", color=COL[arm], lw=1.2, ms=4, capsize=2)
        axc.plot(xb, chi, "o-", color=COL[arm], lw=1.2, ms=4)
        axw.plot(xb, w1, "o-", color=COL[arm], lw=1.2, ms=4)
        pit_all = np.concatenate([np.array(get(arm, nz)["pit"]) for nz in noises], 0)   # (n, 9)
        n = len(pit_all)
        ks = [float(np.max(np.abs(np.sort(pit_all[:, j]) - (np.arange(1, n + 1) - 0.5) / n)) + 0.5 / n)
              for j in range(pit_all.shape[1])]
        in50, in90 = (pit_all > .25) & (pit_all < .75), (pit_all > .05) & (pit_all < .95)
        summ["pooled"][arm] = {
            "n_truths": n, "ks_max": max(ks), "ks_crit_5pct": 1.36 / math.sqrt(n),
            "ks_crit_5pct_bonferroni9": 1.63 / math.sqrt(n),
            "pit_cov50": float(in50.mean()), "pit_cov50_se": float(in50.mean(1).std() / math.sqrt(n)),
            "pit_cov90": float(in90.mean()), "pit_cov90_se": float(in90.mean(1).std() / math.sqrt(n)),
            "chi2_median": float(np.median(np.concatenate([c2_kept(arm, nz) for nz in noises]))),
            "frac_samples_chi2_above_2": float(np.mean(np.concatenate(
                [c2_kept(arm, nz) for nz in noises]) > 2.0)),
            "w1_median": float(np.median(np.concatenate(
                [get(arm, nz)["w1_per_truth"] for nz in noises]))),
            "seconds_total": float(sum(get(arm, nz)["seconds"] for nz in noises))}
        if arm != "raw":
            summ["pooled"][arm]["paired_w1_ratio_to_raw_per_bin"] = {
                f"{x0:.2g}": float(np.median(np.concatenate(
                    [np.array(get(arm, noises[k])["w1_per_truth"]) /
                     np.array(get("raw", noises[k])["w1_per_truth"]) for k in b])))
                for b, x0 in zip(bins, xb)}
        if arm == "accept":
            summ["pooled"][arm]["frac_accepted_per_bin"] = {
                f"{x0:.2g}": float(np.mean([get(arm, noises[k])["frac_accepted"] for k in b]))
                for b, x0 in zip(bins, xb)}
            summ["pooled"][arm]["truths_with_no_accepted_sample"] = int(
                sum(get(arm, nz)["truths_with_no_accepted_sample"] for nz in noises))
        if arm == "gated":
            summ["pooled"][arm]["frac_raw_kept_per_bin"] = {
                f"{x0:.2g}": float(np.mean([get(arm, noises[k])["frac_raw_kept"] for k in b]))
                for b, x0 in zip(bins, xb)}
    for ax, nom, t in ((ax50, 0.5, "50% interval coverage (PIT)"), (ax90, 0.9, "90% interval coverage (PIT)")):
        ax.axhline(nom, color=MUTED, ls="--", lw=0.9)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"{t}\ndashed = calibrated; above = too wide, below = overconfident", fontsize=9)
    axc.axhline(1.0, color=MUTED, ls="--", lw=0.9)
    axc.set_yscale("log")
    axc.set_title("median sample $\\chi^2$ per point vs the measured data\n(the true spectrum scores 1)", fontsize=9)
    axw.set_yscale("log")
    axw.set_title("W$_1$ of the posterior mean (median over truths)\nlower is better", fontsize=9)
    ax50.legend(fontsize=7.5, frameon=False, loc="lower left")
    for ax in axes[0]:
        ax.set_xscale("log")
        ax.set_xlabel("relative noise level")
    # bottom row: pooled PIT histograms per arm + the data-fit rank
    for ax, arm in zip(axes[1, :3], hist_arms):
        pit_all = np.concatenate([np.array(get(arm, nz)["pit"]) for nz in noises], 0).ravel()
        ax.hist(pit_all, bins=10, range=(0, 1), color=COL[arm], rwidth=0.9)
        ax.axhline(len(pit_all) / 10, color=MUTED, ls="--", lw=0.9)
        p = summ["pooled"][arm]
        ax.set_title(f"{LAB[arm]}: PIT of the truth, all levels\n"
                     f"flat = calibrated, hump = too wide;  max KS {p['ks_max']:.3f} "
                     f"(5% crit {p['ks_crit_5pct_bonferroni9']:.3f})", fontsize=8.5)
        ax.set_xlabel("PIT")
    ax = axes[1, 3]
    for a_i, arm in enumerate(ARMS):
        # mean data-fit PIT per noise bin (fraction of samples that fit better than the truth):
        # 0.5 = samples fit the data as well as the truth does; -> 0 = every sample fits worse
        # than the truth; -> 1 = every sample fits better than the truth (noise fitted)
        mu = np.array([fit_pit[arm][b].mean() for b in bins])
        se = np.array([fit_pit[arm][b].std() / math.sqrt(fit_pit[arm][b].size) for b in bins])
        off = 1.0 + 0.06 * (a_i - (len(ARMS) - 1) / 2)
        ax.errorbar(np.array(xb) * off, mu, yerr=se, fmt="o-", color=COL[arm], lw=1.2, ms=4,
                    capsize=2, label=LAB[arm])
        u = fit_pit[arm].ravel()
        n = len(u)
        summ["pooled"][arm]["fit_pit_mean"] = float(u.mean())
        summ["pooled"][arm]["fit_pit_ks"] = float(
            np.max(np.abs(np.sort(u) - (np.arange(1, n + 1) - 0.5) / n)) + 0.5 / n)
        summ["pooled"][arm]["fit_pit_mean_per_bin"] = {f"{x0:.2g}": float(v) for x0, v in zip(xb, mu)}
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("relative noise level")
    ax.set_title("data-fit rank of the truth's $\\chi^2$ among the samples'\n"
                 "0.5 = calibrated; 0 = samples fit worse; 1 = samples fit noise", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False)
    for ax in axes.flat:
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig_.suptitle(f"Refined flow (60k) calibration by noise level: {len(noises)} noise levels x {M} "
                  f"held-out prior draws x {S} samples ({M * len(noises)} truths), one grid / "
                  "correlation draw per level", fontsize=11)
    fig_.savefig(out / f"{PREFIX}.png", dpi=150)
    with open(out / f"{PREFIX}.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ["pooled"], indent=1))
    print(f"-> {out}/{PREFIX}.json + .png")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        o = Path(sys.argv[3])
        o.mkdir(parents=True, exist_ok=True)
        set_design(sys.argv[4])
        run(sys.argv[2], o, int(sys.argv[5]), int(sys.argv[6]))
    else:
        set_design(sys.argv[3] if len(sys.argv) > 3 else "small")
        fig(Path(sys.argv[2]))
