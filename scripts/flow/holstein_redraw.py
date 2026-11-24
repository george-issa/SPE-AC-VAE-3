"""Holstein cell: does the noise source of the per-sample redraw matter, and do
all refined samples reach the data?

`refine_samples(perturb=True)` redraws each sample's target from the WHITENER's
covariance (sigma^2 R_shrunk).  On synthetic data that is the true noise.  On
the Holstein cell the whitener is a shrinkage estimate whose perfect-fit chi^2
is 0.175, not 1: it overstates the bins' own scatter in most whitened
directions (last panel).  This script refines the same 200 flow samples three
ways:

  none        no redraw (every sample fits the measured mean)
  whitener    redraw from sigma^2 R_shrunk            (the current default)
  bootstrap   redraw = mean of the bins resampled with replacement

and reports, per variant, chi^2 / target of the samples against the measured
mean, the width of the 90% band of A(w), and an acceptance check: chi^2 of
each sample against the target it was fitted to (none / bootstrap), the
fraction left above 2x target, and the pole counts of accepted vs rejected
samples (the refinement keeps each sample's pole count fixed).

Measured 2026-11-24 on F_60k: the redraw source moves the sample chi^2 a
little (1.64 -> 1.46 x target) and the bands not at all (width 0.167 vs 0.166);
9-12% of samples cannot reach the data, and those carry a median of 6-7 poles
against 12 for the rest.

A third argument `lm` runs the same comparison with the least-squares solver
(`judge_str.REFINE_LM`) and writes `holstein_redraw_lm.*`.

    python scripts/flow/holstein_redraw.py out/flow/F_60k.pt out/flow/review_20261124 [lm]
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/flow")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import judge_flow as J
import judge_str as JS
from spevae.evaluation import cases as C
from spevae.models.flow import load_flow, refine_samples
from spevae.physics import PoleGreens, SpectralQuadrature

S = 200
COL = {"none": "#eda100", "whitener": "#1baf7a", "bootstrap": "#2a78d6"}
INK, MUTED, GRID = "#1f1e1c", "#52514e", "#d9d8d4"


def main():
    flow_path, out = sys.argv[1], Path(sys.argv[2])
    lm = len(sys.argv) > 3 and sys.argv[3] == "lm"
    refine_kw, stem = (JS.REFINE_LM, "holstein_redraw_lm") if lm else (JS.REFINE, "holstein_redraw")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    flow = load_flow(flow_path).eval()
    mg = C.holstein_cell()
    cov = mg.covariance(1e-10, method="structured")
    sl = cov["slices"]
    N = mg.n_bins
    x = mg.x.float()
    Gfull = mg.mean.float().unsqueeze(0)
    s_enc = (mg.sigma_encoder / math.sqrt(N)).float().unsqueeze(0)
    s_fit = (cov["sigma"] / math.sqrt(N)).float().unsqueeze(0)
    isr = cov["inv_sqrt_R"].float()
    quad = SpectralQuadrature.for_beta(mg.beta)

    def chi2_vs_mean(ps):
        G_rec = PoleGreens(quad)(ps, x).double()
        d = ((G_rec[:, sl] - mg.mean.double()[sl]) / (cov["sigma"].double() / math.sqrt(N))) \
            @ cov["inv_sqrt_R"].double().T
        return ((d * d).sum(-1) / cov["n_independent"]).numpy() / cov["chi2_target"]

    res, bands, kept = {"flow": flow_path, "samples": S, "chi2_target": cov["chi2_target"]}, {}, {}
    for name in ("none", "whitener", "bootstrap"):
        g = torch.Generator().manual_seed(3)                  # same flow samples every variant
        z, norm = flow.sample_coords(Gfull, x, mg.beta, sigma=s_enc, n_samples=S, generator=g)
        G_fit = Gfull[:, sl].expand(S, -1)
        if name == "bootstrap":
            gb = torch.Generator().manual_seed(11)
            idx = torch.randint(N, (S, N), generator=gb)
            G_fit = mg.bins[:, sl][idx].mean(1).float()       # (S, L) bootstrap means
        ps, c2_own = refine_samples(flow, z, norm, mg.beta, x[sl], G_fit, s_fit.expand(S, -1), isr,
                                    1.0, perturb=(name == "whitener"), generator=g, **refine_kw)
        c2 = chi2_vs_mean(ps)
        A = J.A_phys(ps, beta=mg.beta).numpy()
        q = np.quantile(A, [0.05, 0.5, 0.95], axis=0)
        w = q[2] - q[0]
        # acceptance check: chi^2 of each sample against the target IT was fitted to (the measured
        # mean, or its own bootstrap mean).  A converged fit scores at or below the target; a sample
        # still above 2x target could not reach the data from where the flow put it.  (Not available
        # for the whitener redraw, whose targets are drawn inside refine_samples.)
        if name != "whitener":
            own = c2_own.numpy() / cov["chi2_target"]
            keep = own <= 2.0
            qk = np.quantile(A[keep], [0.05, 0.5, 0.95], axis=0)
            wk = qk[2] - qk[0]
            kept[name] = qk
            acc = {"own_target_chi2_over_target_q05_q50_q95": np.quantile(own, [0.05, 0.5, 0.95]).tolist(),
                   "frac_rejected_own_chi2_above_2x_target": float(1.0 - keep.mean()),
                   "accepted_band90_width_mean_|w|<6": float(wk[np.abs(J.W) <= 6].mean()),
                   "accepted_band90_width_bands_2<|w|<5": float(wk[(np.abs(J.W) > 2) & (np.abs(J.W) < 5)].mean()),
                   "n_poles_accepted_median": float(np.median((ps.a.numpy() > 0).sum(-1)[keep])),
                   "n_poles_rejected_median": float(np.median((ps.a.numpy() > 0).sum(-1)[~keep]))
                   if (~keep).any() else None}
        else:
            acc = {}
        res[name] = {
            "chi2_over_target_q05_q50_q95": np.quantile(c2, [0.05, 0.5, 0.95]).tolist(),
            "frac_above_2x_target": float((c2 > 2.0).mean()),
            "band90_width_mean_|w|<6": float(w[np.abs(J.W) <= 6].mean()),
            "band90_width_gap_edge_0.5<|w|<1.5": float(w[(np.abs(J.W) > 0.5) & (np.abs(J.W) < 1.5)].mean()),
            "band90_width_bands_2<|w|<5": float(w[(np.abs(J.W) > 2) & (np.abs(J.W) < 5)].mean()),
            **acc,
        }
        bands[name] = q
        print(name, json.dumps(res[name]), flush=True)
    # the noise mismatch itself: whitened sample-correlation eigenvalues (1 = whitener matches bins)
    b = mg.bins[:, sl].double()
    Z = (b - b.mean(0, keepdim=True)) / cov["sigma"].double()
    Wd = cov["inv_sqrt_R"].double()
    ev = torch.linalg.eigvalsh(Wd @ ((Z.T @ Z) / N) @ Wd.T).flip(0).numpy()
    res["whitened_sample_eigenvalues"] = ev.tolist()
    res["solver"] = refine_kw["solver"]
    with open(out / f"{stem}.json", "w") as fh:
        json.dump(res, fh, indent=1)

    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1.25, 1.25, 1.0, 1.0]})
    core = np.abs(J.W) <= 6
    titles = {"none": "no redraw", "whitener": "redraw from the whitener (current default)",
              "bootstrap": "redraw = bootstrap of the bins"}
    for ax, name in zip(axes[:3], ("none", "whitener", "bootstrap")):
        q = bands[name]
        ax.fill_between(J.W[core], q[0][core], q[2][core], color=COL[name], alpha=0.28, lw=0,
                        label="90% band")
        ax.plot(J.W[core], q[1][core], color=COL[name], lw=1.3, label="median")
        if name in kept:
            for j in (0, 2):
                ax.plot(J.W[core], kept[name][j][core], color=INK, lw=0.7, ls=":",
                        label="90% band, accepted samples only" if j == 0 else None)
        if mg.reference is not None:
            w, A = (np.asarray(v, float) for v in mg.reference)
            m = np.abs(w) <= 6
            ax.plot(w[m], A[m], color=INK, lw=1.0, label="SmoQyDEAC (shipped)")
        r = res[name]["chi2_over_target_q05_q50_q95"]
        ax.set_title(f"{titles[name]}\n$\\chi^2$/target {r[1]:.2f} [{r[0]:.2f}, {r[2]:.2f}]", fontsize=8.5)
        ax.set_xlabel(r"$\omega$")
        ax.set_ylim(-0.02, 0.45)
    axes[0].legend(fontsize=7, frameon=False)
    ax = axes[3]
    for i, name in enumerate(("none", "whitener", "bootstrap")):
        r = res[name]["chi2_over_target_q05_q50_q95"]
        ax.plot([r[0], r[2]], [i, i], color=COL[name], lw=2.2)
        ax.plot(r[1], i, "o", color=COL[name], ms=7)
    ax.axvline(1.0, color=MUTED, ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["no redraw", "whitener", "bootstrap"], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("sample $\\chi^2$/target vs the measured mean\n(5-95% of 200 samples)", fontsize=8.5)
    ax = axes[4]
    ax.semilogy(np.arange(1, len(ev) + 1), ev, ".", color=INK, ms=4)
    ax.axhline(1.0, color=MUTED, ls="--", lw=0.9)
    ax.set_xlabel("whitened direction (sorted)")
    ax.set_title(f"bins' variance / whitener's variance\n(1 = whitener matches; mean {ev.mean():.3f} = target)",
                 fontsize=8.5)
    for a in axes:
        a.grid(color=GRID, lw=0.6)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    rej = res["bootstrap"]["frac_rejected_own_chi2_above_2x_target"]
    fig.suptitle(f"Holstein cell, 200 refined samples of the 60k flow ({refine_kw['solver']}): redraw noise source and sample "
                 f"$\\chi^2$ (whitener vs bootstrap: {res['whitener']['chi2_over_target_q05_q50_q95'][1]:.2f} vs "
                 f"{res['bootstrap']['chi2_over_target_q05_q50_q95'][1]:.2f} x target; mean 90% band width "
                 f"{res['whitener']['band90_width_mean_|w|<6']:.3f} vs {res['bootstrap']['band90_width_mean_|w|<6']:.3f}); "
                 f"{res['none']['frac_rejected_own_chi2_above_2x_target']:.1%}-{rej:.1%} of samples cannot reach the data "
                 f"(median {res['bootstrap']['n_poles_rejected_median']:.0f} poles vs "
                 f"{res['bootstrap']['n_poles_accepted_median']:.0f})", fontsize=10)
    fig.savefig(out / f"{stem}.png", dpi=150)
    print(f"-> {out}/{stem}.json + .png")


if __name__ == "__main__":
    main()
