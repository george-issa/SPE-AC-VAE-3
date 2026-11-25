"""The Holstein DQMC cell through the posterior report, as a figure.

The cell has no known answer; the SmoQyDEAC spectrum shipped with the fixture
is a reference continuation, not the truth.  What the figure shows is what the
tool would hand a user: the band of A(omega), how well the samples fit the
bins, and the weight per frequency window with its interval next to what the
data alone allow.

    python scripts/step3/holstein_report.py out/step3/holstein_F3_60k_s1 \
        out/step3/F3_60k_s1.pt out/step2/data_arm/D24_60k.pt

Writes holstein_report.{json,png,txt} into the output directory.
"""
import json
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

WG = np.linspace(-15.0, 15.0, 30001)
DW = WG[1] - WG[0]
FLOW, VAE, ME, INK, PALE = "#2a78d6", "#eb6834", "#7d7c78", "#0b0b0b", "#d9d8d3"


def window_weight(A, lo, hi):
    return float(A[(WG >= lo) & (WG < hi)].sum() * DW)


def main():
    from spevae import load_model
    from spevae.benchmark import run_maxent
    from spevae.evaluation import cases as C
    from spevae.models.flow import load_flow
    from spevae.physics.poles import PoleSet
    from spevae.posterior import continue_posterior
    from spevae.refine import RefineConfig, refine_latent

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    flow, vae = load_flow(sys.argv[2]).eval(), load_model(sys.argv[3])
    torch.set_num_threads(6)
    mg = C.holstein_cell()
    beta, N = mg.beta, mg.n_bins
    cov = mg.covariance(1e-10, method="structured")
    sl = cov["slices"]
    rep = continue_posterior(flow, mg.mean, mg.x, beta, mg.sigma_encoder / math.sqrt(N),
                             cov["inv_sqrt_R"], bins=mg.bins[:, sl], n_samples=200,
                             chi2_target=cov["chi2_target"], x_fit=mg.x[sl], G_fit=mg.mean[sl],
                             sigma_fit=cov["sigma"] / math.sqrt(N), omega=WG,
                             generator=torch.Generator().manual_seed(0))
    text = rep.summary()
    print(text)
    (out / "holstein_report.txt").write_text(text + "\n")

    rr = refine_latent(vae, mg, RefineConfig(cov_method="structured", min_eigenvalue=1e-10, solver="lm"))
    nu = torch.tensor(WG, dtype=torch.float64)
    A_vae = PoleSet(*(getattr(rr.poles, k).double() for k in ("eps", "gamma", "a", "b"))).spectral(nu)[0].numpy()
    me = run_maxent(mg, min_eigenvalue=1e-10)
    A_me = np.interp(WG, me.omega, me.A, left=0.0, right=0.0)
    ref_w, ref_A = (t.double().numpy() for t in mg.reference)
    A_ref = np.interp(WG, ref_w, ref_A, left=0.0, right=0.0)

    cdf = lambda a: np.cumsum(a) * DW
    w1_ref = {k: float(np.abs(cdf(v) - cdf(A_ref)).sum() * DW)
              for k, v in (("flow", rep.band["mean"]), ("vae", A_vae), ("maxent", A_me))}
    core = np.abs(WG) <= 6
    wins = []
    for w in rep.windows:
        lo, hi = w["window"]
        wins.append({**w, "reference": window_weight(A_ref, lo, hi), "maxent": window_weight(A_me, lo, hi),
                     "vae": window_weight(A_vae, lo, hi)})
    rel = float(rep.diagnostics.get("relative_error_bar", float("nan")))
    res = {"flow": sys.argv[2], "beta": beta, "n_bins": N, "relative_error_bar": rel,
           "accepted": rep.n_accepted, "of": rep.n_samples, "kept_raw": rep.n_kept_raw,
           "chi2_over_target": {"median": float(np.median(rep.chi2) / rep.chi2_target),
                                "q05": float(np.quantile(rep.chi2, 0.05) / rep.chi2_target),
                                "q95": float(np.quantile(rep.chi2, 0.95) / rep.chi2_target)},
           "vae_chi2_over_target": float(rr.chi2_ratio),
           "band90_mean_width": float((rep.band["q95"] - rep.band["q05"])[core].mean()),
           "w1_to_reference": w1_ref, "warnings": rep.warnings, "windows": wins,
           "diagnostics": {k: v for k, v in rep.diagnostics.items()
                           if isinstance(v, (int, float, str, bool, dict, list))}}
    with open(out / "holstein_report.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.6, 0.8, 1.3]})
    m = np.abs(WG) <= 6
    st = slice(None, None, 4)
    a0.fill_between(WG[m][st], rep.band["q05"][m][st], rep.band["q95"][m][st], color=FLOW, alpha=0.25, lw=0,
                    label="flow posterior, 90% band")
    a0.plot(WG[m][st], rep.band["q50"][m][st], color=FLOW, lw=1.5, label="flow posterior, median")
    a0.plot(WG[m][st], A_vae[m][st], color=VAE, lw=1.0, label="VAE latent MAP")
    a0.plot(WG[m][st], A_me[m][st], color=ME, lw=1.0, label="MaxEnt")
    a0.plot(WG[m][st], A_ref[m][st], color=INK, lw=1.3, ls="--",
            label="SmoQyDEAC reference (another continuation, not the truth)")
    a0.set_xlabel(r"$\omega$")
    a0.set_ylabel(r"$A(\omega)$")
    a0.set_ylim(-0.03 * float(rep.band["q95"][m].max()), 1.15 * float(rep.band["q95"][m].max()))
    a0.set_title(f"spectrum; {rep.n_accepted} of {rep.n_samples} samples fit the bins", fontsize=9.5)
    a0.legend(fontsize=7.5, frameon=False, loc="upper left")

    r = rep.chi2 / rep.chi2_target
    a1.hist(r, bins=np.linspace(min(0.6, r.min()), max(1.6, r.max()), 28), color=FLOW, alpha=0.75)
    a1.axvline(1.0, color=INK, lw=1.2)
    a1.axvline(rr.chi2_ratio, color=VAE, lw=1.2)
    a1.set_xlabel(r"$\chi^2$ / value a perfect fit scores")
    a1.set_ylabel("accepted samples")
    a1.set_title(f"misfit of each sample (median {np.median(r):.2f});\nblack = perfect fit, orange = VAE",
                 fontsize=9.5)

    y = np.arange(len(wins))[::-1]
    for yy, w in zip(y, wins):
        a2.plot([w["lo"], w["hi"]], [yy, yy], color=PALE, lw=9, solid_capstyle="butt")
        a2.plot([w["q05"], w["q95"]], [yy, yy], color=FLOW, lw=9, alpha=0.45, solid_capstyle="butt")
        a2.plot([w["q25"], w["q75"]], [yy, yy], color=FLOW, lw=9, solid_capstyle="butt")
        a2.plot([w["reference"]] * 2, [yy - 0.33, yy + 0.33], color=INK, lw=1.6)
        a2.plot([w["maxent"]] * 2, [yy - 0.33, yy + 0.33], color=ME, lw=1.6)
    name = lambda lo, hi: f"[{lo:g}, {hi:g})".replace("inf", "∞")
    a2.set_yticks(y, [name(*w["window"]) for w in wins], fontsize=8.5)
    a2.set_xlabel("spectral weight in the frequency window")
    a2.plot([], [], color=PALE, lw=7, label="allowed by the data alone")
    a2.plot([], [], color=FLOW, lw=7, alpha=0.45, label="posterior 90% interval")
    a2.plot([], [], color=FLOW, lw=7, label="posterior 50% interval")
    a2.plot([], [], color=INK, lw=1.6, label="SmoQyDEAC reference")
    a2.plot([], [], color=ME, lw=1.6, label="MaxEnt")
    a2.legend(fontsize=7.5, frameon=False, loc="center right")
    cal = "calibrated intervals" if rep.diagnostics.get("band_calibrated") else "plain sample quantiles"
    a2.set_title(f"weight per window ({cal})", fontsize=9.5)
    for ax in (a0, a1, a2):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    warn = "; ".join(w.split(":")[0] for w in rep.warnings) if rep.warnings else "none"
    f.suptitle(f"Holstein DQMC cell (beta = {beta:g}, {N} bins, relative error bar {rel:.0e}), "
               f"{Path(sys.argv[2]).stem}: no known answer.  Warnings: {len(rep.warnings)}", fontsize=10.5)
    f.savefig(out / "holstein_report.png", dpi=150)
    print(f"W1 to the reference: {w1_ref}\nwarnings: {warn}\n-> {out}/holstein_report.{{json,png,txt}}")


if __name__ == "__main__":
    main()
