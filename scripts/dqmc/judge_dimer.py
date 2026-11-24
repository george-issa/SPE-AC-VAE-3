"""Real DQMC data with a known answer: the half-filled Hubbard dimer.

Reads the bins written by `hubbard_dimer.jl` (SmoQyDQMC), computes the exact
spectrum by exact diagonalisation, and

  1. compares the measured G(tau) with the exact one in units of the error bar
     (statistical noise vs the Trotter discretisation error of DQMC), and
  2. continues the measured data three ways -- refined flow posterior, VAE
     latent MAP, MaxEnt -- scoring each against the exact cumulative weight.

    python scripts/dqmc/judge_dimer.py out/dqmc/hubbard_dimer_U4.00_b8.00_dt0.050-1 4.0 \
        out/flow/F_60k.pt out/step2/data_arm/D24_60k.pt out/dqmc
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

WG = np.linspace(-15.0, 15.0, 60001)
DW = WG[1] - WG[0]
FLOW, VAE, ME, INK = "#2a78d6", "#eb6834", "#7d7c78", "#0b0b0b"


def main():
    from spevae import load_model
    from spevae.benchmark import run_maxent
    from spevae.data.exact_models import hubbard_cluster
    from spevae.data.readers import read_smoqy_dir
    from spevae.models.flow import load_flow
    from spevae.physics.poles import PoleSet
    from spevae.posterior import continue_posterior
    from spevae.refine import RefineConfig, refine_latent

    sim, U = sys.argv[1], float(sys.argv[2])
    flow, vae, out = load_flow(sys.argv[3]).eval(), load_model(sys.argv[4]), Path(sys.argv[5])
    torch.set_num_threads(6)
    mg = read_smoqy_dir(sim, pair=(0, 1))                       # the two sites are equivalent
    beta, N = mg.beta, mg.n_bins
    exact = hubbard_cluster(np.array([[0.0, -1.0], [-1.0, 0.0]]), U, U / 2.0, beta)
    cov = mg.covariance(1e-10, method="structured")
    sl = cov["slices"]
    x = mg.x.double().numpy()
    G_exact = exact.greens(x)
    mean, sem = mg.mean.double().numpy(), (mg.sigma / math.sqrt(N)).double().numpy()
    zsc = (mean - G_exact) / np.maximum(sem, 1e-300)
    d = torch.tensor(((G_exact - mean)[sl.numpy()] / (cov["sigma"].double().numpy() / math.sqrt(N))))
    chi_exact = float(((d @ cov["inv_sqrt_R"].double().T) ** 2).mean() / cov["chi2_target"])
    rel = float(np.median(sem[1:-1] / np.abs(mean).max()))
    print(f"{N} bins, beta {beta:g}, {mg.n_tau} slices; median relative error bar {rel:.1e}")
    print(f"measured vs exact G(tau): max |z| {np.abs(zsc[1:-1]).max():.1f}, rms z {np.sqrt((zsc[1:-1] ** 2).mean()):.2f}; "
          f"max |dG| {np.abs(mean - G_exact).max():.1e}; chi2 of the EXACT G against the data / target "
          f"{chi_exact:.2f}")

    rep = continue_posterior(flow, mg.mean, mg.x, beta, mg.sigma_encoder / math.sqrt(N),
                             cov["inv_sqrt_R"], bins=mg.bins[:, sl], n_samples=200,
                             chi2_target=cov["chi2_target"], x_fit=mg.x[sl], G_fit=mg.mean[sl],
                             sigma_fit=cov["sigma"] / math.sqrt(N), omega=WG)
    print(rep.summary())
    rr = refine_latent(vae, mg, RefineConfig(cov_method="structured", min_eigenvalue=1e-10, solver="lm"))
    pv = rr.poles
    nu = torch.tensor(WG, dtype=torch.float64)
    A_vae = PoleSet(*(getattr(pv, k).double() for k in ("eps", "gamma", "a", "b"))).spectral(nu)[0].numpy()
    me = run_maxent(mg, min_eigenvalue=1e-10)
    A_me = np.interp(WG, me.omega, me.A, left=0.0, right=0.0)

    cdf_t = exact.cdf(WG)
    cdf = lambda a: np.cumsum(a) * DW
    w1 = {"flow": float(np.abs(cdf(rep.band["mean"]) - cdf_t).sum() * DW),
          "vae": float(np.abs(cdf(A_vae) - cdf_t).sum() * DW),
          "maxent": float(np.abs(cdf(A_me) - cdf_t).sum() * DW)}
    big = exact.weights > 1e-3
    res = {"sim": sim, "U": U, "beta": beta, "n_bins": N, "n_tau": int(mg.n_tau),
           "relative_error_bar": rel, "z_max": float(np.abs(zsc[1:-1]).max()),
           "z_rms": float(np.sqrt((zsc[1:-1] ** 2).mean())),
           "dG_max": float(np.abs(mean - G_exact).max()), "chi2_exact_over_target": chi_exact,
           "exact_lines": {"energies": exact.energies[big].tolist(), "weights": exact.weights[big].tolist()},
           "w1": w1, "vae_chi2_over_target": float(rr.chi2_ratio),
           "flow": {"accepted": rep.n_accepted, "of": rep.n_samples,
                    "chi2_over_target_median": float(np.median(rep.chi2) / rep.chi2_target),
                    "warnings": rep.warnings, "windows": rep.windows, **rep.diagnostics}}
    print("W1 to the exact answer:", {k: round(v, 4) for k, v in w1.items()},
          "| VAE chi2/target", round(rr.chi2_ratio, 2))
    with open(out / "dimer_judge.json", "w") as fh:
        json.dump(res, fh, indent=1)

    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17, 4.2), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1, 1.6, 1]})
    a0.axhspan(-2, 2, color="#e4e3df", lw=0)
    # tau = 0 and tau = beta are pinned by symmetry at half filling (zero variance): not shown
    a0.plot((x * beta)[1:-1], zsc[1:-1], color=INK, lw=1.0)
    a0.set_xlabel(r"$\tau$")
    a0.set_title("(measured $-$ exact) / error bar\n"
                 f"rms {res['z_rms']:.2f}, max {res['z_max']:.1f}; shaded = $\\pm 2\\sigma$", fontsize=9.5)
    m = np.abs(WG) <= 8
    st = slice(None, None, 8)
    a1.fill_between(WG[m][st], rep.band["q05"][m][st], rep.band["q95"][m][st], color=FLOW, alpha=0.25, lw=0,
                    label="flow posterior, 90% band")
    a1.plot(WG[m][st], rep.band["q50"][m][st], color=FLOW, lw=1.3, label="flow posterior, median")
    a1.plot(WG[m][st], A_vae[m][st], color=VAE, lw=1.0, label="VAE latent MAP")
    a1.plot(WG[m][st], A_me[m][st], color=ME, lw=1.0, label="MaxEnt")
    top = 1.25 * float(max(rep.band["q95"][m].max(), A_vae[m].max()))
    a1.vlines(exact.energies[big], 0, exact.weights[big] / exact.weights.max() * 0.92 * top, color=INK,
              lw=1.8, label="exact lines (height = weight)")
    a1.set_ylim(-0.03 * top, top)
    a1.set_xlabel(r"$\omega$")
    a1.set_title(f"spectra;  W$_1$ to the exact answer: flow {w1['flow']:.3f}, VAE {w1['vae']:.3f}, "
                 f"MaxEnt {w1['maxent']:.3f}", fontsize=9.5)
    a1.legend(fontsize=7.5, frameon=False)
    a2.step(WG[m], cdf_t[m], where="post", color=INK, lw=1.6, label="exact")
    a2.plot(WG[m][st], cdf(rep.band["mean"])[m][st], color=FLOW, lw=1.3, label="flow posterior mean")
    a2.plot(WG[m][st], cdf(A_vae)[m][st], color=VAE, lw=1.0, label="VAE")
    a2.plot(WG[m][st], cdf(A_me)[m][st], color=ME, lw=1.0, label="MaxEnt")
    a2.set_xlabel(r"$\omega$")
    a2.set_title("cumulative spectral weight", fontsize=9.5)
    a2.legend(fontsize=7.5, frameon=False)
    for ax in (a0, a1, a2):
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.suptitle(f"Real DQMC data with a known answer: half-filled Hubbard dimer, U = {U:g}, beta = {beta:g}, "
               f"{N} bins, relative error bar {rel:.0e} (SmoQyDQMC; exact spectrum by diagonalisation)",
               fontsize=10.5)
    f.savefig(out / "dimer_judge.png", dpi=150)
    print(f"-> {out}/dimer_judge.json + .png")


if __name__ == "__main__":
    main()
