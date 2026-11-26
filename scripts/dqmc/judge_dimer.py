"""Real DQMC data with a known answer: the half-filled Hubbard dimer, or a ring.

Reads the bins written by `hubbard_dimer.jl` or `hubbard_ring.jl` (SmoQyDQMC;
a directory named hubbard_ring<L>_... is read as a ring of L sites), computes
the exact spectrum by exact diagonalisation, and

  1. compares the measured G(tau) with the exact one in units of the error bar
     (statistical noise vs the Trotter discretisation error of DQMC), and
  2. continues the measured data three ways -- refined flow posterior, VAE
     latent MAP, MaxEnt -- scoring each against the exact cumulative weight.

    python scripts/dqmc/judge_dimer.py out/dqmc/hubbard_dimer_U4.00_b8.00_dt0.050-1 4.0 \
        out/flow/F_60k.pt out/step2/data_arm/D24_60k.pt out/dqmc [floor_factor] [discrete]

`floor_factor` (a number or `auto`) is handed to `continue_posterior`; without
it the report's default applies (`auto` since 2026-11-25; the runs banked
before that used 1, the width floor as trained).  `discrete` states that the
spectrum is a set of lines (`continue_posterior(discrete=True)`), which these
models' spectra are.

Writes <out>/dimer_judge.{json,png} (ring<L>_judge.{json,png} for a ring).  The
json holds, per frequency window, the exact weight and whether it lies inside
the intervals the report states and inside the range the data alone allow.
"""
import json
import math
import os
import re
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
    extra = sys.argv[6:]
    ff = {"discrete": True} if "discrete" in extra else {}
    for a in extra:
        if a != "discrete":
            ff["floor_factor"] = a if a == "auto" else float(a)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    ring = re.search(r"hubbard_ring(\d+)", os.path.basename(os.path.normpath(sim)))
    if ring:                                                    # one orbital, sites averaged by the package
        n_sites = int(ring.group(1))
        mg = read_smoqy_dir(sim)
        hop = np.zeros((n_sites, n_sites))
        for i in range(n_sites):
            hop[i, (i + 1) % n_sites] = hop[(i + 1) % n_sites, i] = -1.0
        tag, what = f"ring{n_sites}", f"half-filled Hubbard ring of {n_sites} sites"
    else:
        mg = read_smoqy_dir(sim, pair=(0, 1))                   # the two sites are equivalent
        hop = np.array([[0.0, -1.0], [-1.0, 0.0]])
        tag, what = "dimer", "half-filled Hubbard dimer"
    beta, N = mg.beta, mg.n_bins
    exact = hubbard_cluster(hop, U, U / 2.0, beta)
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
                             sigma_fit=cov["sigma"] / math.sqrt(N), omega=WG, **ff)
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
    # the exact weight of every window against what the report states for it
    wins, tol = [], 1e-6
    for w in rep.windows:
        lo_w, hi_w = w["window"]
        t = float(exact.weights[(exact.energies >= lo_w) & (exact.energies < hi_w)].sum() / exact.weights.sum())
        row = dict(w, exact=t, inside_50=bool(w["q25"] - tol <= t <= w["q75"] + tol),
                   inside_90=bool(w["q05"] - tol <= t <= w["q95"] + tol))
        if w.get("lo") is not None and w["lo"] == w["lo"]:
            row["inside_data_alone"] = bool(w["lo"] - tol <= t <= w["hi"] + tol)
        wins.append(row)
    hits = {k: int(sum(r.get(k, False) for r in wins)) for k in ("inside_50", "inside_90", "inside_data_alone")}
    print(f"exact window weights inside the stated 50% interval: {hits['inside_50']} of {len(wins)}; 90%: "
          f"{hits['inside_90']} of {len(wins)}; inside the range the data alone allow: "
          f"{hits['inside_data_alone']} of {sum('inside_data_alone' in r for r in wins)}")
    res = {"sim": sim, "model": tag, "stated_discrete": bool(ff.get("discrete")), "n_lines": int(big.sum()),
           "U": U, "beta": beta, "n_bins": N, "n_tau": int(mg.n_tau),
           "relative_error_bar": rel, "z_max": float(np.abs(zsc[1:-1]).max()),
           "z_rms": float(np.sqrt((zsc[1:-1] ** 2).mean())),
           "dG_max": float(np.abs(mean - G_exact).max()), "chi2_exact_over_target": chi_exact,
           "exact_lines": {"energies": exact.energies[big].tolist(), "weights": exact.weights[big].tolist()},
           "w1": w1, "vae_chi2_over_target": float(rr.chi2_ratio),
           "flow": {"accepted": rep.n_accepted, "of": rep.n_samples,
                    "chi2_over_target_median": float(np.median(rep.chi2) / rep.chi2_target),
                    "warnings": rep.warnings, "windows": wins, "window_hits": hits, **rep.diagnostics}}
    print("W1 to the exact answer:", {k: round(v, 4) for k, v in w1.items()},
          "| VAE chi2/target", round(rr.chi2_ratio, 2))
    with open(out / f"{tag}_judge.json", "w") as fh:
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
    f.suptitle(f"Real DQMC data with a known answer: {what}, U = {U:g}, beta = {beta:g}, "
               f"{N} bins, relative error bar {rel:.0e} (SmoQyDQMC; exact spectrum by diagonalisation)",
               fontsize=10.5)
    f.savefig(out / f"{tag}_judge.png", dpi=150)
    print(f"-> {out}/{tag}_judge.json + .png")


if __name__ == "__main__":
    main()
