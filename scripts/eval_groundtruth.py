#!/usr/bin/env python
"""Evaluate the full pipeline against the synthetic ground-truth library.

Each dataset ships DQMC-like bins (1000 x L_tau, endpoint-exclusive) plus the
exact A(omega) that generated them -- independent of this package's training
generator, so this is a genuine out-of-family test.  Per dataset the script
runs the amortised model, latent refinement, consolidation + data refit
(cross-pair positivity: unconstrained b with the exact certificate), and a
MaxEnt baseline, scoring everything against the truth with one metric set.

    python scripts/eval_groundtruth.py --checkpoint out/model-fast-24k.pt \
        --figure out/groundtruth_grid.png --json out/groundtruth.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spevae import RefineConfig, load_model, refine_latent  # noqa: E402
from spevae.benchmark import maxent_available, run_maxent  # noqa: E402
from spevae.data import from_bins  # noqa: E402
from spevae.metrics import spectral_report  # noqa: E402
from spevae.paths import data_path  # noqa: E402
from spevae.physics import (  # noqa: E402
    consolidate_poles,
    positivity_certificate,
    refit_poles_to_greens,
)

DEFAULT_SETS = [
    "gaussian_double", "gapped_mott", "gapped_semicircle", "pseudogap",
    "three_peak_hubbard", "asymmetric_three_peak", "fano", "two_close_peaks",
]


def read_meta(path: str) -> dict:
    keys = ("beta", "dtau", "omega_min", "omega_max", "n_omega")
    out: dict = {}
    for line in open(path):
        m = re.match(r"\s*(\w+)\s*=\s*([-\d.eE+]+)", line)
        if m and m.group(1) in keys:
            out[m.group(1)] = float(m.group(2))
    return out


def load_dataset(name: str, noise: str = "s1e-04_xi0.5"):
    base = data_path("datasets", "synthetic", f"half-filled-{name}")
    d = None
    for sub in sorted(os.listdir(base)):
        cand = base / sub
        if (cand / f"Gbins_{noise}.csv").exists():
            d = cand
            break
    if d is None:
        raise FileNotFoundError(f"no inputs-* under {base} carries "
                                f"Gbins_{noise}.csv")
    meta = read_meta(str(d / f"metadata_{noise}.toml"))
    bins = np.loadtxt(d / f"Gbins_{noise}.csv", delimiter=",")
    beta, dtau = meta["beta"], meta["dtau"]
    tau = torch.arange(bins.shape[1], dtype=torch.float64) * dtau
    mg = from_bins(bins, beta, tau=tau, includes_endpoints=False,
                   label=name, dtype=torch.float64)
    A_true = np.loadtxt(d / "spectral_input.csv")
    if "omega_min" in meta:
        w = np.linspace(meta["omega_min"], meta["omega_max"],
                        int(meta["n_omega"]))
    else:
        # Some datasets predate the [spectral_input] table; the documented
        # legacy grid is [-5, 5].  Loud, because a silently wrong grid once
        # misplaced a ground truth (the inputs-13 incident).
        w = np.linspace(-5.0, 5.0, A_true.size)
        print(f"  [{name}] WARNING: metadata carries no spectral grid; "
              f"assuming the legacy [-5, 5]", flush=True)
    return mg, w, A_true




# --------------------------------------------------------------------------- #
# Regenerated evaluation sets: the library's formulas, this package's verified
# forward map.  The v1-era Gbins were measured inconsistent with their own
# shipped truths (up to ~500 sigma at s=1e-4); these are consistent by
# construction, asserted at generation time.
# --------------------------------------------------------------------------- #

def _N(w, mu, s):
    return np.exp(-0.5 * ((w - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def _semicircle(w, c, r):
    d = np.clip(1.0 - ((w - c) / r) ** 2, 0.0, None)
    return 2.0 / (np.pi * r) * np.sqrt(d)


FORMULAS = {
    "gaussian_double": lambda w: 0.6 * _N(w, 0.0, 0.5) + 0.4 * _N(w, 2.0, 0.5),
    "gapped_mott": lambda w: 0.5 * _N(w, -2.0, 0.4) + 0.5 * _N(w, 2.0, 0.4),
    "gapped_semicircle": lambda w: 0.5 * _semicircle(w, -2.5, 1.5)
    + 0.5 * _semicircle(w, 2.5, 1.5),
    "pseudogap": lambda w: np.sqrt(np.abs(w)) * np.exp(-w**2 / (2 * 1.5**2)),
    "three_peak_hubbard": lambda w: 0.2 * _N(w, -2.5, 0.3)
    + 0.6 * _N(w, 0.0, 0.2) + 0.2 * _N(w, 2.5, 0.3),
    "asymmetric_three_peak": lambda w: 0.15 * _N(w, -3.0, 0.4)
    + 0.55 * _N(w, -0.3, 0.25) + 0.3 * _N(w, 1.8, 0.6),
    "fano": lambda w: (2.0 + w / 0.5) ** 2 / (1.0 + (w / 0.5) ** 2)
    * _N(w, 0.0, 2.0),
    "two_close_peaks": lambda w: 0.5 * _N(w, -0.4, 0.2) + 0.5 * _N(w, 0.4, 0.2),
}


def regen_dataset(name: str, beta: float = 10.0, n_tau: int = 201,
                  n_bins: int = 1000, noise: float = 1e-4, seed: int = 0):
    from spevae.physics import PoleGreens, SpectralQuadrature

    quad = SpectralQuadrature.for_beta(beta, dtype=torch.float64)
    nu = quad.nodes.numpy()
    A_nodes = FORMULAS[name](nu / beta)
    A_nodes = A_nodes / float((torch.tensor(A_nodes) * quad.weights).sum())
    A_dimless = torch.tensor(A_nodes, dtype=torch.float64)

    tau = torch.linspace(0.0, beta, n_tau, dtype=torch.float64)
    K = PoleGreens(quad).kernel_matrix(tau / beta)
    G_clean = (A_dimless * quad.weights) @ K

    g = torch.Generator().manual_seed(seed)
    d = (tau.reshape(-1, 1) - tau.reshape(1, -1)).abs() / beta
    R = 0.8 * torch.exp(-d / 0.05) + 0.2 * torch.eye(n_tau, dtype=torch.float64)
    eta = torch.randn(n_bins, n_tau, generator=g, dtype=torch.float64)         @ torch.linalg.cholesky(R).T
    bins = G_clean.unsqueeze(0) + eta * noise
    mg = from_bins(bins, beta, tau=tau, includes_endpoints=True,
                   label=name, dtype=torch.float64)
    # consistency by construction, asserted rather than trusted
    dev = float((mg.mean - G_clean).abs().max())
    sem = float(mg.sigma_mean.max())
    assert dev < 6 * sem, (name, dev, sem)

    w = np.linspace(-8.0, 8.0, 1600)
    A_true = FORMULAS[name](w)
    A_true = A_true / np.trapezoid(A_true, w)
    return mg, w, A_true


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--source", default="library",
                    choices=["library", "regen"],
                    help="library = v1-era Gbins (measured inconsistent); regen = consistent-by-construction")
    ap.add_argument("--refine-steps", type=int, default=8000)
    ap.add_argument("--refit-steps", type=int, default=4000)
    ap.add_argument("--figure", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    results = {}
    curves = {}

    for name in args.datasets:
        if args.source == "regen":
            mg, w, A_true = regen_dataset(name)
        else:
            mg, w, A_true = load_dataset(name)
        cov = mg.covariance()
        tgt = float(cov["chi2_target"])
        res_scale = math.pi / mg.beta
        wt = torch.tensor(w, dtype=torch.float32)
        rows = {}

        sig_enc = (mg.sigma_encoder / mg.n_bins**0.5).unsqueeze(0).float()
        out = model.continue_(mg.mean.unsqueeze(0).float(), beta=mg.beta,
                              tau=mg.tau.float(), sigma=sig_enc,
                              omega=wt, n_samples=200)
        rows["amortized"] = (out["A"][0].double().numpy(), None)

        rr = refine_latent(model, mg, RefineConfig(steps=args.refine_steps,
                                                   patience=2000))
        rows["refined"] = (rr.spectral(wt)[0].double().numpy(),
                           rr.chi2 / tgt)

        # Two refit starts -- the refined poles and the amortised ones -- and
        # keep the chi2-better result.  At the tightest noise floors the
        # latent refinement can stall (measured on gapped_mott: not
        # prior-binding, a plain non-convex stall), and a data-anchored refit
        # from a mangled start inherits the damage; initialisation diversity
        # is the cheap insurance.
        det = model.continue_(mg.mean.unsqueeze(0).float(), beta=mg.beta,
                              tau=mg.tau.float(), sigma=sig_enc, n_samples=0)
        # Resolution sweep x start diversity.  Selection is the parsimony
        # elbow, NOT argmin-chi2: measured on gaussian_double, the finer
        # resolution won chi2 (93 vs 133) while losing against the truth
        # (l2 0.36 vs 0.23) -- pure-chi2 selection over-resolves, which is
        # the ill-posedness reappearing at the model-selection level.  Rule:
        # fewest live poles among candidates within 1.5x of the best chi2.
        cands = []
        for start in (rr.poles, det["poles"]):
            for mult in (1.0, 2.0):
                merged_s, _ = consolidate_poles(start, mult * res_scale)
                # Two phases: converge on pure chi2, then add positivity
                # scaled to the CONVERGED chi2.  Scaling it to the starting
                # chi2 (previous version) handed bad starts a penalty in the
                # tens of thousands, deforming exactly the low-pole
                # candidates the parsimony rule then favours.
                p1, _, c1 = refit_poles_to_greens(
                    merged_s, mg.beta, mg.x, mg.mean, mg.sigma_mean,
                    inv_sqrt_R=cov["inv_sqrt_R"], slices=cov["slices"],
                    steps=args.refit_steps)
                cand, _, c_s = refit_poles_to_greens(
                    p1, mg.beta, mg.x, mg.mean, mg.sigma_mean,
                    inv_sqrt_R=cov["inv_sqrt_R"], slices=cov["slices"],
                    steps=args.refit_steps // 2,
                    positivity_weight=max(1.0, 0.05 * c1))
                n_live = int((cand.a[0] > 1e-5 * float(cand.a[0].sum())).sum())
                cands.append((cand, c_s, n_live))
        c_min = min(c_s for _, c_s, _ in cands)
        ok = [t for t in cands if t[1] <= 1.5 * c_min]
        final, c, _ = min(ok, key=lambda t: (t[2], t[1]))
        cert = positivity_certificate(final[0:1])[0]
        rows["consolidated"] = (final.spectral(wt)[0].double().numpy(), c / tgt)
        n_poles = int((final.a[0] > 1e-5 * float(final.a[0].sum())).sum())

        if maxent_available():
            me = run_maxent(mg, omega=w)
            rows["maxent"] = (me.A, me.meta.get("chi2_ratio"))

        entry = {"chi2_target": tgt, "n_poles_consolidated": n_poles,
                 "certified_positive": bool(cert.is_positive),
                 "min_A": float(cert.min_A)}
        for label, (A, ratio) in rows.items():
            rep_m = spectral_report(A, A_true, w, resolution=res_scale)
            entry[label] = {
                "w1": rep_m["w1"], "l2_rel": rep_m["l2_rel"],
                "peak_d_omega": rep_m.get("peak_d_omega"),
                "peak_rel_d_height": rep_m.get("peak_rel_d_height"),
                "n_pred": rep_m.get("n_pred"), "n_true": rep_m.get("n_true"),
                "missed": rep_m.get("missed"), "spurious": rep_m.get("spurious"),
                "gap_error": rep_m.get("gap_error"),
                "chi2_ratio": ratio,
            }
        results[name] = entry
        curves[name] = (w, A_true, rows)
        r, cns, mx = entry["refined"], entry["consolidated"], entry.get("maxent", {})
        print(f"[{name:>22}] poles={n_poles:>2} "
              f"W1 ref/cons/ME: {r['w1']:.3f}/{cns['w1']:.3f}/"
              f"{mx.get('w1', float('nan')):.3f}  "
              f"l2: {r['l2_rel']:.3f}/{cns['l2_rel']:.3f}/"
              f"{mx.get('l2_rel', float('nan')):.3f}  "
              f"chi2/tgt cons {cns['chi2_ratio']:.2f}  "
              + ("A>=0" if entry["certified_positive"]
                 else f"minA {entry['min_A']:.0e}"),
              flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"json -> {args.json}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(curves)
        ncol = 4
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow))
        axes = np.atleast_2d(axes)
        for k, (name, (w, A_true, rows)) in enumerate(curves.items()):
            ax = axes[k // ncol][k % ncol]
            ax.plot(w, A_true, "k", lw=1.9, label="truth", zorder=5)
            styles = {"amortized": ("0.72", 0.9), "refined": ("#009E73", 1.2),
                      "consolidated": ("#0072B2", 1.6), "maxent": ("#D55E00", 1.1)}
            for label, (A, _) in rows.items():
                col, lw = styles[label]
                ax.plot(w, A, color=col, lw=lw, label=label)
            ax.set_title(name, fontsize=9)
            ax.grid(alpha=0.25)
            if k == 0:
                ax.legend(frameon=False, fontsize=7)
        for k in range(n, nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("Ground-truth evaluation: out-of-family spectra "
                     f"({os.path.basename(args.checkpoint)})", fontsize=11)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=140)
        print(f"figure -> {args.figure}")


if __name__ == "__main__":
    main()
