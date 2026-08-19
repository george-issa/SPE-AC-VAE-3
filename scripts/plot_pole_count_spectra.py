#!/usr/bin/env python
"""Overlay the A(omega) that checkpoints with different pole counts produce.

Two panels: the benchmark's synthetic two-Lorentzian case (exact answer drawn
in black) and a real dataset (no truth exists; agreement/disagreement between
pole counts is the information).  Deterministic posterior-mean decodes, so the
curves differ only through the trained weights.

    python scripts/plot_pole_count_spectra.py \
        out/model-fast.pt:P=16 out/model-fast-p24.pt:P=24 \
        --bins out/hubbard_U6_b10_bins.txt --beta 10 \
        --out out/pole_count_spectra.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import synthetic_case  # noqa: E402  (scripts/benchmark.py)
from spevae import load_model  # noqa: E402
from spevae.data import read_gissa_cube, read_text_bins  # noqa: E402
from spevae.paths import data_path  # noqa: E402

COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def spectrum(model, mg, omega):
    out = model.continue_(mg.mean.unsqueeze(0).float(), beta=mg.beta,
                          tau=mg.tau.float(),
                          sigma=(mg.sigma_encoder.unsqueeze(0).float()
                                 / mg.n_bins**0.5),
                          omega=omega, n_samples=0)
    return out["A"][0].double().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", metavar="CKPT:LABEL")
    ap.add_argument("--bins", default=None, help="real dataset for panel 2")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--gissa", nargs=4, type=float, default=None,
                    metavar=("BETA", "OMEGA", "N", "LAM"),
                    help="a coupling-resolved Holstein cube cell, by physical "
                         "values; attaches the shipped SmoQyDEAC reference")
    ap.add_argument("--no-synthetic", action="store_true",
                    help="drop the synthetic known-answer panel")
    ap.add_argument("--omega-max", type=float, default=8.0)
    ap.add_argument("--out", default="pole_count_spectra.png")
    args = ap.parse_args()

    omega = torch.linspace(-args.omega_max, args.omega_max, 1601)
    w = omega.double().numpy()

    panels = []
    if not args.no_synthetic:
        mg_syn, truth_poles = synthetic_case()
        panels.append(("synthetic two-Lorentzian (truth known)", mg_syn,
                       truth_poles.spectral(omega.double())[0].numpy(),
                       "exact"))
    if args.bins:
        mg_real = read_text_bins(args.bins, args.beta, dtype=torch.float64)
        panels.append((f"{os.path.basename(args.bins)}  (no truth exists)",
                       mg_real, None, None))
    if args.gissa:
        gb, go, gn, gl = args.gissa
        mg_g = read_gissa_cube(
            str(data_path("datasets", "real", "gIssa", "for_george_ac.jld2")),
            beta=gb, omega=go, n=gn, lam=gl,
            dos_path=str(data_path("datasets", "real", "gIssa",
                                   "george_DOS.jld2")))
        ref = None
        if mg_g.reference is not None:
            import numpy as np
            ws, A_ref = mg_g.reference
            ref = np.interp(w, np.asarray(ws, float), np.asarray(A_ref, float))
        panels.append((f"gIssa Holstein  lam={gl:g} n={gn:g} "
                       f"Omega={go:g} beta={gb:g}", mg_g, ref,
                       "SmoQyDEAC ref (not truth)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.4))
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (title, mg, truth, truth_label) in zip(axes, panels):
        if truth is not None:
            style = dict(color="black", lw=2.0) if truth_label == "exact" \
                else dict(color="0.35", lw=1.6, ls="--")
            ax.plot(w, truth, label=truth_label, zorder=5, **style)
        for i, spec in enumerate(args.checkpoints):
            path, _, label = spec.partition(":")
            model = load_model(path)
            A = spectrum(model, mg, omega)
            ax.plot(w, A, color=COLOURS[i % len(COLOURS)], lw=1.4,
                    label=label or os.path.basename(path))
            # PH-symmetry report: physical when G(tau) = G(beta - tau); the
            # model does not hardwire it, so the residual asymmetry is a
            # meaningful diagnostic of what it learned.
            asym = float(abs(A - A[::-1]).sum() / max(abs(A).sum(), 1e-300))
            print(f"  [{title.split()[0]}] {label or path}: "
                  f"integrated PH asymmetry {asym:.3f}")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$A(\omega)$")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)

    fig.suptitle("Posterior-mean spectra by pole count (same data, same "
                 "training seed/preset)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
