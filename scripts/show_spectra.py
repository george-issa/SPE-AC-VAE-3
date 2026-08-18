#!/usr/bin/env python
"""Show what the model actually produces: A(omega) against the known truth.

    python scripts/show_spectra.py --checkpoint out/model25k.pt --out out/spectra.png

One panel per spectral family, drawn from the held-out generator with a fixed
seed.  The truth is known here, so this is the only view that can say whether
the spectra are right -- as opposed to whether they fit the data, which chi^2
answers and which is a different question.
"""
from __future__ import annotations

import argparse
import math
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spevae import load_model
from spevae.data import SyntheticGenerator
from spevae.metrics import spectral_report

FAMILIES = ["gauss_mixture", "gapped", "metal_hubbard",
            "phonon", "bcs", "semicircle"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--beta-min", type=float, default=6.0)
    ap.add_argument("--beta-max", type=float, default=10.0)
    ap.add_argument("--omega-max", type=float, default=8.0)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--out", default="out/spectra.png")
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    fams = args.families.split(",")
    omega = np.linspace(-args.omega_max, args.omega_max, 1201)
    w_t = torch.as_tensor(omega, dtype=torch.float32)

    n = len(fams)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.1 * nrow),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    rows = []

    for ax, fam in zip(axes, fams):
        gen = SyntheticGenerator(omega_c=20.0,
                                 beta_range=(args.beta_min, args.beta_max),
                                 max_ntau=161, dtau_choices=(0.1, 0.125, 0.2),
                                 families=[fam])
        rng = torch.Generator().manual_seed(args.seed)
        b = gen.batch(4, rng)
        beta = b.beta

        # Truth, converted from dimensionless nu to physical omega.
        # Only the *core* panel: quad.nodes is core-then-reciprocal-tails and is
        # therefore not ascending, which np.interp requires and does not check.
        nc = b.quad.n_core
        nu = b.quad.nodes[:nc].numpy()
        A_true_nu = b.A_true[0, :nc].numpy()
        assert np.all(np.diff(nu) > 0), "core panel must be ascending"
        A_true = np.interp(omega * beta, nu, A_true_nu, left=0.0, right=0.0) * beta

        out = model.continue_(b.G[0:1], beta=beta, tau=(b.x * beta),
                              sigma=b.sigma[0:1], omega=w_t,
                              n_samples=args.n_samples)
        A = out["A"][0].numpy()
        A_sd = out["A_std"][0].numpy() if "A_std" in out else None
        rep = spectral_report(A, A_true, omega, resolution=math.pi / beta)
        eff = float(1.0 / (out["poles"].a[0] ** 2).sum())
        rows.append((fam, beta, rep, eff, float(b.meta["noise"].scale)))

        ax.plot(omega, A_true, "k--", lw=2.0, label="truth")
        if A_sd is not None:
            ax.fill_between(omega, A - A_sd, A + A_sd, color="#0072B2",
                            alpha=0.22, lw=0)
        ax.plot(omega, A, color="#0072B2", lw=1.7, label="spevae")
        ax.axhline(0, color="0.8", lw=0.7, zorder=0)
        ax.set_title(f"{fam}   $\\beta$={beta:.1f}", fontsize=10)
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$A(\omega)$")
        ax.text(0.02, 0.97,
                f"$W_1$={rep['w1']:.3f}\n$L^2_{{rel}}$={rep['l2_rel']:.2f}\n"
                f"$\\Delta\\omega_{{pk}}$={rep['peak_d_omega']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                color="0.25")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"A(omega) from {args.checkpoint} -- held-out synthetic, "
                 f"truth known (band = latent ensemble, {args.n_samples} draws)",
                 fontsize=11)
    fig.savefig(args.out, dpi=150)

    hdr = (f"{'family':16s} {'beta':>5s} {'noise':>8s} {'W1':>7s} {'L2rel':>7s} "
           f"{'pk dw':>7s} {'pk dh':>7s} {'npk p/t':>9s} {'P_eff':>6s}")
    print(hdr); print("-" * len(hdr))
    for fam, beta, r, eff, ns in rows:
        print(f"{fam:16s} {beta:5.1f} {ns:8.1e} {r['w1']:7.3f} {r['l2_rel']:7.3f} "
              f"{r['peak_d_omega']:7.3f} {r['peak_rel_d_height']:7.3f} "
              f"{r['peak_n_pred']:4d}/{r['peak_n_true']:<4d} {eff:6.1f}")
    print(f"\nfigure -> {args.out}")


if __name__ == "__main__":
    main()
