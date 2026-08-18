#!/usr/bin/env python
"""Three-way analytic-continuation benchmark: spevae vs MaxEnt vs SmoQyDEAC.

Two modes:

  --synthetic     a spectrum with a *known* answer, so the table scores accuracy
  --holstein ...  a real DQMC cell, where the table scores agreement only

The synthetic mode is the one that can validate anything, because on real data
no method is ground truth.  Both are worth running: the first says how the
methods behave when the answer is known, the second says whether that behaviour
survives contact with real noise.

Examples
--------
    python scripts/benchmark.py --synthetic --no-spevae
    python scripts/benchmark.py --synthetic --checkpoint out/smoke6k.pt
    python scripts/benchmark.py --holstein 19 1 5 --checkpoint out/smoke6k.pt
    python scripts/benchmark.py --bins mybins.txt --beta 8.0 --figure cmp.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from spevae.benchmark import (  # noqa: E402
    compare, deac_available, format_table, maxent_available,
    run_deac, run_maxent, run_spevae,
)
from spevae.data import from_bins, read_holstein_cube, read_text_bins  # noqa: E402
from spevae.models import SPEVAE  # noqa: E402
from spevae import RefineConfig, load_model as _load  # noqa: E402
from spevae.physics import PoleGreens, PoleSet, SpectralQuadrature  # noqa: E402

from spevae.paths import data_path  # noqa: E402

CUBE = str(data_path("datasets", "real", "george_325.jld2"))


def synthetic_case(beta=10.0, n_tau=81, n_bins=800, noise=1e-3, seed=0):
    """A gapped two-peak spectrum with correlated noise, and its exact answer."""
    quad = SpectralQuadrature(nu_c=beta * 20, h=0.09, dtype=torch.float64)
    poles = PoleSet(
        eps=torch.tensor([[-1.5 * beta, 1.8 * beta]], dtype=torch.float64),
        gamma=torch.tensor([[0.35 * beta, 0.45 * beta]], dtype=torch.float64),
        a=torch.tensor([[0.45, 0.55]], dtype=torch.float64),
        b=torch.zeros(1, 2, dtype=torch.float64),
    )
    tau = torch.linspace(0, beta, n_tau, dtype=torch.float64)
    clean = PoleGreens(quad)(poles, tau / beta)[0]
    g = torch.Generator().manual_seed(seed)
    d = (tau.reshape(-1, 1) - tau.reshape(1, -1)).abs() / beta
    R = 0.8 * torch.exp(-d / 0.1) + 0.2 * torch.eye(n_tau, dtype=torch.float64)
    eta = torch.randn(n_bins, n_tau, generator=g, dtype=torch.float64) @ \
        torch.linalg.cholesky(R).T
    mg = from_bins(clean.unsqueeze(0) + eta * noise, beta, tau=tau,
                   label="synthetic two-peak", dtype=torch.float64)
    return mg, poles.to_physical(beta)


load_model = _load


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--holstein", nargs=3, type=int, metavar=("N", "OMEGA", "BETA"),
                     help="site-Holstein cube indices (n, Omega, beta)")
    src.add_argument("--bins", metavar="FILE",
                     help="your own bins: whitespace-delimited (N_bins x L_tau), "
                          "tau = 0 ... beta inclusive; needs --beta")
    ap.add_argument("--beta", type=float, default=None,
                    help="inverse temperature for --bins")
    ap.add_argument("--sign-flip", dest="sign_flip", action="store_true",
                    default=None, help="force negation (default: detect from the sum rule)")
    ap.add_argument("--no-sign-flip", dest="sign_flip", action="store_false",
                    help="force no negation")
    ap.add_argument("--checkpoint", default=None, help="trained spevae weights")
    ap.add_argument("--no-spevae", action="store_true")
    ap.add_argument("--refine", action="store_true",
                    help="add a latent-refined spevae row")
    ap.add_argument("--refine-steps", type=int, default=2000)
    ap.add_argument("--no-maxent", action="store_true")
    ap.add_argument("--no-deac", action="store_true")
    ap.add_argument("--omega-max", type=float, default=10.0)
    ap.add_argument("--n-omega", type=int, default=801)
    ap.add_argument("--deac-bins", type=int, default=8)
    ap.add_argument("--deac-runs", type=int, default=8)
    ap.add_argument("--deac-generations", type=int, default=20_000)
    ap.add_argument("--min-eigenvalue", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="write results to this .npz")
    ap.add_argument("--figure", default=None, help="save a comparison figure here")
    args = ap.parse_args()

    truth_poles = None
    if args.synthetic:
        mg, truth_poles = synthetic_case()
    elif args.bins:
        if args.beta is None:
            sys.exit("--bins requires --beta")
        mg = read_text_bins(args.bins, args.beta, sign_flip=args.sign_flip,
                            dtype=torch.float64)
    else:
        if not os.path.exists(CUBE):
            sys.exit(f"{CUBE} not found (the Data symlink may be missing)")
        mg = read_holstein_cube(CUBE, *args.holstein)

    print(mg.summary(args.min_eigenvalue))
    print()
    sr = mg.check_sum_rule()
    if not sr["ok"]:
        # Nearly always a sign convention or a grid that stops short of beta.
        # Continuing anyway would produce a confident, wrong spectrum.
        sys.exit(f"G(0)+G(beta) = {sr['norm']:.6f}, expected 1. "
                 f"Try --sign-flip, or check that tau runs 0 ... beta inclusive.")

    # One shared grid, evenly spaced, so DEAC accepts it and the comparison is
    # done without resampling anybody.
    omega = np.linspace(-args.omega_max, args.omega_max, args.n_omega)
    results = []

    if not args.no_spevae:
        if args.checkpoint is None:
            print("[spevae ] skipped: no --checkpoint given")
        else:
            print(f"[spevae ] running ({args.checkpoint}) ...", flush=True)
            _m = load_model(args.checkpoint)
            results.append(run_spevae(_m, mg, omega=omega, n_samples=200,
                                      min_eigenvalue=args.min_eigenvalue))
            if args.refine:
                print("[refine ] optimising z with the weights frozen ...",
                      flush=True)
                results.append(run_spevae(
                    _m, mg, omega=omega, n_samples=0,
                    min_eigenvalue=args.min_eigenvalue,
                    refine=RefineConfig(steps=args.refine_steps,
                                        min_eigenvalue=args.min_eigenvalue)))

    if not args.no_maxent:
        if not maxent_available():
            print("[maxent ] skipped: vendored ana_cont did not import")
        else:
            print("[maxent ] running ...", flush=True)
            results.append(run_maxent(mg, omega=omega,
                                      min_eigenvalue=args.min_eigenvalue))

    if not args.no_deac:
        if not deac_available():
            print("[deac   ] skipped: julia + SmoQyDEAC not reachable "
                  "(set SPEVAE_JULIA_PROJECT)")
        else:
            print(f"[deac   ] running ({args.deac_bins}x{args.deac_runs} runs, "
                  f"{args.deac_generations} generations) ...", flush=True)
            results.append(run_deac(mg, omega=omega, num_bins=args.deac_bins,
                                    runs_per_bin=args.deac_runs,
                                    generations=args.deac_generations,
                                    min_eigenvalue=args.min_eigenvalue))

    if not results:
        sys.exit("no methods ran")

    truth = None
    if truth_poles is not None:
        truth = (omega, truth_poles.spectral(torch.tensor(omega))[0].numpy())
    elif mg.reference is not None:
        ws, A = mg.reference
        # The cube ships a SmoQyDEAC continuation. It is a *reference*, not
        # truth, so it enters as another row rather than as the answer.
        from spevae.benchmark import ContinuationResult
        ref = ContinuationResult(method="deac-ref", omega=ws.double().numpy(),
                                 A=A.double().numpy(),
                                 meta={"solver": "SmoQyDEAC shipped with the cube"})
        results.append(ref.attach_reconstruction(mg, args.min_eigenvalue, "mean"))

    # Without a truth, score against a shipped third-party continuation when the
    # dataset has one -- it is the most independent curve available.  Falling
    # back to results[0] would make whichever method happened to run first the
    # yardstick, which is arbitrary and reads as a claim it is not.
    names = [r.method for r in results]
    ref_name = None
    if truth is None:
        ref_name = next((n for n in ("deac-ref", "deac", "maxent") if n in names),
                        names[0])
    cmp = compare(results, truth=truth, beta=mg.beta, reference=ref_name)
    print()
    print(format_table(cmp))

    if args.figure:
        from spevae.benchmark import plot_comparison
        label = (mg.label or "spevae benchmark")
        plot_comparison(results, mg=mg, truth=truth, title=label,
                        omega_range=(-args.omega_max, args.omega_max),
                        path=args.figure)
        print(f"\nfigure -> {args.figure}")

    if args.out:
        np.savez(args.out,
                 omega=omega,
                 **{f"A_{r.method}": r.on_grid(omega) for r in results},
                 **({"A_truth": truth[1]} if truth is not None else {}))
        with open(args.out + ".json", "w") as f:
            json.dump({"per_method": cmp["per_method"],
                       "agreement_labels": cmp["agreement"]["labels"],
                       "agreement_l2": cmp["agreement"]["l2"].tolist()},
                      f, indent=2, default=str)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
