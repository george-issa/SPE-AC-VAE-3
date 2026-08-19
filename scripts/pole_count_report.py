#!/usr/bin/env python
"""Compare checkpoints trained at different pole counts, fairly.

Every number comes from the same frozen held-out stream (the seed pins
beta/dtau/family/noise draws and the device generator), so rows differ only in
the model.  Reports the spectral metrics plus the *effective* pole count --
the participation ratio 1/sum(a_p^2) -- whose prediction from the minimal-pole
literature is that the data's noise level, not the parameter P, sets how many
poles a fit can spend.

    python scripts/pole_count_report.py out/model-fast.pt out/model-fast-p24.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spevae import load_model  # noqa: E402
from spevae.data.synthetic import SyntheticGenerator  # noqa: E402
from spevae.physics.poles import effective_pole_count  # noqa: E402
from spevae.training import evaluate  # noqa: E402


def report(path: str, gen: SyntheticGenerator, n_batches: int, batch_size: int,
           seed: int) -> dict:
    model = load_model(path)
    m = evaluate(model, gen, n_batches=n_batches, batch_size=batch_size,
                 seed=seed)
    # Effective poles over the same frozen stream, deterministic decode.
    effs = []
    rng = torch.Generator().manual_seed(seed)
    with gen.fixed_scalars(seed):
        batches = [gen.batch(batch_size, rng) for _ in range(n_batches)]
    with torch.no_grad():
        for b in batches:
            out = model(b.G, b.x, b.beta, sigma=b.sigma, deterministic=True)
            a_unit = out.poles.a / out.poles.a.sum(-1, keepdim=True).clamp_min(1e-30)
            effs.append(effective_pole_count(a_unit))
    eff = torch.cat(effs)
    return {
        "path": path,
        "P": model.num_poles,
        "params": model.num_parameters(),
        "w1": m["w1_omega"],
        "l2": m["l2_rel"],
        "chi2_geo": m.get("chi2_geo", float("nan")),
        "AU": f"{m['active_units']:.2f}/{int(m['latent_dim'])}",
        "eff_mean": float(eff.mean()),
        "eff_p90": float(eff.quantile(0.9)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12_345)
    ap.add_argument("--beta-min", type=float, default=6.0)
    ap.add_argument("--beta-max", type=float, default=10.0)
    ap.add_argument("--omega-c", type=float, default=20.0)
    args = ap.parse_args()

    gen = SyntheticGenerator(omega_c=args.omega_c,
                             beta_range=(args.beta_min, args.beta_max))
    rows = [report(p, gen, args.batches, args.batch_size, args.seed)
            for p in args.checkpoints]

    print(f"{'checkpoint':<28} {'P':>3} {'params':>10} {'W1':>8} {'l2rel':>8} "
          f"{'chi2geo':>9} {'AU':>9} {'eff mean':>9} {'eff p90':>8}")
    print("-" * 100)
    for r in rows:
        print(f"{os.path.basename(r['path']):<28} {r['P']:>3} {r['params']:>10,} "
              f"{r['w1']:>8.4f} {r['l2']:>8.4f} {r['chi2_geo']:>9.4g} "
              f"{r['AU']:>9} {r['eff_mean']:>9.2f} {r['eff_p90']:>8.2f}")
    print("\nSame frozen eval stream for every row; rows differ only in the model.")


if __name__ == "__main__":
    main()
