#!/usr/bin/env python
"""Short end-to-end pretraining run: does the pipeline actually learn?

Not a production run -- a few minutes on CPU, small model, narrow beta range.
Its job is to show that the objective moves the metrics in the right direction
and that nothing in the physics/model/loss stack is silently broken.
"""

from __future__ import annotations

import argparse
import json

import torch

from spevae.data import SyntheticGenerator
from spevae.losses import LossWeights
from spevae.models import SPEVAE
from spevae.training import PretrainConfig, evaluate, pretrain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast", choices=["fast", "full"],
                    help="fast: small model, big batch, GPU if present, narrow "
                         "beta -- for A/B experiments. full: production sizes.")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-poles", type=int, default=20)
    ap.add_argument("--latent-dim", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--enc-layers", type=int, default=None)
    ap.add_argument("--families", default=None,
                    help="comma-separated subset, for a focused experiment")
    ap.add_argument("--residue-mode", default="complex", choices=["real", "complex"])
    ap.add_argument("--chi2-transform", default="sqrt",
                    choices=["sqrt", "log1p", "raw"])
    ap.add_argument("--w-chi2", type=float, default=1.0)
    ap.add_argument("--w-w1", type=float, default=3.0)
    ap.add_argument("--w-l2", type=float, default=0.3)
    ap.add_argument("--w-positivity", type=float, default=1.0)
    ap.add_argument("--w-barrier", type=float, default=0.0)
    ap.add_argument("--beta-min", type=float, default=None)
    ap.add_argument("--beta-max", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=None,
                    help="stop after N evals without improvement "
                         "(fast preset default 3; 0 disables)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Presets exist because the default should be the *cheap* thing.  An A/B
    # question ("do complex residues help?") is answered by whichever setting
    # separates the arms soonest, not by production sizes -- in the run this
    # replaced, the two arms differed 14x in chi^2 by step 600 and the remaining
    # 11,400 steps would have added nothing to the verdict.
    FAST = dict(steps=4000, batch_size=128, d_model=96, enc_layers=2,
                beta_min=6.0, beta_max=10.0, patience=3)
    FULL = dict(steps=25000, batch_size=48, d_model=160, enc_layers=3,
                beta_min=2.0, beta_max=20.0, patience=0)
    defaults = FAST if args.preset == "fast" else FULL
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.device is None:
        # MPS wins once the batch is large enough to hide dispatch overhead:
        # measured 1.52 ms/sample at batch 192 against 3.98 on CPU at batch 48.
        args.device = ("mps" if (torch.backends.mps.is_available()
                                 and args.batch_size >= 96) else "cpu")
    families = args.families.split(",") if args.families else None

    torch.manual_seed(args.seed)
    model = SPEVAE(
        latent_dim=args.latent_dim, num_poles=args.num_poles, omega_c=20.0,
        d_model=args.d_model, enc_layers=args.enc_layers, dec_layers=2,
        n_heads=4, d_ff=256, n_matsubara=16, residue_mode=args.residue_mode,
    )
    gen = SyntheticGenerator(
        omega_c=20.0, beta_range=(args.beta_min, args.beta_max), max_ntau=161,
        dtau_choices=(0.1, 0.125, 0.2), device=args.device, families=families,
    )
    print(f"preset={args.preset}  device={args.device}  steps={args.steps}  "
          f"batch={args.batch_size}  d_model={args.d_model}  "
          f"beta=[{args.beta_min},{args.beta_max}]  residues={args.residue_mode}")
    print(f"model parameters: {model.num_parameters():,}")

    model = model.to(args.device)
    base = evaluate(model, gen, n_batches=8, batch_size=args.batch_size,
                    device=args.device)
    print("before:", {k: round(v, 5) for k, v in base.items()})

    def cb(kind, step, d):
        if kind == "train" and step % 100 == 0:
            print(f"  step {step:5d}  loss {d['total']:9.4f}  chi2 {d['chi2']:11.3f}  "
                  f"w1 {d['w1_omega']:.4f}  l2 {d['l2_rel']:.4f}  kl {d['kl']:.3f}  "
                  f"minA {d.get('min_A', 0.0):+.2e}  beta {d['beta']:.2f}  {d['family']}")
        elif kind == "stop":
            print(f"  [early stop @ {step}] {d['reason']}  (best {d['best']:.4f})")
        elif kind == "eval":
            print(f"  [eval @ {step}] " + "  ".join(
                f"{k}={v:.4f}" for k, v in d.items() if "/" not in k))

    cfg = PretrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=6e-4, warmup=60,
        eval_every=max(args.steps // 12, 1), eval_batches=8, device=args.device,
        early_stop_patience=args.patience,
        seed=args.seed,
        weights=LossWeights(chi2=args.w_chi2, w1=args.w_w1, l2=args.w_l2,
                            chi2_transform=args.chi2_transform,
                            positivity=args.w_positivity, barrier=args.w_barrier),
    )
    hist = pretrain(model, gen, cfg, callback=cb)

    final = evaluate(model, gen, n_batches=16, batch_size=args.batch_size,
                     device=args.device)
    print("after: ", {k: round(v, 5) for k, v in final.items()})
    print(f"wall time: {hist['wall_time']:.1f}s")

    imp = (base["w1_omega"] - final["w1_omega"]) / base["w1_omega"]
    print(f"\nW1 (physical omega) improved by {100 * imp:.1f}%  "
          f"[{base['w1_omega']:.4f} -> {final['w1_omega']:.4f}]")

    if args.out:
        from spevae import save_model
        save_model(model, args.out, cli=vars(args), metrics=final,
                   metrics_before=base)
        print(f"saved -> {args.out}")
        with open(args.out + ".metrics.json", "w") as f:
            json.dump({"before": base, "after": final}, f, indent=2)


if __name__ == "__main__":
    main()
