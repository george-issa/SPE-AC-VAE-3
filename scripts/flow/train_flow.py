"""Train the conditional flow-matching posterior (models/flow.py).

Same training distribution as the VAE control it is compared with: the M1
recipe's pole prior (beta = 8, dispersive residue patterns, default noise and
grid draws), batch 128.  Only the model and its objective differ.

    python scripts/flow/train_flow.py --out out/flow/F_6k.pt --steps 6000 --device mps
"""
import argparse
import copy
import json
import math
import sys
import time

sys.path.insert(0, "src")

import torch

from spevae.data.pole_prior import PolePriorGenerator
from spevae.models.flow import FlowPosterior, save_flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--enc-layers", type=int, default=2)
    ap.add_argument("--flow-layers", type=int, default=4)
    ap.add_argument("--no-ot", action="store_true", help="index pairing instead of Hungarian")
    ap.add_argument("--smooth-bank", default=None)
    ap.add_argument("--p-smooth", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    kw = {}
    if args.p_smooth > 0:
        kw = dict(smooth_bank=args.smooth_bank, p_smooth=args.p_smooth)
    gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive",
                             beta_range=(8.0, 8.0), device=dev, **kw)
    gen.seed_scalars(args.seed)
    rng = torch.Generator(device=dev).manual_seed(args.seed)
    grng = torch.Generator().manual_seed(args.seed + 7)

    model = FlowPosterior(num_poles=12, d_model=args.d_model, enc_layers=args.enc_layers,
                          flow_layers=args.flow_layers)
    # standardise the token coordinates on prior draws held apart from training
    with gen.fixed_scalars(10_000):
        fit = [gen.batch(256, torch.Generator(device=dev).manual_seed(10_000 + k)).poles_true
               for k in range(8)]
    model.coords.fit([{k: v.cpu() for k, v in f.items() if k != "n_true"} for f in fit])
    model.to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        q = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q)))

    # fixed held-out batches for the eval loss (bit-identical every eval)
    with gen.fixed_scalars(20_000):
        held = [gen.batch(args.batch_size,
                          torch.Generator(device=dev).manual_seed(20_000 + k))
                for k in range(4)]

    def eval_loss(m):
        m.eval()
        g = torch.Generator().manual_seed(123)
        with torch.no_grad():
            return sum(float(m.loss(b, ot_pairing=not args.no_ot, generator=g))
                       for b in held) / len(held)

    print(f"flow posterior: {model.num_parameters():,} parameters, {args.steps} steps, "
          f"batch {args.batch_size}, device {dev}", flush=True)
    t0 = time.time()
    hist = []
    for step in range(args.steps):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        model.train()
        batch = gen.batch(args.batch_size, rng)
        loss = model.loss(batch, ot_pairing=not args.no_ot, generator=grng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        if step % 100 == 0:
            print(f"[{step:>6}] loss {float(loss):.4f}  lr {lr_at(step):.2e}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if (step + 1) % max(args.steps // 12, 1) == 0 or step + 1 == args.steps:
            el, ee = eval_loss(model), eval_loss(ema)
            hist.append({"step": step + 1, "eval_loss": el, "eval_loss_ema": ee})
            print(f"[eval @ {step + 1}] loss {el:.4f}  ema {ee:.4f}", flush=True)
            save_flow(ema.cpu(), args.out + ".partial", step=step + 1, history=hist)
            ema.to(dev)
    recipe = {k: v for k, v in vars(args).items()}
    save_flow(ema.cpu(), args.out, steps=args.steps, history=hist, recipe=recipe,
              wall_seconds=time.time() - t0)
    print(f"saved -> {args.out}  (wall {time.time() - t0:.0f}s)")
    print(json.dumps(hist[-1]))


if __name__ == "__main__":
    main()
