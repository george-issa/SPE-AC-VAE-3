"""Train the pole-count classifier p(n | G, sigma, beta) on the step-3 prior.

    python scripts/step3/train_count.py --out out/step3/count_6k.pt --steps 6000 --device mps \
        --smooth-bank out/step3/bank/label_bank_trusted.pt

Same data recipe and schedule as `scripts/flow/train_flow.py`.  Prints, on
fixed held-out batches: cross-entropy (nats) against the entropy of the
prior's count marginal, top-1 accuracy, and the probability given to the true
count.
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
from spevae.models.count import CountPosterior, save_count


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
    ap.add_argument("--smooth-bank", required=True)
    ap.add_argument("--p-smooth", type=float, default=0.35)
    ap.add_argument("--p-exact", type=float, default=0.15)
    ap.add_argument("--beta-choices", type=float, nargs="+",
                    default=[1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20])
    ap.add_argument("--noise-range", type=float, nargs=2, default=(1e-5, 3e-2))
    ap.add_argument("--dtau-choices", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.125, 0.2, 0.25, 0.5])
    ap.add_argument("--max-ntau", type=int, default=401)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive", beta_range=(8.0, 8.0),
                             device=dev, p_exact=args.p_exact, smooth_bank=args.smooth_bank,
                             p_smooth=args.p_smooth, beta_choices=tuple(args.beta_choices),
                             noise_scale_range=tuple(args.noise_range),
                             dtau_choices=tuple(args.dtau_choices), max_ntau=args.max_ntau)
    gen.seed_scalars(args.seed)
    rng = torch.Generator(device=dev).manual_seed(args.seed)
    model = CountPosterior(num_poles=12, d_model=args.d_model, enc_layers=args.enc_layers).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        q = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q)))

    with gen.fixed_scalars(20_000):
        held = [gen.batch(args.batch_size, torch.Generator(device=dev).manual_seed(20_000 + k))
                for k in range(16)]
    n_all = torch.cat([b.poles_true["n_true"].cpu().long() for b in held]).clamp(1, 12) - 1
    marg = torch.bincount(n_all, minlength=12).double() / n_all.numel()
    h_prior = float(-(marg[marg > 0] * marg[marg > 0].log()).sum())

    def evaluate(m):
        m.eval()
        ce = acc = pt = 0.0
        with torch.no_grad():
            for b in held:
                lg = m.logits(b.G, b.x, b.beta, sigma=b.sigma).cpu().double()
                n = b.poles_true["n_true"].cpu().long().clamp(1, 12) - 1
                lp = torch.log_softmax(lg, -1)
                ce += float(-lp.gather(1, n[:, None]).mean())
                acc += float((lg.argmax(-1) == n).double().mean())
                pt += float(lp.gather(1, n[:, None]).exp().mean())
        k = len(held)
        return {"cross_entropy": ce / k, "top1": acc / k, "p_true": pt / k}

    print(f"count classifier: {sum(p.numel() for p in model.parameters()):,} parameters, {args.steps} steps, "
          f"device {dev}; entropy of the prior's count marginal {h_prior:.3f} nats "
          f"(most common count {float(marg.max()):.2f})", flush=True)
    t0 = time.time()
    hist = []
    for step in range(args.steps):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        model.train()
        loss = model.loss(gen.batch(args.batch_size, rng))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
        if step % 200 == 0:
            print(f"[{step:>6}] loss {float(loss):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if (step + 1) % max(args.steps // 6, 1) == 0 or step + 1 == args.steps:
            e = {"step": step + 1, **evaluate(ema), "prior_entropy": h_prior}
            hist.append(e)
            print(f"[eval @ {step + 1}] " + "  ".join(f"{k} {v:.3f}" for k, v in e.items() if k != "step"),
                  flush=True)
    recipe = dict(vars(args))
    save_count(ema.cpu(), args.out, steps=args.steps, history=hist, recipe=recipe,
               prior_count_marginal=marg.tolist(), wall_seconds=time.time() - t0)
    print(f"saved -> {args.out}  (wall {time.time() - t0:.0f}s)")
    print(json.dumps(hist[-1]))


if __name__ == "__main__":
    main()
