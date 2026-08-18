r"""Stage 1: amortised supervised pretraining.

What this stage is for
----------------------
It produces *the shipped model*: one set of weights that continues anybody's
:math:`G(\tau)` at any :math:`(\beta, \Delta\tau)` and noise level, in a single
forward pass, with no per-dataset optimisation.  That is the deliverable a
package can actually offer.  This is a deliberate inversion of the more common
arrangement, in which a model is fitted separately to each dataset by minimising
:math:`\chi^2` -- a procedure that has no prior, so nothing prevents it from
choosing whichever of the many :math:`\chi^2 \approx 1` spectra the optimiser
happens to reach first.

Because the target is known here, the objective can constrain the *spectrum*
(:math:`W_1` and :math:`L^2`) and not only the reconstruction.  That is the only
stage at which the ill-posedness can actually be beaten: on real data no loss
can distinguish two spectra that produce the same :math:`G(\tau)` to within
noise, so the ability to tell them apart has to be learned beforehand, from
cases where the answer was known.

Scaling
-------
The generator is infinite, so there is no fixed epoch and no overfitting to a
finite training set; longer runs and larger models monotonically buy accuracy
until the network saturates the information the data contains.  ``d_model``,
``enc_layers``/``dec_layers`` and ``steps`` are the knobs.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import torch

from ..data.synthetic import SyntheticGenerator, TrainingBatch
from ..losses import (LossWeights, floatify, pretrain_loss, spectral_l2,
                      spectral_w1)
from ..models.vae import SPEVAE, active_units

__all__ = ["PretrainConfig", "pretrain", "evaluate"]


@dataclass
class PretrainConfig:
    """Stage-1 settings.

    ``early_stop_patience`` (in *evaluations*, not steps) ends a run once the
    held-out metric stops improving by ``early_stop_min_delta`` in relative
    terms.  Set it for exploratory runs: the step count that answers a question
    is not knowable in advance, and picking one by feel is how a 4-minute
    decision turns into an hour.  Leave it 0 for a production run, where you
    want the whole cosine schedule.
    """

    steps: int = 20_000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 100
    eval_every: int = 1_000
    eval_batches: int = 24
    early_stop_patience: int = 0
    early_stop_metric: str = "w1_omega"
    early_stop_min_delta: float = 0.01
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    weights: LossWeights = field(default_factory=LossWeights)


def _lr_at(step: int, cfg: PretrainConfig) -> float:
    """Linear warmup, then cosine decay to 5% of the peak."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(cfg.warmup, 1)
    t = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    return cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))))


@torch.no_grad()
def evaluate(
    model: SPEVAE,
    generator: SyntheticGenerator,
    n_batches: int = 12,
    batch_size: int = 64,
    seed: int = 12_345,
    device: str = "cpu",
) -> dict[str, float]:
    r"""Held-out metrics, reported per spectral family as well as overall.

    An aggregate number hides the failure that matters: a model can look strong
    on smooth multi-peak spectra and be useless on gapped ones, and the average
    will not say so.  ``w1_omega`` is converted to *physical* frequency
    (:math:`W_1/\beta`) so that it is comparable across cells -- it reads
    directly as a typical peak-placement error in units of :math:`\omega`.
    """
    was_training = model.training
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    per_family: dict[str, list[tuple[float, float, float]]] = {}
    chi2s, aus, min_As = [], [], []

    for _ in range(n_batches):
        batch: TrainingBatch = generator.batch(batch_size, rng)
        out = model(batch.G, batch.x, batch.beta, sigma=batch.sigma,
                    deterministic=True)
        A = out.poles.spectral(batch.quad.nodes.to(out.poles.eps.dtype))
        w1 = float(spectral_w1(A, batch.A_true, batch.quad, beta=batch.beta))
        l2 = float(spectral_l2(A, batch.A_true, batch.quad, relative=True))
        from ..losses import chi2 as _chi2
        c = float(_chi2(out.G_recon, batch.G, batch.sigma,
                        inv_sqrt_R=batch.inv_sqrt_R,
                        n_independent=batch.G.shape[-1] - 1))
        per_family.setdefault(batch.family[0], []).append((w1, l2, c))
        chi2s.append(c)
        aus.append(float(active_units(out.mu)))
        min_As.append(float(A.min()))

    chi2s_sorted = sorted(chi2s)
    metrics = {
        # Median, not mean: chi^2 spans orders of magnitude across cells early in
        # training, and one ill-conditioned batch would otherwise define the number.
        "chi2": float(chi2s_sorted[len(chi2s_sorted) // 2]),
        # Geometric mean, which is the right average for a quantity spanning
        # decades: it uses every batch (unlike the median, which on 8 batches of
        # wildly different difficulty is itself noisy enough to move a headline
        # number by 3x between two evaluations of the *same* model).
        "chi2_geo": float(math.exp(sum(math.log(max(c, 1e-12))
                                       for c in chi2s) / len(chi2s))),
        "active_units": float(sum(aus) / len(aus)),
        # Identically 0 with real residues; strictly negative excursions are the
        # price the complex mode pays, so it must be visible in every eval.
        "min_A": float(min(min_As)),
        "latent_dim": float(model.latent_dim),
    }
    all_w1 = [v[0] for vs in per_family.values() for v in vs]
    all_l2 = [v[1] for vs in per_family.values() for v in vs]
    metrics["w1_omega"] = float(sum(all_w1) / len(all_w1))
    metrics["l2_rel"] = float(sum(all_l2) / len(all_l2))
    for fam, vs in sorted(per_family.items()):
        metrics[f"w1_omega/{fam}"] = float(sum(v[0] for v in vs) / len(vs))

    if was_training:
        model.train()
    return metrics


def pretrain(
    model: SPEVAE,
    generator: SyntheticGenerator,
    cfg: PretrainConfig,
    callback=None,
) -> dict:
    """Run Stage 1.  Returns a history dict; the model is updated in place."""
    device = torch.device(cfg.device)
    model = model.to(device)
    rng = torch.Generator(device=device).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    ema = copy.deepcopy(model).eval() if cfg.ema_decay > 0 else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)

    history: dict[str, list] = {"step": [], "loss": [], "parts": [], "eval": []}
    t0 = time.time()
    model.train()
    best_metric, stale = float("inf"), 0
    history["stopped_early"] = False

    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)

        batch = generator.batch(cfg.batch_size, rng)
        out = model(batch.G, batch.x, batch.beta, sigma=batch.sigma)
        # Encode only: the consistency term reads mu and logvar and nothing else.
        alt = (model.encode(batch.G_alt, batch.x, batch.beta, sigma=batch.sigma)[:2]
               if cfg.weights.consistency > 0 else None)
        loss, parts = pretrain_loss(out, batch, model, cfg.weights, alt_posterior=alt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if ema is not None:
            d = min(cfg.ema_decay, (step + 1) / (step + 10))    # warm up the EMA
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            # The only place the step's scalars are pulled off the device.
            parts = floatify(parts)
            parts["beta"] = batch.beta
            parts["n_tau"] = int(batch.x.numel())
            parts["family"] = batch.family[0]
            parts["lr"] = opt.param_groups[0]["lr"]
            parts["elapsed"] = time.time() - t0
            history["step"].append(step)
            history["loss"].append(parts["total"])
            history["parts"].append(parts)
            if callback is not None:
                callback("train", step, parts)

        if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            m = evaluate(ema if ema is not None else model, generator,
                         n_batches=cfg.eval_batches, batch_size=cfg.batch_size,
                         device=cfg.device)
            m["step"] = step
            history["eval"].append(m)
            if callback is not None:
                callback("eval", step, m)

            if cfg.early_stop_patience > 0:
                v = m.get(cfg.early_stop_metric, float("inf"))
                if v < best_metric * (1.0 - cfg.early_stop_min_delta):
                    best_metric, stale = v, 0
                else:
                    stale += 1
                    if stale >= cfg.early_stop_patience:
                        history["stopped_early"] = True
                        if callback is not None:
                            callback("stop", step, {
                                "reason": f"{cfg.early_stop_metric} did not improve "
                                          f"by {cfg.early_stop_min_delta:.0%} for "
                                          f"{stale} evals",
                                "best": best_metric})
                        break

    if ema is not None:
        model.load_state_dict(ema.state_dict())
    history["wall_time"] = time.time() - t0
    return history
