r"""Stage 2: weight-level fine-tuning on one cell's measured DQMC bins.

Where Stage 1 learns the generic inverse map from a synthetic prior, Stage 2
adapts the *weights* to the one simulation cell the user actually cares about
-- its noise level, its correlation structure, its spectral family -- using no
ground truth at all.  The self-supervision is the physics: the calibrated
:math:`\chi^2` against the measured covariance, the KL to the latent prior,
and the positivity structure in both spaces (spectral penalty + the
complete-monotonicity penalties on :math:`G, G'', G''''`).

Relation to latent refinement
-----------------------------
:mod:`spevae.refine` moves only :math:`z` and is the cheap, always-safe option
for a single spectrum.  Fine-tuning moves weights and is worth it when a cell
has *many* momenta / orbitals to continue, or sits outside the pretraining
distribution.  The two compose: fine-tune once per dataset, refine per input.

What to train
-------------
``train="encoder_only"`` is the default, on a measured failure mode of the
predecessor project: a free :math:`\chi^2` fine-tune discards the sparse pole
budget Stage 1 installs (pole usage 2.30 :math:`\to` 11.08 across Stage 2 on
identical inputs -- surplus poles wake up to fit noise).  Freezing the decoder
keeps the latent :math:`\to` pole map and its learned sparsity; the encoder
adapts to the cell's noise.  ``"all"`` and ``"decoder_only"`` are available
and reported, not forbidden.

Divergence guard
----------------
Real bins contain outliers, and one bad minibatch early in fine-tuning can
throw a pretrained model into a region it never recovers from.  Every step is
screened: a non-finite or grossly out-of-distribution loss (relative to an EMA
of *accepted* losses -- a bad batch must not inflate its own threshold) is
skipped before ``backward()``; a non-finite gradient norm is skipped after.
If an entire epoch-equivalent is skipped, training stops and the best
checkpoint so far is restored.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import torch

from ..data.real import MeasuredGreens
from ..losses import (
    chi2 as chi2_loss,
    floatify,
    kl_divergence,
    negativity_penalties,
    negativity_stencil_weights,
    noise_consistency,
    spectral_positivity,
    width_barrier,
)
from ..models.vae import SPEVAE, active_units

__all__ = ["FinetuneConfig", "finetune"]

_EMA_BETA = 0.95           #: smoothing of the accepted-loss EMA in the guard
_GUARD_ABS = 100.0         #: absolute loss ceiling before any EMA exists
_GUARD_REL = 100.0         #: multiple of the EMA that flags a divergent batch


@dataclass
class FinetuneConfig:
    """Stage-2 configuration.  Defaults follow the pretraining loop where the
    two share a concept, and the predecessor project's measured settings where
    they do not."""

    steps: int = 2_000
    batch_size: int = 32
    lr: float = 1e-4                    # an order below Stage 1: adapting, not learning
    weight_decay: float = 0.0           # decay fights the pretrained prior; off
    warmup: int = 100
    grad_clip: float = 1.0
    train: str = "encoder_only"         # "encoder_only" | "decoder_only" | "all"
    min_eigenvalue: float = 0.0         # correlation floor for the whitener

    # loss weights
    w_chi2: float = 1.0
    chi2_transform: str = "sqrt"
    w_kl: float = 1.0
    free_bits: float = 0.05
    w_consistency: float = 0.1          # bins are free noise realisations: use them
    w_neg: float = 1.0                  # the three G-monotonicity penalties
    w_positivity: float = 1.0           # spectral-side ReLU(-A)^2 (complex mode)
    w_barrier: float = 0.0

    eval_every: int = 100
    early_stop_patience: int = 10       # evals without val improvement
    early_stop_min_delta: float = 1e-3
    seed: int = 0
    device: str = "cpu"


def _set_trainable(model: SPEVAE, mode: str) -> list[str]:
    """Freeze parameter groups; returns the names left trainable."""
    if mode not in ("encoder_only", "decoder_only", "all"):
        raise ValueError(f"train={mode!r}; expected 'encoder_only', "
                         f"'decoder_only' or 'all'")
    for p in model.parameters():
        p.requires_grad_(True)
    if mode == "encoder_only":
        for p in model.decoder.parameters():
            p.requires_grad_(False)
    elif mode == "decoder_only":
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    return [n for n, p in model.named_parameters() if p.requires_grad]


def _lr_at(step: int, cfg: FinetuneConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(cfg.warmup, 1)
    t = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    return cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))))


@torch.no_grad()
def _validate(model: SPEVAE, mg: MeasuredGreens, cov: dict, device) -> dict:
    """Deterministic (z = mu) validation on the bin mean.

    The scheduler and early stopping need a noise-free signal, so validation
    never samples the latent.  ``chi2_ratio`` is measured against the
    *calibrated* target for this whitener -- 1.0 is not the right yardstick on
    shrunk covariances (see ``MeasuredGreens.covariance``).
    """
    was_training = model.training
    model.eval()
    dtype = next(model.parameters()).dtype
    G_mean = mg.mean.unsqueeze(0).to(device, dtype)
    sigma = mg.sigma_mean.unsqueeze(0).to(device, dtype)
    x = mg.x.to(device, dtype)
    out = model(G_mean, x, mg.beta, sigma=sigma, deterministic=True)
    n = cov["n_independent"]
    c = float(chi2_loss(out.G_recon, G_mean, sigma,
                        inv_sqrt_R=cov["inv_sqrt_R"].to(device, G_mean.dtype),
                        n_independent=n))
    # The residual is divided by sigma_mean and the mean's correlation matrix
    # is the same R as a single bin's, so the sqrt(N) factors cancel: the
    # perfect-fit expectation for the mean is chi2_target itself.
    target = float(cov["chi2_target"])
    quad = model.forward_map(mg.beta, device, G_mean.dtype).quad
    A = out.poles.spectral(quad.nodes.to(out.poles.eps.dtype))
    metrics = {
        "val_chi2": c,
        "val_chi2_target": target,
        "val_chi2_ratio": c / max(target, 1e-300),
        "val_min_A": float(A.min()),
        "effective_poles": float(out.effective_poles()[0]),
    }
    if was_training:
        model.train()
    return metrics


def finetune(
    model: SPEVAE,
    mg: MeasuredGreens,
    cfg: FinetuneConfig,
    callback=None,
) -> dict:
    r"""Fine-tune ``model`` on the bins of one measured cell, in place.

    Returns a history dict (``train``/``eval`` records, guard statistics, wall
    time).  The model ends at the best validation checkpoint seen, not at the
    last step.

    Notes
    -----
    * The :math:`\chi^2` term scores individual bins against the *per-bin*
      noise (``mg.sigma``), whitened by the block whitener from
      :meth:`MeasuredGreens.covariance` -- the same calibrated object every
      other part of the package uses.
    * The consistency term pairs two disjoint minibatches of bins.  Bins *are*
      independent noise realisations of one spectrum, so this is the
      "different noisy G, same A" requirement applied to real data, with no
      synthetic pairing needed.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    torch.manual_seed(cfg.seed)
    rng = torch.Generator().manual_seed(cfg.seed)

    trainable = _set_trainable(model, cfg.train)
    if not trainable:
        raise ValueError("nothing left to train")

    cov = mg.covariance(cfg.min_eigenvalue)
    n_ind = cov["n_independent"]
    dtype = torch.float32
    bins = mg.bins.to(device, dtype)
    sigma_bin = mg.sigma.unsqueeze(0).to(device, dtype)        # (1, L) broadcasts
    inv_sqrt_R = cov["inv_sqrt_R"].to(device, dtype)
    x = mg.x.to(device, dtype)
    stencils = {k: v.to(device, dtype)
                for k, v in negativity_stencil_weights(mg.sigma.pow(2)).items()}
    quad = model.forward_map(mg.beta, device, dtype).quad

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.98))

    history: dict = {"train": [], "eval": [], "trainable": trainable,
                     "skipped": 0, "stopped_early": False, "stopped_reason": ""}
    best = {"metric": float("inf"), "state": None, "step": -1}
    stale = 0
    ema_loss = None
    skipped_run = 0                      # consecutive skips
    epoch_len = max(mg.n_bins // cfg.batch_size, 1)
    t0 = time.time()
    model.train()

    def draw(n: int) -> torch.Tensor:
        idx = torch.randperm(mg.n_bins, generator=rng)[:n]
        return bins[idx]

    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)

        G = draw(min(cfg.batch_size, mg.n_bins))
        out = model(G, x, mg.beta, sigma=sigma_bin.expand_as(G))

        raw_chi2 = chi2_loss(out.G_recon, G, sigma_bin,
                             inv_sqrt_R=inv_sqrt_R, n_independent=n_ind)
        if cfg.chi2_transform == "sqrt":
            L_chi2 = torch.sqrt(raw_chi2.clamp_min(1e-12))
        elif cfg.chi2_transform == "log1p":
            L_chi2 = torch.log1p(raw_chi2)
        elif cfg.chi2_transform == "raw":
            L_chi2 = raw_chi2
        else:
            raise ValueError(f"unknown chi2_transform {cfg.chi2_transform!r}")

        L_kl = kl_divergence(out.mu, out.logvar, free_bits=cfg.free_bits)
        total = cfg.w_chi2 * L_chi2 + cfg.w_kl * L_kl
        parts = {"chi2": raw_chi2.detach(), "kl": L_kl.detach()}

        if cfg.w_consistency > 0.0 and mg.n_bins >= 2 * cfg.batch_size:
            G2 = draw(min(cfg.batch_size, mg.n_bins))
            mu2, lv2 = model.encode(G2, x, mg.beta,
                                    sigma=sigma_bin.expand_as(G2))[:2]
            L_cons = noise_consistency(out.mu, out.logvar, mu2, lv2)
            total = total + cfg.w_consistency * L_cons
            parts["consistency"] = L_cons.detach()

        if cfg.w_neg > 0.0:
            neg = negativity_penalties(out.G_recon, stencils)
            for k, v in neg.items():
                total = total + cfg.w_neg * v
                parts[k] = v.detach()

        if cfg.w_positivity > 0.0 and model.residue_mode != "real":
            A = out.poles.spectral(quad.nodes.to(out.poles.eps.dtype))
            L_pos = spectral_positivity(A, quad)
            total = total + cfg.w_positivity * L_pos
            parts["positivity"] = L_pos.detach()

        if cfg.w_barrier > 0.0:
            L_bar = width_barrier(out.poles.gamma, model.gamma_min)
            total = total + cfg.w_barrier * L_bar
            parts["barrier"] = L_bar.detach()

        # ---- divergence guard, phase 1: screen before backward ---- #
        # One small device sync per step, and it is the guard's job to look:
        # screening cannot be done without reading the number.
        loss_val = float(total.detach())
        threshold = _GUARD_ABS if ema_loss is None else max(_GUARD_ABS,
                                                            _GUARD_REL * ema_loss)
        if not math.isfinite(loss_val) or loss_val > threshold:
            history["skipped"] += 1
            skipped_run += 1
            if skipped_run >= epoch_len:
                history["stopped_early"] = True
                history["stopped_reason"] = (
                    f"{skipped_run} consecutive divergent batches "
                    f"(last loss {loss_val:.3g}, threshold {threshold:.3g})")
                break
            continue

        opt.zero_grad(set_to_none=True)
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), cfg.grad_clip)

        # ---- phase 2: screen after backward ---- #
        if not torch.isfinite(gnorm):
            opt.zero_grad(set_to_none=True)
            history["skipped"] += 1
            skipped_run += 1
            if skipped_run >= epoch_len:
                history["stopped_early"] = True
                history["stopped_reason"] = "non-finite gradients for a full epoch"
                break
            continue

        opt.step()
        skipped_run = 0
        # The EMA sees only *accepted* losses, so an outlier cannot raise the
        # very threshold that should have rejected it.
        ema_loss = loss_val if ema_loss is None else \
            _EMA_BETA * ema_loss + (1.0 - _EMA_BETA) * loss_val

        if step % max(cfg.eval_every // 4, 1) == 0 or step == cfg.steps - 1:
            rec = floatify(parts)
            rec.update(step=step, total=loss_val, lr=opt.param_groups[0]["lr"],
                       grad_norm=float(gnorm), elapsed=time.time() - t0,
                       active_units=float(active_units(out.mu)))
            history["train"].append(rec)
            if callback is not None:
                callback("train", step, rec)

        if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            m = _validate(model, mg, cov, device)
            m["step"] = step
            history["eval"].append(m)
            if callback is not None:
                callback("eval", step, m)

            v = m["val_chi2_ratio"]
            if v < best["metric"] * (1.0 - cfg.early_stop_min_delta):
                best = {"metric": v, "step": step,
                        "state": copy.deepcopy(model.state_dict())}
                stale = 0
            else:
                stale += 1
                if cfg.early_stop_patience and stale >= cfg.early_stop_patience:
                    history["stopped_early"] = True
                    history["stopped_reason"] = (
                        f"val_chi2_ratio did not improve for {stale} evals")
                    break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    history["best_step"] = best["step"]
    history["best_val_chi2_ratio"] = best["metric"]
    history["final"] = _validate(model, mg, cov, device)
    history["wall_time"] = time.time() - t0
    # Restore a fully trainable model: the freeze is a property of this stage,
    # not of the weights it leaves behind.
    for p in model.parameters():
        p.requires_grad_(True)
    return history
