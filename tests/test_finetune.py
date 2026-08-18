"""Stage-2 weight-level fine-tuning.

Follows the pattern of test_refine: data generated from the model's own
decoder, so improvement is attributable to the optimisation rather than to
checkpoint luck.  What must hold: fine-tuning improves the calibrated fit,
respects the freeze policy exactly, preserves every structural constraint
(they are architectural, but the test says so explicitly), and the divergence
guard refuses to walk into NaN-land.
"""

import copy

import pytest
import torch

from spevae.data import from_bins
from spevae.models import SPEVAE
from spevae.training import FinetuneConfig, finetune

from test_refine import BETA, decodable_dataset, tiny_model


def perturbed_model(seed=0, residue_mode="complex"):
    """A model whose weights have been knocked away from the data-generating
    ones -- the situation fine-tuning exists for."""
    model = tiny_model(seed=seed, residue_mode=residue_mode)
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return model


def small_cfg(**kw):
    base = dict(steps=120, batch_size=16, lr=3e-4, warmup=10, eval_every=30,
                early_stop_patience=0, seed=0)
    base.update(kw)
    return FinetuneConfig(**base)


def test_finetune_improves_the_calibrated_fit():
    model = perturbed_model(seed=3)
    mg, _, _ = decodable_dataset(model, seed=3, n_tau=41, n_bins=400)
    cov = mg.covariance()
    from spevae.training.finetune import _validate
    before = _validate(model, mg, cov, torch.device("cpu"))
    hist = finetune(model, mg, small_cfg())
    after = hist["final"]
    assert after["val_chi2"] < before["val_chi2"], (before, after)
    assert hist["best_step"] >= 0
    assert hist["skipped"] == 0


@pytest.mark.parametrize("mode,frozen_prefix", [
    ("encoder_only", "decoder."),
    ("decoder_only", "encoder."),
])
def test_freeze_policy_is_exact(mode, frozen_prefix):
    model = perturbed_model(seed=4)
    mg, _, _ = decodable_dataset(model, seed=4, n_tau=41, n_bins=300)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    hist = finetune(model, mg, small_cfg(steps=60, train=mode, eval_every=0,
                                         early_stop_patience=0))
    moved_frozen, moved_free = [], []
    for k, v in model.state_dict().items():
        if torch.equal(v, before[k]):
            continue
        (moved_frozen if k.startswith(frozen_prefix) else moved_free).append(k)
    assert not moved_frozen, f"frozen parameters moved: {moved_frozen[:4]}"
    assert moved_free, "nothing trained at all"
    assert all(not n.startswith(frozen_prefix) for n in hist["trainable"])
    # the freeze must not outlive the stage
    assert all(p.requires_grad for p in model.parameters())


def test_structural_constraints_survive_finetuning():
    model = perturbed_model(seed=5, residue_mode="complex")
    mg, _, _ = decodable_dataset(model, seed=5, n_tau=41, n_bins=300)
    finetune(model, mg, small_cfg(steps=80, eval_every=0))
    with torch.no_grad():
        out = model(mg.bins[:8].float(), mg.x.float(), mg.beta,
                    deterministic=True)
    p = out.poles
    assert (p.gamma > 0).all(), "causality"
    assert (p.a > 0).all(), "positive weights"
    norm = out.norm.reshape(-1, 1)
    assert torch.allclose(p.a.sum(-1, keepdim=True), norm, rtol=1e-4), "sum rule"
    assert torch.allclose(p.b.sum(-1), torch.zeros(p.shape[0]), atol=1e-6), "tail"
    assert (p.b.abs() <= p.a + 1e-8).all(), "|b| <= a"
    assert (p.gamma >= model.gamma_min - 1e-9).all(), "width floor"


def test_divergence_guard_skips_and_stops():
    """A loss forced to be non-finite must be skipped, and a full epoch of
    skips must stop the run with the reason recorded -- not raise, not NaN the
    weights."""
    model = perturbed_model(seed=6)
    mg, _, _ = decodable_dataset(model, seed=6, n_tau=41, n_bins=200)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    # Poison one encoder weight so every forward produces NaN.
    with torch.no_grad():
        next(model.encoder.parameters()).fill_(float("nan"))
        poisoned = {k: v.clone() for k, v in model.state_dict().items()}
    hist = finetune(model, mg, small_cfg(steps=200, eval_every=0))
    assert hist["stopped_early"]
    assert hist["skipped"] > 0
    assert "divergent" in hist["stopped_reason"] or "non-finite" in hist["stopped_reason"]
    # no optimiser step was taken on garbage
    for k, v in model.state_dict().items():
        ref = poisoned[k]
        both_nan = torch.isnan(v) & torch.isnan(ref)
        assert torch.equal(v[~both_nan], ref[~both_nan]), f"{k} moved on NaN loss"
    del before


def test_negativity_penalties_have_correct_shapes_and_zero_floor():
    from spevae.losses import negativity_penalties, negativity_stencil_weights
    var0 = torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0])   # pinned endpoints
    w = negativity_stencil_weights(var0)
    assert w["inv_var0"][0] == 0.0 and w["inv_var0"][-1] == 0.0
    assert w["inv_var2"].shape == (3,) and w["inv_var4"].shape == (1,)
    # a strictly positive, convex G incurs zero penalty
    tau = torch.linspace(0, 1, 5)
    G = torch.exp(-3.0 * tau).unsqueeze(0)
    pen = negativity_penalties(G, w)
    assert all(float(v) == 0.0 for v in pen.values())
    # negative excursions are charged, weighted by 1/var
    G_bad = G.clone()
    G_bad[0, 2] = -0.5
    pen_bad = negativity_penalties(G_bad, w)
    assert float(pen_bad["neg_G"]) > 0.0
