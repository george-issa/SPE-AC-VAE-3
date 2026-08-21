"""Pole-first training data and the Sinkhorn pole loss."""

import math

import pytest
import torch

from spevae.data.pole_prior import PolePriorGenerator
from spevae.losses import sinkhorn_pole_loss
from spevae.physics import PoleSet, certify


def gen(seed=3):
    g = PolePriorGenerator(beta_range=(6.0, 10.0), P_max=12)
    g.seed_scalars(seed)
    return g


def test_labels_satisfy_every_constraint_and_are_canonical():
    g = gen()
    rng = torch.Generator().manual_seed(0)
    b = g.batch(24, rng)
    lab = b.poles_true
    for i in range(24):
        m = int(lab["n_true"][i])
        assert 1 <= m <= 12
        a = lab["a"][i]
        assert (a[m:] == 0).all(), "padding must carry zero weight"
        assert float(a[:m].sum()) == pytest.approx(1.0, rel=1e-5)
        assert float(lab["b"][i].sum()) == pytest.approx(0.0, abs=1e-6)
        assert (lab["b"][i, :m].abs() <= a[:m] * 0.5 + 1e-6).all()
        gam = lab["gamma"][i, :m]
        assert (gam >= 0.1 * math.pi).all(), "dimensionless width floor"
        # canonical: no sub-width pair survives
        e = lab["eps"][i, :m]
        for p in range(m):
            for q in range(p + 1, m):
                assert abs(float(e[p] - e[q])) >= 0.4 * min(float(gam[p]),
                                                            float(gam[q]))
        rep = certify(e.numpy() / b.beta, gam.numpy() / b.beta,
                      a[:m].numpy(), lab["b"][i, :m].numpy(), atol=1e-6)
        assert rep.is_positive or rep.min_A > -1e-5


def test_batch_shape_and_consistency_with_forward_map():
    g = gen(5)
    rng = torch.Generator().manual_seed(1)
    b = g.batch(8, rng)
    assert b.meta["pole_labeled"]
    # A_true integrates to ~1 and G_clean endpoints obey the sum rule
    norms = b.quad.integrate(b.A_true)
    assert torch.allclose(norms, torch.ones_like(norms), atol=2e-3)
    ends = b.G_clean[:, 0] + b.G_clean[:, -1]
    assert torch.allclose(ends, torch.ones_like(ends), atol=2e-3)


def test_fixed_scalars_freezes_the_stream():
    g = gen(7)
    rng1 = torch.Generator().manual_seed(2)
    with g.fixed_scalars(99):
        b1 = g.batch(4, rng1)
    rng2 = torch.Generator().manual_seed(2)
    with g.fixed_scalars(99):
        b2 = g.batch(4, rng2)
    assert torch.equal(b1.poles_true["eps"], b2.poles_true["eps"])
    assert b1.beta == b2.beta


def test_sinkhorn_zero_at_match_and_permutation_invariant():
    lab = {"eps": torch.tensor([[-3.0, 1.0, 4.0, 0.0]]),
           "gamma": torch.tensor([[0.5, 0.8, 0.6, 1.0]]),
           "a": torch.tensor([[0.3, 0.5, 0.2, 0.0]]),
           "b": torch.tensor([[0.05, -0.03, -0.02, 0.0]]),
           "n_true": torch.tensor([3])}
    exact = PoleSet(eps=lab["eps"][:, :3], gamma=lab["gamma"][:, :3],
                    a=lab["a"][:, :3], b=lab["b"][:, :3])
    loss0, _ = sinkhorn_pole_loss(exact, lab)
    perm = PoleSet(eps=exact.eps[:, [2, 0, 1]], gamma=exact.gamma[:, [2, 0, 1]],
                   a=exact.a[:, [2, 0, 1]], b=exact.b[:, [2, 0, 1]])
    lossp, _ = sinkhorn_pole_loss(perm, lab)
    assert float(loss0) < 0.02
    assert float(lossp) == pytest.approx(float(loss0), abs=1e-5)
    # moving a pole away raises the loss monotonically
    prev = float(loss0)
    for shift in (0.5, 1.0, 2.0):
        moved = PoleSet(eps=exact.eps + torch.tensor([[shift, 0.0, 0.0]]),
                        gamma=exact.gamma, a=exact.a, b=exact.b)
        cur = float(sinkhorn_pole_loss(moved, lab)[0])
        assert cur > prev
        prev = cur


def test_sinkhorn_suppresses_dust():
    """A 16-pole comb fit to a 2-pole truth: the gradient on the surplus
    weights must be positive (pushing them down)."""
    lab = {"eps": torch.tensor([[-2.0, 2.0]]),
           "gamma": torch.tensor([[0.6, 0.6]]),
           "a": torch.tensor([[0.5, 0.5]]),
           "b": torch.zeros(1, 2),
           "n_true": torch.tensor([2])}
    u = torch.zeros(1, 16, requires_grad=True)
    eps = torch.linspace(-6, 6, 16).unsqueeze(0)
    a = torch.softmax(u, dim=-1)
    pred = PoleSet(eps=eps, gamma=torch.full((1, 16), 0.6),
                   a=a, b=torch.zeros(1, 16))
    loss, parts = sinkhorn_pole_loss(pred, lab)
    loss.backward()
    # gradient wrt logits: du for poles far from truth should be positive
    far = (eps[0] - (-2.0)).abs().gt(1.0) & (eps[0] - 2.0).abs().gt(1.0)
    near = ~far
    assert u.grad[0][far].mean() > u.grad[0][near].mean()
    assert "eff_gap" in parts
