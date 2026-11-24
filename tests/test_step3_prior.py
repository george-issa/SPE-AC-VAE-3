"""Step-3 prior: exactly diagonalised models and a beta grid in the pole prior."""

import math

import pytest
import torch

from spevae.data.exact_models import EXACT_DISCRETE
from spevae.data.pole_prior import PolePriorGenerator


def test_exact_truths_are_valid_pole_closed_items():
    g = PolePriorGenerator(residue_mode="dispersive", beta_choices=(2.0, 8.0), p_exact=1.0)
    g.seed_scalars(0)
    b = g.batch(48, torch.Generator().manual_seed(0))
    assert b.beta in (2.0, 8.0)
    assert all(f.startswith("exact:") and f.split(":")[1] in EXACT_DISCRETE for f in b.family)
    lab = b.poles_true
    live = lab["a"] > 0
    assert torch.allclose(lab["a"].sum(-1), torch.ones(48), atol=1e-5)
    assert (lab["b"] == 0).all() and (lab["n_true"] >= 1).all() and (lab["n_true"] <= 12).all()
    floor = 1.05 * 0.1 * math.pi                                   # dimensionless width floor
    assert float(lab["gamma"][live].min()) >= floor - 1e-5
    # pole-closed: the data are the label's own Green's function, so the sum rule is exact
    assert float((b.G_clean[:, 0] + b.G_clean[:, -1] - 1).abs().max()) < 2e-5
    assert (b.G_clean > 0).all()


def test_beta_grid_and_mixture_fractions():
    g = PolePriorGenerator(residue_mode="dispersive", beta_choices=(1.0, 4.0, 20.0), p_exact=0.3,
                           noise_scale_range=(1e-5, 3e-2))
    g.seed_scalars(3)
    betas, n_exact, n = set(), 0, 0
    for k in range(12):
        b = g.batch(32, torch.Generator().manual_seed(k))
        betas.add(b.beta)
        n_exact += sum(f.startswith("exact:") for f in b.family)
        n += 32
        assert 1e-5 * 0.99 <= b.meta["noise"].scale <= 3e-2 * 1.01
    assert betas <= {1.0, 4.0, 20.0} and len(betas) >= 2
    assert 0.2 < n_exact / n < 0.4


def test_defaults_leave_the_step2_stream_untouched():
    a = PolePriorGenerator(residue_mode="dispersive", beta_range=(8.0, 8.0))
    b = PolePriorGenerator(residue_mode="dispersive", beta_range=(8.0, 8.0), p_exact=0.0,
                           beta_choices=None)
    a.seed_scalars(5)
    b.seed_scalars(5)
    x, y = a.batch(16, torch.Generator().manual_seed(1)), b.batch(16, torch.Generator().manual_seed(1))
    assert torch.equal(x.G, y.G) and x.family == y.family
    with pytest.raises(ValueError):
        PolePriorGenerator(p_exact=1.5)
