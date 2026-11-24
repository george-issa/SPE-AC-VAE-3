"""Step 2: smooth label-bank truths inside the pole prior."""

import pytest
import torch

from spevae.data.label_bank import build_label_bank
from spevae.data.pole_prior import PolePriorGenerator
from spevae.physics import PoleGreens


@pytest.fixture(scope="module")
def bank8():
    return build_label_bank(n_per_family=2, seed=5, beta_range=(8.0, 8.0),
                            families=("semicircle", "gauss_mixture"), P_max=12)


def _gen(**kw):
    g = PolePriorGenerator(beta_range=(8.0, 8.0), dtau_choices=(0.125,),
                           residue_mode="dispersive", **kw)
    g.seed_scalars(3)
    return g


def test_p_smooth_zero_leaves_the_prior_bit_identical(bank8):
    a = _gen().batch(16, torch.Generator().manual_seed(1))
    b = _gen(smooth_bank=bank8, p_smooth=0.0).batch(16, torch.Generator().manual_seed(1))
    assert torch.equal(a.G, b.G)
    for k in ("eps", "gamma", "a", "b"):
        assert torch.equal(a.poles_true[k], b.poles_true[k])


def test_smooth_items_carry_bank_labels_and_exact_inputs(bank8):
    batch = _gen(smooth_bank=bank8, p_smooth=1.0).batch(8, torch.Generator().manual_seed(2))
    assert batch.meta["n_smooth"] == 8
    assert all(f.startswith("smooth:") for f in batch.family)
    lab = batch.poles_true
    for i in range(8):
        m = int(lab["n_true"][i])
        # the label is one of the bank's, in the training convention (nu = beta*omega)
        hits = [j for j in range(len(bank8)) if int(bank8.n_true[j]) == m and
                torch.allclose(lab["eps"][i, :m], bank8.eps[j, :m] * 8.0)]
        assert hits, i
        j = hits[0]
        assert torch.allclose(lab["a"][i, :m], bank8.a[j, :m])
        assert torch.allclose(lab["b"][i, :m], bank8.b[j, :m])
        assert float(lab["a"][i, m:].abs().sum()) == 0.0          # zero-weight padding
        # the INPUT is the exact smooth truth, not the label's rendering
        tr = bank8.truth(j)
        nu = batch.quad.nodes.double().numpy()
        A_exact = torch.tensor(tr.A(nu / 8.0) / 8.0, dtype=batch.A_true.dtype)
        assert torch.allclose(batch.A_true[i], A_exact)
    K = PoleGreens(batch.quad).kernel_matrix(batch.x)
    assert torch.allclose(batch.G_clean, (batch.A_true * batch.quad.weights) @ K)
    # sum rule of the inputs: G(0) + G(beta) = 1
    s = batch.G_clean[:, 0] + batch.G_clean[:, -1]
    assert torch.allclose(s, torch.ones_like(s), atol=2e-3)


def test_mixture_fraction_and_beta_guard(bank8):
    g = _gen(smooth_bank=bank8, p_smooth=0.3)
    n = sum(g.batch(32, torch.Generator().manual_seed(k)).meta["n_smooth"] for k in range(20))
    assert 0.2 < n / 640 < 0.4
    wrong = PolePriorGenerator(beta_range=(10.0, 10.0), smooth_bank=bank8, p_smooth=0.5)
    with pytest.raises(ValueError, match="no labels at beta"):
        wrong.batch(4, torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="needs a smooth_bank"):
        PolePriorGenerator(p_smooth=0.5)
