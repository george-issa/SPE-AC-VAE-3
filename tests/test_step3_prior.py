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


def test_line_spectra_in_the_bank_give_exact_data_and_their_pole_label(tmp_path):
    """An exact line spectrum in the label bank: the training data are the lines' own G(tau)
    in closed form, the target is the canonical pole label, the node spectrum is the lines
    binned onto the nodes."""
    import numpy as np
    from spevae.data import exact_models as X
    from spevae.data.label_bank import LabelBank, draw_bank_truths, extract_smooth_label
    beta = 8.0
    truths = draw_bank_truths(2, seed=11, families=("holstein_atom",), beta_choices=(beta,))
    assert all(isinstance(t, X.DiscreteSpectrum) for t in truths)
    labs = [extract_smooth_label(t, P_max=12) for t in truths]
    P, n = 12, len(truths)
    cols = {k: torch.zeros(n, P) for k in ("eps", "gamma", "a", "b")}
    cols["gamma"] += 1.0
    for row, lab in enumerate(labs):
        for k in cols:
            cols[k][row, :lab.n] = torch.tensor(getattr(lab, k)).float()
    bank = LabelBank(P_max=P, beta=torch.full((n,), beta), n_true=torch.tensor([lab.n for lab in labs]),
                     resid=torch.tensor([lab.resid for lab in labs]).float(),
                     neg_weight=torch.tensor([lab.neg_weight for lab in labs]).float(),
                     family=[t.family for t in truths], params=[t.params for t in truths],
                     ladders=[lab.ladder for lab in labs], meta={}, **cols)
    path = str(tmp_path / "bank.pt")
    bank.save(path)
    torch.save({row: (torch.tensor(t.energies), torch.tensor(t.weights / t.weights.sum()))
                for row, t in enumerate(truths)}, path + ".lines.pt")
    with pytest.raises(ValueError):
        LabelBank.load(path).truth(0)                              # lines are not rebuilt from parameters

    g = PolePriorGenerator(residue_mode="dispersive", beta_choices=(beta,), smooth_bank=path, p_smooth=1.0,
                           dtau_choices=(0.125,), dtype=torch.float64)
    g.seed_scalars(0)
    b = g.batch(6, torch.Generator().manual_seed(0))
    assert all(f == "lines:holstein_atom" for f in b.family)
    x = b.x.double().numpy()
    for i in range(6):
        row = next(r for r in range(n) if int(b.poles_true["n_true"][i]) == labs[r].n
                   and torch.allclose(b.poles_true["eps"][i, :labs[r].n].float(), cols["eps"][r, :labs[r].n] * beta))
        t = truths[row]
        exact = t.greens(x) / t.weights.sum()
        assert np.abs(b.G_clean[i].numpy() - exact).max() < 1e-12      # closed form, not quadrature
        # the label reproduces those data (a dense ladder is representable) ...
        from spevae.physics import PoleGreens, PoleSet
        lab = PoleSet(*(b.poles_true[k][i:i + 1] for k in ("eps", "gamma", "a", "b")))
        assert float((PoleGreens(b.quad)(lab, b.x)[0] - b.G_clean[i]).abs().max()) < 1e-4
        # ... and the node spectrum carries the lines' weight, line by line
        assert abs(float((b.A_true[i] * b.quad.weights).sum()) - 1.0) < 1e-9
        nodes = b.quad.nodes / beta
        lo = float((b.A_true[i] * b.quad.weights * (nodes < 0)).sum())
        assert abs(lo - float((t.weights / t.weights.sum())[t.energies < 0].sum())) < 1e-6
