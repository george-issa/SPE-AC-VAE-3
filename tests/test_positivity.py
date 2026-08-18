r"""The exact positivity certificate.

The certificate's claim is sharp: ``is_positive`` iff :math:`A(\omega) \ge 0`
on the *whole* real line, decided from the pole parameters alone.  Every test
therefore checks it against a brute-force dense scan -- the scan can only ever
prove negativity, the certificate proves both directions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spevae.physics.poles import PoleSet
from spevae.physics.positivity import PositivityReport, certify, positivity_certificate


def scan_min(eps, gamma, a, b, lo=-60.0, hi=60.0, n=400_001):
    w = np.linspace(lo, hi, n)
    d = np.subtract.outer(w, np.asarray(eps, float))
    A = ((np.asarray(a, float) * np.asarray(gamma, float) - np.asarray(b, float) * d)
         / (d * d + np.asarray(gamma, float) ** 2)).sum(-1) / np.pi
    return float(A.min()), float(w[int(np.argmin(A))]), w, A


def random_set(rng, P, b_amp):
    eps = rng.uniform(-8, 8, P)
    gamma = rng.uniform(0.4, 2.0, P)
    a = rng.uniform(0.2, 1.0, P)
    a = a / a.sum()
    t = rng.uniform(-b_amp, b_amp, P)
    b = a * (t - (a * t).sum())          # the decoder's own construction
    return eps, gamma, a, b


def test_real_residues_certify_positive():
    rng = np.random.default_rng(0)
    for P in (1, 3, 8):
        eps, gamma, a, _ = random_set(rng, P, 0.0)
        rep = certify(eps, gamma, a, np.zeros(P))
        assert rep.is_positive
        assert rep.negative_weight == 0.0
        assert rep.min_A >= 0.0


def test_positive_despite_complex_residues():
    """The case the certificate exists for: b != 0 yet A >= 0 everywhere.

    Two broad overlapping poles with a small opposite-signed dispersive pair:
    the b-parts cancel pointwise well enough that the total stays positive.
    The dense scan confirms; the certificate must agree WITHOUT a grid.
    """
    eps = np.array([-0.5, 0.5])
    gamma = np.array([2.0, 2.0])
    a = np.array([0.5, 0.5])
    b = np.array([0.05, -0.05])
    m, _, _, _ = scan_min(eps, gamma, a, b, -500, 500, 2_000_001)
    assert m >= 0.0, "test construction is wrong: the scan already finds A < 0"
    rep = certify(eps, gamma, a, b)
    assert rep.is_positive
    assert float(np.abs(b).max()) > 0.0


def test_negative_excursion_is_found_and_quantified():
    """A sharp pole with a strong dispersive part goes negative; the report
    must locate the dip and integrate it, matching brute force."""
    eps = np.array([-3.0, 3.0])
    gamma = np.array([0.5, 0.5])
    a = np.array([0.5, 0.5])
    b = np.array([0.4, -0.4])
    rep = certify(eps, gamma, a, b)
    m_scan, w_scan, w, A = scan_min(eps, gamma, a, b, -200, 200, 4_000_001)
    assert m_scan < 0
    assert not rep.is_positive
    assert rep.min_A == pytest.approx(m_scan, rel=1e-4)
    # The construction is antisymmetric, so the minimum is doubly degenerate at
    # +-w*; accept either twin by checking the *depth at the reported location*.
    d = rep.argmin_omega - eps
    A_at = float(((a * gamma - b * d) / (d * d + gamma**2)).sum() / np.pi)
    assert A_at == pytest.approx(m_scan, rel=1e-4)
    assert abs(rep.argmin_omega) == pytest.approx(abs(w_scan), abs=1e-2)
    neg_scan = float(np.trapezoid(np.clip(-A, 0.0, None), w))
    assert rep.negative_weight == pytest.approx(neg_scan, rel=1e-3)
    assert rep.n_real_roots >= 2


def test_certificate_matches_scan_on_random_sets():
    """Fuzz both verdict directions against the scan."""
    rng = np.random.default_rng(7)
    n_pos = n_neg = 0
    for _ in range(40):
        eps, gamma, a, b = random_set(rng, int(rng.integers(2, 7)),
                                      float(rng.uniform(0.0, 1.0)))
        rep = certify(eps, gamma, a, b)
        m, _, _, _ = scan_min(eps, gamma, a, b, -300, 300, 600_001)
        if rep.is_positive:
            n_pos += 1
            assert m >= -1e-7, f"certified positive but scan finds {m}"
        else:
            n_neg += 1
            assert m < 0 or rep.min_A > -1e-8, \
                f"certified negative (min {rep.min_A}) but scan floor is {m}"
    assert n_pos > 0 and n_neg > 0, "fuzz did not exercise both verdicts"


def test_scale_covariance():
    """Physical vs dimensionless units: verdict and negative weight invariant,
    min_A scales by beta, argmin by 1/beta."""
    beta = 7.3
    eps = np.array([-14.0, 6.0, 11.0])
    gamma = np.array([1.5, 2.5, 1.0])
    a = np.array([0.3, 0.4, 0.3])
    b = np.array([0.25, -0.05, -0.2])
    dimless = certify(eps, gamma, a, b)
    physical = certify(eps / beta, gamma / beta, a, b)
    assert dimless.is_positive == physical.is_positive
    if not dimless.is_positive:
        assert physical.min_A == pytest.approx(beta * dimless.min_A, rel=1e-6)
        assert physical.argmin_omega == pytest.approx(dimless.argmin_omega / beta,
                                                      rel=1e-5)
        assert physical.negative_weight == pytest.approx(dimless.negative_weight,
                                                         rel=1e-6)


def test_batched_poleset_interface_and_pruning():
    ps = PoleSet(
        eps=torch.tensor([[-2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]]),
        gamma=torch.tensor([[1.0, 1.0, 1.0], [0.4, 0.4, 1.0]]),
        a=torch.tensor([[0.5, 0.5, 1e-14], [0.5, 0.5, 1e-14]]),
        b=torch.tensor([[0.0, 0.0, 0.0], [0.45, -0.45, 0.0]]),
    )
    reports = positivity_certificate(ps)
    assert len(reports) == 2
    assert reports[0].is_positive          # pure Lorentzians (third pole pruned)
    assert not reports[1].is_positive      # sharp poles + strong dispersion
    assert isinstance(str(reports[1]), str)


def test_all_zero_poles_are_positive():
    ps = PoleSet(eps=torch.zeros(1, 4), gamma=torch.ones(1, 4),
                 a=torch.zeros(1, 4), b=torch.zeros(1, 4))
    rep = positivity_certificate(ps)[0]
    assert rep.is_positive and rep.negative_weight == 0.0


def test_large_dimensionless_positions_are_conditioned():
    """beta*omega ~ 10^3 coefficients must not destroy the root finder."""
    eps = np.array([-3000.0, -1000.0, 1000.0, 3000.0])
    gamma = np.array([40.0, 60.0, 60.0, 40.0])
    a = np.array([0.25, 0.25, 0.25, 0.25])
    b = np.array([12e-3, -4e-3, 4e-3, -12e-3]) * 20
    rep = certify(eps, gamma, a, b)
    m, _, _, _ = scan_min(eps, gamma, a, b, -30_000, 30_000, 4_000_001)
    assert rep.is_positive == (m >= -1e-12), (rep, m)
