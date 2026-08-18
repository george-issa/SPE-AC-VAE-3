"""Tests for the real-DQMC reading path.

The file-format readers are skipped when the datasets are not present, but the
universal :func:`from_bins` entry point and the convention checks always run --
those are what a user hits when bringing their own data, and they are where the
silent mistakes live.
"""

import math
import os

import numpy as np
import pytest
import torch

from spevae.data import MeasuredGreens, from_bins, read_holstein_cube
from spevae.paths import data_path
from spevae.physics import PoleGreens, PoleSet, SpectralQuadrature

CUBE = str(data_path("datasets", "real", "george_325.jld2"))
have_cube = os.path.exists(CUBE)


def synthetic_bins(n_bins=400, n_tau=41, beta=10.0, seed=0, sign=+1.0):
    """Bins from a known two-pole spectrum, with correlated noise."""
    quad = SpectralQuadrature(nu_c=beta * 20.0, h=0.09, dtype=torch.float64)
    poles = PoleSet(
        eps=torch.tensor([[-1.5 * beta, 1.8 * beta]], dtype=torch.float64),
        gamma=torch.tensor([[0.4 * beta, 0.5 * beta]], dtype=torch.float64),
        a=torch.tensor([[0.45, 0.55]], dtype=torch.float64),
        b=torch.zeros(1, 2, dtype=torch.float64),
    )
    tau = torch.linspace(0, beta, n_tau, dtype=torch.float64)
    clean = PoleGreens(quad)(poles, tau / beta)[0]
    g = torch.Generator().manual_seed(seed)
    d = (tau.reshape(-1, 1) - tau.reshape(1, -1)).abs() / beta
    R = 0.85 * torch.exp(-d / 0.1) + 0.15 * torch.eye(n_tau, dtype=torch.float64)
    L = torch.linalg.cholesky(R)
    noise = (torch.randn(n_bins, n_tau, generator=g, dtype=torch.float64) @ L.T) * 1e-3
    return sign * (clean.unsqueeze(0) + noise), tau, beta


def test_from_bins_reads_the_sum_rule():
    bins, tau, beta = synthetic_bins()
    mg = from_bins(bins, beta, tau=tau, dtype=torch.float64)
    sr = mg.check_sum_rule()
    assert sr["ok"], sr
    assert abs(sr["norm"] - 1.0) < 5e-3


def test_sum_rule_check_catches_a_forced_wrong_sign():
    """With detection overridden, the sum rule still flags the mistake."""
    bins, tau, beta = synthetic_bins(sign=-1.0)
    wrong = from_bins(bins, beta, tau=tau, sign_flip=False, dtype=torch.float64)
    assert not wrong.check_sum_rule()["ok"]
    assert wrong.check_sum_rule()["deviation"] > 1.5
    fixed = from_bins(bins, beta, tau=tau, sign_flip=True, dtype=torch.float64)
    assert fixed.check_sum_rule()["ok"]


def test_sum_rule_check_still_fails_when_neither_sign_works():
    """Detection must not paper over a dataset that is not a fermionic G."""
    bins, tau, beta = synthetic_bins()
    mg = from_bins(3.7 * bins, beta, tau=tau, dtype=torch.float64)
    assert not mg.check_sum_rule()["ok"]


def test_endpoint_dependence_is_excluded_from_the_covariance():
    bins, tau, beta = synthetic_bins(n_tau=41)
    mg = from_bins(bins, beta, tau=tau, dtype=torch.float64)
    assert mg.n_independent == mg.n_tau - 1
    cov = mg.covariance()
    assert cov["inv_sqrt_R"].shape == (mg.n_tau - 1, mg.n_tau - 1)
    assert cov["sigma"].numel() == mg.n_tau - 1


def test_chi2_target_approaches_one_when_bins_outnumber_slices():
    """The calibration is only trustworthy when N_bins >> L_tau."""
    bins, tau, beta = synthetic_bins(n_bins=4000, n_tau=21)
    mg = from_bins(bins, beta, tau=tau, dtype=torch.float64)
    assert mg.covariance()["chi2_target"] > 0.9

    # Same data, far fewer bins: the target falls away from 1, and a run
    # "reaching chi2 = 1" there would be reaching nothing in particular.
    starved = from_bins(bins[:22], beta, tau=tau, dtype=torch.float64)
    assert starved.covariance()["chi2_target"] < 0.85
    assert starved.covariance()["chi2_target"] < mg.covariance()["chi2_target"] - 0.1


def test_eigenvalue_floor_bounds_the_whitener():
    """The floor caps the whitener's amplification at 1/sqrt(floor)."""
    bins, tau, beta = synthetic_bins(n_bins=45, n_tau=41)
    mg = from_bins(bins, beta, tau=tau, dtype=torch.float64)
    for f in (0.05, 0.25, 0.5):
        R_inv_sqrt = mg.covariance(f)["inv_sqrt_R"]
        lam = torch.linalg.eigvalsh(R_inv_sqrt)
        assert float(lam.max()) <= 1.0 / math.sqrt(f) + 1e-6
    unfloored = float(torch.linalg.eigvalsh(
        mg.covariance(0.0)["inv_sqrt_R"]).max())
    # Compare at a floor that actually binds for this dataset: the unfloored
    # amplification must exceed 1/sqrt(f), otherwise flooring is a no-op and
    # the strict inequality below would compare a number with itself.
    f_binding = 0.5
    assert unfloored > 1.0 / math.sqrt(f_binding), (
        "test data too well-conditioned for the floor to bind; raise f_binding")
    floored = float(torch.linalg.eigvalsh(
        mg.covariance(f_binding)["inv_sqrt_R"]).max())
    assert unfloored > floored


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        MeasuredGreens(bins=torch.zeros(5), tau=torch.zeros(5), beta=1.0)
    with pytest.raises(ValueError):
        MeasuredGreens(bins=torch.zeros(4, 5), tau=torch.zeros(6), beta=1.0)
    with pytest.raises(ValueError):
        MeasuredGreens(bins=torch.zeros(1, 5), tau=torch.zeros(5), beta=1.0)


@pytest.mark.skipif(not have_cube, reason="site-Holstein cube not present")
@pytest.mark.parametrize("n_idx,expected_n", [(19, 1.00), (3, 0.20), (9, 0.50)])
def test_holstein_cube_conventions(n_idx, expected_n):
    """G(0) = 1 - n/2 and the sum rule, read straight off the file."""
    mg = read_holstein_cube(CUBE, n_idx=n_idx, omega_idx=1, beta_idx=5)
    assert mg.check_sum_rule()["ok"], mg.summary()
    assert abs(float(mg.mean[0]) - (1.0 - expected_n / 2)) < 5e-3
    assert mg.includes_endpoints
    assert mg.n_tau == int(round(mg.beta / 0.1)) + 1


@pytest.mark.skipif(not have_cube, reason="site-Holstein cube not present")
def test_holstein_cube_attaches_a_reference_spectrum():
    mg = read_holstein_cube(CUBE, n_idx=19, omega_idx=1, beta_idx=5)
    assert mg.reference is not None
    ws, A = mg.reference
    assert ws.numel() == A.numel()
    assert float(A.min()) > -1e-6, "reference DOS should be non-negative"
    norm = float(torch.trapezoid(A.double(), ws.double()))
    assert abs(norm - 1.0) < 0.05, f"reference DOS norm = {norm}"


def test_sign_convention_is_detected_by_default():
    """Neither convention is 'the' convention; the sum rule settles it."""
    bins, tau, beta = synthetic_bins()
    pos = from_bins(bins, beta, tau=tau, dtype=torch.float64)
    neg = from_bins(-bins, beta, tau=tau, dtype=torch.float64)
    assert pos.check_sum_rule()["ok"] and neg.check_sum_rule()["ok"]
    assert pos.meta["sign_flip"] is False
    assert neg.meta["sign_flip"] is True
    assert torch.allclose(pos.bins, neg.bins)
    # An explicit choice still wins over detection.
    forced = from_bins(-bins, beta, tau=tau, sign_flip=False, dtype=torch.float64)
    assert not forced.check_sum_rule()["ok"]
