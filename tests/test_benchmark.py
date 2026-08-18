"""Tests for the three-way benchmark harness.

The solver wrappers are skipped when their backends are absent, but the parts
that would silently corrupt a comparison table -- convention agreement between
the three kernels, the shared scoring path, and the "no method is ground truth"
labelling -- always run.
"""

import math

import numpy as np
import pytest
import torch

from spevae.benchmark import (
    ContinuationResult,
    agreement_matrix,
    backtransform,
    compare,
    deac_available,
    format_table,
    maxent_available,
    reconstruction_chi2,
    run_deac,
    run_maxent,
)
from spevae.data import from_bins
from spevae.physics import PoleGreens, PoleSet, SpectralQuadrature, fermi_kernel

DT = torch.float64
BETA = 10.0


def known_case(n_tau=81, n_bins=800, noise=1e-3, seed=0):
    """A two-peak spectrum with correlated noise, and its exact answer."""
    quad = SpectralQuadrature(nu_c=BETA * 20, h=0.09, dtype=DT)
    poles = PoleSet(
        eps=torch.tensor([[-1.5 * BETA, 1.8 * BETA]], dtype=DT),
        gamma=torch.tensor([[0.35 * BETA, 0.45 * BETA]], dtype=DT),
        a=torch.tensor([[0.45, 0.55]], dtype=DT),
        b=torch.zeros(1, 2, dtype=DT),
    )
    tau = torch.linspace(0, BETA, n_tau, dtype=DT)
    clean = PoleGreens(quad)(poles, tau / BETA)[0]
    g = torch.Generator().manual_seed(seed)
    d = (tau.reshape(-1, 1) - tau.reshape(1, -1)).abs() / BETA
    R = 0.8 * torch.exp(-d / 0.1) + 0.2 * torch.eye(n_tau, dtype=DT)
    eta = torch.randn(n_bins, n_tau, generator=g, dtype=DT) @ torch.linalg.cholesky(R).T
    mg = from_bins(clean.unsqueeze(0) + eta * noise, BETA, tau=tau, dtype=DT)
    return mg, poles.to_physical(BETA)


# --------------------------------------------------------------------------- #
# The thing that makes a three-way comparison meaningful at all
# --------------------------------------------------------------------------- #

def test_all_three_kernels_agree():
    """spevae, ana_cont and SmoQyDEAC must define the same K(tau, omega).

    A sign or normalisation mismatch between solvers produces plausible-looking
    but incomparable spectra, and nothing downstream would catch it.  The
    conventions are written out here explicitly and checked against this
    package's own kernel:

      spevae   : 1 / (exp(x nu) + exp((x-1) nu)),  x = tau/beta, nu = beta omega
      ana_cont : exp(-tau omega) / (1 + exp(-omega)) on the same rescaled axes
      SmoQyDEAC: 1 / (exp(omega tau) + exp(-omega (beta - tau)))
    """
    tau = torch.linspace(0.0, BETA, 17, dtype=DT)
    omega = torch.linspace(-6.0, 6.0, 23, dtype=DT)
    x = (tau / BETA).reshape(-1, 1)
    nu = (BETA * omega).reshape(1, -1)

    ours = fermi_kernel(x, nu)
    ana_cont = torch.exp(-x * nu) / (1.0 + torch.exp(-nu))
    deac = 1.0 / (torch.exp(omega.reshape(1, -1) * tau.reshape(-1, 1))
                  + torch.exp(-omega.reshape(1, -1) * (BETA - tau.reshape(-1, 1))))

    assert torch.allclose(ours, ana_cont, rtol=1e-12, atol=1e-14)
    assert torch.allclose(ours, deac, rtol=1e-10, atol=1e-14)


def test_backtransform_converges_to_the_forward_map_as_the_grid_widens():
    """The shared scoring path is exact only up to the grid it is given.

    Missing spectral weight beyond the window lands almost entirely at
    ``tau = 0`` and ``tau = beta``, where the kernel is 1, and for a pole
    expansion it decays only as ``1/omega_max``.  This is why an analytic method
    supplies its own reconstruction (see ``attach_reconstruction``).
    """
    mg, poles = known_case(n_bins=50)
    quad = SpectralQuadrature(nu_c=BETA * 200, h=0.09, dtype=DT)
    G_exact = PoleGreens(quad)(poles.to_dimensionless(BETA), mg.tau / BETA)[0].numpy()

    errs = []
    for w_max in (20.0, 200.0, 2000.0):
        omega = np.linspace(-w_max, w_max, 40001)
        A = poles.spectral(torch.as_tensor(omega, dtype=DT))[0].numpy()
        errs.append(float(np.abs(backtransform(A, omega, mg.tau.numpy(), BETA)
                                 - G_exact).max()))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-4
    # The error sits at the endpoints, not spread through the interior.
    omega = np.linspace(-20, 20, 40001)
    A = poles.spectral(torch.as_tensor(omega, dtype=DT))[0].numpy()
    d = np.abs(backtransform(A, omega, mg.tau.numpy(), BETA) - G_exact)
    assert d[0] > 50 * d[len(d) // 2]


def test_reconstruction_chi2_is_calibrated():
    """A perfect reconstruction scores at the calibrated target, not at 1."""
    mg, poles = known_case(n_bins=2000, n_tau=41)
    quad = SpectralQuadrature(nu_c=BETA * 200, h=0.09, dtype=DT)
    G = PoleGreens(quad)(poles.to_dimensionless(BETA), mg.tau / BETA)[0].numpy()
    val = reconstruction_chi2(G, mg, against="mean")
    target = mg.covariance()["chi2_target"]
    assert 0.2 < val / target < 5.0, f"chi2={val}, target={target}"


def test_exact_reconstruction_can_be_supplied_instead_of_the_grid_one():
    mg, poles = known_case(n_bins=400, n_tau=41)
    omega = np.linspace(-10, 10, 801)
    A = poles.spectral(torch.as_tensor(omega, dtype=DT))[0].numpy()
    quad = SpectralQuadrature(nu_c=BETA * 200, h=0.09, dtype=DT)
    G_exact = PoleGreens(quad)(poles.to_dimensionless(BETA), mg.tau / BETA)[0].numpy()

    truncated = ContinuationResult("grid", omega, A).attach_reconstruction(mg)
    exact = ContinuationResult("exact", omega, A).attach_reconstruction(
        mg, G_recon=G_exact)
    assert exact.chi2 < truncated.chi2, (exact.chi2, truncated.chi2)
    assert exact.meta["reconstruction"] == "exact forward map"
    assert truncated.meta["reconstruction"] == "grid back-transform"


def test_mean_and_bin_chi2_differ_by_the_bin_count():
    mg, poles = known_case(n_bins=100, n_tau=41)
    omega = np.linspace(-20, 20, 2001)
    A = poles.spectral(torch.as_tensor(omega, dtype=DT))[0].numpy()
    G = backtransform(A, omega, mg.tau.numpy(), BETA)
    ratio = (reconstruction_chi2(G, mg, against="mean")
             / reconstruction_chi2(G, mg, against="bin"))
    assert abs(ratio - mg.n_bins) < 0.01 * mg.n_bins


# --------------------------------------------------------------------------- #
# Comparator bookkeeping
# --------------------------------------------------------------------------- #

def _fake(method, centre, omega):
    A = np.exp(-0.5 * ((omega - centre) / 0.6) ** 2)
    A /= np.trapezoid(A, omega)
    return ContinuationResult(method=method, omega=omega, A=A)


def test_agreement_matrix_is_symmetric_with_zero_diagonal():
    omega = np.linspace(-8, 8, 801)
    rs = [_fake("a", 0.0, omega), _fake("b", 0.5, omega), _fake("c", 2.0, omega)]
    labels, M = agreement_matrix(rs)
    assert labels == ["a", "b", "c"]
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)
    assert M[0, 2] > M[0, 1], "further apart should score further apart"


def test_compare_against_a_reference_is_labelled_as_agreement_not_accuracy():
    omega = np.linspace(-8, 8, 801)
    rs = [_fake("a", 0.0, omega), _fake("b", 0.4, omega)]
    cmp = compare(rs, reference="a", beta=BETA)
    assert "vs_truth" not in cmp
    assert cmp["vs_reference"]["reference"] == "a"
    assert "not accuracy" in cmp["vs_reference"]["caveat"]
    assert "b" in cmp["vs_reference"]["scores"]


def test_compare_against_truth_scores_every_method():
    omega = np.linspace(-8, 8, 801)
    truth = (omega, _fake("t", 0.0, omega).A)
    rs = [_fake("a", 0.0, omega), _fake("b", 1.0, omega)]
    cmp = compare(rs, truth=truth, beta=BETA)
    assert set(cmp["vs_truth"]) == {"a", "b"}
    assert cmp["vs_truth"]["a"]["w1"] < cmp["vs_truth"]["b"]["w1"]
    assert cmp["resolution"] == pytest.approx(math.pi / BETA)
    assert "pairwise L2" in format_table(cmp)


def test_compare_rejects_a_bad_reference_name():
    omega = np.linspace(-8, 8, 401)
    with pytest.raises(ValueError, match="not among the results"):
        compare([_fake("a", 0.0, omega)], reference="nope")


def test_non_overlapping_grids_are_rejected():
    a = _fake("a", 0.0, np.linspace(-8, -4, 201))
    b = _fake("b", 0.0, np.linspace(4, 8, 201))
    with pytest.raises(ValueError, match="do not overlap"):
        agreement_matrix([a, b])


def test_resolution_smoothing_suppresses_stochastic_roughness():
    """Peak counts are only comparable once both sides are at one resolution."""
    from spevae.metrics import find_peaks

    omega = np.linspace(-8, 8, 1601)
    smooth = _fake("s", 0.0, omega).A
    rng = np.random.default_rng(0)
    rough = smooth * (1.0 + 0.25 * rng.standard_normal(omega.size))
    assert len(find_peaks(rough, omega)) > 10
    assert len(find_peaks(rough, omega, resolution=math.pi / BETA)) <= 3
    assert len(find_peaks(smooth, omega, resolution=math.pi / BETA)) == 1


# --------------------------------------------------------------------------- #
# The solvers themselves
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not maxent_available(), reason="ana_cont did not import")
def test_maxent_recovers_a_known_two_peak_spectrum():
    mg, poles = known_case()
    omega = np.linspace(-10, 10, 801)
    res = run_maxent(mg, omega=omega)
    A_true = poles.spectral(torch.as_tensor(omega, dtype=DT))[0].numpy()

    assert abs(res.norm - np.trapezoid(A_true, omega)) < 0.05
    assert res.min_A > -1e-3, "MaxEnt should stay non-negative here"
    assert np.isfinite(res.chi2) and np.isfinite(res.chi2_target)
    assert res.G_recon is not None and res.G_recon.shape == mg.tau.shape

    from spevae.metrics import matched_peak_error
    m = matched_peak_error(res.A, A_true, omega, resolution=math.pi / BETA)
    assert m["n_matched"] == 2, m
    assert m["d_omega"] < 0.3, m


@pytest.mark.skipif(not maxent_available(), reason="ana_cont did not import")
def test_maxent_diagonal_and_full_covariance_both_run():
    mg, _ = known_case(n_tau=41, n_bins=200)
    omega = np.linspace(-10, 10, 401)
    a = run_maxent(mg, omega=omega, use_full_covariance=False)
    b = run_maxent(mg, omega=omega, use_full_covariance=True)
    assert a.meta["full_covariance"] is False
    assert b.meta["full_covariance"] is True
    assert np.isfinite(a.A).all() and np.isfinite(b.A).all()


@pytest.mark.skipif(not deac_available(), reason="julia + SmoQyDEAC unavailable")
def test_deac_recovers_a_known_two_peak_spectrum():
    mg, poles = known_case()
    omega = np.linspace(-10, 10, 201)
    res = run_deac(mg, omega=omega, num_bins=4, runs_per_bin=4, generations=4000)
    A_true = poles.spectral(torch.as_tensor(omega, dtype=DT))[0].numpy()

    assert abs(res.norm - np.trapezoid(A_true, omega)) < 0.05
    assert res.A_std is not None and (res.A_std >= 0).all()
    from spevae.metrics import matched_peak_error
    m = matched_peak_error(res.A, A_true, omega, resolution=math.pi / BETA)
    assert m["n_matched"] == 2, m
    assert m["d_omega"] < 0.3, m


@pytest.mark.skipif(not deac_available(), reason="julia + SmoQyDEAC unavailable")
def test_deac_rejects_an_uneven_grid():
    mg, _ = known_case(n_tau=41, n_bins=100)
    with pytest.raises(ValueError, match="evenly spaced"):
        run_deac(mg, omega=np.geomspace(1.0, 10.0, 51))


def test_deac_reports_a_missing_backend_loudly():
    """A missing comparator must raise, never quietly vanish from a table."""
    import shutil

    mg, _ = known_case(n_tau=21, n_bins=50)
    if shutil.which("julia") is not None:
        pytest.skip("julia present; the no-julia path cannot be exercised here")
    with pytest.raises(RuntimeError, match="julia not found"):
        run_deac(mg)
