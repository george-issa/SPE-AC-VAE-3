"""Tests for the synthetic data generator, the noise model, and the objectives.

Two of these check load-bearing quantitative claims rather than mere plumbing:

* ``test_chi2_is_calibrated_to_one`` -- the reconstruction loss really does
  average 1 at the noise floor, so "chi2 approached 1" means something.
* ``test_w1_is_linear_in_peak_displacement`` -- the Wasserstein term really is
  linear in how far a peak is misplaced, which is the entire reason it is used
  instead of L2.  The companion test shows L2 saturating on the same data.
"""

import math

import pytest
import torch

from spevae.data import (
    FAMILIES,
    NoiseModel,
    SyntheticGenerator,
    correlation_matrix,
    expected_chi2,
    ledoit_wolf_correlation,
    sample_correlated_noise,
    sample_spectra,
    whitener,
)
from spevae.losses import (
    chi2,
    kl_divergence,
    noise_consistency,
    spectral_l2,
    spectral_multiscale_l2,
    spectral_w1,
)
from spevae.physics import PoleGreens, SpectralQuadrature

DT = torch.float64


# --------------------------------------------------------------------------- #
# Spectral families
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_family_is_positive_and_normalised(family):
    quad = SpectralQuadrature(nu_c=200.0, h=0.09, dtype=DT)
    rng = torch.Generator().manual_seed(0)
    spec = sample_spectra(quad, beta=10.0, batch_size=16, rng=rng,
                          families=[family], dtype=DT)
    assert (spec.A >= 0).all(), f"{family}: negative spectral weight"
    assert torch.isfinite(spec.A).all()
    assert torch.allclose(quad.integrate(spec.A), torch.ones(16, dtype=DT), atol=1e-8)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_family_produces_a_valid_greens_function(family):
    """The generated (A, G) pair must satisfy the sum rule G(0)+G(beta)=1."""
    quad = SpectralQuadrature(nu_c=200.0, h=0.09, dtype=DT)
    rng = torch.Generator().manual_seed(1)
    spec = sample_spectra(quad, beta=10.0, batch_size=8, rng=rng,
                          families=[family], dtype=DT)
    x = torch.linspace(0, 1, 101, dtype=DT)
    K = PoleGreens(quad).kernel_matrix(x)
    G = (spec.A * quad.weights) @ K
    assert (G > 0).all(), f"{family}: non-positive G(tau)"
    assert torch.allclose(G[:, 0] + G[:, -1], torch.ones(8, dtype=DT), atol=1e-7)


def test_sharp_families_are_broadened_to_the_resolution_floor():
    """Unbroadened band edges would make the (A, G) targets mutually inconsistent."""
    quad = SpectralQuadrature(nu_c=200.0, h=0.09, dtype=DT)
    rng = torch.Generator().manual_seed(2)
    spec = sample_spectra(quad, beta=10.0, batch_size=4, rng=rng,
                          families=["box"], dtype=DT)
    core = spec.A[:, : quad.n_core]
    # A hard box would jump by its full height in one step; after broadening to
    # gamma_min ~ 3.5 h the largest single-step change is a small fraction of it.
    jump = (core[:, 1:] - core[:, :-1]).abs().amax(-1)
    assert (jump < 0.25 * core.amax(-1)).all()


def test_lorentzian_family_exposes_exact_pole_labels():
    quad = SpectralQuadrature(nu_c=200.0, h=0.09, dtype=DT)
    rng = torch.Generator().manual_seed(3)
    spec = sample_spectra(quad, beta=10.0, batch_size=6, rng=rng,
                          families=["lorentz_mixture"], dtype=DT)
    assert spec.poles is not None
    eps, gam, w = spec.poles
    assert torch.allclose(w.sum(-1), torch.ones(6, dtype=DT), atol=1e-6)
    assert (gam > 0).all()


def test_other_families_expose_no_pole_labels():
    """Parameter-space supervision is only meaningful when the target *is* poles."""
    quad = SpectralQuadrature(nu_c=200.0, h=0.09, dtype=DT)
    rng = torch.Generator().manual_seed(4)
    spec = sample_spectra(quad, beta=10.0, batch_size=4, rng=rng,
                          families=["gauss_mixture"], dtype=DT)
    assert spec.poles is None


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #

def test_correlated_noise_has_the_requested_covariance():
    x = torch.linspace(0, 1, 40, dtype=DT)
    G = torch.ones(20_000, 40, dtype=DT)
    model = NoiseModel(scale=1.0, xi=0.15, proportional=0.0, nugget=0.2)
    rng = torch.Generator().manual_seed(0)
    noise, sigma, R = sample_correlated_noise(G, x, model, rng)
    emp = (noise.T @ noise) / noise.shape[0] / (sigma[0].reshape(-1, 1) * sigma[0].reshape(1, -1))
    assert (emp - R).abs().max() < 0.05


def test_nugget_keeps_the_whitener_conditioned():
    """Without an independent component the correlation matrix is near-singular.

    Worst for the smooth (Gaussian) kernel, whose eigenvalues decay
    exponentially -- whitening with its inverse then demands an essentially exact
    fit along directions carrying no information.
    """
    x = torch.linspace(0, 1, 100, dtype=DT)
    for kind, min_gain in (("gauss", 1e4), ("exp", 5.0)):
        lam_no = torch.linalg.eigvalsh(
            correlation_matrix(x, xi=0.4, kind=kind, nugget=1e-12))
        lam_yes = torch.linalg.eigvalsh(
            correlation_matrix(x, xi=0.4, kind=kind, nugget=0.15))
        cond_no = float(lam_no.max() / lam_no.clamp_min(1e-30).min())
        cond_yes = float(lam_yes.max() / lam_yes.min())
        assert cond_no / cond_yes > min_gain, (
            f"{kind}: cond {cond_no:.3g} -> {cond_yes:.3g}")
        # The smallest eigenvalue is floored by the nugget itself, so the
        # whitener can never amplify by more than 1/sqrt(nugget).
        assert float(lam_yes.min()) >= 0.15 - 1e-9


def test_whitener_inverts_the_correlation():
    x = torch.linspace(0, 1, 60, dtype=DT)
    R = correlation_matrix(x, xi=0.1, nugget=0.1)
    W = whitener(R)
    assert torch.allclose(W @ R @ W, torch.eye(60, dtype=DT), atol=1e-8)


def test_ledoit_wolf_recovers_the_correlation_and_shrinks_sensibly():
    x = torch.linspace(0, 1, 50, dtype=DT)
    R = correlation_matrix(x, xi=0.12, nugget=0.15)
    Lch = torch.linalg.cholesky(R)
    rng = torch.Generator().manual_seed(0)
    bins = torch.randn(4000, 50, generator=rng, dtype=DT) @ Lch.T * 1e-3
    sigma, R_lw, rho = ledoit_wolf_correlation(bins)
    assert 0.0 <= rho <= 1.0
    assert (sigma - 1e-3).abs().max() < 1e-4
    assert (R_lw - R).abs().max() < 0.15
    # With many bins the estimate is good, so shrinkage should be light.
    assert rho < 0.3


def test_expected_chi2_is_at_most_one_and_falls_with_shrinkage():
    x = torch.linspace(0, 1, 60, dtype=DT)
    R = correlation_matrix(x, xi=0.2, nugget=0.1)
    vals = [expected_chi2(R, rho) for rho in (0.0, 0.1, 0.3, 0.6)]
    assert abs(vals[0] - 1.0) < 1e-9
    assert all(v <= 1.0 + 1e-12 for v in vals)
    assert vals[1] > vals[2] > vals[3], "more shrinkage should pull the target lower"


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #

def test_chi2_is_calibrated_to_one():
    """At the noise floor the whitened reconstruction loss averages 1."""
    x = torch.linspace(0, 1, 64, dtype=DT)
    R = correlation_matrix(x, xi=0.15, nugget=0.2)
    model = NoiseModel(scale=1e-3, xi=0.15, proportional=0.0, nugget=0.2)
    G_true = torch.full((4000, 64), 0.5, dtype=DT)
    rng = torch.Generator().manual_seed(0)
    noise, sigma, _ = sample_correlated_noise(G_true, x, model, rng)
    val = float(chi2(G_true, G_true + noise, sigma, inv_sqrt_R=whitener(R)))
    assert abs(val - 1.0) < 0.05, f"chi2 at the noise floor = {val}"


def test_whitening_reweights_smooth_versus_rough_errors():
    """What the correlation actually buys, stated precisely.

    Diagonal weighting judges an error only by its per-slice size.  The whitened
    loss knows *which directions the data constrains*: a rough error the same
    size as the noise is heavily penalised (the data pins rough directions far
    more tightly than the marginal variances suggest), while a smooth one is
    discounted (it lies along the dominant noise mode).  Ignoring the correlation
    therefore does not simply rescale chi^2 -- it optimises the wrong thing.
    """
    L = 64
    x = torch.linspace(0, 1, L, dtype=DT)
    R = correlation_matrix(x, xi=0.3, nugget=0.05)
    W = whitener(R)
    G = torch.full((1, L), 0.5, dtype=DT)
    sigma = torch.full((1, L), 1e-3, dtype=DT)

    rough = sigma * torch.tensor([(-1.0) ** i for i in range(L)], dtype=DT)
    smooth = sigma * torch.ones(1, L, dtype=DT)

    diag_rough = float(chi2(G + rough, G, sigma))
    diag_smooth = float(chi2(G + smooth, G, sigma))
    full_rough = float(chi2(G + rough, G, sigma, inv_sqrt_R=W))
    full_smooth = float(chi2(G + smooth, G, sigma, inv_sqrt_R=W))

    assert abs(diag_rough - diag_smooth) < 1e-9, "diagonal cannot tell them apart"
    assert full_rough > 10 * full_smooth, (
        f"whitened loss should separate them: rough={full_rough:.3g}, "
        f"smooth={full_smooth:.3g}")


def _delta_like(quad, centre, width=1.0):
    nu = quad.nodes.reshape(1, -1)
    A = torch.exp(-0.5 * ((nu - centre) / width) ** 2) / (width * math.sqrt(2 * math.pi))
    return A / quad.integrate(A).reshape(-1, 1)


def test_w1_is_linear_in_peak_displacement():
    """The property L2 lacks, and the reason W1 is the primary spectral term."""
    quad = SpectralQuadrature(nu_c=200.0, h=0.09, dtype=DT)
    ref = _delta_like(quad, 0.0)
    shifts = [2.0, 4.0, 8.0, 16.0]
    w1 = [float(spectral_w1(_delta_like(quad, s), ref, quad)) for s in shifts]
    for s, v in zip(shifts, w1):
        assert abs(v - s) < 0.05 * s, f"W1 at shift {s} is {v}, expected ~{s}"


def test_l2_saturates_once_peaks_stop_overlapping():
    """The failure mode W1 exists to avoid: no gradient towards the right position."""
    quad = SpectralQuadrature(nu_c=200.0, h=0.09, dtype=DT)
    ref = _delta_like(quad, 0.0)
    far = [float(spectral_l2(_delta_like(quad, s), ref, quad)) for s in (8.0, 16.0, 32.0)]
    assert max(far) - min(far) < 0.02 * max(far), "L2 should be flat once separated"
    # ... whereas W1 keeps growing over exactly the same range.
    w1 = [float(spectral_w1(_delta_like(quad, s), ref, quad)) for s in (8.0, 16.0, 32.0)]
    assert w1[2] > 3.5 * w1[0]


def test_spectral_distances_vanish_for_identical_spectra():
    quad = SpectralQuadrature(nu_c=100.0, h=0.09, dtype=DT)
    A = _delta_like(quad, 3.0, width=2.0)
    assert float(spectral_w1(A, A, quad)) < 1e-12
    assert float(spectral_l2(A, A, quad)) < 1e-20
    assert float(spectral_multiscale_l2(A, A, quad)) < 1e-18


def test_w1_reported_in_physical_frequency_is_beta_free():
    """The same physical displacement scores the same at any beta."""
    vals = []
    for beta in (5.0, 20.0):
        quad = SpectralQuadrature(nu_c=beta * 20.0, h=0.09, dtype=DT)
        ref = _delta_like(quad, 0.0, width=0.05 * beta)
        shifted = _delta_like(quad, 1.5 * beta, width=0.05 * beta)   # 1.5 in omega
        vals.append(float(spectral_w1(shifted, ref, quad, beta=beta)))
    assert abs(vals[0] - vals[1]) < 0.02 * vals[0]
    assert abs(vals[0] - 1.5) < 0.05


def test_free_bits_removes_the_gradient_below_the_floor():
    mu = torch.zeros(32, 4, dtype=DT, requires_grad=True)
    logvar = torch.zeros(32, 4, dtype=DT, requires_grad=True)
    # At the prior the per-dimension KL is 0, i.e. below any positive floor.
    kl = kl_divergence(mu, logvar, free_bits=0.1)
    assert abs(float(kl.detach()) - 0.4) < 1e-12
    kl.backward()
    assert float(mu.grad.abs().max()) == 0.0

    # Well above the floor the penalty is active again.
    mu2 = torch.full((32, 4), 3.0, dtype=DT, requires_grad=True)
    kl2 = kl_divergence(mu2, torch.zeros(32, 4, dtype=DT), free_bits=0.1)
    kl2.backward()
    assert float(mu2.grad.abs().max()) > 0.0


def test_noise_consistency_vanishes_for_identical_posteriors():
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(16, 5, generator=g, dtype=DT)
    lv = torch.randn(16, 5, generator=g, dtype=DT)
    assert float(noise_consistency(mu, lv, mu, lv)) < 1e-12
    assert float(noise_consistency(mu, lv, mu + 1.0, lv)) > 0.0


# --------------------------------------------------------------------------- #
# Batch generation
# --------------------------------------------------------------------------- #

def test_generator_produces_consistent_paired_realisations():
    gen = SyntheticGenerator(omega_c=20.0, beta_range=(5.0, 15.0), max_ntau=101,
                             dtype=DT)
    rng = torch.Generator().manual_seed(0)
    b = gen.batch(12, rng)
    assert b.G.shape == b.G_alt.shape == b.G_clean.shape
    assert not torch.allclose(b.G, b.G_alt), "the two draws must be independent"
    # Both are the same clean signal plus noise of the same declared size.
    for noisy in (b.G, b.G_alt):
        z = (noisy - b.G_clean) / b.sigma
        assert float(z.abs().mean()) < 3.0
    assert torch.allclose(b.G_clean[:, 0] + b.G_clean[:, -1],
                          torch.ones(12, dtype=DT), atol=1e-7)


def test_generator_varies_beta_and_grid_across_batches():
    gen = SyntheticGenerator(omega_c=20.0, beta_range=(2.0, 20.0), max_ntau=201,
                             dtype=DT)
    rng = torch.Generator().manual_seed(0)
    betas, lengths, fams = set(), set(), set()
    for _ in range(24):
        b = gen.batch(4, rng)
        betas.add(round(b.beta, 3))
        lengths.add(int(b.x.numel()))
        fams.add(b.family[0])
    assert len(betas) > 10, "beta should vary between batches"
    assert len(lengths) > 3, "the time grid should vary between batches"
    assert len(fams) > 3, "the spectral family should vary between batches"
