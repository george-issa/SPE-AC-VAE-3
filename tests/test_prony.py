"""MPM-style extraction: rank-determined poles, no descent, no texture.

The load-bearing claims: the spline-exact Matsubara transform matches the
closed-form pole Matsubara to ~1e-8; a clean pole-representable G is
recovered with the EXACT pole count and parameters (no wobble by
construction); the noise floor controls the extracted rank.
"""

import math

import numpy as np
import pytest
import torch

from spevae.physics import (
    PoleGreens,
    PoleSet,
    SpectralQuadrature,
    greens_matsubara,
    matsubara_from_bins,
    mpm_extract,
)

BETA = 10.0


def clean_G(poles: PoleSet, n_tau=201):
    quad = SpectralQuadrature(nu_c=BETA * 20.0, h=0.05, dtype=torch.float64)
    tau = torch.linspace(0, BETA, n_tau, dtype=torch.float64)
    G = PoleGreens(quad)(poles.to_dimensionless(BETA).to(torch.float64),
                         tau / BETA)[0]
    return G, tau


def test_matsubara_transform_matches_closed_form():
    """Two regimes, both measured: smooth-tailed G transforms at ~1e-11;
    Lorentzian-tailed G is limited to ~1e-5 by the endpoint log-singularity
    of G' (the documented reason mpm_extract is two-pass)."""
    poles = PoleSet(eps=torch.tensor([[-1.5, 1.8]]),
                    gamma=torch.tensor([[0.35, 0.45]]),
                    a=torch.tensor([[0.45, 0.55]]),
                    b=torch.tensor([[0.05, -0.05]]))
    G, tau = clean_G(poles)
    om, F, _ = matsubara_from_bins(G, tau, BETA, n_max=30)
    zp = torch.complex(poles.eps, -poles.gamma)[0].numpy()
    rp = torch.complex(poles.a, poles.b)[0].numpy()
    F_exact = (rp[None, :] / (1j * om[:, None] - zp[None, :])).sum(1)
    assert np.abs(F - F_exact).max() < 5e-5

    # smooth-tailed control: the transform itself is near-exact
    quad = SpectralQuadrature(nu_c=BETA * 20.0, h=0.05, dtype=torch.float64)
    nu = quad.nodes.double()
    A = torch.exp(-0.5 * (nu / BETA) ** 2)
    A = A / (A * quad.weights.double()).sum()
    K = PoleGreens(quad).kernel_matrix(tau / BETA).double()
    Gs = (A * quad.weights.double()) @ K
    om2, Fs, _ = matsubara_from_bins(Gs, tau, BETA, n_max=30)
    w_ = nu.numpy() / BETA
    Fs_exact = np.array([((A * quad.weights.double()).numpy()
                          / (1j * o - w_)).sum() for o in om2])
    assert np.abs(Fs - Fs_exact).max() < 1e-9


def test_exact_recovery_of_a_pole_representable_G():
    truth = PoleSet(eps=torch.tensor([[-2.0, 0.4, 2.4]]),
                    gamma=torch.tensor([[0.5, 0.3, 0.7]]),
                    a=torch.tensor([[0.3, 0.45, 0.25]]),
                    b=torch.tensor([[0.06, -0.02, -0.04]]))
    G, tau = clean_G(truth)
    ps, rep = mpm_extract(G, tau, BETA, n_max=30)
    assert rep.n_poles == 3, rep
    order = torch.argsort(ps.eps[0])
    for got, want, tol in ((ps.eps[0][order], truth.eps[0], 3e-3),
                           (ps.gamma[0][order], truth.gamma[0], 8e-3),
                           (ps.a[0][order], truth.a[0], 3e-3),
                           (ps.b[0][order], truth.b[0], 5e-3)):
        assert torch.allclose(got.double(), want.double(), atol=tol), (got, want)
    assert rep.matsubara_residual < 5e-5
    # sum rules exact
    assert float(ps.a.sum()) == pytest.approx(float(G[0] + G[-1]), rel=1e-6)
    assert float(ps.b.sum()) == pytest.approx(0.0, abs=1e-7)


def test_noise_floor_controls_the_rank():
    """With noise, the extraction must return FEW poles (rank-truncated),
    not a texture comb -- the anti-wobble property, stated as a test."""
    truth = PoleSet(eps=torch.tensor([[-2.0, 2.0]]),
                    gamma=torch.tensor([[0.6, 0.6]]),
                    a=torch.tensor([[0.5, 0.5]]), b=torch.zeros(1, 2))
    G, tau = clean_G(truth)
    g = torch.Generator().manual_seed(5)
    bins = G.unsqueeze(0) + torch.randn(400, G.numel(), generator=g,
                                        dtype=torch.float64) * 1e-4
    mean = bins.mean(0)
    cov = ((bins - mean).T @ (bins - mean)) / (bins.shape[0] - 1) / bins.shape[0]
    ps, rep = mpm_extract(mean, tau, BETA, cov_mean=cov, n_max=30)
    assert rep.n_poles <= 6, rep          # few, noise-limited
    assert rep.n_poles >= 2
    # and the recovered spectrum has no comb: compare to truth
    w = torch.linspace(-8, 8, 1601)
    A_t = truth.spectral(w)[0]
    A_p = ps.spectral(w)[0]
    l2 = float(((A_p - A_t)**2).sum().sqrt() / (A_t**2).sum().sqrt())
    assert l2 < 0.2, l2
