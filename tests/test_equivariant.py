"""Exact PH (tau-reflection) equivariance by group averaging (models/equivariant.py).

The claims are structural, so they are tested on randomly initialised models
(no training can be responsible): F[PG] = M F[G] for the amortised pass, the
same for z-descent run on the wrapper, every output constraint preserved, the
kernel identity the construction rests on, and refusal of grids the reflection
does not map onto themselves.
"""

import pytest
import torch

from spevae.models import SPEVAE
from spevae.models.equivariant import (MirrorSymmetrized, mirror_poles,
                                       reflect_tau)
from spevae.physics import SpectralQuadrature
from spevae.physics.forward import PoleGreens
from spevae.physics.poles import PoleSet
from spevae.refine import z_descend

BETA = 8.0


def base_model(seed=0, residue_mode="complex"):
    torch.manual_seed(seed)
    return SPEVAE(latent_dim=6, num_poles=6, omega_c=20.0, d_model=32,
                  enc_layers=2, dec_layers=2, n_heads=4, d_ff=64, n_matsubara=8,
                  residue_mode=residue_mode).eval()


def asym_greens(L=65, batch=3, seed=0):
    """Clearly PH-asymmetric data on a uniform endpoint-inclusive grid."""
    g = torch.Generator().manual_seed(seed)
    P = 3
    eps = (torch.rand(batch, P, generator=g) * 3 - 0.5) * BETA   # shifted up
    gam = (0.2 + 0.4 * torch.rand(batch, P, generator=g)) * BETA
    a = torch.rand(batch, P, generator=g) + 0.2
    a = a / a.sum(-1, keepdim=True)
    truth = PoleSet(eps, gam, a, torch.zeros_like(a))
    x = torch.linspace(0.0, 1.0, L)
    G = PoleGreens(SpectralQuadrature.for_beta(BETA))(truth, x)
    sigma = torch.full_like(G, 1e-3)
    return G, x, sigma


def test_mirror_poles_reflects_the_spectrum():
    p = PoleSet(torch.tensor([[0.7, -1.9]]), torch.tensor([[0.3, 0.8]]),
                torch.tensor([[0.6, 0.4]]), torch.tensor([[0.2, -0.2]]))
    w = torch.linspace(-6, 6, 1201)
    assert torch.allclose(mirror_poles(p).spectral(w), p.spectral(w).flip(-1),
                          atol=1e-6)


def test_kernel_identity_G_reflects_under_mirror():
    """PG of a spectrum equals G of its mirror: the symmetry the construction uses."""
    G, x, _ = asym_greens()
    p = PoleSet(torch.tensor([[2.0, -5.0]]), torch.tensor([[1.5, 3.0]]),
                torch.tensor([[0.7, 0.3]]), torch.tensor([[0.1, -0.1]]))
    fwd = PoleGreens(SpectralQuadrature.for_beta(BETA))
    assert torch.allclose(fwd(mirror_poles(p), x), reflect_tau(fwd(p, x)),
                          atol=1e-6)


@pytest.mark.parametrize("residue_mode", ["complex", "dispersive"])
def test_amortised_pass_is_exactly_equivariant(residue_mode):
    F = MirrorSymmetrized(base_model(seed=1, residue_mode=residue_mode))
    G, x, sigma = asym_greens()
    w = torch.linspace(-30.0, 30.0, 3001)
    with torch.no_grad():
        A = F(G, x, BETA, sigma=sigma, deterministic=True).poles.spectral(w)
        A_r = F(reflect_tau(G), x, BETA, sigma=reflect_tau(sigma),
                deterministic=True).poles.spectral(w)
    assert torch.allclose(A_r, A.flip(-1), atol=1e-6), float((A_r - A.flip(-1)).abs().max())
    # and the base model is NOT (otherwise the test proves nothing)
    with torch.no_grad():
        b = F.base(G, x, BETA, sigma=sigma, deterministic=True).poles.spectral(w)
        b_r = F.base(reflect_tau(G), x, BETA, sigma=reflect_tau(sigma),
                     deterministic=True).poles.spectral(w)
    assert (b_r - b.flip(-1)).abs().max() > 1e-3


def test_constraints_survive_the_average():
    F = MirrorSymmetrized(base_model(seed=2))
    z = torch.randn(8, F.latent_dim)
    norm = torch.rand(8) + 0.5
    p = F.decode(z, norm, BETA)
    assert p.shape[1] == F.num_poles == 2 * F.base.num_poles
    assert (p.gamma >= F.gamma_min - 1e-9).all()
    assert (p.a > 0).all()
    assert torch.allclose(p.a.sum(-1), norm, rtol=1e-5)
    assert torch.allclose(p.b.sum(-1), torch.zeros(8), atol=1e-6)
    assert (p.b.abs() <= p.a + 1e-8).all()


def test_swapped_latent_is_the_mirrored_spectrum():
    F = MirrorSymmetrized(base_model(seed=3))
    Lz = F.base.latent_dim
    z = torch.randn(4, F.latent_dim)
    zs = torch.cat([z[:, Lz:], z[:, :Lz]], -1)
    w = torch.linspace(-30.0, 30.0, 3001)
    A = F.decode(z, torch.ones(4), BETA).spectral(w)
    As = F.decode(zs, torch.ones(4), BETA).spectral(w)
    assert torch.allclose(As, A.flip(-1), atol=1e-6)


def test_z_descent_on_the_wrapper_stays_equivariant():
    F = MirrorSymmetrized(base_model(seed=4))
    G, x, sigma = asym_greens(batch=1)
    w = torch.linspace(-4.0, 4.0, 801)
    r = z_descend(F, G[0], x, BETA, sigma=sigma[0], steps=60, patience=1000)
    rr = z_descend(F, reflect_tau(G[0]), x, BETA, sigma=reflect_tau(sigma[0]),
                   steps=60, patience=1000)
    A, Ar = r.poles.spectral(w), rr.poles.spectral(w)
    assert r.chi2 < r.chi2_init                     # it did descend
    assert abs(r.chi2 - rr.chi2) <= 1e-4 * max(r.chi2, 1.0)
    assert torch.allclose(Ar, A.flip(-1), atol=1e-4), float((Ar - A.flip(-1)).abs().max())


def test_asymmetric_grid_is_refused():
    F = MirrorSymmetrized(base_model())
    G, x, sigma = asym_greens()
    x_bad = torch.cat([x[:-1], torch.tensor([0.999])])
    with pytest.raises(ValueError, match="symmetric"):
        F(G, x_bad, BETA, sigma=sigma)
