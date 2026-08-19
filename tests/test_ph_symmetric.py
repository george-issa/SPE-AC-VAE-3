"""The particle-hole-symmetric decoder mode.

The claim is structural: A(omega) = A(-omega) identically in the weights, for
every latent code, at initialisation and after any amount of training -- plus
every existing constraint surviving the mirroring.  Tests probe random codes,
the exactness of the generator's target symmetrisation, checkpoint round
trips, and the forward map's G(tau) = G(beta - tau).
"""

import numpy as np
import pytest
import torch

from spevae import load_model, save_model
from spevae.data.synthetic import SyntheticGenerator
from spevae.models import SPEVAE
from spevae.physics import SpectralQuadrature

BETA = 10.0


def sym_model(seed=0, num_poles=8, residue_mode="complex"):
    torch.manual_seed(seed)
    return SPEVAE(latent_dim=6, num_poles=num_poles, omega_c=20.0,
                  d_model=32, enc_layers=2, dec_layers=2, n_heads=4, d_ff=64,
                  n_matsubara=8, residue_mode=residue_mode,
                  ph_symmetric=True).eval()


def test_odd_pole_count_is_rejected():
    with pytest.raises(ValueError, match="even"):
        sym_model(num_poles=7)


@pytest.mark.parametrize("residue_mode", ["real", "complex"])
def test_spectrum_is_symmetric_for_random_codes(residue_mode):
    model = sym_model(seed=1, residue_mode=residue_mode)
    z = torch.randn(16, model.latent_dim)
    poles = model.decode(z, torch.ones(16), BETA)
    w = torch.linspace(-25.0, 25.0, 2001)
    A = poles.spectral(w)
    assert torch.allclose(A, A.flip(-1), atol=1e-6), \
        float((A - A.flip(-1)).abs().max())


def test_all_constraints_survive_the_mirroring():
    model = sym_model(seed=2)
    z = torch.randn(8, model.latent_dim)
    norm = torch.rand(8) + 0.5
    p = model.decode(z, norm, BETA)
    assert p.shape[1] == model.num_poles          # full count, not the half
    assert (p.gamma >= model.gamma_min - 1e-9).all(), "width floor"
    assert (p.a > 0).all(), "positive weights"
    assert torch.allclose(p.a.sum(-1), norm, rtol=1e-5), "sum rule x norm"
    assert torch.allclose(p.b.sum(-1), torch.zeros(8), atol=1e-7), "zero-sum b"
    assert (p.b.abs() <= p.a + 1e-8).all(), "|b| <= a"
    # mirror pairing is exact, half by half
    h = model.num_poles // 2
    assert torch.allclose(p.eps[:, :h], -p.eps[:, h:])
    assert torch.allclose(p.gamma[:, :h], p.gamma[:, h:])
    assert torch.allclose(p.a[:, :h], p.a[:, h:])
    assert torch.allclose(p.b[:, :h], -p.b[:, h:])


def test_greens_function_is_ph_symmetric():
    model = sym_model(seed=3)
    tau = torch.linspace(0.0, BETA, 81)
    x = tau / BETA
    z = torch.randn(4, model.latent_dim)
    poles = model.decode(z, torch.ones(4), BETA)
    G = model.forward_map(BETA, x.device, x.dtype)(
        poles, x, kernel_matrix=model.kernel_matrix(BETA, x))
    assert torch.allclose(G, G.flip(-1), atol=1e-6), \
        float((G - G.flip(-1)).abs().max())


def test_reflection_index_is_exact():
    quad = SpectralQuadrature(nu_c=120.0, h=0.09, dtype=torch.float64)
    pi = quad.reflection_index()
    assert torch.equal(quad.nodes[pi], -quad.nodes)
    assert torch.allclose(quad.weights[pi], quad.weights)
    assert torch.equal(pi[pi], torch.arange(quad.n_nodes))   # an involution


def test_generator_symmetrises_targets_and_greens():
    gen = SyntheticGenerator(omega_c=20.0, beta_range=(6.0, 10.0),
                             ph_symmetric=True)
    gen.seed_scalars(5)
    rng = torch.Generator().manual_seed(5)
    batch = gen.batch(16, rng)
    pi = batch.quad.reflection_index()
    assert torch.allclose(batch.A_true, batch.A_true[:, pi], atol=1e-6)
    assert torch.allclose(batch.G_clean, batch.G_clean.flip(-1), atol=1e-5)
    assert batch.poles_true is None
    assert batch.meta["ph_symmetric"] is True
    # normalisation preserved by the symmetrisation
    norms = batch.quad.integrate(batch.A_true)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_checkpoint_round_trip_preserves_the_mode(tmp_path):
    model = sym_model(seed=6)
    path = tmp_path / "sym.pt"
    save_model(model, str(path))
    loaded = load_model(str(path))
    assert loaded.ph_symmetric
    z = torch.randn(3, model.latent_dim)
    a = model.decode(z, torch.ones(3), BETA)
    b = loaded.decode(z, torch.ones(3), BETA)
    for u, v in ((a.eps, b.eps), (a.gamma, b.gamma), (a.a, b.a), (a.b, b.b)):
        assert torch.equal(u, v)


def test_full_forward_pass_stays_symmetric_on_asymmetric_input():
    """Even fed asymmetric data, the symmetric decoder cannot emit an
    asymmetric spectrum -- the structural claim, stated adversarially."""
    model = sym_model(seed=7)
    tau = torch.linspace(0.0, BETA, 61)
    g = torch.exp(-2.0 * tau) + 0.3 * torch.exp(-1.5 * (BETA - tau))  # asym
    g = (g / (g[0] + g[-1])).unsqueeze(0)
    out = model(g, tau / BETA, BETA, deterministic=True)
    w = torch.linspace(-8.0, 8.0, 1601)
    A = out.spectral(w)
    assert torch.allclose(A, A.flip(-1), atol=1e-6)
