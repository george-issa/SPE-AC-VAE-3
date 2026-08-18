"""Structural-symmetry tests for the model.

Every property here must hold *identically*, for random weights, not merely
after training.  A failure means the guarantee has become a training objective,
which is the thing the design is meant to avoid.
"""

import math

import pytest
import torch

from spevae.models import (
    DEFAULT_GAMMA_MIN,
    PoleConstraints,
    SPEVAE,
    active_units,
    spectral_norm_from_greens,
)
from spevae.physics import MATSUBARA_NU0, SpectralQuadrature

DT = torch.float64
OMEGA = torch.linspace(-8, 8, 801, dtype=DT)


def make_model(residue_mode="real", latent_dim=6, num_poles=8, seed=0, **kw):
    torch.manual_seed(seed)
    m = SPEVAE(latent_dim=latent_dim, num_poles=num_poles, omega_c=20.0,
               residue_mode=residue_mode, d_model=32, enc_layers=2, dec_layers=2,
               n_heads=4, d_ff=64, n_matsubara=8, **kw)
    return m.to(DT)


def physical_greens(B=4, L=41, beta=10.0, seed=0):
    """A batch of plausible G(tau): positive, obeying G(0)+G(beta)=1."""
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(0, 1, L, dtype=DT)
    nu = 4.0 * torch.randn(B, 3, generator=g, dtype=DT)
    w = torch.softmax(torch.randn(B, 3, generator=g, dtype=DT), dim=-1)
    K = 1.0 / (torch.exp(x[None, None, :] * nu[..., None])
               + torch.exp((x - 1)[None, None, :] * nu[..., None]))
    return (w.unsqueeze(-1) * K).sum(1), x


# --------------------------------------------------------------------------- #
# The constraint layer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("residue_mode", ["real", "complex"])
def test_constraints_hold_for_arbitrary_raw_outputs(residue_mode):
    """Causality, positivity, sum rule and zero-sum tail, for wild activations."""
    con = PoleConstraints(residue_mode=residue_mode, nu_max=200.0)
    g = torch.Generator().manual_seed(0)
    B, P = 16, 10
    for scale in (1e-3, 1.0, 50.0):
        u = [scale * torch.randn(B, P, generator=g, dtype=DT) for _ in range(4)]
        poles = con(*u)
        assert (poles.gamma >= DEFAULT_GAMMA_MIN).all(), "width floor violated"
        assert torch.isfinite(poles.gamma).all()
        assert (poles.a > 0).all(), "negative spectral weight"
        assert torch.allclose(poles.a.sum(-1), torch.ones(B, dtype=DT), atol=1e-12)
        assert torch.allclose(poles.b.sum(-1), torch.zeros(B, dtype=DT), atol=1e-12)
        assert (poles.eps.abs() <= 200.0 * (1 + 1e-9)).all(), "position bound violated"
        assert (poles.b.abs() <= poles.a + 1e-12).all(), "|b| <= a violated"


def test_real_mode_gives_pointwise_positive_spectra():
    con = PoleConstraints(residue_mode="real")
    g = torch.Generator().manual_seed(1)
    u = [3.0 * torch.randn(8, 12, generator=g, dtype=DT) for _ in range(4)]
    A = con(*u).spectral(torch.linspace(-400, 400, 20001, dtype=DT))
    assert (A >= 0).all()


def test_complex_mode_can_go_negative():
    """Documents the trade-off rather than hiding it."""
    con = PoleConstraints(residue_mode="complex")
    g = torch.Generator().manual_seed(2)
    u = [3.0 * torch.randn(64, 12, generator=g, dtype=DT) for _ in range(4)]
    A = con(*u).spectral(torch.linspace(-400, 400, 20001, dtype=DT))
    assert (A < 0).any(), "complex residues should be able to produce A < 0"


def test_norm_scaling_is_exactly_linear():
    con = PoleConstraints(residue_mode="complex")
    g = torch.Generator().manual_seed(3)
    u = [torch.randn(5, 7, generator=g, dtype=DT) for _ in range(4)]
    base = con(*u, norm=torch.ones(5, dtype=DT))
    for c in (0.0, 2.5, -1.25):
        scaled = con(*u, norm=torch.full((5,), c, dtype=DT))
        assert torch.allclose(scaled.a, c * base.a, atol=1e-14)
        assert torch.allclose(scaled.b, c * base.b, atol=1e-14)
        assert torch.allclose(scaled.eps, base.eps, atol=1e-14)
        assert torch.allclose(scaled.gamma, base.gamma, atol=1e-14)


def test_width_floor_is_beta_independent_in_physical_units():
    """gamma_min is a constant in dimensionless units, i.e. ~ pi/beta in omega."""
    con = PoleConstraints()
    u = [torch.zeros(1, 4, dtype=DT) for _ in range(4)]
    poles = con(*u)
    for beta in (5.0, 10.0, 20.0):
        phys = poles.to_physical(beta)
        assert torch.allclose(phys.gamma * beta, poles.gamma, atol=1e-14)
    assert abs(DEFAULT_GAMMA_MIN - 0.1 * MATSUBARA_NU0) < 1e-12


def test_spectral_norm_read_off_the_endpoints():
    G, _ = physical_greens()
    assert torch.allclose(spectral_norm_from_greens(G),
                          torch.ones(G.shape[0], dtype=DT), atol=1e-12)


# --------------------------------------------------------------------------- #
# The full model: the symmetries the user asked for
# --------------------------------------------------------------------------- #

def test_zero_greens_function_maps_to_zero_spectral_function():
    """The headline structural symmetry: G == 0  =>  A == 0, exactly."""
    model = make_model()
    L, beta = 41, 10.0
    G = torch.zeros(3, L, dtype=DT)
    x = torch.linspace(0, 1, L, dtype=DT)
    out = model(G, x, beta, deterministic=True)
    assert torch.allclose(out.norm, torch.zeros(3, dtype=DT), atol=0)
    assert torch.allclose(out.poles.a, torch.zeros_like(out.poles.a), atol=0)
    assert torch.allclose(out.poles.b, torch.zeros_like(out.poles.b), atol=0)
    assert torch.allclose(out.spectral(OMEGA), torch.zeros(3, OMEGA.numel(), dtype=DT))
    assert torch.allclose(out.G_recon, torch.zeros_like(G), atol=1e-14)


@pytest.mark.parametrize("c", [0.5, 3.0, -2.0])
def test_estimator_is_exactly_one_homogeneous(c):
    """A[cG] = c A[G]: the scale lives entirely in the residues."""
    model = make_model(seed=1)
    G, x = physical_greens(B=3, L=41)
    beta = 10.0
    base = model(G, x, beta, deterministic=True)
    scaled = model(c * G, x, beta, deterministic=True)
    assert torch.allclose(scaled.spectral(OMEGA), c * base.spectral(OMEGA), atol=1e-11)
    assert torch.allclose(scaled.G_recon, c * base.G_recon, atol=1e-11)
    # Poles themselves are scale-invariant.
    assert torch.allclose(scaled.poles.eps, base.poles.eps, atol=1e-12)
    assert torch.allclose(scaled.poles.gamma, base.poles.gamma, atol=1e-12)


def test_model_output_satisfies_every_constraint():
    model = make_model(residue_mode="complex", seed=2)
    G, x = physical_greens(B=4, L=61)
    out = model(G, x, beta=10.0)
    p = out.poles
    assert (p.gamma > 0).all(), "causality: all poles in the lower half plane"
    assert (p.a > 0).all(), "positive spectral weight per pole"
    assert torch.allclose(p.a.sum(-1), out.norm, atol=1e-10), "sum rule"
    assert torch.allclose(p.b.sum(-1), torch.zeros(4, dtype=DT), atol=1e-10), "1/w^2 tail"
    assert (p.gamma >= DEFAULT_GAMMA_MIN).all(), "width floor"


def test_reconstruction_obeys_the_sum_rule():
    """G_recon(0) + G_recon(beta) reproduces the data's own norm."""
    model = make_model(seed=3)
    G, x = physical_greens(B=4, L=81)
    out = model(G, x, beta=10.0, deterministic=True)
    assert torch.allclose(out.G_recon[:, 0] + out.G_recon[:, -1], out.norm, atol=1e-7)


def test_reconstruction_is_positive_in_real_mode():
    model = make_model(residue_mode="real", seed=4)
    G, x = physical_greens(B=4, L=81)
    out = model(G, x, beta=10.0, deterministic=True)
    assert (out.G_recon > 0).all()


# --------------------------------------------------------------------------- #
# Grid / beta independence
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("L", [21, 51, 101, 201])
def test_same_weights_accept_any_number_of_time_slices(L):
    model = make_model(seed=5)
    G, x = physical_greens(B=2, L=L)
    out = model(G, x, beta=10.0, deterministic=True)
    assert out.G_recon.shape == (2, L)
    assert torch.isfinite(out.mu).all()


@pytest.mark.parametrize("beta", [2.0, 5.0, 10.0, 20.0])
def test_same_weights_accept_any_beta(beta):
    model = make_model(seed=6)
    G, x = physical_greens(B=2, L=int(10 * beta) + 1)
    out = model(G, x, beta=beta, deterministic=True)
    assert torch.isfinite(out.G_recon).all()
    # The pole-position bound tracks beta so that the physical cutoff is fixed.
    assert out.poles.eps.abs().max() <= 0.8 * beta * model.omega_c + 1e-9
    assert (out.poles_physical().gamma >= DEFAULT_GAMMA_MIN / beta - 1e-12).all()


def test_beta_universality_of_the_underlying_map():
    """Same G(x) and same z at two betas give the same *dimensionless* poles.

    This is the property that lets one trained network serve every simulation
    cell: beta enters the decoder only through the position bound and through
    the conditioning input, not through the shape of the inverse problem.
    """
    model = make_model(seed=7)
    z = torch.randn(3, model.latent_dim, dtype=DT)
    norm = torch.ones(3, dtype=DT)
    p5 = model.decode(z, norm, beta=5.0)
    p20 = model.decode(z, norm, beta=20.0)

    # Widths are unconditionally beta-independent.
    assert torch.allclose(p5.gamma, p20.gamma, atol=1e-9)
    # Positions agree wherever the smaller beta's bound is not active.  The
    # bound itself *is* beta-dependent, and correctly so: it encodes a fixed
    # *physical* cutoff |omega| <= 0.8 * omega_c, which is beta * that in
    # dimensionless units.
    inside = p5.eps.abs() < 0.99 * model.nu_max(5.0)
    assert inside.any()
    assert torch.allclose(p5.eps[inside], p20.eps[inside], atol=1e-9)
    # In physical units the same code therefore means a four-times-narrower
    # feature at four times the beta -- the correct temperature scaling.
    assert torch.allclose(p5.to_physical(5.0).gamma * 5.0,
                          p20.to_physical(20.0).gamma * 20.0, atol=1e-9)


def test_padding_mask_isolates_samples():
    model = make_model(seed=8)
    L = 61
    G, x = physical_greens(B=3, L=L)
    mask = torch.ones(3, L, dtype=torch.bool)
    mask[1, 40:] = False
    G_masked = G * mask
    out = model(G_masked, x, beta=10.0, mask=mask, deterministic=True)
    assert (out.G_recon[1, 40:] == 0).all()


# --------------------------------------------------------------------------- #
# Inference API and diagnostics
# --------------------------------------------------------------------------- #

def test_continue_returns_a_grid_free_pole_set():
    model = make_model(seed=9)
    G, _ = physical_greens(B=2, L=41)
    beta = 10.0
    out = model.continue_(G, beta=beta, omega=OMEGA)
    assert out["A"].shape == (2, OMEGA.numel())
    # Re-evaluating at 10x resolution needs no rerun of the network.
    fine = torch.linspace(-8, 8, 8001, dtype=DT)
    assert out["poles"].spectral(fine).shape == (2, 8001)


def test_continue_ensemble_reports_a_spread():
    model = make_model(seed=10)
    G, _ = physical_greens(B=2, L=41)
    out = model.continue_(G, beta=10.0, omega=OMEGA, n_samples=8)
    assert out["A_std"].shape == out["A"].shape
    assert (out["A_std"] >= 0).all()


def test_active_units_detects_a_dead_encoder():
    assert int(active_units(torch.zeros(64, 5, dtype=DT))) == 0
    assert int(active_units(torch.randn(4096, 5, dtype=DT))) == 5


def test_gradients_flow_to_every_parameter():
    model = make_model(seed=11)
    G, x = physical_greens(B=3, L=41)
    out = model(G, x, beta=10.0)
    ((out.G_recon - G) ** 2).mean().backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, f"no finite gradient for: {missing[:8]}"


def test_quadrature_guard_rejects_an_unresolvable_width_floor():
    """Too coarse a rule for the declared width floor is refused, not silently wrong."""
    model = SPEVAE(latent_dim=4, num_poles=4, quad_nodes_per_width=1.0,
                   d_model=16, enc_layers=1, dec_layers=1, n_heads=2, d_ff=32)
    with pytest.raises(ValueError, match="resolution limit"):
        model.forward_map(10.0, torch.device("cpu"), torch.float32)


def test_quadrature_guard_rejects_an_unaffordable_node_count():
    """Lowering the width floor refines the rule; the cost of that is made explicit."""
    model = SPEVAE(latent_dim=4, num_poles=4, gamma_min=1e-3,
                   d_model=16, enc_layers=1, dec_layers=1, n_heads=2, d_ff=32)
    with pytest.raises(ValueError, match="max_quad_nodes"):
        model.forward_map(10.0, torch.device("cpu"), torch.float32)


def test_caches_stay_bounded_over_many_betas():
    """Regression: an unbounded (beta -> kernel matrix) cache is a memory leak.

    beta is drawn continuously during pretraining, so every step is a cache miss
    *and* a new entry.  Each kernel matrix is megabytes; an unbounded dict
    reached tens of gigabytes within a few thousand steps and got the first long
    training run OOM-killed.
    """
    model = make_model(seed=12)
    G, _ = physical_greens(B=2, L=41)
    for i in range(40):
        beta = 4.0 + 0.137 * i                     # a different cell every time
        x = torch.linspace(0, 1, 41, dtype=DT)
        model(G, x, beta, deterministic=True)
    assert len(model._kernel_cache) <= 4
    assert len(model._quad_cache) <= 4
    assert model.cache_bytes() < 50 * 1024 * 1024

    model.clear_caches()
    assert model.cache_bytes() == 0


def test_cache_is_still_used_when_beta_repeats():
    """Bounding the cache must not disable it for the single-cell case."""
    model = make_model(seed=13)
    G, x = physical_greens(B=2, L=41)
    model(G, x, beta=10.0, deterministic=True)
    K1 = model.kernel_matrix(10.0, x)
    model(G, x, beta=10.0, deterministic=True)
    assert model.kernel_matrix(10.0, x) is K1
