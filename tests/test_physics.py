"""Physics-invariant tests for the kernel, pole algebra and forward map.

These are the load-bearing checks of the package: every downstream claim about
causality, normalisation, tails and positivity reduces to something asserted
here.  Everything runs in float64 so a failure means a real error rather than
accumulated float32 noise.
"""

import math

import numpy as np
import pytest
import torch
from scipy import integrate

from spevae.physics import (
    MATSUBARA_NU0,
    PoleGreens,
    PoleSet,
    SpectralQuadrature,
    effective_pole_count,
    fermi_kernel,
    gaussian_mixture_spectrum,
    greens_matsubara,
    matsubara_from_tau,
    per_pole_extrema,
    spectral_from_poles,
    tail_coefficients,
)

DT = torch.float64
GAMMA_FLOOR = 0.1 * MATSUBARA_NU0            # the package's width floor
X_PROBE = [0.0, 0.005, 0.05, 0.3, 0.5, 0.77, 1.0]


def random_poles(B=4, P=6, seed=0, complex_residues=False,
                 gamma_min=GAMMA_FLOOR, gamma_spread=5.0, eps_spread=8.0):
    """A random, *constraint-satisfying* pole set in dimensionless units."""
    g = torch.Generator().manual_seed(seed)
    eps = eps_spread * torch.randn(B, P, generator=g, dtype=DT)
    gamma = gamma_min + torch.rand(B, P, generator=g, dtype=DT) * gamma_spread
    a = torch.softmax(torch.randn(B, P, generator=g, dtype=DT), dim=-1)
    if complex_residues:
        t = 0.5 * torch.tanh(torch.randn(B, P, generator=g, dtype=DT))
        b = a * (t - (a * t).sum(-1, keepdim=True))     # sum(b)=0, a=0 => b=0
    else:
        b = torch.zeros_like(a)
    return PoleSet(eps, gamma, a, b)


def adaptive_reference(poles: PoleSet, xs):
    """Independent ground truth: adaptive quadrature of A(nu) K(x, nu) over R."""
    eps = poles.eps[0].numpy()
    gam = poles.gamma[0].numpy()
    a = poles.a[0].numpy()
    b = poles.b[0].numpy()

    def A(nu):
        d = nu - eps
        return np.sum((a * gam - b * d) / (d * d + gam * gam)) / np.pi

    brk = sorted(set(np.sort(eps).tolist()))
    out = []
    for x in xs:
        def f(nu, x=x):
            return A(nu) * np.exp(-np.logaddexp(x * nu, (x - 1.0) * nu))
        v = integrate.quad(f, -np.inf, brk[0], limit=400,
                           epsabs=1e-14, epsrel=1e-13)[0]
        for lo, hi in zip(brk[:-1], brk[1:]):
            v += integrate.quad(f, lo, hi, limit=400,
                                epsabs=1e-14, epsrel=1e-13)[0]
        v += integrate.quad(f, brk[-1], np.inf, limit=400,
                            epsabs=1e-14, epsrel=1e-13)[0]
        out.append(v)
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #

def test_kernel_endpoint_identities():
    nu = torch.linspace(-800, 800, 4001, dtype=DT)
    k0 = fermi_kernel(torch.zeros(1, dtype=DT), nu)
    k1 = fermi_kernel(torch.ones(1, dtype=DT), nu)
    assert torch.allclose(k0, torch.sigmoid(nu), atol=1e-16, rtol=1e-15)
    assert torch.allclose(k1, torch.sigmoid(-nu), atol=1e-16, rtol=1e-15)
    # The identity behind the sum rule G(0) + G(beta) = int A.
    assert torch.max((k0 + k1 - 1.0).abs()) < 1e-16


def test_kernel_no_overflow_and_symmetry():
    nu = torch.tensor([-1e6, -1e3, -1.0, 0.0, 1.0, 1e3, 1e6], dtype=DT)
    x = torch.tensor([0.0, 1e-6, 0.25, 0.5, 0.75, 1.0 - 1e-6, 1.0], dtype=DT)
    K = fermi_kernel(x[:, None], nu[None, :])
    assert torch.isfinite(K).all()
    assert (K >= 0).all() and (K <= 1).all()
    assert torch.allclose(fermi_kernel(x, torch.zeros(1, dtype=DT)),
                          torch.full_like(x, 0.5))
    # K(1-x, -nu) = K(x, nu): the particle-hole reflection of the kernel.
    assert torch.allclose(K, fermi_kernel((1 - x)[:, None], -nu[None, :]))


# --------------------------------------------------------------------------- #
# Quadrature accuracy against an independent reference
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("complex_residues", [False, True])
@pytest.mark.parametrize("eps_spread,nu_c", [(8.0, 120.0), (60.0, 400.0)])
def test_forward_map_matches_adaptive_quadrature(complex_residues, eps_spread, nu_c):
    poles = random_poles(B=1, P=6, seed=3, complex_residues=complex_residues,
                         eps_spread=eps_spread)
    quad = SpectralQuadrature(nu_c=nu_c, h=0.09, n_tail=32, dtype=DT)
    G = PoleGreens(quad)(poles, torch.tensor(X_PROBE, dtype=DT))[0].numpy()
    ref = adaptive_reference(poles, X_PROBE)
    assert np.abs(G - ref).max() < 1e-8, f"max error {np.abs(G - ref).max():.3e}"


def test_accuracy_holds_at_the_width_floor():
    """Worst case: every pole pinned at gamma_min, spread across the band."""
    poles = random_poles(B=1, P=6, seed=1, complex_residues=True,
                         gamma_min=GAMMA_FLOOR, gamma_spread=0.0, eps_spread=60.0)
    quad = SpectralQuadrature(nu_c=400.0, h=0.09, n_tail=32, dtype=DT)
    G = PoleGreens(quad)(poles, torch.tensor(X_PROBE, dtype=DT))[0].numpy()
    ref = adaptive_reference(poles, X_PROBE)
    err = np.abs(G - ref).max()
    # Four orders below the DQMC bin-mean noise (~2.5e-5).
    assert err < 1e-8, f"max error {err:.3e}"


def test_width_floor_sits_above_the_quadratures_resolution_limit():
    quad = SpectralQuadrature.for_beta(beta=10.0, omega_c=20.0, dtype=DT)
    assert GAMMA_FLOOR > quad.min_resolvable_gamma


def test_spacing_dependence_is_exponential_as_predicted():
    """Poisson summation predicts error ~ 2 exp(-2 pi gamma / h); check the trend."""
    poles = random_poles(B=1, P=4, seed=1, gamma_min=GAMMA_FLOOR,
                         gamma_spread=0.0, eps_spread=8.0)
    ref = adaptive_reference(poles, X_PROBE)
    x = torch.tensor(X_PROBE, dtype=DT)
    errs = []
    for h in (0.20, 0.15, 0.09):
        quad = SpectralQuadrature(nu_c=120.0, h=h, n_tail=32, dtype=DT)
        errs.append(float(np.abs(PoleGreens(quad)(poles, x)[0].numpy() - ref).max()))
    assert errs[0] > errs[1] > errs[2]
    assert errs[0] / errs[2] > 1e3, "error should fall by orders of magnitude, not linearly"


# --------------------------------------------------------------------------- #
# Exact structural properties of the forward map
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("complex_residues", [False, True])
def test_sum_rule_at_the_endpoints(complex_residues):
    """G(0) + G(beta) = sum(a), because K(0,nu) + K(1,nu) = 1 identically."""
    poles = random_poles(complex_residues=complex_residues)
    quad = SpectralQuadrature(nu_c=120.0, h=0.09, dtype=DT)
    x = torch.linspace(0, 1, 51, dtype=DT)
    G = PoleGreens(quad)(poles, x)
    assert torch.allclose(G[:, 0] + G[:, -1], poles.a.sum(-1), atol=1e-8)


def test_reflection_maps_G_tau_to_G_beta_minus_tau():
    """(eps, gamma, a, b) -> (-eps, gamma, a, -b) is exactly tau -> beta - tau."""
    poles = random_poles(complex_residues=True)
    refl = PoleSet(-poles.eps, poles.gamma, poles.a, -poles.b)
    quad = SpectralQuadrature(nu_c=120.0, h=0.09, dtype=DT)
    fwd = PoleGreens(quad)
    x = torch.linspace(0, 1, 65, dtype=DT)
    assert torch.allclose(fwd(refl, x), torch.flip(fwd(poles, x), dims=[1]), atol=1e-12)


def test_forward_map_is_linear_in_the_residues():
    """A -> cA implies G -> cG: the property that makes zero map to zero."""
    poles = random_poles(complex_residues=True)
    quad = SpectralQuadrature(nu_c=120.0, h=0.09, dtype=DT)
    fwd = PoleGreens(quad)
    x = torch.linspace(0, 1, 33, dtype=DT)
    for c in (0.0, 0.37, -2.5):
        scaled = PoleSet(poles.eps, poles.gamma, c * poles.a, c * poles.b)
        assert torch.allclose(fwd(scaled, x), c * fwd(poles, x), atol=1e-12)


def test_G_is_positive_for_real_residues():
    poles = random_poles(complex_residues=False)
    quad = SpectralQuadrature(nu_c=120.0, h=0.09, dtype=DT)
    G = PoleGreens(quad)(poles, torch.linspace(0, 1, 201, dtype=DT))
    assert (G > 0).all()


def test_delta_limit_reproduces_the_bare_kernel():
    """gamma -> 0 makes a unit pole reproduce K(x, eps), with a residual O(gamma).

    The leftover is *physical* Lorentzian smearing, not quadrature error, so the
    test checks the linear-in-gamma scaling rather than a bare tolerance -- a
    quadrature bug would not halve when gamma halves.
    """
    eps0 = 2.5
    x = torch.linspace(0, 1, 41, dtype=DT)
    ref = fermi_kernel(x.unsqueeze(0), torch.tensor(eps0, dtype=DT))
    errs = []
    for gamma0 in (2e-2, 1e-2):
        quad = SpectralQuadrature(nu_c=40.0, h=gamma0 / 3.5, n_tail=32, dtype=DT)
        assert gamma0 > quad.min_resolvable_gamma
        poles = PoleSet(torch.full((1, 1), eps0, dtype=DT),
                        torch.full((1, 1), gamma0, dtype=DT),
                        torch.ones(1, 1, dtype=DT),
                        torch.zeros(1, 1, dtype=DT))
        errs.append((PoleGreens(quad)(poles, x) - ref).abs().max().item())
    assert errs[0] < 1e-2
    assert 1.7 < errs[0] / errs[1] < 2.3, f"expected O(gamma) scaling, got {errs}"


def test_under_resolved_poles_are_wrong_by_a_lot():
    """The resolution limit is real, not conservative: below it the rule is useless.

    Guards against anyone lowering the decoder's width floor without also
    refining the quadrature -- the failure is silent in the loss (a wrong G is
    still a smooth G) but enormous in magnitude.
    """
    quad = SpectralQuadrature(nu_c=120.0, h=0.09, dtype=DT)
    gamma0 = 1e-2
    assert gamma0 < quad.min_resolvable_gamma
    poles = PoleSet(torch.full((1, 1), 2.5, dtype=DT),
                    torch.full((1, 1), gamma0, dtype=DT),
                    torch.ones(1, 1, dtype=DT),
                    torch.zeros(1, 1, dtype=DT))
    x = torch.linspace(0, 1, 41, dtype=DT)
    G = PoleGreens(quad)(poles, x)
    ref = fermi_kernel(x.unsqueeze(0), torch.tensor(2.5, dtype=DT))
    assert (G - ref).abs().max() > 0.1


def test_tau_space_and_matsubara_forward_maps_agree():
    """FT of the quadrature G(x) reproduces the closed-form g(i nu_n)."""
    poles = random_poles(B=2, P=5, seed=7, complex_residues=True)
    quad = SpectralQuadrature(nu_c=120.0, h=0.09, dtype=DT)
    x = torch.linspace(0, 1, 8001, dtype=DT)
    G = PoleGreens(quad)(poles, x)

    n = torch.arange(6, dtype=DT)
    num = matsubara_from_tau(G, x, n_max=6)
    exact = greens_matsubara(poles, n)
    assert torch.allclose(num[..., 0], exact.real, atol=1e-5)
    assert torch.allclose(num[..., 1], exact.imag, atol=1e-5)


def test_gaussian_mixture_shares_the_quadrature():
    """The synthetic-data spectrum and the model spectrum use one rule."""
    g = torch.Generator().manual_seed(11)
    w = torch.softmax(torch.randn(3, 2, generator=g, dtype=DT), dim=-1)
    mu = 12.0 * torch.randn(3, 2, generator=g, dtype=DT)
    sigma = 1.0 + 6.0 * torch.rand(3, 2, generator=g, dtype=DT)
    quad = SpectralQuadrature(nu_c=200.0, h=0.09, dtype=DT)

    A = gaussian_mixture_spectrum(w, mu, sigma, quad.nodes)
    assert torch.allclose(quad.integrate(A), w.sum(-1), atol=1e-10)

    K = PoleGreens(quad).kernel_matrix(torch.linspace(0, 1, 41, dtype=DT))
    G = (A * quad.weights) @ K
    assert torch.allclose(G[:, 0] + G[:, -1], w.sum(-1), atol=1e-10)


def test_forward_map_is_differentiable():
    poles = random_poles(B=2, P=4, seed=31, complex_residues=True)
    for t in (poles.eps, poles.gamma, poles.a, poles.b):
        t.requires_grad_(True)
    quad = SpectralQuadrature(nu_c=120.0, h=0.09, dtype=DT)
    PoleGreens(quad)(poles, torch.linspace(0, 1, 33, dtype=DT)).square().sum().backward()
    for t in (poles.eps, poles.gamma, poles.a, poles.b):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def test_grouped_inference_over_ragged_grids():
    poles = random_poles(B=5, P=4, seed=5)
    quad = SpectralQuadrature(nu_c=120.0, h=0.09, dtype=DT)
    fwd = PoleGreens(quad)
    xs = [torch.linspace(0, 1, 21, dtype=DT), torch.linspace(0, 1, 61, dtype=DT)]
    index = torch.tensor([0, 1, 0, 1, 1])
    out = fwd.forward_grouped(poles, xs, index)
    assert [o.numel() for o in out] == [21, 61, 21, 61, 61]


# --------------------------------------------------------------------------- #
# Pole algebra
# --------------------------------------------------------------------------- #

def test_spectral_function_is_nonnegative_for_real_residues():
    poles = random_poles(B=4, P=8, seed=13, complex_residues=False)
    A = poles.spectral(torch.linspace(-60, 60, 4001, dtype=DT))
    assert (A >= 0).all()


def test_complex_residues_make_individual_poles_go_negative():
    """Documented pathology: b != 0 forces min_omega A_p < 0, scaling as 1/gamma."""
    poles = random_poles(B=2, P=5, seed=17, complex_residues=True)
    amax, amin = per_pole_extrema(poles.gamma, poles.a, poles.b)
    assert (amin < 0).any()
    p = poles[0:1]
    p = PoleSet(p.eps[:, :1], p.gamma[:, :1], p.a[:, :1], p.b[:, :1])
    span = 400 * p.gamma.item()
    w = torch.linspace(p.eps.item() - span, p.eps.item() + span, 400001, dtype=DT)
    A = p.spectral(w)
    assert abs(A.max().item() - amax[0, 0].item()) < 1e-5 * abs(amax[0, 0].item())
    assert abs(A.min().item() - amin[0, 0].item()) < 1e-5 * abs(amin[0, 0].item())


def test_tail_coefficients():
    """c1 = 0 under the zero-sum constraint; c2 matches a large-omega evaluation."""
    poles = random_poles(B=2, P=4, seed=19, complex_residues=True)
    c1, c2 = tail_coefficients(poles.eps, poles.gamma, poles.a, poles.b)
    assert torch.allclose(c1, torch.zeros_like(c1), atol=1e-14)
    w = torch.tensor([1e6], dtype=DT)
    assert torch.allclose(poles.spectral(w)[:, 0] * w[0] ** 2, c2, rtol=1e-5)


def test_positive_tail_for_real_residues():
    """b = 0 makes the 1/omega^2 tail coefficient strictly positive."""
    poles = random_poles(B=3, P=5, seed=21, complex_residues=False)
    _, c2 = tail_coefficients(poles.eps, poles.gamma, poles.a, poles.b)
    assert (c2 > 0).all()


def test_lorentzian_weight_normalisation():
    """Each Lorentzian integrates to a_p; window + analytic tail is exact."""
    poles = random_poles(B=1, P=3, seed=23, complex_residues=False)
    eps, gam, a = poles.eps[0], poles.gamma[0], poles.a[0]
    W = 1e4
    window = (a / math.pi) * (torch.atan((W - eps) / gam) + torch.atan((W + eps) / gam))
    omega = torch.linspace(-W, W, 400001, dtype=DT)
    numeric = torch.trapezoid(poles.spectral(omega), omega, dim=-1)[0]
    assert abs(numeric.item() - window.sum().item()) < 1e-5


def test_effective_pole_count():
    a = torch.tensor([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]], dtype=DT)
    assert torch.allclose(effective_pole_count(a),
                          torch.tensor([4.0, 1.0], dtype=DT), atol=1e-6)


def test_unit_conversion_roundtrip_and_rescaling():
    poles = random_poles(B=2, P=3, seed=29)
    beta = torch.tensor([7.5, 13.0], dtype=DT)
    back = poles.to_physical(beta).to_dimensionless(beta)
    assert torch.allclose(back.eps, poles.eps, atol=1e-12)
    assert torch.allclose(back.gamma, poles.gamma, atol=1e-12)

    # A_phys(omega) = beta * A_dimless(beta * omega): the spectrum rescales,
    # the weights do not.
    omega = torch.linspace(-4, 4, 501, dtype=DT)
    A_phys = poles.to_physical(beta).spectral(omega)
    A_dim = poles.spectral(omega.unsqueeze(0) * beta.unsqueeze(-1))
    assert torch.allclose(A_phys, A_dim * beta.unsqueeze(-1), atol=1e-10)


def test_spectral_from_poles_accepts_per_sample_grids():
    poles = random_poles(B=3, P=4, seed=33)
    shared = torch.linspace(-10, 10, 101, dtype=DT)
    per_sample = shared.unsqueeze(0).expand(3, -1)
    assert torch.allclose(spectral_from_poles(poles.eps, poles.gamma, poles.a,
                                              poles.b, shared),
                          spectral_from_poles(poles.eps, poles.gamma, poles.a,
                                              poles.b, per_sample))
