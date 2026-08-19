"""Pole consolidation: the minimal-pole-style output.

The load-bearing claims: sub-resolution combs collapse to few poles while the
spectrum and the Green's function are preserved to far below the data noise
(chi^2-neutrality is *measured* through the exact forward map, not assumed);
both sum rules survive exactly; structure coarser than the resolution is left
alone.
"""

import math

import pytest
import torch

from spevae.physics import (
    PoleGreens,
    PoleSet,
    SpectralQuadrature,
    consolidate_poles,
    positivity_certificate,
)

BETA = 10.0
RES = math.pi / BETA          # first Matsubara: the thermal resolution scale


def comb(centre=0.0, half_span=0.75, n=10, gamma=0.4, weight=1.0, tilt=0.0):
    """n narrow poles tiling one broad feature (physical units)."""
    e = torch.linspace(centre - half_span, centre + half_span, n)
    g = torch.full((n,), gamma)
    a = torch.full((n,), weight / n)
    t = torch.linspace(-1.0, 1.0, n) * tilt
    b = a * (t - (a * t).sum() / a.sum())
    return e, g, a, b


def as_set(*parts):
    e = torch.cat([p[0] for p in parts]).unsqueeze(0)
    g = torch.cat([p[1] for p in parts]).unsqueeze(0)
    a = torch.cat([p[2] for p in parts]).unsqueeze(0)
    b = torch.cat([p[3] for p in parts]).unsqueeze(0)
    return PoleSet(eps=e, gamma=g, a=a, b=b)


def greens_dev(p1: PoleSet, p2: PoleSet):
    """Max |G1 - G2| through the exact forward map (dimensionless internally)."""
    quad = SpectralQuadrature(nu_c=BETA * 20.0, h=0.09, dtype=torch.float64)
    fwd = PoleGreens(quad)
    x = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    G = [fwd(p.to_dimensionless(BETA).to(torch.float64), x)[0] for p in (p1, p2)]
    return float((G[0] - G[1]).abs().max())


def test_comb_collapses_and_preserves_the_physics():
    raw = as_set(comb(centre=-1.5, weight=0.45, tilt=0.3),
                 comb(centre=+1.8, weight=0.55, tilt=-0.2))
    merged, (rep,) = consolidate_poles(raw, RES)
    assert rep.n_in == 20
    assert rep.n_out <= 8, rep                # 2x half_span 0.75 vs res 0.314
    assert rep.matsubara_err_polished < 5e-3, rep
    # both sum rules -- exact in the float64 pipeline, read back at the
    # input's float32 resolution
    assert float(merged.a.sum()) == pytest.approx(float(raw.a.sum()), rel=1e-6)
    assert float(merged.b.sum()) == pytest.approx(0.0, abs=1e-7)
    # the data cannot tell them apart: G deviation far below any noise floor
    assert greens_dev(raw, merged) < 2e-3
    # and the spectra agree on a dense grid
    w = torch.linspace(-8, 8, 3001)
    A1, A2 = raw.spectral(w)[0], merged.spectral(w)[0]
    l2 = float(((A1 - A2) ** 2).sum().sqrt() / (A1**2).sum().sqrt())
    assert l2 < 0.08, l2


def test_two_separated_features_give_two_poles_with_the_right_weights():
    raw = as_set(comb(centre=-1.5, half_span=0.12, n=6, weight=0.45),
                 comb(centre=+1.8, half_span=0.12, n=6, weight=0.55))
    merged, (rep,) = consolidate_poles(raw, RES)
    assert rep.n_out == 2, rep
    a = merged.a[0][merged.a[0] > 0]
    e = merged.eps[0][merged.a[0] > 0]
    order = torch.argsort(e)
    assert a[order][0] == pytest.approx(0.45, abs=5e-3)
    assert a[order][1] == pytest.approx(0.55, abs=5e-3)
    assert float(e[order][0]) == pytest.approx(-1.5, abs=0.05)
    assert float(e[order][1]) == pytest.approx(+1.8, abs=0.05)


def test_resolved_structure_is_left_alone():
    e = torch.tensor([[-3.0, -1.0, 1.2, 3.5]])
    g = torch.tensor([[0.4, 0.3, 0.5, 0.6]])
    a = torch.tensor([[0.2, 0.3, 0.3, 0.2]])
    b = torch.tensor([[0.05, -0.02, -0.04, 0.01]])
    raw = PoleSet(eps=e, gamma=g, a=a, b=b)
    merged, (rep,) = consolidate_poles(raw, RES)
    assert rep.n_out == 4
    order = torch.argsort(merged.eps[0])
    assert torch.allclose(merged.eps[0][order], e[0], atol=1e-6)
    assert torch.allclose(merged.gamma[0][order], g[0], atol=1e-6)
    assert torch.allclose(merged.a[0][order], a[0], atol=1e-6)
    assert torch.allclose(merged.b[0][order], b[0], atol=1e-6)


def test_polish_does_not_regress_and_positivity_survives_a_smooth_comb():
    raw = as_set(comb(centre=0.0, half_span=1.0, n=12, gamma=0.5))
    merged_np, (rep_np,) = consolidate_poles(raw, RES, polish_steps=0)
    merged, (rep,) = consolidate_poles(raw, RES)
    assert rep.matsubara_err_polished <= rep_np.matsubara_err_merged + 1e-12
    assert positivity_certificate(merged)[0].is_positive


def test_batch_padding_and_inert_fill():
    s1 = as_set(comb(centre=0.0, half_span=0.1, n=4))          # -> 1 cluster
    s2 = as_set(comb(centre=-2.0, half_span=0.1, n=2),
                comb(centre=2.0, half_span=0.1, n=2))          # -> 2 clusters
    raw = PoleSet(eps=torch.cat([s1.eps, s2.eps]),
                  gamma=torch.cat([s1.gamma, s2.gamma]),
                  a=torch.cat([s1.a, s2.a]),
                  b=torch.cat([s1.b, s2.b]))
    merged, reps = consolidate_poles(raw, RES)
    assert len(reps) == 2 and merged.shape[0] == 2
    assert merged.shape[1] == max(r.n_out for r in reps)
    # padding poles are spectrally inert
    for i, r in enumerate(reps):
        pad_a = merged.a[i][r.n_out:]
        assert (pad_a == 0).all()


def test_zero_weight_dust_is_absorbed_not_seeded():
    base = comb(centre=1.0, half_span=0.1, n=4)
    dust = (torch.tensor([-5.0]), torch.tensor([0.5]),
            torch.tensor([1e-9]), torch.tensor([0.0]))
    raw = as_set(base, dust)
    merged, (rep,) = consolidate_poles(raw, RES)
    assert rep.n_out == 1
    assert rep.dropped_weight < 1e-6
    assert float(merged.a.sum()) == pytest.approx(float(base[2].sum()), rel=1e-4)


def test_refit_recovers_a_two_pole_truth_from_a_comb():
    """The minimal-pole claim, end to end without any network: bins generated
    from a known 2-pole spectrum, a comb approximation of it consolidated at a
    coarse resolution and refit to the data must land back on the truth."""
    from spevae.data import from_bins
    from spevae.physics import refit_poles_to_greens

    truth = PoleSet(eps=torch.tensor([[-1.5, 1.8]]),
                    gamma=torch.tensor([[0.35, 0.45]]),
                    a=torch.tensor([[0.45, 0.55]]),
                    b=torch.zeros(1, 2))
    quad = SpectralQuadrature(nu_c=BETA * 20.0, h=0.09, dtype=torch.float64)
    tau = torch.linspace(0, BETA, 81, dtype=torch.float64)
    clean = PoleGreens(quad)(truth.to_dimensionless(BETA).to(torch.float64),
                             tau / BETA)[0]
    g = torch.Generator().manual_seed(3)
    noise = torch.randn(400, 81, generator=g, dtype=torch.float64) * 1e-4
    mg = from_bins(clean.unsqueeze(0) + noise, BETA, tau=tau,
                   dtype=torch.float64)

    comb_approx = as_set(comb(centre=-1.5, half_span=0.5, n=8, gamma=0.45,
                              weight=0.45),
                         comb(centre=+1.8, half_span=0.5, n=8, gamma=0.5,
                              weight=0.55))
    merged, (rep,) = consolidate_poles(comb_approx, 4.0 * RES)
    assert rep.n_out == 2
    cov = mg.covariance()
    final, before, after = refit_poles_to_greens(
        merged, BETA, mg.x, mg.mean, mg.sigma_mean,
        inv_sqrt_R=cov["inv_sqrt_R"], slices=cov["slices"], steps=2500)
    assert after < before
    assert after < 50.0, f"refit did not approach the data: {after}"
    order = torch.argsort(final.eps[0])
    assert float(final.eps[0][order][0]) == pytest.approx(-1.5, abs=0.05)
    assert float(final.eps[0][order][1]) == pytest.approx(+1.8, abs=0.05)
    assert float(final.a[0][order][0]) == pytest.approx(0.45, abs=0.03)
    assert float(final.a[0][order][1]) == pytest.approx(0.55, abs=0.03)
    assert float(final.a[0].sum()) == pytest.approx(1.0, rel=1e-5)
    assert float(final.b[0].sum()) == pytest.approx(0.0, abs=1e-7)
