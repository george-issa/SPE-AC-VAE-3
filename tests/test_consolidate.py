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


def test_mirror_refit_is_exactly_symmetric_and_recovers_a_symmetric_truth():
    """PH-symmetric truth (poles at +-2 with opposite b), bins generated from
    it; the mirror-constrained refit must recover it with A(w) = A(-w)
    holding identically, not approximately."""
    from spevae.data import from_bins
    from spevae.physics import refit_poles_to_greens

    truth = PoleSet(eps=torch.tensor([[-2.0, 2.0]]),
                    gamma=torch.tensor([[0.4, 0.4]]),
                    a=torch.tensor([[0.5, 0.5]]),
                    b=torch.tensor([[0.1, -0.1]]))
    quad = SpectralQuadrature(nu_c=BETA * 20.0, h=0.09, dtype=torch.float64)
    tau = torch.linspace(0, BETA, 81, dtype=torch.float64)
    clean = PoleGreens(quad)(truth.to_dimensionless(BETA).to(torch.float64),
                             tau / BETA)[0]
    g = torch.Generator().manual_seed(11)
    noise = torch.randn(400, 81, generator=g, dtype=torch.float64) * 1e-4
    mg = from_bins(clean.unsqueeze(0) + noise, BETA, tau=tau,
                   dtype=torch.float64)
    cov = mg.covariance()

    # deliberately asymmetric comb init: the fold must handle it
    init = as_set(comb(centre=-2.1, half_span=0.3, n=5, gamma=0.5, weight=0.48),
                  comb(centre=+1.9, half_span=0.3, n=5, gamma=0.5, weight=0.52))
    merged, _ = consolidate_poles(init, 4.0 * RES)
    final, before, after = refit_poles_to_greens(
        merged, BETA, mg.x, mg.mean, mg.sigma_mean,
        inv_sqrt_R=cov["inv_sqrt_R"], slices=cov["slices"], steps=2500,
        ph_symmetric=True, fold_tolerance=4.0 * RES)
    assert after < before and after < 100.0, (before, after)

    w = torch.linspace(-30.0, 30.0, 4001, dtype=final.eps.dtype)
    A = final.spectral(w)
    assert torch.allclose(A, A.flip(-1), atol=1e-7), "A(w) != A(-w)"
    assert float(final.b.sum()) == pytest.approx(0.0, abs=1e-12)
    assert float(final.a.sum()) == pytest.approx(1.0, rel=1e-6)
    # the mirror pair structure is explicit
    h = final.shape[1] // 2
    assert torch.allclose(final.eps[0][:h], -final.eps[0][h:])
    assert torch.allclose(final.b[0][:h], -final.b[0][h:])
    # and the recovered feature sits where the truth put it
    i = int(torch.argmax(final.a[0][:h]))
    assert abs(float(final.eps[0][:h][i])) == pytest.approx(2.0, abs=0.1)


def test_positivity_weight_removes_the_in_gap_dip():
    """A pole set constructed to dip negative in its gap: the refit with
    positivity pressure must fix the dip without leaving the noise floor."""
    from spevae.data import from_bins
    from spevae.physics import refit_poles_to_greens

    truth = PoleSet(eps=torch.tensor([[-2.0, 2.0]]),
                    gamma=torch.tensor([[0.5, 0.5]]),
                    a=torch.tensor([[0.5, 0.5]]),
                    b=torch.zeros(1, 2))
    quad = SpectralQuadrature(nu_c=BETA * 20.0, h=0.09, dtype=torch.float64)
    tau = torch.linspace(0, BETA, 81, dtype=torch.float64)
    clean = PoleGreens(quad)(truth.to_dimensionless(BETA).to(torch.float64),
                             tau / BETA)[0]
    g = torch.Generator().manual_seed(13)
    noise = torch.randn(300, 81, generator=g, dtype=torch.float64) * 3e-4
    mg = from_bins(clean.unsqueeze(0) + noise, BETA, tau=tau,
                   dtype=torch.float64)
    cov = mg.covariance()

    # init whose strong opposite dispersive parts dip negative between peaks
    init = PoleSet(eps=torch.tensor([[-2.0, 2.0]]),
                   gamma=torch.tensor([[0.5, 0.5]]),
                   a=torch.tensor([[0.5, 0.5]]),
                   b=torch.tensor([[0.24, -0.24]]))
    final, _, after = refit_poles_to_greens(
        init, BETA, mg.x, mg.mean, mg.sigma_mean,
        inv_sqrt_R=cov["inv_sqrt_R"], slices=cov["slices"], steps=4000,
        positivity_weight=30.0)
    cert = positivity_certificate(final[0:1])[0]
    assert cert.is_positive or cert.min_A > -1e-6, cert
    assert after < 100.0, f"positivity pressure destroyed the fit: {after}"


def test_pair_positivity_condition_is_exact():
    """The derived closed form: a mirror pair is non-negative everywhere iff
    |b| <= a*gamma/eps.  Checked at and just beyond the boundary."""
    import numpy as np
    rng = np.random.default_rng(4)
    w = np.linspace(-400, 400, 800001)
    for _ in range(200):
        eps = rng.uniform(0.1, 8.0); gam = rng.uniform(0.05, 3.0); a = 1.0
        for frac, expect_nonneg in ((0.999, True), (1.02, False)):
            b = frac * a * gam / eps * (1 if rng.random() < 0.5 else -1)
            t1 = (a/2*gam - b/2*(w-eps)) / ((w-eps)**2 + gam**2)
            t2 = (a/2*gam + b/2*(w+eps)) / ((w+eps)**2 + gam**2)
            A = (t1 + t2) / np.pi
            if expect_nonneg:
                assert A.min() > -1e-12, (eps, gam, b, A.min())
            else:
                assert A.min() < 0, (eps, gam, b)


def test_pair_bound_refit_is_certified_positive_by_construction():
    """Mirror refit with the structural pair bound: certificate must pass for
    the result, whatever the data wanted."""
    from spevae.data import from_bins
    from spevae.physics import refit_poles_to_greens

    truth = PoleSet(eps=torch.tensor([[-2.0, 2.0]]),
                    gamma=torch.tensor([[0.4, 0.4]]),
                    a=torch.tensor([[0.5, 0.5]]),
                    b=torch.tensor([[0.15, -0.15]]))
    quad = SpectralQuadrature(nu_c=BETA * 20.0, h=0.09, dtype=torch.float64)
    tau = torch.linspace(0, BETA, 81, dtype=torch.float64)
    clean = PoleGreens(quad)(truth.to_dimensionless(BETA).to(torch.float64),
                             tau / BETA)[0]
    g = torch.Generator().manual_seed(17)
    noise = torch.randn(300, 81, generator=g, dtype=torch.float64) * 2e-4
    mg = from_bins(clean.unsqueeze(0) + noise, BETA, tau=tau,
                   dtype=torch.float64)
    cov = mg.covariance()
    init = as_set(comb(centre=-2.0, half_span=0.2, n=4, gamma=0.5, weight=0.5),
                  comb(centre=+2.0, half_span=0.2, n=4, gamma=0.5, weight=0.5))
    merged, _ = consolidate_poles(init, 4.0 * RES)
    final, before, after = refit_poles_to_greens(
        merged, BETA, mg.x, mg.mean, mg.sigma_mean,
        inv_sqrt_R=cov["inv_sqrt_R"], slices=cov["slices"], steps=3000,
        ph_symmetric=True, fold_tolerance=4.0 * RES, pair_bound=True)
    assert after < before
    cert = positivity_certificate(final[0:1])[0]
    assert cert.is_positive, cert
    # the bound holds pole by pole
    e, gm, a, b = final.eps[0], final.gamma[0], final.a[0], final.b[0]
    live = a > 0
    assert (b[live].abs() <= a[live] * gm[live] / e[live].abs() + 1e-9).all()
    # symmetry still exact
    w = torch.linspace(-30, 30, 3001, dtype=final.eps.dtype)
    A = final.spectral(w)
    assert torch.allclose(A, A.flip(-1), atol=1e-7)


def test_sharpness_verdict_discriminates():
    """A genuinely narrow pole at high SNR must be declared data-demanded;
    a broad structure must come back undecidable."""
    from spevae.data import from_bins
    from spevae.physics import sharpness_verdict

    quad = SpectralQuadrature(nu_c=BETA * 20.0, h=0.05, dtype=torch.float64)
    tau = torch.linspace(0, BETA, 101, dtype=torch.float64)
    g = torch.Generator().manual_seed(23)

    def bins_from(truth, noise):
        clean = PoleGreens(quad)(truth.to_dimensionless(BETA).to(torch.float64),
                                 tau / BETA)[0]
        eta = torch.randn(400, 101, generator=g, dtype=torch.float64) * noise
        return from_bins(clean.unsqueeze(0) + eta, BETA, tau=tau,
                         dtype=torch.float64)

    # narrow pole (gamma far below pi/beta) at tiny noise: smearing must be
    # violently rejected by the data
    narrow = PoleSet(eps=torch.tensor([[-1.0, 1.0]]),
                     gamma=torch.tensor([[0.05, 0.05]]),
                     a=torch.tensor([[0.5, 0.5]]), b=torch.zeros(1, 2))
    mg = bins_from(narrow, 1e-5)
    cov = mg.covariance()
    v = sharpness_verdict(narrow, BETA, mg.x, mg.mean, mg.sigma_mean,
                          inv_sqrt_R=cov["inv_sqrt_R"], slices=cov["slices"],
                          steps=1200)
    assert v["data_demanded"], v["n_dchi2"]

    # broad structure: floor at pi/beta binds nothing; undecidable
    broad = PoleSet(eps=torch.tensor([[-1.0, 1.0]]),
                    gamma=torch.tensor([[0.8, 0.8]]),
                    a=torch.tensor([[0.5, 0.5]]), b=torch.zeros(1, 2))
    mg2 = bins_from(broad, 1e-4)
    cov2 = mg2.covariance()
    v2 = sharpness_verdict(broad, BETA, mg2.x, mg2.mean, mg2.sigma_mean,
                           inv_sqrt_R=cov2["inv_sqrt_R"], slices=cov2["slices"],
                           steps=1200)
    assert not v2["data_demanded"], v2["n_dchi2"]


def test_rank_scan_escapes_the_comb_aliasing_trap():
    """The measured trap: a dense 6-pole comb fits a 3-rung truth at chi^2 ~ 1
    and blocks every single-pole pruning move.  The rank scan must select
    n = 3 and land the rungs, because the 3-pole refit reaches the noise
    floor and the elbow prefers the smallest count that does."""
    from spevae.physics.consolidate import rank_scan

    beta, noise = 8.0, 1e-4
    truth = PoleSet(eps=torch.tensor([[0.5, 1.5, 2.5]]) * beta,
                    gamma=torch.tensor([[0.3, 0.3, 0.3]]) * beta,
                    a=torch.tensor([[0.540, 0.297, 0.163]]),
                    b=torch.zeros(1, 3)).to_physical(1.0)  # already physical scale
    truth = PoleSet(eps=torch.tensor([[0.5, 1.5, 2.5]]),
                    gamma=torch.tensor([[0.3, 0.3, 0.3]]),
                    a=torch.tensor([[0.540, 0.297, 0.163]]),
                    b=torch.zeros(1, 3))
    quad = SpectralQuadrature(nu_c=beta * 20.0, h=0.09, dtype=torch.float64)
    fwd = PoleGreens(quad)
    x = torch.linspace(0.0, 1.0, 65, dtype=torch.float64).unsqueeze(0)
    G = fwd(truth.to_dimensionless(beta).to(torch.float64), x[0])
    g = torch.Generator().manual_seed(0)
    sigma = torch.full_like(G, noise)
    G = G + sigma * torch.randn(G.shape, generator=g, dtype=torch.float64)

    # the alias: a denser, finer comb spanning the same envelope, plus dust
    dense = PoleSet(
        eps=torch.tensor([[0.45, 1.0, 1.45, 2.0, 2.55, 3.1, -0.9, 4.0]]),
        gamma=torch.tensor([[0.28, 0.3, 0.3, 0.28, 0.3, 0.3, 0.5, 0.4]]),
        a=torch.tensor([[0.40, 0.16, 0.17, 0.12, 0.09, 0.04, 0.01, 0.01]]),
        b=torch.zeros(1, 8))

    sel, trace = rank_scan(dense, beta, x, G, sigma, n_max=6,
                           refit_steps=3000, polish_steps=1000)
    # The COUNT is what the sequential LR recovers deterministically.
    assert trace.selected_n == 3, str(trace)
    # The selected fit is statistically equivalent to the truth basin
    # (chi2 at truth is ~0.92; 9.5/L_eff of slack is the LR resolution).
    assert trace.chi2_selected < 1.1, str(trace)
    got = torch.sort(sel.eps[0]).values
    want = torch.tensor([0.5, 1.5, 2.5], dtype=got.dtype)
    # Dominant rung: pinned by the data.  Faint rungs: recovered only to the
    # measured information limit (several chi2-equivalent minima within ~0.5).
    assert abs(float(got[0] - want[0])) < 0.15, got
    assert (got - want).abs().max() < 0.6, got
    # And the selected spectrum is indistinguishable from the truth's.
    quad2 = SpectralQuadrature(nu_c=beta * 20.0, h=0.09)
    from spevae.losses import spectral_w1
    w1 = float(spectral_w1(sel.to_dimensionless(beta).spectral(quad2.nodes),
                           truth.to_dimensionless(beta).spectral(quad2.nodes),
                           quad2, beta=beta))
    assert w1 < 0.06, w1


def test_ladder_verdict_classifies_the_banked_exemplars():
    """The three dual-output chi2(n) curves are the calibration set: the
    asym triplet is pole-closed (cliff then shelf), the Holstein is
    continuum-like (steady multiplicative arrival, no shelf), and the
    fano band -- a resolved band at the comb/smooth boundary -- is
    honestly ambiguous."""
    from spevae.physics import ladder_verdict
    ns = list(range(1, 9))
    triplet = [12677.2, 9.7518, 0.99002, 0.86568, 1.00604, 1.14400,
               1.58054, 2.29217]
    fano = [29181.1, 127.876, 15.3683, 2.21945, 0.93060, 1.08756,
            1.19818, 22.1196]
    holstein = [94996.9, 1134.45, 209.754, 1.93786, 0.92890, 0.18872,
                0.34359, 1.60462]
    v_t = ladder_verdict(ns, triplet, floor_target=1.0)
    v_f = ladder_verdict(ns, fano, floor_target=1.0)
    v_h = ladder_verdict(ns, holstein, floor_target=0.175)
    assert v_t.verdict == "pole-closed", str(v_t)
    assert v_t.n_arrival == 3 and v_t.shelf >= 2
    assert v_f.verdict == "ambiguous", str(v_f)
    assert v_h.verdict == "continuum-like", str(v_h)
    assert v_h.n_arrival == 6 and v_h.shelf <= 1
    # noiseless dichotomy sanity: a machine-crash curve is pole-closed,
    # a clean geometric decay ending at its floor is continuum-like
    crash = [1e2, 1e-1, 1e-9, 1.1e-9, 1.2e-9, 1.0e-9]
    geom = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]
    assert ladder_verdict(range(1, 7), crash).verdict == "pole-closed"
    assert ladder_verdict(range(1, 7), geom).verdict == "continuum-like"
