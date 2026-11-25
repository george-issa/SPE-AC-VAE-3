"""The posterior report: prior-free brackets and the accept/reject bookkeeping."""

import math
import os

import numpy as np
import pytest
import torch

from spevae.evaluation import cases as C
from spevae.posterior import DEFAULT_WINDOWS, continue_posterior, prior_free_bounds


def _true_window_weights(truth, beta):
    p = truth.poles
    eps, gam, a, b = (getattr(p, k)[0].double() for k in ("eps", "gamma", "a", "b"))

    def cdf(w):
        if math.isinf(w):
            return 0.0 if w < 0 else float(a.sum())
        u = w * beta - eps
        return float((a * (torch.atan(u / gam) / math.pi + 0.5)
                      - b * torch.log(u * u + gam * gam) / (2 * math.pi)).sum())
    return [cdf(hi) - cdf(lo) for lo, hi in DEFAULT_WINDOWS]


def test_prior_free_brackets_contain_the_truth_and_tighten_with_the_noise():
    widths = {}
    for noise in (1e-2, 1e-4):
        inst = C.draw(C.realize("lorentz pair"), noise, 7)
        tr = inst.truth
        bnd = prior_free_bounds(inst.G[0], tr.x, tr.beta, inst.sigma[0], inst.isr)
        want = _true_window_weights(tr, tr.beta)
        for b, w in zip(bnd, want):
            assert b["feasible"] and b["lo"] <= b["hi"]
            assert b["lo"] - 1e-6 <= w <= b["hi"] + 1e-6, (b, w)       # the truth is admitted
        widths[noise] = sum(b["hi"] - b["lo"] for b in bnd)
    assert widths[1e-4] < widths[1e-2]                                 # better data, tighter brackets


def test_posterior_report_on_the_reference_flow():
    path = "out/flow/F_60k.pt"
    if not os.path.exists(path):
        pytest.skip("reference flow checkpoint not present")
    from spevae.models.flow import load_flow
    flow = load_flow(path).eval()
    inst = C.draw(C.realize("lorentz pair"), 1e-3, 7)
    rep = continue_posterior(flow, inst.G, inst.truth.x, 8.0, inst.sigma, inst.isr, n_samples=48,
                             generator=torch.Generator().manual_seed(1))
    assert rep.n_accepted >= 40 and rep.n_samples == 48
    assert 0.8 < np.median(rep.chi2) < 1.2                             # samples fit at the noise level
    assert (rep.band["q05"] <= rep.band["q95"] + 1e-12).all()
    assert abs(sum(w["mean"] for w in rep.windows) - 1.0) < 1e-6       # window weights partition the norm
    want = _true_window_weights(inst.truth, 8.0)
    inside = [w["q05"] - 0.05 <= t <= w["q95"] + 0.05 for w, t in zip(rep.windows, want)]
    assert sum(inside) >= 6
    assert all(w["lo"] - 1e-6 <= w["q95"] and w["q05"] <= w["hi"] + 1e-6 for w in rep.windows)
    assert "posterior from" in rep.summary() and not rep.warnings


def test_small_error_bars_draw_the_systematics_warning_unless_a_floor_is_set():
    path = "out/flow/F_60k.pt"
    if not os.path.exists(path):
        pytest.skip("reference flow checkpoint not present")
    from spevae.models.flow import load_flow
    flow = load_flow(path).eval()
    inst = C.draw(C.realize("lorentz pair"), 5e-5, 7)
    kw = dict(n_samples=16, bounds=False, max_births=0, generator=torch.Generator().manual_seed(1))
    rep = continue_posterior(flow, inst.G, inst.truth.x, 8.0, inst.sigma, inst.isr, **kw)
    assert any("Trotter" in w for w in rep.warnings)
    rep = continue_posterior(flow, inst.G, inst.truth.x, 8.0, inst.sigma, inst.isr,
                             sigma_floor=1e-3, **kw)
    assert not any("Trotter" in w for w in rep.warnings)
    assert rep.diagnostics["sigma_floor"] == 1e-3 and rep.diagnostics["relative_error_bar"] < 1e-4


def test_band_table_recovers_a_known_overdispersion():
    """Samples twice as wide as the truth's distribution: the table finds the narrower level."""
    from statistics import NormalDist
    from spevae.posterior import BAND_TARGETS, band_levels, fit_band_table

    rng = np.random.default_rng(0)
    n, S = 4000, 64
    truth = rng.normal(size=(n, 1))
    samples = 2.0 * rng.normal(size=(n, S))                       # over-dispersed by a factor 2
    pit = ((samples < truth).sum(1) + rng.random(n)) / (S + 1)
    need = np.abs(2 * pit - 1)[:, None]
    noise, beta = np.full(n, 1e-3), np.full(n, 8.0)
    table = fit_band_table(need, noise, beta)
    lev = band_levels(table, 1e-3, 8.0)
    nd = NormalDist()
    for t in BAND_TARGETS:                                        # exact: 2 Phi(z_t / 2) - 1
        want = 2 * nd.cdf(nd.inv_cdf(0.5 + t / 2) / 2.0) - 1
        assert abs(lev[t] - want) < 0.03, (t, lev[t], want)
    assert band_levels(None, 1e-3, 8.0) == {t: t for t in BAND_TARGETS}
    # strata: a stratum with too few truths falls back to the pooled levels
    noise[:50] = 1e-5
    t2 = fit_band_table(need, noise, beta, noise_edges=(1e-4,), min_truths=100)
    assert t2["truths"] == [[50], [n - 50]] and t2["levels"][0][0] == t2["pooled"]
    assert band_levels(t2, 3e-6, 8.0) == band_levels(t2, 1e-5, 8.0)


def test_a_band_table_narrows_the_window_intervals_and_nothing_else():
    path = "out/flow/F_60k.pt"
    if not os.path.exists(path):
        pytest.skip("reference flow checkpoint not present")
    from spevae.models.flow import load_flow
    flow = load_flow(path).eval()
    inst = C.draw(C.realize("lorentz pair"), 1e-3, 7)
    table = {"targets": [0.5, 0.68, 0.9, 0.95], "noise_edges": [], "beta_edges": [],
             "levels": [[[0.4, 0.55, 0.8, 0.88]]]}
    kw = dict(n_samples=48, bounds=True)
    a = continue_posterior(flow, inst.G, inst.truth.x, 8.0, inst.sigma, inst.isr, band_calibration=None,
                           generator=torch.Generator().manual_seed(3), **kw)
    b = continue_posterior(flow, inst.G, inst.truth.x, 8.0, inst.sigma, inst.isr, band_calibration=table,
                           generator=torch.Generator().manual_seed(3), **kw)
    # samples and the pointwise band of A(omega): untouched
    assert torch.equal(a.poles.eps, b.poles.eps)
    for k in ("mean", "q05", "q16", "q25", "q50", "q75", "q84", "q95"):
        assert np.array_equal(a.band[k], b.band[k])
    assert (a.band["q05"] <= a.band["q25"] + 1e-12).all() and (a.band["q75"] <= a.band["q95"] + 1e-12).all()
    # window intervals: narrower, nested, and never wider than the plain ones
    for wa, wb in zip(a.windows, b.windows):
        assert wb["q95"] - wb["q05"] <= wa["q95"] - wa["q05"] + 1e-12
        assert wb["q05"] <= wb["q25"] + 1e-12 and wb["q75"] <= wb["q95"] + 1e-12
    assert sum(w["q95"] - w["q05"] for w in b.windows) < 0.97 * sum(w["q95"] - w["q05"] for w in a.windows)
    assert b.diagnostics["band_calibrated"] and not a.diagnostics["band_calibrated"]
    assert "calibrated on held-out" in b.summary() and "not calibrated" in a.summary()


def test_weight_at_width_floor_counts_only_pinned_poles():
    from spevae.posterior import weight_at_width_floor
    g0 = 0.1 * math.pi
    a = torch.tensor([[0.6, 0.4, 0.0], [0.2, 0.8, 0.0], [0.5, 0.5, 0.0]])
    gam = torch.tensor([[g0 * 1.01, 2.0, 1.0], [g0 * 1.2, 0.9, 1.0], [g0 * 3.0, g0 * 1.3, 1.0]])
    # per sample 0.6, 0.2, 0.0 -> median 0.2; empty slots (a = 0) never count
    assert abs(weight_at_width_floor(a, gam, g0) - 0.2) < 1e-12
    assert weight_at_width_floor(a, torch.full_like(gam, 5.0), g0) == 0.0


def test_broad_spectra_do_not_draw_the_width_floor_warning():
    path = "out/flow/F_60k.pt"
    if not os.path.exists(path):
        pytest.skip("reference flow checkpoint not present")
    from spevae.models.flow import load_flow
    flow = load_flow(path).eval()
    inst = C.draw(C.realize("semicircle"), 1e-3, 7)
    rep = continue_posterior(flow, inst.G, inst.truth.x, 8.0, inst.sigma, inst.isr, n_samples=32,
                             bounds=False, generator=torch.Generator().manual_seed(2))
    assert rep.diagnostics["weight_at_width_floor"] < 0.1
    assert not any("minimum width" in w for w in rep.warnings)


def test_auto_floor_refits_line_spectra_and_leaves_broad_ones_alone():
    path = next((p for p in ("out/step3/F3_60k_s1.pt", "out/flow/F_60k.pt") if os.path.exists(p)), None)
    if path is None:
        pytest.skip("no flow checkpoint present")
    from spevae.models.flow import load_flow
    from spevae.posterior import FLOOR_FACTOR
    flow = load_flow(path).eval()
    beta = 8.0
    # two exact lines at omega = +-1 (a Hubbard atom, U = 2)
    x = torch.linspace(0.0, 1.0, 65)
    e = torch.tensor([-1.0, 1.0], dtype=torch.float64) * beta
    G = (0.5 * torch.exp(-x.double()[:, None] * e) / (1.0 + torch.exp(-e))).sum(-1).float()
    sigma = torch.full_like(G, 1e-4 * float(G.max()))
    G = G + sigma * torch.randn(65, generator=torch.Generator().manual_seed(5))
    table = {"targets": [0.5, 0.68, 0.9, 0.95], "noise_edges": [], "beta_edges": [],
             "levels": [[[0.4, 0.58, 0.84, 0.92]]]}
    kw = dict(n_samples=32, bounds=False, sigma_floor=None, band_calibration=table)
    a = continue_posterior(flow, G, x, beta, sigma, floor_factor=1.0,
                           generator=torch.Generator().manual_seed(2), **kw)
    b = continue_posterior(flow, G, x, beta, sigma, generator=torch.Generator().manual_seed(2), **kw)
    assert a.diagnostics["floor_factor"] == 1.0 and b.diagnostics["floor_factor"] == FLOOR_FACTOR
    assert any("floor_factor='auto'" in w for w in a.warnings)
    assert any("fitted again" in w for w in b.warnings)
    # narrow lines flagged: the band table is withheld and the window intervals span both
    # readings of the narrow poles, as fitted and as sharp lines
    for r in (a, b):
        assert r.diagnostics["band_table_withheld"] and not r.diagnostics["band_calibrated"]
        assert r.diagnostics["band_levels"][0.9] == 0.9 and "not applied" in r.summary()
        assert r.diagnostics["line_shape_readings"] and "both readings" in r.summary()
        for w in r.windows:
            assert w["q05"] == min(w["as_fitted"][0], w["sharp_lines"][0])
            assert w["q95"] == max(w["as_fitted"][1], w["sharp_lines"][1])
    # the lines sit at +-1, half a unit from the window edges at +-0.5 and +-1.5: read as sharp
    # lines their weight is all inside [0.5, 1.5); as fitted, part of it leaks into the neighbours
    mid = next(w for w in b.windows if w["window"] == (0.5, 1.5))
    assert mid["sharp_lines"][1] > mid["as_fitted"][1] - 1e-9 and mid["sharp_lines"][1] > 0.49
    assert b.diagnostics["weight_at_trained_width_floor"] == a.diagnostics["weight_at_width_floor"] >= 0.1
    assert b.diagnostics["weight_below_trained_width_floor"] > 0.1     # the fit used the room
    gmin = flow.coords.gamma_min / beta
    assert float(a.poles.gamma[a.poles.a != 0].min()) >= gmin * (1 - 1e-6)
    assert float(b.poles.gamma[b.poles.a != 0].min()) >= gmin / FLOOR_FACTOR * (1 - 1e-6)
    assert b.n_accepted >= a.n_accepted
    # a broad spectrum never triggers it: the two reports are the same report
    inst = C.draw(C.realize("semicircle"), 1e-3, 7)
    c, d = (continue_posterior(flow, inst.G, inst.truth.x, 8.0, inst.sigma, inst.isr, n_samples=24,
                               bounds=False, floor_factor=ff, generator=torch.Generator().manual_seed(2))
            for ff in (1.0, "auto"))                 # "auto" is the default
    assert d.diagnostics["floor_factor"] == 1.0 and not d.diagnostics["line_shape_readings"]
    assert "sharp_lines" not in d.windows[0]
    assert np.array_equal(c.band["mean"], d.band["mean"]) and np.array_equal(c.chi2, d.chi2)
    with pytest.raises(ValueError):
        continue_posterior(flow, inst.G, inst.truth.x, 8.0, inst.sigma, inst.isr, n_samples=4,
                           floor_factor="lower")


def test_central_interval_is_exact_at_any_sample_count():
    from spevae.posterior import central_interval
    rng = np.random.default_rng(0)
    for n in (25, 60, 200):
        x = rng.standard_normal((4000, n + 1))                 # n samples and one further draw
        hit = np.mean([lo <= r[-1] <= hi for r in x for lo, hi in [central_interval(r[:-1], 0.9)]])
        assert abs(hit - 0.9) < 0.015, (n, hit)
    lo, hi = central_interval(np.arange(10.0), 0.9)            # too few samples: the range
    assert (lo, hi) == (0.0, 9.0)
