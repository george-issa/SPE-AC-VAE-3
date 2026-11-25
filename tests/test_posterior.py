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
