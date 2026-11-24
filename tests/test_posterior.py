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
