"""Structured-template fits and model selection."""

import torch

from spevae.physics import PoleSet, SpectralQuadrature, PoleGreens
from spevae.physics.templates import (TemplateFit, chi2_quantile_95,
                                      fit_ladder, ladder_starts, select_model)

BETA = 8.0


def _probe(eps, gam, w, noise=1e-4, seed=0):
    truth = PoleSet(eps=torch.tensor([eps]) * BETA,
                    gamma=torch.tensor([gam]) * BETA,
                    a=torch.tensor([w]), b=torch.zeros(1, len(eps)))
    quad = SpectralQuadrature.for_beta(BETA, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    G = PoleGreens(quad)(truth.to(torch.float64), x)
    g = torch.Generator().manual_seed(seed)
    sigma = torch.full_like(G, noise)
    return x, G + sigma * torch.randn(G.shape, generator=g,
                                      dtype=torch.float64), sigma


def test_ladder_template_recovers_structure_sharply():
    # 3-rung truth: the shared spacing must pin ALL rungs, including the
    # faint one the free fits could only place to +-0.3
    e = [0.5, 1.5, 2.5]
    a = [0.55 ** k for k in range(3)]; s = sum(a)
    x, G, sigma = _probe(e, [0.3] * 3, [v / s for v in a])
    fit = fit_ladder(BETA, x, G, sigma, m=3,
                     starts=[(0.4, 0.6), (0.6, 1.2), (0.5, 2.0)])
    assert fit.chi2 < 1.1, fit
    assert abs(fit.params["eps0"] - 0.5) < 0.05, fit
    assert abs(fit.params["Omega"] - 1.0) < 0.05, fit
    assert abs(fit.params["gamma"] - 0.3) < 0.05, fit


def test_wrong_rung_count_fits_worse_at_equal_complexity():
    e = [0.5, 1.5, 2.5]
    a = [0.55 ** k for k in range(3)]; s = sum(a)
    x, G, sigma = _probe(e, [0.3] * 3, [v / s for v in a])
    starts = [(0.4, 0.6), (0.6, 1.2), (0.5, 2.0)]
    f3 = fit_ladder(BETA, x, G, sigma, m=3, starts=starts)
    f2 = fit_ladder(BETA, x, G, sigma, m=2, starts=starts)
    f5 = fit_ladder(BETA, x, G, sigma, m=5, starts=starts)
    assert f3.chi2 < f2.chi2
    best = select_model([f2, f3, f5], L_eff=65)
    assert best.m == 3, [str(f) for f in (f2, f3, f5)]


def test_template_beats_free_poles_via_parsimony():
    # a free 3-pole fit with marginally better chi2 must NOT displace the
    # 4-parameter template: 8 extra parameters demand ~15.5 in total chi^2
    tpl = TemplateFit(name="ladder", m=3, k_params=4, chi2=0.92,
                      poles=None, params={})
    free = TemplateFit(name="free", m=3, k_params=12, chi2=0.88,
                       poles=None, params={})
    assert select_model([tpl, free], L_eff=65).name == "ladder"
    # ...but a decisive improvement does displace it
    free2 = TemplateFit(name="free", m=5, k_params=12, chi2=0.60,
                        poles=None, params={})
    assert select_model([tpl, free2], L_eff=65).name == "free"


def test_quantile_and_starts_helpers():
    assert abs(chi2_quantile_95(4) - 9.49) < 0.1
    amort = PoleSet(eps=torch.tensor([[0.5, 1.4, 2.6, -0.2]]),
                    gamma=torch.full((1, 4), 0.3),
                    a=torch.tensor([[0.5, 0.3, 0.15, 0.05]]),
                    b=torch.zeros(1, 4))
    st = ladder_starts(amort)
    assert len(st) >= 2
    assert all(o > 0 for _, o in st)
