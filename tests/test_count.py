"""The pole-count posterior and the resampling that applies it."""

import os

import pytest
import torch

from spevae.models.count import (CountPosterior, count_weights, load_count, resample_by_count,
                                 save_count)


def test_count_weights_reproduce_the_target_marginal():
    p = torch.tensor([0.5, 0.3, 0.2] + [0.0] * 9)
    counts = torch.tensor([1] * 2 + [2] * 10 + [3] * 4 + [7] * 4)
    w = count_weights(p, counts)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    for n, want in ((1, 0.5), (2, 0.3), (3, 0.2), (7, 0.0)):
        assert abs(float(w[counts == n].sum()) - want) < 1e-6      # p is float32
    # a floor keeps every count that occurs alive
    assert float(count_weights(p, counts, floor=0.12)[counts == 7].sum()) > 0.0


def test_resampling_matches_counts_and_reports_what_is_missing():
    p = torch.tensor([0.0, 0.6, 0.0, 0.4] + [0.0] * 8)           # two or four poles
    counts = torch.tensor([2] * 5 + [3] * 40 + [5] * 55)          # no four-pole sample at all
    take, d = resample_by_count(p, counts, generator=torch.Generator().manual_seed(0), floor=0.0)
    assert take.numel() == counts.numel() and bool((counts[take] == 2).all())
    assert abs(d["count_mass_covered"] - 0.6) < 1e-6 and d["count_distinct_samples"] == 5
    assert abs(d["count_ess"] - 5.0) < 1e-9
    # counts already distributed as the target: nothing changes but the order statistics
    counts = torch.tensor([2] * 60 + [4] * 40)
    take, d = resample_by_count(p, counts, generator=torch.Generator().manual_seed(0), floor=0.0)
    assert abs(d["count_ess"] - 100.0) < 1e-6 and d["count_distinct_samples"] == 100


def test_classifier_shapes_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = CountPosterior(num_poles=12, d_model=32, enc_layers=1, n_heads=4, d_ff=64)
    x = torch.linspace(0, 1, 33)
    G = torch.rand(3, 33) * 0.4 + 0.05
    p = m.probs(G, x, 8.0, sigma=torch.full((3, 33), 1e-3))
    assert p.shape == (3, 12) and torch.allclose(p.sum(-1), torch.ones(3), atol=1e-6)
    path = save_count(m, str(tmp_path / "c.pt"), note="t")
    m2 = load_count(path)
    assert torch.allclose(m2.probs(G, x, 8.0, sigma=torch.full((3, 33), 1e-3)), p, atol=1e-6)
    assert m2.meta["note"] == "t"


def test_posterior_with_a_count_model_keeps_its_size_and_fit():
    path = "out/flow/F_60k.pt"
    if not os.path.exists(path):
        pytest.skip("reference flow checkpoint not present")
    from spevae.evaluation import cases as C
    from spevae.models.flow import load_flow
    from spevae.posterior import continue_posterior

    flow = load_flow(path).eval()

    class TwoPoles(torch.nn.Module):                      # a classifier that is sure of two poles
        def probs(self, G, x, beta, sigma=None):
            p = torch.full((1, 12), 1e-6)
            p[0, 1] = 1.0
            return p / p.sum()

    inst = C.draw(C.realize("lorentz pair"), 1e-3, 7)
    kw = dict(n_samples=48, bounds=False)
    plain = continue_posterior(flow, inst.G, inst.truth.x, 8.0, inst.sigma, inst.isr,
                               generator=torch.Generator().manual_seed(1), **kw)
    rep = continue_posterior(flow, inst.G, inst.truth.x, 8.0, inst.sigma, inst.isr,
                             count_model=TwoPoles(), count_floor=0.0,
                             generator=torch.Generator().manual_seed(1), **kw)
    assert rep.poles.a.shape == plain.poles.a.shape and rep.chi2.shape == plain.chi2.shape
    d = rep.diagnostics
    assert 0.0 <= d["count_mass_covered"] <= 1.0 and d["count_ess"] >= 1.0
    had_two = bool(((plain.poles.a != 0).sum(-1) == 2).any())
    if had_two:
        assert bool(((rep.poles.a != 0).sum(-1) == 2).all()) and d["count_mass_covered"] > 0.99
    else:
        assert any("count posterior" in w for w in rep.warnings)
