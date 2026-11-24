"""The least-squares solver and the exact pole Jacobian it runs on."""

import math

import torch

from spevae.physics import PoleGreens, PoleSet, SpectralQuadrature, lm_solve, rowwise_jacobian
from spevae.physics.lsq import BUDGET, STATIONARY, STOPPED

DD = torch.float64


def test_pole_jacobian_matches_autograd():
    torch.manual_seed(0)
    quad = SpectralQuadrature.for_beta(8.0, dtype=DD)
    fwd = PoleGreens(quad)
    x = torch.linspace(0, 1, 33, dtype=DD)
    B, P = 3, 5
    th = torch.stack([torch.randn(B, P) * 8, 2.6 + torch.rand(B, P) * 6, torch.rand(B, P),
                      torch.randn(B, P) * 0.2], -1).to(DD).requires_grad_(True)
    G = fwd(PoleSet(th[..., 0], th[..., 1], th[..., 2], th[..., 3]), x)
    auto = torch.stack([torch.autograd.grad(G[:, k].sum(), th, retain_graph=True)[0]
                        for k in range(G.shape[1])], 1)                    # (B, L, P, 4)
    d = th.detach()
    G2, dG = fwd.jacobian(PoleSet(d[..., 0], d[..., 1], d[..., 2], d[..., 3]), x)
    assert dG.shape == (B, 33, P, 4)
    assert torch.allclose(G2, G.detach(), atol=1e-13)
    assert float((dG - auto).abs().max()) < 1e-11


def test_rowwise_jacobian_is_the_per_row_derivative():
    torch.manual_seed(1)
    u = torch.randn(4, 3, dtype=DD, requires_grad=True)
    M = torch.randn(5, 3, dtype=DD)
    y = torch.tanh(u @ M.T)                                 # rows independent
    J = rowwise_jacobian(y, u)
    want = (1 - y.detach() ** 2).unsqueeze(-1) * M          # (4, 5, 3)
    assert torch.allclose(J, want, atol=1e-12)


def _exp_problem():
    t = torch.linspace(0, 2, 30, dtype=DD)
    true = torch.tensor([[2.0, 1.5], [0.7, 0.3], [1.2, 0.9]], dtype=DD)
    y = true[:, :1] * torch.exp(-true[:, 1:] * t)

    def fun(u, jac):
        e = torch.exp(-u[:, 1:] * t)
        J = torch.stack([e, -u[:, :1] * t * e], -1) if jac else None
        return u[:, :1] * e - y, J
    return fun, true


def test_lm_solves_each_row_and_reports_why_it_stopped():
    fun, true = _exp_problem()
    u, F, info = lm_solve(fun, torch.ones(3, 2, dtype=DD), max_iter=60)
    assert torch.allclose(u, true, atol=1e-7)
    assert float(F.max()) < 1e-16
    assert (info.status == STATIONARY).all() and info.finished.all()
    # every row's objective is monotone: a step is kept only if it lowers it
    H = torch.stack(info.history)
    assert bool((H[1:] <= H[:-1] + 1e-18).all())


def test_lm_budget_stop_and_frozen_coordinates():
    fun, true = _exp_problem()
    u0 = torch.ones(3, 2, dtype=DD)
    _, _, info = lm_solve(fun, u0, max_iter=2)
    assert (info.status == BUDGET).all() and not info.finished.any()   # ran out: not a result
    # a stop criterion ends a row as soon as it holds
    u, _, info = lm_solve(fun, u0, max_iter=60,
                          stop=lambda u, r: (r * r).sum(-1) < 1e-3)
    assert (info.status == STOPPED).all()
    assert float(((fun(u, False)[0]) ** 2).sum(-1).max()) < 1e-3
    # coordinates masked out of `live` do not move
    live = torch.tensor([[1.0, 0.0]] * 3, dtype=DD)
    u, _, _ = lm_solve(fun, u0, max_iter=30, live=live)
    assert torch.equal(u[:, 1], u0[:, 1]) and not torch.allclose(u[:, 0], u0[:, 0])
