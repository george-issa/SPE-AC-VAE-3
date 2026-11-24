r"""Batched Levenberg--Marquardt for the pipeline's least-squares fits.

Why this exists
---------------
Every per-dataset fit in this package -- the latent descent, the refinement of
posterior samples -- minimises

.. math:: F(u) = \tfrac12 \lVert r(u) \rVert^2 ,

a whitened data residual stacked with a Gaussian anchor.  The Hessian of such
a problem is badly conditioned by construction: the kernel's singular values
fall exponentially, so at relative noise :math:`\sigma` the curvature spans
roughly :math:`\sigma^{-2}` between the best- and worst-determined directions.
A first-order method needs a step count that grows with that ratio.  Measured
2026-11-24: 3000 Adam steps leave the refinement at :math:`\chi^2 \approx 5`
per point at noise 3e-5, and the 10,000-step "manifold floor" descents were
all still descending when they hit the cap.  Gauss--Newton steps use the
curvature and converge in tens of iterations.

The solver is deliberately plain: Marquardt-scaled damping per row, a step is
kept only if it lowers that row's objective (so every row's objective is
monotone), and each row reports *why* it stopped -- a fit that ran out of
iterations is an intermediate answer, not a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

__all__ = ["lm_solve", "rowwise_jacobian", "LMInfo", "BUDGET", "STATIONARY", "STOPPED"]

#: Row statuses in :class:`LMInfo`.
BUDGET, STATIONARY, STOPPED = 0, 1, 2


@dataclass
class LMInfo:
    """How each row of a :func:`lm_solve` call ended."""

    iterations: torch.Tensor        # (N,) iterations spent while the row was active
    status: torch.Tensor            # (N,) BUDGET / STATIONARY / STOPPED
    history: list                   # per iteration: (N,) objective values

    @property
    def stationary(self) -> torch.Tensor:
        return self.status == STATIONARY

    @property
    def finished(self) -> torch.Tensor:
        """Rows that ended for a reason other than running out of iterations."""
        return self.status != BUDGET


def rowwise_jacobian(y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    r"""``dy[i] / du[i]`` for a row-independent map, as ``(N, q, n)``.

    ``y`` is ``(N, q)`` computed from ``u`` (``(N, ...)``, requiring grad) with
    no coupling between rows -- true of a decoder or a coordinate map applied
    per sample.  One vectorised backward pass when every operation supports
    it, a loop over the ``q`` outputs otherwise.
    """
    N, q = y.shape
    try:
        eye = torch.eye(q, dtype=y.dtype, device=y.device).unsqueeze(1).expand(q, N, q)
        (g,) = torch.autograd.grad(y, u, grad_outputs=eye, is_grads_batched=True,
                                   retain_graph=True)
        return g.reshape(q, N, -1).transpose(0, 1)
    except RuntimeError:
        rows = [torch.autograd.grad(y[:, j].sum(), u, retain_graph=True)[0].reshape(N, -1)
                for j in range(q)]
        return torch.stack(rows, 1)


def lm_solve(
    fun: Callable[[torch.Tensor, bool], tuple[torch.Tensor, torch.Tensor | None]],
    u0: torch.Tensor,
    max_iter: int = 60,
    live: torch.Tensor | None = None,
    stop: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    mu0: float = 1e-2,
    ftol: float = 1e-8,
    patience: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, LMInfo]:
    r"""Minimise :math:`\tfrac12\lVert r(u)\rVert^2` independently for each row.

    Parameters
    ----------
    fun
        ``fun(u, need_jacobian) -> (r, J)`` with ``u`` ``(N, n)``, ``r``
        ``(N, m)`` and ``J`` ``(N, m, n)`` (``None`` when not requested).
        Rows must be independent.
    u0
        ``(N, n)`` starting points.
    live
        ``(N, n)`` mask of coordinates allowed to move (default: all).
    stop
        Optional ``stop(u, r) -> (N,) bool``: rows for which the fit should
        end as soon as the criterion holds (e.g. the data misfit has reached
        the calibrated noise floor -- going further is fitting noise).
    ftol, patience
        A row is *stationary* after ``patience`` consecutive accepted steps
        each lowering its objective by less than ``ftol`` (relative), or after
        twelve consecutive rejected steps (the damping has grown by 4^12 and
        even a heavily damped step no longer lowers the objective).

    Returns
    -------
    (u, F, info)
        Final points, their objectives, and an :class:`LMInfo`.
    """
    u = u0.detach().clone()
    N, n = u.shape
    live = torch.ones_like(u) if live is None else live.to(u.dtype)
    dead = torch.diag_embed(1.0 - live)
    mu = torch.full((N,), float(mu0), dtype=u.dtype, device=u.device)
    status = torch.zeros(N, dtype=torch.long, device=u.device)
    iters = torch.zeros(N, dtype=torch.long, device=u.device)
    small = torch.zeros(N, dtype=torch.long, device=u.device)
    rejected = torch.zeros(N, dtype=torch.long, device=u.device)

    r, J = fun(u, True)
    F = 0.5 * (r * r).sum(-1)
    history = [F.clone()]
    if stop is not None:
        status[stop(u, r)] = STOPPED
    active = status == BUDGET
    for _ in range(int(max_iter)):
        if not bool(active.any()):
            break
        A = J.transpose(1, 2) @ J
        g = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1) * live
        scale = torch.diagonal(A, dim1=1, dim2=2).clamp_min(1e-12) * live
        step = torch.linalg.solve(A * live.unsqueeze(1) * live.unsqueeze(2)
                                  + mu.reshape(-1, 1, 1) * torch.diag_embed(scale) + dead,
                                  -g.unsqueeze(-1)).squeeze(-1)
        step = torch.where(active.unsqueeze(-1), step, torch.zeros_like(step))
        un = u + step
        r2, _ = fun(un, False)
        F2 = 0.5 * (r2 * r2).sum(-1)
        ok = active & torch.isfinite(F2) & (F2 < F)
        rel = (F - F2) / F.clamp_min(1e-300)
        u = torch.where(ok.unsqueeze(-1), un, u)
        small = torch.where(ok & (rel < ftol), small + 1,
                            torch.where(ok, torch.zeros_like(small), small))
        rejected = torch.where(ok, torch.zeros_like(rejected), rejected + active.long())
        mu = torch.where(ok, mu / 3.0, mu * 4.0).clamp(1e-6, 1e12)
        iters += active.long()
        status[active & ((small >= patience) | (rejected >= 12))] = STATIONARY
        r, J = fun(u, True)
        F = 0.5 * (r * r).sum(-1)
        history.append(F.clone())
        if stop is not None:
            status[(status == BUDGET) & stop(u, r)] = STOPPED
        active = status == BUDGET
    return u, F, LMInfo(iterations=iters, status=status, history=history)
