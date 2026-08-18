r"""Exact, grid-free pointwise-positivity certificate for a pole expansion.

With complex residues each pole contributes a Lorentzian *plus* a dispersive
part, so :math:`A(\omega) \ge 0` is a collective property of the whole set.  It
is also a *decidable* one: over the common (strictly positive) denominator,

.. math::
    A(\omega) \;=\; \frac{\mathcal{N}(\omega)}
                         {\pi \prod_p \big[(\omega-\varepsilon_p)^2 + \gamma_p^2\big]},
    \qquad
    \mathcal{N}(\omega) = \sum_p \big[a_p\gamma_p - b_p(\omega-\varepsilon_p)\big]
                          \prod_{q \neq p}\big[(\omega-\varepsilon_q)^2 + \gamma_q^2\big],

so :math:`A \ge 0` on all of :math:`\mathbb{R}` **iff** the polynomial
:math:`\mathcal{N}` (degree :math:`2P-2`, because the :math:`\omega^{2P-1}`
coefficient is :math:`-\sum_p b_p = 0` by construction) has no real root of odd
multiplicity.  Real roots come from the companion matrix -- a finite
computation on the pole parameters, with no :math:`\omega` grid anywhere.

Where :math:`A` does go negative, the excursion is quantified exactly:
the antiderivative of a pole expansion is elementary,

.. math::
    \int A \,d\omega \;=\; \sum_p \Big[\frac{a_p}{\pi}
        \arctan\frac{\omega-\varepsilon_p}{\gamma_p}
        - \frac{b_p}{2\pi}\,\ln\big((\omega-\varepsilon_p)^2+\gamma_p^2\big)\Big]
        + \text{const},

(the logs cancel at :math:`|\omega| \to \infty` because :math:`\sum_p b_p = 0`),
so the total negative weight :math:`\int \max(-A, 0)\,d\omega` is a finite sum
of antiderivative differences between consecutive real roots.

Everything here runs in float64 on the CPU: it is an *inference-time*
diagnostic and reporting tool, not a training loss.  For the training-time soft
penalty see :func:`spevae.losses.spectral_positivity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .poles import PoleSet

__all__ = ["PositivityReport", "positivity_certificate", "certify"]

#: Relative magnitude below which a leading polynomial coefficient is treated
#: as the exact zero that the sum rules make it.
_TRIM = 1e-12


@dataclass
class PositivityReport:
    """The verdict for one pole set, in the units the poles were given in."""

    is_positive: bool          #: A(w) >= -atol everywhere on the real line
    min_A: float               #: most negative value found (>= 0 if positive)
    argmin_omega: float | None  #: where; None when no negative region exists
    negative_weight: float     #: exact integral of max(-A, 0) over the line
    n_real_roots: int          #: real zero crossings of A
    atol: float                #: tolerance the verdict was issued at

    def __str__(self) -> str:
        if self.is_positive:
            return f"certified A >= 0 (atol={self.atol:.1e})"
        return (f"A < 0: min_A={self.min_A:.3e} at omega={self.argmin_omega:.4g}, "
                f"negative weight {self.negative_weight:.3e}")


def _numerator_coefficients(eps, gamma, a, b) -> np.ndarray:
    """Ascending coefficients of the numerator polynomial ``N`` of ``pi * A``.

    Built with prefix/suffix convolutions of the per-pole quadratics so each
    ``prod_{q != p}`` costs one convolution instead of ``P - 1``.
    """
    P = eps.shape[0]
    quads = [np.array([e * e + g * g, -2.0 * e, 1.0]) for e, g in zip(eps, gamma)]
    prefix = [np.array([1.0])]
    for q in quads:
        prefix.append(np.convolve(prefix[-1], q))
    suffix = [np.array([1.0])]
    for q in reversed(quads):
        suffix.append(np.convolve(suffix[-1], q))
    suffix.reverse()

    N = np.zeros(2 * P, dtype=np.float64)
    for p in range(P):
        # linear factor a_p gamma_p - b_p (w - eps_p), ascending
        lin = np.array([a[p] * gamma[p] + b[p] * eps[p], -b[p]])
        term = np.convolve(np.convolve(prefix[p], lin), suffix[p + 1])
        N[: term.shape[0]] += term
    return N


def _antiderivative(omega: float, eps, gamma, a, b) -> float:
    """Elementary antiderivative of ``A`` at a finite ``omega``."""
    d = omega - eps
    return float(
        (a / math.pi * np.arctan(d / gamma)
         - b / (2.0 * math.pi) * np.log(d * d + gamma * gamma)).sum()
    )


def _spectral(omega, eps, gamma, a, b):
    d = np.subtract.outer(np.atleast_1d(omega), eps)          # (M, P)
    return ((a * gamma - b * d) / (d * d + gamma * gamma)).sum(-1) / math.pi


def certify(
    eps: np.ndarray,
    gamma: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    atol: float = 1e-10,
) -> PositivityReport:
    r"""Certify pointwise positivity for one pole set (1-D arrays of length P).

    The verdict is scale-covariant in the right way: rescaling frequencies
    (physical vs dimensionless units) rescales ``min_A`` by the same factor and
    leaves ``is_positive`` and ``negative_weight`` unchanged -- the latter
    because :math:`\int \mathrm{ReLU}(-A)\,d\omega` is a pure spectral weight.

    ``atol`` is an *absolute* tolerance on :math:`A` in the given units; the
    default is far below any physically meaningful spectral density.
    """
    eps = np.asarray(eps, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if eps.ndim != 1:
        raise ValueError("certify() takes one pole set; use positivity_certificate "
                         "for a batched PoleSet")
    if not (gamma > 0).all():
        raise ValueError("certify() requires gamma > 0 (poles off the real axis)")

    # Condition the polynomial: an affine frequency map w = c + s*u leaves the
    # sign pattern of A untouched and keeps the quadratics' coefficients O(1)
    # even when |eps| is O(10^3) (large-beta dimensionless units).
    c = float(eps.mean())
    s = float(max(np.abs(eps - c).max(), gamma.max(), 1.0))
    eps_u, gamma_u = (eps - c) / s, gamma / s

    N = _numerator_coefficients(eps_u, gamma_u, a, b)
    scale = np.abs(N).max()
    if scale == 0.0:                     # A identically zero (all residues zero)
        return PositivityReport(True, 0.0, None, 0.0, 0, atol)
    N = N / scale
    # Trim the coefficients the constraints make exactly zero (sum b = 0 kills
    # the top one; floating point leaves ~1e-17 residue that np.roots would
    # otherwise turn into one enormous spurious root).
    top = N.shape[0]
    while top > 1 and abs(N[top - 1]) < _TRIM:
        top -= 1
    N = N[:top]

    if top <= 1:                          # constant numerator
        val = float(_spectral(c, eps, gamma, a, b)[0])
        ok = val >= -atol
        return PositivityReport(ok, min(val, 0.0) if not ok else val, None if ok else c,
                                0.0, 0, atol)

    roots = np.roots(N[::-1])             # descending order for np.roots
    real = np.sort(np.unique(np.real(
        roots[np.abs(roots.imag) <= 1e-9 * (1.0 + np.abs(roots.real))]
    )))

    if real.size == 0:
        # No real zero crossing: A has one sign everywhere; read it anywhere.
        val = float(_spectral(c, eps, gamma, a, b)[0])
        if val >= -atol:
            return PositivityReport(True, val, None, 0.0, 0, atol)
        # A < 0 everywhere cannot happen with sum a > 0, but report faithfully.
        return PositivityReport(False, val, c, float("inf"), 0, atol)

    # Sample each sign interval at its midpoint (outer intervals: one width out).
    real_w = c + s * real                                     # back to input units
    span = max(real_w[-1] - real_w[0], float(gamma.max()), 1e-6)
    probes = np.concatenate([
        [real_w[0] - span], 0.5 * (real_w[:-1] + real_w[1:]), [real_w[-1] + span],
    ])
    vals = _spectral(probes, eps, gamma, a, b)

    neg = vals < -atol
    if not neg.any():
        return PositivityReport(True, float(vals.min()), None, 0.0,
                                int(real_w.size), atol)

    # Exact negative weight: antiderivative differences over negative intervals.
    # Interval i is (edge[i-1], edge[i]) with edges = roots and +-infinity;
    # F(-inf) = -sum(a)/2, F(+inf) = +sum(a)/2 (the b-logs cancel, sum b = 0).
    F = np.array([_antiderivative(w, eps, gamma, a, b) for w in real_w])
    F_lo, F_hi = -a.sum() / 2.0, a.sum() / 2.0
    edges_F = np.concatenate([[F_lo], F, [F_hi]])
    neg_weight = float(np.maximum(-(edges_F[1:] - edges_F[:-1])[neg], 0.0).sum())

    # Locate the most negative point by bounded golden search inside each
    # negative interval (outer intervals get a generous finite bracket -- the
    # tail decays monotonically, so the interior minimum is what matters).
    from scipy.optimize import minimize_scalar
    lo_edges = np.concatenate([[real_w[0] - 50.0 * span], real_w])
    hi_edges = np.concatenate([real_w, [real_w[-1] + 50.0 * span]])
    min_A, argmin = float(vals.min()), float(probes[int(np.argmin(vals))])
    for i in np.flatnonzero(neg):
        r = minimize_scalar(lambda w: float(_spectral(w, eps, gamma, a, b)[0]),
                            bounds=(float(lo_edges[i]), float(hi_edges[i])),
                            method="bounded")
        if r.fun < min_A:
            min_A, argmin = float(r.fun), float(r.x)

    return PositivityReport(False, min_A, argmin, neg_weight,
                            int(real_w.size), atol)


def positivity_certificate(
    poles: PoleSet,
    atol: float = 1e-10,
) -> list[PositivityReport]:
    """Certify every pole set in a batched :class:`PoleSet`.

    Weightless poles are pruned first: the decoder's ``softmax`` keeps unused
    weights at ~1e-8 rather than 0, and a pole carrying no weight cannot change
    the verdict but doubles the polynomial degree for nothing.
    """
    eps = poles.eps.detach().double().cpu().numpy()
    gamma = poles.gamma.detach().double().cpu().numpy()
    a = poles.a.detach().double().cpu().numpy()
    b = poles.b.detach().double().cpu().numpy()
    if eps.ndim == 1:
        eps, gamma, a, b = (v[None] for v in (eps, gamma, a, b))

    out = []
    for i in range(eps.shape[0]):
        total = a[i].sum()
        keep = (a[i] > 1e-12 * max(total, 1e-300)) | (np.abs(b[i]) > 1e-15)
        if not keep.any():
            out.append(PositivityReport(True, 0.0, None, 0.0, 0, atol))
            continue
        out.append(certify(eps[i][keep], gamma[i][keep], a[i][keep], b[i][keep],
                           atol=atol))
    return out
