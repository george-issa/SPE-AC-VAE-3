r"""The simple-pole expansion (SPE) and its exact algebra.

Representation
--------------
A spectral function is represented by :math:`P` complex poles
:math:`z_p = \varepsilon_p - i\gamma_p` (:math:`\gamma_p > 0`) and complex
residues :math:`r_p = a_p + i b_p`:

.. math::
    g(z) \;=\; \sum_{p=1}^{P} \frac{r_p}{z - z_p},
    \qquad
    A(\omega) \;=\; -\frac{1}{\pi}\,\mathrm{Im}\, g(\omega + i0^+)
    \;=\; \frac{1}{\pi} \sum_p
          \frac{a_p \gamma_p - b_p(\omega - \varepsilon_p)}
               {(\omega - \varepsilon_p)^2 + \gamma_p^2} .

There is **no frequency grid**: :math:`A` is a closed-form function of
:math:`\omega`, evaluable at any point and to any resolution after training.

Why the four constraints are exactly the physics
------------------------------------------------
Expanding :math:`g(z) = \sum_k M_k / z^{k+1}` at large :math:`|z|` gives
:math:`M_0 = \sum_p r_p`.  A fermionic single-particle Green's function has
:math:`g(z) \to 1/z`, so

* :math:`\sum_p a_p = 1` — the spectral sum rule :math:`\int A\,d\omega = 1`;
* :math:`\sum_p b_p = 0` — kills the :math:`1/\omega` term of :math:`A`, leaving
  the physical :math:`A(\omega) \simeq c_2/\omega^2` tail with
  :math:`c_2 = \frac{1}{\pi}\sum_p (a_p\gamma_p - b_p\varepsilon_p)`;
* :math:`\gamma_p > 0` — every pole in the lower half plane, so
  :math:`g` is analytic in the upper half plane: **causality**;
* :math:`a_p \ge 0` — each pole carries non-negative spectral weight, so
  :math:`a_p` reads directly as the fraction of the spectrum in feature
  :math:`p`.

Positivity
----------
With ``b = 0`` the expansion is a *positive mixture of Lorentzians*,

.. math:: A(\omega) = \sum_p a_p \, \frac{\gamma_p/\pi}{(\omega-\varepsilon_p)^2 + \gamma_p^2},

which is non-negative **pointwise, by construction**, normalised, causal, and
:math:`\sim 1/\omega^2` at large :math:`|\omega|` — all four simultaneously and
without any training pressure.  It is the discrete analogue of the observation
that respecting the Nevanlinna analytic structure yields intrinsically positive
and normalised spectra (Fei, Yeh & Gull, *PRL* **126**, 056402 (2021)).

With :math:`b_p \neq 0` — the package default, because physical line shapes are
asymmetric — each pole acquires a dispersive (Fano-like) component and the
*individual* :math:`A_p` is no longer sign-definite:

.. math::
    \min_\omega A_p = \frac{a_p - \sqrt{a_p^2+b_p^2}}{2\pi\gamma_p} < 0
    \quad\text{whenever } b_p \neq 0 ,

with magnitude diverging as :math:`\gamma_p \to 0`.  Positivity of the total
then relies on cross-pole cancellation, which :math:`\sum_p b_p = 0` enforces
in the *integral* while the :math:`|b_p| \le a_p` bound and the training-time
penalty keep the pointwise excursions small.  Whether a given pole set is in
fact non-negative everywhere is *decidable* without any grid — see
:mod:`spevae.physics.positivity` for the exact certificate — and the verdict
ships with every reported spectrum.  ``residue_mode="real"`` remains available
when a structural guarantee matters more than asymmetry.
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import torch

__all__ = [
    "PoleSet",
    "spectral_from_poles",
    "tail_coefficients",
    "windowed_moments",
    "effective_pole_count",
    "per_pole_extrema",
    "spectral_norm",
]


@dataclass
class PoleSet:
    r"""A batch of pole expansions.

    All tensors have shape ``(B, P)``.  ``eps`` and ``gamma`` are in whatever
    frequency units the caller is working in -- dimensionless
    :math:`\tilde\nu = \beta\omega` inside the network, physical :math:`\omega`
    after :meth:`to_physical`.  ``a`` and ``b`` are the real and imaginary parts
    of the residues and are *unit-free* (they carry spectral weight, which is
    dimensionless).
    """

    eps: torch.Tensor
    gamma: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor

    def __post_init__(self) -> None:
        shapes = {t.shape for t in (self.eps, self.gamma, self.a, self.b)}
        if len(shapes) != 1:
            raise ValueError(f"PoleSet components must share a shape, got {shapes}")

    # -- unit conversion ---------------------------------------------------- #

    def to_physical(self, beta: float | torch.Tensor) -> "PoleSet":
        r"""Convert dimensionless poles (:math:`\beta\omega`) to physical :math:`\omega`.

        Residues are unchanged: spectral weight is dimensionless, so
        :math:`\sum a_p = 1` holds in both unit systems.  Widths and positions
        scale as :math:`1/\beta`.
        """
        if torch.is_tensor(beta) and beta.ndim == 1:
            beta = beta.unsqueeze(-1)
        return PoleSet(self.eps / beta, self.gamma / beta, self.a, self.b)

    def to_dimensionless(self, beta: float | torch.Tensor) -> "PoleSet":
        """Inverse of :meth:`to_physical`."""
        if torch.is_tensor(beta) and beta.ndim == 1:
            beta = beta.unsqueeze(-1)
        return PoleSet(self.eps * beta, self.gamma * beta, self.a, self.b)

    # -- derived quantities ------------------------------------------------- #

    @property
    def poles(self) -> torch.Tensor:
        r""":math:`z_p = \varepsilon_p - i\gamma_p` as a complex tensor."""
        return torch.complex(self.eps, -self.gamma)

    @property
    def residues(self) -> torch.Tensor:
        r""":math:`r_p = a_p + i b_p` as a complex tensor."""
        return torch.complex(self.a, self.b)

    def spectral(self, omega: torch.Tensor) -> torch.Tensor:
        """:math:`A(\\omega)` on an arbitrary grid; see :func:`spectral_from_poles`."""
        return spectral_from_poles(self.eps, self.gamma, self.a, self.b, omega)

    def detach(self) -> "PoleSet":
        return PoleSet(self.eps.detach(), self.gamma.detach(),
                       self.a.detach(), self.b.detach())

    def to(self, *args, **kwargs) -> "PoleSet":
        return PoleSet(self.eps.to(*args, **kwargs), self.gamma.to(*args, **kwargs),
                       self.a.to(*args, **kwargs), self.b.to(*args, **kwargs))

    def __getitem__(self, idx) -> "PoleSet":
        return PoleSet(self.eps[idx], self.gamma[idx], self.a[idx], self.b[idx])

    @property
    def shape(self) -> torch.Size:
        return self.eps.shape


def spectral_from_poles(
    eps: torch.Tensor,
    gamma: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate :math:`A(\omega)` from a pole expansion.

    Parameters
    ----------
    eps, gamma, a, b
        ``(B, P)`` pole positions, half-widths (> 0), and residue real/imaginary
        parts.
    omega
        ``(M,)`` frequency grid, or ``(B, M)`` for a per-sample grid.

    Returns
    -------
    torch.Tensor
        ``(B, M)`` spectral function.

    Notes
    -----
    Evaluated in the real form
    :math:`\frac{1}{\pi}\sum_p [a_p\gamma_p - b_p(\omega-\varepsilon_p)] /
    [(\omega-\varepsilon_p)^2 + \gamma_p^2]` rather than via complex division,
    which avoids a complex intermediate and keeps the ``b = 0`` path exactly
    non-negative in floating point.
    """
    if omega.ndim == 1:
        omega = omega.unsqueeze(0)          # (1, M)
    w = omega.unsqueeze(1)                  # (B|1, 1, M)
    d = w - eps.unsqueeze(-1)               # (B, P, M)
    den = d * d + (gamma * gamma).unsqueeze(-1)
    num = (a * gamma).unsqueeze(-1) - b.unsqueeze(-1) * d
    return (num / den).sum(dim=1) / math.pi


def spectral_norm(a: torch.Tensor) -> torch.Tensor:
    r""":math:`\int A(\omega)\,d\omega = \sum_p a_p`, exactly (each Lorentzian integrates to 1)."""
    return a.sum(dim=-1)


def tail_coefficients(
    eps: torch.Tensor,
    gamma: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Leading large-:math:`|\omega|` tail coefficients of :math:`A`.

    Returns ``(c1, c2)`` in :math:`A(\omega) \simeq c_1/\omega + c_2/\omega^2`:

    .. math::
        c_1 = -\frac{1}{\pi}\sum_p b_p, \qquad
        c_2 = \frac{1}{\pi}\sum_p \big(a_p\gamma_p - b_p\varepsilon_p\big).

    A physical spectral function has :math:`c_1 = 0`; the decoder enforces this
    by construction, so this function is a *diagnostic*, not a penalty.
    """
    c1 = -b.sum(dim=-1) / math.pi
    c2 = (a * gamma - b * eps).sum(dim=-1) / math.pi
    return c1, c2


def windowed_moments(
    eps: torch.Tensor,
    gamma: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    omega: torch.Tensor,
    order: int = 2,
) -> torch.Tensor:
    r"""Spectral moments :math:`\int_{-W}^{W} \omega^k A(\omega)\,d\omega`.

    **Windowed on purpose.**  A simple pole expansion has a :math:`1/\omega^2`
    tail, so :math:`M_k` for :math:`k \ge 1` diverges over the whole real line
    (logarithmically for :math:`k=1`).  Only :math:`M_0` is grid-free; every
    higher moment is a window-dependent quantity and is reported as such.

    ``omega`` must be a uniform ascending grid; integration is by the trapezoid
    rule on that grid.
    """
    A = spectral_from_poles(eps, gamma, a, b, omega)     # (B, M)
    w = omega if omega.ndim == 1 else omega[0]
    out = []
    for k in range(order + 1):
        integrand = A * (w**k).unsqueeze(0)
        out.append(torch.trapezoid(integrand, w, dim=-1))
    return torch.stack(out, dim=-1)                      # (B, order+1)


def effective_pole_count(a: torch.Tensor) -> torch.Tensor:
    r"""Participation ratio :math:`1 / \sum_p a_p^2` -- a threshold-free pole count.

    Equals :math:`N` when :math:`N` poles share the weight equally and 1 when a
    single pole carries all of it.  Unlike a magnitude threshold it cannot be
    inflated by a dusting of near-zero weights, which makes it the right
    quantity for asking whether the model spends only the poles the data
    supports.  Assumes :math:`\sum_p a_p = 1`.
    """
    return 1.0 / (a.pow(2).sum(dim=-1) + 1e-12)


def per_pole_extrema(
    gamma: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Closed-form :math:`\max_\omega A_p` and :math:`\min_\omega A_p` per pole.

    .. math::
        \max_\omega A_p = \frac{a_p + \sqrt{a_p^2+b_p^2}}{2\pi\gamma_p},
        \qquad
        \min_\omega A_p = \frac{a_p - \sqrt{a_p^2+b_p^2}}{2\pi\gamma_p} .

    Both scale as :math:`1/\gamma_p`, which is the quantitative statement of why
    a pole collapsing onto the real axis blows up the spectrum.  The ``min`` is
    identically zero when ``b = 0`` and strictly negative otherwise.
    """
    rad = torch.sqrt(a * a + b * b)
    scale = 1.0 / (2.0 * math.pi * gamma)
    return (a + rad) * scale, (a - rad) * scale
