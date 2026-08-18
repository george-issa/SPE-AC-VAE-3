r"""The fermionic imaginary-time kernel, in dimensionless form.

Conventions
-----------
Throughout this package the imaginary-time Green's function is the
*positive* (SmoQy / DQMC) convention

.. math::
    G(\tau) \;=\; \big\langle \mathcal{T}\, c(\tau)\, c^\dagger(0) \big\rangle
    \;=\; \int_{-\infty}^{\infty}\! d\omega \; A(\omega)\,
          \frac{e^{-\tau\omega}}{1 + e^{-\beta\omega}},
    \qquad 0 \le \tau \le \beta ,

with :math:`A(\omega) \ge 0` and :math:`\int A(\omega)\,d\omega = 1`.  The two
endpoints then carry the fermionic anticommutator,

.. math::
    G(0) = 1 - n/2, \qquad G(\beta) = n/2,
    \qquad G(0) + G(\beta) = \int A\,d\omega = 1 ,

which is the sum rule this package uses to fix the overall spectral norm.

The dimensionless reduction
---------------------------
Substituting :math:`x = \tau/\beta \in [0,1]` and :math:`\nu = \beta\omega`,

.. math::
    G(x) \;=\; \int d\nu \; \tilde A(\nu)\, K(x, \nu),
    \qquad K(x,\nu) = \frac{e^{-x\nu}}{1+e^{-\nu}}
                    = \frac{1}{e^{x\nu} + e^{(x-1)\nu}},

with :math:`\tilde A(\nu) = A(\nu/\beta)/\beta` and
:math:`\int \tilde A\,d\nu = \int A\,d\omega`.

**β has disappeared.**  The inverse problem "given :math:`G` on a grid in
:math:`x\in[0,1]`, find :math:`\tilde A`" is one single problem, identical for
every inverse temperature.  β and Δτ re-enter only through (i) *where* on
:math:`[0,1]` the data is sampled and (ii) the noise level.  This is why a
single trained network in this package covers all (β, Δτ) rather than needing
one model per simulation cell, and why the pole-width floor
:math:`\tilde\gamma_{\min} \sim \pi` (the first dimensionless Matsubara
frequency) is a β-independent constant.

Numerics
--------
:func:`fermi_kernel` is evaluated as ``exp(-logaddexp(x*nu, (x-1)*nu))``, which
is exact and overflow-free for every real ``(x, nu)`` with ``0 <= x <= 1``:
both branches underflow gracefully to 0 instead of producing ``inf/inf``.  No
argument clamping is used anywhere (clamping the exponent, as an earlier
implementation did at :math:`\pm 50`, silently biases broad poles at large β).

The identity that makes the sum rule exact at the discrete level is

.. math::  K(0,\nu) + K(1,\nu) = \sigma(\nu) + \sigma(-\nu) = 1

for every :math:`\nu`, where :math:`\sigma` is the logistic function.  Any
quadrature rule whose weights sum to the measure of the domain therefore
reproduces :math:`G(0) + G(\beta) = \int A` *exactly*, to machine precision --
see :mod:`spevae.physics.forward`.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "fermi_kernel",
    "matsubara_nu",
    "MATSUBARA_NU0",
    "to_dimensionless",
    "from_dimensionless",
]

#: First fermionic Matsubara frequency in dimensionless units, ``beta * (pi/beta)``.
#: The natural resolution scale of the problem: a spectral feature narrower than
#: this is not resolvable from data on ``tau in [0, beta]``.
MATSUBARA_NU0: float = math.pi


def fermi_kernel(x: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    r"""Dimensionless fermionic kernel :math:`K(x,\nu) = 1/(e^{x\nu}+e^{(x-1)\nu})`.

    Parameters
    ----------
    x
        Dimensionless imaginary time :math:`\tau/\beta`, expected in ``[0, 1]``.
    nu
        Dimensionless frequency :math:`\beta\omega`.  Broadcast against ``x``.

    Returns
    -------
    torch.Tensor
        ``K`` broadcast to ``torch.broadcast_shapes(x.shape, nu.shape)``.

    Notes
    -----
    Special values, all exact under this implementation:
    ``K(0, nu) = sigmoid(nu)``, ``K(1, nu) = sigmoid(-nu)``,
    ``K(x, 0) = 1/2``, and ``K(0, nu) + K(1, nu) = 1``.
    """
    a = x * nu
    b = (x - 1.0) * nu
    return torch.exp(-torch.logaddexp(a, b))


def matsubara_nu(n: torch.Tensor | int) -> torch.Tensor:
    r"""Dimensionless fermionic Matsubara frequencies :math:`\nu_n = (2n+1)\pi`.

    These are :math:`\beta\omega_n` for :math:`\omega_n = (2n+1)\pi/\beta`, and
    are the poles of :func:`fermi_kernel` in the complex ``nu`` plane (at
    :math:`\nu = \pm i(2n+1)\pi`).  A pole of the spectral representation with
    dimensionless width :math:`\tilde\gamma \ll \pi` lies far inside the first
    Matsubara spacing and is therefore not resolvable from the data.
    """
    if not torch.is_tensor(n):
        n = torch.as_tensor(n)
    return (2.0 * n + 1.0) * math.pi


def to_dimensionless(
    tau: torch.Tensor,
    beta: float | torch.Tensor,
) -> torch.Tensor:
    """Map an imaginary-time grid to ``x = tau / beta``."""
    return tau / beta


def from_dimensionless(
    nu: torch.Tensor,
    beta: float | torch.Tensor,
) -> torch.Tensor:
    """Map a dimensionless frequency (or pole coordinate) back to ``omega = nu / beta``."""
    return nu / beta
