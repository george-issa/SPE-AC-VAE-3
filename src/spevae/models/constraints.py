r"""The constraint layer: raw network outputs :math:`\rightarrow` a physical pole set.

Everything the physics demands is imposed here, *structurally*, on the last line
of the decoder.  Nothing below is a penalty, a regulariser, or something the
network is trained to approximate: each property holds identically for every
parameter value, at initialisation, mid-training, and out of distribution.

======================================  ==========================================
property                                 how it is secured
======================================  ==========================================
causality, :math:`\gamma_p > 0`          ``gamma = gamma_min + softplus(u)``
positive weights, :math:`a_p > 0`        ``a = softmax(u)``
sum rule, :math:`\sum_p a_p = 1`         ``softmax`` normalisation
zero-sum tail, :math:`\sum_p b_p = 0`    ``b = a * (t - <t>_a)``; identically 0 by default
:math:`A(\omega) \ge 0` pointwise        ``b = 0`` (positive Lorentzian mixture)
:math:`A \sim 1/\omega^2`                follows from :math:`\sum_p b_p = 0`
:math:`G \equiv 0 \Rightarrow A \equiv 0`  residues scaled by the measured norm
:math:`A[cG] = c\,A[G]`                  encoder sees ``G/N``, decoder rescales by ``N``
width floor :math:`\gamma_p \ge \gamma_{\min}`  additive floor inside the softplus
======================================  ==========================================

Why ``softmax`` rather than ``|a| / sum|a|``
--------------------------------------------
The normalised-absolute-value construction has a kink at :math:`a_p = 0` whose
subgradient is zero, so a pole that reaches zero weight is *trapped*: it can
never be revived.  ``softmax`` is smooth everywhere, keeps every weight strictly
positive, and reaches the same simplex.  Unused poles decay towards -- but never
reach -- zero weight, so their gradients stay alive.

Why the width floor is a constant in dimensionless units
--------------------------------------------------------
The finest structure recoverable from data on :math:`\tau \in [0,\beta]` is set
by the first Matsubara frequency :math:`\omega_0 = \pi/\beta`, i.e.
:math:`\tilde\nu_0 = \pi` after the dimensionless reduction.  A pole with
:math:`\tilde\gamma \ll \pi` is narrower than anything the data can resolve, and
sharpening it further only fits noise while driving
:math:`\max_\omega A_p = a_p/\pi\gamma_p` to infinity.  The floor
:math:`\tilde\gamma_{\min} = 0.1\pi` is therefore a *physical* statement about
resolution, and it is the same number at every :math:`\beta` -- which is only
true because the network works in dimensionless units.

Why ``eps`` is parameterised through ``sinh``
---------------------------------------------
Dimensionless pole positions span :math:`|\tilde\varepsilon| \lesssim \beta W`,
which is a few hundred at large :math:`\beta`, while the resolution scale stays
:math:`O(\pi)`.  A plain linear or ``tanh`` head would need outputs accurate to
:math:`10^{-3}` of its range to place a low-energy feature.  ``sinh`` gives
near-linear sensitivity around :math:`\tilde\varepsilon = 0` and exponential
reach into the band edges, so both regimes are learnable from :math:`O(1)`
activations.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..physics.kernel import MATSUBARA_NU0
from ..physics.poles import PoleSet

__all__ = ["PoleConstraints", "DEFAULT_GAMMA_MIN"]

#: Width floor in dimensionless units: one tenth of the first Matsubara frequency.
DEFAULT_GAMMA_MIN: float = 0.1 * MATSUBARA_NU0


class PoleConstraints(nn.Module):
    r"""Map raw decoder heads to a constraint-satisfying :class:`PoleSet`.

    Parameters
    ----------
    residue_mode
        ``"real"`` (default) sets :math:`b \equiv 0`, giving a positive mixture
        of Lorentzians: :math:`A(\omega) \ge 0` pointwise by construction.
        ``"complex"`` frees the dispersive part subject to
        :math:`\sum_p b_p = 0` and :math:`|b_p| \le a_p`, which buys asymmetric
        line shapes at the cost of pointwise positivity -- appropriate for
        correlators that are genuinely not positive-definite, and *not*
        recommended for a fermionic single-particle spectral function.
    gamma_min
        Width floor, dimensionless.  Must exceed the quadrature's
        ``min_resolvable_gamma``.
    gamma_scale
        Sets the typical width at zero activation:
        :math:`\gamma(0) = \gamma_{\min} + \gamma_{\rm scale}\ln 2`.
    nu_max
        Soft bound on :math:`|\tilde\varepsilon_p|`.  Should sit inside the
        quadrature's core range (``0.8 * nu_c`` is the package default) so that
        no pole lands in the reciprocal tail panels.
    eps_scale
        Slope of the ``sinh`` position map at the origin, in units of
        :math:`\tilde\nu`.  The default :math:`\pi` makes one unit of activation
        worth roughly one Matsubara frequency.
    """

    def __init__(
        self,
        residue_mode: str = "real",
        gamma_min: float = DEFAULT_GAMMA_MIN,
        gamma_scale: float = MATSUBARA_NU0,
        nu_max: float = 240.0,
        eps_scale: float = MATSUBARA_NU0,
    ) -> None:
        super().__init__()
        if residue_mode not in ("real", "complex"):
            raise ValueError(
                f"residue_mode={residue_mode!r}; expected 'real' or 'complex'."
            )
        self.residue_mode = residue_mode
        self.gamma_min = float(gamma_min)
        self.gamma_scale = float(gamma_scale)
        self.nu_max = float(nu_max)
        self.eps_scale = float(eps_scale)

    def forward(
        self,
        u_eps: torch.Tensor,
        u_gamma: torch.Tensor,
        u_a: torch.Tensor,
        u_b: torch.Tensor | None = None,
        norm: torch.Tensor | None = None,
        nu_max: float | None = None,
    ) -> PoleSet:
        r"""Apply every constraint.

        Parameters
        ----------
        u_eps, u_gamma, u_a, u_b
            ``(B, P)`` raw head outputs.  ``u_b`` is ignored in ``"real"`` mode.
        norm
            ``(B,)`` spectral norm :math:`N = \int A\,d\omega`, read from the
            data as :math:`G(0)+G(\beta)`.  Residues are scaled by it, which is
            what makes the whole map exactly 1-homogeneous and sends
            :math:`G \equiv 0` to :math:`A \equiv 0`.  Defaults to 1.
        nu_max
            Per-call override of the position bound.  The physically meaningful
            bound is a *frequency* one, :math:`|\omega| \le \omega_{\max}`, which
            in dimensionless units is :math:`\beta\omega_{\max}` and therefore
            depends on the cell being continued; the caller supplies it.

        Returns
        -------
        PoleSet
            Dimensionless poles with ``sum(a) == norm`` and ``sum(b) == 0``.
        """
        nu_max = self.nu_max if nu_max is None else float(nu_max)

        # Position: sinh reach, bounded by clamping the *pre-activation* at
        # asinh(nu_max / eps_scale).  Clamping there rather than squashing the
        # output with a tanh keeps the map exactly linear-in-sinh everywhere
        # inside the bound, so a given activation means the same dimensionless
        # pole at every beta -- squashing would make the interior of the band
        # depend on where the (beta-dependent) bound happens to sit.
        u_max = min(math.asinh(nu_max / self.eps_scale), 8.0)
        eps = self.eps_scale * torch.sinh(torch.clamp(u_eps, -u_max, u_max))

        # Width: strictly positive and floored at the resolution limit.  Left
        # unbounded above on purpose -- a very broad pole is a featureless
        # background, which the quadrature's reciprocal tail panels integrate
        # exactly, and any upper cap would have to be beta-dependent and would
        # therefore break the exact beta-universality of the decoder.
        gamma = self.gamma_min + self.gamma_scale * F.softplus(u_gamma)

        # Weights: strictly positive, exactly on the unit simplex.
        a = torch.softmax(u_a, dim=-1)

        if self.residue_mode == "real" or u_b is None:
            b = torch.zeros_like(a)
        else:
            # b_p = a_p (t_p - sum_q a_q t_q) with |t| <= 1/2.
            #   sum_p b_p = sum_p a_p t_p - (sum_q a_q t_q) sum_p a_p = 0   [sum a = 1]
            #   a_p -> 0  =>  b_p -> 0, so a weightless pole is structurally inert
            #   |t_p - <t>_a| <= 1  =>  |b_p| <= a_p, bounding the dispersive dip
            t = 0.5 * torch.tanh(u_b)
            b = a * (t - (a * t).sum(dim=-1, keepdim=True))

        if norm is not None:
            scale = norm.reshape(-1, 1).to(a.dtype)
            a = a * scale
            b = b * scale

        return PoleSet(eps=eps, gamma=gamma, a=a, b=b)

    def extra_repr(self) -> str:
        return (f"residue_mode={self.residue_mode!r}, gamma_min={self.gamma_min:.4g}, "
                f"gamma_scale={self.gamma_scale:.4g}, nu_max={self.nu_max:g}, "
                f"eps_scale={self.eps_scale:.4g}")


def spectral_norm_from_greens(
    G: torch.Tensor,
    includes_endpoints: bool = True,
    fallback: float = 1.0,
) -> torch.Tensor:
    r"""Read :math:`N = \int A\,d\omega` off the data as :math:`G(0) + G(\beta)`.

    This is exact for the fermionic anticommutator sum rule and requires no
    fitting.  It is the single quantity that carries the overall scale, which is
    why removing it from the encoder's input (and re-applying it in the decoder)
    makes the estimator exactly 1-homogeneous:

    .. math:: A[cG] = c\,A[G] \quad\text{for all } c,
              \qquad\text{in particular } A[0] = 0 .

    Parameters
    ----------
    G
        ``(B, L)`` Green's function on an ascending grid.
    includes_endpoints
        Whether ``G[:, 0]`` is :math:`\tau=0` and ``G[:, -1]`` is
        :math:`\tau=\beta`.  If ``False`` the sum rule is not directly readable
        and ``fallback`` is used (1 for a fermionic single-particle
        :math:`G`); the homogeneity property is then lost, so prefer supplying
        endpoint-inclusive data.
    """
    if includes_endpoints:
        return G[:, 0] + G[:, -1]
    return torch.full((G.shape[0],), float(fallback), dtype=G.dtype, device=G.device)
