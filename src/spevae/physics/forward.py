r"""Exact forward maps: spectral function :math:`\rightarrow` :math:`G(\tau)`.

The forward map is the only place physics enters the reconstruction loss, so it
must be accurate to well below the DQMC noise floor (bin :math:`\sigma \sim
5\times10^{-4}`, bin-mean :math:`\sigma \sim 2.5\times10^{-5}`) *uniformly over
the pole configurations the decoder can emit* -- not just over the ones it
happens to emit early in training.

Why not the tangent substitution
--------------------------------
The obvious rule for a Lorentzian is :math:`\nu = \varepsilon + \gamma\tan(\pi
t/2)`, which flattens the pole's own measure and covers :math:`\mathbb{R}`
exactly.  It is however adapted to the *pole* and not to the *kernel*: the
Fermi factor turns over on a scale :math:`\Delta\nu \sim 1` at :math:`\nu = 0`,
which in :math:`t` coordinates is a boundary layer of width
:math:`2\gamma/\pi\varepsilon^2` sitting at :math:`|t| \to 1`.  Resolving it
needs

.. math:: n_{\rm nodes} \gtrsim \pi^{3/2}\,|\varepsilon|^{3/2}/\gamma^{1/2},

i.e. ~225 nodes at :math:`\beta\omega = 8` but ~4·10^4 at :math:`\beta\omega =
250`.  Measured convergence agrees with that scaling.  Any fixed node budget
therefore silently loses accuracy as :math:`\beta` or the bandwidth grows,
which is a poor property for a package meant to run at whatever :math:`\beta`
the user simulated.

The rule used here
------------------
:class:`SpectralQuadrature` is a *fixed, shared* rule for
:math:`\int_{-\infty}^{\infty} f(\nu)\,d\nu` built from two pieces:

1. **Core** -- the trapezoid rule on a uniform grid over
   :math:`[-\nu_c, \nu_c]` with spacing :math:`h`.  For an integrand analytic in
   a strip of half-width :math:`\gamma`, Poisson summation gives error
   :math:`\approx 2e^{-2\pi\gamma/h}`: *spectral* accuracy, uniform in where the
   feature sits.  At the package's width floor
   :math:`\tilde\gamma_{\min} = 0.1\pi` and :math:`h = 0.1` that is
   :math:`5\times10^{-9}`, independent of :math:`\varepsilon`.
2. **Tails** -- :math:`\nu = \pm\nu_c/u`, :math:`u \in (0,1]`, with
   Gauss--Legendre in :math:`u`.  The Jacobian :math:`\nu_c/u^2` exactly cancels
   the :math:`1/\nu^2` decay of a pole expansion, so the transformed integrand
   is bounded and smooth.  This is what makes the fat Lorentzian tails exact
   rather than truncated -- a uniform grid alone would need
   :math:`\nu_c \sim 10^6` to reach the same accuracy.

Because the nodes are shared across the batch and across pole configurations,
the whole forward map collapses to one matrix product

.. math:: G_{b\ell} \;=\; \sum_m \big(w_m \tilde A_{bm}\big)\, K(x_\ell, \nu_m),

with a :math:`(M, L)` kernel matrix that depends only on the time grid.  Memory
is :math:`O(BM + ML)` instead of the :math:`O(BPNL)` of a per-pole rule, and the
same rule integrates *any* spectral function -- pole expansions, Gaussian
mixtures, user-supplied curves -- so training data and model predictions are
never compared across two different quadratures.

Shared time grid
----------------
:class:`PoleGreens` takes one time grid per call.  β and Δτ are properties of a
simulation cell, so batching by cell is the natural data layout; mixing
:math:`\beta` *across* batches (not within one) gives the network the same
coverage at a fraction of the memory.  :meth:`PoleGreens.forward_grouped`
handles a ragged collection of grids at inference time.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import special
from torch import nn

from .kernel import MATSUBARA_NU0, fermi_kernel
from .poles import PoleSet, spectral_from_poles

__all__ = [
    "SpectralQuadrature",
    "PoleGreens",
    "gaussian_mixture_spectrum",
    "greens_matsubara",
    "matsubara_from_tau",
]


class SpectralQuadrature(nn.Module):
    r"""Fixed nodes and weights for :math:`\int_{-\infty}^{\infty} f(\nu)\,d\nu`.

    Parameters
    ----------
    nu_c
        Core half-range.  Poles are expected to satisfy
        :math:`|\varepsilon_p| + 10\gamma_p \lesssim \nu_c`; the decoder bounds
        them accordingly.  For physical bandwidth :math:`\omega_c` take
        ``nu_c = beta * omega_c``.
    h
        Core spacing.  Must be small compared with the narrowest feature:
        the error is :math:`2e^{-2\pi\gamma_{\min}/h}`, so
        ``h <= gamma_min / 3`` is a safe rule.
    n_tail
        Gauss--Legendre nodes per reciprocal tail panel.
    """

    def __init__(
        self,
        nu_c: float = 300.0,
        h: float = 0.09,
        n_tail: int = 32,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if nu_c <= 0 or h <= 0:
            raise ValueError("nu_c and h must be positive")

        n_core = int(round(2.0 * nu_c / h))
        core = np.linspace(-nu_c, nu_c, n_core + 1)
        h_eff = float(core[1] - core[0])
        w_core = np.full(n_core + 1, h_eff)
        w_core[0] = w_core[-1] = 0.5 * h_eff                 # trapezoid

        u, wu = special.roots_legendre(n_tail)
        u = 0.5 * (u + 1.0)                                  # (0, 1)
        wu = 0.5 * wu
        jac = nu_c / u**2                                    # |dnu/du|
        hi = nu_c / u
        w_hi = wu * jac
        lo = -hi
        w_lo = wu * jac

        nodes = np.concatenate([core, hi, lo])
        weights = np.concatenate([w_core, w_hi, w_lo])

        self.nu_c = float(nu_c)
        self.h = h_eff
        self.n_core = n_core + 1
        self.n_tail = int(n_tail)
        self.register_buffer("nodes", torch.as_tensor(nodes, dtype=dtype))
        self.register_buffer("weights", torch.as_tensor(weights, dtype=dtype))

    @classmethod
    def for_beta(
        cls,
        beta: float,
        omega_c: float = 20.0,
        gamma_min: float = 0.1 * MATSUBARA_NU0,
        nodes_per_width: float = 3.5,
        **kw,
    ) -> "SpectralQuadrature":
        r"""Rule sized for one simulation cell.

        ``nu_c = beta * omega_c`` covers the physical bandwidth; ``h`` is tied to
        the dimensionless width floor rather than to :math:`\beta`, because the
        floor *is* the resolution limit of the data (:math:`\tilde\gamma_{\min}`
        is a fixed fraction of the first Matsubara frequency :math:`\pi`).

        ``nodes_per_width = 3.5`` puts the Poisson-summation error at
        :math:`2e^{-7\pi} \approx 5\times10^{-10}`; measured worst-case error
        with every pole pinned at the floor is :math:`\sim 10^{-10}`, four to
        five orders of magnitude below the DQMC bin-mean noise.  Dropping to
        3.0 costs three orders of magnitude (:math:`\sim 5\times10^{-9}`), and
        2.0 is already at :math:`10^{-5}` -- the dependence is exponential, so
        this is not a knob to economise on casually.
        """
        return cls(nu_c=float(beta) * float(omega_c),
                   h=float(gamma_min) / float(nodes_per_width), **kw)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.numel())

    @property
    def min_resolvable_gamma(self) -> float:
        r"""Narrowest pole this rule integrates to :math:`\lesssim 10^{-9}`.

        From the Poisson-summation error :math:`2e^{-2\pi\gamma/h}`, this is
        :math:`\gamma = h \ln(2\times10^{9}) / 2\pi \approx 3.4\,h`.  The
        decoder's width floor must sit above it or the reconstruction loss will
        be reading quadrature error rather than physics.
        """
        return float(self.h) * math.log(2e9) / (2.0 * math.pi)

    def integrate(self, f_nodes: torch.Tensor) -> torch.Tensor:
        """Integrate values sampled at :attr:`nodes`; ``f_nodes`` is ``(..., M)``."""
        return (f_nodes * self.weights).sum(-1)

    def extra_repr(self) -> str:
        return (f"nu_c={self.nu_c:g}, h={self.h:g}, "
                f"n_core={self.n_core}, n_tail={self.n_tail}, M={self.n_nodes}")


class PoleGreens(nn.Module):
    r"""Pole expansion :math:`\rightarrow` :math:`G(x)` on a dimensionless time grid.

    Examples
    --------
    >>> quad = SpectralQuadrature(nu_c=60.0, h=0.1)
    >>> fwd = PoleGreens(quad)
    >>> poles = PoleSet(eps=torch.zeros(1, 1), gamma=torch.full((1, 1), 3.0),
    ...                 a=torch.ones(1, 1), b=torch.zeros(1, 1))
    >>> G = fwd(poles, torch.linspace(0, 1, 11))
    >>> bool(torch.isclose(G[0, 0] + G[0, -1], torch.tensor(1.0), atol=1e-5))
    True
    """

    def __init__(self, quad: SpectralQuadrature) -> None:
        super().__init__()
        self.quad = quad

    def kernel_matrix(self, x: torch.Tensor) -> torch.Tensor:
        r""":math:`K(x_\ell, \nu_m)` as an ``(M, L)`` matrix."""
        nu = self.quad.nodes.to(x.dtype)
        return fermi_kernel(x.reshape(1, -1), nu.reshape(-1, 1))

    def forward(
        self,
        poles: PoleSet,
        x: torch.Tensor,
        kernel_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Evaluate :math:`G(x)` for a batch of pole sets on one shared grid.

        Parameters
        ----------
        poles
            ``(B, P)`` pole set in **dimensionless** units (:math:`\beta\omega`).
        x
            ``(L,)`` dimensionless times :math:`\tau/\beta`, ascending, normally
            including both endpoints 0 and 1.
        kernel_matrix
            Optional precomputed :meth:`kernel_matrix` for ``x`` -- worth caching
            when the same grid is reused across many batches.

        Returns
        -------
        torch.Tensor
            ``(B, L)``.
        """
        x = x.reshape(-1).to(poles.eps.dtype)
        K = self.kernel_matrix(x) if kernel_matrix is None else kernel_matrix
        nodes = self.quad.nodes.to(poles.eps.dtype)
        A = spectral_from_poles(poles.eps, poles.gamma, poles.a, poles.b, nodes)
        return (A * self.quad.weights.to(A.dtype)) @ K

    def forward_grouped(
        self,
        poles: PoleSet,
        xs: list[torch.Tensor],
        index: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Ragged inference: ``index[b]`` selects which grid in ``xs`` sample ``b`` uses."""
        out: list[torch.Tensor | None] = [None] * poles.shape[0]
        for gi, x in enumerate(xs):
            sel = torch.nonzero(index == gi, as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            G = self.forward(poles[sel], x)
            for k, b in enumerate(sel.tolist()):
                out[b] = G[k]
        return [o for o in out if o is not None]


def gaussian_mixture_spectrum(
    weights: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    r"""Gaussian-mixture spectral function sampled at ``nu``.

    .. math::
        \tilde A(\nu) = \sum_c \frac{w_c}{\sqrt{2\pi}\,\sigma_c}\,
                        \exp\!\big[-(\nu-\mu_c)^2/2\sigma_c^2\big] .

    Evaluated on :class:`SpectralQuadrature` nodes this is integrated to
    spectral accuracy as long as :math:`\sigma_c \gtrsim h`: the Poisson-summation
    error of the core panel is :math:`2e^{-2\pi^2\sigma^2/h^2}`, which is at
    machine precision already for :math:`\sigma = h`.  Gaussians have no
    algebraic tail, so the reciprocal panels contribute essentially nothing.

    All arguments are dimensionless (:math:`\nu = \beta\omega`); ``weights``,
    ``mu``, ``sigma`` are ``(B, C)`` and ``nu`` is ``(M,)``.
    """
    z = (nu.reshape(1, 1, -1) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    norm = weights / (sigma * math.sqrt(2.0 * math.pi))
    return (norm.unsqueeze(-1) * torch.exp(-0.5 * z * z)).sum(dim=1)


def greens_matsubara(poles: PoleSet, n: torch.Tensor) -> torch.Tensor:
    r"""Exact dimensionless Matsubara Green's function :math:`g(i\nu_n)`.

    .. math:: g(i\nu_n) = \sum_p \frac{r_p}{i\nu_n - z_p},
              \qquad \nu_n = (2n+1)\pi .

    Exact for the pole ansatz -- no quadrature -- which makes it the cheapest
    independent check on :class:`PoleGreens` and the natural interface to
    Matsubara-axis solvers.

    Parameters
    ----------
    poles
        ``(B, P)`` pole set, dimensionless units.
    n
        ``(M,)`` non-negative Matsubara indices.
    """
    nu_n = (2.0 * n.to(poles.eps.dtype) + 1.0) * math.pi
    z = torch.complex(poles.eps, -poles.gamma).unsqueeze(-1)
    r = torch.complex(poles.a, poles.b).unsqueeze(-1)
    i_nu = torch.complex(torch.zeros_like(nu_n), nu_n)
    return (r / (i_nu - z)).sum(dim=1)


def matsubara_from_tau(
    G: torch.Tensor,
    x: torch.Tensor,
    n_max: int = 24,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Fourier-transform :math:`G(x)` to the Matsubara axis.

    .. math:: g(i\nu_n) \;=\; -\int_0^1 dx\; e^{i\nu_n x}\, G(x),
              \qquad \nu_n = (2n+1)\pi .

    The minus sign is the price of the *positive* time-domain convention used
    throughout this package (:math:`G = +\langle \mathcal{T} c c^\dagger\rangle`,
    so :math:`G` here is minus the textbook :math:`\mathcal{G}`); with it,
    ``matsubara_from_tau`` and :func:`greens_matsubara` return the same object,
    which is checked in the test suite.

    Used to give the encoder a **fixed-size, β-free, grid-free** summary of a
    variable-length :math:`G(\tau)`: however many time slices the simulation
    used, the first ``n_max`` Matsubara coefficients are the same physical
    quantity.  Integration is by the trapezoid rule on the supplied ``x``, so
    only low :math:`n` are trustworthy on a coarse grid -- which is exactly the
    range that carries signal rather than discretisation noise.

    Parameters
    ----------
    G, x
        ``(B, L)`` values and ascending dimensionless times.
    n_max
        Number of Matsubara frequencies returned.
    mask
        ``(B, L)`` boolean marking valid samples.

    Returns
    -------
    torch.Tensor
        ``(B, n_max, 2)`` holding ``(Re g, Im g)``.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0).expand(G.shape[0], -1)
    n = torch.arange(n_max, device=G.device, dtype=G.dtype)
    nu_n = (2.0 * n + 1.0) * math.pi
    ph = nu_n[None, :, None] * x[:, None, :]
    integ_r = torch.cos(ph) * G[:, None, :]
    integ_i = torch.sin(ph) * G[:, None, :]

    dx = x[:, 1:] - x[:, :-1]
    if mask is not None:
        dx = dx * (mask[:, 1:] & mask[:, :-1]).to(G.dtype)

    def _trapz(f: torch.Tensor) -> torch.Tensor:
        return (0.5 * (f[..., 1:] + f[..., :-1]) * dx[:, None, :]).sum(-1)

    return -torch.stack([_trapz(integ_r), _trapz(integ_i)], dim=-1)
