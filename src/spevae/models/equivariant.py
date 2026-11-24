r"""Exact particle-hole (τ-reflection) equivariance by finite-group averaging.

The symmetry
------------
The fermionic kernel satisfies :math:`K(\beta-\tau, -\omega) = K(\tau, \omega)`
identically, so if :math:`G(\tau)` has spectrum :math:`A(\omega)` then the
reflected data :math:`(PG)(\tau) = G(\beta-\tau)` has spectrum
:math:`(MA)(\omega) = A(-\omega)`.  This holds for *every* Green's function,
symmetric or not: it is a property of the inverse problem, not of the data.  The
ideal estimator is therefore equivariant under the group :math:`\{1, P\}`,

.. math:: \hat A[PG] = M\,\hat A[G] .

On poles, :math:`M` is :math:`(\varepsilon, \gamma, a, b) \mapsto
(-\varepsilon, \gamma, a, -b)`.

The construction (Jang, Bakshi & Chern, arXiv:2609.27132)
---------------------------------------------------------
Any map :math:`f` becomes exactly equivariant under a finite group by averaging
its transformed outputs over transformed inputs,

.. math:: F(X) = \frac{1}{|G|}\sum_{g} D_Y(g^{-1})\, f\big[D_X(g)\,X\big],

with no constraint on the network itself.  Here :math:`|G| = 2` and the
average of two spectra is the *union* of the two pole sets with every residue
halved, :math:`F(G) = \tfrac12 f(G) \cup \tfrac12 M f(PG)`.  The constrained
output set is closed under this: :math:`\sum a = N`, :math:`\sum b = 0`, the
per-pole :math:`|b|`/width bounds, and pointwise positivity all survive (an
average of two non-negative spectra is non-negative).  Exact 1-homogeneity
survives because :math:`N = G(0) + G(\beta)` is reflection invariant.

The latent is the pair :math:`(z_1, z_2)` -- the codes of :math:`G` and of
:math:`PG` -- and the group acts on it by the swap.  Because
:meth:`MirrorSymmetrized.encode` / :meth:`~MirrorSymmetrized.decode` expose
that pair as one ``2 * latent_dim`` vector, every latent-space routine in
:mod:`spevae.refine` (z-descent, MAP refinement, the manifold floor) runs on the
wrapper unchanged and *stays* equivariant: its objective is swap-symmetric and
Adam is permutation-equivariant, so the descent from a swapped start is the
swapped descent.

Grid requirement: reflection maps the grid to itself only when
:math:`x \to 1 - x` permutes it (uniform endpoint-inclusive DQMC grids do).
Anything else would need interpolation, which breaks exactness, so it is
refused.
"""

from __future__ import annotations

import torch
from torch import nn

from ..physics.poles import PoleSet
from .constraints import spectral_norm_from_greens
from .vae import ACResult, SPEVAE

__all__ = ["mirror_poles", "union_half", "reflect_tau", "check_symmetric_grid",
           "MirrorSymmetrized"]


def mirror_poles(p: PoleSet) -> PoleSet:
    r""":math:`A(\omega) \to A(-\omega)` on a pole set: :math:`(\varepsilon, \gamma, a, b) \to (-\varepsilon, \gamma, a, -b)`."""
    return PoleSet(-p.eps, p.gamma, p.a, -p.b)


def union_half(p: PoleSet, q: PoleSet) -> PoleSet:
    r"""Pole set of :math:`\tfrac12(A_p + A_q)`: concatenate, halve every residue."""
    return PoleSet(torch.cat([p.eps, q.eps], -1), torch.cat([p.gamma, q.gamma], -1),
                   0.5 * torch.cat([p.a, q.a], -1), 0.5 * torch.cat([p.b, q.b], -1))


def check_symmetric_grid(x: torch.Tensor, atol: float = 1e-6) -> None:
    """Raise unless ``x -> 1 - x`` maps the (last-axis) grid onto itself in reverse order."""
    if not torch.allclose(x.flip(-1), 1.0 - x, atol=atol, rtol=0.0):
        raise ValueError(
            "tau-reflection needs a grid symmetric under x -> 1 - x (e.g. uniform, "
            "endpoint-inclusive); this grid is not, and interpolating would break "
            "exact equivariance")


def reflect_tau(t: torch.Tensor | None) -> torch.Tensor | None:
    r""":math:`G(\tau) \to G(\beta - \tau)` on a symmetric grid: reverse the slice axis."""
    return None if t is None else t.flip(-1)


class MirrorSymmetrized(nn.Module):
    r"""Exactly PH-equivariant wrapper around a trained :class:`SPEVAE`.

    Drop-in for the base model wherever ``forward``/``encode``/``decode``/
    ``kernel_matrix``/``forward_map`` are used.  It owns no parameters of its
    own -- ``parameters()`` are the base model's -- so it can wrap an existing
    checkpoint (inference-time symmetrisation) or be trained through
    (training-time projection, the paper's protocol; halve the KL weight
    there, since the latent is the concatenation of two posteriors).
    """

    def __init__(self, base: SPEVAE) -> None:
        super().__init__()
        self.base = base
        self.latent_dim = 2 * base.latent_dim
        self.num_poles = 2 * base.num_poles
        self.omega_c = base.omega_c
        self.gamma_min = base.gamma_min
        self.residue_mode = base.residue_mode
        self.ph_symmetric = base.ph_symmetric
        self.config = dict(base.config, mirror_symmetrized=True)

    # -- shared machinery: delegate ------------------------------------------ #
    def nu_max(self, beta: float) -> float:
        return self.base.nu_max(beta)

    def forward_map(self, beta, device, dtype):
        return self.base.forward_map(beta, device, dtype)

    def kernel_matrix(self, beta, x):
        return self.base.kernel_matrix(beta, x)

    def clear_caches(self) -> None:
        self.base.clear_caches()

    def num_parameters(self) -> int:
        return self.base.num_parameters()

    reparameterise = staticmethod(SPEVAE.reparameterise)

    # -- the group average ---------------------------------------------------- #
    def encode(self, G, x, beta, sigma=None, mask=None, includes_endpoints=True,
               norm=None):
        r"""``G -> (mu, logvar, N)`` with ``mu = [mu(G), mu(PG)]`` (width ``2 * Lz``)."""
        check_symmetric_grid(x)
        if norm is None:
            norm = spectral_norm_from_greens(G, includes_endpoints)
        B = G.shape[0]
        G2 = torch.cat([G, reflect_tau(G)], 0)
        s2 = None if sigma is None else torch.cat([sigma, reflect_tau(sigma)], 0)
        m2 = None if mask is None else torch.cat([mask, reflect_tau(mask)], 0)
        x2 = x if x.ndim == 1 else torch.cat([x, x], 0)   # grid maps onto itself
        mu, logvar, _ = self.base.encode(G2, x2, beta, sigma=s2, mask=m2,
                                         includes_endpoints=includes_endpoints,
                                         norm=torch.cat([norm, norm], 0))
        return (torch.cat([mu[:B], mu[B:]], -1),
                torch.cat([logvar[:B], logvar[B:]], -1), norm)

    def decode(self, z, norm, beta) -> PoleSet:
        r"""``(z1, z2) -> 1/2 dec(z1) U 1/2 M dec(z2)``  (``2 * P`` poles)."""
        Lz = self.base.latent_dim
        B = z.shape[0]
        p = self.base.decode(torch.cat([z[:, :Lz], z[:, Lz:]], 0),
                             torch.cat([norm, norm], 0), beta)
        return union_half(p[:B], mirror_poles(p[B:]))

    def forward(self, G, x, beta, sigma=None, mask=None, deterministic=False,
                includes_endpoints=True) -> ACResult:
        mu, logvar, norm = self.encode(G, x, beta, sigma=sigma, mask=mask,
                                       includes_endpoints=includes_endpoints)
        z = mu if deterministic else self.reparameterise(mu, logvar)
        poles = self.decode(z, norm, beta)
        fwd = self.forward_map(beta, G.device, G.dtype)
        G_recon = fwd(poles, x, kernel_matrix=self.kernel_matrix(beta, x))
        if mask is not None:
            G_recon = G_recon * mask.to(G_recon.dtype)
        return ACResult(mu=mu, logvar=logvar, z=z, poles=poles, G_recon=G_recon,
                        norm=norm, beta=float(beta))
