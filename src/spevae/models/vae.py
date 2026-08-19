r"""The analytic-continuation VAE: :math:`G(\tau) \rightarrow q(z\mid G) \rightarrow` poles :math:`\rightarrow A(\omega)`.

Why a VAE and not a regressor
-----------------------------
Analytic continuation is ill-posed in two directions at once, and a
*distribution* over spectra is the honest object for both:

* **Many-to-one.**  Different noise realisations of the same simulation carry the
  same physics.  A stochastic encoder can map them into overlapping posteriors,
  so the noise is absorbed by the latent uncertainty rather than being
  transcribed into spurious spectral wiggles.  The training objective makes this
  explicit through a noise-consistency term (see :mod:`spevae.losses`).
* **One-to-many.**  Two nearby :math:`G(\tau)` can be compatible with genuinely
  different :math:`A(\omega)`, because the kernel annihilates high-frequency
  structure.  Sampling :math:`z` and decoding repeatedly gives an *ensemble* of
  admissible spectra whose spread localises where the data stops constraining
  the answer -- exactly the region a point estimate would silently fabricate.

The caveat that governs how results may be reported: a spread is only an
uncertainty if the encoder is actually reading its input.  Posterior collapse
produces a beautifully tight ensemble that means nothing, so
:func:`active_units` is computed alongside every result and belongs in every
table.  See ``docs/DESIGN.md``.

Exact 1-homogeneity
-------------------
:meth:`SPEVAE.forward` normalises by :math:`N = G(0)+G(\beta)` before encoding
and multiplies the residues by :math:`N` after decoding.  Hence

.. math:: A[cG] = c\,A[G] \quad \text{for every } c, \qquad A[0] = 0 ,

identically in the weights, at any point in training.  Note that the *poles*
are scale-invariant while the *residues* are 1-homogeneous, which is the correct
structure: rescaling a Green's function changes how much spectral weight there
is, not where it sits.
"""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass

import torch
from torch import nn

from ..physics.forward import PoleGreens, SpectralQuadrature
from ..physics.poles import PoleSet, effective_pole_count
from .constraints import DEFAULT_GAMMA_MIN, spectral_norm_from_greens
from .decoder import PoleDecoder
from .encoder import GreensEncoder

__all__ = ["SPEVAE", "ACResult", "active_units"]


@dataclass
class ACResult:
    """Everything one forward pass produces."""

    mu: torch.Tensor                 # (B, Lz) posterior mean
    logvar: torch.Tensor             # (B, Lz) posterior log-variance
    z: torch.Tensor                  # (B, Lz) sampled (or mean) code
    poles: PoleSet                   # (B, P) dimensionless
    G_recon: torch.Tensor            # (B, L)
    norm: torch.Tensor               # (B,) spectral norm read from the data
    beta: float

    def poles_physical(self) -> PoleSet:
        r"""Poles in physical frequency units :math:`\omega = \tilde\nu/\beta`."""
        return self.poles.to_physical(self.beta)

    def spectral(self, omega: torch.Tensor) -> torch.Tensor:
        r""":math:`A(\omega)` on any grid, at any resolution -- no retraining, no rebinning."""
        return self.poles_physical().spectral(omega.to(self.mu.dtype))

    def effective_poles(self) -> torch.Tensor:
        """Participation ratio of the weights: how many poles the fit actually spends."""
        return effective_pole_count(self.poles.a / self.norm.reshape(-1, 1))

    def positivity(self, atol: float = 1e-10) -> list:
        r"""Exact pointwise-positivity certificate for every sample in the batch.

        Decided from the pole parameters alone (no :math:`\omega` grid): see
        :func:`spevae.physics.positivity.positivity_certificate`.  In
        ``residue_mode="real"`` every report is trivially positive; in complex
        mode this is the number to print next to any spectrum.
        """
        from ..physics.positivity import positivity_certificate
        return positivity_certificate(self.poles_physical(), atol=atol)


def active_units(mu: torch.Tensor, threshold: float = 1e-2) -> torch.Tensor:
    r"""Number of latent dimensions whose posterior mean actually varies with the input.

    :math:`\mathrm{AU} = \#\{u : \mathrm{Var}_x[\mu_u(x)] > \text{threshold}\}`
    (Burda et al., 2016).  ``AU = 0`` means the encoder's output is a constant
    function of :math:`G`: the model has become a fixed :math:`z \mapsto
    A(\omega)` map that ignores which sample it was given.  Such a run can still
    have an excellent :math:`\chi^2` and a plausible mean spectrum, so
    :math:`\chi^2` alone cannot detect it -- which is why this number has to be
    reported next to every result rather than checked occasionally.
    """
    return (mu.var(dim=0, unbiased=False) > threshold).sum()


class SPEVAE(nn.Module):
    r"""Physics-constrained VAE for analytic continuation.

    Parameters
    ----------
    latent_dim
        Latent width.  Match it to the intrinsic dimensionality of the physics
        family in the training set; an over-wide bottleneck makes collapse
        cheaper for the optimiser without adding expressivity.
    num_poles
        Upper bound on the pole count (see :class:`~spevae.models.decoder.PoleDecoder`).
    omega_c
        Physical frequency cutoff.  Sets both the quadrature range
        (:math:`\nu_c = \beta\omega_c`) and, at 80% of it, the pole position
        bound.  It should comfortably exceed the bandwidth of the system being
        continued.
    residue_mode
        ``"complex"`` (default) frees the dispersive residue part, which is what
        lets a pole produce an *asymmetric* line shape -- physical spectra are
        "almost Lorentzian, but not exactly", and a symmetric-only basis pays for
        every asymmetry by stacking extra poles.  It also removes the hard floor
        on the :math:`1/\omega^2` tail that the real mode carries
        (:math:`c_2 \ge \gamma_{\min}/\pi`), which is what caps attainable peak
        heights.  The cost is that :math:`A(\omega) \ge 0` is no longer
        structural -- it becomes a penalty plus a reported ``min_A`` -- so print
        that number with any spectrum.  ``"real"`` restores the guarantee.
    ph_symmetric
        Mirror-paired decoder: the network emits ``num_poles/2`` poles and the
        constraint layer adds :math:`(-\varepsilon_p, \gamma_p, a_p/2, -b_p/2)`
        for each, making :math:`A(\omega) = A(-\omega)` **identical in the
        weights** -- particle-hole symmetry as architecture.  Use only for
        data where the symmetry is exact (e.g. half filling); train with a
        ``SyntheticGenerator(ph_symmetric=True)`` so the targets respect what
        the decoder enforces.  On asymmetric data a symmetric decoder is a
        structurally wrong prior, not a regulariser.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        num_poles: int = 12,
        omega_c: float = 20.0,
        residue_mode: str = "complex",
        ph_symmetric: bool = False,
        gamma_min: float = DEFAULT_GAMMA_MIN,
        d_model: int = 128,
        enc_layers: int = 4,
        dec_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 512,
        n_matsubara: int = 24,
        dropout: float = 0.0,
        quad_nodes_per_width: float = 3.5,
        quad_tail_nodes: int = 32,
        max_quad_nodes: int = 400_000,
        cache_limit: int = 4,
    ) -> None:
        super().__init__()
        # Every constructor argument, recorded so a checkpoint can rebuild the
        # exact architecture without the caller having to remember it.  A
        # state_dict alone is not a model: the widths, the pole count and the
        # residue mode are all needed, and getting one wrong loads silently
        # wrong weights or fails with a shape error days later.
        self.config = dict(
            latent_dim=int(latent_dim), num_poles=int(num_poles),
            omega_c=float(omega_c), residue_mode=str(residue_mode),
            ph_symmetric=bool(ph_symmetric),
            gamma_min=float(gamma_min), d_model=int(d_model),
            enc_layers=int(enc_layers), dec_layers=int(dec_layers),
            n_heads=int(n_heads), d_ff=int(d_ff),
            n_matsubara=int(n_matsubara), dropout=float(dropout),
            quad_nodes_per_width=float(quad_nodes_per_width),
            quad_tail_nodes=int(quad_tail_nodes),
            max_quad_nodes=int(max_quad_nodes), cache_limit=int(cache_limit),
        )
        self.latent_dim = int(latent_dim)
        self.num_poles = int(num_poles)
        self.omega_c = float(omega_c)
        self.gamma_min = float(gamma_min)
        self.residue_mode = residue_mode
        self.ph_symmetric = bool(ph_symmetric)
        self._quad_h = float(gamma_min) / float(quad_nodes_per_width)
        self._quad_tail_nodes = int(quad_tail_nodes)
        self._max_quad_nodes = int(max_quad_nodes)
        self._cache_limit = max(int(cache_limit), 1)

        self.encoder = GreensEncoder(
            latent_dim=latent_dim, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, d_ff=d_ff, n_matsubara=n_matsubara, dropout=dropout,
        )
        self.decoder = PoleDecoder(
            latent_dim=latent_dim, num_poles=num_poles, d_model=d_model,
            n_layers=dec_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            residue_mode=residue_mode, gamma_min=gamma_min,
            ph_symmetric=ph_symmetric,
        )
        self._quad_cache: dict[tuple, PoleGreens] = {}
        self._kernel_cache: dict[tuple, torch.Tensor] = {}
        self._grid_fingerprints: dict[int, str] = {}

    # -- quadrature bookkeeping -------------------------------------------- #
    #
    # Both caches are *bounded*.  They exist for the common case of many batches
    # on one simulation cell, where the quadrature and the (M, L) kernel matrix
    # are worth reusing.  During pretraining, though, beta is drawn continuously,
    # so every step is a cache miss *and* a new entry: an unbounded dict grows by
    # the size of one kernel matrix per step -- ~5 MB at beta = 12 -- and reaches
    # tens of gigabytes within a few thousand steps.  That is how the first long
    # training run died.

    def _cache_put(self, cache: dict, key, value, limit: int):
        cache[key] = value
        while len(cache) > limit:
            cache.pop(next(iter(cache)))          # dicts are insertion-ordered
        return value

    def nu_max(self, beta: float) -> float:
        r"""Dimensionless pole-position bound for this cell, :math:`0.8\,\beta\omega_c`."""
        return 0.8 * float(beta) * self.omega_c

    def forward_map(self, beta: float, device, dtype) -> PoleGreens:
        """Quadrature rule for one inverse temperature (bounded cache)."""
        key = (round(float(beta), 9), str(device), str(dtype))
        fwd = self._quad_cache.get(key)
        if fwd is None:
            quad = SpectralQuadrature(
                nu_c=float(beta) * self.omega_c, h=self._quad_h,
                n_tail=self._quad_tail_nodes, dtype=dtype,
            ).to(device)
            if self.gamma_min <= quad.min_resolvable_gamma:
                raise ValueError(
                    f"gamma_min={self.gamma_min:.4g} is below the quadrature's "
                    f"resolution limit {quad.min_resolvable_gamma:.4g}; refine the "
                    f"rule (raise quad_nodes_per_width) or raise gamma_min."
                )
            if quad.n_nodes > self._max_quad_nodes:
                raise ValueError(
                    f"quadrature needs {quad.n_nodes:,} nodes at beta={beta:g} "
                    f"(nu_c={beta * self.omega_c:g}, h={self._quad_h:.3g}), above "
                    f"max_quad_nodes={self._max_quad_nodes:,}. The node count is "
                    f"2*beta*omega_c/h, so lower omega_c, raise gamma_min, or raise "
                    f"the cap deliberately."
                )
            fwd = self._cache_put(self._quad_cache, key, PoleGreens(quad),
                                  self._cache_limit)
        return fwd

    def _grid_fingerprint(self, x: torch.Tensor) -> str:
        """Content hash of a grid tensor, memoised per tensor object.

        Keying the kernel cache on (length, endpoints) alone lets two different
        *non-uniform* grids with the same length and endpoints silently share a
        kernel matrix.  Hashing the actual values closes that hole; the memo
        means a caller reusing one grid tensor pays the device sync once.  The
        memo is keyed by ``id(x)`` (tensors cannot live in a WeakKeyDictionary:
        its lookups compare keys with ``==``, which is elementwise), and a
        ``weakref.finalize`` evicts the entry the moment the tensor is freed,
        so a recycled id can never return a stale fingerprint.
        """
        key = id(x)
        fp = self._grid_fingerprints.get(key)
        if fp is None:
            fp = hashlib.sha1(x.detach().cpu().numpy().tobytes()).hexdigest()
            self._grid_fingerprints[key] = fp
            weakref.finalize(x, self._grid_fingerprints.pop, key, None)
        return fp

    def kernel_matrix(self, beta: float, x: torch.Tensor) -> torch.Tensor:
        """Kernel matrix for one (beta, grid); reused across batches on that cell."""
        key = (round(float(beta), 9), int(x.numel()), self._grid_fingerprint(x),
               str(x.device), str(x.dtype))
        K = self._kernel_cache.get(key)
        if K is None:
            K = self._cache_put(
                self._kernel_cache, key,
                self.forward_map(beta, x.device, x.dtype).kernel_matrix(x),
                self._cache_limit)
        return K

    def clear_caches(self) -> None:
        """Drop the cached quadratures and kernel matrices."""
        self._quad_cache.clear()
        self._kernel_cache.clear()

    def cache_bytes(self) -> int:
        """Approximate memory held by the caches, for diagnostics."""
        n = sum(K.numel() * K.element_size() for K in self._kernel_cache.values())
        n += sum(f.quad.nodes.numel() * f.quad.nodes.element_size() * 2
                 for f in self._quad_cache.values())
        return int(n)

    # -- pieces -------------------------------------------------------------- #

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(-20.0, 20.0)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(
        self,
        G: torch.Tensor,
        x: torch.Tensor,
        beta: float,
        sigma: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        includes_endpoints: bool = True,
        norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""``G -> (mu, logvar, N)``, with the scale stripped before encoding."""
        if norm is None:
            norm = spectral_norm_from_greens(G, includes_endpoints)
        denom = torch.where(norm.abs() > 1e-12, norm, torch.ones_like(norm))
        g_hat = G / denom.reshape(-1, 1)
        sigma_hat = None if sigma is None else sigma / denom.reshape(-1, 1).abs()
        mu, logvar = self.encoder(g_hat, x, sigma_hat=sigma_hat, beta=beta, mask=mask)
        return mu, logvar, norm

    def decode(self, z: torch.Tensor, norm: torch.Tensor, beta: float) -> PoleSet:
        return self.decoder(z, norm=norm, nu_max=self.nu_max(beta))

    def forward(
        self,
        G: torch.Tensor,
        x: torch.Tensor,
        beta: float,
        sigma: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        deterministic: bool = False,
        includes_endpoints: bool = True,
    ) -> ACResult:
        r"""One full pass.

        Parameters
        ----------
        G
            ``(B, L)`` imaginary-time Green's function.
        x
            ``(L,)`` dimensionless times :math:`\tau/\beta`, one grid per batch.
        beta
            Inverse temperature of this batch (a scalar: β is a property of the
            simulation cell, and batching by cell keeps the forward map a single
            matrix product).
        sigma
            ``(B, L)`` per-slice standard deviations, if known.
        deterministic
            Use :math:`z = \mu` instead of sampling.  The right choice for
            validation and for reporting a single spectrum; sample when you want
            the ensemble.
        """
        mu, logvar, norm = self.encode(
            G, x, beta, sigma=sigma, mask=mask, includes_endpoints=includes_endpoints
        )
        z = mu if deterministic else self.reparameterise(mu, logvar)
        poles = self.decode(z, norm, beta)
        fwd = self.forward_map(beta, G.device, G.dtype)
        G_recon = fwd(poles, x, kernel_matrix=self.kernel_matrix(beta, x))
        if mask is not None:
            G_recon = G_recon * mask.to(G_recon.dtype)
        return ACResult(mu=mu, logvar=logvar, z=z, poles=poles,
                        G_recon=G_recon, norm=norm, beta=float(beta))

    # -- inference ----------------------------------------------------------- #

    @torch.no_grad()
    def continue_(
        self,
        G: torch.Tensor,
        beta: float,
        tau: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        n_samples: int = 0,
        includes_endpoints: bool = True,
    ) -> dict:
        r"""Analytically continue a batch of Green's functions.

        Parameters
        ----------
        G
            ``(B, L)`` or ``(L,)`` imaginary-time data.
        beta
            Inverse temperature.
        tau
            ``(L,)`` imaginary times; defaults to a uniform endpoint-inclusive
            grid ``linspace(0, beta, L)``.
        omega
            Real-frequency grid on which to report :math:`A`.  Optional -- the
            pole set is grid-free, and ``poles`` in the result lets a caller
            evaluate :math:`A` anywhere afterwards.
        n_samples
            ``0`` decodes the posterior mean (a single spectrum).  ``n > 0``
            draws ``n`` latent samples and returns the ensemble mean and
            standard deviation, which is the model's statement about where the
            data stops determining the answer.

        Returns
        -------
        dict
            ``poles`` (physical units), ``G_recon``, ``norm``, ``A`` /
            ``A_std`` when ``omega`` is given, plus ``mu`` and ``logvar``.
        """
        self.eval()
        if G.ndim == 1:
            G = G.unsqueeze(0)
        L = G.shape[-1]
        if tau is None:
            tau = torch.linspace(0.0, beta, L, dtype=G.dtype, device=G.device)
        x = (tau / beta).to(G.dtype)

        mu, logvar, norm = self.encode(
            G, x, beta, sigma=sigma, includes_endpoints=includes_endpoints
        )
        K = self.kernel_matrix(beta, x)
        fwd = self.forward_map(beta, G.device, G.dtype)

        if n_samples <= 0:
            poles = self.decode(mu, norm, beta)
            out = {
                "poles": poles.to_physical(beta),
                "G_recon": fwd(poles, x, kernel_matrix=K),
                "norm": norm, "mu": mu, "logvar": logvar,
            }
            if omega is not None:
                out["A"] = out["poles"].spectral(omega.to(G.dtype))
            return out

        As, Gs = [], []
        poles_last = None
        for _ in range(int(n_samples)):
            z = self.reparameterise(mu, logvar)
            poles = self.decode(z, norm, beta)
            poles_last = poles
            Gs.append(fwd(poles, x, kernel_matrix=K))
            if omega is not None:
                As.append(poles.to_physical(beta).spectral(omega.to(G.dtype)))
        out = {
            "poles": poles_last.to_physical(beta),
            "G_recon": torch.stack(Gs).mean(0),
            "norm": norm, "mu": mu, "logvar": logvar,
        }
        if omega is not None:
            stack = torch.stack(As)
            out["A"] = stack.mean(0)
            out["A_std"] = stack.std(0)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
