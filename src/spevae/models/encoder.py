r"""Transformer encoder: a variable-length, β-agnostic, noise-aware map :math:`G(\tau) \to q(z\mid G)`.

The problem this solves
-----------------------
DQMC gives :math:`G` on :math:`L_\tau = \beta/\Delta\tau + 1` points, so the
input dimension changes with both the inverse temperature and the time step.  A
fixed-width convolutional stack -- the natural first thing to write, and what
the predecessor project used -- has to be rebuilt and retrained for every
simulation cell, which is fatal for a package other people are meant to run on
their own data.

Two changes remove the dependence entirely:

1. **Dimensionless time.**  Tokens carry :math:`x = \tau/\beta \in [0,1]`, not
   :math:`\tau`.  Under the reduction in :mod:`spevae.physics.kernel` the
   inverse problem itself is β-free, so :math:`x` is the coordinate the physics
   actually cares about.  β is supplied separately as a conditioning scalar,
   because it still sets the *prior* over spectra (bandwidth in units of
   temperature) even though it does not change the map.
2. **A transformer over time slices instead of a fixed-width stack.**  Each
   :math:`\tau` point is a token; a padding mask handles ragged lengths.  Any
   :math:`L_\tau`, uniform or not, is a valid input, and nothing about the
   architecture changes between :math:`\beta = 5` and :math:`\beta = 20`.

Scale invariance
----------------
The encoder is given :math:`\hat G = G/N` with :math:`N = G(0) + G(\beta)`, never
:math:`G` itself.  It is therefore *exactly* invariant under
:math:`G \to cG`, and the decoder's rescaling by :math:`N` makes the full
estimator exactly 1-homogeneous -- which is how "a zero Green's function maps to
a zero spectral function" becomes a property of the architecture rather than
something training has to discover.

Noise awareness
---------------
Per-slice error bars enter as a token feature.  Analytic continuation is
ill-posed in a noise-dependent way: how much structure is recoverable is set by
the signal-to-noise ratio at each :math:`\tau`, so a network that cannot see the
error bars has to guess a fixed noise level and will over-resolve clean data and
over-smooth noisy data.  Feeding :math:`\log(\hat\sigma)` lets one trained model
cover the whole range.

Global summary tokens
---------------------
A ``CLS`` token carries quantities that are properties of the whole curve rather
than of one slice: :math:`\log\beta`, :math:`\log L_\tau`, the noise level, the
curve's own location and scale, and the leading Matsubara coefficients
:math:`g(i\nu_n)` of :math:`\hat G`.  The Matsubara block is a fixed-size,
grid-free, β-free re-encoding of the same data -- the representation every
classical analytic-continuation method works in -- so the network gets the
canonical summary for free rather than having to synthesise it through attention.

Feature conditioning (this matters more than it looks)
------------------------------------------------------
Within one batch the time grid, its Fourier features, β and the noise scale are
all *identical across samples*; only :math:`\hat G` varies.  Fed raw, the
constant channels dominate the embedding's magnitude and the per-sample signal
survives as a ~2% perturbation, which after a few normalised layers leaves the
posterior mean essentially constant -- zero active units, from step one, with no
KL pressure needed to explain it.  Three choices avoid that:

* **Per-sample standardisation along** :math:`\tau`.  :math:`\hat G` is
  standardised using its own mean and spread, and those two numbers are handed
  to the ``CLS`` token separately.  This is a per-sample operation, so it behaves
  identically at batch size 1, and it puts the *shape* -- the thing that
  differs -- at :math:`O(1)`.
* **A signed-log channel.**  :math:`G(\tau)` falls by orders of magnitude across
  the interval, and the decay rate is where the spectral support is written; on
  a linear scale that information is compressed into the first few slices.
* **Scale-matched auxiliary channels** and a ``LayerNorm`` on both embeddings, so
  no single feature sets the magnitude of the residual stream.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..physics.forward import matsubara_from_tau

__all__ = ["TimeFeatures", "GreensEncoder"]


class TimeFeatures(nn.Module):
    r"""Continuous positional features for :math:`x = \tau/\beta \in [0,1]`.

    Learned positional embeddings would tie the model to one grid, and raw
    :math:`x` alone resolves the endpoints poorly -- yet :math:`G` varies fastest
    exactly there, where the high-frequency spectral weight lives.  The feature
    set is :math:`[\,x,\; 1-x,\; \log x,\; \log(1-x),\; \sin/\cos(2\pi k x)\,]`,
    which is smooth, grid-free, and symmetric under the particle--hole reflection
    :math:`x \to 1-x` (the features map onto each other, so the network can
    represent that symmetry without being forced into it).
    """

    def __init__(self, n_fourier: int = 8, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("freqs", 2.0 * math.pi * torch.arange(1, n_fourier + 1).float())

    @property
    def dim(self) -> int:
        return 4 + 2 * int(self.freqs.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` dimensionless times -> ``(B, L, dim)``."""
        f = self.freqs.to(x.dtype)
        ang = x.unsqueeze(-1) * f
        return torch.cat(
            [
                x.unsqueeze(-1),
                (1.0 - x).unsqueeze(-1),
                torch.log(x.unsqueeze(-1) + self.eps),
                torch.log((1.0 - x).unsqueeze(-1) + self.eps),
                torch.sin(ang),
                torch.cos(ang),
            ],
            dim=-1,
        )


class GreensEncoder(nn.Module):
    r"""Transformer encoder over imaginary-time slices.

    A pre-LayerNorm :class:`torch.nn.TransformerEncoder` (GELU feed-forward)
    reads one token per slice plus a ``CLS`` summary token; ``mu`` and
    ``logvar`` are read off the ``CLS`` output.

    Parameters
    ----------
    latent_dim
        Size of the latent code.  Keep it near the intrinsic dimensionality of
        the physics family being modelled: an over-wide bottleneck leaves most
        dimensions inert and makes posterior collapse cheaper for the optimiser.
    d_model, n_layers, n_heads, d_ff
        Transformer geometry.  Capacity here is what lets longer training buy
        better spectra, so these are the knobs to scale.
    n_matsubara
        Number of Matsubara coefficients in the summary token.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        n_fourier: int = 8,
        n_matsubara: int = 24,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_matsubara = int(n_matsubara)
        self.time_features = TimeFeatures(n_fourier)

        # Per-slice token: standardised G, signed-log G, raw G, relative noise,
        # has_sigma, + time features.
        token_in = 5 + self.time_features.dim
        self.token_proj = nn.Linear(token_in, d_model)
        self.token_norm = nn.LayerNorm(d_model)

        # Whole-curve token: log beta, log L, log noise, has_sigma, curve mean,
        # log curve spread, log Matsubara RMS, + the normalised Matsubara block.
        self.cls_proj = nn.Linear(7 + 2 * self.n_matsubara, d_model)
        self.cls_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        # Start with an informative, fairly sharp posterior.  A near-zero
        # initialisation of `to_mu` is tempting (it starts close to the prior and
        # keeps the KL small) but it makes z a constant, so the decoder receives
        # no signal to learn from and the encoder has to bootstrap itself out of
        # a flat region -- an easy way to sit at zero active units for thousands
        # of steps.  Default weights plus a negative logvar bias
        # (sigma ~ 0.2) let the code carry information from step one; the KL term
        # is what pushes it back towards the prior, not the initialisation.
        nn.init.zeros_(self.to_mu.bias)
        nn.init.constant_(self.to_logvar.bias, -3.0)
        nn.init.normal_(self.to_logvar.weight, std=1e-3)

    def forward(
        self,
        g_hat: torch.Tensor,
        x: torch.Tensor,
        sigma_hat: torch.Tensor | None = None,
        beta: torch.Tensor | float | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Encode a batch of *scale-normalised* Green's functions.

        Parameters
        ----------
        g_hat
            ``(B, L)`` :math:`\hat G = G/N`.
        x
            ``(L,)`` or ``(B, L)`` dimensionless times.
        sigma_hat
            ``(B, L)`` normalised per-slice standard deviations, or ``None``.
        beta
            ``(B,)`` or scalar inverse temperature (conditioning only).
        mask
            ``(B, L)`` boolean; ``False`` marks padding.

        Returns
        -------
        (mu, logvar)
            Each ``(B, latent_dim)``.
        """
        B, L = g_hat.shape
        if x.ndim == 1:
            x = x.unsqueeze(0).expand(B, -1)
        x = x.to(g_hat.dtype)
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=g_hat.device)
        valid = mask.to(g_hat.dtype)

        has_sigma = 0.0 if sigma_hat is None else 1.0
        n_valid = valid.sum(-1).clamp_min(1.0)

        # Per-sample location/scale of the curve, computed over valid slices only.
        g_mean = (g_hat * valid).sum(-1, keepdim=True) / n_valid.unsqueeze(-1)
        g_var = (((g_hat - g_mean) ** 2) * valid).sum(-1, keepdim=True) / n_valid.unsqueeze(-1)
        g_scale = g_var.clamp_min(1e-24).sqrt()
        g_std = (g_hat - g_mean) / g_scale                        # shape, O(1)
        g_slog = torch.sign(g_hat) * torch.log1p(g_hat.abs() / 1e-4) / 4.0

        if sigma_hat is not None:
            log_sigma = torch.log(sigma_hat.clamp_min(1e-30))
            mean_log_sigma = (log_sigma * valid).sum(-1, keepdim=True) / n_valid.unsqueeze(-1)
            rel_log_sigma = (log_sigma - mean_log_sigma) / 2.0     # noise *profile*
        else:
            mean_log_sigma = torch.zeros(B, 1, device=g_hat.device, dtype=g_hat.dtype)
            rel_log_sigma = torch.zeros_like(g_hat)

        tokens = torch.cat(
            [
                g_std.unsqueeze(-1),
                g_slog.unsqueeze(-1),
                (4.0 * g_hat).unsqueeze(-1),
                rel_log_sigma.unsqueeze(-1),
                torch.full_like(g_hat, has_sigma).unsqueeze(-1),
                self.time_features(x),
            ],
            dim=-1,
        )
        h = self.token_norm(self.token_proj(tokens))

        # Whole-curve summary.
        if beta is None:
            log_beta = torch.zeros(B, 1, device=g_hat.device, dtype=g_hat.dtype)
        else:
            beta_t = torch.as_tensor(beta, device=g_hat.device, dtype=g_hat.dtype)
            log_beta = torch.log(beta_t.reshape(-1).expand(B).clamp_min(1e-6)).unsqueeze(-1)

        mats = matsubara_from_tau(g_hat * valid, x, n_max=self.n_matsubara, mask=mask)
        mats_rms = mats.flatten(1).pow(2).mean(-1, keepdim=True).clamp_min(1e-30).sqrt()
        cls_in = torch.cat(
            [
                log_beta,
                torch.log(n_valid).unsqueeze(-1),
                (mean_log_sigma + 8.0) / 4.0,
                torch.full((B, 1), has_sigma, device=g_hat.device, dtype=g_hat.dtype),
                4.0 * g_mean,
                torch.log(g_scale),
                torch.log(mats_rms),
                # Scale-normalised so the Matsubara *shape* is O(1); the overall
                # magnitude is already supplied as its log, above.
                (mats.reshape(B, -1) / mats_rms),
            ],
            dim=-1,
        )
        cls = self.cls_norm(self.cls_proj(cls_in)).unsqueeze(1)

        h = torch.cat([cls, h], dim=1)
        pad = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], 1)
        h = self.transformer(h, src_key_padding_mask=~pad)
        summary = self.norm(h[:, 0])
        return self.to_mu(summary), self.to_logvar(summary)
