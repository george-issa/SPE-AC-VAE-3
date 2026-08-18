r"""Decoder: latent code :math:`z \rightarrow` a set of :math:`P` constrained poles.

The poles are a **set**, not a vector.  All :math:`P!` relabellings describe the
same :math:`A(\omega)`, so an MLP that emits :math:`4P` numbers in a fixed order
has to break that symmetry arbitrarily, and gradients from different samples
pull the same output slot toward different physical features.

This decoder instead uses :math:`P` learned *pole queries* refined by
self-attention and conditioned on :math:`z` -- the set-prediction pattern from
DETR-style detection, which is the right inductive bias for "emit an unordered
collection of objects".  Queries communicate with each other at every layer, so
they can divide the spectrum up between themselves (this pole takes the
quasiparticle peak, that one takes the upper Hubbard band) instead of each
independently regressing the whole thing.

Pole-count as regularisation
----------------------------
:math:`P` is an upper bound, not a target.  The ``softmax`` weight head lets
surplus poles fade to negligible weight, and
:func:`spevae.physics.poles.effective_pole_count` measures how many the model
actually spends.  A spread of :math:`P` at fixed data is therefore a
regularisation sweep, and the effective count -- not :math:`P` -- is the number
to report.

Initialisation
--------------
Position-head biases are initialised on a spread in the ``sinh`` pre-activation,
so at step zero the poles already tile the band instead of piling up at
:math:`\omega = 0`.  Starting degenerate is the classic way a set decoder gets
stuck: identical queries receive identical gradients and never separate.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..physics.poles import PoleSet
from .constraints import DEFAULT_GAMMA_MIN, PoleConstraints

__all__ = ["PoleDecoder"]


class PoleDecoder(nn.Module):
    r"""Latent code :math:`\rightarrow` :class:`PoleSet`.

    Parameters
    ----------
    latent_dim
        Width of ``z``.
    num_poles
        Upper bound :math:`P` on the number of poles.
    n_context
        Number of memory tokens the latent is expanded into.  More than one lets
        different queries attend to different aspects of ``z``.
    residue_mode, gamma_min, nu_max
        Passed to :class:`~spevae.models.constraints.PoleConstraints`.
    eps_init_span
        Half-width of the initial spread of position-head biases in
        pre-activation units; :math:`\pm 3` maps to roughly
        :math:`\tilde\varepsilon \in [-31, 31]`, i.e. :math:`\pm 3` in physical
        units at :math:`\beta = 10`.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        num_poles: int = 12,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 512,
        n_context: int = 4,
        dropout: float = 0.0,
        residue_mode: str = "complex",
        gamma_min: float = DEFAULT_GAMMA_MIN,
        gamma_scale: float = math.pi,
        nu_max: float = 240.0,
        eps_scale: float = math.pi,
        eps_init_span: float = 3.0,
    ) -> None:
        super().__init__()
        self.num_poles = int(num_poles)
        self.n_context = int(n_context)

        self.queries = nn.Parameter(torch.randn(num_poles, d_model) * 0.02)
        self.context = nn.Linear(latent_dim, n_context * d_model)
        self.d_model = d_model

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head_eps = nn.Linear(d_model, 1)
        self.head_gamma = nn.Linear(d_model, 1)
        self.head_a = nn.Linear(d_model, 1)
        # In "real" mode b is identically zero, so a dispersive head would be a
        # parameter that can never receive a gradient.  Omit it rather than
        # carry dead weights that quietly inflate the parameter count.
        self.head_b = nn.Linear(d_model, 1) if residue_mode == "complex" else None

        self.constraints = PoleConstraints(
            residue_mode=residue_mode, gamma_min=gamma_min,
            gamma_scale=gamma_scale, nu_max=nu_max, eps_scale=eps_scale,
        )
        self._init_heads(eps_init_span)

    def _init_heads(self, span: float) -> None:
        """Default head weights; per-pole biases spread across the band.

        The heads keep PyTorch's default initialisation on purpose.  Shrinking
        them (a natural instinct, to start from a "neutral" spectrum) makes the
        poles nearly independent of ``z``, so early gradients carry almost no
        information about the latent code and the encoder has nothing to learn
        from.
        """
        for head in (self.head_eps, self.head_gamma, self.head_a, self.head_b):
            if head is None:
                continue
            nn.init.zeros_(head.bias)
        # A shared Linear head gives every pole the same bias, so the per-pole
        # offsets live in their own parameters.  Spreading the position offsets
        # is what stops the queries from starting degenerate: identical queries
        # get identical gradients and never separate.
        self.eps_bias = nn.Parameter(torch.linspace(-span, span, self.num_poles))
        self.gamma_bias = nn.Parameter(torch.zeros(self.num_poles))
        self.a_bias = nn.Parameter(torch.zeros(self.num_poles))
        if self.head_b is not None:
            self.b_bias = nn.Parameter(torch.zeros(self.num_poles))
        else:
            self.register_parameter("b_bias", None)

    def forward(
        self,
        z: torch.Tensor,
        norm: torch.Tensor | None = None,
        nu_max: float | None = None,
    ) -> PoleSet:
        r"""Decode ``(B, latent_dim)`` to a ``(B, P)`` constrained pole set.

        ``nu_max`` is the dimensionless position bound :math:`\beta\omega_{\max}`
        for the cell being decoded; it varies with :math:`\beta` even though the
        network's parameters do not.
        """
        B = z.shape[0]
        memory = self.context(z).reshape(B, self.n_context, self.d_model)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        h = self.norm(self.transformer(q, memory))

        u_eps = self.head_eps(h).squeeze(-1) + self.eps_bias
        u_gamma = self.head_gamma(h).squeeze(-1) + self.gamma_bias
        u_a = self.head_a(h).squeeze(-1) + self.a_bias
        u_b = None if self.head_b is None else self.head_b(h).squeeze(-1) + self.b_bias
        return self.constraints(u_eps, u_gamma, u_a, u_b, norm=norm, nu_max=nu_max)
