r"""Synthetic training batches: spectrum :math:`\rightarrow` clean :math:`G` :math:`\rightarrow` noisy bins.

One batch is one *simulated cell*: a single :math:`(\beta, \Delta\tau)` and a
single noise process, with ``batch_size`` independent spectra drawn from one
family.  That mirrors how DQMC data actually arrives, keeps the forward map a
single matrix product, and makes stratified evaluation ("how do we do on gapped
spectra at :math:`\beta = 20`?") a matter of grouping batches.

:math:`\beta`, :math:`\Delta\tau`, the noise amplitude, the noise correlation
length and the spectral family all vary *between* batches.  Over an epoch the
network therefore sees the whole range, which is what makes one set of weights
usable on someone else's simulation rather than only on the cell it was tuned
for.

Paired realisations
-------------------
Every batch carries **two independent noise draws of the same clean**
:math:`G`.  This is what turns "different noisy Green's functions should map to
the same spectral function" from an aspiration into a training signal: the
consistency term in :mod:`spevae.losses` penalises the disagreement between the
two posteriors directly, at zero extra forward-map cost.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

import torch

from ..physics.forward import PoleGreens, SpectralQuadrature
from .noise import NoiseModel, sample_correlated_noise, whitener
from .spectra import SpectrumBatch, sample_spectra

__all__ = ["TrainingBatch", "SyntheticGenerator"]


@dataclass
class TrainingBatch:
    """One simulated cell's worth of training data."""

    G: torch.Tensor                  # (B, L) noisy realisation 1
    G_alt: torch.Tensor              # (B, L) noisy realisation 2, same clean signal
    G_clean: torch.Tensor            # (B, L)
    sigma: torch.Tensor              # (B, L) per-slice standard deviation
    inv_sqrt_R: torch.Tensor         # (L, L) whitener for the shared correlation
    x: torch.Tensor                  # (L,) dimensionless times
    beta: float
    A_true: torch.Tensor             # (B, M) on the quadrature nodes
    quad: SpectralQuadrature
    family: list[str]
    poles_true: tuple | None = None
    meta: dict = field(default_factory=dict)

    def to(self, device=None, dtype=None) -> "TrainingBatch":
        def _c(t):
            return t if t is None else t.to(device=device, dtype=dtype)
        return TrainingBatch(
            G=_c(self.G), G_alt=_c(self.G_alt), G_clean=_c(self.G_clean),
            sigma=_c(self.sigma), inv_sqrt_R=_c(self.inv_sqrt_R), x=_c(self.x),
            beta=self.beta, A_true=_c(self.A_true), quad=self.quad,
            family=self.family, poles_true=self.poles_true, meta=self.meta,
        )


class SyntheticGenerator:
    r"""Sampler for pretraining batches.

    Parameters
    ----------
    omega_c
        Physical frequency cutoff; must match the model's, since it fixes the
        quadrature both are evaluated on.
    gamma_min
        Resolution floor, dimensionless.  Sets both the quadrature spacing and
        the broadening applied to sharp spectral families.
    beta_range
        Log-uniform range for the inverse temperature.
    dtau_choices
        Time steps to draw from; together with :math:`\beta` these set
        :math:`L_\tau = \beta/\Delta\tau + 1`.
    max_ntau
        Cap on the number of time slices, for memory.
    noise_scale_range
        Log-uniform range for the relative noise amplitude.  The wide default
        spans a quick exploratory run through a long production one, so a single
        trained model covers both instead of being implicitly tuned to one.
    """

    def __init__(
        self,
        omega_c: float = 20.0,
        gamma_min: float = 0.1 * math.pi,
        beta_range: tuple[float, float] = (2.0, 20.0),
        dtau_choices: tuple[float, ...] = (0.05, 0.1, 0.125, 0.2, 0.25),
        max_ntau: int = 401,
        noise_scale_range: tuple[float, float] = (3e-5, 3e-2),
        noise_xi_range: tuple[float, float] = (0.01, 0.4),
        families: list[str] | None = None,
        omega_bandwidth: tuple[float, float] = (1.0, 12.0),
        quad_nodes_per_width: float = 3.5,
        quad_tail_nodes: int = 32,
        cache_limit: int = 4,
        ph_symmetric: bool = False,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.omega_c = float(omega_c)
        self.gamma_min = float(gamma_min)
        self.beta_range = beta_range
        self.dtau_choices = tuple(dtau_choices)
        self.max_ntau = int(max_ntau)
        self.noise_scale_range = noise_scale_range
        self.noise_xi_range = noise_xi_range
        self.families = families
        self.omega_bandwidth = omega_bandwidth
        self.quad_h = float(gamma_min) / float(quad_nodes_per_width)
        self.quad_tail_nodes = int(quad_tail_nodes)
        self.cache_limit = max(int(cache_limit), 1)
        # Particle-hole symmetrise every target: A(nu) -> [A(nu) + A(-nu)]/2,
        # exact through the quadrature's reflection permutation.  For training
        # a ph_symmetric decoder: the generic families are asymmetric on
        # purpose, and a mirrored decoder cannot fit what it structurally
        # excludes.
        self.ph_symmetric = bool(ph_symmetric)
        self.device = device
        self.dtype = dtype
        self._quad_cache: dict[float, SpectralQuadrature] = {}
        # Scalar/discrete draws live here.  Keeping them off the accelerator
        # removes ~9 device synchronisations per batch, each of which also
        # forced the queued model compute to finish before the next step
        # could be built -- worth ~half the wall time of a training step.
        self._cpu_rng = torch.Generator()

    def quadrature(self, beta: float) -> SpectralQuadrature:
        """Quadrature rule for one beta, with a **bounded** cache.

        beta is drawn continuously, so almost every call is a miss; an unbounded
        cache would therefore grow without limit over a long run while never
        being hit.  The small cache still pays for itself when the same beta is
        reused (evaluation, single-cell sweeps).
        """
        key = round(float(beta), 6)
        q = self._quad_cache.get(key)
        if q is None:
            q = SpectralQuadrature(nu_c=beta * self.omega_c, h=self.quad_h,
                                   n_tail=self.quad_tail_nodes, dtype=self.dtype)
            if self.device is not None:
                q = q.to(self.device)
            self._quad_cache[key] = q
            while len(self._quad_cache) > self.cache_limit:
                self._quad_cache.pop(next(iter(self._quad_cache)))
        return q

    def seed_scalars(self, seed: int) -> None:
        """Seed the CPU stream used for beta, dtau, family and noise parameters."""
        self._cpu_rng.manual_seed(int(seed))

    @contextlib.contextmanager
    def fixed_scalars(self, seed: int):
        """Freeze the scalar stream at ``seed``; restore it on exit.

        Evaluation must draw the *same* sequence of (beta, dtau, family, noise
        model) every time it runs, or the eval curve mixes model improvement
        with whichever cells happened to be drawn -- chi^2 can then "improve"
        3x between two evaluations of the same weights.  Wrapping the eval loop
        in this context makes held-out batches genuinely fixed while leaving
        the training stream exactly where it was.
        """
        state = self._cpu_rng.get_state()
        self._cpu_rng.manual_seed(int(seed))
        try:
            yield self
        finally:
            self._cpu_rng.set_state(state)

    def sample_grid(self, rng: torch.Generator) -> tuple[float, torch.Tensor]:
        r"""Draw :math:`(\beta, x)` for one cell."""
        lo, hi = (math.log(b) for b in self.beta_range)
        u = torch.rand((), generator=self._cpu_rng).item()
        beta = math.exp(lo + u * (hi - lo))
        j = int(torch.randint(len(self.dtau_choices), (1,),
                              generator=self._cpu_rng).item())
        dtau = self.dtau_choices[j]
        n_tau = min(int(round(beta / dtau)) + 1, self.max_ntau)
        n_tau = max(n_tau, 9)
        x = torch.linspace(0.0, 1.0, n_tau, device=self.device, dtype=self.dtype)
        return beta, x

    def batch(self, batch_size: int, rng: torch.Generator) -> TrainingBatch:
        """Draw one cell: shared grid and noise process, independent spectra."""
        beta, x = self.sample_grid(rng)
        quad = self.quadrature(beta)

        spec: SpectrumBatch = sample_spectra(
            quad, beta, batch_size, rng, families=self.families,
            omega_bandwidth=self.omega_bandwidth, gamma_min=self.gamma_min,
            device=self.device, dtype=self.dtype, crng=self._cpu_rng,
        )
        A_true = spec.A
        poles_true = spec.poles
        if self.ph_symmetric:
            pi = quad.reflection_index().to(A_true.device)
            A_true = 0.5 * (A_true + A_true[:, pi])
            poles_true = None            # the labels no longer describe A_true

        K = PoleGreens(quad).kernel_matrix(x)                       # (M, L)
        G_clean = (A_true * quad.weights.to(A_true.dtype)) @ K       # (B, L)

        model = NoiseModel.sample(self._cpu_rng,
                                  scale_range=self.noise_scale_range,
                                  xi_range=self.noise_xi_range)
        kind = "exp" if torch.rand((), generator=self._cpu_rng).item() < 0.5 \
            else "gauss"
        noise1, sigma, R = sample_correlated_noise(G_clean, x, model, rng, kind=kind)
        noise2, _, _ = sample_correlated_noise(G_clean, x, model, rng, kind=kind)

        return TrainingBatch(
            G=G_clean + noise1,
            G_alt=G_clean + noise2,
            G_clean=G_clean,
            sigma=sigma,
            inv_sqrt_R=whitener(R),
            x=x,
            beta=beta,
            A_true=A_true,
            quad=quad,
            family=spec.family,
            poles_true=poles_true,
            meta={"noise": model, "correlation": kind,
                  "ph_symmetric": self.ph_symmetric, **spec.meta},
        )
