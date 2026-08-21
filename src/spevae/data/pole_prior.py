r"""Pole-first training data: exact, canonical labels by construction.

The truth is *sampled as poles* -- hierarchical motifs mirroring physical
spectra -- so :math:`A(\omega)` is a closed form, :math:`G(\tau)` comes from
the verified forward map, and the label lives exactly in the model's output
space: zero training error is achievable, and every residual is pure
optimization signal.

Motif hierarchy
---------------
Level 1 (global): number of features, bandwidth, particle-hole symmetry flag.
Level 2 (features): centre, half-width, Dirichlet weight, and a motif type:
``lorentz`` (one pole), ``smooth_band`` (overlapping cluster),
``edged_band`` (the measured semicircle structure: narrow poles at the band
edges, broad in the body), ``fano`` (strong structured asymmetry),
``ladder`` (satellite replicas).
Level 3 (poles): the cluster realisation, governed by the spacing-to-width
knob :math:`\rho`: :math:`\rho \lesssim 1.5` blends into a smooth envelope,
:math:`\rho \gtrsim 3` is genuinely resolved structure -- sampling the range
teaches the comb/no-comb boundary itself.

Label hygiene (what makes ``P_true`` well-defined): labels are
**canonicalized** (sub-width pairs merged -- minimal pole representations are
unique, so after merging the label is *the* answer) and **certified**
(:math:`A \ge 0` checked exactly; asymmetries shrunk on the rare violation).
"""

from __future__ import annotations

import contextlib
import math

import torch

from ..physics.forward import PoleGreens, SpectralQuadrature
from ..physics.poles import PoleSet
from ..physics.positivity import certify
from .noise import NoiseModel, sample_correlated_noise, whitener
from .synthetic import TrainingBatch

__all__ = ["PolePriorGenerator", "MixedGenerator"]

MOTIFS = ("lorentz", "smooth_band", "edged_band", "fano", "ladder")


class PolePriorGenerator:
    """Drop-in sibling of :class:`SyntheticGenerator` emitting pole labels.

    ``batch()`` returns a :class:`TrainingBatch` whose ``poles_true`` is a
    dict of dimensionless padded tensors ``eps/gamma/a/b (B, P_max)`` plus
    ``n_true (B,)``; padding entries carry exactly zero weight.  ``A_true``
    is evaluated from the labels, so the legacy spectral evaluation keeps
    working unchanged.
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
        P_max: int = 12,
        quad_nodes_per_width: float = 3.5,
        quad_tail_nodes: int = 32,
        cache_limit: int = 4,
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
        self.P_max = int(P_max)
        self.quad_h = float(gamma_min) / float(quad_nodes_per_width)
        self.quad_tail_nodes = int(quad_tail_nodes)
        self.cache_limit = max(int(cache_limit), 1)
        self.device = device
        self.dtype = dtype
        self._quad_cache: dict[float, SpectralQuadrature] = {}
        self._cpu_rng = torch.Generator()

    # -- rng plumbing (mirrors SyntheticGenerator) -------------------------- #

    def seed_scalars(self, seed: int) -> None:
        self._cpu_rng.manual_seed(int(seed))

    @contextlib.contextmanager
    def fixed_scalars(self, seed: int):
        state = self._cpu_rng.get_state()
        self._cpu_rng.manual_seed(int(seed))
        try:
            yield self
        finally:
            self._cpu_rng.set_state(state)

    def _u(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand((), generator=self._cpu_rng))

    def _i(self, n: int) -> int:
        return int(torch.randint(n, (1,), generator=self._cpu_rng))

    # -- one pole configuration (physical units) ---------------------------- #

    def _feature(self, motif: str, centre: float, half: float, floor: float):
        """Return (eps, gamma, a_rel, t) lists for one feature; a_rel sums 1."""
        # spacing / width knob, quadratically biased toward the smooth end:
        # this distribution IS the explicit smoothness prior -- uniform rho
        # said "resolved bumps as likely as smooth bands", which is not the
        # physical expectation.  median ~1.5, p90 ~3.0.
        u = self._u(0.0, 1.0)
        rho = 0.8 + 2.7 * u * u
        if motif == "lorentz":
            g = max(self._u(0.5, 2.0) * half, floor)
            return [centre], [g], [1.0], [self._u(-0.4, 0.4)]
        if motif == "ladder":
            m = 2 + self._i(3)
            Om = self._u(0.6, 1.6) * half
            g = max(self._u(0.3, 0.8) * Om / rho, floor)
            eps = [centre + k * Om for k in range(m)]
            a = [self._u(0.5, 1.0) * (0.55 ** k) for k in range(m)]
            s = sum(a)
            return eps, [g] * m, [v / s for v in a], [0.0] * m
        m = 2 + self._i(4)                          # smooth/edged/fano: 2..5
        span = 2.0 * half
        spacing = span / max(m - 1, 1)
        g_body = max(spacing / rho, floor)
        eps = [centre - half + k * spacing for k in range(m)]
        gam = [g_body] * m
        if motif == "edged_band" and m >= 3:
            g_edge = max(g_body * self._u(0.15, 0.5), floor)
            gam[0] = gam[-1] = g_edge
        a = [1.0 + self._u(-0.3, 0.3) for _ in range(m)]
        if motif == "smooth_band":                   # envelope-weighted
            a = [v * (0.4 + math.sin(math.pi * (k + 0.5) / m))
                 for k, v in enumerate(a)]
        s = sum(a)
        a = [v / s for v in a]
        if motif == "fano":
            slope = self._u(0.5, 0.9) * (1 if self._u(0, 1) < 0.5 else -1)
            t = [max(-0.45, min(0.45, slope * (2 * k / max(m - 1, 1) - 1)))
                 for k in range(m)]
        else:
            amp = self._u(0.0, 0.35)
            t = [max(-0.45, min(0.45, amp * (2 * self._u(0, 1) - 1)))
                 for _ in range(m)]
        return eps, gam, a, t

    def sample_poles(self, beta: float) -> tuple[list, list, list, list]:
        """One canonical, certified pole configuration in physical units."""
        floor = 1.05 * self.gamma_min / beta
        W = self._u(1.5, 0.45 * self.omega_c)        # half-bandwidth
        ph = self._u(0, 1) < 0.3
        K = 1 + self._i(3)
        eps, gam, a_rel, t = [], [], [], []
        weights = torch.zeros(K)                     # feature weights
        for k in range(K):
            weights[k] = self._u(0.2, 1.0)
        weights = weights / weights.sum()
        for k in range(K):
            motif = MOTIFS[self._i(len(MOTIFS))]
            centre = self._u(-0.8, 0.8) * W
            half = self._u(0.15, 0.6) * W / K + floor
            e, g, ar, tt = self._feature(motif, centre, half, floor)
            eps += e
            gam += g
            a_rel += [v * float(weights[k]) for v in ar]
            t += tt
        if ph:                                        # mirror everything
            eps = eps + [-v for v in eps]
            gam = gam + gam
            a_rel = [v / 2 for v in a_rel] * 2
            t = t + [-v for v in t]

        # truncate to P_max by weight, renormalise
        order = sorted(range(len(eps)), key=lambda i: -a_rel[i])[: self.P_max]
        eps = [eps[i] for i in order]
        gam = [gam[i] for i in order]
        a_rel = [a_rel[i] for i in order]
        t = [t[i] for i in order]
        s = sum(a_rel)
        a = [v / s for v in a_rel]

        # canonicalize: merge sub-width pairs so P_true is the unique minimal
        # count (weighted merge; iterate to a fixed point)
        changed = True
        while changed and len(eps) > 1:
            changed = False
            for i in range(len(eps) - 1):
                for j in range(i + 1, len(eps)):
                    if abs(eps[i] - eps[j]) < 0.45 * min(gam[i], gam[j]):
                        wsum = a[i] + a[j]
                        eps[i] = (a[i] * eps[i] + a[j] * eps[j]) / wsum
                        gam[i] = (a[i] * gam[i] + a[j] * gam[j]) / wsum
                        t[i] = (a[i] * t[i] + a[j] * t[j]) / wsum
                        a[i] = wsum
                        for lst in (eps, gam, a, t):
                            lst.pop(j)
                        changed = True
                        break
                if changed:
                    break

        # residues: b = a (t - <t>_a), certified positive (shrink b on the
        # rare violation -- exact certificate, no grid)
        for shrink in range(4):
            tm = sum(ai * ti for ai, ti in zip(a, t))
            b = [ai * (ti - tm) for ai, ti in zip(a, t)]
            rep = certify(eps, gam, a, b, atol=1e-9)
            if rep.is_positive or all(abs(v) < 1e-12 for v in b):
                break
            t = [0.5 * v for v in t]
        else:
            b = [0.0] * len(a)
        return eps, gam, a, b

    # -- batches ------------------------------------------------------------ #

    def quadrature(self, beta: float) -> SpectralQuadrature:
        key = round(float(beta), 9)
        q = self._quad_cache.get(key)
        if q is None:
            q = SpectralQuadrature(nu_c=float(beta) * self.omega_c,
                                   h=self.quad_h,
                                   n_tail=self.quad_tail_nodes,
                                   dtype=self.dtype)
            if self.device is not None:
                q = q.to(self.device)
            self._quad_cache[key] = q
            while len(self._quad_cache) > self.cache_limit:
                self._quad_cache.pop(next(iter(self._quad_cache)))
        return q

    def batch(self, batch_size: int, rng: torch.Generator) -> TrainingBatch:
        lo, hi = (math.log(b) for b in self.beta_range)
        beta = math.exp(self._u(lo, hi))
        dtau = self.dtau_choices[self._i(len(self.dtau_choices))]
        n_tau = max(min(int(round(beta / dtau)) + 1, self.max_ntau), 9)
        x = torch.linspace(0.0, 1.0, n_tau, device=self.device,
                           dtype=self.dtype)
        quad = self.quadrature(beta)

        P = self.P_max
        lab = {k: torch.zeros(batch_size, P) for k in ("eps", "gamma", "a", "b")}
        lab["gamma"] += 1.0                          # inert padding widths
        n_true = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            e, g, a, b = self.sample_poles(beta)
            m = len(e)
            n_true[i] = m
            lab["eps"][i, :m] = torch.tensor(e) * beta      # dimensionless
            lab["gamma"][i, :m] = torch.tensor(g) * beta
            lab["a"][i, :m] = torch.tensor(a)
            lab["b"][i, :m] = torch.tensor(b)
        lab = {k: v.to(self.dtype).to(self.device) if self.device is not None
               else v.to(self.dtype) for k, v in lab.items()}
        lab["n_true"] = n_true

        truth = PoleSet(eps=lab["eps"], gamma=lab["gamma"],
                        a=lab["a"], b=lab["b"])
        A_true = truth.spectral(quad.nodes)
        K = PoleGreens(quad).kernel_matrix(x)
        G_clean = (A_true * quad.weights.to(A_true.dtype)) @ K

        model = NoiseModel.sample(self._cpu_rng,
                                  scale_range=self.noise_scale_range,
                                  xi_range=self.noise_xi_range)
        kind = "exp" if float(torch.rand((), generator=self._cpu_rng)) < 0.5 \
            else "gauss"
        noise1, sigma, R = sample_correlated_noise(G_clean, x, model, rng,
                                                   kind=kind)
        noise2, _, _ = sample_correlated_noise(G_clean, x, model, rng,
                                               kind=kind)
        return TrainingBatch(
            G=G_clean + noise1, G_alt=G_clean + noise2, G_clean=G_clean,
            sigma=sigma, inv_sqrt_R=whitener(R), x=x, beta=beta,
            A_true=A_true, quad=quad,
            family=["pole_mixture"] * batch_size,
            poles_true=lab,
            meta={"noise": model, "correlation": kind, "pole_labeled": True},
        )


class MixedGenerator:
    """Interleave pole-labeled and family batches (coverage + labels).

    Each ``batch()`` call flips a coin from its own scalar stream: pole-prior
    batches carry exact labels and take the pole-space losses; family batches
    keep the legacy spectral supervision.  ``fixed_scalars`` freezes all three
    streams, so held-out evaluation stays bit-reproducible.
    """

    def __init__(self, pole_gen, family_gen, p_pole: float = 0.5,
                 p_pole_final: float | None = None) -> None:
        self.pole_gen = pole_gen
        self.family_gen = family_gen
        self.p_pole = float(p_pole)
        # A linear curriculum: the coin bias moves from p_pole to
        # p_pole_final as set_progress() advances.  Held-out evaluation must
        # stay comparable across evals (and across arms), so under
        # fixed_scalars the bias is pinned to 0.5 regardless of schedule.
        self.p_pole_final = None if p_pole_final is None else float(p_pole_final)
        self._progress = 0.0
        self._eval_frozen = False
        self._cpu_rng = torch.Generator()

    def set_progress(self, t: float) -> None:
        self._progress = min(max(float(t), 0.0), 1.0)

    def _p_eff(self) -> float:
        if self._eval_frozen:
            return 0.5
        if self.p_pole_final is None:
            return self.p_pole
        return self.p_pole + (self.p_pole_final - self.p_pole) * self._progress

    def seed_scalars(self, seed: int) -> None:
        self._cpu_rng.manual_seed(int(seed))
        self.pole_gen.seed_scalars(seed + 1)
        self.family_gen.seed_scalars(seed + 2)

    @contextlib.contextmanager
    def fixed_scalars(self, seed: int):
        state = self._cpu_rng.get_state()
        frozen = self._eval_frozen
        self._cpu_rng.manual_seed(int(seed))
        self._eval_frozen = True
        with self.pole_gen.fixed_scalars(seed + 1), \
                self.family_gen.fixed_scalars(seed + 2):
            try:
                yield self
            finally:
                self._cpu_rng.set_state(state)
                self._eval_frozen = frozen

    def batch(self, batch_size: int, rng: torch.Generator):
        coin = float(torch.rand((), generator=self._cpu_rng))
        gen = self.pole_gen if coin < self._p_eff() else self.family_gen
        return gen.batch(batch_size, rng)
