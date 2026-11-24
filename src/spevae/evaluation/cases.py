r"""The frozen evaluation cases: truths, noise draws and the real-data cell.

Every judge script used to rebuild its own copy of the probe truths, with
small differences in construction between copies.  This module is now the
single definition.  The constructions reproduce the step-1/step-2 judge
scripts exactly (same float paths), so the suite's seed-7 rows can be
checked against the banked numbers in ``out/``.

Conventions (identical to the judges):

- Synthetic cells use :math:`\beta = 8` and an endpoint grid of 65 points,
  :math:`x = \tau/\beta = \mathrm{linspace}(0, 1, 65)`.
- Truth pole sets are dimensionless (:math:`\nu = \beta\omega`); spectra live
  on the quadrature nodes of :meth:`SpectralQuadrature.for_beta`.
- Noise is correlated with correlation length :math:`\xi = 0.1`, drawn by
  :func:`sample_correlated_noise` from ``torch.Generator().manual_seed(seed)``.
  Seed 7 is the banked draw; seeds come in pairs (7, 8), (9, 10), … so the
  paired-noise consistency of a method can be measured on every pair.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

import numpy as np
import torch

from ..data.label_bank import GAMMA_MIN, SmoothTruth
from ..data.noise import NoiseModel, sample_correlated_noise
from ..physics import PoleGreens, SpectralQuadrature
from ..physics.poles import PoleSet

__all__ = [
    "BETA", "NTAU", "XI", "LEGACY_SEED", "CASES", "Case", "Truth", "Instance",
    "realize", "draw", "seeds_for", "holstein_cell", "HOLSTEIN_FIXTURE",
]

BETA = 8.0
NTAU = 64
XI = 0.1
LEGACY_SEED = 7          # the single noise draw every banked number used


# --------------------------------------------------------------- truths --- #
@dataclass(frozen=True)
class PoleSpec:
    """A pole-closed truth in physical units; ``t`` sets the asymmetry b."""

    eps: tuple
    gam: tuple
    a: tuple
    t: tuple | None = None          # None: b = 0 exactly (step-1 b-free probes)


@dataclass(frozen=True)
class SmoothSpec:
    """A smooth truth: a label-bank family and its parameters."""

    family: str
    params: tuple                   # sorted (key, value) pairs, hashable


@dataclass(frozen=True)
class Case:
    """One frozen truth with the tags used to stratify the scorecard."""

    name: str
    spec: PoleSpec | SmoothSpec
    tags: tuple = ()
    n_true: int | None = None


def _ladder(e0, Om, m, g, d):
    e = [e0 + k * Om for k in range(m)]
    a = [d ** k for k in range(m)]
    s = sum(a)
    return e, [g] * m, [v / s for v in a]


def _mirrored_ladder():
    # judge_arm.py: the 2-rung ladder (1.2, 1.0, gamma 0.30, d 0.45) and its mirror
    em, gm, am = _ladder(1.2, 1.0, 2, 0.30, 0.45)
    e = [-v for v in reversed(em)] + em
    g = list(reversed(gm)) + gm
    a = [v / 2 for v in reversed(am)] + [v / 2 for v in am]
    return e, g, a


def _smooth(family, **params):
    return SmoothSpec(family, tuple(sorted(params.items())))


_KS = np.linspace(-1, 1, 5)
_E3, _G3, _A3 = _ladder(0.5, 1.0, 3, 0.30, 0.55)
_E4, _G4, _A4 = _ladder(-0.4, 0.8, 4, 0.25, 0.60)
_EM, _GM, _AM = _mirrored_ladder()

#: The frozen truth set (suite v1).  Names match the banked JSON keys.
CASES: dict[str, Case] = {c.name: c for c in [
    # step-1 pole-closed probes (judge_arm.py), b = 0
    Case("3-rung ladder", PoleSpec(tuple(_E3), tuple(_G3), tuple(_A3), (0.0,) * 3),
         ("pole-closed", "ladder"), 3),
    Case("4-rung ladder", PoleSpec(tuple(_E4), tuple(_G4), tuple(_A4), (0.0,) * 4),
         ("pole-closed", "ladder"), 4),
    Case("mirrored ladder", PoleSpec(tuple(_EM), tuple(_GM), tuple(_AM), None),
         ("pole-closed", "ladder", "mirrored"), 4),
    Case("edged band", PoleSpec(tuple((0.3 + _KS * 1.4).tolist()),
                                (0.12, 0.45, 0.45, 0.45, 0.12), (0.2,) * 5, (0.0,) * 5),
         ("pole-closed", "band"), 5),
    Case("irregular triplet", PoleSpec((0.4, 1.1, 2.6), (0.35, 0.25, 0.45),
                                       (0.5, 0.3, 0.2), (0.0,) * 3),
         ("pole-closed",), 3),
    # b-channel probes (asym_judge.py)
    Case("asym triplet", PoleSpec((0.4, 1.1, 2.6), (0.35, 0.25, 0.45),
                                  (0.5, 0.3, 0.2), (0.35, -0.4, -0.25)),
         ("pole-closed", "b-channel"), 3),
    Case("fano band", PoleSpec(tuple((0.3 + _KS * 1.2).tolist()), (0.5,) * 5, (0.2,) * 5,
                               tuple(np.clip(0.7 * _KS, -0.45, 0.45).tolist())),
         ("pole-closed", "b-channel", "band"), 5),
    # reconstruction gallery (recon_gallery.py)
    Case("lorentz pair", PoleSpec((-1.8, 1.4), (0.4, 0.6), (0.55, 0.45), (0.2, -0.2)),
         ("pole-closed", "b-channel", "two-sided"), 2),
    Case("4-pole ladder", PoleSpec((0.5, 1.6, 2.7, 3.8), (0.35,) * 4,
                                   (0.45, 0.28, 0.17, 0.10), (0.0,) * 4),
         ("pole-closed", "ladder"), 4),
    # smooth truths (label-bank families; in-distribution once step 2 trains on them)
    Case("gauss pair", _smooth("gauss_mixture", weights=(0.6, 0.4), mus=(-1.2, 1.8),
                               sigmas=(0.5, 0.9), W=4.0), ("smooth",)),
    Case("gapped pair", _smooth("gapped_pair", weights=(0.45, 0.55), mus=(-2.2, 2.2),
                                sigmas=(0.4, 0.4), W=4.0), ("smooth", "gapped")),
    Case("semicircle", _smooth("semicircle", centre=0.3, radius=4.0,
                               sigma_res=GAMMA_MIN / BETA, W=4.0), ("smooth", "band")),
    Case("box", _smooth("box", lo=-3.0, hi=4.0, sigma_res=GAMMA_MIN / BETA, W=4.0),
         ("smooth", "band")),
    Case("pseudogap", _smooth("pseudogap", weights=(1.2062, -0.2062), mus=(0.1, 0.1),
                              sigmas=(2.0, 0.4851), W=4.0), ("smooth", "gapped")),
]}


@dataclass
class Truth:
    """A realised truth: spectrum on the quadrature nodes and clean data."""

    case: Case
    beta: float
    x: torch.Tensor                 # (L,)
    quad: SpectralQuadrature
    A: torch.Tensor                 # (1, M) dimensionless, on quad.nodes
    G_clean: torch.Tensor           # (1, L)
    poles: PoleSet | None           # dimensionless; None for smooth truths


@dataclass
class Instance:
    """One noisy dataset drawn from a truth."""

    truth: Truth
    noise: float
    seed: int
    G: torch.Tensor                 # (1, L)
    sigma: torch.Tensor             # (1, L)
    R: torch.Tensor                 # (L, L)
    isr: torch.Tensor               # (L, L) inverse square root of R

    @property
    def key(self) -> str:
        return f"{self.truth.case.name}@{self.noise:g}#{self.seed}"


@lru_cache(maxsize=4)
def _quad(beta: float) -> SpectralQuadrature:
    return SpectralQuadrature.for_beta(beta)


def realize(case: Case | str, beta: float = BETA, ntau: int = NTAU) -> Truth:
    """Build the truth exactly as the judge scripts did."""
    if isinstance(case, str):
        case = CASES[case]
    quad = _quad(beta)
    x = torch.linspace(0.0, 1.0, ntau + 1)
    spec = case.spec
    if isinstance(spec, PoleSpec):
        if spec.t is None:
            b = torch.zeros(1, len(spec.eps))
        else:
            a_, t_ = np.array(spec.a, float), np.array(spec.t, float)
            b = torch.tensor([(a_ * (t_ - (a_ * t_).sum() / a_.sum())).tolist()]).float()
        ps = PoleSet(eps=torch.tensor([list(spec.eps)]) * beta,
                     gamma=torch.tensor([list(spec.gam)]) * beta,
                     a=torch.tensor([list(spec.a)]).float(), b=b)
        return Truth(case, beta, x, quad, ps.spectral(quad.nodes),
                     PoleGreens(quad)(ps, x), ps)
    params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.params}
    tr = SmoothTruth(spec.family, beta, params)
    A = (torch.tensor(tr.A(quad.nodes.numpy() / beta)) / beta).float()
    K = PoleGreens(quad).kernel_matrix(x)
    return Truth(case, beta, x, quad, A.unsqueeze(0),
                 ((A * quad.weights) @ K).unsqueeze(0), None)


def draw(truth: Truth, noise: float, seed: int) -> Instance:
    """One correlated-noise draw, identical to the judges' noise path."""
    n, sigma, R = sample_correlated_noise(truth.G_clean, truth.x,
                                          NoiseModel(scale=noise, xi=XI),
                                          torch.Generator().manual_seed(seed))
    ev, V = torch.linalg.eigh(R)
    isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    return Instance(truth, noise, seed, (truth.G_clean + n).reshape(1, -1),
                    sigma.reshape(1, -1), R, isr)


def seeds_for(n_seeds: int) -> list[int]:
    """The suite's noise seeds: the banked seed first, then consecutive pairs."""
    return [LEGACY_SEED + k for k in range(n_seeds)]


# ------------------------------------------------------- real-data cell --- #
#: The Holstein judge cell (beta 8, Omega 1, n 1, lambda 0.325) extracted from
#: the gIssa cube by scripts/eval/build_holstein_fixture.py.  The sha256 pins
#: the bins, so a rebuilt or edited fixture cannot silently change the judge.
HOLSTEIN_FIXTURE = {
    "file": "holstein_b8_w1_n1_l0325.npz",
    "label": "gissa lam=0.325 n=1 Omega=1 beta=8",
    "bins_sha256": "f92de452f5f10e2a984357b9709aca480f6a966c4d8f14cf09e655779c34b169",
}


def _bins_digest(bins: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bins, dtype=np.float64).tobytes()).hexdigest()


def holstein_cell(path: str | None = None):
    """The real-data judge cell as a :class:`MeasuredGreens`, from the fixture.

    Built through :func:`from_bins` with the same arguments as
    :func:`read_gissa_cube`, so it is the identical object the banked
    Holstein numbers were computed on -- without the 9.9 GB cube.
    """
    from ..data.real import from_bins

    if path is None:
        path = resources.files("spevae.evaluation") / "fixtures" / HOLSTEIN_FIXTURE["file"]
    with np.load(path, allow_pickle=False) as f:
        bins = f["bins"]
        ref = (f["ref_omega"], f["ref_A"]) if "ref_A" in f else None
        meta = {k[5:]: f[k].item() for k in f.files if k.startswith("meta_")}
    want = HOLSTEIN_FIXTURE["bins_sha256"]
    if want is not None and _bins_digest(bins) != want:
        raise ValueError(f"Holstein fixture {path} does not match the pinned digest")
    mg = from_bins(bins, float(meta["beta"]), includes_endpoints=True,
                   label=HOLSTEIN_FIXTURE["label"], reference=ref, dtype=torch.float64)
    mg.meta.update(meta)
    return mg


def truth_on_grid(case: Case | str, omega: np.ndarray, beta: float = BETA) -> np.ndarray:
    """Physical A_true(omega) on any grid, for figures."""
    if isinstance(case, str):
        case = CASES[case]
    spec = case.spec
    if isinstance(spec, PoleSpec):
        tr = realize(case, beta)
        nu = torch.tensor(np.asarray(omega) * beta, dtype=torch.float32)
        return (tr.poles.spectral(nu)[0] * beta).double().numpy()
    params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.params}
    return np.asarray(SmoothTruth(spec.family, beta, params).A(np.asarray(omega, float)))
