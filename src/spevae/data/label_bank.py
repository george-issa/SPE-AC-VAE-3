r"""Step-2 label bank: noise-independent MPM labels for smooth spectral families.

Why this exists
---------------
Step 1's measured deficiency is expressiveness: the decoder manifold cannot
express smooth, non-pole-closed spectra (Holstein manifold floor
:math:`\chi^2 = 24.0` against a target of :math:`0.175`).  Step 2 grows the
training prior by the smooth catalog -- Gaussian mixtures, gapped pairs,
semicircle, box, pseudogap -- and this module supplies their **labels**: for
each smooth truth, one fixed, canonical pole set in the model's output space.

Labels are **noise-independent** (user decision 2026-08-26, consistent with
step 1's exact labels and with real data where :math:`\sigma` is only
estimated): the extraction below runs at machine precision on the truth's
exact Matsubara samples and never sees a noise realisation.  The noise does
its teaching on the *input* side only -- the mass-weighted OT loss absorbs
unresolvable detail and the network learns :math:`\sigma`-calibrated hedging.

The canonical extraction protocol (THE definition of the label)
---------------------------------------------------------------
1. Exact Matsubara samples :math:`F(i\omega_n) = \int A(\omega)/(i\omega_n -
   \omega)\,d\omega` for :math:`n < 64`, from closed forms (Faddeeva
   :math:`w` for Gaussian components; principal square roots / logs for the
   semicircle and box cores, resolution-broadened by Gauss--Legendre
   quadrature of the exact convolution -- all verified to :math:`10^{-15}`
   against brute-force integration).
2. A rank ladder :math:`n = 1 \ldots P_{\max}`.  Each rung is a constrained
   nonlinear least-squares fit of an :math:`n`-pole expansion to the
   Matsubara samples, **parameterised exactly as the decoder's constraint
   layer**: :math:`a = \mathrm{softmax}(s)` (sum rule exact, and no mass can
   hide in a far-away ghost pole -- the failure mode of an unconstrained
   fit, observed at 4% of the total weight), :math:`b = a(t - \langle
   t\rangle_a)` with :math:`t = \tfrac12\tanh v` (zero-sum, bounded
   asymmetry), :math:`\gamma = \gamma_{\rm floor} + e^u` (width floor).  The
   label therefore lives in the model's reachable set *by construction*.
   The fit runs inside a hard parameter box (:data:`EPS_MAX`,
   :data:`GAM_MAX`): Matsubara data cannot price a pole far outside the
   sampled window, and an unbounded fit was measured parking 1e-3 of real
   spectral weight at :math:`|\varepsilon| \sim 10^{20}` -- invisible in
   the Matsubara residual, a 1e-3 error in :math:`G(0)`.  Rungs are seeded
   from an AAA sweep on the samples (per-rank causal configurations) and by
   splitting the previous rung's heaviest pole; plain AAA interpolation
   alone is NOT enough -- its stopping rule controls interpolation error,
   not the causal-pole-sum error, which was measured stuck at
   :math:`10^{-1}` for Gaussian tails at :math:`\beta \ge 8` while the
   polished ladder reaches :math:`10^{-8}`--:math:`10^{-13}`.  Every
   polish is a budgeted homotopy: a residual block compares label and
   truth spectra mollified at the resolution scale :math:`\pi/\beta`
   (the smoothness selector -- the Matsubara objective alone is blind in
   the kernel's null space and was measured returning spike combs,
   floor-width poles peaking 10x above a smooth truth at residual 1e-7),
   its weight descends :data:`W_LADDER`, and the label is the smoothest
   fit whose PURE Matsubara residual still meets the budget
   :data:`LABEL_RES_BUDGET` (or 1.5x the rank's own data floor when that
   is higher).  The recorded residual is always the pure data-fidelity
   number of the shipped configuration.
3. Rung selection: the smallest :math:`n` whose relative Matsubara residual
   is within 5% of the best rung (and never above
   :math:`\max(10^{-9}, \cdot)`) -- minimal rank at the achievable
   fidelity.  The ladder never exceeds :math:`P_{\max}` (capacity-matched
   truncation; measured residual at :math:`P_{\max}` is typically
   :math:`10^{-7}`--:math:`10^{-11}` across families and :math:`\beta \in
   [2, 20]`, well below the :math:`10^{-4}` roadmap bound; the exception
   is a truth whose features sit AT the resolution floor, where the
   plateau is floor-limited -- :math:`\sim 10^{-3}` at :math:`\beta = 2` --
   and the per-entry ``resid`` records exactly how far the output space
   can follow that truth).
4. Hygiene, as in the pole prior: sub-weight dust dropped, sub-width pairs
   merged when the merge costs no fidelity (a stacked narrow-core /
   broad-base pair at one centre can BE the minimal representation of a
   smooth shape), sum rules restored exactly, and the positivity
   certificate used to *bound* the dip: when negative weight exceeds 0.5%
   the label is re-polished under a tightened asymmetry cap
   (:math:`t_{\rm cap} = 0.25 \to 0.125 \to 0`, the last being a positive
   Lorentzian mixture, positive by construction) -- re-fitting, never
   post-hoc damage to :math:`b`, which was measured to cost 17x in
   residual.  A bound, not a ban; enforcement proper stays at inference.

Everything the extraction did ships with the label: the full rank ladder,
final residual, negative weight -- the bank's report is the pinning document
for the step-2 unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.special import erf, wofz

from ..physics.poles import PoleSet
from ..physics.positivity import certify
from ..physics.prony import aaa_poles

__all__ = [
    "SMOOTH_FAMILIES",
    "SmoothTruth",
    "SmoothLabel",
    "sample_smooth_truth",
    "extract_smooth_label",
    "LabelBank",
    "build_label_bank",
]

SMOOTH_FAMILIES = ("gauss_mixture", "gapped_pair", "semicircle", "box",
                   "pseudogap")

#: Dimensionless resolution floor, matching the decoder and the pole prior.
GAMMA_MIN = 0.1 * math.pi
#: Protocol constants.  These DEFINE the canonical label -- changing any of
#: them changes every label in every bank, so they are module constants, not
#: knobs.  N_MATSUBARA=64 puts the top sample at 127*pi/beta: ~omega_c at
#: beta=20, far above it at small beta.
N_MATSUBARA = 64
N_GL = 480              # Gauss-Legendre nodes; 240 was measured 2e-6 short
                        # of convergence at beta=20, 480 agrees with 960 to
                        # 1e-12
AAA_MAX_TERMS = 32
LM_MAX_NFEV = 200
RES_TARGET = 1e-9       # machine-defensible fidelity bar for rung selection
RES_FLOOR = 3e-13       # stop the ladder outright below this
DUST_TOL = 1e-8         # weights below this fraction are padding, not poles
MERGE_FRAC = 0.45       # sub-width merge criterion, as in the pole prior
DIP_TOL = 5e-3          # max negative weight fraction, as in the pole prior
#: Hard parameter box for the fit.  Matsubara data cannot see a pole at
#: |eps| >> omega_max (1/(z - eps) -> 0), so an unbounded fit can park real
#: spectral weight at infinity -- measured: a 1e-3-weight pole at |eps|~1e20,
#: invisible in the Matsubara residual (3e-7) but a 1e-3 error in G(0).
#: The box keeps every pole where the data can price it.
EPS_MAX = 25.0          # physical; omega_c = 20 plus margin
GAM_MAX = 20.0          # physical; a pole broader than the band is not physics
#: The smoothness budget.  The Matsubara objective alone is blind in the
#: kernel's null space, and among the many data-equivalent pole sets the
#: unregularized polish was MEASURED returning spike combs: floor-width
#: poles peaking 10x above a smooth truth at Matsubara residual 1e-7 (the
#: first rendered overlays caught it).  Every polish therefore carries a
#: residual block comparing the label's and the truth's spectra, both
#: mollified at the resolution scale sigma = SIGMA_MOLL/beta (the first
#: Matsubara frequency -- structure below it is unrecoverable, so the
#: block constrains exactly what the data cannot), and its weight is a
#: descending homotopy: the label is the SMOOTHEST fit whose pure
#: Matsubara residual still meets the budget -- max(1e-6, 1.5x the rank's
#: own data floor), 30x below the quietest training noise.  A fixed
#: weight cannot do this job: 0.5 alone was measured costing 1e-4--8e-3
#: in data fidelity, and a pure re-polish afterwards re-sharpened the
#: spikes it had removed.
SIGMA_MOLL = math.pi    # dimensionless mollification scale (nu units)
W_LADDER = (0.5, 0.05, 0.005)                # A-block homotopy weights
LABEL_RES_BUDGET = 1e-6                      # data-fidelity budget
A_GRID_N = 160          # points of the mollified comparison grid


# --------------------------------------------------------------------------- #
# smooth truths: exact A(omega) and exact Stieltjes transform
# --------------------------------------------------------------------------- #

_SQRT2 = math.sqrt(2.0)
_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl_nodes(n: int = N_GL) -> tuple[np.ndarray, np.ndarray]:
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def _a_gauss(w, mu, sigma):
    return np.exp(-0.5 * ((w - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def _s_gauss(z, mu, sigma):
    r"""Stieltjes transform of a unit Gaussian, exact for Im z > 0.

    :math:`S(z) = -i\sqrt{\pi}\,w(\zeta)/(\sqrt2\sigma)`,
    :math:`\zeta = (z-\mu)/(\sqrt2\sigma)`; the :math:`\sigma \to 0` limit is
    :math:`1/(z-\mu)`, which fixes the prefactor.
    """
    zeta = (z - mu) / (_SQRT2 * sigma)
    return -1j * math.sqrt(math.pi) / (_SQRT2 * sigma) * wofz(zeta)


@dataclass
class SmoothTruth:
    """One smooth spectral truth: family + parameters + beta.

    Internally reduced to components of three kinds -- signed Gaussians
    (closed forms), and resolution-broadened semicircle / box cores (exact
    convolution integrals by Gauss--Legendre; the semicircle's edge
    singularity is absorbed by the :math:`u = D\sin\theta` substitution, so
    both integrands are analytic and the quadrature converges spectrally).
    ``params`` is the full record: the truth is reconstructible from
    ``(family, beta, params)`` alone, which is what the bank stores.
    """

    family: str
    beta: float
    params: dict

    def _components(self):
        p = self.params
        if self.family in ("gauss_mixture", "gapped_pair", "pseudogap"):
            return [("gauss", w, mu, s)
                    for w, mu, s in zip(p["weights"], p["mus"], p["sigmas"])]
        if self.family == "semicircle":
            return [("semicircle", 1.0, p["centre"], p["radius"], p["sigma_res"])]
        if self.family == "box":
            return [("box", 1.0, p["lo"], p["hi"], p["sigma_res"])]
        raise ValueError(f"unknown family {self.family!r}")

    def A(self, omega: np.ndarray) -> np.ndarray:
        """Exact :math:`A(\\omega)` in physical units."""
        omega = np.asarray(omega, float)
        out = np.zeros_like(omega)
        for comp in self._components():
            if comp[0] == "gauss":
                _, w, mu, s = comp
                out = out + w * _a_gauss(omega, mu, s)
            elif comp[0] == "semicircle":
                _, w, c, D, sres = comp
                # On the REAL axis the integrand is a width-sres Gaussian
                # sweeping the support, so the node count must track
                # D/sres; the fixed N_GL rule is fine for the off-axis
                # transforms (their scale is pi/beta) but was measured
                # aliasing the beta ~ 20 truth renderings into a band of
                # oscillation.
                n_gl = min(max(N_GL, int(10.0 * D / sres)), 20_000)
                t, wt = _gl_nodes(n_gl)
                th = 0.5 * math.pi * t
                u = c + D * np.sin(th)
                integ = np.cos(th) ** 2 * _a_gauss(omega[..., None], u, sres)
                out = out + w * (integ * wt).sum(-1)   # (2/pi)*(pi/2) jacobian = 1
            else:
                _, w, lo, hi, sres = comp
                phi = lambda x: 0.5 * (1.0 + erf(x / _SQRT2))
                out = out + w * (phi((hi - omega) / sres)
                                 - phi((lo - omega) / sres)) / (hi - lo)
        return out

    def stieltjes(self, z: np.ndarray) -> np.ndarray:
        r"""Exact :math:`S(z) = \int A(\omega)/(z-\omega)\,d\omega`, Im z > 0."""
        z = np.asarray(z, complex)
        out = np.zeros_like(z)
        t, wt = _gl_nodes()
        for comp in self._components():
            if comp[0] == "gauss":
                _, w, mu, s = comp
                out = out + w * _s_gauss(z, mu, s)
            elif comp[0] == "semicircle":
                _, w, c, D, sres = comp
                th = 0.5 * math.pi * t
                u = c + D * np.sin(th)
                integ = np.cos(th) ** 2 * _s_gauss(z[..., None], u, sres)
                out = out + w * (integ * wt).sum(-1)
            else:
                _, w, lo, hi, sres = comp
                u = 0.5 * (hi + lo) + 0.5 * (hi - lo) * t
                out = out + w * 0.5 * (_s_gauss(z[..., None], u, sres)
                                       * wt).sum(-1)
        return out

    def matsubara(self, n_w: int = N_MATSUBARA) -> tuple[np.ndarray, np.ndarray]:
        """The extraction's data: :math:`(\\omega_n, F(i\\omega_n))`, exact."""
        om = (2.0 * np.arange(n_w) + 1.0) * math.pi / self.beta
        return om, self.stieltjes(1j * om)

    def mollified_A(self, omega: np.ndarray, sigma: float) -> np.ndarray:
        """:math:`(A * N_\\sigma)(\\omega)`, exact: Gaussian components widen
        in quadrature; the broadened cores widen their resolution kernel."""
        omega = np.asarray(omega, float)
        out = np.zeros_like(omega)
        for comp in self._components():
            if comp[0] == "gauss":
                _, w, mu, s = comp
                out = out + w * _a_gauss(omega, mu, math.hypot(s, sigma))
            elif comp[0] == "semicircle":
                _, w, c, D, sres = comp
                s_eff = math.hypot(sres, sigma)
                n_gl = min(max(N_GL, int(10.0 * D / s_eff)), 20_000)
                t, wt = _gl_nodes(n_gl)
                th = 0.5 * math.pi * t
                u = c + D * np.sin(th)
                integ = np.cos(th) ** 2 * _a_gauss(omega[..., None], u, s_eff)
                out = out + w * (integ * wt).sum(-1)
            else:
                _, w, lo, hi, sres = comp
                s_eff = math.hypot(sres, sigma)
                phi = lambda x: 0.5 * (1.0 + erf(x / _SQRT2))
                out = out + w * (phi((hi - omega) / s_eff)
                                 - phi((lo - omega) / s_eff)) / (hi - lo)
        return out

    def support_bound(self) -> float:
        """|omega| beyond which the truth is negligible (for grids)."""
        L = 0.0
        for comp in self._components():
            if comp[0] == "gauss":
                _, _, mu, s = comp
                L = max(L, abs(mu) + 5.0 * s)
            elif comp[0] == "semicircle":
                _, _, c, D, sres = comp
                L = max(L, abs(c) + D + 5.0 * sres)
            else:
                _, _, lo, hi, sres = comp
                L = max(L, max(abs(lo), abs(hi)) + 5.0 * sres)
        return L


# --------------------------------------------------------------------------- #
# family samplers (physical units, physics-scaled ranges, K <= 3, mirroring)
# --------------------------------------------------------------------------- #

def _mirror_gaussians(weights, mus, sigmas):
    w = [v / 2 for v in weights] * 2
    m = mus + [-v for v in mus]
    s = sigmas + sigmas
    return w, m, s


def sample_smooth_truth(
    family: str,
    beta: float,
    rng: torch.Generator,
    omega_bandwidth: tuple[float, float] = (1.0, 12.0),
    p_mirror: float = 0.3,
) -> SmoothTruth:
    """Draw one truth.  Ranges mirror :mod:`spevae.data.spectra`; component
    counts obey the K <= 3 mechanism vocabulary; every width is clamped to
    the resolution floor (structure finer than :math:`\\pi/\\beta` is not
    recoverable and teaching it teaches hallucination)."""

    def u(lo, hi):
        return lo + (hi - lo) * float(torch.rand((), generator=rng))

    def logu(lo, hi):
        return math.exp(u(math.log(lo), math.log(hi)))

    if family not in SMOOTH_FAMILIES:
        raise ValueError(f"unknown smooth family {family!r}")
    floor = 1.05 * GAMMA_MIN / beta
    W = u(*omega_bandwidth)

    if family == "gauss_mixture":
        C = 1 + int(torch.randint(3, (1,), generator=rng))
        mus = [u(-0.8, 0.8) * W for _ in range(C)]
        sigmas = [max(logu(0.08, 0.5) * W, floor) for _ in range(C)]
        raw = [u(0.2, 1.0) for _ in range(C)]
        weights = [v / sum(raw) for v in raw]
        if u(0.0, 1.0) < p_mirror:
            weights, mus, sigmas = _mirror_gaussians(weights, mus, sigmas)
        return SmoothTruth(family, beta, {"weights": weights, "mus": mus,
                                          "sigmas": sigmas, "W": W})

    if family == "gapped_pair":
        centre = (u(0.1, 0.6) + u(0.15, 0.45)) * W      # gap/2 + lobe offset
        sig = max(logu(0.05, 0.25) * W, floor)
        w0 = 0.5 if u(0.0, 1.0) < p_mirror else u(0.25, 0.75)
        return SmoothTruth(family, beta, {
            "weights": [w0, 1.0 - w0], "mus": [-centre, centre],
            "sigmas": [sig, sig], "W": W})

    if family == "semicircle":
        return SmoothTruth(family, beta, {
            "centre": u(-0.15, 0.15) * W,
            "radius": u(0.4, 1.0) * W,
            "sigma_res": max(GAMMA_MIN / beta, 1e-3), "W": W})

    if family == "box":
        return SmoothTruth(family, beta, {
            "lo": u(-1.0, -0.1) * W, "hi": u(0.1, 1.0) * W,
            "sigma_res": max(GAMMA_MIN / beta, 1e-3), "W": W})

    # pseudogap: Gaussian background times (1 - c * gaussian dip at the Fermi
    # level).  The product of Gaussians is a Gaussian, so this is an exact
    # signed two-component mixture, positive pointwise for c < 1.
    sb = max(u(0.25, 0.7) * W, 3.0 * floor)
    sg = max(logu(0.1, 0.5) * sb, floor)
    c = u(0.3, 0.95)
    shift = u(-0.05, 0.05) * W
    sp = 1.0 / math.sqrt(1.0 / sb**2 + 1.0 / sg**2)
    k = (sp / sb)                                    # dip mass factor (mu_g = 0)
    norm = 1.0 - c * k
    return SmoothTruth(family, beta, {
        "weights": [1.0 / norm, -c * k / norm],
        "mus": [shift, shift], "sigmas": [sb, sp],
        "W": W, "depth": c, "sigma_bg": sb, "sigma_dip": sg})


# --------------------------------------------------------------------------- #
# the constrained polish: n poles, decoder-shaped parameter space
# --------------------------------------------------------------------------- #

def _unpack(th: np.ndarray, n: int, gfloor: float, t_cap: float = 0.5):
    eps = th[:n]
    gam = gfloor + np.exp(np.clip(th[n:2 * n], -30.0, 30.0))
    s = th[2 * n:3 * n]
    a = np.exp(s - s.max())
    a = a / a.sum()
    if t_cap > 0.0:
        t = t_cap * np.tanh(th[3 * n:])
    else:
        t = np.zeros(n)
    b = a * (t - (a * t).sum())
    return eps, gam, a, b, t


def _pack(eps, gam, a, b, gfloor, t_cap: float = 0.5):
    a = np.maximum(a, 1e-12)
    cap = max(t_cap, 1e-9)
    t = np.clip(b / a, -0.998 * cap, 0.998 * cap)   # centred t is recovered
    th = [                                          # up to the gauge <t>_a=0
        np.clip(eps, -EPS_MAX + 1e-6, EPS_MAX - 1e-6),
        np.log(np.clip(gam - gfloor, 1e-6, GAM_MAX * (1 - 1e-9))),
        np.clip(np.log(a / a.sum()), -29.0, 29.0),
    ]
    if t_cap > 0.0:
        th.append(np.arctanh(np.clip(t / t_cap, -0.998, 0.998)))
    return np.concatenate(th)


def _bounds(n: int, t_cap: float):
    lo = [np.full(n, -EPS_MAX), np.full(n, math.log(1e-6)), np.full(n, -30.0)]
    hi = [np.full(n, EPS_MAX), np.full(n, math.log(GAM_MAX)), np.full(n, 30.0)]
    if t_cap > 0.0:
        lo.append(np.full(n, -12.0))
        hi.append(np.full(n, 12.0))
    return np.concatenate(lo), np.concatenate(hi)


def _pole_F(om, eps, gam, a, b):
    r = a + 1j * b
    z = eps - 1j * gam
    return (r[None, :] / (1j * om[:, None] - z[None, :])).sum(1)


def _rel_resid(om, F, eps, gam, a, b) -> float:
    return float(np.linalg.norm(_pole_F(om, eps, gam, a, b) - F)
                 / np.linalg.norm(F))


def _ablock(truth: "SmoothTruth"):
    """The homotopy's mollified-A comparison: (grid, target, sigma, scale)."""
    sm = SIGMA_MOLL / truth.beta
    L = truth.support_bound() + 4.0 * sm
    wgrid = np.linspace(-L, L, A_GRID_N)
    target = truth.mollified_A(wgrid, sm)
    return wgrid, target, sm, float(target.max())


def _voigt_kernel(wgrid, eps, gam, sm):
    r"""Complex kernel H and its :math:`\partial_z` for the mollified block.

    :math:`(A_p * N_\sigma)(\omega) = -\tfrac1\pi \mathrm{Im}[r_p
    S_N(\omega - z_p)]` with :math:`S_N(z) = -i\sqrt\pi\,w(\zeta)/(\sqrt2
    \sigma)` -- the Lorentzian-Gaussian convolution is the Faddeeva
    function again, so the block is exact and differentiable in closed
    form (:math:`w' = -2\zeta w + 2i/\sqrt\pi`).
    """
    z = eps - 1j * gam
    zeta = (wgrid[:, None] - z[None, :]) / (_SQRT2 * sm)
    cw = -1j * math.sqrt(math.pi) / (_SQRT2 * sm)
    wz = wofz(zeta)
    H = cw * wz
    dH_dz = cw * (-2.0 * zeta * wz + 2j / math.sqrt(math.pi)) \
        * (-1.0 / (_SQRT2 * sm))
    return H, dH_dz


def _deriv_blocks(K, Kz, r, a, t, T, gam, gfloor, t_cap):
    """Complex parameter-derivative of L = K @ r for any kernel K(z_p)."""
    L = K @ r
    Ga = K @ a
    d_eps = Kz * r[None, :]
    d_u = -1j * (gam - gfloor)[None, :] * d_eps
    d_s = K * r[None, :] - a[None, :] * L[:, None] \
        - 1j * (a * (t - T))[None, :] * Ga[:, None]
    blocks = [d_eps, d_u, d_s]
    if t_cap > 0.0:
        tau = t_cap * (1.0 - (t / t_cap) ** 2)   # dt/dv, with tanh v = t/cap
        blocks.append(1j * (tau * a)[None, :] * (K - Ga[:, None]))
    return np.concatenate(blocks, axis=1)


def _residual_vec(th, n, om, F, Fs, gfloor, t_cap=0.5, ab=None):
    eps, gam, a, b, _ = _unpack(th, n, gfloor, t_cap)
    d = (_pole_F(om, eps, gam, a, b) - F) / Fs
    rows = [d.real, d.imag]
    if ab is not None:
        wgrid, target, sm, As, wA = ab
        H, _ = _voigt_kernel(wgrid, eps, gam, sm)
        Am = -(H @ (a + 1j * b)).imag / math.pi
        rows.append(wA * (Am - target) / As)
    return np.concatenate(rows)


def _residual_jac(th, n, om, F, Fs, gfloor, t_cap=0.5, ab=None):
    """Analytic Jacobian of :func:`_residual_vec` (checked against finite
    differences in the unit tests; the softmax and centred-t couplings make
    the cross terms easy to get silently wrong)."""
    eps, gam, a, b, t = _unpack(th, n, gfloor, t_cap)
    r = a + 1j * b
    z = eps - 1j * gam
    T = (a * t).sum()
    C = 1.0 / (1j * om[:, None] - z[None, :])
    Jc = _deriv_blocks(C, C * C, r, a, t, T, gam, gfloor, t_cap) / Fs
    rows = [Jc.real, Jc.imag]
    if ab is not None:
        wgrid, target, sm, As, wA = ab
        H, dH = _voigt_kernel(wgrid, eps, gam, sm)
        Ja = _deriv_blocks(H, dH, r, a, t, T, gam, gfloor, t_cap)
        rows.append(-(wA / (math.pi * As)) * Ja.imag)
    return np.concatenate(rows, axis=0)


def _polish(om, F, cfg, gfloor, t_cap: float = 0.5, ab=None):
    """Deterministic bounded trust-region fit in the constrained space,
    analytic Jacobian.  The bounds are the parameter box above; ``t_cap``
    tightens the asymmetry when the dip bound asks for it; ``ab`` adds the
    weighted mollified-A block."""
    n = cfg[0].size
    Fs = np.abs(F).max()
    args = (n, om, F, Fs, gfloor, t_cap, ab)
    sol = least_squares(_residual_vec, _pack(*cfg, gfloor, t_cap),
                        jac=_residual_jac, args=args, method="trf",
                        bounds=_bounds(n, t_cap), tr_solver="exact",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15,
                        max_nfev=LM_MAX_NFEV)
    eps, gam, a, b, _ = _unpack(sol.x, n, gfloor, t_cap)
    return eps, gam, a, b


def _polish_smooth(om, F, cfg, gfloor, ab, t_cap: float = 0.5):
    """The canonical budgeted polish: the smoothest fit that is still a
    data-fidelity fit.  A pure fit sets the rank's data floor; the A-block
    weight then descends from smooth to sharp (warm-started, so the search
    stays on the smooth side of the basin) and the first fit whose PURE
    Matsubara residual meets the budget wins.  When no smoothing budget is
    feasible (floor-limited or rank-limited), the pure fit -- reached from
    the smooth side -- is the label."""
    cfg0 = _polish(om, F, cfg, gfloor, t_cap)
    res0 = _rel_resid(om, F, *cfg0)
    target = max(LABEL_RES_BUDGET, 1.5 * res0)
    cur = cfg
    for wA in W_LADDER:
        cur = _polish(om, F, cur, gfloor, t_cap, ab=(*ab, wA))
        if _rel_resid(om, F, *cur) <= target:
            return cur
    cur = _polish(om, F, cur, gfloor, t_cap)
    return cur if _rel_resid(om, F, *cur) <= res0 else cfg0


def _aaa_seed_sets(om, F, max_terms=AAA_MAX_TERMS):
    """Best causal seed configuration per rank, from an AAA term sweep.

    AAA supplies *positions* only -- its interpolation-error stopping rule
    says nothing about the causal-pole-sum error (measured: interpolation at
    1e-12 with the causal subset stuck at 1e-1 for Gaussian tails), so every
    seed goes through the constrained polish before it counts.
    """
    def residues(zs):
        K = np.concatenate([1.0 / (1j * om[:, None] - zs[None, :]),
                            np.ones((om.size, 1), complex)], axis=1)
        sol, *_ = np.linalg.lstsq(K, F, rcond=None)
        return sol[:-1]

    sets: dict[int, tuple] = {}
    for m in range(2, max_terms + 1):
        poles, mm = aaa_poles(1j * om, F, tol=0.0, max_terms=m)
        if mm < m:
            break
        causal = poles[np.imag(poles) < 0]
        if causal.size == 0:
            continue
        r = residues(causal)
        F_fit = (r[None, :] / (1j * om[:, None] - causal[None, :])).sum(1)
        res = float(np.linalg.norm(F_fit - F) / np.linalg.norm(F))
        k = causal.size
        if k not in sets or res < sets[k][2]:
            sets[k] = (causal, r, res)
    return sets


def _seed_cfg(zs, r, gfloor):
    eps = zs.real.copy()
    gam = np.maximum(-zs.imag, 1.5 * gfloor)
    a = np.maximum(r.real, 1e-4 * max(float(np.abs(r.real).max()), 1e-12))
    a = a / a.sum()
    b = np.clip(r.imag, -0.45 * a, 0.45 * a)
    return eps, gam, a, b


def _truncate_by_weight(cfg, n):
    eps, gam, a, b = cfg
    order = np.argsort(-a)[:n]
    keep = (eps[order], gam[order], a[order], b[order])
    s = keep[2].sum()
    return keep[0], keep[1], keep[2] / s, keep[3] / s


def _split_heaviest(cfg, gfloor):
    eps, gam, a, b = (v.copy() for v in cfg)
    i = int(np.argmax(a))
    e0, g0, a0, b0 = eps[i], gam[i], a[i], b[i]
    g1 = max(0.6 * g0, 1.2 * gfloor)
    eps = np.concatenate([np.delete(eps, i), [e0 - 0.5 * g0, e0 + 0.5 * g0]])
    gam = np.concatenate([np.delete(gam, i), [g1, g1]])
    a = np.concatenate([np.delete(a, i), [a0 / 2, a0 / 2]])
    b = np.concatenate([np.delete(b, i), [b0 / 2, b0 / 2]])
    return eps, gam, a, b


def _ladder(om, F, beta, P_max, ab):
    """Rung n -> (config, residual) for n = 1..P_max, with early stop."""
    gfloor = 1.05 * GAMMA_MIN / beta
    seeds = _aaa_seed_sets(om, F)
    full = max(seeds) if seeds else None
    rungs: dict[int, tuple] = {}
    prev = None
    for n in range(1, P_max + 1):
        cands = []
        if n in seeds:
            cands.append(_seed_cfg(*seeds[n][:2], gfloor))
        elif full is not None and full > n:
            cands.append(_truncate_by_weight(_seed_cfg(*seeds[full][:2],
                                                       gfloor), n))
        if prev is not None:
            cands.append(_split_heaviest(prev, gfloor))
        if not cands:
            cands.append((np.zeros(1), np.ones(1) * max(1.0, 2 * gfloor),
                          np.ones(1), np.zeros(1)))
        best = None
        for c in cands:
            if c[0].size != n:
                continue
            eps, gam, a, b = _polish_smooth(om, F, c, gfloor, ab)
            res = _rel_resid(om, F, eps, gam, a, b)
            if best is None or res < best[1]:
                best = ((eps, gam, a, b), res)
        if best is None:
            continue
        rungs[n] = best
        prev = best[0]
        if best[1] < RES_FLOOR:
            break
        # Far above the budget (100x) the truth is floor-limited: stop when
        # two extra rungs buy < 10%.  The gate matters twice over -- (i)
        # stopping at the first in-budget rung was measured to cost 2-3x in
        # label W1 (higher rungs keep buying real spectral fidelity after
        # the data residual is satisfied), and (ii) Gaussian tails converge
        # SLOWLY through the 1e-5..1e-6 zone (measured: 1.1e-5, 6.2e-6,
        # 6.1e-6, then 3e-7), which an ungated plateau rule mistakes for a
        # floor; true floors sit at 1e-3 and above.
        if best[1] > 100.0 * LABEL_RES_BUDGET and n >= 4 and len(rungs) >= 3:
            r3 = [rungs[k][1] for k in sorted(rungs)[-3:]]
            if r3[-1] > 0.9 * min(r3[:-1]):
                break
    return rungs, gfloor


# --------------------------------------------------------------------------- #
# canonical label
# --------------------------------------------------------------------------- #

@dataclass
class SmoothLabel:
    """The canonical label plus everything the extraction measured."""

    eps: np.ndarray
    gamma: np.ndarray
    a: np.ndarray
    b: np.ndarray
    resid: float                       #: rel. Matsubara residual of the label
    ladder: list[tuple[int, float]]    #: [(rank, residual)] for every rung
    neg_weight: float                  #: certified negative weight fraction
    n_shrink: int                      #: b-shrink rounds the dip bound needed

    @property
    def n(self) -> int:
        return int(self.eps.size)

    def poleset(self) -> PoleSet:
        """Physical-units :class:`PoleSet` with batch dimension 1."""
        return PoleSet(eps=torch.tensor(self.eps).unsqueeze(0).float(),
                       gamma=torch.tensor(self.gamma).unsqueeze(0).float(),
                       a=torch.tensor(self.a).unsqueeze(0).float(),
                       b=torch.tensor(self.b).unsqueeze(0).float())

    def rank_at(self, tol: float) -> int:
        """Smallest ladder rank whose residual is <= tol (the rank the data
        would support at that noise level -- rank grows as tol shrinks)."""
        ok = [n for n, r in self.ladder if r <= tol]
        return min(ok) if ok else max(n for n, _ in self.ladder)


def _canonical_cleanup(om, F, cfg, gfloor, bar, ab):
    """Dust drop, plus sub-width merges that do not cost fidelity.

    A merge is *tried* for every pair closer than the pole prior's
    sub-width criterion, but accepted only if the merged-and-re-polished
    fit stays within the label's fidelity bar: a stacked pair at one centre
    (narrow core + broad base) can be the genuinely minimal representation
    of a smooth shape, and merging load-bearing structure would silently
    degrade the label below its claimed residual.
    """
    eps, gam, a, b = cfg

    def dedust(eps, gam, a, b):
        keep = a > DUST_TOL
        if not keep.all():
            eps, gam, a, b = eps[keep], gam[keep], a[keep], b[keep]
            s = a.sum()
            a, b = a / s, b / s
        return eps, gam, a, b

    eps, gam, a, b = dedust(eps, gam, a, b)
    rejected: set[tuple[int, int]] = set()
    for _ in range(eps.size + 1):
        n = eps.size
        pair = next(((i, j) for i in range(n - 1) for j in range(i + 1, n)
                     if abs(eps[i] - eps[j]) < MERGE_FRAC * min(gam[i], gam[j])
                     and (round(float(eps[i]), 9), round(float(eps[j]), 9))
                     not in rejected), None)
        if pair is None:
            break
        i, j = pair
        key = (round(float(eps[i]), 9), round(float(eps[j]), 9))
        w = a[i] + a[j]
        m_eps = np.concatenate([np.delete(eps, [i, j]),
                                [(a[i] * eps[i] + a[j] * eps[j]) / w]])
        m_gam = np.concatenate([np.delete(gam, [i, j]),
                                [(a[i] * gam[i] + a[j] * gam[j]) / w]])
        m_a = np.concatenate([np.delete(a, [i, j]), [w]])
        m_b = np.concatenate([np.delete(b, [i, j]), [b[i] + b[j]]])
        m = _polish_smooth(om, F, (m_eps, m_gam, m_a, m_b), gfloor, ab)
        if _rel_resid(om, F, *m) <= bar:
            eps, gam, a, b = dedust(*m)
        else:
            rejected.add(key)
    return eps, gam, a, b


def extract_smooth_label(truth: SmoothTruth, P_max: int = 12) -> SmoothLabel:
    """Run the canonical protocol on one truth.  Deterministic."""
    om, F = truth.matsubara()
    ab = _ablock(truth)
    rungs, gfloor = _ladder(om, F, truth.beta, P_max, ab)
    ladder = sorted((n, res) for n, (_, res) in rungs.items())
    # Minimal rank at the achievable fidelity: the smallest rung within 5%
    # of the best.  (An earlier 3x slack was measured to matter: a
    # floor-limited truth plateaus at 1e-3, where a 3x-worse rung is a
    # materially worse label, not a plateau wiggle.)
    res_best = min(r for _, r in ladder)
    bar = max(RES_TARGET, 1.05 * res_best)
    n_label = min(n for n, r in ladder if r <= bar)
    eps, gam, a, b = _canonical_cleanup(om, F, rungs[n_label][0], gfloor, bar,
                                        ab)

    # Dip bound, as in the pole prior: measured with the exact certificate,
    # acted on only when the negative weight exceeds the tolerance -- a
    # bound, not a ban.  The remedy is a re-polish under a tighter asymmetry
    # cap, NOT a post-hoc shrink of b: damaging b outside the optimizer was
    # measured to cost 17x in residual (3.3e-3 -> 5.7e-2) where the capped
    # re-fit reoptimizes positions and weights alongside it.  t_cap = 0
    # (a positive Lorentzian mixture) certifies by construction, so the
    # ladder always terminates.
    n_shrink = 0
    rep = certify(eps.tolist(), gam.tolist(), a.tolist(), b.tolist(),
                  atol=1e-9)
    for cap in (0.25, 0.125, 0.0):
        if rep.is_positive or rep.negative_weight <= DIP_TOL:
            break
        n_shrink += 1
        eps, gam, a, b = _polish_smooth(om, F, (eps, gam, a, b), gfloor, ab,
                                  t_cap=cap)
        rep = certify(eps.tolist(), gam.tolist(), a.tolist(), b.tolist(),
                      atol=1e-9)

    # exact sum rules (cleanup and the cap ladder work at the dust level)
    a = a / a.sum()
    b = b - b.sum() * a
    b = np.clip(b, -a, a)

    return SmoothLabel(
        eps=eps, gamma=gam, a=a, b=b,
        resid=_rel_resid(om, F, eps, gam, a, b),
        ladder=ladder,
        neg_weight=0.0 if rep.is_positive else float(rep.negative_weight),
        n_shrink=n_shrink,
    )


# --------------------------------------------------------------------------- #
# the bank
# --------------------------------------------------------------------------- #

_BANK_VERSION = 1


@dataclass
class LabelBank:
    """Padded, saveable collection of labeled smooth truths.

    Tensors are in **physical** units with per-entry ``beta`` (labels are
    :math:`\\beta`-dependent through the Matsubara sample positions -- rank
    grows with :math:`\\beta`, which is physics, not an artifact); padding
    carries zero weight and inert unit widths, as in the pole prior.
    """

    P_max: int
    beta: torch.Tensor                 # (N,)
    eps: torch.Tensor                  # (N, P_max) physical
    gamma: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    n_true: torch.Tensor               # (N,)
    resid: torch.Tensor                # (N,)
    neg_weight: torch.Tensor           # (N,)
    family: list[str]
    params: list[dict]
    ladders: list[list[tuple[int, float]]]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.beta.numel())

    def truth(self, i: int) -> SmoothTruth:
        return SmoothTruth(self.family[i], float(self.beta[i]), self.params[i])

    def label(self, i: int) -> PoleSet:
        m = int(self.n_true[i])
        return PoleSet(eps=self.eps[i, :m].unsqueeze(0),
                       gamma=self.gamma[i, :m].unsqueeze(0),
                       a=self.a[i, :m].unsqueeze(0),
                       b=self.b[i, :m].unsqueeze(0))

    def labels_dimensionless(self) -> dict:
        """Padded label dict in the training convention (``nu = beta*omega``);
        padding keeps inert unit widths, as the pole prior does."""
        beta = self.beta.unsqueeze(-1)
        pad = self.a == 0
        return {"eps": self.eps * beta,
                "gamma": torch.where(pad, torch.ones_like(self.gamma),
                                     self.gamma * beta),
                "a": self.a, "b": self.b, "n_true": self.n_true}

    def save(self, path) -> None:
        torch.save({
            "version": _BANK_VERSION, "P_max": self.P_max,
            "beta": self.beta, "eps": self.eps, "gamma": self.gamma,
            "a": self.a, "b": self.b, "n_true": self.n_true,
            "resid": self.resid, "neg_weight": self.neg_weight,
            "family": self.family, "params": self.params,
            "ladders": self.ladders, "meta": self.meta,
        }, path)

    @classmethod
    def load(cls, path) -> "LabelBank":
        d = torch.load(path, weights_only=False)
        if d.pop("version") != _BANK_VERSION:
            raise ValueError(f"bank version mismatch in {path}")
        return cls(**d)


def _extract_entry(args):
    truth, P_max = args
    return extract_smooth_label(truth, P_max=P_max)


def build_label_bank(
    n_per_family: int = 256,
    seed: int = 1,
    beta_range: tuple[float, float] = (2.0, 20.0),
    families: tuple[str, ...] = SMOOTH_FAMILIES,
    P_max: int = 12,
    workers: int | None = None,
    progress=None,
) -> LabelBank:
    """Draw truths (one rng stream, so the bank is seed-deterministic
    regardless of ``workers``) and extract every label.

    ``progress`` is an optional callable ``(done, total) -> None``.
    """
    rng = torch.Generator().manual_seed(int(seed))
    lo, hi = (math.log(v) for v in beta_range)
    truths: list[SmoothTruth] = []
    for _ in range(int(n_per_family)):
        for fam in families:
            beta = math.exp(lo + (hi - lo) * float(torch.rand((), generator=rng)))
            truths.append(sample_smooth_truth(fam, beta, rng))

    jobs = [(t, P_max) for t in truths]
    labels: list[SmoothLabel] = []
    if workers is not None and workers > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=int(workers)) as ex:
            for i, lab in enumerate(ex.map(_extract_entry, jobs, chunksize=4)):
                labels.append(lab)
                if progress is not None:
                    progress(i + 1, len(jobs))
    else:
        for i, job in enumerate(jobs):
            labels.append(_extract_entry(job))
            if progress is not None:
                progress(i + 1, len(jobs))

    N = len(truths)
    out = {k: torch.zeros(N, P_max) for k in ("eps", "gamma", "a", "b")}
    out["gamma"] += 1.0                       # inert padding widths
    n_true = torch.zeros(N, dtype=torch.long)
    resid = torch.zeros(N)
    negw = torch.zeros(N)
    for i, lab in enumerate(labels):
        m = lab.n
        n_true[i] = m
        out["eps"][i, :m] = torch.tensor(lab.eps).float()
        out["gamma"][i, :m] = torch.tensor(lab.gamma).float()
        out["a"][i, :m] = torch.tensor(lab.a).float()
        out["b"][i, :m] = torch.tensor(lab.b).float()
        resid[i] = lab.resid
        negw[i] = lab.neg_weight

    return LabelBank(
        P_max=P_max,
        beta=torch.tensor([t.beta for t in truths]),
        eps=out["eps"], gamma=out["gamma"], a=out["a"], b=out["b"],
        n_true=n_true, resid=resid, neg_weight=negw,
        family=[t.family for t in truths],
        params=[t.params for t in truths],
        ladders=[lab.ladder for lab in labels],
        meta={"seed": int(seed), "beta_range": tuple(beta_range),
              "families": tuple(families), "n_per_family": int(n_per_family),
              "protocol": {"N_MATSUBARA": N_MATSUBARA, "N_GL": N_GL,
                           "AAA_MAX_TERMS": AAA_MAX_TERMS,
                           "RES_TARGET": RES_TARGET, "DUST_TOL": DUST_TOL,
                           "MERGE_FRAC": MERGE_FRAC, "DIP_TOL": DIP_TOL,
                           "SIGMA_MOLL": SIGMA_MOLL, "W_LADDER": W_LADDER,
                           "LABEL_RES_BUDGET": LABEL_RES_BUDGET,
                           "A_GRID_N": A_GRID_N, "EPS_MAX": EPS_MAX,
                           "GAM_MAX": GAM_MAX}},
    )
