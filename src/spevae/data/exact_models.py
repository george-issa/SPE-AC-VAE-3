r"""Exactly solvable fermionic models: spectra whose answer is known.

Why this exists
---------------
Every truth the model had trained on was an invented shape -- pole motifs,
Gaussian mixtures, boxes.  They cover a space of curves; they carry no
physics.  This module supplies single-particle spectral functions of
Hamiltonians that can be solved exactly, so the training prior contains
what band edges, van Hove singularities, correlation gaps, coherence peaks,
multiplets and phonon ladders actually look like, and so the pipeline can be
judged on cases where the right answer is not a matter of opinion.

Two kinds of model, two routes into the training set.

**Continuum models** have a closed (or one-dimensional-integral) lattice
Green's function :math:`G(z)` in the upper half plane:

==================  =========================================================
``tb_chain``        1D tight binding (inverse-square-root band edges)
``tb_square``       2D square lattice, optional :math:`t'` (log van Hove)
``tb_triangular``   2D triangular lattice (asymmetric band)
``tb_honeycomb``    honeycomb lattice (Dirac point, two van Hove peaks)
``tb_bethe``        Bethe lattice (semicircle)
``tb_hypercubic``   :math:`d = \infty` hypercubic lattice (Gaussian)
``ssh_chain``       dimerised chain (band gap, four square-root edges)
``bcs``             mean-field s-wave superconductor on any symmetric band
``resonant_level``  non-interacting Anderson impurity in a band
``falicov_kimball`` Falicov--Kimball model on the Bethe lattice (exact DMFT;
                    a correlation gap opens at :math:`U = 2t`)
==================  =========================================================

A truth is the model with a quasiparticle lifetime :math:`\eta`:
:math:`A(\omega) = -\tfrac1\pi\,\mathrm{Im}\,G(\omega + \mu + i\eta)`.  That
is the exact spectrum convolved with a Lorentzian of half-width
:math:`\eta`, so its Stieltjes transform is :math:`G(z + \mu + i\eta)`
exactly -- no singular integral is ever taken, and
:class:`ExactTruth` offers the same interface as
:class:`~spevae.data.label_bank.SmoothTruth` (``A``, ``stieltjes``,
``matsubara``, ``mollified_A``, ``support_bound``), so these families go
through the label bank unchanged.  :math:`\eta` is never below the
resolution floor: structure finer than that is not recoverable, and
teaching it teaches hallucination.

**Discrete models** are finite Hamiltonians diagonalised exactly
(:func:`lehmann`): the Hubbard atom, Hubbard clusters of two to four sites,
an Anderson impurity with a few bath levels, and the Holstein(-Hubbard)
atom with a truncated phonon.  Their spectra are sums of delta functions,
:class:`DiscreteSpectrum`.  Broadened by a lifetime :math:`\eta` such a
spectrum *is* a pole set, so it enters the pole prior with an exact label
and no fit (:func:`sample_exact_poles`); unbroadened it gives known-answer
test data through the closed-form :meth:`DiscreteSpectrum.greens`.

Conventions: physical units, :math:`\int A = 1`,
:math:`G(\tau) = \int A(\omega)\, e^{-\tau\omega}/(1 + e^{-\beta\omega})\,
d\omega` (so :math:`\omega > 0` is electron addition), as everywhere in the
package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.special import wofz

__all__ = [
    "EXACT_CONTINUUM", "EXACT_DISCRETE", "ExactTruth", "sample_exact_truth",
    "DiscreteSpectrum", "lehmann", "hubbard_cluster", "holstein_atom",
    "sample_discrete_spectrum", "sample_exact_poles", "lattice_greens",
]

EXACT_CONTINUUM = ("tb_chain", "tb_square", "tb_triangular", "tb_honeycomb", "tb_bethe",
                   "tb_hypercubic", "ssh_chain", "bcs", "resonant_level", "falicov_kimball")
EXACT_DISCRETE = ("hubbard_atom", "hubbard_dimer", "hubbard_trimer", "hubbard_ring4",
                  "anderson_cluster", "holstein_atom")

GAMMA_MIN = 0.1 * math.pi          # dimensionless width floor, as in the label bank
_SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# lattice Green's functions, Im z > 0
# --------------------------------------------------------------------------- #

def _inv_sqrt(u, b):
    r""":math:`1/\sqrt{u^2 - b^2}` on the branch that behaves as :math:`1/u`.

    This is :math:`\frac{1}{2\pi}\int dk\,(u - b\cos k)^{-1}`; written as
    :math:`1/(u\sqrt{1 - b^2/u^2})` with the principal root it is analytic
    everywhere off the real segment :math:`|u| < |b|`.
    """
    return 1.0 / (u * np.sqrt(1.0 - (b / u) ** 2))


def _k_average(a_of_k, b_of_k, z, scale, n_max=16384):
    r"""Midpoint rule over one Brillouin-zone direction of the exact inner
    integral: :math:`\langle 1/\sqrt{(z - a(k))^2 - b(k)^2}\rangle_k`.

    The integrand is periodic and analytic for Im z > 0, so the rule
    converges geometrically, with a rate set by how close its branch points
    come to the real k axis -- about Im z / scale.  The node count follows
    that distance.
    """
    z = np.asarray(z, complex)
    dist = max(float(np.min(np.abs(z.imag))), 1e-6)
    n = int(min(n_max, max(256, 2 ** math.ceil(math.log2(48.0 * scale / dist)))))
    k = (np.arange(n) + 0.5) * (2.0 * math.pi / n)
    a, b = a_of_k(k), b_of_k(k)
    flat = z.reshape(-1)
    out = np.empty_like(flat)
    step = max(1, (1 << 22) // n)                       # bound the (points, n) block
    for i in range(0, flat.size, step):
        blk = flat[i:i + step, None]
        out[i:i + step] = _inv_sqrt(blk - a, b).mean(-1)
    return out.reshape(z.shape)


def lattice_greens(model: str, z, **p) -> np.ndarray:
    r"""Local Green's function :math:`G(z)` of a named model, Im z > 0."""
    z = np.asarray(z, complex)
    if model == "tb_chain":
        return _inv_sqrt(z, 2.0 * p["t"])
    if model == "tb_bethe":
        t = p["t"]
        return (z - z * np.sqrt(1.0 - (2.0 * t / z) ** 2)) / (2.0 * t * t)
    if model == "tb_hypercubic":
        t = p["t"]
        return -1j * math.sqrt(math.pi / 2.0) / t * wofz(z / (_SQRT2 * t))
    if model == "tb_square":
        t, tp = p["t"], p.get("tp", 0.0)
        # eps = -2t cos kx - 2t cos ky - 4t' cos kx cos ky = a(ky) + b(ky) cos kx
        return _k_average(lambda k: -2.0 * t * np.cos(k),
                          lambda k: -2.0 * t - 4.0 * tp * np.cos(k), z,
                          scale=2.0 * abs(t) + 4.0 * abs(tp))
    if model == "tb_triangular":
        t = p["t"]
        # eps = -2t[cos k1 + cos k2 + cos(k1 - k2)]
        #     = -2t cos k2 - 4t cos(k2/2) cos(k1 - k2/2)
        return _k_average(lambda k: -2.0 * t * np.cos(k),
                          lambda k: -4.0 * t * np.cos(0.5 * k), z, scale=4.0 * abs(t))
    if model == "tb_honeycomb":
        t = p["t"]
        # G_AA = <z / (z^2 - |f_k|^2)>, |f_k|^2 = t^2 [3 + 2cos k1 + 2cos k2 + 2cos(k1-k2)]
        return z * _k_average_E(z * z - 3.0 * t * t, t * t, z)
    if model == "ssh_chain":
        t1, t2 = p["t1"], p["t2"]
        return z * _inv_sqrt(z * z - t1 * t1 - t2 * t2, 2.0 * t1 * t2)
    if model == "bcs":
        gap = p["gap"]
        s = z * np.sqrt(1.0 - (gap / z) ** 2)            # sqrt(z^2 - gap^2), Im s > 0
        return (z / s) * lattice_greens(p["band"], s, **p["band_params"])
    if model == "resonant_level":
        bath = lattice_greens(p["bath"], z, **p["bath_params"])
        return 1.0 / (z - p["e_d"] - p["V"] ** 2 * bath)
    if model == "falicov_kimball":
        return _fk_bethe(z, p["t"], p["U"], p["w1"])
    raise ValueError(f"unknown exact model {model!r}")


def _k_average_E(E, tt, z):
    """Honeycomb inner average in the E = z^2 - 3t^2 plane (E may sit in
    either half plane; the branch of :func:`_inv_sqrt` is valid in both)."""
    # distance of the E-plane branch points from the real k axis, in E units:
    # Im E = 2 Re(z) Im(z) vanishes at Re z = 0, where |dE/dz| -> 0 as well, so
    # use the z-plane distance times the typical slope 2 sqrt(3) t.
    slope = 2.0 * math.sqrt(3.0 * tt)
    dist = max(float(np.min(np.abs(z.imag))), 1e-6) * slope
    n = int(min(16384, max(256, 2 ** math.ceil(math.log2(48.0 * 4.0 * tt / dist)))))
    k = (np.arange(n) + 0.5) * (2.0 * math.pi / n)
    a, b = 2.0 * tt * np.cos(k), 4.0 * tt * np.cos(0.5 * k)
    flat = np.asarray(E, complex).reshape(-1)
    out = np.empty_like(flat)
    step = max(1, (1 << 22) // n)
    for i in range(0, flat.size, step):
        out[i:i + step] = _inv_sqrt(flat[i:i + step, None] - a, b).mean(-1)
    return out.reshape(np.shape(E))


def _fk_bethe(z, t, U, w1, levels: int = 48):
    r"""Falicov--Kimball model on the Bethe lattice (exact in :math:`d=\infty`).

    :math:`G = \frac{1-w_1}{\zeta - t^2 G} + \frac{w_1}{\zeta - U - t^2 G}`,
    a cubic in :math:`G`.  The physical root is the one continuous with
    :math:`1/\zeta` at large Im :math:`\zeta`, so it is followed down from
    there by Newton steps (a root picked by sign alone can be the wrong one).
    """
    z = np.asarray(z, complex)
    t2, t4 = t * t, t ** 4

    def f_df(G, zeta):
        c2 = -t2 * (2.0 * zeta - U)
        c1 = zeta * (zeta - U) + t2
        c0 = -(zeta - (1.0 - w1) * U)
        return t4 * G ** 3 + c2 * G * G + c1 * G + c0, 3.0 * t4 * G * G + 2.0 * c2 * G + c1

    y0 = np.maximum(z.imag, 1e-9)
    top = max(60.0 * (2.0 * t + abs(U)), 10.0 * float(y0.max()))
    G = 1.0 / (z.real + 1j * top)
    for lam in np.linspace(0.0, 1.0, levels + 1)[1:]:
        zeta = z.real + 1j * np.exp((1.0 - lam) * math.log(top) + lam * np.log(y0))
        for _ in range(6):
            f, df = f_df(G, zeta)
            G = G - f / df
    for _ in range(4):
        f, df = f_df(G, z)
        G = G - f / df
    return G


# --------------------------------------------------------------------------- #
# continuum truths
# --------------------------------------------------------------------------- #

def _band_edge(model: str, p: dict) -> float:
    """|omega| bound of the unbroadened support (about the model's own zero)."""
    if model in ("tb_chain", "tb_bethe"):
        return 2.0 * abs(p["t"])
    if model == "tb_hypercubic":
        return 5.0 * abs(p["t"])
    if model == "tb_square":
        return 4.0 * abs(p["t"]) + 4.0 * abs(p.get("tp", 0.0))
    if model == "tb_triangular":
        return 6.0 * abs(p["t"])
    if model == "tb_honeycomb":
        return 3.0 * abs(p["t"])
    if model == "ssh_chain":
        return abs(p["t1"]) + abs(p["t2"])
    if model == "bcs":
        return math.hypot(_band_edge(p["band"], p["band_params"]), p["gap"])
    if model == "resonant_level":
        return max(_band_edge(p["bath"], p["bath_params"]), abs(p["e_d"])) + abs(p["V"])
    if model == "falicov_kimball":
        return 2.0 * abs(p["t"]) + abs(p["U"])
    raise ValueError(model)


@dataclass
class ExactTruth:
    """One exact continuum truth: model + parameters + lifetime + beta.

    ``params`` holds ``model`` (the model's own parameters), ``mu`` (the
    chemical potential: the spectrum is shifted so the Fermi level sits at
    :math:`\\omega = 0`) and ``eta``.  Reconstructible from ``(family, beta,
    params)``, which is what a :class:`~spevae.data.label_bank.LabelBank`
    stores.
    """

    family: str
    beta: float
    params: dict

    def _g(self, z):
        p = self.params
        return lattice_greens(self.family, np.asarray(z, complex) + p["mu"] + 1j * p["eta"],
                              **p["model"])

    def A(self, omega: np.ndarray) -> np.ndarray:
        """Exact :math:`A(\\omega)` in physical units."""
        return -self._g(np.asarray(omega, float)).imag / math.pi

    def stieltjes(self, z: np.ndarray) -> np.ndarray:
        r""":math:`\int A(\omega)/(z-\omega)\,d\omega` for Im z > 0, exact."""
        return self._g(z)

    def matsubara(self, n_w: int = 64) -> tuple[np.ndarray, np.ndarray]:
        om = (2.0 * np.arange(n_w) + 1.0) * math.pi / self.beta
        return om, self.stieltjes(1j * om)

    def support_bound(self) -> float:
        p = self.params
        return _band_edge(self.family, p["model"]) + abs(p["mu"]) + 12.0 * p["eta"]

    def mollified_A(self, omega: np.ndarray, sigma: float) -> np.ndarray:
        r""":math:`(A * N_\sigma)(\omega)` by the trapezoid rule on a grid that
        resolves both the lifetime and the mollifier (spectrally accurate: the
        integrand is smooth and the Gaussian kills the ends)."""
        omega = np.asarray(omega, float)
        h = 0.25 * min(self.params["eta"], sigma)
        L = float(np.max(np.abs(omega))) + 8.0 * sigma
        n = int(min(2 * math.ceil(L / h) + 1, 60001))
        u = np.linspace(-L, L, n)
        Au = self.A(u) * (u[1] - u[0])
        out = np.empty_like(omega)
        flat = omega.reshape(-1)
        res = out.reshape(-1)
        norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        for i in range(0, flat.size, 256):
            d = (flat[i:i + 256, None] - u[None, :]) / sigma
            res[i:i + 256] = norm * (np.exp(-0.5 * d * d) * Au).sum(-1)
        return out


def sample_exact_truth(family: str, beta: float, rng: torch.Generator,
                       omega_bandwidth: tuple[float, float] = (1.0, 12.0)) -> ExactTruth:
    """Draw one continuum truth.  The half-bandwidth scale matches the smooth
    families'; the lifetime is log-uniform from the resolution floor up."""

    def u(lo, hi):
        return lo + (hi - lo) * float(torch.rand((), generator=rng))

    def logu(lo, hi):
        return math.exp(u(math.log(lo), math.log(hi)))

    def pick(seq):
        return seq[int(torch.randint(len(seq), (1,), generator=rng))]

    if family not in EXACT_CONTINUUM:
        raise ValueError(f"unknown exact continuum family {family!r}")
    floor = 1.05 * GAMMA_MIN / beta
    W = u(*omega_bandwidth)                      # half-bandwidth scale
    eta = floor * logu(1.0, 8.0)
    mu = u(-0.5, 0.5) * W if u(0.0, 1.0) > 0.3 else 0.0     # 30% particle-hole symmetric

    def band(name):
        if name == "tb_chain":
            return {"t": W / 2.0}
        if name == "tb_bethe":
            return {"t": W / 2.0}
        if name == "tb_hypercubic":
            return {"t": W / 2.5}
        if name == "tb_square":
            return {"t": W / 4.0}
        if name == "tb_honeycomb":
            return {"t": W / 3.0}
        raise ValueError(name)

    if family in ("tb_chain", "tb_bethe", "tb_hypercubic", "tb_honeycomb"):
        model = band(family)
    elif family == "tb_square":
        t = W / 4.0
        model = {"t": t, "tp": (u(-0.45, 0.45) * t if u(0.0, 1.0) < 0.5 else 0.0)}
    elif family == "tb_triangular":
        model = {"t": (W / 4.5) * (1.0 if u(0.0, 1.0) < 0.5 else -1.0)}
    elif family == "ssh_chain":
        r = u(0.25, 0.95)                         # t2 / t1: gap = 2 |t1 - t2|
        t1 = W / (1.0 + r)
        model = {"t1": t1, "t2": r * t1}
    elif family == "bcs":
        name = pick(("tb_bethe", "tb_chain", "tb_square", "tb_hypercubic"))
        model = {"band": name, "band_params": band(name), "gap": u(0.05, 0.5) * W}
        mu = 0.0                                  # the closed form needs a symmetric band
    elif family == "resonant_level":
        name = pick(("tb_bethe", "tb_hypercubic", "tb_chain"))
        bp = band(name)
        model = {"bath": name, "bath_params": bp, "e_d": u(-0.7, 0.7) * W,
                 "V": u(0.15, 0.6) * W / 2.0}
    else:                                         # falicov_kimball
        t = W / (2.0 + u(0.0, 3.0))
        U = (W - 2.0 * t) * 2.0                   # bands reach about +-(U/2 + 2t) = +-W
        w1 = 0.5 if u(0.0, 1.0) < 0.5 else u(0.2, 0.8)
        model = {"t": t, "U": U, "w1": w1}
        mu = U / 2.0 + (u(-0.3, 0.3) * W if u(0.0, 1.0) > 0.3 else 0.0)
    return ExactTruth(family, float(beta), {"model": model, "mu": float(mu),
                                            "eta": float(eta), "W": float(W)})


# --------------------------------------------------------------------------- #
# exact diagonalisation: discrete spectra
# --------------------------------------------------------------------------- #

@dataclass
class DiscreteSpectrum:
    r"""An exact spectrum :math:`A(\omega) = \sum_i w_i\,\delta(\omega - \varepsilon_i)`."""

    energies: np.ndarray
    weights: np.ndarray
    beta: float
    family: str = ""
    params: dict = field(default_factory=dict)

    def greens(self, x: np.ndarray) -> np.ndarray:
        r"""Exact :math:`G(\tau)` at :math:`x = \tau/\beta \in [0, 1]`."""
        x = np.asarray(x, float).reshape(-1, 1)
        nu = self.beta * self.energies.reshape(1, -1)
        # e^{-x nu} / (1 + e^{-nu}), written to stay finite for either sign of nu
        k = np.where(nu >= 0.0,
                     np.exp(-x * np.abs(nu)) / (1.0 + np.exp(-np.abs(nu))),
                     np.exp(-(1.0 - x) * np.abs(nu)) / (1.0 + np.exp(-np.abs(nu))))
        return (k * self.weights.reshape(1, -1)).sum(-1)

    def cdf(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, float)
        return (self.weights[None, :] * (self.energies[None, :] <= omega.reshape(-1, 1))
                ).sum(-1).reshape(omega.shape)

    def broadened(self, omega: np.ndarray, eta: float) -> np.ndarray:
        d = np.asarray(omega, float)[..., None] - self.energies
        return (self.weights * eta / math.pi / (d * d + eta * eta)).sum(-1)

    # -- what the canonical label protocol reads (label_bank.extract_smooth_label) -- #

    def stieltjes(self, z: np.ndarray) -> np.ndarray:
        r""":math:`\sum_i w_i/(z-\varepsilon_i)` with the weights normalised to one, exact."""
        z = np.asarray(z, complex)
        return (self.weights / self.weights.sum() / (z[..., None] - self.energies)).sum(-1)

    def matsubara(self, n_w: int = 64) -> tuple[np.ndarray, np.ndarray]:
        om = (2.0 * np.arange(n_w) + 1.0) * math.pi / self.beta
        return om, self.stieltjes(1j * om)

    def support_bound(self) -> float:
        return float(np.abs(self.energies[self.weights > 1e-6 * self.weights.sum()]).max()) + 1.0

    def mollified_A(self, omega: np.ndarray, sigma: float) -> np.ndarray:
        r""":math:`(A * N_\sigma)(\omega)`: a Gaussian of width ``sigma`` on every line, exact."""
        d = np.asarray(omega, float)[..., None] - self.energies
        return (self.weights / self.weights.sum() * np.exp(-0.5 * (d / sigma) ** 2)
                / (sigma * math.sqrt(2.0 * math.pi))).sum(-1)

    def poles(self, eta: float, P_max: int = 12, merge_frac: float = 0.45):
        """The spectrum with lifetime ``eta`` as a pole set: lines closer than
        ``merge_frac * eta`` merged (they are one feature at that width), the
        ``P_max`` heaviest kept, weights renormalised.  Returns
        ``(eps, gamma, a, kept_weight)``."""
        order = np.argsort(self.energies)
        e, w = list(self.energies[order]), list(self.weights[order])
        i = 0
        while i < len(e) - 1:
            if e[i + 1] - e[i] < merge_frac * eta:
                ws = w[i] + w[i + 1]
                e[i] = (w[i] * e[i] + w[i + 1] * e[i + 1]) / ws
                w[i] = ws
                del e[i + 1], w[i + 1]
            else:
                i += 1
        e, w = np.array(e), np.array(w)
        keep = np.argsort(-w)[:P_max]
        kept = float(w[keep].sum())
        return e[keep], np.full(keep.size, float(eta)), w[keep] / kept, kept


def _fermion_ops(n_orb: int) -> list[np.ndarray]:
    """Annihilation operators on the Fock space of ``n_orb`` orbitals."""
    dim = 1 << n_orb
    ops = []
    states = np.arange(dim)
    for j in range(n_orb):
        c = np.zeros((dim, dim))
        occ = (states >> j) & 1 == 1
        src = states[occ]
        sign = np.array([(-1.0) ** bin(int(s) & ((1 << j) - 1)).count("1") for s in src])
        c[src ^ (1 << j), src] = sign
        ops.append(c)
    return ops


def lehmann(H: np.ndarray, c_dag: np.ndarray, beta: float, tol: float = 1e-12,
            family: str = "", params: dict | None = None) -> DiscreteSpectrum:
    r"""Exact finite-temperature spectrum of a finite Hamiltonian.

    .. math::
        A(\omega) = \frac1Z \sum_{n,m} \big(e^{-\beta E_n} + e^{-\beta E_m}\big)
        \,|\langle m|c^\dagger|n\rangle|^2\; \delta\big(\omega - (E_m - E_n)\big)

    Lines at the same energy are merged; the weights sum to one (the
    anticommutator) to rounding.
    """
    E, V = np.linalg.eigh(H)
    p = np.exp(-beta * (E - E[0]))
    p /= p.sum()
    M = V.T @ c_dag @ V                              # <m| c^dag |n>
    W = (M * M) * (p[None, :] + p[:, None])
    dE = E[:, None] - E[None, :]
    w, e = W.reshape(-1), dE.reshape(-1)
    keep = w > tol
    w, e = w[keep], e[keep]
    order = np.argsort(e)
    e, w = e[order], w[order]
    key = np.round(e, 9)
    uniq, inv = np.unique(key, return_inverse=True)
    wsum = np.bincount(inv, weights=w)
    esum = np.bincount(inv, weights=w * e) / wsum
    return DiscreteSpectrum(esum, wsum / wsum.sum(), float(beta), family, params or {})


def hubbard_cluster(hop: np.ndarray, U, mu: float, beta: float, site: int = 0,
                    family: str = "hubbard_cluster", params: dict | None = None):
    r"""Spin-up spectrum on ``site`` of
    :math:`H = \sum_{ij\sigma} h_{ij} c^\dagger_{i\sigma}c_{j\sigma}
    + \sum_i U_i n_{i\uparrow}n_{i\downarrow} - \mu N`.

    ``hop`` is the symmetric one-body matrix (hoppings off the diagonal, site
    energies on it); ``U`` a scalar or one value per site.
    """
    hop = np.asarray(hop, float)
    n = hop.shape[0]
    U = np.full(n, float(U)) if np.isscalar(U) else np.asarray(U, float)
    c = _fermion_ops(2 * n)                          # orbital i + n * spin
    cd = [o.T for o in c]
    num = [cd[k] @ c[k] for k in range(2 * n)]
    H = np.zeros_like(c[0])
    for s in range(2):
        for i in range(n):
            for j in range(n):
                if hop[i, j] != 0.0:
                    H += hop[i, j] * cd[i + n * s] @ c[j + n * s]
    for i in range(n):
        H += U[i] * num[i] @ num[i + n]
    H -= mu * sum(num)
    return lehmann(H, cd[site], beta, family=family, params=params)


def holstein_atom(omega0: float, g: float, U: float, mu: float, beta: float,
                  offset: float = 0.0, n_boson: int = 48,
                  params: dict | None = None) -> DiscreteSpectrum:
    r"""Holstein--Hubbard atom,
    :math:`H = \Omega\, b^\dagger b + g\,(b + b^\dagger)(n - c) + U n_\uparrow
    n_\downarrow - \mu n`, with the phonon truncated at ``n_boson`` quanta.

    The spectrum is the phonon ladder of the independent-boson model, with
    exact finite-temperature weights; the truncation is converged when the
    highest retained levels carry no weight (checked in the tests).
    """
    c = _fermion_ops(2)
    cd = [o.T for o in c]
    n_el = cd[0] @ c[0] + cd[1] @ c[1]
    I_f, I_b = np.eye(4), np.eye(n_boson)
    b = np.diag(np.sqrt(np.arange(1, n_boson)), 1)
    x = b + b.T
    H = (omega0 * np.kron(I_f, b.T @ b) + g * np.kron(n_el - offset * I_f, x)
         + U * np.kron(cd[0] @ c[0] @ cd[1] @ c[1], I_b) - mu * np.kron(n_el, I_b))
    return lehmann(H, np.kron(cd[0], I_b), beta, family="holstein_atom", params=params)


def sample_discrete_spectrum(family: str, beta: float, rng: torch.Generator,
                             omega_bandwidth: tuple[float, float] = (1.0, 9.0),
                             omega_max: float = 14.0) -> DiscreteSpectrum:
    r"""Draw one exactly diagonalised model.  Draws whose lines (carrying more
    than 1e-3 of the weight) reach beyond ``omega_max`` are redrawn: the
    package's band box is :math:`|\omega| \le 20` and comparison methods use
    a narrower window still."""
    if family not in EXACT_DISCRETE:
        raise ValueError(f"unknown exact discrete family {family!r}")
    for _ in range(50):
        spec = _draw_discrete(family, beta, rng, omega_bandwidth)
        if float(np.abs(spec.energies[spec.weights > 1e-3]).max()) <= omega_max:
            break
    return spec


def _draw_discrete(family, beta, rng, omega_bandwidth):
    def u(lo, hi):
        return lo + (hi - lo) * float(torch.rand((), generator=rng))

    W = u(*omega_bandwidth)
    half = u(0.0, 1.0) < 0.4                         # particle-hole symmetric draws

    if family == "hubbard_atom":
        U = u(0.3, 1.8) * W
        mu = U / 2.0 if half else u(-0.4, 1.4) * U
        p = {"U": U, "mu": mu}
        return hubbard_cluster(np.zeros((1, 1)), U, mu, beta, family=family, params=p)
    if family in ("hubbard_dimer", "hubbard_trimer", "hubbard_ring4"):
        n = {"hubbard_dimer": 2, "hubbard_trimer": 3, "hubbard_ring4": 4}[family]
        t = u(0.15, 0.6) * W
        U = u(0.0, 1.6) * W
        hop = np.zeros((n, n))
        for i in range(n - 1):
            hop[i, i + 1] = hop[i + 1, i] = -t
        if n == 4 or (n == 3 and u(0.0, 1.0) < 0.5):
            hop[0, n - 1] = hop[n - 1, 0] = -t        # close the ring
        mu = U / 2.0 if half else U / 2.0 + u(-0.6, 0.6) * W
        p = {"t": t, "U": U, "mu": mu, "ring": bool(hop[0, n - 1] != 0.0)}
        return hubbard_cluster(hop, U, mu, beta, site=int(u(0.0, 1.0) * n) % n,
                               family=family, params=p)
    if family == "anderson_cluster":
        nb = 1 + int(torch.randint(3, (1,), generator=rng))
        U = u(0.3, 1.6) * W
        e_d = -U / 2.0 if half else u(-1.2, 0.2) * U
        e_b = sorted(u(-0.8, 0.8) * W for _ in range(nb))
        if half:
            e_b = [0.5 * (a - b_) for a, b_ in zip(e_b, reversed(e_b))]
        V = [u(0.1, 0.45) * W for _ in range(nb)]
        hop = np.diag([e_d] + list(e_b))
        for k in range(nb):
            hop[0, k + 1] = hop[k + 1, 0] = V[k]
        p = {"U": U, "e_d": e_d, "e_bath": list(e_b), "V": V}
        return hubbard_cluster(hop, [U] + [0.0] * nb, 0.0, beta, family=family, params=p)
    # holstein_atom: Omega sets the ladder spacing, g^2/Omega^2 the mean phonon number.
    # With E(n) = -mu n + U n_up n_dn - (g^2/Omega)(n - c)^2 the two addition
    # energies are -mu - lam Omega (1 - 2c) and -mu + U - lam Omega (3 - 2c); mu is
    # drawn around their midpoint so the ladders sit about the Fermi level
    # instead of tens of Omega below it.
    omega0 = u(0.15, 0.35) * W
    lam = u(0.2, 2.5)                                 # g^2 / Omega^2
    g = math.sqrt(lam) * omega0
    U = u(0.0, 1.0) * W if u(0.0, 1.0) < 0.5 else 0.0
    offset = 1.0 if half else 0.0                     # coupling to (n - 1): symmetric at mu = U/2
    centre = U / 2.0 - lam * omega0 * (2.0 - 2.0 * offset)
    mu = centre if half else centre + u(-0.4, 0.4) * W
    nb = int(min(96, max(40, 12 * lam + 30)))
    p = {"omega0": omega0, "g": g, "U": U, "mu": mu, "offset": offset, "n_boson": nb}
    return holstein_atom(omega0, g, U, mu, beta, offset=offset, n_boson=nb, params=p)


def sample_exact_poles(beta: float, rng: torch.Generator, P_max: int = 12,
                       gamma_min: float = GAMMA_MIN, omega_c: float = 20.0,
                       families: tuple[str, ...] = EXACT_DISCRETE,
                       min_kept: float = 0.97, max_tries: int = 8):
    """One pole-closed training truth from an exactly diagonalised model.

    The model's delta spectrum is given a lifetime drawn log-uniformly from
    the resolution floor up, which makes it a pole set exactly -- the label
    needs no fit.  A draw is rejected (and redrawn) when the ``P_max``
    heaviest features hold less than ``min_kept`` of the weight or a line
    falls outside the band box.  Returns physical-unit lists
    ``(eps, gamma, a, b)`` with ``b = 0`` and the family name.
    """
    floor = 1.05 * gamma_min / beta
    for _ in range(max_tries):
        fam = families[int(torch.randint(len(families), (1,), generator=rng))]
        spec = sample_discrete_spectrum(fam, beta, rng)
        r = float(torch.rand((), generator=rng))
        eta = floor * math.exp(r * math.log(6.0))
        eps, gam, a, kept = spec.poles(eta, P_max)
        if kept >= min_kept and float(np.max(np.abs(eps))) < 0.9 * omega_c:
            return eps.tolist(), gam.tolist(), a.tolist(), [0.0] * len(a), fam
    return eps.tolist(), gam.tolist(), a.tolist(), [0.0] * len(a), fam
