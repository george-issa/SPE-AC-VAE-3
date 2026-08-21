r"""Minimal-pole extraction from Matsubara data (MPM/AAA-style, grid-free).

The gradient-based refit inherits texture because the comb directions are
:math:`\chi^2`-flat and local descent leaves debris there.  This module takes
the opposite route, after Zhang & Gull (PRB **110**, 035154 (2024)) and the
minimal pole representation (Zhang, Erpenbeck, Yu & Gull, JCP **162**,
214111 (2025)): work on the Matsubara samples, build the data's rational
structure by adaptive AAA interpolation, and let the **noise floor decide the
rank** -- modes below it are never instantiated, so texture does not exist in
the output, while high-contrast structure (band edges) sits above the floor
and is captured.

Pipeline: bins :math:`\to` spline-exact Fourier transform to
:math:`G(i\omega_n)` with propagated per-point noise :math:`\to` AAA with a
noise-scaled stopping rule :math:`\to` poles of the barycentric interpolant
:math:`\to` causal filtering + weight least squares :math:`\to` (caller)
constrained polish, certificate, ledger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .poles import PoleSet

__all__ = ["matsubara_from_bins", "aaa_poles", "mpm_extract", "MPMReport"]


def _interval_moments(om: float, h: float, kmax: int = 5) -> np.ndarray:
    r""":math:`I_k = \int_0^h t^k e^{i\omega t} dt`, stable for all sizes of ωh."""
    out = np.zeros(kmax + 1, dtype=complex)
    z = 1j * om
    if abs(om * h) < 0.5:
        # series: I_k = h^{k+1} sum_m (i om h)^m / (m! (k+m+1))
        for k in range(kmax + 1):
            term, s = 1.0 + 0j, 0.0 + 0j
            for m in range(0, 40):
                s += term / (k + m + 1)
                term *= z * h / (m + 1)
                if abs(term) < 1e-20:
                    break
            out[k] = h ** (k + 1) * s
    else:
        e = np.exp(z * h)
        out[0] = (e - 1.0) / z
        for k in range(1, kmax + 1):
            out[k] = (h ** k * e - k * out[k - 1]) / z
    return out


def matsubara_from_bins(
    G_mean: torch.Tensor,
    tau: torch.Tensor,
    beta: float,
    n_max: int,
    cov_mean: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Spline-exact :math:`F(i\omega_n) = -\int_0^\beta e^{i\omega_n\tau} G(\tau)\,d\tau`.

    A quintic spline of the measured mean is integrated against
    :math:`e^{i\omega_n\tau}` *analytically on every interval* (series/
    recursion switch keeps the moments stable at small and large
    :math:`\omega h`), so the transform error is the interpolation error
    alone -- :math:`O(h^6)`, validated against the closed-form pole Matsubara.
    With this sign convention a pole set satisfies
    :math:`F(i\omega_n) = \sum_p r_p/(i\omega_n - z_p)`.

    Returns ``(omega_n, F, sigma_F)``; ``sigma_F`` is the per-point noise
    propagated through the transform's exact linear weights (from
    ``cov_mean``; zeros if not given).
    """
    from scipy.interpolate import PPoly, make_interp_spline

    t = tau.detach().double().cpu().numpy()
    g = G_mean.detach().double().cpu().numpy()
    L = t.size
    om = (2.0 * np.arange(n_max) + 1.0) * math.pi / float(beta)
    k_spl = 5 if L >= 8 else 3
    kmax = k_spl

    def ft_of(values: np.ndarray) -> np.ndarray:
        pp = PPoly.from_spline(make_interp_spline(t, values, k=k_spl))
        # PPoly may carry repeated breakpoints from the B-spline knots; use
        # its own x
        xs = pp.x
        hs = np.diff(xs)
        C = pp.c[::-1, :]                      # ascending coeffs, (k+1, n_int)
        out = np.zeros(n_max, dtype=complex)
        for j, o in enumerate(om):
            # uniform grids: one moment set reused across equal-h intervals
            acc = 0.0 + 0j
            cache: dict[float, np.ndarray] = {}
            for i in range(hs.size):
                if hs[i] <= 0:
                    continue
                key = round(float(hs[i]), 14)
                I = cache.get(key)
                if I is None:
                    I = _interval_moments(o, float(hs[i]), kmax=kmax)
                    cache[key] = I
                acc += np.exp(1j * o * xs[i]) * np.dot(C[: kmax + 1, i], I)
            out[j] = acc
        return out

    F = -ft_of(g)

    if cov_mean is not None:
        # exact linear weights by applying the transform to unit vectors
        W = np.zeros((n_max, L), dtype=complex)
        for basis in range(L):
            e_vec = np.zeros(L)
            e_vec[basis] = 1.0
            W[:, basis] = -ft_of(e_vec)
        C = cov_mean.detach().double().cpu().numpy()
        var = np.einsum("ij,jk,ik->i", W, C, W.conj()).real
        sigma_F = np.sqrt(np.maximum(var, 0.0))
    else:
        sigma_F = np.zeros(n_max)
    return om, F, sigma_F


def aaa_poles(
    z: np.ndarray,
    f: np.ndarray,
    tol: float,
    max_terms: int = 40,
    sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    r"""AAA rational approximation; returns the interpolant's poles.

    Standard greedy AAA (Nakatsukasa, Sete & Trefethen): add the sample of
    largest residual as a support point, get barycentric weights from the
    smallest singular vector of the Loewner matrix, stop when the residual
    falls below ``tol`` -- the noise floor, which is what makes the *rank*
    (and hence the pole count) data-determined.  Poles are the finite
    generalised eigenvalues of the barycentric pencil.
    """
    z = np.asarray(z, complex)
    f = np.asarray(f, complex)
    # Per-point noise weighting: an unweighted absolute stopping rule grants
    # the quietest samples up to tol/sigma_j ~ 13 sigma of slack (measured),
    # under-resolving the rank the data actually supports.  With sigma given,
    # residuals are measured in units of each point's own noise and ``tol``
    # is in sigma units.
    wgt = (1.0 / np.maximum(np.asarray(sigma, float), 1e-300)
           if sigma is not None else np.ones(f.size))
    M = z.size
    mask = np.ones(M, bool)
    support: list[int] = []
    R = np.full(M, f.mean(), complex)
    w = np.array([1.0 + 0j])

    for _ in range(min(max_terms, M - 2)):
        j = int(np.argmax(np.abs(f - R) * wgt * mask))
        support.append(j)
        mask[j] = False
        zs, fs = z[support], f[support]
        # Loewner matrix on the remaining samples
        zi, fi = z[mask], f[mask]
        C = 1.0 / (zi[:, None] - zs[None, :])
        A = fi[:, None] * C - C * fs[None, :]
        _, _, Vh = np.linalg.svd(A)
        w = Vh[-1].conj()
        num = C @ (w * fs)
        den = C @ w
        R = np.full(M, np.nan, complex)
        R[mask] = num / den
        R[support] = f[support]
        err = np.max(np.abs(f[mask] - R[mask]) * wgt[mask]) \
            if mask.any() else 0.0
        if err < tol:
            break

    zs = z[support]
    m = zs.size
    # generalized eigenvalue pencil for the barycentric poles
    B = np.eye(m + 1, dtype=complex)
    B[0, 0] = 0.0
    E = np.zeros((m + 1, m + 1), dtype=complex)
    E[0, 1:] = w
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(zs)
    from scipy.linalg import eig
    ev = eig(E, B, right=False)
    poles = ev[np.isfinite(ev)]
    return poles, m



def _causal_poles_of_F(om: np.ndarray, F: np.ndarray, norm: float,
                       tol: float, max_terms: int = 40) -> np.ndarray:
    r"""Causal pole locations of :math:`F`, via AAA on the tail-free function.

    A barycentric interpolant of type :math:`(m{-}1, m{-}1)` is a pole sum
    *plus a constant*, while :math:`F` has none -- discarding the constant
    corrupts the far Matsubara tail (measured: an 8.5% pole-sum residual on
    an AAA fit that was itself accurate).  So AAA runs on
    :math:`H(z) = z F(z) - \mathrm{norm}`, which decays like :math:`1/z`
    (no constant term possible), has exactly :math:`F`'s pole locations, and
    whose value at the origin-side samples is well scaled.  ``tol`` is in
    units of :math:`H`.
    """
    H = 1j * om * F - norm
    poles, _ = aaa_poles(1j * om, H, tol=tol, max_terms=max_terms)
    return poles[np.imag(poles) < -1e-12]


@dataclass
class MPMReport:
    """What the extraction did."""

    n_matsubara: int
    noise_floor: float
    aaa_terms: int
    n_poles_raw: int          #: causal poles before weight cleanup
    n_poles: int              #: shipped
    matsubara_residual: float  #: rel. misfit of the final pole set on F(iw_n)

    def __str__(self) -> str:
        return (f"MPM: {self.n_matsubara} Matsubara points at floor "
                f"{self.noise_floor:.2e}; AAA {self.aaa_terms} terms -> "
                f"{self.n_poles_raw} causal poles -> {self.n_poles} kept; "
                f"residual {self.matsubara_residual:.2e}")


def mpm_extract(
    G_mean: torch.Tensor,
    tau: torch.Tensor,
    beta: float,
    cov_mean: torch.Tensor | None = None,
    n_max: int = 40,
    tol_scale: float = 5.0,
    min_tol: float = 1e-9,
    weight_floor: float = 1e-4,
    two_pass: bool = True,
) -> tuple[PoleSet, MPMReport]:
    r"""Bins-mean :math:`\to` minimal pole set, by rank not by descent.

    ``tol_scale`` multiplies the largest propagated Matsubara noise to set
    the AAA stopping tolerance (the SVD-truncation analogue); ``min_tol``
    floors it for effectively noiseless input.  Causality is enforced by
    keeping lower-half-plane poles only; weights come from a final
    noise-weighted least squares on all Matsubara points; near-zero-weight
    poles (Froissart doublets) are dropped and the sum rule is restored by
    rescaling.  The caller owns any further constrained polish, certificate
    and ledger.
    """
    om, F, sig = matsubara_from_bins(G_mean, tau, beta, n_max, cov_mean)
    floor = max(float(sig.max()) * tol_scale if sig.size else 0.0, min_tol)

    # Two-pass reference subtraction.  With 1/omega^2 spectral tails, G'(tau)
    # is log-singular at the endpoints and no spline transform beats ~1e-5
    # (measured; a smooth-tailed G transforms at 1e-11).  Pass 1 extracts a
    # crude pole set from the 1e-5-accurate transform; its closed-form
    # Matsubara series is exact, and the tau-residual G - G_ref is ~1e-5 in
    # amplitude with correspondingly weak singular content, so the pass-2
    # transform is accurate to the noise floor.  Self-referencing: no external
    # prior needed (a caller-supplied reference would work identically).
    if two_pass:
        # upper Matsubara axis only: a strictly causal pole set cannot
        # satisfy the reflection identity F(-iw) = F(iw)* as a function, so
        # reflected samples are contradictory data that corrupt the fit
        # (measured: junk poles, 4.5% residual).
        floor1 = max(floor, 3e-5)
        p1, _ = aaa_poles(1j * om, F, tol=floor1)
        causal1 = p1[np.imag(p1) < -1e-12]
        if causal1.size:
            K1 = np.concatenate(
                [1.0 / (1j * om[:, None] - causal1[None, :]),
                 np.ones((om.size, 1), complex)], axis=1)
            sol1, *_ = np.linalg.lstsq(K1, F, rcond=None)
            r1 = sol1[:-1]                    # constant column absorbed, dropped
            from .forward import PoleGreens, SpectralQuadrature
            ref = PoleSet(
                eps=torch.tensor(np.real(causal1)).unsqueeze(0),
                gamma=torch.tensor(-np.imag(causal1)).unsqueeze(0),
                a=torch.tensor(np.real(r1)).unsqueeze(0),
                b=torch.tensor(np.imag(r1)).unsqueeze(0)).to(torch.float64)
            quad_ref = SpectralQuadrature.for_beta(float(beta),
                                                   dtype=torch.float64)
            x = (tau.double() / float(beta))
            G_ref = PoleGreens(quad_ref)(
                ref.to_dimensionless(float(beta)), x)[0]
            _, F_res, _ = matsubara_from_bins(
                G_mean.double() - G_ref, tau, beta, n_max)
            F_ref_exact = (r1[None, :]
                           / (1j * om[:, None] - causal1[None, :])).sum(1)
            F = F_res + F_ref_exact

    # plain-F fit: multiplying by omega (the H-transform variant) amplifies
    # exactly the noisiest tail samples and starves the fit of modes at the
    # noise floor (measured: 2-3 spiky poles, chi2/tgt ~1e4).  The barycentric
    # constant is handled where it lives instead: an explicit constant column
    # in the residue least squares, absorbed and then dropped.
    if sig.size and sig.max() > 0:
        poles_c, m = aaa_poles(1j * om, F, tol=float(tol_scale), sigma=sig)
    else:
        tol_F = max(floor, 1e-6 * float(np.abs(F).max()))
        poles_c, m = aaa_poles(1j * om, F, tol=tol_F)
    causal = poles_c[np.imag(poles_c) < -1e-12]
    n_raw = causal.size
    if n_raw == 0:
        raise RuntimeError("MPM found no causal poles; raise n_max or check data")

    # noise-weighted least squares for complex residues on the causal poles
    Wt = 1.0 / np.maximum(sig, max(floor / tol_scale, 1e-30))
    K = np.concatenate([1.0 / (1j * om[:, None] - causal[None, :]),
                        np.ones((om.size, 1), complex)], axis=1)
    sol, *_ = np.linalg.lstsq(K * Wt[:, None], F * Wt, rcond=None)
    r_c, const = sol[:-1], sol[-1]

    a = np.real(r_c)
    keep = a > weight_floor * max(a.sum(), 1e-300)
    causal, r_c, a = causal[keep], r_c[keep], a[keep]
    b = np.imag(r_c)
    # exact constraint restoration: sum a = measured norm, sum b = 0
    norm = float(G_mean[0] + G_mean[-1])
    a = a * (norm / a.sum())
    b = b - b.mean() if b.size else b
    b = np.clip(b, -a, a)
    b = b - (b.sum() / a.sum()) * a          # zero-sum, weight-proportional
    b = np.clip(b, -a, a)

    ps = PoleSet(
        eps=torch.tensor(np.real(causal)).unsqueeze(0).float(),
        gamma=torch.tensor(-np.imag(causal)).unsqueeze(0).float(),
        a=torch.tensor(a).unsqueeze(0).float(),
        b=torch.tensor(b).unsqueeze(0).float(),
    )
    r_final = a + 1j * b
    F_fit = (r_final[None, :] / (1j * om[:, None] - causal[None, :])).sum(1)
    resid = float(np.linalg.norm(F_fit - F) / np.linalg.norm(F))
    rep = MPMReport(n_matsubara=n_max, noise_floor=floor, aaa_terms=m,
                    n_poles_raw=n_raw, n_poles=int(a.size),
                    matsubara_residual=resid)
    return ps, rep
