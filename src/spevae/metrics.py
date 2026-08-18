r"""Spectral metrics: peaks, heights, gaps.

Aggregate distances (:math:`L^2`, :math:`W_1`) are what you optimise; *these* are
what a condensed-matter reader actually checks.  "Is the gap there and is it the
right size?" is not answered by an :math:`L^2` norm -- two spectra can be close
in :math:`L^2` with one gapped and one not, and far apart in :math:`L^2` while
agreeing on every feature a physicist would name.

All functions take :math:`A(\omega)` on a grid and are pure NumPy/Torch, with no
model dependency, so they apply equally to this package's output, to a
maximum-entropy result, and to a `SmoQyDEAC.jl` reference -- which is the point:
a three-way comparison has to be scored with one ruler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

__all__ = [
    "Peak",
    "smooth_to_resolution",
    "find_peaks",
    "matched_peak_error",
    "gap_metrics",
    "spectral_report",
]


@dataclass
class Peak:
    omega: float
    height: float
    weight: float           # integrated weight of the peak's basin
    prominence: float


def _as_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def smooth_to_resolution(A, omega, width: float):
    r"""Gaussian-smooth a spectrum to a stated frequency resolution.

    Necessary before counting peaks in a *comparison*.  Stochastic methods
    return visibly rough spectra -- a differential-evolution continuation of the
    same two-peak spectrum that maximum entropy renders with 3 maxima can carry
    50 -- so raw peak counts measure how noisy each method is, not whether the
    physics agrees.  Smoothing both sides to a common, physically meaningful
    scale asks the question that was actually intended.

    The natural scale is the resolution limit :math:`\pi/\beta` (the first
    Matsubara frequency): structure finer than that is not recoverable from data
    on :math:`\tau\in[0,\beta]`, so counting it as a feature is counting noise.
    """
    a = _as_numpy(A).astype(float)
    w = _as_numpy(omega).astype(float)
    if width <= 0:
        return a
    h = float(np.mean(np.diff(w)))
    n = max(int(np.ceil(4.0 * width / h)), 1)
    k = np.exp(-0.5 * (np.arange(-n, n + 1) * h / width) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(a, n, mode="edge"), k, mode="valid")


def find_peaks(
    A,
    omega,
    rel_prominence: float = 0.05,
    rel_height: float = 0.02,
    resolution: float = 0.0,
) -> list[Peak]:
    r"""Local maxima of :math:`A(\omega)`, filtered by prominence.

    Parameters
    ----------
    A, omega
        1-D arrays of equal length; ``omega`` ascending.
    rel_prominence
        Minimum prominence as a fraction of :math:`\max A`.  Prominence -- the
        drop required to reach a higher peak -- rather than raw height, because
        a shoulder on a large peak is not a separate feature, and a small but
        isolated satellite is.
    rel_height
        Minimum height as a fraction of :math:`\max A`.
    resolution
        Gaussian width to smooth to first (see :func:`smooth_to_resolution`).
        Pass :math:`\pi/\beta` when comparing methods; leave at 0 to inspect a
        single spectrum as returned.

    Returns
    -------
    list[Peak]
        Sorted by descending height.
    """
    w = _as_numpy(omega).astype(float)
    a = smooth_to_resolution(A, w, resolution)
    if a.ndim != 1:
        raise ValueError("find_peaks expects a single spectrum")
    amax = float(a.max()) if a.size else 0.0
    if amax <= 0:
        return []

    idx = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:])) + 1
    peaks: list[Peak] = []
    for i in idx:
        h = float(a[i])
        if h < rel_height * amax:
            continue
        # Prominence: descend both ways until the curve rises above this peak.
        left = a[:i]
        higher_l = np.flatnonzero(left > h)
        base_l = float(left[higher_l[-1]:].min()) if higher_l.size else float(left.min()) if left.size else 0.0
        right = a[i + 1:]
        higher_r = np.flatnonzero(right > h)
        base_r = float(right[:higher_r[0]].min()) if higher_r.size else float(right.min()) if right.size else 0.0
        prom = h - max(base_l, base_r)
        if prom < rel_prominence * amax:
            continue
        # Basin weight: integrate between the half-height crossings.
        half = 0.5 * h
        lo = i
        while lo > 0 and a[lo] > half:
            lo -= 1
        hi = i
        while hi < a.size - 1 and a[hi] > half:
            hi += 1
        weight = float(np.trapezoid(a[lo:hi + 1], w[lo:hi + 1])) if hi > lo else 0.0
        peaks.append(Peak(omega=float(w[i]), height=h, weight=weight, prominence=prom))

    return sorted(peaks, key=lambda p: -p.height)


def matched_peak_error(
    A_pred,
    A_true,
    omega,
    rel_prominence: float = 0.05,
    resolution: float = 0.0,
) -> dict:
    r"""Compare peak sets by optimal matching.

    Peaks are an unordered set, so the comparison is a linear assignment on
    :math:`|\Delta\omega|` (Hungarian), not a positional zip.  Unmatched peaks on
    either side are reported as counts rather than folded into the averages: a
    method that invents a spurious satellite and a method that misses a real one
    are different errors, and averaging them together hides both.

    Returns
    -------
    dict
        ``n_pred``, ``n_true``, ``n_matched``, ``d_omega`` (mean absolute
        position error over matched peaks), ``d_omega_max``, ``rel_d_height``
        (mean relative height error), ``missed``, ``spurious``.
    """
    p = find_peaks(A_pred, omega, rel_prominence, resolution=resolution)
    t = find_peaks(A_true, omega, rel_prominence, resolution=resolution)
    out = {"n_pred": len(p), "n_true": len(t), "n_matched": 0,
           "d_omega": float("nan"), "d_omega_max": float("nan"),
           "rel_d_height": float("nan"),
           "missed": len(t), "spurious": len(p)}
    if not p or not t:
        return out

    cost = np.abs(np.array([[pi.omega - tj.omega for tj in t] for pi in p]))
    rows, cols = linear_sum_assignment(cost)
    d_w = np.array([abs(p[r].omega - t[c].omega) for r, c in zip(rows, cols)])
    d_h = np.array([abs(p[r].height - t[c].height) / max(t[c].height, 1e-30)
                    for r, c in zip(rows, cols)])
    out.update(n_matched=len(rows), d_omega=float(d_w.mean()),
               d_omega_max=float(d_w.max()), rel_d_height=float(d_h.mean()),
               missed=len(t) - len(rows), spurious=len(p) - len(rows))
    return out


def gap_metrics(A, omega, rel_threshold: float = 0.05) -> dict:
    r"""Detect and measure a spectral gap around :math:`\omega = 0`.

    A gap is reported when :math:`A(\omega) < \text{rel\_threshold}\cdot\max A`
    over a contiguous interval containing (or adjacent to) the Fermi level.  The
    threshold is relative because absolute smallness is not meaningful -- what
    matters is suppression relative to the coherent weight elsewhere.

    Returns
    -------
    dict
        ``gapped`` (bool), ``gap`` (width), ``edge_lower``, ``edge_upper``,
        ``A_at_zero``, ``rel_A_at_zero``.
    """
    a = _as_numpy(A).astype(float)
    w = _as_numpy(omega).astype(float)
    amax = float(a.max()) if a.size else 0.0
    i0 = int(np.argmin(np.abs(w)))
    a0 = float(a[i0])
    out = {"gapped": False, "gap": 0.0, "edge_lower": float("nan"),
           "edge_upper": float("nan"), "A_at_zero": a0,
           "rel_A_at_zero": a0 / amax if amax > 0 else float("nan")}
    if amax <= 0 or a0 >= rel_threshold * amax:
        return out

    thr = rel_threshold * amax
    lo = i0
    while lo > 0 and a[lo] < thr:
        lo -= 1
    hi = i0
    while hi < a.size - 1 and a[hi] < thr:
        hi += 1
    out.update(gapped=True, gap=float(w[hi] - w[lo]),
               edge_lower=float(w[lo]), edge_upper=float(w[hi]))
    return out


def spectral_report(A_pred, A_true, omega, rel_prominence: float = 0.05,
                    resolution: float = 0.0) -> dict:
    r"""Everything a results table should carry for one spectrum pair.

    Combines the aggregate distances with the feature-level comparisons, plus
    ``min_A`` -- the most negative excursion of the predicted spectrum.  In the
    default (real-residue) mode that is identically zero by construction, and
    reporting it is how a reader confirms it rather than taking it on trust.

    Pass ``resolution = pi/beta`` when comparing *methods*: the peak counts and
    the matched-peak errors are otherwise dominated by how rough each solver's
    output happens to be rather than by where the physics is.  The aggregate
    distances (:math:`L^2`, :math:`L^\infty`, :math:`W_1`) are always computed on
    the unsmoothed curves, because roughness is a real property of an answer and
    should not be hidden.
    """
    a_p = _as_numpy(A_pred).astype(float)
    a_t = _as_numpy(A_true).astype(float)
    w = _as_numpy(omega).astype(float)

    l2 = float(np.sqrt(np.trapezoid((a_p - a_t) ** 2, w)))
    l2_rel = l2 / max(float(np.sqrt(np.trapezoid(a_t ** 2, w))), 1e-30)
    linf = float(np.max(np.abs(a_p - a_t)))

    cdf_p = np.concatenate([[0.0], np.cumsum(0.5 * (a_p[1:] + a_p[:-1]) * np.diff(w))])
    cdf_t = np.concatenate([[0.0], np.cumsum(0.5 * (a_t[1:] + a_t[:-1]) * np.diff(w))])
    cdf_p /= max(cdf_p[-1], 1e-30)
    cdf_t /= max(cdf_t[-1], 1e-30)
    w1 = float(np.trapezoid(np.abs(cdf_p - cdf_t), w))

    rep = {"l2": l2, "l2_rel": l2_rel, "linf": linf, "w1": w1,
           "norm_pred": float(np.trapezoid(a_p, w)),
           "norm_true": float(np.trapezoid(a_t, w)),
           "min_A_pred": float(a_p.min()),
           "resolution": float(resolution)}
    rep.update({f"peak_{k}": v for k, v in
                matched_peak_error(a_p, a_t, w, rel_prominence,
                                   resolution=resolution).items()})
    gp = gap_metrics(smooth_to_resolution(a_p, w, resolution), w)
    gt = gap_metrics(smooth_to_resolution(a_t, w, resolution), w)
    rep.update({"gap_pred": gp["gap"], "gap_true": gt["gap"],
                "gapped_pred": gp["gapped"], "gapped_true": gt["gapped"],
                "gap_error": abs(gp["gap"] - gt["gap"])})
    return rep
