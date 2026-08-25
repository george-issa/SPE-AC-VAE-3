r"""DQMC-like noise: correlated in imaginary time, and honest about it.

What DQMC noise actually looks like
-----------------------------------
Bin-to-bin fluctuations of :math:`G(\tau)` are **strongly correlated across
:math:`\tau`** -- neighbouring time slices come from the same configurations --
and the covariance eigenvalue spectrum is heavy-tailed, with the leading mode
often carrying ~90% of the trace.  Treating the noise as independent per slice
makes the reconstruction loss badly miscalibrated: it counts :math:`L_\tau`
independent constraints where the data supplies far fewer, so a fit that looks
like :math:`\chi^2 \approx 1` may be over-fitting a handful of noise modes.

Two consequences are built in here:

1. Training data is generated with correlated noise, not white noise, so the
   network sees realistic input.
2. The reconstruction loss whitens with the covariance, in the factorised form
   :math:`C = D R D` (:math:`D` diagonal standard deviations, :math:`R` the
   correlation matrix).  Factorising is not cosmetic: per-sample noise
   *amplitudes* differ while the correlation *structure* is a property of the
   simulation, so one :math:`R^{-1/2}` serves a whole batch and the per-sample
   scale is a cheap elementwise division.

Estimating :math:`R` from bins
------------------------------
With :math:`N_{\rm bins} \sim L_\tau` the sample covariance sits in the
Marchenko--Pastur regime: its small eigenvalues are noise, and inverting them
blows up the whitener.  The historically common fix -- truncating the
eigendecomposition at 99% of the trace -- silently discards exactly the
directions that encode high-frequency spectral structure (on one real cell it
kept 4 of 100 directions).  :func:`ledoit_wolf_correlation` instead shrinks
towards the identity with the closed-form optimal intensity (Ledoit & Wolf,
2004), keeping every direction at the cost of a mild, *quantified* bias.

That bias must be accounted for when reading :math:`\chi^2`.  Under shrinkage the
value at a perfect fit is not 1 but

.. math::
    \mathbb{E}[\chi^2] = \frac{1}{L}\sum_k \frac{\lambda_k}{\rho\mu + (1-\rho)\lambda_k}
    \;\le\; 1 ,

which :func:`expected_chi2` returns.  Reporting "we reached the noise floor"
against 1 rather than against this number is how a :math:`\chi^2 < 1` mystery
gets manufactured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "NoiseModel",
    "correlation_matrix",
    "sample_correlated_noise",
    "ledoit_wolf_correlation",
    "whitener",
    "expected_chi2",
    "chi2_target",
]


@dataclass
class NoiseModel:
    r"""Parameters of the synthetic DQMC-like noise process.

    Attributes
    ----------
    scale
        Overall standard deviation at the reference point, relative to
        :math:`\max_\tau |G|`.  Sampled log-uniformly across training so one
        model covers everything from a short exploratory run to a long
        production one.
    xi
        Correlation length in units of :math:`\beta` (i.e. in :math:`x`).
    proportional
        Fraction of the noise amplitude that follows :math:`|G(\tau)|` rather
        than being constant across :math:`\tau`.  Real DQMC sits between the two
        extremes, and which one dominates changes where in :math:`\tau` the data
        is informative -- so both are covered.
    nugget
        Fraction of the variance that is *independent* between slices.  Not a
        numerical convenience: a purely smooth correlation model is nearly
        rank-deficient (:math:`e^{-|\Delta x|/\xi}` on 100 slices at
        :math:`\xi = 0.4` has eigenvalues spanning six orders of magnitude), and
        whitening with its inverse demands an essentially exact fit along
        directions that carry no real information -- which inflates
        :math:`\chi^2` by many orders of magnitude and lets a handful of
        near-null directions dominate the gradient.  Real bin noise always has
        an independent component; including it keeps the whitener conditioned
        *and* keeps the model honest.
    """

    scale: float = 1e-3
    xi: float = 0.1
    proportional: float = 0.5
    nugget: float = 0.15

    @staticmethod
    def sample(rng: torch.Generator, device=None,
               scale_range: tuple[float, float] = (3e-5, 3e-2),
               xi_range: tuple[float, float] = (0.01, 0.4),
               nugget_range: tuple[float, float] = (0.05, 0.6)) -> "NoiseModel":
        """Draw the noise parameters.

        ``rng`` must be a **CPU** generator: these are four Python floats, and
        drawing them on an accelerator costs a device synchronisation each,
        which also drains any queued compute and serialises the whole training
        step.  Nothing here benefits from being on device.
        """
        def _lu(lo, hi):
            u = torch.rand((), generator=rng).item()
            return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        p = torch.rand((), generator=rng).item()
        n = _lu(*nugget_range)
        return NoiseModel(scale=_lu(*scale_range), xi=_lu(*xi_range),
                          proportional=p, nugget=n)


def correlation_matrix(
    x: torch.Tensor,
    xi: float,
    kind: str = "exp",
    nugget: float = 0.15,
) -> torch.Tensor:
    r"""Correlation matrix :math:`R` on a dimensionless time grid.

    .. math:: R = (1-n)\,R_{\rm smooth} + n\,I

    ``kind="exp"`` gives :math:`R_{ij} = e^{-|x_i-x_j|/\xi}` (Ornstein--Uhlenbeck,
    rough sample paths); ``kind="gauss"`` gives
    :math:`e^{-(x_i-x_j)^2/2\xi^2}` (smooth paths, a much heavier-tailed
    eigenvalue spectrum).  Sampling both across training covers the range of
    conditioning real covariances show.  ``nugget`` is the independent fraction
    (see :class:`NoiseModel`).
    """
    d = (x.reshape(-1, 1) - x.reshape(1, -1)).abs()
    if kind == "exp":
        R = torch.exp(-d / max(xi, 1e-8))
    elif kind == "gauss":
        R = torch.exp(-0.5 * (d / max(xi, 1e-8)) ** 2)
    else:
        raise ValueError(f"unknown correlation kind {kind!r}")
    n = float(min(max(nugget, 1e-6), 1.0))
    eye = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
    return (1.0 - n) * R + n * eye


def sample_correlated_noise(
    G: torch.Tensor,
    x: torch.Tensor,
    model: NoiseModel,
    rng: torch.Generator,
    kind: str = "exp",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Draw correlated noise for a batch of clean Green's functions.

    Returns
    -------
    (noise, sigma, R)
        ``noise`` and ``sigma`` are ``(B, L)``; ``R`` is the shared ``(L, L)``
        correlation matrix, which the loss whitens with.
    """
    B, L = G.shape
    amp = G.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    sigma = model.scale * amp * (
        (1.0 - model.proportional) + model.proportional * G.abs() / amp
    )
    sigma = sigma.clamp_min(1e-12)

    R = correlation_matrix(x, model.xi, kind=kind, nugget=model.nugget)
    Lch = torch.linalg.cholesky(R)
    w = torch.randn(B, L, generator=rng, device=G.device, dtype=G.dtype)
    return (w @ Lch.T) * sigma, sigma, R


def ledoit_wolf_correlation(
    bins: torch.Tensor,
    min_eigenvalue: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    r"""Shrunk correlation estimate from raw bins.

    Splits the sample covariance :math:`S = D R D` and shrinks the *correlation*
    matrix towards the identity,

    .. math:: R_{\rm LW} = \rho I + (1-\rho) R ,

    with :math:`\rho` the closed-form Ledoit--Wolf intensity.  Shrinking
    :math:`R` rather than :math:`C` keeps the per-slice variances exactly as
    measured -- they are well estimated even when the off-diagonal structure is
    not -- and confines the regularisation to the part that is genuinely
    ill-determined.

    **When Ledoit--Wolf is not enough.**  The intensity :math:`\rho` is chosen to
    minimise the distance to the identity target, so a *strongly* correlated
    :math:`R` -- which is far from the identity, giving a large :math:`d^2` --
    gets almost no shrinkage.  That is optimal in the Frobenius sense and
    inadequate for whitening: with :math:`N_{\rm bins} \lesssim L_\tau` the
    sample correlation is rank-deficient, LW returns :math:`\rho \sim 10^{-2}`,
    and the near-null directions still blow up.  Measured on a real
    site-Holstein cell (100 bins, 101 slices): :math:`\rho = 0.007`, and the
    calibrated perfect-fit target falls to :math:`\mathbb{E}[\chi^2] \approx
    0.11`.  In that regime "we reached :math:`\chi^2 \approx 1`" is not a
    statement about the noise floor at all.

    ``min_eigenvalue`` puts an explicit floor on the eigenvalues of
    :math:`R_{\rm LW}` (whose mean eigenvalue is 1 before flooring, since
    Ledoit--Wolf preserves the trace), which bounds the whitener's amplification
    by exactly :math:`1/\sqrt{\text{floor}}`.  Raise it whenever
    :math:`N_{\rm bins}` is not comfortably larger than :math:`L_\tau`, and
    always report :func:`chi2_target` alongside the result.

    Be clear about what the floor does and does not fix.  It bounds the
    whitener, which stops a handful of near-null directions from dominating the
    gradient.  It does **not** restore :math:`\mathbb{E}[\chi^2] = 1`: when the
    covariance is genuinely under-determined, no reweighting recovers
    information the bins do not contain.  The remedy for that is more bins or
    fewer time slices.

    Parameters
    ----------
    bins
        ``(N, L)`` raw bin measurements.
    min_eigenvalue
        Lower bound on the eigenvalues of the returned correlation matrix.
        ``0`` (default) is pure Ledoit--Wolf.

    Returns
    -------
    (sigma, R_lw, rho)
        ``sigma`` is ``(L,)``, ``R_lw`` is ``(L, L)``, ``rho`` in ``[0, 1]``.
    """
    X = bins.detach().cpu().to(torch.float64)   # device first: MPS has no float64
    N, L = X.shape
    if N < 2:
        raise ValueError("need at least two bins to estimate a covariance")
    Xc = X - X.mean(0, keepdim=True)
    S = (Xc.T @ Xc) / N
    sigma = torch.sqrt(torch.diagonal(S).clamp_min(1e-300))
    R = S / (sigma.reshape(-1, 1) * sigma.reshape(1, -1))

    Z = Xc / sigma.reshape(1, -1)                     # standardised bins
    d2 = ((R - torch.eye(L, dtype=R.dtype, device=R.device)) ** 2).sum()
    z_norm_sq = (Z * Z).sum(-1)
    b2 = (z_norm_sq ** 2).sum() / (N * N) - (R * R).sum() / N
    b2 = b2.clamp(min=0.0, max=float(d2))
    rho = float((b2 / d2).clamp(0.0, 1.0)) if float(d2) > 0 else 0.0

    R_lw = rho * torch.eye(L, dtype=R.dtype, device=R.device) + (1.0 - rho) * R

    if min_eigenvalue > 0.0:
        # Clamp and stop: renormalising the trace back to L afterwards would
        # scale every eigenvalue down by L/sum, pushing the smallest back below
        # the floor and quietly breaking the amplification bound the floor is
        # there to provide.  The trace grows slightly instead, which costs
        # nothing -- chi2_target reads the calibration off whatever matrix is
        # actually used, so no downstream number depends on tr(R) = L.
        w, V = torch.linalg.eigh(0.5 * (R_lw + R_lw.T))
        R_lw = V @ torch.diag(w.clamp_min(float(min_eigenvalue))) @ V.T

    return (sigma.to(device=bins.device, dtype=bins.dtype),
            R_lw.to(device=bins.device, dtype=bins.dtype), rho)


def chi2_target(R_used: torch.Tensor, R_sample: torch.Tensor) -> float:
    r"""Perfect-fit :math:`\chi^2` for *any* regularised correlation estimate.

    .. math:: \mathbb{E}[\chi^2] = \tfrac{1}{L}\operatorname{tr}\!
              \big(R_{\rm used}^{-1} R_{\rm true}\big)

    with the sample correlation standing in for the truth.  Generalises
    :func:`expected_chi2` (which is the closed form for plain Ledoit--Wolf) to
    an arbitrary whitener, including an eigenvalue-floored one.  Equals 1 exactly
    when the estimate used *is* the sample correlation.

    The usual caveat applies and is not small: the smallest sample eigenvalues
    are downward-biased in the Marchenko--Pastur regime, so the true target is
    somewhat higher than what this returns from :math:`S` alone.  It is a
    *lower* bound on where the noise floor sits, and far better than assuming 1.
    """
    A = R_used.detach().cpu().to(torch.float64)
    B = R_sample.detach().cpu().to(torch.float64)
    L = A.shape[0]
    return float(torch.linalg.solve(A, B).diagonal().sum() / L)


def whitener(R: torch.Tensor) -> torch.Tensor:
    r""":math:`R^{-1/2}` by symmetric eigendecomposition.

    Always computed in float64 on the CPU, then cast back: inverting the small
    eigenvalues of a correlation matrix in float32 loses accuracy exactly where
    the whitener is largest, and MPS has no float64 at all.  This runs once per
    batch on an ``(L, L)`` matrix, so the round trip costs nothing.
    """
    Rd = R.detach().cpu().to(torch.float64)   # device first: MPS has no float64
    w, V = torch.linalg.eigh(0.5 * (Rd + Rd.T))
    w = w.clamp_min(1e-12)
    out = V @ torch.diag(w.rsqrt()) @ V.T
    return out.to(device=R.device, dtype=R.dtype)


def expected_chi2(R_sample: torch.Tensor, rho: float) -> float:
    r"""Calibrated :math:`\mathbb{E}[\chi^2]` at a perfect fit under LW shrinkage.

    .. math::
        \mathbb{E}[\chi^2] = \frac{1}{L}\sum_k \frac{\lambda_k}{\rho\mu + (1-\rho)\lambda_k},
        \qquad \mu = \tfrac1L\sum_k \lambda_k = 1 ,

    with :math:`\lambda_k` the eigenvalues of the *unshrunk* sample correlation
    matrix.  Always :math:`\le 1` by Jensen (the summand is concave), with the
    gap growing like :math:`\rho(1-\rho)\,\mathrm{Var}(\lambda)`.  Since DQMC
    correlation spectra are heavy-tailed, even modest shrinkage pulls the target
    visibly below 1 -- so this, not 1, is the number a converged fit should be
    compared against.
    """
    lam = torch.linalg.eigvalsh(
        R_sample.detach().cpu().to(torch.float64)).clamp_min(0.0)
    mu = lam.mean()
    return float((lam / (rho * mu + (1.0 - rho) * lam)).mean())


def structured_shrunk_correlation(
    bins: torch.Tensor,
    x: torch.Tensor,
    min_eigenvalue: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    r"""Shrink the sample correlation towards an exponential-decay target.

    Ledoit--Wolf's identity target is the least informed choice possible for
    DQMC bins, whose true :math:`\tau`-correlations decay smoothly -- so LW
    assigns almost no shrinkage exactly when regularisation is needed most
    (see :func:`ledoit_wolf_correlation`).  Here the target is the physical
    one, :math:`T_{ij} = e^{-|x_i - x_j|/\xi}` with :math:`\xi` read off the
    sample correlation's own diagonal decay, and the intensity is the
    Schafer--Strimmer optimum for an arbitrary target,

    .. math:: \rho^* = \sum_{i\ne j} \widehat{\mathrm{Var}}(r_{ij}) \Big/
              \sum_{i\ne j} (r_{ij} - t_{ij})^2 .

    Because the target now carries the real off-diagonal structure, large
    intensities are cheap: the estimator can shrink hard without flattening
    correlations that are actually there.

    Returns ``(sigma, R_shrunk, rho, xi)``.
    """
    X = bins.detach().cpu().to(torch.float64)
    N, L = X.shape
    if N < 2:
        raise ValueError("need at least two bins to estimate a covariance")
    xd = x.detach().cpu().to(torch.float64).reshape(-1)
    Xc = X - X.mean(0, keepdim=True)
    S = (Xc.T @ Xc) / N
    sigma = torch.sqrt(torch.diagonal(S).clamp_min(1e-300))
    R = S / (sigma.reshape(-1, 1) * sigma.reshape(1, -1))

    # xi from the mean superdiagonal decay: ln r(lag) ~ -dx(lag)/xi over the
    # lags where the mean correlation is solidly positive
    lags, logr, dxs = [], [], []
    for k in range(1, L):
        rk = float(torch.diagonal(R, offset=k).mean())
        if rk < 0.05:
            break
        lags.append(k)
        logr.append(math.log(rk))
        dxs.append(float((xd[k:] - xd[:-k]).mean()))
    if len(lags) >= 2:
        A = torch.tensor(dxs, dtype=torch.float64)
        b = -torch.tensor(logr, dtype=torch.float64)
        xi = float((A * A).sum() / (A * b).sum().clamp_min(1e-12))
        xi = max(min(xi, 100.0 * float(xd[-1] - xd[0])), 1e-6)
    else:
        xi = 1e-6                                  # effectively uncorrelated
    T = torch.exp(-(xd.reshape(-1, 1) - xd.reshape(1, -1)).abs() / xi)

    Z = Xc / sigma.reshape(1, -1)
    off = ~torch.eye(L, dtype=torch.bool)
    # Var(r_ij) ~ (1/N^2) sum_k (z_ki z_kj - r_ij)^2  (moment estimate)
    W2 = torch.einsum("ki,kj->ij", Z * Z, Z * Z) / N - R * R
    var_r = (W2 / N).clamp_min(0.0)
    d2 = ((R - T)[off] ** 2).sum()
    rho = float((var_r[off].sum() / d2).clamp(0.0, 1.0)) if float(d2) > 0 else 1.0

    R_sh = rho * T + (1.0 - rho) * R
    if min_eigenvalue > 0.0:
        w, V = torch.linalg.eigh(0.5 * (R_sh + R_sh.T))
        R_sh = V @ torch.diag(w.clamp_min(float(min_eigenvalue))) @ V.T
    return (sigma.to(device=bins.device, dtype=bins.dtype),
            R_sh.to(device=bins.device, dtype=bins.dtype), rho, xi)
