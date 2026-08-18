r"""A common result type for every analytic-continuation method.

Three solvers with three interfaces produce one object, so that comparison code
never branches on which method produced a spectrum, and so that adding a fourth
method means writing a wrapper rather than editing the comparator.

Why the reconstruction is carried alongside the spectrum
--------------------------------------------------------
:math:`A(\omega)` alone cannot be checked.  Every method returns *something*, and
a plausible-looking curve that does not reproduce the input :math:`G(\tau)` is
worse than useless -- it looks like a result.  Every wrapper therefore back-
transforms its own answer through the *same* forward map
(:mod:`spevae.physics.forward`), independent of whatever kernel discretisation
the solver used internally, and reports the resulting :math:`\chi^2`.  A
disagreement between a solver's internal :math:`\chi^2` and this one is a
convention mismatch, and is worth catching before it becomes a figure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.real import MeasuredGreens
from ..physics.forward import SpectralQuadrature
from ..physics.kernel import fermi_kernel

__all__ = ["ContinuationResult", "backtransform", "reconstruction_chi2"]


def backtransform(
    A: np.ndarray,
    omega: np.ndarray,
    tau: np.ndarray,
    beta: float,
) -> np.ndarray:
    r"""Independent forward map: :math:`G(\tau) = \int A(\omega) K(\tau,\omega)\,d\omega`.

    Trapezoid on whatever :math:`\omega` grid the method returned.  Deliberately
    *not* the solver's own back-transform: using one map for all methods is what
    makes the reported :math:`\chi^2` values comparable, and it catches a
    convention error in any single wrapper.

    **Grid truncation.**  The integral runs only over the grid supplied, so any
    weight outside it is missed.  For a grid-based solver that is correct -- its
    answer *is* the grid, and it has no weight elsewhere.  For a method whose
    output is analytic and fat-tailed (a pole expansion decays as
    :math:`1/\omega^2`) it is not: the missing tail lands almost entirely at
    :math:`\tau \in \{0, \beta\}`, where :math:`K \to 1`, and can reach
    :math:`10^{-2}` for broad poles on a :math:`\pm 20` grid.  Such a method
    should pass its own exact reconstruction to
    :meth:`ContinuationResult.attach_reconstruction` instead of relying on this.
    """
    w = torch.as_tensor(np.asarray(omega), dtype=torch.float64)
    a = torch.as_tensor(np.asarray(A), dtype=torch.float64)
    t = torch.as_tensor(np.asarray(tau), dtype=torch.float64)
    K = fermi_kernel((t / beta).reshape(1, -1), (beta * w).reshape(-1, 1))  # (M, L)
    return torch.trapezoid(a.reshape(-1, 1) * K, w, dim=0).numpy()


def reconstruction_chi2(
    G_pred: np.ndarray,
    mg: MeasuredGreens,
    min_eigenvalue: float = 0.0,
    against: str = "mean",
) -> float:
    r"""Covariance-whitened :math:`\chi^2` of a reconstruction against the data.

    ``against="mean"`` compares to the bin mean with the standard error of the
    mean (the usual choice when a method returns one spectrum for the whole
    dataset); ``against="bin"`` uses the single-bin scale instead.  The two
    differ by :math:`N_{\rm bins}`, so mixing them across a comparison table
    would be meaningless -- hence the explicit argument.
    """
    cov = mg.covariance(min_eigenvalue)
    n = cov["n_independent"]
    sigma = cov["sigma"]
    if against == "mean":
        sigma = sigma / math.sqrt(mg.n_bins)
    elif against != "bin":
        raise ValueError(f"against must be 'mean' or 'bin', got {against!r}")
    d = (torch.as_tensor(G_pred, dtype=torch.float64)[:n]
         - mg.mean.double()[:n]) / sigma.double()
    d = d @ cov["inv_sqrt_R"].double().T
    return float((d * d).sum() / n)


@dataclass
class ContinuationResult:
    r"""One method's answer for one dataset.

    Attributes
    ----------
    method
        Short label used in tables and plots, e.g. ``"spevae"``, ``"maxent"``.
    omega, A
        Real-frequency grid and spectral function.
    A_std
        Per-frequency spread when the method provides one (DEAC's bin-to-bin
        error, spevae's latent ensemble).  ``None`` otherwise -- MaxEnt returns a
        point estimate.
    G_recon
        Back-transform of ``A`` on the data's own :math:`\tau` grid.
    chi2, chi2_target
        Whitened reconstruction :math:`\chi^2` and the value a *perfect* fit
        would give under the same whitener.  Report the ratio: with
        :math:`N_{\rm bins} \lesssim L_\tau` the target is well below 1 and
        "reached :math:`\chi^2 = 1`" means nothing.
    """

    method: str
    omega: np.ndarray
    A: np.ndarray
    G_recon: np.ndarray | None = None
    A_std: np.ndarray | None = None
    chi2: float = float("nan")
    chi2_target: float = float("nan")
    runtime: float = float("nan")
    meta: dict = field(default_factory=dict)

    @property
    def norm(self) -> float:
        r""":math:`\int A\,d\omega` on the returned grid (misses any tail beyond it)."""
        return float(np.trapezoid(self.A, self.omega))

    @property
    def min_A(self) -> float:
        """Most negative excursion -- 0 for a method that guarantees positivity."""
        return float(np.min(self.A))

    @property
    def chi2_ratio(self) -> float:
        r""":math:`\chi^2 / \chi^2_{\rm target}`: 1 means the noise floor was reached."""
        if not np.isfinite(self.chi2_target) or self.chi2_target <= 0:
            return float("nan")
        return self.chi2 / self.chi2_target

    def attach_reconstruction(
        self,
        mg: MeasuredGreens,
        min_eigenvalue: float = 0.0,
        against: str = "mean",
        G_recon: np.ndarray | None = None,
    ) -> "ContinuationResult":
        r"""Back-transform and score.

        ``G_recon`` lets a method supply its *own exact* reconstruction instead
        of the grid back-transform.  Use it whenever the method's spectrum is
        analytic with algebraic tails -- otherwise the grid truncation described
        in :func:`backtransform` is charged to the method as a fit error, which
        would penalise an exact answer for being untruncated.
        """
        self.G_recon = (backtransform(self.A, self.omega,
                                      mg.tau.double().numpy(), mg.beta)
                        if G_recon is None else np.asarray(G_recon, dtype=float))
        self.meta.setdefault("reconstruction",
                             "grid back-transform" if G_recon is None
                             else "exact forward map")
        self.chi2 = reconstruction_chi2(self.G_recon, mg, min_eigenvalue, against)
        self.chi2_target = mg.covariance(min_eigenvalue)["chi2_target"]
        return self

    def on_grid(self, omega: np.ndarray) -> np.ndarray:
        """Linear interpolation onto a common grid, for pairwise comparison."""
        return np.interp(np.asarray(omega), self.omega, self.A, left=0.0, right=0.0)

    def summary(self) -> str:
        return (f"{self.method:10s} norm={self.norm:7.4f}  min(A)={self.min_A:+.2e}  "
                f"chi2={self.chi2:9.3f}  target={self.chi2_target:7.4f}  "
                f"ratio={self.chi2_ratio:8.2f}  {self.runtime:6.1f}s")


class _Timer:
    """Context manager returning wall time, used by the wrappers."""

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self._t0
        return False
