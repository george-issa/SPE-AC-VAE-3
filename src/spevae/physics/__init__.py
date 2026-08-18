"""Exact physics of the analytic-continuation problem: kernel, pole algebra, forward map."""

from .kernel import (
    MATSUBARA_NU0,
    fermi_kernel,
    from_dimensionless,
    matsubara_nu,
    to_dimensionless,
)
from .forward import (
    PoleGreens,
    SpectralQuadrature,
    gaussian_mixture_spectrum,
    greens_matsubara,
    matsubara_from_tau,
)
from .poles import (
    PoleSet,
    effective_pole_count,
    per_pole_extrema,
    spectral_from_poles,
    spectral_norm,
    tail_coefficients,
    windowed_moments,
)

__all__ = [
    "MATSUBARA_NU0",
    "fermi_kernel",
    "matsubara_nu",
    "to_dimensionless",
    "from_dimensionless",
    "PoleSet",
    "spectral_from_poles",
    "spectral_norm",
    "tail_coefficients",
    "windowed_moments",
    "effective_pole_count",
    "per_pole_extrema",
    "SpectralQuadrature",
    "PoleGreens",
    "greens_matsubara",
    "matsubara_from_tau",
    "gaussian_mixture_spectrum",
]
