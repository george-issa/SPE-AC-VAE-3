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
from .consolidate import (
    ConsolidationReport,
    FixedPointTrace,
    LadderVerdict,
    consolidate_poles,
    feature_ledger,
    ladder_verdict,
    minimal_pole_fixed_point,
    pole_significance_table,
    refit_poles_to_greens,
    sharpness_verdict,
)
from .positivity import PositivityReport, certify, positivity_certificate
from .prony import MPMReport, aaa_poles, matsubara_from_bins, mpm_extract

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
    "PositivityReport",
    "certify",
    "positivity_certificate",
    "ConsolidationReport",
    "consolidate_poles",
    "LadderVerdict",
    "ladder_verdict",
    "refit_poles_to_greens",
    "minimal_pole_fixed_point",
    "FixedPointTrace",
    "pole_significance_table",
    "sharpness_verdict",
    "feature_ledger",
    "mpm_extract",
    "matsubara_from_bins",
    "aaa_poles",
    "MPMReport",
]
