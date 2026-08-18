"""Benchmarking against the established analytic-continuation solvers.

Three methods, one interface (:class:`~spevae.benchmark.result.ContinuationResult`),
one forward map for scoring, one set of metrics:

- :func:`~spevae.benchmark.maxent.run_maxent` -- maximum entropy, via the
  vendored `ana_cont` (Kaufmann & Held).
- :func:`~spevae.benchmark.deac.run_deac` -- differential evolution, via
  `SmoQyDEAC.jl` in a Julia subprocess.
- :func:`~spevae.benchmark.compare.run_spevae` -- this package.

Availability is checked, never assumed: :func:`maxent_available` and
:func:`deac_available` let a driver skip a comparator it cannot run instead of
crashing halfway through a sweep -- but ``run_deac`` itself raises rather than
returning nothing, so a missing solver never becomes a silent absence in a table.
"""

from .compare import agreement_matrix, compare, format_table, run_spevae
from .plot import plot_comparison
from .deac import DEFAULT_JULIA_PROJECT, deac_available, run_deac
from .maxent import maxent_available, run_maxent
from .result import ContinuationResult, backtransform, reconstruction_chi2

__all__ = [
    "ContinuationResult",
    "backtransform",
    "reconstruction_chi2",
    "run_maxent",
    "maxent_available",
    "run_deac",
    "deac_available",
    "DEFAULT_JULIA_PROJECT",
    "run_spevae",
    "compare",
    "agreement_matrix",
    "format_table",
    "plot_comparison",
]
