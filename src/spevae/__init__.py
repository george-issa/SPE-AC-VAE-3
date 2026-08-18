"""spevae -- analytic continuation of imaginary-time Green's functions.

A physics-constrained variational autoencoder that maps a noisy DQMC
:math:`G(\\tau)` to a real-frequency spectral function :math:`A(\\omega)`
represented as a simple pole expansion -- causal, normalised, positive, and
free of any frequency grid by construction.

See :mod:`spevae.physics` for the exact forward map and the constraint algebra,
and ``docs/DESIGN.md`` for the design rationale and literature context.
"""

__version__ = "0.1.0"

from . import physics  # noqa: F401
from .diagnostics import AnsatzFloor, ansatz_chi2_floor, tail_weight_bound  # noqa: E402
from .io import load_checkpoint, load_model, save_model  # noqa: E402
from .refine import RefineConfig, RefineResult, refine_latent  # noqa: E402

__all__ = [
    "physics",
    "ansatz_chi2_floor",
    "AnsatzFloor",
    "tail_weight_bound",
    "load_model",
    "save_model",
    "load_checkpoint",
    "refine_latent",
    "RefineConfig",
    "RefineResult",
    "__version__",
]
