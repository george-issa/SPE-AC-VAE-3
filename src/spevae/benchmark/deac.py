r"""Differential-evolution analytic continuation, via `SmoQyDEAC.jl`.

DEAC is structurally different from both maximum entropy and this package: it
does not regularise with an entropy term and it does not learn a prior. It runs
a differential-evolution search over :math:`A(\omega)` against the
covariance-whitened :math:`\chi^2`, many times, and averages the results.  That
makes it the most valuable of the three comparators -- agreement between
methods that fail in *different* ways is real evidence, whereas two methods with
the same bias agreeing tells you nothing.

Reference: J. C. Neuhaus, N. Nichols, D. Banerjee, B. Cohen-Stead, T. A. Maier,
A. Del Maestro, S. Johnston, *SmoQyDEAC.jl: A differential evolution package for
the analytic continuation of imaginary time correlation functions*,
SciPost Phys. Codebases **39** (2024).

How it is called
----------------
Through a Julia subprocess with a plain-text interface (see ``deac_driver.jl``).
No `PyCall`/`juliacall` bridge: those tie the Python process to a Julia runtime
for the whole session, and the payload here is two arrays.  The cost is process
startup plus Julia's compile time (~10 s cold), which is negligible against a
DEAC run.

DEAC bootstraps internally when there are fewer bins than time slices -- which
is exactly the regime that makes the covariance estimate unreliable in the first
place (see :meth:`~spevae.data.real.MeasuredGreens.covariance`).  Bootstrapping
does not create information, so a run in that regime should be read with the
same caution as a :math:`\chi^2` in that regime.  The wrapper reports whether it
happened.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

from ..data.real import MeasuredGreens
from .result import ContinuationResult, _Timer

__all__ = ["deac_available", "run_deac", "DEFAULT_JULIA_PROJECT"]

#: Julia environment holding SmoQyDEAC.  ``@deac`` is a *named* shared
#: environment (``~/.julia/environments/deac``); override with ``$SPEVAE_JULIA_PROJECT``.
DEFAULT_JULIA_PROJECT = os.environ.get("SPEVAE_JULIA_PROJECT", "@deac")

_DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deac_driver.jl")


def deac_available(julia_project: str = DEFAULT_JULIA_PROJECT,
                   timeout: float = 180.0) -> bool:
    """Whether a Julia with SmoQyDEAC can be reached (loads the package to check)."""
    if shutil.which("julia") is None:
        return False
    try:
        p = subprocess.run(
            ["julia", f"--project={julia_project}", "-e", "using SmoQyDEAC"],
            capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode == 0
    except Exception:
        return False


def run_deac(
    mg: MeasuredGreens,
    omega: np.ndarray | None = None,
    num_bins: int = 10,
    runs_per_bin: int = 10,
    generations: int = 20_000,
    population: int = 20,
    seed: int = 8675309,
    fitness: float = 1.0,
    find_fitness_floor: bool = True,
    julia_project: str = DEFAULT_JULIA_PROJECT,
    n_omega: int = 401,
    omega_max: float = 15.0,
    min_eigenvalue: float = 0.0,
    timeout: float = 7200.0,
    workdir: str | None = None,
    verbose: bool = False,
) -> ContinuationResult:
    r"""Continue ``mg`` with SmoQyDEAC.

    Parameters
    ----------
    omega
        Output grid.  DEAC needs it **evenly spaced**; a non-uniform grid is
        rejected rather than silently resampled.  Defaults to
        ``linspace(-omega_max, omega_max, n_omega)`` -- coarser than the MaxEnt
        default because DEAC's cost scales with the number of frequencies.
    num_bins, runs_per_bin, generations
        Statistics and search effort.  The defaults are a fast setting suitable
        for a comparison table; raise them for a production number.
    find_fitness_floor
        Let DEAC discover the best attainable :math:`\chi^2` and target it,
        rather than assuming 1 is reachable.  On by default for the same reason
        this package reports ``chi2_target``: with a poorly conditioned
        covariance, 1 is the wrong goal.
    workdir
        Keep intermediates here instead of a temporary directory.

    Returns
    -------
    ContinuationResult
        ``A_std`` carries DEAC's own bin-to-bin spread.

    Raises
    ------
    RuntimeError
        If Julia is missing, SmoQyDEAC will not load, or the driver fails -- with
        the subprocess's stderr attached, since a silent fallback here would put
        a missing comparator into a table as an absence rather than an error.
    """
    if shutil.which("julia") is None:
        raise RuntimeError(
            "julia not found on PATH; SmoQyDEAC comparison unavailable. "
            "Install Julia and `Pkg.add(\"SmoQyDEAC\")` into the environment "
            f"named by SPEVAE_JULIA_PROJECT (currently {julia_project!r})."
        )

    if omega is None:
        omega = np.linspace(-omega_max, omega_max, n_omega)
    omega = np.asarray(omega, dtype=float)
    d = np.diff(omega)
    if omega.size < 2 or not np.allclose(d, d[0], rtol=1e-9, atol=1e-12):
        raise ValueError("SmoQyDEAC requires an evenly spaced omega grid")

    if not mg.includes_endpoints:
        raise ValueError(
            "SmoQyDEAC requires tau = 0 ... beta inclusive; this dataset's grid "
            "stops short of beta.")

    owned = workdir is None
    wd = workdir or tempfile.mkdtemp(prefix="spevae_deac_")
    os.makedirs(wd, exist_ok=True)
    try:
        np.savetxt(os.path.join(wd, "bins.txt"),
                   mg.bins.double().numpy(), fmt="%.17g")
        with open(os.path.join(wd, "params.txt"), "w") as f:
            for k, v in {
                "beta": repr(float(mg.beta)),
                "omega_min": repr(float(omega[0])),
                "omega_max": repr(float(omega[-1])),
                "n_omega": str(int(omega.size)),
                "num_bins": str(int(num_bins)),
                "runs_per_bin": str(int(runs_per_bin)),
                "generations": str(int(generations)),
                "population": str(int(population)),
                "seed": str(int(seed)),
                "fitness": repr(float(fitness)),
                "find_fitness_floor": "true" if find_fitness_floor else "false",
            }.items():
                f.write(f"{k} = {v}\n")

        cmd = ["julia", f"--project={julia_project}", _DRIVER, wd]
        with _Timer() as t:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"SmoQyDEAC driver failed (exit {proc.returncode}).\n"
                f"cmd: {' '.join(cmd)}\n--- stderr ---\n{proc.stderr[-4000:]}")
        if verbose:
            print(proc.stdout)

        spec = np.loadtxt(os.path.join(wd, "spectrum.txt"))
        info = {}
        info_path = os.path.join(wd, "info.txt")
        if os.path.exists(info_path):
            for line in open(info_path):
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k.strip()] = v.strip()
    finally:
        if owned:
            shutil.rmtree(wd, ignore_errors=True)

    res = ContinuationResult(
        method="deac", omega=spec[:, 0], A=spec[:, 1], A_std=spec[:, 2],
        runtime=t.elapsed,
        meta={"solver": f"SmoQyDEAC {info.get('smoqydeac_version', '?')}",
              "num_bins": num_bins, "runs_per_bin": runs_per_bin,
              "generations": generations, "population": population,
              "find_fitness_floor": find_fitness_floor,
              "zeroth_moment": info.get("zeroth_moment"),
              "avg_generations": info.get("avg_generations"),
              # DEAC bootstraps when bins < 5 * tau slices; flag it, because the
              # extra samples are resampled, not new information.
              "bootstrapped": mg.n_bins < 5 * mg.n_tau,
              "julia_runtime": info.get("runtime")},
    )
    return res.attach_reconstruction(mg, min_eigenvalue, against="mean")
