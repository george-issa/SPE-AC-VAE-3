r"""Readers for the DQMC outputs the package meets in the wild.

Every reader funnels into :func:`spevae.data.real.from_bins`, so whatever the
on-disk layout, the user receives the same :class:`MeasuredGreens` -- sign
convention detected from the sum rule, endpoint policy recorded, covariance
machinery attached.  Ported from the predecessor project's ``data_real.py``,
where each of these layouts was reverse-engineered against real files.

Supported sources
-----------------
* **SmoQyDQMC simulation directories** (:func:`read_smoqy_dir`) -- both the
  v1 layout (``time-displaced/greens/position/bin-*_pID-*.jld2``, one JLD2
  object-reference file per bin) and the v2.x layout
  (``bins/bins_pID-*.h5`` with the
  ``CORRELATIONS/STANDARD/TIME-DISPLACED/greens/POSITION`` dataset).
  ``beta``/``dtau``/``L_tau`` come from ``model_summary.toml``.
* **The coupling-resolved Holstein cube** (:func:`read_gissa_cube`) --
  ``for_george_ac.jld2`` (9.9 GB, un-chunked; cell reads are cached) with the
  SmoQyDEAC reference DOS in the separate ``george_DOS.jld2``.  Cells are
  addressed by *physical values* (``beta=10, omega=1.0, n=1.0, lam=0.35``),
  not raw axis indices.
* **Long-format text bins** (:func:`read_long_text_bins`) -- a stack of
  per-bin ``[tau_index, G, error]`` blocks.

Axis-order warning
------------------
Julia writes column-major; ``h5py`` reads row-major, so every Julia array
appears with its axes *reversed*.  The slice expressions below encode that,
and the shape checks exist because a silently transposed read produces
plausible numbers with the wrong physics.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np
import torch

from .real import MeasuredGreens, from_bins

__all__ = [
    "read_model_params",
    "read_smoqy_dir",
    "read_long_text_bins",
    "read_gissa_cube",
    "gissa_cube_axes",
]


def read_model_params(sim_dir: str) -> dict:
    """``beta``, ``dtau``, ``L_tau`` from a SmoQyDQMC ``model_summary.toml``.

    A line scrape rather than a TOML parser: the file is machine-written, the
    three keys sit at the top level, and this keeps the reader dependency-free.
    """
    path = os.path.join(sim_dir, "model_summary.toml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"model_summary.toml not found in {sim_dir}")
    params: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("beta"):
                params["beta"] = float(line.split("=")[1])
            elif line.startswith("dtau"):
                params["dtau"] = float(line.split("=")[1])
            elif line.startswith("L_tau"):
                params["L_tau"] = int(line.split("=")[1])
    missing = {"beta", "dtau", "L_tau"} - set(params)
    if missing:
        raise ValueError(f"{sorted(missing)} not found in {path}")
    return params


# --------------------------------------------------------------------------- #
# SmoQyDQMC simulation directories (v1 JLD2-per-bin and v2.x HDF5 layouts)
# --------------------------------------------------------------------------- #

def _read_smoqy_v1_bins(sim_dir: str, r1: int, r2: int) -> np.ndarray:
    """v1 layout: one JLD2 file per bin, correlations behind an object ref."""
    import h5py

    bin_dir = os.path.join(sim_dir, "time-displaced", "greens", "position")
    files = [f for f in os.listdir(bin_dir)
             if re.match(r"bin-\d+_pID-\d+\.jld2", f)]
    if not files:
        raise FileNotFoundError(f"no bin-*_pID-*.jld2 files in {bin_dir}")
    files.sort(key=lambda s: (int(re.search(r"pID-(\d+)", s).group(1)),
                              int(re.search(r"bin-(\d+)_", s).group(1))))

    rows = []
    for name in files:
        with h5py.File(os.path.join(bin_dir, name), "r") as f:
            ref = f["correlations"][()].flat[0]
            corr = f[ref][()] if isinstance(ref, h5py.h5r.Reference) else ref
            # structured (L_tau+1, N_R2, N_R1) with ('re','im') fields
            rows.append(corr["re"][:, r2, r1].copy())
    return np.stack(rows, axis=0)                       # (N_bins, L_tau+1)


def _read_smoqy_v2_bins(sim_dir: str, r1: int, r2: int) -> np.ndarray:
    """v2.x layout: bins/bins_pID-*.h5, axes reversed from the Julia shape
    (N_bins, Lx, Ly, L_tau+1, N_pairs) -> (N_pairs, L_tau+1, Ly, Lx, N_bins)."""
    import h5py

    bin_dir = os.path.join(sim_dir, "bins")
    files = sorted(
        (f for f in os.listdir(bin_dir) if re.match(r"bins_pID-\d+\.h5", f)),
        key=lambda s: int(re.search(r"pID-(\d+)", s).group(1)))
    if not files:
        raise FileNotFoundError(f"no bins_pID-*.h5 files in {bin_dir}")

    blocks = []
    for name in files:
        with h5py.File(os.path.join(bin_dir, name), "r") as f:
            data = f["CORRELATIONS/STANDARD/TIME-DISPLACED/greens/POSITION"]
            if data.ndim != 5:
                raise ValueError(f"{name}: expected a 5-axis POSITION dataset, "
                                 f"got shape {data.shape}")
            blocks.append(data[0, :, r2, r1, :].real.T)  # (N_bins, L_tau+1)
    return np.concatenate(blocks, axis=0)


def read_smoqy_dir(
    sim_dir: str,
    r1: int = 0,
    r2: int = 0,
    dtype: torch.dtype = torch.float64,
) -> MeasuredGreens:
    r"""Read the local (or ``(r1, r2)``-displaced) :math:`G(\tau)` bins from a
    SmoQyDQMC simulation directory, either storage layout.

    The endpoint-inclusive grid :math:`\tau = 0 \dots \beta` is kept: the
    :math:`\tau=\beta` slice is what makes the spectral norm readable, and
    :class:`MeasuredGreens` already excludes it from the covariance block.
    The sign convention is detected from the sum rule, as everywhere else.
    """
    params = read_model_params(sim_dir)
    if os.path.isdir(os.path.join(sim_dir, "bins")):
        bins = _read_smoqy_v2_bins(sim_dir, r1, r2)
        layout = "v2.x (bins_pID-*.h5)"
    elif os.path.isdir(os.path.join(sim_dir, "time-displaced")):
        bins = _read_smoqy_v1_bins(sim_dir, r1, r2)
        layout = "v1 (bin-*_pID-*.jld2)"
    else:
        raise FileNotFoundError(
            f"{sim_dir} has neither bins/ (v2.x) nor time-displaced/ (v1)")

    n_expected = params["L_tau"] + 1
    if bins.shape[1] != n_expected:
        raise ValueError(
            f"{sim_dir}: read {bins.shape[1]} tau slices but model_summary.toml "
            f"says L_tau={params['L_tau']} (expected {n_expected} inclusive "
            f"points) -- the axis order assumption is violated, refusing to "
            f"hand back silently transposed data")

    mg = from_bins(bins, params["beta"], includes_endpoints=True,
                   label=os.path.basename(os.path.normpath(sim_dir)),
                   dtype=dtype)
    mg.meta.update(**params, r1=r1, r2=r2, source=sim_dir, layout=layout)
    return mg


# --------------------------------------------------------------------------- #
# Long-format text bins
# --------------------------------------------------------------------------- #

def read_long_text_bins(
    path: str,
    beta: float,
    sign_flip: bool | None = None,
    dtype: torch.dtype = torch.float64,
) -> MeasuredGreens:
    r"""Parse a stack of per-bin ``[tau_index, G, (error)]`` blocks.

    Each block runs ``tau_index = 0 .. L`` (endpoint-inclusive); the block
    length and the bin count are inferred, and the index column is *validated*
    to be a clean tiling rather than trusted.  The sign convention is detected
    from the sum rule unless forced.
    """
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"expected >=2 columns [tau_index, G, (error)], got "
                         f"shape {raw.shape} from {path}")
    tau_idx = raw[:, 0].astype(int)
    g = raw[:, 1].astype(np.float64)
    n_full = int(tau_idx.max()) + 1
    if g.size % n_full != 0:
        raise ValueError(f"{path}: {g.size} rows is not a whole multiple of "
                         f"the block length {n_full}")
    n_bins = g.size // n_full
    if not np.array_equal(tau_idx, np.tile(np.arange(n_full), n_bins)):
        raise ValueError(f"{path}: tau_index is not a clean 0..{n_full - 1} "
                         f"tiling across {n_bins} blocks")

    mg = from_bins(g.reshape(n_bins, n_full), beta, sign_flip=sign_flip,
                   includes_endpoints=True, label=os.path.basename(path),
                   dtype=dtype)
    mg.meta.update(source=path, format="long-text")
    return mg


# --------------------------------------------------------------------------- #
# Coupling-resolved Holstein cube (for_george_ac.jld2 + george_DOS.jld2)
# --------------------------------------------------------------------------- #

_GISSA_DTAU = 0.1
_GISSA_LAM = (0.125, 0.25, 0.3, 0.325, 0.35, 0.375)
_GISSA_N = tuple(round(0.05 * i, 10) for i in range(1, 21))
_GISSA_OMEGA = (0.5, 1.0, 1.5, 2.0)
_GISSA_BETA = tuple(float(b) for b in range(5, 21))


def gissa_cube_axes() -> dict:
    """The physical values each cube axis takes, for discovery."""
    return {"lam": _GISSA_LAM, "n": _GISSA_N, "omega": _GISSA_OMEGA,
            "beta": _GISSA_BETA, "dtau": _GISSA_DTAU}


def _axis_index(name: str, values: tuple, requested: float) -> int:
    diffs = [abs(v - requested) for v in values]
    i = int(np.argmin(diffs))
    if diffs[i] > 1e-6:
        raise ValueError(f"{name}={requested} is not on the cube axis "
                         f"{values}")
    return i


@lru_cache(maxsize=32)
def _gissa_cell(ac_path: str, beta_idx: int, omega_idx: int, n_idx: int,
                lam_idx: int, n_keep: int) -> np.ndarray:
    """One (N_bins, n_keep) block.  Cached: the file is un-chunked, so every
    strided cell read costs seconds and callers tend to want a cell twice."""
    import h5py

    with h5py.File(ac_path, "r") as f:
        # Julia [lam, n, Om, beta, bins, tau] -> h5py (tau, bins, beta, Om, n, lam)
        raw = f["G_r"][:n_keep, :, beta_idx, omega_idx, n_idx, lam_idx]
    return np.ascontiguousarray(raw.T)


def read_gissa_cube(
    ac_path: str,
    *,
    beta: float,
    omega: float,
    n: float,
    lam: float,
    dos_path: str | None = None,
    dtype: torch.dtype = torch.float64,
) -> MeasuredGreens:
    r"""One cell of the coupling-resolved Holstein cube, by physical values.

    :math:`\tau` is zero-padded on disk to the :math:`\beta=20` length; only
    the valid :math:`10\beta + 1` endpoint-inclusive slices are returned.  The
    data is already in the positive convention (:math:`G(0) = 1 - n/2`), and
    the sum-rule detection in :func:`from_bins` confirms rather than assumes.

    ``dos_path`` attaches the shipped SmoQyDEAC continuation as a *reference*
    when that cell was actually continued (in the shipped file, only
    :math:`n = 1.0`); an un-continued cell simply gets no reference.
    """
    bi = _axis_index("beta", _GISSA_BETA, beta)
    oi = _axis_index("omega", _GISSA_OMEGA, omega)
    ni = _axis_index("n", _GISSA_N, n)
    li = _axis_index("lam", _GISSA_LAM, lam)
    n_keep = int(round(beta / _GISSA_DTAU)) + 1

    bins = _gissa_cell(os.path.abspath(ac_path), bi, oi, ni, li, n_keep)

    ref = None
    if dos_path is not None:
        import h5py
        with h5py.File(dos_path, "r") as f:
            A = f["A"][:, bi, oi, ni, li]
            ws = f["ws"][()]
        if np.any(A != 0.0):
            ref = (np.asarray(ws), np.asarray(A))

    mg = from_bins(bins, float(beta), includes_endpoints=True,
                   label=f"gissa lam={lam:g} n={n:g} Omega={omega:g} "
                         f"beta={beta:g}",
                   reference=ref, dtype=dtype)
    mg.meta.update(lam=lam, n=n, omega=omega, beta=float(beta),
                   dtau=_GISSA_DTAU, source=ac_path,
                   reference_source=dos_path if ref is not None else None)
    return mg
