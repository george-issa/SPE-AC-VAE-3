"""The ported DQMC readers, exercised against the real local datasets.

Each test skips when its dataset is absent, so CI without the Data symlink
stays green -- but on a machine with the data these are the tests that catch
a silently transposed axis, which produces plausible numbers with the wrong
physics and nothing downstream would notice.
"""

import os

import numpy as np
import pytest
import torch

from spevae.data import (
    gissa_cube_axes,
    read_gissa_cube,
    read_long_text_bins,
    read_model_params,
    read_smoqy_dir,
)
from spevae.paths import data_path

REAL = data_path("datasets", "real")
HUBBARD = str(REAL / "hubbard_square_U6.00_mu0.00_L4_b10.00-1")
BOND_HOLSTEIN = None
if REAL.is_dir():
    for cand in sorted(p for p in os.listdir(REAL)
                       if p.startswith("bond_holstein")):
        pos = REAL / cand / "time-displaced" / "greens" / "position"
        if pos.is_dir() and any(f.endswith(".jld2") for f in os.listdir(pos)):
            BOND_HOLSTEIN = str(REAL / cand)
            break
RICHARD = str(REAL / "gL96all_richard.txt")
GISSA_AC = str(REAL / "gIssa" / "for_george_ac.jld2")
GISSA_DOS = str(REAL / "gIssa" / "george_DOS.jld2")


def _sane(mg, beta):
    assert mg.beta == pytest.approx(beta)
    assert mg.includes_endpoints
    assert mg.n_bins >= 2
    sr = mg.check_sum_rule(tol=2e-2)
    assert sr["ok"], f"sum rule violated after sign detection: {sr}"
    # G must decay away from the endpoints -- a transposed read does not
    mid = mg.mean[mg.n_tau // 2]
    assert mid < mg.mean[0] and mid < mg.mean[-1]
    assert (mg.mean > -1e-6).all(), "negative G(tau) after sign handling"


@pytest.mark.skipif(not os.path.isdir(HUBBARD), reason="Hubbard dataset absent")
def test_smoqy_v2_directory():
    params = read_model_params(HUBBARD)
    mg = read_smoqy_dir(HUBBARD)
    _sane(mg, params["beta"])
    assert mg.n_tau == params["L_tau"] + 1
    assert mg.meta["layout"].startswith("v2")


@pytest.mark.skipif(BOND_HOLSTEIN is None or not os.path.isdir(BOND_HOLSTEIN or ""),
                    reason="bond-Holstein dataset absent")
def test_smoqy_v1_directory():
    params = read_model_params(BOND_HOLSTEIN)
    mg = read_smoqy_dir(BOND_HOLSTEIN)
    _sane(mg, params["beta"])
    assert mg.meta["layout"].startswith("v1")


@pytest.mark.skipif(not os.path.exists(RICHARD), reason="long-text bins absent")
def test_long_text_bins_detect_the_negative_convention():
    mg = read_long_text_bins(RICHARD, beta=12.0)
    _sane(mg, 12.0)
    # the source stores G = -<T c c^dag>; detection must have flipped it
    assert mg.meta.get("sign_flip") is True or float(mg.mean[0]) > 0


@pytest.mark.skipif(not os.path.exists(GISSA_AC), reason="gissa cube absent")
def test_gissa_cube_half_filling_cell_with_reference():
    axes = gissa_cube_axes()
    assert 1.0 in axes["n"] and 0.35 in axes["lam"]
    mg = read_gissa_cube(GISSA_AC, beta=10.0, omega=1.0, n=1.0, lam=0.35,
                         dos_path=GISSA_DOS if os.path.exists(GISSA_DOS) else None)
    _sane(mg, 10.0)
    assert mg.n_tau == 101                       # 10*beta + 1, padding dropped
    # half filling: G(0) = 1 - n/2 = 0.5 and PH symmetry G(tau) = G(beta - tau)
    assert float(mg.mean[0]) == pytest.approx(0.5, abs=0.02)
    asym = float((mg.mean - mg.mean.flip(0)).abs().max() / mg.mean.max())
    assert asym < 0.05, f"PH asymmetry {asym} -- axis order suspect"
    if os.path.exists(GISSA_DOS):
        assert mg.reference is not None          # n = 1.0 was continued
        ws, A = mg.reference
        assert np.all(np.asarray(A) >= -1e-12)


@pytest.mark.skipif(not os.path.exists(GISSA_AC), reason="gissa cube absent")
def test_gissa_cube_rejects_off_axis_values():
    with pytest.raises(ValueError, match="not on the cube axis"):
        read_gissa_cube(GISSA_AC, beta=10.0, omega=1.0, n=0.33, lam=0.35)


def test_long_text_parser_validates_the_tiling(tmp_path):
    bad = tmp_path / "bad.txt"
    rows = [(i, 0.5, 0.01) for i in range(5)] + [(i, 0.5, 0.01) for i in range(4)]
    np.savetxt(bad, np.array(rows))
    with pytest.raises(ValueError):
        read_long_text_bins(str(bad), beta=1.0)
