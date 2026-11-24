"""Extract the Holstein judge cell from the gIssa cube into a small fixture.

The cube is 9.9 GB and lives outside the repository; every Holstein judge
read it by absolute path.  This writes the one cell the judges use
(beta 8, Omega 1, n 1, lambda 0.325) -- its raw bins plus the shipped
SmoQyDEAC reference continuation -- to
src/spevae/evaluation/fixtures/, and checks that the fixture reproduces the
cube-read MeasuredGreens bit for bit (bins, covariance, whitener, target).

Usage (from the repo root):
    python scripts/eval/build_holstein_fixture.py [CUBE_DIR]
"""
import hashlib
import os
import sys

sys.path.insert(0, "src")

import numpy as np
import torch

from spevae.data.readers import read_gissa_cube
from spevae.evaluation import cases

CUBE_DIR = (sys.argv[1] if len(sys.argv) > 1 else
            "/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/gIssa")
CELL = dict(beta=8.0, omega=1.0, n=1.0, lam=0.325)

mg = read_gissa_cube(os.path.join(CUBE_DIR, "for_george_ac.jld2"),
                     dos_path=os.path.join(CUBE_DIR, "george_DOS.jld2"), **CELL)
bins = mg.bins.double().numpy()
out = os.path.join("src/spevae/evaluation/fixtures", cases.HOLSTEIN_FIXTURE["file"])
payload = {"bins": bins}
if mg.reference is not None:
    payload["ref_omega"] = np.asarray(mg.reference[0], float)
    payload["ref_A"] = np.asarray(mg.reference[1], float)
for k, v in dict(beta=8.0, omega=1.0, n=1.0, lam=0.325, dtau=mg.meta["dtau"],
                 source="gIssa cube for_george_ac.jld2 (reference: george_DOS.jld2)").items():
    payload[f"meta_{k}"] = np.array(v)
np.savez_compressed(out, **payload)
digest = hashlib.sha256(np.ascontiguousarray(bins, dtype=np.float64).tobytes()).hexdigest()

fx = cases.holstein_cell(out)
assert torch.equal(fx.bins, mg.bins), "bins differ"
assert torch.equal(fx.tau, mg.tau), "tau differs"
c0 = mg.covariance(1e-10, method="structured")
c1 = fx.covariance(1e-10, method="structured")
for k in ("sigma", "inv_sqrt_R", "R"):
    if k in c0:
        assert torch.equal(c0[k], c1[k]), f"covariance[{k}] differs"
assert c0["chi2_target"] == c1["chi2_target"]
assert torch.equal(mg.statistical_slices, fx.statistical_slices)
print(f"bins {bins.shape}, reference {'yes' if mg.reference is not None else 'no'}, "
      f"chi2_target {c0['chi2_target']:.5f}, file {os.path.getsize(out)/1e3:.0f} kB")
print(f"bins_sha256 = {digest}")
