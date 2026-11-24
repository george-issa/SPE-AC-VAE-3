"""Pack the Hubbard-dimer DQMC bins into a small versioned fixture.

Reads a `hubbard_dimer.jl` output directory, averages the two equivalent
sites bin by bin, and writes the binned local G(tau) with the run's
parameters to src/spevae/evaluation/fixtures/.  The exact spectrum is not
stored: `evaluation.cases.hubbard_dimer_cell` recomputes it by exact
diagonalisation from the parameters.

    python scripts/dqmc/build_dimer_fixture.py out/dqmc/hubbard_dimer_U4.00_b8.00_dt0.050-1
"""
import hashlib
import os
import re
import sys

sys.path.insert(0, "src")

import numpy as np

from spevae.data.readers import read_smoqy_dir
from spevae.evaluation import cases

sim = sys.argv[1]
mg = read_smoqy_dir(sim, pair=(0, 1))
info = open(os.path.join(sim, "simulation_info_sID-1_pID-0.toml")).read()
get = lambda k: re.search(rf"^{k}\s*=\s*(.+)$", info, re.M).group(1).strip().strip('"')
bins = mg.bins.double().numpy()
payload = {"bins": bins}
for k, v in dict(U=float(get("U")), t=1.0, beta=float(get("beta")), dtau=float(get("dtau")),
                 bin_size=int(get("bin_size")), seed=int(get("seed")),
                 source=f"SmoQyDQMC {get('smoqydqmc_version')}, scripts/dqmc/hubbard_dimer.jl, "
                        "half filling, symmetric Trotter, two sites averaged").items():
    payload[f"meta_{k}"] = np.array(v)
out = os.path.join("src/spevae/evaluation/fixtures", cases.DIMER_FIXTURE["file"])
np.savez_compressed(out, **payload)
digest = hashlib.sha256(np.ascontiguousarray(bins, dtype=np.float64).tobytes()).hexdigest()
print(f"bins {bins.shape}, file {os.path.getsize(out) / 1e3:.0f} kB")
print(f"bins_sha256 = {digest}")
