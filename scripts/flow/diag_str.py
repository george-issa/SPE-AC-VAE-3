"""Where is refined-flow overconfidence?  Coverage by noise x grid x noise redraw.

    python scripts/flow/diag_str.py out/flow/F_60k.pt out/flow/str_F60k/diag.json
"""
import json
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/flow")

import torch

import judge_str as JS
from spevae.data.pole_prior import PolePriorGenerator
from spevae.models.flow import load_flow, refine_samples
from spevae.physics.poles import PoleSet

torch.set_num_threads(6)
flow = load_flow(sys.argv[1]).eval()
S, M = 16, 16
rows = []
for dtau in (0.25, 0.05):
    for nz in (3e-2, 1e-3, 3e-5):
        gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive",
                                 beta_range=(8.0, 8.0), dtau_choices=(dtau,),
                                 noise_scale_range=(nz, nz))
        with gen.fixed_scalars(6060):
            b = gen.batch(M, torch.Generator().manual_seed(6061))
        truth = PoleSet(b.poles_true["eps"], b.poles_true["gamma"], b.poles_true["a"],
                        b.poles_true["b"])
        for name, pert in (("raw", None), ("refine", False), ("refine+redraw", True)):
            g = torch.Generator().manual_seed(21)
            t = time.time()
            z, norm = flow.sample_coords(b.G, b.x, 8.0, sigma=b.sigma, n_samples=S, generator=g)
            if pert is None:
                ps = flow.coords.decode(z, norm=norm, nu_max=flow.nu_max(8.0))
            else:
                ps, _ = refine_samples(flow, z, norm, 8.0, b.x, b.G.repeat_interleave(S, 0),
                                       b.sigma.repeat_interleave(S, 0), b.inv_sqrt_R, 1.0,
                                       perturb=pert, generator=g, **JS.REFINE)
            m, _ = JS.metrics(ps, b, S, truth)
            m.update(dtau=dtau, noise=nz, config=name, n_tau=int(b.x.numel()))
            rows.append(m)
            print(f"dtau {dtau:.2f} (L={b.x.numel():3d}) noise {nz:7.0e} {name:14s} chi2 "
                  f"{m['chi2_median']:8.3g}  cov50 {m['cover50']:.2f} cov90 {m['cover90']:.2f} "
                  f"band90 {m['band90']:.2f}  W1 {m['w1_mean']:.4f}  ({time.time() - t:.0f}s)",
                  flush=True)
json.dump(rows, open(sys.argv[2], "w"), indent=1)
