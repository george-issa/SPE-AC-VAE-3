"""THE step-2 gate, applied to the map arm: the Holstein manifold floor.

Reproduces the step-1 boundary proof (checkpoint_A.py) for each model
given on argv as tag=path: z-refine MAP (3k steps), then the anchor
released (prior_weight = 0, 10k steps) -- the resulting chi2 is the
decoder MANIFOLD's expressiveness floor on this cell, free of
regularisation or optimiser excuses.  Banked step-1 numbers: s1_metric
floor 23.96 (137x the calibrated 0.175 +- 0.061 target); free poles
off-manifold 0.56.  Appends results to out/step2/holstein_floor.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from spevae import load_model
from spevae.data.readers import read_gissa_cube
from spevae.refine import RefineConfig, refine_latent

CUBE = ("/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/"
        "gIssa/for_george_ac.jld2")
OUT = Path("out/step2/holstein_floor.json")

mg = read_gissa_cube(CUBE, beta=8.0, omega=1.0, n=1.0, lam=0.325)
print(mg.summary())

report = json.load(open(OUT)) if OUT.exists() else {}
for kv in sys.argv[1:]:
    tag, path = kv.split("=", 1)
    model = load_model(path)
    rr = refine_latent(model, mg, RefineConfig(cov_method="structured",
                                               min_eigenvalue=1e-10,
                                               steps=3000))
    print(f"{tag}: z-refine (MAP) chi2 {rr.chi2_init:.1f} -> {rr.chi2:.2f} "
          f"(target {rr.chi2_target:.3f})", flush=True)
    rb = refine_latent(model, mg, RefineConfig(cov_method="structured",
                                               min_eigenvalue=1e-10,
                                               steps=10000, prior_weight=0.0,
                                               patience=3000))
    print(f"{tag}: manifold floor chi2 {rb.chi2:.2f} "
          f"(ratio {rb.chi2 / rb.chi2_target:.1f}x target)", flush=True)
    report[tag] = {"chi2_map": float(rr.chi2), "chi2_init": float(rr.chi2_init),
                   "floor": float(rb.chi2), "target": float(rb.chi2_target),
                   "ratio": float(rb.chi2 / rb.chi2_target)}
with open(OUT, "w") as fh:
    json.dump(report, fh, indent=1)
print(f"-> {OUT}")
