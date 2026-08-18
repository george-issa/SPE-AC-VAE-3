# Tutorial: continuing a real DQMC dataset, end to end

This walks the full pipeline on a real half-filled Hubbard simulation
(SmoQyDQMC, U = 6, β = 10, 4×4 lattice, 800 bins) exactly as run during
validation of this package. Every number below is a measured output, not an
illustration. Substitute your own data at any step.

## 0. Get bins

Any `(N_bins × L_τ)` whitespace-delimited text file with τ = 0 … β inclusive
works directly. If your data lives in a SmoQyDQMC simulation directory or one
of the Holstein cubes, the readers export it:

```python
from spevae.data import read_smoqy_dir
import numpy as np

mg = read_smoqy_dir("path/to/hubbard_square_U6.00_mu0.00_L4_b10.00-1")
np.savetxt("hubbard_bins.txt", mg.bins.numpy())
print(mg.summary())
```

The summary is worth reading before anything else — it is the data-quality
report the rest of the pipeline builds on:

```
beta      : 10   dtau = 0.05
grid      : 201 slices, endpoint-inclusive
bins      : 800
sum rule  : G(0)+G(beta) = 1.000000 (ok)
LW rho    : 0.0533  (floor 0)  ->  chi2 at a perfect fit = 0.3900
bins/slices: 800/200
```

Two things to note. The **sign convention was detected**, not assumed (the sum
rule is satisfied by exactly one sign). And the **perfect-fit χ² is 0.39, not
1** — the calibrated target for this covariance estimator. Every χ² this
package prints comes with that target next to it.

## 1. Pretrain once (or reuse a checkpoint)

```bash
spevae pretrain --preset fast --out out/model.pt --device mps
```

A 27-minute laptop run. On the frozen held-out stream, every metric improved
at every one of twelve evaluations — W₁: 1.05 → 0.35, relative L²: 0.38 →
0.19 — with active units steady at 11.75/12 (no posterior collapse). Longer
runs and the `full` preset keep buying accuracy; the generator is infinite, so
there is nothing to overfit.

## 2. Continue

```bash
spevae continue --bins hubbard_bins.txt --beta 10.0 --checkpoint out/model.pt \
                --out hubbard.npz --figure hubbard.png
```

Output (amortised: one forward pass, ~0.3 s):

```
chi2 = 193.7  (calibrated perfect-fit target 0.39, ratio 497)
exactly-pinned slices (PH symmetry): max deviation 7.94e-04
effective poles: 12.70 of 16
positivity: certified A >= 0 (atol=1.0e-10)
```

The spectrum is Mott-gapped with Hubbard bands centred near ±U/2, the
asymmetry degrees of freedom (Im r ≠ 0, Σ Im r = 0) are active, and the
**positivity certificate** — an exact statement about the pole set, no ω grid
anywhere — passes. The ratio 497 is the honest signature of an amortised model
that has not seen this cell; the figure's uncertainty band says *where* the
answer is undetermined (here: the fine structure inside the bands, not the gap
edges).

Note the pinned-slice line: at half filling the algorithm fixes
G(0) = G(β) = 1/2 to machine precision. Those slices are exact constraints,
excluded from the whitened χ² and reported separately.

## 3. Fine-tune to the cell (Stage 2)

```bash
spevae finetune --bins hubbard_bins.txt --beta 10.0 --checkpoint out/model.pt \
                --out out/model-hubbard.pt --steps 2000 --device mps
```

```
best step 599: chi2_ratio 348.6, 0 skipped batches, stopped early
final: chi2=135.8 target=0.3895 eff_poles=12.76
```

χ² 194 → 136 with **effective poles pinned at ~12.75 throughout** — the
default `--train encoder_only` preserves the sparse pole budget pretraining
installed (the predecessor project measured what happens without the freeze:
pole usage inflated 2.30 → 11.08 as surplus poles woke up to fit noise).

## 4. Refine per input (Stage 3)

```bash
spevae continue --bins hubbard_bins.txt --beta 10.0 \
                --checkpoint out/model-hubbard.pt --refine --refine-steps 3000 \
                --out hubbard_refined.npz
```

```
latent refinement: 3000 steps, chi2 136 -> 13.06 (calibrated target 0.3895)
   [prior binding -- treat with care]
positivity: A < 0: min_A=-3.345e-03 at omega=0.6934, negative weight 1.760e-03
```

Refinement walks χ² down another order of magnitude — and **both safety
diagnostics fire**, which is the point of having them:

- *prior binding*: the refined code sits far enough from the encoder posterior
  that the prior term is now limiting. The fit is no longer a sample from the
  learned distribution; treat it as a MAP estimate with the training prior as
  regulariser.
- *the certificate reports a real (small) positivity violation*: −3.3×10⁻³ at
  ω ≈ 0.69, total negative weight 0.18% of the spectrum. You see it because
  it is measured exactly, not because a grid happened to sample it.

Whether to prefer the certified-positive χ² = 136 spectrum or the
χ² = 13 spectrum with a 0.18% dip is a physics judgement — the package's job
is to hand you both with the diagnostics attached.

## 5. Cross-check against independent methods

```bash
python scripts/benchmark.py --bins hubbard_bins.txt --beta 10.0 \
       --checkpoint out/model-hubbard.pt --refine --figure cmp.png
```

runs MaxEnt (vendored ana_cont) and SmoQyDEAC (Julia) on the *same* bins with
the *same* whitened covariance and scores everything through one forward map.
On the site-Holstein cell that ships a SmoQyDEAC reference, that reference
lands at χ²/target = 0.98 — independent confirmation of the calibration — and
this package's refined result at 1.39, with MaxEnt genuinely fitting below the
floor at 0.23 (~800 grid degrees of freedom against 100 slices).

The three methods fail differently. Where they agree, the feature is in the
data. Where they disagree, the inversion is underdetermined there — and that
localisation is information, not embarrassment.

## 6. Use the poles

```python
import numpy as np
z = np.load("hubbard_refined.npz")
z["poles_omega"], z["poles_gamma"]      # z_p = omega_p - i*gamma_p (physical)
z["residues_re"], z["residues_im"]      # a_p + i b_p;  sum a = norm, sum b = 0
```

The pole set is exact input for downstream work — Padé-free evaluation of
G(z) anywhere in the complex plane, real-time propagation, hybridisation
fitting — with no ω grid to interpolate off. That reuse is a feature of the
representation, argued for in the minimal-pole-representation literature
(Zhang *et al.*, JCP **162**, 214111 (2025)).
