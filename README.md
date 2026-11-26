# spevae

Analytic continuation of noisy imaginary-time Green's functions to
real-frequency spectral functions, using a physics-constrained variational
autoencoder: a **transformer encoder** reads G(τ) as a sequence of time-slice
tokens, and the decoder emits a **simple pole expansion** — never a frequency
grid.

```
G(τ) from DQMC ──▶ transformer ──▶ q(z|G) ──▶ decoder ──▶ {ε_p − iγ_p,  a_p + i b_p}
  any β, any Δτ      encoder       (sample)                       │
                                                                  ▼
                                   A(ω) = −(1/π) Σ_p Im[ r_p / (ω − z_p) ]
                                   causal · normalised · asymmetric · certified ≥ 0
```

**The encoder is a transformer.** Every imaginary-time slice is one token — its
value of G (standardised and signed-log), its error bar, and continuous position
features of x = τ/β — and a `CLS` token adds β, L_τ, the noise level and the
leading Matsubara coefficients. Self-attention over the slices, with a padding
mask for ragged lengths, is what lets one set of weights read any grid; q(z|G)
is read off the `CLS` output (`src/spevae/models/encoder.py`,
[`docs/DESIGN.md`](docs/DESIGN.md) §6).

The pole set **is** the result: evaluate `A(ω)` anywhere, at any resolution,
after the fact — and reuse the poles directly in downstream many-body codes.
`A(ω)` on a grid exists only as a plotting convenience.

## The physics constraints

Hardwired in the decoder's constraint layer — they hold identically at
initialisation, mid-training, and out of distribution
(`src/spevae/models/constraints.py`, asserted in `tests/`):

| property | mechanism | status |
|---|---|---|
| **causality** — all poles in the lower half-plane | `γ = γ_min + softplus(·)` | structural |
| **positive weights**, Re r_p = a_p > 0 | `softmax` | structural |
| **sum rule**, Σ Re r_p = ∫A dω = measured norm | `softmax` × N | structural |
| **zero-sum asymmetry**, Σ Im r_p = 0 with b_p ≠ 0 | `b = a·(t − ⟨t⟩_a)`, `t = tanh(·)/2` | structural |
| **bounded asymmetry**, \|b_p\| ≤ a_p | same map | structural |
| **1/ω² tail** with the physical coefficient | follows from Σ b_p = 0 | structural |
| **width floor** γ ≥ 0.1 π/β (the Matsubara resolution limit) | additive floor | structural |
| **G ≡ 0 ⇒ A ≡ 0**, and A[cG] = c·A[G] exactly | encoder sees G/N, decoder rescales by N | structural |
| **A(ω) ≥ 0 pointwise** | penalty in training + **exact certificate** at inference | certified per result |

Positivity with complex residues is a *collective* property — it cannot be
hardwired per pole without giving up asymmetry — but it is **decidable without
any ω grid**: `spevae.physics.positivity` reduces A ≥ 0 on all of ℝ to a
real-root count of a polynomial built from the pole parameters, and where A
does dip negative it locates the dip and integrates its weight in closed form.
Every reported spectrum carries the verdict. `residue_mode="real"` remains one
flag away when a structural guarantee matters more than asymmetry.

## Why one model covers every (β, Δτ)

With x = τ/β and ν = βω the fermionic kernel becomes K(x, ν) = 1/(e^{xν} +
e^{(x−1)ν}) — **β is gone from the kernel**. One set of weights serves every
simulation cell; the number of time slices is just a sequence length for the
transformer encoder. See [`docs/DESIGN.md`](docs/DESIGN.md) §3.

## Install

```bash
conda env create -f environment.yml
conda activate spevae
pip install -e ".[dev]"
pytest                    # physics invariants, symmetries, calibration, CLI
```

## Use: the posterior report

The current model is a conditional flow over pole sets (`spevae.models.flow`)
trained on pole mixtures, smooth bands and exactly solvable models. Its samples
are fitted to your data and returned as a posterior together with the evidence
for it (`spevae.posterior`). Given a flow checkpoint, `spevae continue` prints
that report:

```bash
spevae continue --bins mybins.txt --beta 8.0 --checkpoint flow.pt \
                --out report.npz --figure report.png
```

- **fit**: how many of the samples fit the bins, and how well;
- **weight per frequency window** with its 5–95% interval, next to the range
  that any non-negative, normalised spectrum consistent with the data could
  take ("allowed by the data alone") — the part of the answer that does not
  depend on the training prior;
- **warnings**: narrow lines the model was not trained on, error bars smaller
  than the systematics it can resolve, every pole in use, and so on.

`report.npz` holds the band of A(ω) on a grid and every accepted pole set;
`report.npz.json` the window table, warnings and diagnostics. From Python the
same call is `spevae.posterior.continue_measured(flow, measured)`.

If the system is an atom, a molecule or a finite cluster, say so with
`--discrete` (`discrete=True`). Many close lines look like a continuous band
in the data, the report cannot tell the two apart, and without the statement
its intervals are too narrow for the lines. Do not use it for an extended
system: there it makes the intervals about ten times wider than needed.

Trained range: fermionic G(τ), β from 1 to 20, relative error bars from 1e-5
to 3e-2, Δτ from 0.02 to 0.5. Measured against exact answers over that range
and on three DQMC runs of exactly solvable clusters
([`docs/TRAINING.md`](docs/TRAINING.md), step 3; drawn in `figures/`). Where
it falls short:

- spectra of a few sharp lines at relative error bars of 1e-5, or at β near
  2: MaxEnt is more accurate there (by up to 1.6 and 1.2 times); in most of
  these cases the report flags narrow lines and says so;
- continuous spectra at relative error bars of 1e-4 and below: the stated 90%
  interval holds the exact weight 79–86% of the time; the misses are below
  0.005 of the total weight;
- a spectrum of many close lines that is not stated to be discrete: intervals
  too narrow, and no warning.

No checkpoint is published yet; the training recipe is
`scripts/step3/run_lines.sh` (label bank from
`scripts/step3/build_physics_bank.py`; about three hours on an Apple GPU).

## Use: the variational autoencoder

```bash
# Stage 1 — pretrain on the infinite synthetic-family generator
spevae pretrain --preset fast --out out/model.pt --device mps

# continue your own bins: sign detected, chi2 calibrated, positivity certified
spevae continue --bins mybins.txt --beta 8.0 --checkpoint out/model.pt \
                --out result.npz --figure result.png

# Stage 2 — adapt the weights to one dataset (no ground truth needed)
spevae finetune --bins mybins.txt --beta 8.0 --checkpoint out/model.pt \
                --out out/model-tuned.pt

# per-input latent refinement (weights frozen) at continue time, and a
# minimal-pole-style output: merge sub-resolution poles, then refit the few
# survivors directly to the data (chi2-neutrality verified and printed)
spevae continue --bins mybins.txt --beta 8.0 --checkpoint out/model-tuned.pt \
                --refine --consolidate --out result.npz
```

[`docs/TUTORIAL.md`](docs/TUTORIAL.md) walks all of this on a real half-filled
Hubbard dataset, with every printed number a measured output.

`--bins` takes whitespace-delimited `(N_bins × L_τ)` text, `τ = 0 … β`
inclusive. The sign convention is **detected, not assumed** — the sum rule
`G(0) + G(β) = 1` is satisfied by exactly one sign — and the run aborts rather
than producing a confident, wrong spectrum when neither works.
`--figure` also writes the pole/residue-plane figure (per-feature uncertainty
as per-pole latent clouds) and the latent-space collapse diagnostic.

Or from Python:

```python
import torch
from spevae import load_model

model = load_model("out/model.pt")
out = model.continue_(G, beta=10.0, tau=tau,
                      omega=torch.linspace(-8, 8, 2001), n_samples=200)
out["poles"]     # the grid-free result: evaluate .spectral(any_omega) later
out["A"]         # convenience grid;  out["A_std"] = latent-ensemble spread
```

Readers for common formats (`spevae.data`): SmoQyDQMC simulation directories
(both storage layouts, auto-detected), the site-Holstein and coupling-resolved
Holstein cubes (cells addressed by physical values), long-format
`[tau_index, G, err]` text, and plain row-per-bin text.

## Three training stages

1. **Pretrain** (`spevae.training.pretrain`) — supervised, on nine qualitatively
   different spectral families generated on the fly across wide (β, Δτ,
   bandwidth, noise) ranges. No fixed epoch, no finite training set: longer
   runs and larger models buy accuracy monotonically. Held-out evaluation uses
   a genuinely frozen batch stream, so those curves are comparable across runs
   and checkpoints.
2. **Fine-tune** (`spevae.training.finetune`) — Stage 2 on your own bins:
   calibrated per-bin χ², free-bits KL, noise-consistency across bins (bins
   *are* independent noise realisations of one spectrum — "different noisy G,
   same A" applied to real data), the spectral positivity penalty, and
   complete-monotonicity penalties on G, G″, G⁗. Default trains the encoder
   only, preserving the pole budget Stage 1 installs; a two-phase divergence
   guard keeps outlier batches from wrecking a pretrained model.
3. **Refine** (`spevae.refine`) — per-input MAP optimisation of z with weights
   frozen, stopping at the *calibrated* noise floor.

The VAE structure is not decoration: different noise realisations of the same
physics map to overlapping posteriors (enforced by a symmetric-KL consistency
term), while genuinely ambiguous data decodes to a visibly spread ensemble —
`A_std` localises where the data stops determining the answer. Report
`active_units` next to any band: a collapsed encoder produces beautifully
tight, meaningless ensembles.

## Benchmarking

```bash
python scripts/benchmark.py --synthetic --checkpoint out/model.pt        # known answer
python scripts/benchmark.py --holstein 19 1 5 --checkpoint out/model.pt  # bundled cube
python scripts/benchmark.py --bins mybins.txt --beta 8.0 --figure cmp.png
```

Three solvers, one interface, one forward map for scoring, one metric set:

- **MaxEnt** — [`ana_cont`][anacont], vendored (MIT). Nothing to install.
- **SmoQyDEAC** — [`SmoQyDEAC.jl`][deac] via a Julia subprocess
  (`$SPEVAE_JULIA_PROJECT`, default `@deac`).
- **spevae** — this package, with `certified_positive` / `min_A_exact` /
  `negative_weight` columns from the positivity certificate.

All three kernels are asserted equal to 1e-12 in the test suite. Every result
is scored through one whitened covariance, so the χ² column means the same
thing in every row. The three methods fail *differently* — MaxEnt broadens
sharp structure, DEAC returns rough spectra, spevae is bounded by its training
distribution — which is exactly why agreement matters: where all three agree,
the feature is in the data; where they disagree, the inversion is
underdetermined *there*.

## Evaluation

```bash
spevae eval --model A=out/a.pt --model B=out/b.pt --tier fast --out out/eval/a_vs_b
```

One frozen suite judges every method: 14 synthetic truths × 2 noise levels ×
8 noise draws plus the real Holstein cell (a 253 kB fixture built by
`scripts/eval/build_holstein_fixture.py`; not versioned), each
spevae inference pipeline next to MaxEnt and DEAC, all scored by the same code.
Error bars are hierarchical bootstraps over training seeds and noise draws;
model comparisons are paired on the same draws. `scripts/eval/acceptance.py`
checks that the suite reproduces the banked judge numbers before it is trusted.
Details: [`docs/EVAL.md`](docs/EVAL.md).

## Reading the results honestly

- **χ² ≈ target does not mean the spectrum is right.** It means the fit is
  consistent with the data at the noise level; the kernel maps a large family
  of spectra to indistinguishable G(τ).
- **The perfect-fit χ² is not 1** under a shrunk covariance estimator —
  `spevae.data.chi2_target` returns the calibrated value (measured 0.112 on a
  real site-Holstein cell, where the shipped SmoQyDEAC reference lands at 0.98
  of it and this package's refined result at 1.39). Comparing to 1 is how
  spurious "χ² < 1 anomalies" get manufactured.
- **`effective_poles`** (participation ratio) is the pole count to report;
  `num_poles` is an upper bound. Prediction quality should be flat-to-better
  as `num_poles` grows — surplus poles fade under the `softmax`, they don't
  fit noise (the noise budget for recoverable poles is real; see
  `docs/DESIGN.md` §2).
- **`active_units`** belongs next to every uncertainty band.

## Provenance

This is the third iteration. `../SPE-AC-VAE` (v1) established the ansatz and
the calibrated-χ² protocol; `../SPE-AC-VAE-2` rewrote it grid-free and
β-universal. v3 started from a **line-by-line audit of v2** (13 defects found
and fixed, including a mis-calibrated whitener on synthetic data and an
evaluation stream that was not actually held out), made complex residues the
default with the exact positivity certificate, added Stage-2 fine-tuning, the
real-data readers, the CLI, and the pole/latent figures. Every claim in this
README is asserted by a test or measured in `docs/DESIGN.md`.

## Method attribution

The pole/residue output ansatz with hardwired sum rule and zero-sum tail, the
covariance-whitened χ² calibration idea, the variance-weighted G/G″/G⁗
negativity penalties, and the two-stage pretrain/fine-tune protocol are due to
**Benjamin Cohen-Stead** (UT-Knoxville). See `docs/DESIGN.md` §10 and the
reference list there — including the closest published prior work
(Kliczkowski *et al.*, PRB **110**, 115119 (2024): a VAE-type encoder with a
fixed grid-kernel decoder) and the pole-representation literature this package
builds on (Zhang & Gull, PRB **110**, 035154 (2024); Huang & Liang, PRB
**108**, 235143 (2023); Fei, Yeh & Gull, PRL **126**, 056402 (2021)).

## Licence

MIT.

[anacont]: https://doi.org/10.1016/j.cpc.2022.108519
[deac]: https://scipost.org/SciPostPhysCodeb.39
