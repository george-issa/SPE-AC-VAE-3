# Training the production model

This document records how the shipped model should be trained so that a user
can trust it, and why each practice is there.  Everything marked *measured*
comes from an experiment in this repository, not from taste.  DESIGN.md covers
what the model is; this covers how to train it well.

## 1. What the pretraining campaign established

The stage the ill-posedness is actually beaten at is pretraining (DESIGN.md
§7): on real data no loss can distinguish two spectra that fit G(τ) equally
well, so the ability to tell them apart has to be learned where the answer is
known.  The 2026 pole-first campaign tested how, in controlled A/Bs at fixed
budget (6k steps, fast preset, identical seeds and frozen judges):

- **Mixed data wins** (*measured*).  50/50 pole-labeled batches (motif prior
  with exact canonical pole labels, OT loss in pole space) and legacy A-space
  family batches (W₁/L² losses) beats either alone.  Pole-only training gives
  a ~1.9× pole-placement advantage on pole-labeled truths but loses badly on
  family coverage (W₁ 0.495 vs 0.276); family-only training cannot place
  poles.  The mixed model held most of both: legacy W₁ 0.392, pole-stream W₁
  0.279 vs the control's 0.491, and the lowest ripple of any arm on both
  streams.
- **OT ε-annealing bought nothing** (*measured*).  Fixed ε = 0.1 matched the
  annealed schedule.  Keep the simpler recipe.
- **The stack-coherence term hurt** (*measured*).  Pulling co-assigned poles
  toward their transport centroid raised ripple on every stream at weight
  1.0.  The code remains (`pole_coherence`), default 0.0; do not turn it on
  without a micro-tier sweep showing otherwise.
- **The family-coverage gap is interference, not just data budget**
  (*measured*, and this falsified the first hypothesis).  Scaling the mixed
  recipe 6k → 24k narrowed the legacy-stream gap (W₁ 0.392 → 0.329) while
  the pole-space advantage grew (pole-stream W₁ 0.279 → 0.255 vs the 24k
  control's 0.444) — but the matched-budget control stayed ahead on family
  coverage (0.231), and the mixed model underperforms even the 6k pure
  control (0.276) despite seeing ~12k family batches.  The two objectives
  interfere; more steps alone will not close it.
- **The curriculum breaks the trade-off** (*measured*, micro tier, 3
  seeds).  Sweeping p_pole at fixed β = 8: every constant ratio pays a
  family-coverage cost over pure-family training (legacy W₁ 0.572–0.581 vs
  0.545), monotone in the ratio — micro-scale confirmation of the
  interference.  A linear 0.9 → 0.1 curriculum (pole-heavy early,
  family-heavy late) *eliminates* it: legacy W₁ 0.527 ± 0.008, at or below
  the pure-family arm with a far tighter seed spread, while retaining a
  1.5× pole-placement advantage (pole-stream W₁ 0.401 ± 0.001 vs
  0.590 ± 0.055) and better legacy L².  The reading: early pole supervision
  shapes the decoder's representation; late family-heavy training refines
  A-space accuracy without erasing it.  Flags: `--p-pole 0.9
  --p-pole-final 0.1`.
- **Full-width confirmation: a strictly better trade-off point, with the
  usual constants caveat** (*measured*, fast 6k).  Versus constant 50/50
  at identical budget the curriculum improved family coverage (legacy W₁
  0.392 → 0.326), kept a 1.47× pole advantage (0.334 vs the control's
  0.491), and had the lowest ripple of all three arms on both streams.
  The micro-tier promise of *matching* pure-family coverage did not fully
  transfer — a 0.05 residual gap to the family-only control remains —
  exactly the promote-recipes-not-constants caveat in action.
- **At 24k the residual halves and the gate passes** (*measured*).
  Curriculum legacy W₁ 0.254 vs the matched control's 0.231 (gap
  0.050 → 0.024 as budget went 6k → 24k, i.e. shrinking with budget,
  unlike constant mixing) with χ²_geo within 15% of the control and a
  1.41× pole-placement advantage retained (0.315 vs 0.444).  The
  curriculum is the production recipe; the remaining ~10% relative
  coverage cost buys the pole-space accuracy that family-only training
  never acquires at any measured budget.

## 2. Where the time goes, and the tier ladder

Per-step cost is overhead-bound, not compute-bound (*measured*: data
generation is 8–36 ms/batch; a full step is ~106–165 ms with the GPU ~17%
busy on sequences of n_τ ≤ 176).  Wall time is therefore ≈ steps × a fixed
per-step cost, nearly independent of how wide the training distribution is.
Cheap experiments come from *fewer steps*, and a narrow distribution
converges in fewer steps.  Hence the ladder — each rung is one command-line
change, and results are only ever promoted upward:

| tier | flags | cost | answers |
|---|---|---|---|
| micro | `--beta-min 8 --beta-max 8 --dtau 0.125 --families …`, ~1k steps | 1–2 min | does this loss/knob help at all? (run 3 seeds) |
| identity | micro + one family, 12k steps | ~10 min | ceiling: can the model invert its own forward map? |

Identity-tier results on the Lorentzian world (*measured*, 12k steps,
β = 8): both supervision modes pass the A-space gate — probe W₁ 0.04–0.11
(A-space losses) and 0.02–0.06 (pole OT losses, `--data poles --motifs
lorentz`) against a thermal resolution of π/β ≈ 0.39, shape error ~1% —
with the pole-loss variant reaching up to 3.6× better peak placement.
Two shared residuals, both informative: (i) neither drives χ² to 1 at the
lowest noise (χ² ~10²–10³ at σ ≈ 10⁻⁴; the amortized pass leaves a
~10⁻³-level G-residual that barely affects A — the per-dataset refine
stage exists to close exactly this); (ii) the pole-supervised model still
*stacks* (9–15 effective poles on 1–2-pole truths) — OT's free
co-location means minimal labels do not induce minimal outputs;
minimality remains the consolidation stage's job.
| fast 6k | preset defaults, β ∈ [6,10] | ~16 min | does the winner survive full width? |
| fast 24k | `--steps 24000` | ~1 h | pre-production confirmation |
| full | `--preset full`, β ∈ [2,20], larger d_model, ≥100k steps | hours | the shipped model |

**Micro probes must match the training grid exactly** (*measured*, the hard
way).  A micro-trained model has seen exactly one τ-grid; probing the
Lorentzian identity model on a 64-point midpoint grid instead of its
65-point endpoint grid turned W₁ 0.04 into W₁ 1.35 — a catastrophic score
that was pure grid mismatch, not model failure.  Narrow tiers are
single-grid specialists; only wide training buys grid robustness.  Any
probe of a micro checkpoint must reuse the generator's own (β, Δτ) grid
convention.

Promote **recipes, not constants**: which terms are on and their rough
ratios transfer between tiers; tuned magnitudes and convergence step counts
do not (gradient scales differ across families).  Never warm-start a wider
tier from a narrower tier's checkpoint — a model converged on a narrow
distribution is a biased prior, and a from-scratch fast-tier run costs
minutes.  `--init-from` is for rescuing killed runs and extending
same-distribution runs only.

## 3. Fixed judges

Every claim of improvement must come from yardsticks that never move during
a campaign:

1. **Frozen legacy stream** (`scripts/pole_count_report.py`): the full
   family generator under `fixed_scalars`, so rows differ only in the model.
   W₁, relative L², χ²_geo, active units, effective pole count.
2. **Frozen pole-labeled stream**: same protocol on `PolePriorGenerator`;
   this is where pole-placement accuracy and the ripple/ripple-excess
   metrics live (excess = prediction's sub-thermal oscillation minus the
   truth's own — the pole-comb artefact isolated from real structure).
3. **Named probes**: single physical cases scored end to end (the
   two-Lorentzian probe; the ground-truth suite in
   `scripts/eval_groundtruth.py` — semicircle, pseudogap, gapped Mott,
   double Gaussian — with the honest-resolution convention of DESIGN.md §8).

Aggregates hide the failure that matters, so every eval stratifies per
family; the release gate below is stated per family, not on the mean.

## 4. The production run

Recipe, from the campaign:

- `--data mixed --p-pole 0.9 --p-pole-final 0.1` (the measured curriculum;
  constant 50/50 pays a coverage cost — see §1), `--pole-eps1 0.1` (no
  annealing), coherence off.
- **No positivity pressure in training** (design decision, 2026-08-21):
  the positivity loss defaults to 0 and training labels are allowed small
  negative dips (bounded at 0.5% negative spectral weight via `dip_tol`,
  not banned).  Forcing A ≥ 0 during training biases the prior — small
  dips are sometimes what a good reconstruction needs.  Positivity is
  enforced where the user is owed it: the consolidation refit and the
  certificate gate at inference (release gate 4 is unchanged).
- Full preset: β ∈ [2,20], all Δτ choices, all families, complex residues.
- EMA on (0.999) — the shipped weights are the EMA weights.
- Full cosine schedule; **no early stopping** on the production run.  Early
  stopping is for exploratory tiers where the question is cheap; the
  production run should spend its entire schedule (*measured*: eval metrics
  were still descending at every milestone of every run we did).
- Rolling `.partial` checkpoints stay on; a multi-hour run must not be
  losable to one kill (this happened three times before the guard existed).
- Scale in this order: steps first (pure data budget, monotone), then
  d_model/enc_layers (capacity), then P (`--num-poles`) only with the
  noise-regime map in mind — past the noise-supported rank extra poles buy
  nothing on noisy data (Gull's noiseless monotonicity does not transfer;
  DESIGN.md §7).
- Run ≥3 seeds of the final configuration.  The seed spread on the frozen
  judges is the epistemic error bar quoted for the model itself; if the
  spread straddles a release criterion the run was too short.

### On Perlmutter (or any A100 machine)

The port is `--device cuda` plus a Slurm script; the code is plain PyTorch.
Worth adding when the production run is set up, in this order of payoff:
worker-process prefetch of batches (generation currently serializes with the
step), `torch.compile` (fuses the many small ops that dominate the step),
bf16 autocast, larger batch size (near-free throughput when overhead-bound).
Use multiple GPUs for *seeds and arms*, not DDP — the model is far too small
to shard.  Keep micro-tier iteration local: queue wait exceeds runtime.

## 5. Reliability gates before shipping a checkpoint

A user should be able to run the model blind.  A checkpoint is releasable
when, and only when:

1. **No per-family regression** on the frozen legacy stream vs the previous
   released checkpoint (W₁ and L² per family, not the average).
2. **Pole-stream advantage retained**: pole-labeled W₁ at least as good as
   the released model's; ripple excess ≤ 0 (smoother than the truths
   themselves, never noisier).
3. **Ground-truth suite** at both noise regimes of the noise-regime map
   (high noise where we beat MaxEnt, low noise where MaxEnt wins) with
   honest-resolution scoring; no case may cross from win to loss.
4. **Structural invariants spot-checked at inference**: G≡0 → A≡0 exactly
   (1-homogeneity), Σ Re a = 1 × measured norm, positivity certificate on
   the decoded pole set, all γ above the width floor.
5. **χ² calibration** on at least one real dataset (Hubbard/Holstein bins):
   χ²/target within [0.8, 1.5] after the statistical-slice exclusion; a
   model that cannot reach its own noise floor on real bins is not shipped.
6. **UQ coverage** (once calibrated intervals exist): nominal 68/95%
   intervals must cover at their stated rates on the frozen streams.

Ship with the eval harness versioned alongside the weights: the checkpoint,
the judge scripts, and the frozen-stream seeds together are the release.

## 6. Practices with negative results on record

Kept here so they are not re-tried by accident (details in git history and
DESIGN.md):

- Stack-coherence loss at weight 1.0 — raised ripple everywhere.
- OT ε-annealing — no measurable benefit over fixed ε.
- argmin-χ² model selection — over-resolves; use the parsimony elbow.
- MPM/AAA as a noisy-bin engine — four falsified iterations; it is a
  noiseless validation engine and a label generator only.
- Warm-starting across distribution widths — see §2.
