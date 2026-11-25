# Evaluation: `spevae eval`

One frozen suite that every continuation method goes through: spevae checkpoints
under each inference pipeline, MaxEnt, DEAC, and any future method. Before this
suite, the judges were ~50 one-off scripts with hardcoded paths, one noise draw
per probe, and no baselines in the judged results. The suite replaces them for
judging; the scripts stay as the historical record.

```bash
# fast tier (minutes): CI, screens
spevae eval --model s1=out/step1/s1_metric.pt --model M1=out/step2/map_M1_60k.pt \
            --tier fast --out out/eval/s1_vs_M1_fast

# training-seed replicates: repeat a tag
spevae eval --model M1=run_seed0.pt --model M1=run_seed1.pt --model M1=run_seed2.pt ...

# acceptance: reproduce the banked numbers (needs out/ on this machine)
python scripts/eval/acceptance.py out/eval/acceptance
```

Outputs: `scorecard.json` (provenance, every per-draw record, the summary, paired
comparisons), `summary.md` (tables), `w1_scorecard.png`, `holstein.png`,
`spectra_<noise>.png`.

## What is judged

**Synthetic cases** (`evaluation/cases.py`, the single definition): 14 truths at
β = 8 on the 65-point endpoint grid — nine pole-closed (ladders, mirrored ladder,
edged band, irregular and asymmetric triplets, Fano band, Lorentzian pair) and five
smooth (Gaussian pair, gapped pair, semicircle, box, pseudogap). Correlated noise
(ξ = 0.1) at 1e-4 and 1e-3. Seeds 7, 8, 9, …: seed 7 is the banked draw; seeds come
in pairs so the paired-noise consistency is measured on every pair.

**Real cell**: the Holstein cell (β = 8, Ω = 1, n = 1, λ = 0.325), shipped as a
253 kB fixture (`evaluation/fixtures/`, built by `scripts/eval/build_holstein_fixture.py`
from the 9.9 GB cube and pinned by sha256). It reproduces the cube-read cell bit for
bit. The shipped SmoQyDEAC continuation of the same cell is a free comparator
(`deac_shipped`).

**Methods** (`evaluation/methods.py`):

| spec | what it is |
|---|---|
| `amortized` | one deterministic forward pass |
| `zd800`, `zd3000` | latent MAP descent (weights frozen, prior anchor 1), Adam for that many steps; on real data `refine_latent` |
| `zdlm` | the same MAP objective minimised by Levenberg–Marquardt (`physics/lsq.py`) to a stationary point, or on a real cell to the calibrated noise floor; each row records `converged`, `stationary` and the iteration count |
| `scan_minimal`, `scan_full` | z-descent 1500 (MAP 3000 on real data) → anchored rank scan (anchor 0.5 synthetic / 1.0 real, pre-refit 3000, drop reduction): the LR-selected minimal form and the prior-started full rendering (the dual output) |
| `maxent` | ana_cont MaxEnt, flat default model, χ²-kink α. Synthetic cases get the exact noise covariance σᵢRᵢⱼσⱼ, the same information spevae's whitener uses |
| `deac` | SmoQyDEAC (real cells only in v1; it needs raw bins) |
| `deac_shipped` | the SmoQyDEAC result shipped with the real cell |

Not yet in the suite: the template menu (J2/J3 judges), DEAC on synthetic cases
(needs synthetic bins), the count-fidelity judge (J10), and held-out smooth draws
(the five smooth truths come from the label-bank families, so they become
in-distribution once step 2 trains on the bank).

**Known-answer judge** (`scripts/eval/exact_models_judge.py`, outside the frozen
suite for now): 20 exact lattice truths and 12 exactly diagonalised ones from
`data/exact_models.py`, on one noise draw, continued by the refined flow, the VAE
latent MAP and MaxEnt, scored by W₁ between cumulative weights (exact step functions
for the delta spectra). Until a checkpoint is trained on these families the numbers
are an out-of-prior test; afterwards they are in-distribution on held-out parameters,
and generalisation has to be read from a leave-one-family-out arm.

## One scorer

Every method returns a pole set or a spectrum on a grid; everything else is computed
by the same code (`score_synthetic`, `score_real`):

- **W₁** to the truth in physical units (`spectral_w1(..., beta)`);
- **χ²**: whitened, per point, against the noisy data, through one forward map (the
  quadrature of `SpectralQuadrature.for_beta`, the same map that made the clean
  data). On the real cell: the `refine_latent` convention — structured covariance,
  eigenvalue floor 1e-10, statistical slices, σ/√N, divided by the number of
  independent slices — reported as a ratio to χ²_target;
- **negative weight**: the exact positivity certificate for pole sets, the
  quadrature of the negative part for grids;
- **pole count**: poles carrying > 0.1% of the weight; the scan's selected n.

Each method's own χ² is kept as `chi2_internal` for the acceptance check and is never
used for ranking. The scorer's χ² equals z-descent's own χ² exactly on synthetic data
(tested).

**Fixed step counts are not convergence (measured 2026-11-24).** The `zd<N>` methods
and the banked Holstein "manifold floor" (Adam, 10,000 steps, prior off) stop at a step
cap. Re-run that day, the floor descent of every reference checkpoint was still
falling at the cap (`converged=False`), and Levenberg–Marquardt on the same objective
reaches the noise target for all four — s1_metric (banked floor 23.96) in 662
iterations, M1 (3.83) in 81, ctrl60 (3.22) in 52, D24 (0.21) in 44. The floor record
therefore carries both: `floor` / `chi2_map` (the banked Adam protocol, kept so the
acceptance check still reproduces, now with `floor_converged`, `floor_steps_run`,
`map_converged`, `map_steps_run`) and `floor_lm` / `map_lm` with a status of
`at target`, `stationary` (a genuine stationary point above the target) or `budget`
(ran out of iterations — an intermediate answer, not a result). Quote the
least-squares numbers; read a banked "floor" as a descent-speed measurement.

**Known convention split (pre-existing, measured 2026-11-23):** on real data,
`refine_latent` whitens with σ_cov/√N while the rank scan is called with
`mg.sigma_mean`, larger by √(N/(N−1)) (0.125% at N = 400). The two χ² conventions
therefore differ by 0.25%; the scorer uses the first, which is the one χ²_target is
computed under.

## Statistics

`evaluation/stats.py`. Means with percentile-bootstrap 95% intervals that resample
training runs first and noise draws within each run second, so one lucky checkpoint
cannot narrow the interval. Model comparisons are **paired** (both models saw the
same draws): the geometric-mean error ratio with a paired bootstrap interval and the
fraction of draws on which the second model is better.

**The paired interval is a *t* interval on the log-ratios** (changed 2026-11-24).
It was a percentile bootstrap of the mean log-ratio; with two pairs that is exactly
[min, max], so it "excluded 1" whenever the two pairs agreed in sign. Simulated rates
of the `**` flag for identical arms (600 trials through `compare_pairs`): 49% / 19% /
11% at n = 2 / 4 / 8 before, 4.2% / 5.5% / 4.8% now. Reports rendered before this
date carry the old flags; the scan-pipeline rows (first seed pair only, n = 2) were
never significance statements.

**Known limit, not fixed.** `cases.draw` seeds the noise generator with the seed
alone, so every case and both noise levels reuse the same eight standardised noise
patterns; cells are not independent draws of the noise. Changing it would move every
banked number, so it stays; the step-3 judges (`scripts/step3/calibration.py`,
`scripts/eval/exact_models_judge.py` with a seed argument) draw their own noise per
truth.

**Read the intervals with care.** Reconstruction error varies far less across noise
draws than across training seeds (measured: two 6k M1-recipe seeds differ 2.4× on the
irregular triplet while noise-draw intervals are a few percent wide). An interval
over noise draws alone says how stable a *checkpoint* is, not how good a *recipe* is.
Recipe claims need several training seeds per arm.

## Determinism

Every worker pins torch to 2 threads; tasks carry explicit seeds and are independent,
so results do not depend on worker count or scheduling. The banked judges that pinned
2 threads are reproduced bit for bit (see the acceptance record).

## Tiers

| tier | contents | wall (6 workers) |
|---|---|---|
| `fast` | streams; amortized and MaxEnt on all 14 cases × 2 noises × 8 seeds; zd800 on the 3 screen probes × 4 seeds; Holstein: amortized, MAP, scan, MaxEnt, shipped DEAC | ~2 min |
| `full` | adds zd800 and zd3000 on all cases × 8 seeds, the scan pipeline on all cases (first seed pair), the Holstein manifold floor, DEAC on Holstein | 33 min for two models (5 workers) |
| `lm` | `zdlm` on all cases × 8 seeds and on Holstein, plus the floor record; set its rows next to a `full` card of the same models (same cases, same draws) | |
| `accept` | the banked judges' exact protocols on the banked draws | ~5 min |

## Acceptance record (2026-11-23)

`python scripts/eval/acceptance.py` on s1_metric and M1-60k against every banked
artifact in `out/` (212 checks): **204 pass, 0 fail, 8 drift**.

- **Bit for bit** (rtol 1e-9): both frozen streams; z-descent 3000 χ² on the
  12-truth set at both noise levels; the scan pipeline's χ² and selected n on the
  Fano band, the asymmetric triplet and the Holstein cell (minimal and full
  rendering); Holstein amortized χ², MAP χ², and the manifold floor.
- **W₁** (rtol 1e-4): amortized, z-descent 3000 and scan-pipeline W₁ agree to
  ≤ 6e-5. The judges evaluated spectra on float32 nodes (most) or float64 nodes
  (dual output); the suite uses one convention.
- **Drift** (8, all from `map_judge60k.json`, banked 2026-09-01 under the previous
  Python/torch environment): six paired-consistency values (a difference of two
  near-equal reconstructions, 0.3–8% relative) and M1's semicircle at 1e-4
  (W₁ +1.3%, χ² +4.4%, consistency 2.3×). Thread count is not the cause — z-descent
  800 on that case is identical at 1, 2, 4 and 8 threads — so this measures how much
  z-descent on that case amplifies float-level differences between library
  versions. Differences between arms smaller than that on that case are not
  meaningful.

Not covered: the template-menu judges (J2/J3: "seven probes green"), the anchor-weight
sweep (J5) and count fidelity (J10). They join in v1.1.
