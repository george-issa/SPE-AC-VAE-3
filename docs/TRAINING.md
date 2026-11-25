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

Residual (i) is confirmed to be refine's job (*measured*): staged
whitened refit on the three probes — no consolidation — takes χ² from
30–540 to ≈ 0.87 within the first 500–1000 steps, and the refined
spectra are near-exact: W₁ 0.0008–0.016, with the pole-loss model's
single-Lorentzian case recovering the true pole to four digits.  The
amortized pass supplies the basin; refine supplies the digits.  (χ²
settling ~1σ below 1 on 65 points is the expected mild noise
absorption of an unconsolidated 16-pole fit.)

**Ladder extension** (*measured*, `--motifs ladder`, the next pole-closed
family up: 2–4-rung combs, decaying weights, up to 12 poles with
mirroring).  A(ω) accuracy transfers at every budget (consolidated W₁
0.015–0.031, χ² ≈ 0.85); exact structural recovery is harder and its
residuals decompose three ways, each measured: (i) *budget* — 12k → 24k
fixed the 3-rung count (5 → 3 poles) and brought dominant rungs to 2–3
digits; monotone, as the basin picture predicts; (ii) *information* —
at noise 3×10⁻⁵ the 3-rung case becomes essentially exact (W₁ 0.008,
weakest rung recovered), so its 10⁻⁴-noise residual was the
identifiability edge, not the model; (iii) *basin* — the 4-rung's ~0.08
position offsets and the mirrored case's 6-poles-asymmetric answer are
unchanged by better data: only a stronger amortized basin (or, for
genuinely PH-symmetric data, the mirror-constrained consolidation mode)
removes them.  Negative result, recorded: swapping the smooth-biased ρ
draw for a resolved-comb prior (ρ ~ U(2, 4.5)) at matched budget made
*everything worse* (amortized W₁ ~2× worse, counts 5/5/6) — the
resolved world is the harder curriculum, and the smooth-biased prior's
tail transfers better than a resolved bulk trained to the same cost.
The smooth-biased ρ stays.

**Step 1 of the scaling ladder: PASSED — the recipe is the width metric
alone** (*measured*, β = 8, full motif breadth, pole-loss only, factorial
A/B/C).  Arm A (metric + concentration 1.0), B (neither), C (metric
only), 60k each, judged on frozen streams, the legacy canary and the
five fixed probes through the full inference pipeline.  C dominates:
best stream (W₁ 0.197, l2 0.103, ripple 0.075), best canary (0.438),
and all five probes passed with correct structure — including the
mirrored case (so concentration was NOT load-bearing there) and the
irregular triplet with the *correct* peaks(3) hypothesis.  A passes all
probes only after inference hardening and B fails mirrored + misses the
4-rung faint rung.  Attribution: the width-relative metric carries every
win; concentration at 1.0 adds nothing at breadth and over-concentrates
multi-peak proposals (the measured triplet regression).  Production
objective going forward: `--pole-metric width`, no concentration.
Inference hardening adopted along the way: RankScanTrace exposes per-n
fits; templates start from the scan's n-matched fits.  Legacy canary
baseline banked for step 2 (MPM labels) to close: W₁ 0.438.

**The b channel probed end to end** (*measured*, step 1 extension).
`peaks_asym` (b free per peak; Σb = 0 and |b| ≤ a structural; k = 4n−2)
plus two certified b ≠ 0 probes.  Fit topology, learned the hard way:
symmetric fits *absorb* real asymmetry into shifted positions — a
separated basin that staged b-release and even an exact linear b-solve
at converged geometry both fail to escape; only joint optimisation from
asymmetric t-sign multi-starts reaches it.  Verdicts: the fano band is
recovered cleanly by peaks_asym(4) (χ² 0.90, W₁ 0.021, beating free(5)
with 6 fewer parameters; the symmetric menu sat at 2.1); the b = 0 menu
on the asymmetric triplet over-counts exactly as constraint (iv)
predicts (peaks(4) — an extra symmetric pole buying the asymmetry).
With the working asym entries the triplet becomes a measured
near-degeneracy: peaks_asym(3) (true structure, k = 10, χ² 1.054) vs
peaks(4) (k = 11, χ² 0.963) differ by ~6 total χ² — the
asymmetry-vs-extra-peak discrimination is genuinely at the evidence
edge at σ = 10⁻⁴, and the LR walk initially kept peaks(4) by 5.9 vs
the 3.84 toll.  **Resolved by the borderline fine-polish rule extended
to select_model** (*measured*): re-polishing both competitors at floor
quality (warm-started, 12k steps, lr/2) dropped peaks_asym(3) from
1.054 to 0.880 — *below* its pricier rival — and the true structure
wins with W₁ 0.0115.  The degeneracy was optimiser orbit, not physics:
6 units of total χ² was exactly the convergence slack the rank scan's
version of this rule was built for.  free(n) is now charged its
effective k = 4n−2 (sum rule and Σb = 0 each remove one dof).

**The latent landscape, measured** (*measured*, fano + asym-triplet
probes, arm C).  Amortized inference is one feedforward pass — nothing
descends at inference — and the experiment shows what that costs: the
posterior-*mean* readout sits 30–40× above the decoder manifold's best
fit for the same data (χ² 530 vs 15 on fano; 211 vs 6.6 on the
triplet).  Gradient descent of χ²(decode(z)) in latent space finds far
better manifold points: fano W₁ 0.097 → 0.029 (approaching
template-winner quality) while staying on the physics manifold — no
absorber poles, no width-inflation trades, because the decoder never
learned to emit them.  Posterior *sampling* is the weaker effect
(best-of-64 only modestly beats the mean): the encoder is mis-centred
more than multimodal-and-sharp.  Two boundaries also measured: z-descent
cannot reach the noise floor (χ² 6–15 vs target ~0.9 — decoder
resolution; pole-space refits still own the digits), and better χ² does
not guarantee better W₁ within the data-equivalent class (triplet
0.054 → 0.059).  Consequence for the pipeline: z-descent is the natural
pre-scan stage — manifold-constrained basin selection feeding the scan
and template nomination — with per-dataset finetune (stage 2, encoder-
only, anchored) as the escalation for out-of-distribution data.

**z-descent wired in, and it rewrites checkpoint A** (*measured*).
With `refine_latent` (structured whitener) feeding the scan on the
Holstein cell: the proposal improves on-manifold χ² 916 → 37, and the
rank scan HEALS — a clean monotone cliff to χ² 0.56 at n = 6 (vs the
old collapse at 4,880, non-monotone).  The winner flips from
mirrored_ladder(3) (χ² 3.5) to free(6) (0.56), and the real-data gap
shrinks **20× → 3.2×**: most of the "spectral mismatch" was proposal
failure, not vocabulary.  The physics reading sharpens rather than
reverses — free(6) matches the mirrored ladder's pole COUNT (3 mirrored
pairs = 6) but rejects the ladder's shared-width/geometric-decay
constraints at 6× in χ²; the data is asking for the menu entry we don't
have yet: **mirrored_peaks(n)** — n free mirror pairs, the gap-Δ +
sidebands vocabulary item.  On the b-probes with z-descent starts: fano
upgrades (peaks_asym(3), polished 1.134 → 0.967, W₁ 0.0123, k = 10 —
more parsimonious AND more accurate than the previous asym(4) verdict);
the asym triplet's scan count locks at the true n = 3, and the menu
surfaces a genuine equal-k tie — free(3) at 0.920 vs peaks_asym(3) at
0.934 (Δχ²·L ≈ 1, data-equivalent) where the free arrangement carries
5× worse W₁; whether structured entries deserve the tie-break at equal
complexity and equal fit is an open selection-policy question,
deliberately not auto-decided.

**The anchored rank scan** (*measured*, built from the merge-confound
finding + the χ²+L2 question): per-n refits carry the manifold anchor
term in their descent objective while the LR count test reads pure χ²
(the prior shapes optimisation, never votes on evidence); candidates
descend by drop-lightest from a pre-refit of the FULL configuration
(the cold merge is gone from the anchored path).  Same seeds, same LR
rule as the old scan: identical certified counts (5/3/6 — all true or
band-consistent), better χ² everywhere, and **2–3× better minimal-form
W₁** (fano 0.054 → 0.024, triplet 0.062 → 0.023); the Holstein reaches
χ² 0.106 — inside the calibration band — in one integrated pass,
matching the refit-then-gate answer with cleaner machinery.  The
remaining gap to the 16-pole prior-started results (fano 0.011) is
representational (5 vs 16 poles), not procedural.  Canonical per-dataset
order is now: z-descent → **anchored rank scan** (pre-refit, drop init,
w ≈ 0.5–1) → menu.

## Step 2 roadmap (locked scope: one variable — the training data)

**Goal**: grow the decoder manifold to express smooth, non-pole-closed
spectra — the measured deficiency behind every remaining gap.

**What changes**: the pole prior gains MPM-labeled smooth families
(Gaussian mixtures, semicircle, box, gapped pairs, pseudogap — the
legacy catalog).  **Labels are noise-INDEPENDENT** (user correction
2026-08-26, consistent with step 1's exact labels and with real data
where σ is only estimated): one fixed label per truth — the
machine-precision MPM extraction, canonically truncated by weight to
P_max (capacity-matched; truncation residual ~10⁻⁴–10⁻⁶, far below any
realistic noise; deterministic extraction protocol = the canonical
label).  The noise does its teaching on the INPUT side only, exactly as
in step 1 — the mass-weighted OT loss absorbs unresolvable detail and
the network learns σ-calibrated hedging.  Pole-loss only; no ω-grid
enters training.  **What stays frozen**: objective (width metric, no
concentration), β = 8, all judges.  **Latent width is the one
in-step factorial** (user remark: AU has been pinned at cap in every
run): arms at latent 24 (control) and 32/48, judged identically — the
data change is the step; capacity is swept within it, as the objective
was within step 1.

**Build order**: (1) label bank + unit tests (bank reproduces A_true to
noise-resolution tolerance; rank grows with SNR); (2) generator
integration (smooth families as new motif classes); (3) training A/B vs
s1_metric at 60k; (4) smooth-shape probes (gaussian pair, semicircle,
gapped) + their menu templates via the same noise-matched pole maps;
(5) real-data checkpoint B.

**Pass gates (all banked numbers)**: manifold floor on Holstein
24.0 → down materially (THE gate — but see the 2026-11-24 re-measurement
in the step-2 roadmap: the 24.0 was a first-order descent stopped at its
step cap, not a floor); legacy canary 0.438 → down (its
families become in-distribution — now a direct target); seven step-1
probes stay green; new smooth probes recover their noise-matched labels;
Holstein w-sensitivity flag flattens; dual-output full renderings lose
their spikes at equal calibration (the current gold Holstein curve is
the "before" picture).

**Coverage protocol** (how "wide, rich, meaningful enough" is made a
measured property): (1) mechanism vocabulary + composition (K ≤ 3,
mirroring, physics-scaled ranges) — coverage in mechanism space, not
function space; (2) a literature audit table: published DQMC spectral
phenomenology per target model → motif composition, gaps becoming new
families with menu entries and probes; (3) leave-one-family-out
training arms — the generalization matrix replacing coverage claims;
(4) real data as permanent auditor via the manifold floor and the
w-sensitivity flag, closing the loop gap → mechanism → family →
retrain; (5) no prior-hacking: ranges from physics scales, every
distribution change A/B'd (the ρ lesson); (6) stratified judging:
per-family streams, per-family probes, rotating canary, AU vs latent
width.  Label-truncation residual: estimated 10⁻⁴–10⁻⁶ from geometric
rational-approximation decay + measured MPM ladders; pinned per family
by the bank's first unit test.

**Grid convention hardened NOW, not deferred** (user directive): the
80/81 lesson is enforced by construction — `prepare_inference(mg)` is
the blessed real-data entry point (encoder = full native grid, fits =
statistical slices, whitener attached), z_descend carries the warning,
and a regression test pins the convention on sum-rule-redundant data.

**Deferred, in order, each its own arm**: latent-organization heads
(on/off A/B on the step-2 checkpoint); β-extension 8 → [6,10] → [2,20]
with grid-robustness augmentation; Perlmutter port +
production run; UQ calibration; MaxEnt/SmoQyDEAC benchmarks; release
gates.

**Foundation-model track** (adopted 2026-11-23, user direction: keep it
active alongside the stepwise protocol; every item is an arm judged by
`spevae eval`).  The information limit is irreducible, but the measured
gaps are model-limited (Holstein manifold floor 3.8 vs target 0.175;
amortized χ² 10²–10³; best-of-64 posterior samples fail to cover
balanced pairs), and those are what the ideas below target.

| idea | where it lives now | status | next arm |
|---|---|---|---|
| learned prior × exact likelihood | decoder manifold + N(0,I) latent = prior over pole sets; z-descent = MAP under the exact χ² (the kernel enters only through the forward map, so the decoder is kernel-agnostic; the encoder is the kernel-specific proposal) | in place, MAP only | generative prior over pole sets (flow matching / diffusion) with exact-likelihood guidance → posterior samples, calibrated bands, one prior for every kernel |
| conditional generative decoder | — | not started | keep the encoder, replace the deterministic decoder with conditional flow matching over pole sets; head-to-head vs the VAE |
| data diversity over parameters | step-2 data arm (smooth label-bank truths) | running | physics-sourced corpus: small-cluster ED (exact pole labels), DMFT, exactly solvable models |
| scaling curves | latent width 24/32/48 factorial in the step-2 screen | running | network-size ladder, on the generative model where capacity matters more |
| test-time compute + verifier | z-descent, anchored rank scan, LR count test, positivity certificate (~60 s per dataset) | in place | best-of-N / guided sampling once a generative prior exists |
| fine-tune on real data vs the likelihood | `spevae finetune` (encoder only) | exists, unused in judging | judge it on the Holstein cell |
| breadth across kernels | fermionic G(τ) only | not started | bosonic / conductivity kernels through the same decoder (forward-map swap) |

**Step-2 data arm, 6k screen** (*measured*, 2026-11-23;
`scripts/step2/data_arm_screen.sh`, judged by `spevae eval --tier screen`,
`out/eval/step2_screen6k/`).  Bank: v1-protocol labels at β = 8, 1024
truths per family (seed 2), filtered to data gap < 3e-5 (5088 of 5120 kept;
`scripts/step2/filter_bank.py`); every kept label reachable by the M1
decoder, loss load Σb²/a ≤ 0.25.  Arms: M1 recipe + smooth bank truths at
p_smooth 0.3, latent 24 / 32 / 48; control the M1 recipe at 6k, two seeds
(equiv_ctrl6k, conc/ctrl_s2_6k).  (i) Streams: canary 0.511 → 0.424 /
0.433 / 0.440 (the families become in-distribution), pole stream
0.259 → 0.271 / 0.275 / 0.300.  (ii) Amortized pass: better on all 10
smooth cells (gapped 0.31 → 0.14, semicircle 0.43 → 0.33, box 0.48 → 0.37,
pseudogap 0.28 → 0.22), worse on 13–16 of 18 pole-closed cells.  (iii)
z-descent 3000: D24 better on 7 of 10 smooth cells, none worse; even on
pole-closed (8 better / 6 worse, mostly inside the control's seed spread);
the smooth gap to MaxEnt narrows (gapped 5.4× → 2.6×, gauss 4.6× → 1.8×,
pseudogap 3.8× → 2.3×) and spevae now beats MaxEnt on semicircle and box at
1e-3 (2.2–2.3×).  Latent factorial: 24 wins (best streams, smooth tally,
Holstein MAP 0.176); AU stays at cap in every arm.  **Promoted: D24.**
(iv) **THE gate is training-length dependent**: at 6k every model, control
included, reaches the Holstein manifold floor at target (0.171–0.176 vs
0.175; the descent stops at target) and MAP χ² 0.18–0.36, while M1-60k sits
at floor 3.83 / MAP 15.1.  Longer training narrows the decoder manifold
away from the real cell — the real-data auditor of the coverage protocol
firing.  [*Re-measured 2026-11-24*: wrong reading.  Every one of these floors is an
Adam descent that either stopped at the target or hit its 10,000-step
cap; converged, every checkpoint reaches the target.  Training length
changes how quickly first-order descent gets there, not the manifold.]  The 60k A/B therefore keeps snapshots every 5k steps for both
arms (`--keep-snapshots`, `scripts/step2/data_arm_60k.sh`).

**Step-2 data arm at 60k: the floor gate as banked falls 15× — an optimiser
effect, see the re-measurement below** (*measured*, 2026-11-24;
`scripts/step2/data_arm_60k.sh`, `scripts/step2/floor_curve.py`,
`out/step2/data_arm/floor_curve.{json,png}`).  D24 (M1 recipe + trusted
smooth bank, p_smooth 0.3, latent 24) vs the M1 recipe retrained with the
same code, both with snapshots every 5k steps.  Final (EMA) models on the
Holstein cell: manifold floor **0.21 (1.2× target) vs 3.22 (18×)**, MAP χ²
2.37 vs 29.8; the original M1-60k reference sits at 3.83 / 15.1.  Along
training (raw-weight snapshots) the control swings — floor 0.19–1.0, MAP
2–34 — while D24 stays at the target for most of the run (floor 0.17–0.54,
MAP 0.18–2.6); both rise in the last two snapshots (55k, 60k: the end of
the cosine schedule), which the EMA final averages out for D24 but not for
the control.  This refines the 6k reading: the control's real-data fit is
unstable along training, and the smooth data stabilises it, rather than
training length narrowing the manifold monotonically.
**Full scorecard** (`spevae eval --tier full`, `out/eval/step2_60k/`):
streams D24 / ctrl60 / M1 — pole 0.189 / 0.218 / 0.230 (D24 now best on
the pole stream too; at 6k it trailed), canary 0.315 / 0.466 / 0.504.
Versus ctrl60, D24 is better on all 10 smooth cells under every method;
on the 18 pole-closed cells zd3000 14 better / 4 worse, certified minimal
form 9/4, full rendering 9/6.  Versus MaxEnt: better on all 18 pole-closed
cells; smooth (full rendering, 1e-4) gapped 9.2× → 1.6× behind, gauss
3.6× → 2.0×, pseudogap 2.4× → 1.1×, and ahead 4× on semicircle and box.
Holstein: certified minimal χ²/target 0.78 (ctrl60 1.33, M1 0.88), full
rendering 0.32.  Gates: floor ✓ as banked (3.83 → 0.21; converged, both
reach the target — the gate that separates the arms is the MAP fit under
the model's own prior, below), canary ✓, smooth probes ✓,
pole-closed parity ✓ on the suite proxy (the template-menu "seven probes"
judges are not in the suite yet); the full-rendering spike gate is only
partly met — peak height 0.42 → 0.31 and roughness Σ|A''| 1.44 → 1.20 vs
M1, but equal to ctrl60 (1.20) and still a comb over the 2 < |ω| < 5
bands (`holstein_full_renderings.png`), consistent with the v1 labels'
own wiggles around smooth shapes; w-sensitivity not measured.  **D24-60k
is the step-2 checkpoint.**

**Conditional flow-matching posterior, first screen** (*measured*,
2026-11-23; `models/flow.py`, `scripts/flow/{train_flow,judge_flow}.py`,
`out/flow/judge_F6k/`).  Pole tokens (12 slots × (asinh ε/π, log(γ−γ_min),
log a, atanh d/D) + a presence channel; empty slots at the mean with
presence −1 — a first version marking them by log a = −14 put them ~8σ out
and let them dominate the loss, 27 vs ~1), Hungarian-paired flow matching,
a 4-layer transformer over the tokens cross-attending to the VAE's own
encoder outputs; 1.0M parameters, 6k steps, the exact training
distribution of the VAE control (M1 recipe, `equiv_ctrl6k`, 0.82M).
Judged on 256 held-out prior draws × 100 samples:
(i) **calibration**: SBC max KS 0.197 vs the VAE's 0.290 (5% critical
0.085) — neither calibrated; the flow's rank histograms hump (too wide),
the VAE's are U-shaped (overconfident).  90% interval coverage 0.97 vs
0.62; pointwise 90% band coverage 0.955 vs 0.522.
(ii) **data consistency fails, and the breakdown says why**: sample χ²
median 3.0e3 vs the truth's 1.0; by noise level, flow 8.1 / 724 / 7.1e4
and VAE 2.95 / 54 / 2.5e3 at 1e-2 / 1e-3 / 1e-4, while flow coverage stays
0.95 at every level (VAE 0.79 / 0.60 / 0.40).  The flow's posterior does
NOT contract with SNR — its coverage at low noise is the trivial kind (wide
enough to contain everything).  Both amortized models leave the data's
precision unused; the VAE pipeline recovers it by per-instance descent.
(iii) two-sided pairs 80/20 → 50/50: the flow's side-weight median tracks
the truth (0.78 / 0.62 / 0.55 / 0.51) with q05–q95 containing it; the 6k
M1 control also gets the side weights (the s1_metric mirror attractor is
not reproduced by this control), so the test does not discriminate here.
(iv) suite truths and Holstein: bands contain the truth (0.94–1.0, 0.73
semicircle/box) but sample χ² is 10²–10⁵ and Holstein χ²/target ~8e4.
Reading: the amortized flow is a broad, roughly covering posterior that
ignores data precision.  Note the published conditional-diffusion
continuation (arXiv:2608.13123) never checks χ² of its samples.  Next:
(a) 60k training (does amortized precision grow with budget?);
(b) sample-then-refine with the exact likelihood (each sample pulled to the
data with a proximity term), re-judged for calibration.

**Sample-then-refine fixes the flow's precision gap** (*measured*,
2026-11-23; `models/flow.refine_samples`, `scripts/flow/judge_str.py`,
`out/flow/str_F6k/`).  Each flow sample's token coordinates minimise
½χ²(exact likelihood) + ½λ‖u − u_k‖² (tether to its own start; presence
kept), all samples as one batch.  Tether scan (32 prior draws × 32 samples
per noise level, 300 steps): at noise 1e-2, λ = 1 is calibrated — χ² 0.94
(truth 1.01), 50/90% coverage 0.53/0.90, band 0.90, W₁ 0.56 → 0.25;
λ ≤ 300 barely matters at lower noise and per-sample noise redraws change
little — the optimiser budget binds (χ² 2.7 at 1e-3, 175 at 1e-4).
Budget scan (16 draws × 32): at 1e-3, 3000 steps with cosine lr
0.05 → 0.005 and error bars annealed 30× → 1 over the first 70% give χ²
0.92, coverage 0.49/0.89, band 0.89, W₁ 0.064 (raw 0.57) — calibrated; at
1e-4, a 100× anneal gives χ² 1.85 / 1.49 at 3000 / 5000 steps, coverage
0.64/0.92 and 0.60/0.90, band 0.91/0.90, W₁ 0.053/0.049 — near-calibrated,
50% intervals slightly wide.  Chosen schedule: 3000 steps, anneal from an
absolute 1e-2 (`anneal_from`), λ = 1, no redraw.
**Final judge of that schedule** (`out/flow/str_F6k/final.json`): on 96
held-out draws of the FULL training distribution (noise 3e-5–3e-2, all grid
spacings and correlation lengths) [*corrected 2026-11-24*: the generator
draws one noise model and one grid per batch, so these 96 draws are three
settings — noise 1.8e-2, 1.0e-4 and 2.1e-2; and the 50% / 90% coverages
quoted in this and the next two paragraphs are to be read against 0.47 /
0.85, not 0.50 / 0.90 — see the re-measurement below], 32 samples each — refined: SBC max KS
0.156 (5% critical 0.139), coverage 0.44/0.78, band 0.83, χ² 0.94, W₁ 0.46;
raw flow: KS 0.250, 0.67/0.95, band 0.92, χ² 2.67, W₁ 0.67; VAE: KS 0.250,
0.39/0.69, band 0.57, χ² 1.66, W₁ 0.64.  Best of the three on calibration
and accuracy, but over the full noise/grid range the 90% intervals cover
78% — mildly overconfident there, where the fixed-grid per-noise scans were
calibrated; the per-noise/per-grid breakdown is the open diagnostic.  On
the 14 frozen truths (outside the training prior for the 5 smooth ones):
posterior-mean W₁ 0.020–0.091 at 1e-4 and 0.041–0.096 at 1e-3 (VAE one-pass
0.05–0.82), sample χ² 0.85–1.07 (pseudogap 1.6 at 1e-4), band coverage
0.90–1.00 at 1e-4 except pseudogap 0.74, 0.68–0.96 at 1e-3.  **Holstein**:
200 refined samples fit the real cell at χ²/target median 1.05 (q05 0.41,
q95 6.0; raw flow ~8e4).  The bands are tight on the gap edge and the
±1 peaks and wide over the 2 < |ω| < 5 bands, where the median sits between
the shipped SmoQyDEAC humps and the VAE pipeline's sharp replicas — the
data-versus-prior split made visible.
**At 60k the refined flow is MORE overconfident, and the cause is
measured** (`out/flow/str_F60k/`, `scripts/flow/diag_str.py`).  F_60k's raw
posterior is sharper than F_6k's (full-distribution χ² 2.67 → 1.21, W₁
0.67 → 0.49), but refined coverage drops 0.78 → 0.68.  Breakdown (16 draws
× 16 samples per cell, grids of 33 and 161 slices): converged refinement
without noise redraw collapses the spread along data-determined directions
— 90% coverage 0.54–0.58 at noise 3e-2 with χ² 0.88 (fitting the noise),
0.75–0.78 at 1e-3 — and a per-sample noise redraw (randomize-then-optimize)
restores it: 0.83–0.85 at 3e-2 (χ² 1.01), 0.86–0.88 at 1e-3.  It mattered
nothing at 300 steps because the samples never converged there.  Two open
items: at 3e-2 refining hurts the posterior mean (W₁ 0.97 → 1.39 on the
coarse grid; the raw flow is already consistent there), and at 3e-5 the
3000-step schedule does not converge (χ² 6–15, coverage 0.77–0.82).
Default changed to redraw on.
**With the redraw, the refined 60k flow is calibrated on the full training
distribution** (`out/flow/str_F60k_redraw/final.json`; 96 held-out draws ×
32 samples, noise 3e-5–3e-2, all grids): SBC max KS **0.125 < 0.139** (the
5% critical value, all 9 statistics), coverage 0.56 / 0.89, pointwise band
0.906, sample χ² 0.98, posterior-mean W₁ 0.43 — against raw flow KS 0.229
(0.56/0.94, W₁ 0.48) and the 60k VAE control KS 0.281 (0.36/0.64, band
0.55, W₁ 0.60).  Suite truths: band coverage median 0.96 at 1e-3 and 0.95
at 1e-4 (minimum 0.60/0.72), χ² 0.98/1.17.  Holstein: χ²/target median
1.64 (q05 0.94, q95 3.9) — redrawn samples each fit a different noise
realisation, so their χ² against the one measured mean sits above the
target by construction.  The first spevae output with tested, calibrated
error bars.  [*Superseded 2026-11-24*: on 1000 truths at 40 noise levels,
with a converged solver, the refined posterior is conservative rather than
calibrated — 50% intervals cover 60%, 90% cover 94% — and the Holstein
excess was the optimiser, not the redraw: see the re-measurement below.]
Caveats: one training seed, 96 draws; calibration is
in-distribution (real data cannot be checked this way); the 3e-5 and 3e-2
items above remain.  Summary figures (`scripts/flow/str_summary_fig.py`,
from the two final.json files, no reruns): `summary_full.png` (KS,
coverage, χ², W₁ and Holstein spread, redraw vs no-redraw vs raw vs VAE)
and `summary_suite.png` (per-truth W₁ vs the VAE's amortized output, band
coverage at both noise levels), both in `out/flow/str_F60k_redraw/`.

**Solver and calibration re-measurement** (*measured*, 2026-11-24;
`physics/lsq.py`, `models/flow.refine_samples`, `refine.{refine_latent,
z_descend}`, `scripts/flow/{stratified_sbc,lm_refine_test,holstein_redraw}.py`,
`scripts/step2/floor_lm.py`, `scripts/eval/temperature_transfer.py`;
artifacts `out/flow/review_20261124/`, `out/step2/floor_lm/`,
`out/eval/{lm_60k,temperature_transfer,acceptance_20261124}/`).  A review of
the two days above found two faults in how the flow was judged and one in
how every per-dataset fit since step 1 was optimised.  This entry
supersedes the readings it names; the bracketed notes elsewhere in this
file point here.
(i) **The coverage reference was biased.**  `judge_str.metrics` covers the
truth with `np.quantile` intervals of S samples; a perfectly calibrated
posterior scores (S−1)/(S+1) × nominal on that estimator — 0.47 / 0.85 at
S = 32, 0.44 / 0.81 at S = 16 (simulated).  Coverage is now read from
randomised PIT values (rank + U)/(S + 1), exactly uniform at any S.
(ii) **The "full distribution" test was three noise settings**, because
`PolePriorGenerator.batch` draws one noise model and one grid per batch.
`stratified_sbc.py` gives every noise level its own batch.
(iii) **First-order descent was not converging, anywhere.**  Both fits are
nonlinear least squares whose curvature spans ~σ⁻² between directions, so
Adam needs step counts that grow with it.  A batched Levenberg–Marquardt
with the exact pole Jacobian (closed form, checked against autograd to
1e-15) replaces it: `refine_samples(solver="lm")` is now the default,
`RefineConfig(solver="lm")` / `z_descend(solver="lm")` are available (Adam
stays the latent default until the scorecards are re-banked; the acceptance
check still passes 204 / 0 fail / 8 drift).  *Refinement*, identical
samples, redrawn targets and objective (16 truths × 16 samples per level):
at noise 3e-5 Adam (3000 steps) ends at median χ² 5.25 with 72% of samples
above 2, LM at 1.24 with 26%, posterior-mean W₁ 0.034 → 0.025; at 1e-4
1.18 → 1.05, W₁ 0.076 → 0.054; equal at 1e-3; 11–14 s against 115–220 s.
*Latent MAP*, `spevae eval --tier lm` against the banked `zd3000` rows (224
paired runs per model, same cases and draws): W₁ geometric-mean ratio 0.92
(D24), 0.86 (ctrl60), 0.90 (M1), better in 67–75% of runs, median χ²
1.34 vs 1.42 (D24), median runtime 1.9 s vs 7.1 s; at noise 1e-4 D24
improves on 11 of 14 cases (Lorentzian pair 0.065 → 0.042, Gaussian pair
0.056 → 0.036, semicircle 0.029 → 0.019, pseudogap 0.070 → 0.040), at
1e-3 the two agree to three digits.
(iv) **The Holstein "manifold floor" was never a floor**
(`out/step2/floor_lm/`).  Every banked floor is a 10,000-step Adam descent
that reports `converged = False`:

| checkpoint | banked floor (Adam 10k) | Adam 40k | LM, prior off | \|z\| there | LM MAP from the encoder mean | MAP objective, 5 starts |
|---|---|---|---|---|---|---|
| s1_metric | 23.96 (137×) | 19.4, falling | target, 662 it | 212 | 56× (stationary) | 1410–4574 |
| M1 | 3.83 (22×) | 0.57, falling | target, 81 it | 102 | 21× (budget) | 2783–3861 |
| ctrl60 | 3.22 (18×) | 0.176 at step 22,576 | target, 52 it | 92 | 47× (stationary) | 2336–3315 |
| D24 | 0.21 (1.2×) | 0.175 at step 13,506 | target, 44 it | 59 | 1.00× (at target) | 511–3163 |

Every manifold — step 1's included — contains a noise-level fit of the
cell, at a latent norm 12–43× that of a typical prior draw (4.9).  The
step-1 "boundary proof" and the "gate falls 6.3× / 15×" headlines measured
how fast first-order descent travels there.  What does separate the arms is
the fit under the model's own prior: from the encoder mean D24's MAP
reaches the noise target while ctrl60 and M1 stop 47× and 21× above it —
but the MAP landscape has several stationary points (D24 lands at 1.0–11.7×
target over five starts, ctrl60 at 7.8–47×, M1 at 21–32×), so a single MAP
number is a start-dependent quantity; quote the range.  `spevae eval` now
records both protocols and how each fit ended (docs/EVAL.md).
(v) **Calibration on 1000 truths** (40 noise levels 3e-5–3e-2 × 25 truths
× 32 samples, grids and correlations cycled; `stratified_lm.{json,png}`).
Refine + redraw with LM: 50% / 90% intervals cover **0.596 / 0.943**
(±0.007 / 0.003), max KS 0.100 against a critical 0.052 — *conservative,
not calibrated*: error bars about 15–25% too wide, by noise 0.67 / 0.97 at
4e-5 falling to 0.55–0.58 / 0.91–0.95 above 6e-4.  Sample χ² median 0.99
(raw flow 178), 5.2% of samples above 2.  The data-fit rank (the truth's
χ² ranked among the samples'; uniform if calibrated, mean 0.5) is 0.53
overall — 0.28 at 4e-5 (some samples still fit worse than the truth),
0.55–0.63 above 2.5e-4 (samples fit slightly better than the truth) —
against 0.02 for the raw flow at every level: the raw flow never fits as
well as the truth, including at 2e-2 where its median χ² looks acceptable.
Posterior-mean W₁ relative to the raw flow: 0.20 at 4e-5, 0.44 at 6e-4,
0.93 at 3.6e-3, 1.02 at 8.7e-3, **1.20 at 2.1e-2** — refinement hurts at
the top of the noise range.  Two additions measured on the same draws:
keeping a raw sample that already fits the data within noise (*gated*; 12%
of samples at 8.7e-3, 29% at 2.1e-2) brings those ratios to 0.98 and 1.05
at unchanged coverage (0.605 / 0.947); rejecting refined samples whose χ²
is implausible for a posterior draw (*accept*; 72% kept at 4e-5, 89% at
1e-4, ≥ 93% above) lowers W₁ at the low-noise end by 15–25% (0.039 → 0.029
at 4e-5) and gives 0.585 / 0.939.  The excess width is uniform over the
seven window-weight statistics (50% coverage 0.63–0.72 at noise ≤ 1.2e-4;
the peak position is near nominal at 0.54) and is the raw flow's
(0.73 / 0.97): the flow does not contract with the data's precision,
the likelihood repairs the directions it determines, and the rest keep the
flow's spread.
(vi) **The pole count is the refinement's blind spot.**  It is fixed by the
flow sample; P(sample count = true count) is 0.12–0.29 at every noise
level and does not sharpen as the noise falls.  After LM the samples left
above χ² = 2 are mostly those with fewer poles than the truth (47% of them
against 14% of the rest at 3e-5; 22% against 2% at 1e-4).
(vii) **Holstein, converged** (`holstein_redraw_lm.{json,png}`, 200
samples): with LM the refined samples fit the cell at χ²/target 1.21
[0.69, 1.98] under the whitener redraw and 0.94 [0.56, 1.63] when each
sample's target is a bootstrap of the bins, with 5% and 2.5% above 2×
target — against 1.64 [0.94, 3.87] and 1.46 [0.85, 3.89] with 26–30% above
2× under Adam.  The excess the previous entry attributed to the redraw
"by construction" was the optimiser.  Without a redraw LM fits the mean at
0.23× target: noise fitting, so the redraw is not optional.  1.5% of
samples cannot reach the data; they carry a median of 2 poles against 12
(the slot cap) for the rest.  The whitener overstates the bins' own scatter
by more than 10× in 57 of 80 whitened directions (it costs at most 1–2
resolvable kernel modes); a bootstrap redraw is the consistent choice on
real cells (bands 4% narrower).
(viii) **Tried and not adopted.**  Data fission (the flow conditioned on a
deliberately noisier copy, the likelihood on the independent remainder):
no better anywhere, worse W₁ at low noise.  One global quantile
recalibration map fitted on even noise levels and tested on odd ones:
coverage 0.52 / 0.91 on the held-out half but max KS still 0.097 against
0.073 — the excess is not a single rescaling.
(ix) **Temperature** (`out/eval/temperature_transfer/`): five suite truths
rebuilt at β = 4, 8, 16 (Δτ = 0.125, noise 1e-3, 4 draws).  The β = 8
checkpoints do not degrade away from 8: geometric-mean W₁ 0.050 / 0.060 /
0.065 (D24 + LM MAP) and 0.062 / 0.059 / 0.046 (refined flow mean) against
MaxEnt 0.144 / 0.137 / 0.146.  The one case MaxEnt wins at every β is the
Gaussian pair (0.048–0.068 against 0.074–0.170 and 0.079–0.087).
**Judging rules from here on**: coverage from randomised PIT; one batch
per noise level; the data-fit rank in every calibration judge; a floor or
MAP is quoted only with its `converged` / `stationary` flag; the refined
flow is compared with the VAE pipeline at equal test-time compute, not
with the amortized pass.  **Open**: the flow's excess width (it needs to
use the data's precision — noise-scaled input features are the first
candidate); pole births and deaths during refinement; the Gaussian / gapped
/ pseudogap gap to MaxEnt (D24 + LM MAP is 1.4–3× behind at both noise
levels, with the smooth truths in-distribution for D24); an out-of-prior flag for real cells; one
training seed everywhere; no real QMC cell with a known spectrum.

**Step 3 opened: physics-sourced truths, a temperature grid, the posterior
report** (2026-11-24; user decisions of that day: fermionic G(τ) only for
the first reliable release; every model with an exactly known spectrum goes
into the training set; the tool reports one posterior band under a stated
prior, with its evidence).
(i) **Exactly solvable models** (`data/exact_models.py`, 29 tests, each
formula checked against an independent route — brute-force Brillouin-zone
sums, the elliptic-integral closed form, the textbook atomic limit, the
independent-boson Poisson ladder).  *Continuum*: lattice Green's functions
G(z) of the 1D chain, square (with t′), triangular and honeycomb lattices,
the Bethe and d = ∞ hypercubic lattices, the dimerised (SSH) chain,
mean-field BCS on any symmetric band, the resonant level in a band, and the
Falicov–Kimball model on the Bethe lattice (exact DMFT, gap at U = 2t).  A
truth is the model with a lifetime η ≥ the resolution floor, A(ω) =
−Im G(ω + μ + iη)/π, whose Stieltjes transform is G(z + μ + iη) exactly —
no singular integral is taken, and the label-bank protocol applies
unchanged.  *Discrete*: the Hubbard atom, Hubbard clusters of 2–4 sites, an
Anderson impurity with 1–3 bath levels and the Holstein–Hubbard atom, by
exact diagonalisation (finite-T Lehmann sums); with a lifetime these ARE
pole sets, so they enter the pole prior with exact labels and no fit, and
unbroadened they give delta-spectrum test data with G(τ) in closed form.
(ii) **Labels**: the step-2 extraction on 30 lattice truths (10 families ×
β = 1, 8, 20) gives 5–12 poles with data gap 1e-10–1.2e-5, all positive,
2–7 s each.
(iii) **Known-answer baseline, before any of this is trained on**
(`scripts/eval/exact_models_judge.py`, `out/eval/exact_models/`; β = 8,
noise 1e-3, 32 truths, the current F_60k / D24 checkpoints and MaxEnt on
the same draw, W₁ between cumulative weights on a 6e4-point grid).
Lattice truths (20): lowest W₁ — refined flow 11, VAE latent MAP 6,
MaxEnt 3; geometric means 0.123 / 0.181 / 0.229; the flow's samples fit
(χ² 0.96) where the VAE's MAP often does not (median 1.52, up to 10).
Exactly diagonalised truths (12): flow 5, VAE 1, MaxEnt 6; geometric means
0.246 / 0.386 / 0.280.  These errors are 2–5× the suite's at the same noise
(0.05–0.17): band edges, van Hove peaks, coherence peaks and multiplets are
outside every prior trained so far, and on finite-cluster spectra the
pipeline is no better than MaxEnt.  This is the number the step-3 training
has to move.
(iv) **Prior generator**: `beta_choices` (a β grid — bank labels are
β-dependent), `p_exact` (exactly diagonalised truths drawn on the fly,
3 ms each), precomputed exact spectra of the bank truths on the quadrature
nodes (`<bank>.nodes.pt`; lattice sums at β = 20 cost seconds on the real
axis).  Defaults leave the step-2 stream bit-identical.
(v) **Bank** (`scripts/step3/build_physics_bank.py`, `out/step3/bank/`):
5 smooth + 10 lattice families × β ∈ {1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20} ×
64 truths = 9600 labels in 81 min (12 workers); **9417 trusted** (data gap
< 3e-5 and dip within tolerance): 573–640 of 640 per family (lowest:
gapped pair 573, Gaussian mixture 612, SSH chain 618), 913–955 of 960 per
β (lowest at β = 1 and 20); median data gap 2e-8–3e-7, median 7–11 poles.
(vi) **The posterior report** (`posterior.continue_posterior`): refine +
redraw (a bootstrap of the bins on real cells), raw samples that already
fit kept as drawn, implausible misfits rejected; the band; the prior stated
in words; per-window spectral weight next to the **prior-free bracket** —
the range any non-negative normalised spectrum can put in the window with
every whitened residual within 3.5σ (two linear programs, nothing learned);
warnings for a low accept rate, the pole-slot cap and out-of-range inputs.
Measured: on the in-prior Lorentzian pair at 1e-3 the posterior spans a
median 15% of the bracket and is correct; on the Holstein cell 15–20%, 88%
of samples accepted, χ²/target 0.96 [0.55, 1.34].  Hard-edged window
weights are never well determined by the kernel, so the share is reported
as a number and is not a warning.
(vii) **Ranges of the first release** (fermionic G(τ)): β 1–20 on the grid
above; Δτ from {0.02, 0.05, 0.1, 0.125, 0.2, 0.25, 0.5} with 9–401 slices
(finer grids are thinned at inference: beyond ~400 slices neighbouring
points add no resolvable information); relative error bars 1e-5–3e-2.
(viii) **Pole births** (`models/flow.add_pole`, `scripts/flow/birth_test.py`,
`out/flow/review_20261124/birth_test.{json,png}`): a sample still above the
accept threshold after least squares gets one more pole — light, at the
position and width of a pole borrowed from a sample that fits the same
data — and a second fit from where the first ended, up to two rounds, kept
only when it fits better.  Same flow samples and redrawn targets, 16 truths
× 16 samples per level: the share of samples that fit rises 0.68 → 0.88 at
noise 3e-5, 0.85 → 0.95 at 1e-4, 0.96 → 1.00 at 3e-4, 0.95 → 1.00 at 1e-3;
W₁ of the accepted ensemble 0.0185 → 0.0203, 0.048 → 0.040, 0.105 → 0.098,
0.070 → 0.070; coverage unchanged.  It repairs the "cannot fit" samples; it
does not narrow the bands.  On by default in `continue_posterior`
(`max_births=2`).
(ix) **First real DQMC data with a known answer**
(`scripts/dqmc/{hubbard_dimer.jl,judge_dimer.py}`, `out/dqmc/`; SmoQyDQMC
2.0.11 in its own Julia project).  Half-filled Hubbard dimer, U = 4t,
β = 8, Δτ = 0.05, symmetric Trotter, 400 bins of 50 measurements; exact
spectrum by diagonalisation (four lines: ±1.83 with weight 0.42 each,
±3.83 with 0.08 each).  The measured G(τ) agrees with the exact one at
the level of the error bars (relative error bar 1.1e-3, rms z 0.91, max
2.5; χ² of the exact G against the data 1.34× target) — no Trotter bias
visible at this precision.  Continuation of the measured data, W₁ to the
exact cumulative weight: refined flow posterior **0.328**, VAE latent MAP
0.405, MaxEnt 0.777.  All 200 flow samples fit (χ²/target 0.61, none
needed a birth); the two main lines are placed correctly, the ±3.83
satellites are not resolved by any method (the flow's band has a shoulder
there); the samples use all 12 pole slots, which the report flags.  One
model, one parameter point: a first data point, not a validation.
(x) **Simulator feedback in the flow — judged in (xv): no gain, not adopted**
(`FlowPosterior(feedback=True)`, `train_flow.py --feedback`; cf. Holzschuh
& Thuerey, arXiv:2410.22573).  The measured defect it targets: the raw
flow's samples never fit the data as well as the truth does (data-fit rank
0.02 at every noise level), so after refinement the directions the data do
not pin keep the raw flow's excess spread (coverage 0.60 / 0.94).  With
feedback every velocity evaluation is two passes: the first gives an
estimate of the clean sample; the exact likelihood is evaluated there — one
damped Gauss–Newton step in token coordinates and log χ², float64 on the
CPU — and the second pass receives both.  Zero-initialised, trained with
the plain regression on both passes.  Checked: the step vanishes at the
truth (|step| < 5e-2 prior s.d. on noise-free data) and not away from it.
A/B at 6k steps on the step-3 recipe (F3_6k vs F3fb_6k), read by the
calibration judge below and the known-answer judge.
(xi) **Calibration judge for the reported posterior**
(`scripts/step3/calibration.py`): 12 noise levels 1e-5–3e-2 with β cycling
over {1, 2, 4, 8, 12, 20} and the grid spacing over seven values, 24
held-out truths per level from the step-3 prior (all three sources), each
through `continue_posterior` as a user's data would be; PIT coverage of
the spectral weight in the report's seven windows, the data-fit rank, and
the window-weight error, per noise level and per truth source.
(xii) **First step-3 screen: the flow learns the new truths, the refinement
erases the gain** (*measured*, 2026-11-24; `out/step3/F3_6k.pt`, 6000 steps,
18 min on MPS, p_smooth 0.35 / p_exact 0.15 / pole motifs 0.5, β grid,
noise 1e-5–3e-2; known-answer judge at β = 8, noise 1e-3, same 32 truths
and draws; W₁ geometric means of the flow posterior mean).

| flow | lattice, raw | lattice, refined | exact-diag., raw | exact-diag., refined |
|---|---|---|---|---|
| F_60k (step-2 recipe, 60k steps) | 0.249 | 0.123 | 0.500 | 0.246 |
| F_6k (step-2 recipe, 6k steps) | 0.567 | 0.132 | 0.742 | 0.257 |
| F3_6k (step-3 recipe, 6k steps) | 0.403 | 0.140 | 0.658 | 0.241 |

(VAE latent MAP 0.181 / 0.386, MaxEnt 0.229 / 0.280.)  At equal training
length the step-3 recipe makes the RAW flow clearly better on these spectra
(0.567 → 0.403, 0.742 → 0.658), and length matters as much (0.567 → 0.249
from 6k to 60k) — but after refinement all three flows land on the same
error.  The refined answer is set by the least-squares step from a weakly
tethered start, not by what the flow knows: better priors do not reach the
output through this pipeline.  This is the same defect as the excess band
width in (v) of the re-measurement, seen from the accuracy side, and it
makes the feedback arm (x) the decisive one: a flow whose own samples fit
the data needs no refinement, so what it learned survives.  Until that arm
is judged, longer step-3 training would improve only the raw column.
(xiii) **Tether strength of the refinement: 10, not 1** (*measured*,
2026-11-24; `scripts/step3/tether_scan.py` → `out/step3/tether_scan.{json,png}`;
the judges take the tether as an argument and tag their outputs `_lam<t>`).
The refinement minimises ½χ² + ½λ‖u − u_k‖² from each flow sample u_k; λ = 1
was chosen when the fits did not converge.  Known answers (β = 8, noise 1e-3,
W₁ geometric mean, lattice / exactly diagonalised):

| flow | λ = 1 | λ = 10 | λ = 100 |
|---|---|---|---|
| F_6k (step-2 recipe) | 0.132 / 0.257 | 0.096 / 0.267 | 0.087 / 0.287 |
| F3_6k (step-3 recipe) | 0.140 / 0.241 | 0.095 / 0.241 | 0.080 / 0.262 |
| F_60k (step-2 recipe) | 0.123 / 0.246 | 0.088 / 0.257 | — |

Held-out step-3 prior (288 truths, F3_6k, reported posterior):

| λ | coverage 50% / 90% | data-fit rank | window-weight error | accepted |
|---|---|---|---|---|
| 1 | 0.630 / 0.957 | 0.50 | 0.0562 | 0.96 |
| 10 | 0.612 / 0.957 | 0.47 | 0.0443 | 0.96 |
| 100 | 0.629 / 0.954 | 0.29 | 0.0459 | 0.94 |

Real data: DQMC dimer W₁ 0.328 / 0.318 / 0.374 (F_60k) and 0.355 / 0.336 /
0.362 (F3_6k) at λ = 1 / 10 / 100; Holstein cell χ²/target 0.97 → 0.99,
178 of 200 accepted, band width unchanged (0.165 → 0.160).
Reading: λ = 10 lowers the lattice error by a third and the in-prior
window-weight error by a fifth with the misfit still calibrated; λ = 100
stops the samples short of the data (rank 0.29) — rejected.  At λ = 10 the
flow has the lowest W₁ on 16 of 20 lattice and 9 of 12 exactly diagonalised
truths (F3_6k) and is behind MaxEnt on one family, the Holstein atom (0.232
against 0.222).  `continue_posterior(lam=10)` is the default.
Two things λ does NOT do.  (a) It does not make the answer depend on the
flow: three flows whose raw errors differ by 2.3× land within 9% of each
other at λ = 10 (0.088–0.096).  The gain is the fit being held in place,
not the prior showing through; (xii) stands.  (b) It does not move the
coverage: 0.61–0.63 / 0.96 at every λ.  The excess width is therefore not
made by the refinement.  The raw flow is already too wide (0.69 / 0.97) and
the refinement inherits it in every direction the data leave open.

(xiv) **Sampling temperature — tested, not adopted** (*measured*,
2026-11-24; `FlowPosterior.base_scale`, `scripts/step3/
compare_calibrations.py` → `out/step3/temperature_scan.{json,png}`).  Base
noise scaled by T at sampling, F3_6k, λ = 10, same 288 truths: reported
posterior 0.612 / 0.957 (T = 1), 0.584 / 0.947 (0.9), 0.567 / 0.931 (0.8);
window-weight error 0.0443 → 0.0445 → 0.0467; KS 0.130 → 0.115 → 0.110
(critical 0.094).  It narrows the bands on average but not evenly: at
T = 0.8 the β = 1, noise 8e-4 level reads 0.42 / 0.86 while β = 8, noise
9e-5 still reads 0.67 / 0.95.  One scalar cannot meet a per-band gate, and
an error bar that is too narrow somewhere is worse for a tool than one that
is too wide everywhere.  The attribute stays (default 1) as a per-checkpoint
constant should a better-trained flow make it uniform.
Standing statement for the report until that changes: the bands are
conservative — the nominal 50% band holds the truth about 61% of the time,
the 90% band about 96%.

(xv) **Simulator feedback — judged, not adopted** (*measured*, 2026-11-24;
F3fb_6k against F3_6k, same recipe and length; `out/step3/feedback_ab.{json,png}`).

| | raw: coverage, fit rank, weight error | λ = 1: coverage, fit rank, weight error | λ = 10: coverage, fit rank, weight error | known answers, λ = 1 / 10 |
|---|---|---|---|---|
| plain | 0.693 / 0.967, 0.01, 0.186 | 0.630 / 0.957, 0.50, 0.0562 | 0.612 / 0.957, 0.47, 0.0443 | 0.140 / 0.241, 0.095 / 0.241 |
| feedback | 0.697 / 0.972, 0.02, 0.150 | 0.621 / 0.955, 0.50, 0.0512 | 0.601 / 0.951, 0.47, 0.0451 | 0.140 / 0.244, 0.095 / 0.241 |

The feedback makes the raw samples 19% more accurate and leaves everything
that is reported unchanged: the raw samples still do not fit the data (rank
0.02), the raw coverage is the same, and after the fit the two models are
indistinguishable.  It costs 1.5× the training time (27 against 18 minutes
for 6k steps) and a float64 CPU solve per velocity evaluation at sampling.
A 6k-step screen cannot rule out that a long run would learn to use it, but
nothing here pays for one.  The code path stays (`--feedback`), off.
Why it probably cannot work in this form: the network's output is accurate
to about one part in ten of a prior standard deviation, while the data pin
some directions to one part in 10³–10⁵; a Gauss–Newton hint of that
precision is noise to the second pass.

(xvi) **Where the training matters: weak data** (*measured*, 2026-11-24;
known answers at β = 8, λ = 10, W₁ geometric mean, lattice / exactly
diagonalised; `out/eval/exact_models*/exact_models_b8_n0.01_lam10.json`).

| | noise 1e-3 | noise 1e-2 |
|---|---|---|
| F_6k (step-2 recipe, 6k) | 0.096 / 0.267 | 0.354 / 0.580 |
| F3_6k (step-3 recipe, 6k) | 0.095 / 0.241 | 0.336 / 0.558 |
| F_60k (step-2 recipe, 60k) | 0.088 / 0.257 | 0.319 / 0.557 |
| VAE latent MAP | 0.181 / 0.386 | 0.527 / 0.711 |
| MaxEnt | 0.229 / 0.280 | 0.578 / 0.702 |

At both noise levels the three flows are within 10% of one another and
40–60% below MaxEnt on the lattice spectra (17–18 of 20 lowest at 1e-2).
The step-3 recipe is 4–5% better than the step-2 recipe at equal length
and ten times the length is worth 10%: the full-length step-3 runs should
gain on both counts, modestly.  The large factor is the pipeline, not the
checkpoint.

(xvii) **Part of the excess width is a mixture over pole counts**
(*measured*, 2026-11-24; `scripts/step3/count_diagnostic.py` →
`out/step3/count_diag_F3_6k/`; `models/count.py`, `scripts/step3/
train_count.py`; F3_6k, λ = 10).  On 192 held-out pole-set truths (6 noise /
β cells, 96 samples each) the reported samples carry +1.35 poles on average
and their count is off by −6 … +8; 9% have the true count *whatever the
true count is* — the flow's count is the prior's.  Coverage inside groups
of samples, same truths: all 0.615 / 0.954, true count 0.528 / 0.919; within
one pole 0.565 / 0.925; two or more too many 0.614 / 0.959.  Samples with
too many poles are over-dispersed (rms z of the group mean 0.6–0.8), those
with too few are overconfident (1.2–2.0).  On the full step-3 prior (288
truths, all kinds) conditioning on the true count takes 0.60–0.65 / 0.94–0.97
to 0.57–0.60 / 0.91–0.96: a third to a half of the excess, not all of it.
The count is only weakly identifiable.  A classifier p(n | G, σ, β) on the
same encoder reaches cross-entropy 2.12 nats against a prior entropy of 2.36
after 6k steps (2.03 at 10k of a 30k run), gives the true count probability
0.15, and does not sharpen as the noise falls: several poles can make one
smooth feature.  Resampling the reported samples to its count posterior
(`continue_posterior(count_model=...)`, off by default; paired arm of the
calibration judge, 96 samples per truth) moves coverage 0.647 / 0.962 →
0.626 / 0.950 and the window-weight error 0.0467 → 0.0447 at an effective
sample size of about 70%.  Right direction, a sixth of the gap.
A classifier trained five times longer (30k steps: cross-entropy 1.89, true
count at probability 0.22) re-scored on the same samples
(`calibration.py recount`) gives 0.611 / 0.937 at an effective sample size
of 58%: more of the gap, still under half.  It does not pay once the bands
are calibrated (xviii): each arm given its own band table, the
count-matched 50% and 90% bands are 3–11% *wider* at equal held-out
coverage, and the window-weight error is unchanged (0.0451 against 0.0467).
Count matching stays in the code, off, and no classifier ships.

(xviii) **Band table: the reported bands are calibrated, the samples are
not** (*measured*, 2026-11-24; `posterior.band_levels`, `fit_band_table`,
`scripts/step3/fit_band_calibration.py`, sidecar `<checkpoint>.bands.json`
read by `load_flow`).  For every held-out truth and window the central level
of its sample ensemble at which the truth is just covered is |2·PIT − 1|;
the level to quote for a target coverage is the matching quantile over
held-out truths.  F3_6k, 288 truths (96 samples each), fitted on the even
truths of every cell and tested on the odd ones:

| | 50% band | 68% | 90% band | 95% | strata rms z |
|---|---|---|---|---|---|
| plain sample quantiles | 0.642 | 0.807 | 0.954 | 0.981 | 3.46 |
| pooled table (central 38% / 53% / 79% / 88% of the samples) | 0.508 ± 0.022 | 0.670 | 0.896 ± 0.014 | 0.930 | 0.80 |

By noise band 0.46–0.55 / 0.89–0.90, by β 0.42–0.55 / 0.85–0.94, all within
1.6 standard errors of nominal at 24–48 truths per stratum; β = 1 reads low
(0.42 ± 0.06 / 0.85 ± 0.04) and is the stratum to watch — there the plain
quantiles are already closest to calibrated.  Independent check on the exact
lattice spectra, pointwise in ω (a statistic the table was not fitted on):
the 90% band holds the exact A(ω) at 0.975 → 0.923 of frequencies at noise
1e-3 and 0.990 → 0.944 at 1e-2 (lowest single spectrum 0.80 / 0.84).
This is a calibration of the *bands*, by construction valid for data that
resemble the training prior and stratified only as far as it has been
tested; the samples in `PosteriorReport.poles` stay over-dispersed and say
so.  The table for a shipped checkpoint must come from the crossed design
(`calibration.py … reps`), with strata, and its held-out test goes in the
release notes.

(xix) **First full-length step-3 model, seed 1** (*measured*, 2026-11-24;
`out/step3/F3_60k_s1.pt`, `scripts/step3/run_full.sh`, 60k steps, 3.5 h on
MPS with the judges running beside it; λ = 10 throughout; the equal-length
control is F_60k, the step-2 recipe at β = 8 only).

*Calibration on the step-3 prior.*  Same 288 truths as the screen: raw
window-weight error 0.186 → 0.089, reported posterior 0.612 / 0.957 →
0.566 / 0.923, error 0.0443 → 0.0415.  Crossed design (`calibration.py … 6`
repetitions: every noise level at six β, 1728 truths): plain sample
quantiles 0.579 ± 0.006 / 0.935 ± 0.003, flat in β (0.570–0.591 /
0.927–0.946 for β = 1 … 20) and 0.61 / 0.93, 0.57 / 0.94, 0.59 / 0.95,
0.53 / 0.93 from the lowest to the highest noise band; data-fit rank 0.49
(0.43 in the lowest noise band, 0.44–0.53 by β); accepted 0.94–1.00.  The
raw flow is as over-dispersed as ever (0.723 / 0.973): length halves its
error and does not contract it.

*Band table* (`F3_60k_s1.pt.bands.json`, one row per band of relative error
bar, fitted on 2304 truths = 8 repetitions; central level of the samples
quoted for 50 / 68 / 90 / 95%):

| relative error bar | 50% | 68% | 90% | 95% |
|---|---|---|---|---|
| < 1e-4 | 0.412 | 0.587 | 0.874 | 0.946 |
| 1e-4 – 1e-3 | 0.429 | 0.598 | 0.841 | 0.908 |
| 1e-3 – 1e-2 | 0.425 | 0.592 | 0.841 | 0.913 |
| ≥ 1e-2 | 0.474 | 0.639 | 0.877 | 0.931 |

Tests that are not nominal by construction: (a) fitted on six repetitions
and scored on two fresh ones (576 truths): 0.510 ± 0.011 / 0.880 ± 0.008;
(b) leave-one-repetition-out over all eight: by β — which the table does
not stratify — 0.488–0.508 / 0.895–0.908, by repetition 0.479–0.519 /
0.883–0.916 (rms z 0.9 / 1.1).  A table with β strata as well is no better
and was not adopted.  (An early split read 0.43 ± 0.02 for the 50% band at
error bars ≥ 1e-2; the fresh repetitions read 0.54 ± 0.04 — noise.)
*The table is applied to window weights only.*  Pointwise, on the exact
lattice spectra, plain quantiles hold A(ω) at 0.952 / 0.970 / 0.985 of
frequencies (β = 8; noise 1e-4 / 1e-3 / 1e-2), 0.967 (β = 2) and 0.899
(β = 16) at the 90% level; the table's levels would give 0.937 / 0.935 /
0.957 / 0.938 and **0.832** at β = 16, where the lattice spectra carry
structure finer than twelve poles follow.  So `PosteriorReport.band` is
plain sample quantiles and `windows` is calibrated — this corrects (xviii),
which applied the table to both.

*Known answers* (`out/eval/exact_models_F3_60k_s1/`, W₁ geometric mean,
lattice / exactly diagonalised; flow lowest of 20 / of 12 in brackets):

| β, noise | F3_60k_s1 | F_60k | VAE | MaxEnt |
|---|---|---|---|---|
| 8, 1e-4 | 0.041 / 0.160 [13, 4] | 0.044 / 0.175 [12, 2] | 0.055 / 0.234 | 0.172 / **0.124** |
| 8, 1e-3 | 0.088 / 0.236 [16, 9] | 0.088 / 0.257 [15, 7] | 0.181 / 0.386 | 0.229 / 0.280 |
| 8, 1e-2 | 0.298 / 0.526 [18, 9] | 0.319 / 0.557 [17, 8] | 0.527 / 0.711 | 0.578 / 0.702 |
| 2, 1e-3 | 0.142 / 0.406 [11, 3] | 0.228 / 0.456 [7, 1] | 0.232 / 0.497 | 0.465 / **0.323** |
| 16, 1e-3 | 0.109 / 0.204 [17, 11] | 0.097 / 0.230 [18, 11] | 0.235 / 0.368 | 0.213 / 0.333 |

On every lattice family at every setting the flow is at or below MaxEnt
(worst ratio 0.98, honeycomb at β = 16).  The step-3 recipe is worth 0–7% at
β = 8, 38% at β = 2 (outside F_60k's training) and costs 12% on the lattice
spectra at β = 16.  **Line spectra are where it loses**: at β = 8, noise
1e-4 and at β = 2 MaxEnt is lower on the exactly diagonalised families
(Hubbard atom 0.092 against 0.042, Anderson cluster 0.169 against 0.088).
The exact lines given the narrowest width the pole form allows
(γ = 0.1π/β, positions and weights exact) cost W₁ 0.157 at β = 8 and 0.471
at β = 2, and the flow scores 0.160 and 0.406.  [*Corrected 2026-11-25,
(xx)*: that agreement is one of averages only.  Case by case the flow runs
from half to 2.5× the floor-only cost, so the comparison does not show that
the floor is the cause.  Tested directly in (xx): it is the binding limit
at β = 8, noise 1e-4, and it is not what loses at β = 2.]
The report now says when this is happening
(`posterior.weight_at_width_floor`, warning at ≥ 10%): the share of
spectral weight on poles pinned at the floor is 0.00 on every lattice
spectrum (100 runs over the five settings; warning on 0 of 20 through
`continue_posterior`) and 0.2–0.9 on isolated lines (warning on 7 of 12
line spectra: Hubbard atom 0.86, Anderson cluster 0.44–0.70, dimer 0.63;
silent on the trimer and other closely spaced lines, where the deficit to
MaxEnt is also small).  It fires on the real DQMC dimer (0.18) and not on
the Holstein cell.

*Suite* (`scripts/flow/suite_recon.py`, 14 truths, banked draw, W₁ of the
posterior mean, geometric mean): 0.0295 at noise 1e-4 and 0.0627 at 1e-3,
against 0.0291 / 0.0583 for F_60k at the same tether and 0.0296 / 0.0833
for the banked run (F_60k, λ = 1); VAE 0.043 / 0.102, MaxEnt 0.125 / 0.194.
Lowest on 8 and 10 of 14.  Better than F_60k on the smooth cases
(semicircle 0.078 → 0.064, box 0.086 → 0.059 at 1e-3), worse on the
pseudogap (0.055 → 0.118); MaxEnt keeps the Gaussian pair (0.012 against
0.034 at 1e-4) and the pseudogap — Lorentzian tails again.

*Real data.*  DQMC Hubbard dimer (`scripts/dqmc/judge_dimer.py`,
`out/dqmc/F3_60k_s1/`): W₁ to the exact spectrum 0.330 (VAE 0.405, MaxEnt
0.777), every sample accepted, no warnings.  The exact weight lies inside
the 90% window interval in 5 of 7 windows, with or without the table and on
three seeds: the upper satellite (ω = +3.83, weight 0.073) is put at ≤ 0.05
and its weight on the main line; the data-only bracket for that window is
0–0.235.  Holstein cell (`scripts/step3/holstein_report.py`,
`out/step3/holstein_F3_60k_s1/`): 177 of 200 accepted, χ²/target 0.99, W₁
to the SmoQyDEAC reference 0.149 (F_60k 0.145; MaxEnt 0.107, VAE 0.193 —
the reference is itself a continuation), but the samples use all 12 pole
slots (warning) and the band is 1.7× wider than F_60k's.

*Count matching* on the crossed design: 0.579 / 0.935 → 0.564 / 0.920; off,
as decided in (xvii).

(xx) **The width floor, tested directly: binding at β = 8 and small error
bars, not the cause at β = 2** (*measured*, 2026-11-25;
`scripts/step3/width_floor_test.py`, `width_floor_figure.py`,
`out/step3/width_floor/`).  The floor is γ ≥ 0.1π/β, a tenth of the first
Matsubara frequency, and the quadrature step is the floor / 3.5, so the
cost of every forward evaluation scales as 1/floor.  The test needs no
retraining: the same 64 proposals of F3_60k_s1 are refined against the same
data and the same noise redraws with the floor as trained and with the
floor divided by 4 (coordinates re-expressed, quadrature refined 4×), the
latter also at tether 1.  W₁ of the posterior mean, geometric mean, 12 line
spectra / 20 lattice spectra per setting; ratios with a paired 95% t
interval on the log-ratios:

| setting | as trained | floor / 4 | floor / 4, tether 1 | MaxEnt |
|---|---|---|---|---|
| lines, β = 8, noise 1e-4 | 0.160 | 0.141 | 0.131 | 0.124 |
| lines, β = 2, noise 1e-3 | 0.406 | 0.396 | 0.412 | 0.323 |
| lattice, β = 8, noise 1e-3 | 0.088 | 0.089 | — | 0.229 |
| lattice, β = 8, noise 1e-4 | 0.041 | 0.042 | — | 0.172 |

*β = 8, noise 1e-4.*  Lowered / as trained 0.88 [0.79, 0.99] (tether 1:
0.82 [0.70, 0.95]).  Against MaxEnt the flow goes from 1.29 [1.02, 1.62]
to 1.14 [0.96, 1.34] (tether 1: 1.05 [0.88, 1.26]): no longer separable
from MaxEnt on 12 truths, and not shown to be below it.  Eight of the 12
truths use the room (weight on poles below the trained floor); on those
0.141 → 0.117 → 0.103 against MaxEnt 0.098, and the four with dense lines
(trimer, 4-ring, Holstein atom) do not move.  The cleanest case, the
Hubbard atom: with the trained floor the samples cannot reach the data
(median χ² per point 7.0, 27% accepted, 88% of the weight on poles at the
floor, peak height that of a floor-width Lorentzian while MaxEnt's is 2.3×
taller); with the floor / 4, χ² 1.6, 45% accepted, W₁ 0.078 → 0.051
(MaxEnt 0.042).  The second atom still does not fit at floor / 4 (χ² 5.2):
a factor 4 is not always enough at this error bar.
*β = 2, noise 1e-3.*  Lowered / as trained 0.98 [0.94, 1.01], tether 1
1.02 [0.95, 1.09]; against MaxEnt 1.26 → 1.23.  Every sample already fits
(χ² 0.85, accepted 98%) and the posterior mean is two to three times
broader than floor-width lines at the exact positions
(`width_floor.png`, Hubbard trimer: W₁ 0.416, of which 0.062 from weight
more than 1 away from every line; floor-width lines 0.497, of which 0.256).
The loss there is the spread of line positions over the samples, not
their width.  Cause not established.
*Lattice spectra.*  Lowered / as trained 1.009 [1.004, 1.014] and 1.012
[1.006, 1.017]; no pole goes below the trained floor on any of the 40
runs.  The fit takes 3.3× as long (4.5 s → 15 s for 64 samples at β = 8).
Reading: a retrain with a lower floor is not justified by this — the gain
at β = 8 is available in the fit alone, and β = 2 would not be fixed by it.
Not done here: the lowered floor as an option of `refine_samples` /
`continue_posterior` (the test patches the coordinates and the quadrature
from outside), switched on when `weight_at_width_floor` fires, and coverage
of the window intervals on that path — both in (xxii); a true line
component (γ = 0 has a closed form) for the cases a factor 4 does not reach
— still open.

(xxi) **Three training seeds of the full-length model agree** (*measured*,
2026-11-25; `scripts/step3/run_full.sh "1 2 3" 60000`,
`scripts/step3/seed_summary.py`, `out/step3/seeds.{json,png}`).  Seeds 2
and 3 are the recipe of (xix) with only the training seed changed, each
judged by the same chain: calibration on 288 held-out truths of the prior
and the known-answer set at five settings.  W₁ of the posterior mean,
geometric mean, seeds 1 / 2 / 3:

| setting | lattice (n = 20) | MaxEnt | lines (n = 12) | MaxEnt |
|---|---|---|---|---|
| β = 8, noise 1e-4 | 0.041 / 0.042 / 0.041 | 0.172 | 0.160 / 0.151 / 0.160 | 0.124 |
| β = 8, noise 1e-3 | 0.088 / 0.088 / 0.088 | 0.229 | 0.236 / 0.233 / 0.234 | 0.280 |
| β = 8, noise 1e-2 | 0.298 / 0.292 / 0.294 | 0.578 | 0.526 / 0.527 / 0.523 | 0.702 |
| β = 2, noise 1e-3 | 0.142 / 0.143 / 0.142 | 0.465 | 0.406 / 0.409 / 0.411 | 0.323 |
| β = 16, noise 1e-3 | 0.109 / 0.111 / 0.110 | 0.213 | 0.204 / 0.212 / 0.206 | 0.333 |

Spread (max − min over the mean) 0–2% on lattice spectra, 1–6% on line
spectra.  The verdict against MaxEnt is the same for all three seeds at all
ten settings: ahead on eight, behind on line spectra at β = 8, noise 1e-4
and at β = 2.  Both the wins and the two losses belong to the method, not
to a training run.
*Held-out prior.*  Plain sample coverage 0.566 / 0.923, 0.564 / 0.928,
0.579 / 0.928 (nominal 0.50 / 0.90); data-fit rank 0.49, 0.48, 0.49; error
of the refined mean 0.0415, 0.0398, 0.0392 (unrefined 0.089, 0.087, 0.090).
*One band table for all seeds.*  The table fitted on seed 1's calibration
sets, applied unchanged to the 288 held-out truths of the other seeds
(`fit_band_calibration.py out/step3/cal_F3_60k_s<k> out/step3/F3_60k_s1.pt
--evaluate`, `band_table_test.json`): seed 2 0.477 ± 0.016 / 0.901 ± 0.010,
seed 3 0.495 ± 0.016 / 0.906 ± 0.009 at nominal 0.50 / 0.90 (plain
quantiles on the same truths 0.565 / 0.930 and 0.581 / 0.934).  Over the
ten strata (four noise bands, six β) the deviations scatter as sampling
noise would: rms z 0.83 and 0.88, largest |z| 1.70 and 1.88, against 2.4
and 2.5 (largest 7.2 and 4.6) for plain quantiles.  The largest stratum
gap is 0.06; with 24–120 truths per stratum the ±0.03 of gate (3) is not
resolved stratum by stratum on one seed's 288 truths — pooled it holds.
Reading: the over-coverage of the plain samples and its correction are
properties of the recipe, so one table serves any seed; seed 1 stays the
checkpoint carried forward, being the one with the full set of judges.

(xxii) **Narrow lines in the report: the lowered floor as the fit's default,
intervals that span both line shapes, and where the intervals still fail**
(*measured*, 2026-11-25; `spevae.models.flow.LoweredFloor`,
`continue_posterior(floor_factor="auto")`, `posterior.central_interval`,
`scripts/step3/floor_option_judge.py`, `out/step3/floor_option/`).
*What changed in the package.*  (a) `LoweredFloor` re-expresses the flow's
proposals in coordinates whose width floor is divided by a factor;
`refine_samples` sizes its quadrature from the floor of the coordinates it
is given.  (b) `floor_factor="auto"`, now the default of
`continue_posterior`: fit as trained; if a tenth or more of the accepted
samples' weight sits on poles within 1.25 trained floors (the warning of
69bf13f), fit the same proposals again, against the same noise redraws,
with the floor divided by 4, and report that fit.  (c) Whenever that share
is reached, the window intervals span both readings of those poles — as
fitted, and as sharp lines at their positions — and the band table is not
applied.  (d) `central_interval`: with n accepted samples the intervals are
taken at the central level × (n + 1)/(n − 1), the rank convention the band
table was fitted in; before, a stated 90% was the central 90% of n samples,
which holds a further draw 0.87 of the time at n = 64 and 0.84 at n = 30.
With `floor_factor=1` and no narrow lines the report is bit-identical to
before except for (d).
*Judge.*  The five settings of the known-answer judge; per setting the 20
exact lattice spectra and 24 exactly diagonalised line spectra (the judge's
12 and 12 further draws); the report with 64 samples, with the option and
with the floor as trained on the same data and seed; MaxEnt on the same
data.  W₁ of the posterior mean on the line spectra, geometric mean, ratios
with a paired 95% t interval:

| setting | flagged | as trained | option | MaxEnt | option / as trained | option / MaxEnt |
|---|---|---|---|---|---|---|
| β = 8, noise 1e-4 | 14 of 24 | 0.164 | 0.145 | 0.143 | 0.881 [0.813, 0.956] | 1.011 [0.880, 1.162] |
| β = 8, noise 1e-3 | 11 | 0.254 | 0.241 | 0.291 | 0.947 [0.900, 0.996] | 0.827 [0.735, 0.930] |
| β = 8, noise 1e-2 | 7 | 0.545 | 0.539 | 0.684 | 0.989 [0.977, 1.002] | 0.789 [0.696, 0.894] |
| β = 2, noise 1e-3 | 12 | 0.431 | 0.417 | 0.367 | 0.968 [0.941, 0.996] | 1.137 [1.017, 1.272] |
| β = 16, noise 1e-3 | 5 | 0.224 | 0.218 | 0.338 | 0.975 [0.953, 0.998] | 0.645 [0.560, 0.742] |

On the 49 flagged spectra alone the ratio is 0.885 [0.847, 0.925] (0.805 at
β = 8, noise 1e-4, where the share of samples that fit rises from 0.76 to
0.88); 35 improve by more than 2%, 2 get worse by more than 2% (both at
β = 2).  At β = 8, noise 1e-4 the gap to MaxEnt on line spectra closes
(1.15 [1.01, 1.30] → 1.01 [0.88, 1.16]); at β = 2 it does not (1.18 → 1.14
[1.02, 1.27]).  None of the 100 lattice spectra is flagged (the share at the
floor is 0.000 on every one), so their fits are untouched.  A flagged fit
takes four to five times as long (47 s against 12 s for 64 samples at β = 8
with the machine shared).
*Window intervals against the exact window weights*, all settings pooled,
every window counted (50% / 90% nominal):

| | spectra | samples as fitted | as the report states them |
|---|---|---|---|
| lattice | 100 | 0.581 / 0.929 | 0.501 ± 0.027 / 0.893 ± 0.014 |
| line spectra, flagged | 49 | 0.350 / 0.708 | 0.519 ± 0.040 / 0.860 ± 0.021 |
| line spectra, not flagged | 71 | 0.360 / 0.728 | 0.290 ± 0.030 / 0.638 ± 0.035 |

The lattice row is the band table carried from prior draws to exact lattice
models: nominal pooled, 0.81 ± 0.03 at β = 8, noise 1e-4 and 0.84 ± 0.03 at
β = 2 (misses of order 1e-3 in weight), 0.99 at noise 1e-2.
What fails on line spectra is the shape of a line at a window edge, not its
position or weight.  Grouping the windows of the line spectra by the
distance of the nearest exact line to a window edge, the central 50% / 90%
of the samples as fitted hold the exact weight 0.49 / 0.91 of the time at
0.5 or more, 0.30 / 0.77 at 0.25–0.5, 0.24 / 0.58 at 0.1–0.25 and
0.28 / 0.45 below 0.1: a Lorentzian of the floor's width leaks
atan(γ/d)/π of its weight across an edge at distance d (2.5% at β = 8,
10% at β = 2 for d = 0.5), a sharp line leaks nothing, and the data do not
say which it is.  Spanning both readings is what (c) does; on the flagged
spectra it takes the near-edge windows from 0.26 / 0.57 to 0.68 / 0.96
(0.1–0.25) and from 0.31 / 0.72 to 0.54 / 0.85 (0.25–0.5), at intervals
about 1.5 times as wide.
*Not fixed.*  The 71 line spectra that are not flagged — dense sets of lines,
which the flow draws with broad poles: the share at the floor is exactly
zero on 67 of them and below 0.08 on the rest, so no threshold would flag
them — keep intervals that
are too narrow, 0.29 / 0.64 pooled and 0.12 / 0.27 at β = 2, and there the
band table makes them narrower still.  Nothing in G(τ) distinguishes such a
spectrum from a continuous one; a user who knows the system is finite has to
say so, and the report has no switch for that yet.  The bracket allowed by
the data alone held the exact weight in all 1540 windows (840 line, 700
lattice): it is the statement to use for a spectrum that may be discrete.
*Held-out prior draws* (repetition 8 of `calibration.py`, 288 truths never
used for the band table; `out/step3/cal_F3_60k_s1_rep8_floor{1,auto}`,
`_report`).  29 are flagged (10%: 15 of 141 pole mixtures, 7 of 45
broadened line models, 7 of 102 bank spectra).  With and without the second
fit, on the same truths and seeds: table coverage 0.501 / 0.887 against
0.502 / 0.885, data-fit rank 0.514 / 0.513; on the flagged 29 the
window-weight error ratio is 1.01 ± 0.03.  The intervals exactly as the
report now states them: 0.520 ± 0.016 / 0.897 ± 0.011 overall; 0.497 / 0.887
on the 259 not flagged (band table); 0.724 / 0.985 on the 29 flagged (plain
quantiles, both readings — conservative, as intended for in-prior truths
whose poles really are Lorentzians); per noise band 0.50–0.53 / 0.88–0.92.
*Real DQMC dimer* (`out/dqmc/F3_60k_s1_floor_auto/`): flagged at 18%; W₁
0.330 → 0.327 (MaxEnt 0.777, VAE 0.405); the exact weight is inside the
stated 90% interval in 5 of 7 windows as before, the two misses smaller
(0.008 and 0.016 in weight, from 0.020 and 0.027).  The Holstein cell is not
flagged and its report is unchanged.
*A lower floor still* (`width_floor_test.py ... 16 lines`,
`out/step3/width_floor_16/`; the protocol of (xx), 12 line spectra per
setting).  Dividing the floor by 16 instead of 4 gains little: W₁ ratio
0.962 [0.925, 1.000] at β = 8, noise 1e-4 (0.141 → 0.136; MaxEnt 0.124) and
0.978 [0.947, 1.009] at β = 2 (0.396 → 0.388; MaxEnt 0.323), at three times
the fitting time of the factor 4 (54 s against 18 s and 4 s as trained, 64
samples at β = 8).  The factor stays at 4.  Width is therefore no longer
what limits the line spectra: making the poles sharper in the fit, down to
effectively a line, moves the error by 2–4%.
Reading: the option costs nothing where the spectrum is continuous and is
the default; flagged line spectra now come with intervals close to their
stated level; unflagged line spectra and line spectra at β = 2 remain
outside what this prior represents.  What is left is in the proposals —
where the flow puts lines and how many it draws — and the flow was trained
on no spectrum of sharp lines (its line models are broadened to at least
the floor).  A line token (γ = 0, closed-form kernel) with exact line
spectra in the prior is the candidate remedy; it is a retrain, and whether
it closes the gap at β = 2 is not established by anything measured here.

(xxiii) **Poles pinned at the trained floor start the second fit pinned at
the lowered one; and what pole labels of exact line data would be**
(*measured*, 2026-11-25; `LoweredFloor.recode(follow_within=...)`,
`posterior.FLOOR_FOLLOW`, `out/step3/floor_option/` against
`out/step3/floor_option_keep/`; `scripts/step3/line_label_probe.py`,
`out/step3/line_labels/`).
*Why.*  At β = 2 the positions and weights of the line groups are right
(on four spectra looked at sample by sample: weight of each half of the
spectrum within 0.02, its mean position within 0.1) and the loss is in the
tails: a pole
kept at the trained floor is a Lorentzian of half-width 0.157 there, 10% of
its weight more than 0.5 away, and the second fit of (xxii) left it that
wide because the data are indifferent.  A pole the flow pins at its floor
says "as narrow as I can draw"; in the second fit it now keeps its offset
above the floor instead of its width, so it starts pinned at the lowered
floor and the data widen it if they need to (`FLOOR_FOLLOW = False`
restores (xxii)).  Wider poles are untouched.
*Judge of (xxii) again*, same data and seeds; W₁ on the 24 line spectra per
setting, geometric mean:

| setting | as trained | (xxii) | now | MaxEnt | now / MaxEnt |
|---|---|---|---|---|---|
| β = 8, noise 1e-4 | 0.164 | 0.145 | 0.139 | 0.143 | 0.969 [0.846, 1.110] |
| β = 8, noise 1e-3 | 0.254 | 0.241 | 0.233 | 0.291 | 0.799 [0.705, 0.906] |
| β = 8, noise 1e-2 | 0.545 | 0.539 | 0.532 | 0.684 | 0.778 [0.687, 0.882] |
| β = 2, noise 1e-3 | 0.431 | 0.417 | 0.399 | 0.367 | 1.087 [0.971, 1.217] |
| β = 16, noise 1e-3 | 0.224 | 0.218 | 0.215 | 0.338 | 0.636 [0.551, 0.733] |

On the 49 flagged spectra: 0.930 [0.915, 0.944] of (xxii) (43 better by more
than 2%, 2 worse), 0.823 [0.784, 0.863] of the floor as trained (45 better,
3 worse).  At β = 2 the line spectra are no longer measurably behind MaxEnt.
The window intervals of the flagged spectra, every window counted: samples
as fitted 0.414 / 0.854 (0.350 / 0.708 in (xxii)); as the report states them
0.542 ± 0.041 / 0.915 ± 0.017 (0.519 / 0.860) — at their stated level.
Unflagged line spectra (0.290 / 0.638), lattice spectra (0.501 / 0.893, none
flagged) and the bracket (1540 of 1540 windows) are as in (xxii).  Held-out
prior draws (`cal_F3_60k_s1_rep8_follow`): stated intervals 0.523 / 0.897;
on the 29 flagged the window-weight error is 0.94 ± 0.07 of (xxii)'s and the
intervals are wider (0.76 / 0.99).  Real DQMC dimer: W₁ 0.325 (0.327), 5 of 7
windows, misses 0.008 and 0.012.
*Pole labels of exact line data.*  The flow is taught the exactly
diagonalised models as broadened lines — Lorentzians of the floor's width or
more, data computed from the broadened spectrum — while the lattice models
are taught as exact data with the canonical pole label as target.
`DiscreteSpectrum` now carries what the label protocol reads, and the probe
runs it on the judge's 12 line spectra per β (noise-free data, trained
floor), scoring the label like a continuation next to the flow and MaxEnt at
noise 1e-3:

| β | label | flow | MaxEnt | label reproduces G | there: label / flow / MaxEnt |
|---|---|---|---|---|---|
| 2 | 0.375 | 0.406 | 0.323 | 4 of 12 | 0.161 / 0.320 / 0.255 |
| 8 | 0.171 | 0.236 | 0.280 | 4 of 12 | 0.087 / 0.220 / 0.232 |
| 16 | 0.130 | 0.204 | 0.333 | 2 of 12 | 0.084 / 0.265 / 0.435 |

Where the label reproduces the exact G(τ) (dense sets of lines: the spectra
the report cannot flag) it is two to three times closer to the truth than
the flow's posterior mean; where the floor limits it (isolated sharp lines)
it is still closer at β = 16 and worse at β = 2.  The label is a ceiling —
it sees noise-free data — so this bounds what teaching line spectra the way
lattice spectra are taught could gain; it does not measure it.  That is a
change of the training data and a retrain, with no change of the model.

(xxiv) **Leave-families-out arm: 5% on lattice families never seen, nothing
measurable on line families** (*measured*, 2026-11-25;
`scripts/step3/run_leave_out.sh`, `leave_out_bank.py`,
`leave_out_summary.py`, `out/step3/F3_60k_lo.pt`,
`out/step3/leave_out.{json,png}`).  The recipe of (xix), seed 1, with five
lattice families removed from the label bank (tb_square, tb_honeycomb,
ssh_chain, bcs, falicov_kimball: 3157 of 9417 labels) and three line
families removed from the generator (hubbard_dimer, hubbard_ring4,
holstein_atom); judged on the known-answer set at the five settings.  Per
spectrum and setting, W₁ of the leave-out model over W₁ of the full model
(seed 1), geometric mean with a 95% t interval; the other two seeds of the
full model give the size of a ratio that means nothing:

| | n | leave-out / full | other seed / full | leave-out / MaxEnt | full / MaxEnt |
|---|---|---|---|---|---|
| lattice, never seen | 50 | 1.052 [1.028, 1.078] | 1.006 [0.997, 1.014] | 0.482 [0.390, 0.597] | 0.458 |
| lattice, trained on | 50 | 1.001 [0.979, 1.022] | 0.993 [0.981, 1.005] | 0.308 [0.260, 0.365] | 0.308 |
| lines, never seen | 30 | 1.014 [0.990, 1.039] | 0.981 [0.964, 0.998] | 0.901 [0.818, 0.992] | 0.888 |
| lines, trained on | 30 | 1.017 [0.997, 1.038] | 1.014 [1.000, 1.028] | 0.951 [0.784, 1.154] | 0.935 |

By family the largest are bcs 1.085 [1.011, 1.165], hubbard_dimer 1.073
[1.029, 1.120] and tb_honeycomb 1.055 [1.009, 1.103]; holstein_atom and
hubbard_ring4 are at 0.98 and 0.99.  The leave-out model is below MaxEnt on
40 of the 50 never-seen lattice spectra (the full model on 42).  The real
DQMC dimer through the leave-out model, which never saw a Hubbard dimer:
W₁ 0.331 (full model 0.325, MaxEnt 0.777, VAE 0.405), the exact weight
inside the stated interval in 5 of 7 windows as for the full model
(`out/dqmc/F3_60k_lo/`).
Reading: having seen a lattice family is worth about 5%, an order of
magnitude less than the margin over MaxEnt; the known-answer scores of the
full model are therefore not an artefact of training on the judged
families.  One arm, one seed, eight families withheld at once — not a
family-by-family leave-one-out.

**Gates for calling it shippable**, and where seed 1 stands (2026-11-25,
after (xxiv)):
(1) known-answer W₁ at or below MaxEnt on every family — *met on the ten
lattice families at five settings; on line spectra ahead of MaxEnt at
β = 8, noise 1e-3 and 1e-2 and at β = 16, level at β = 8, noise 1e-4
(0.97 [0.85, 1.11]) and at β = 2 (1.09 [0.97, 1.22]) once the fit lowers
the floor ((xxii), (xxiii))*; (2) a
leave-families-out arm — *done with eight families withheld at once: 5%
worse on the five lattice families never seen (1.05 [1.03, 1.08]; still
0.48 of MaxEnt's error), no measurable cost on the three line families, the
real DQMC dimer unchanged ((xxiv))*;
(3) coverage within ±0.03 of nominal per noise band and per β — *met on
draws of the prior by the intervals as the report states them (fresh 288
truths: 0.52 / 0.90, per noise band 0.50–0.53 / 0.88–0.92; earlier: by β
0.49–0.51 / 0.90–0.91, by repetition 0.48–0.52 / 0.88–0.92), and pooled on
100 exact lattice spectra (0.50 / 0.89), where two settings are low at the
90% level (0.81 at β = 8, noise 1e-4; 0.84 at β = 2).  On exact line
spectra: met where narrow lines are flagged (0.54 / 0.92, (xxiii)), not met
where they are not (0.29 / 0.64, (xxii)); the bracket from the data alone
holds in every window*;
(4) data-fit rank 0.45–0.55 — *0.49 pooled, 0.43 below
error bars of 1e-4*; (5) the suite no worse than the banked run — *met
(0.0295 / 0.0627 against 0.0296 / 0.0833); 7% behind F_60k at equal
tether at 1e-3*; (6) three training seeds — *met: 0–2% apart on lattice
spectra, 1–6% on line spectra, same verdict against MaxEnt at every
setting, seed 1's band table transfers ((xxi))*;
(7) a real DQMC run of a model with a known spectrum — *done: below MaxEnt
and the VAE, the upper satellite and its neighbour window outside their
intervals by 0.01 in weight*.

**Step-2 build (1): the smooth-family label bank — built, judged,
pinned** (*measured*, 2026-08-27).  `data/label_bank.py` +
`tests/test_label_bank.py` (10 tests) + `scripts/step2/
build_label_bank.py`; the bank is 1280 labels (256/family, seed 1,
β ∈ [2, 20], P_max = 12) with report, judge arrays and three figures in
`out/step2/`.  The canonical protocol is module constants, and every
clause was forced by a measurement: (i) exact Matsubara data from
closed-form Stieltjes transforms (Faddeeva w for Gaussian components;
Gauss-Legendre with the edge-absorbing u = D sinθ substitution for the
broadened semicircle/box; verified to 1e-15 against brute force — no
τ-grid, no spline transform, no noise); (ii) the fit lives in the
decoder's constrained output space (softmax a, centred-tanh b, width
floor) inside a hard parameter box — two ghost-pole failures are on
record: an unconstrained-sum fit parked 4% of the weight invisibly, and
a softmax'd-but-unbounded fit parked 1e-3 of weight at |ε| ~ 1e20,
invisible in the Matsubara residual (3e-7) but a 1e-3 error in G(0);
(iii) AAA supplies *seeds only* — its interpolation stopping rule says
nothing about the causal-pole-sum error (interpolation at 1e-12 with
the causal subset stuck at 1e-1 for Gaussian tails at β ≥ 8); each
rank rung 1…12 is an analytic-Jacobian trust-region fit
(finite-difference-verified); (iv) the **smoothness budget**: the pure
data objective returned spike combs — floor-width poles peaking 10×
above a smooth truth at residual 1e-7, caught by the rendered overlays
— so every polish carries a mollified-A block at the resolution scale
π/β whose weight descends a homotopy, and the label is the smoothest
fit whose PURE Matsubara residual meets max(1e-6, 1.5× the rank's own
floor).  A fixed weight fails both ways (0.5 alone cost 1e-4–8e-3 in
data fidelity; a pure re-polish afterwards re-sharpened the spikes),
and stopping the ladder at the first in-budget rung cost 2–3× in W₁;
(v) dips bounded, not banned: > 0.5% negative weight triggers a
re-polish under a tightened asymmetry cap — post-hoc b-shrinking cost
17× in residual; merges are fidelity-gated (a stacked pair at one
centre can BE the minimal representation).

**The label A-fidelity floor is the residue-phase bound, NOT
information content** (*measured*, 2026-09-01, forced by the user's
challenge against Zhang & Gull PRB 110, 035154 / arXiv:2504.01163).
The papers reconciled first: arXiv:2504.01163 fits poles to A(ω) GIVEN
on the real axis (1e5 real-frequency points — no continuation; 1e-12
with 9–19 poles), and PRB 110, 035154's own Fig. 2 splits by spectral
type — the near-exact recoveries are its *discrete* (pole-representable)
cases, matching our step-1 exact-recovery tests, while its *continuum*
case reaches err(A) ≈ 1e-3 (L1) at machine-precision ε with M = 11
free complex poles, β = 200, N_ω = 2001.  Then the decisive local
experiment: on OUR truths at β = 8 with only 64 Matsubara points,
UNCONSTRAINED Prony (free complex residues, no cap, no floor) reaches
machine data-residual (1e-12–1e-13) with 7–8 poles and L1(A) =
1e-3 (gauss pair) / 0.014 (semicircle) / 0.016 (box) — so the noiseless
data DOES pin A down, and the earlier "null-space" reading of the label
wiggles was wrong in emphasis.  What binds is the SPE residue
constraint set, isolated dial by dial: the width floor does NOT bind
(free pole depths ≥ 0.11 vs floor 0.041); rank does NOT bind (free fits
use ≤ 8 ≤ P_max); positivity barely binds (free weights all positive,
Σa = 1, dips ≤ 1e-3).  The binding constraint is **|b| ≤ a**: optimal
band-edge poles carry |b|/a ≈ 1.9–2.1, and Gaussian bulk wants
dispersive-dominant poles (b finite as a → 0) that the decoder's
b = a(t − ⟨t⟩ₐ) map cannot express at any cap.  Cap isolation on the
floor-broadened semicircle (t_cap 0.5 / 1.0 / 2.5, all else fixed):
L1 0.187 / 0.041 / 0.014 with residual 4.9e-7 / 1.1e-7 / 7.1e-14 and
negative weight 8.3e-4 / 1.6e-5 / 1.7e-5 — the cap alone recovers
free-fit quality for edge-type spectra WITH positivity essentially
intact.  Consequences: (i) the bank v1 labels stay as built — they are
capacity-matched to the CURRENT decoder, and a label the decoder cannot
emit teaches an unreachable target; (ii) the residue-phase bound is now
a measured expressiveness ceiling of the architecture itself (plausibly
part of the Holstein manifold-floor 24.0 story) and joins the deferred
arms as a candidate: raise the asymmetry cap jointly in the decoder
constraint layer, the pole prior, and the label protocol — one
variable, its own A/B, dips still bounded by the exact certificate at
the standing 0.5% tolerance.

**The candidate residue map, designed and measured** (*measured*,
2026-09-01, follow-up).  Broad sampling first killed the naive option:
free Prony sets are NOT safe to adopt — for Gaussian-bulk and
pseudogap truths the machine-precision representation uses poles with
large NEGATIVE spectral weight (a down to −5.9 behind cancellations),
brittle as targets and meaningless as OT mass; only the edge families
come out all-positive (their transforms are Markov-function-like).
The safe candidate keeps the mass doctrine and decouples only the
dispersive channel: a = softmax (positive, Σa = 1), γ ≥ floor, ε in
the box, and **b_p = d_p − (Σ_q d_q)·a_p with |d_p| ≤ D** — zero-sum
exact by construction, b no longer tied to a (today's map is
b = a(t − ⟨t⟩ₐ), |t| ≤ ½, which both caps |b|/a below 1 AND forces
b → 0 with a).  Measured reachable fidelity of that space at β = 8,
64 noiseless points, pure data objective (L1(A) vs truth; current
label / safe space / unsafe free): gauss pair 0.29 / **0.010–0.015** /
0.001; semicircle 0.055→(this probe) 0.187 / **0.014** / 0.014; box
0.087 / **0.016** / 0.016; pseudogap — / **0.004–0.015** / —; data
residuals 1e-8 to 9e-14, and certified negative weight ≤ 3e-4
everywhere, 20× below the dip tolerance, with NO per-pole surrogate.
D = 0.3 suffices for edge families; D = 0.8 buys the bulk families
(free-fit |b| ≤ 0.2 for edges; safe fits spend |b| ≤ 0.8 where the
unsafe optimum spent negative weight).  Still an arm, not an act: the
change touches the decoder constraint layer + pole prior + label
protocol together, and the open risk is behaviour under noise (the
dispersive channel adds freedom noise can excite) — exactly what the
standing judges measure.

**The map arm, screened on the step-1 world** (*measured*, 2026-09-01;
scope amendment by user directive: the residue-map arm runs BEFORE the
step-2 data arm, on the discrete step-1 prior, so the noise-excitation
question is answered where the channel adds zero needed expressiveness
and any degradation is unambiguous).  Implementation:
`residue_mode="dispersive"` in the decoder constraint layer
(D = 0.8 = `physics.poles.DISPERSIVE_SCALE`), matching d-draws in the
pole prior (`--prior-residue-mode`), unit-tested (reaches |b|/a > 2 and
dispersive-dominant poles; Σb = 0 exact; PH mirror pairwise; prior dips
certificate-bounded).  Three 6k screens, s1_metric recipe otherwise
(β = 8, data poles, width metric, no concentration, seed 1): ctrl
(step-1 map), M0 (dispersive decoder, step-1 prior — pure architecture),
M1 (dispersive decoder + widened prior).  Judged on common fixed probes
via z-descent with PAIRED noise realisations
(`out/step2/map_screen6k.*`): (i) **no noise excitation** — paired-
realisation consistency W₁ is equal or better for M0/M1 on every probe
at both noise levels (e.g. triplet 0.0087/0.0084/0.0081 at 1e-3), and
every reconstruction certifies exactly positive (negw = 0); (ii)
pole-closed probes neutral-to-better (fano W₁ 0.028 → 0.022/0.021;
triplet 0.051 → 0.059 M0 / 0.047 M1 at 1e-4); (iii) the smooth
out-of-vocabulary payoff is large: **semicircle W₁ 0.199 → 0.144 (M0)
→ 0.056 (M1)** at noise 1e-3, and at quiet noise χ² 58.2 → 9.2 → 3.0 —
M0's gain at IDENTICAL training data is the predicted free bonus:
z-descent exploiting the wider reachable manifold.  The ctrl semicircle
render is a spike comb peaking 3× above the truth; M1's is a mildly
rippled semicircle.  Screen verdict: M1 promoted to the judged 60k
tier against s1_metric (the banked 60k control with the same recipe).

**M1 at the judged 60k tier: THE gate falls 6.3x; one probe regression
on record** (*measured*, 2026-09-01, `out/step2/holstein_floor.json`,
`map_judge60k*.json`).  Headline: the Holstein manifold floor (z-refine,
anchor released, 10k steps) drops **23.89 → 3.77** (136x → 21.5x the
calibrated target) — the harness first reproduced the banked s1_metric
number (23.89 vs 23.96), so the drop is real and attributable to the
residue map + widened prior ALONE, with zero smooth training data.  A
substantial part of the step-1 "manifold floor 24" was the |b| <= a
bound, as the free-Prony diagnosis predicted; the remaining 3.77 → 0.56
gap (6.7x) is what the smooth-family data arm now owns.
[*Re-measured 2026-11-24*: both numbers are 10,000-step Adam descents that had not
converged.  With a least-squares solver the s1_metric and the M1 latent
both reach the noise target with the anchor released (662 and 81
iterations).  What the map arm changed is how fast, and how far from the
prior, the fit is found — not whether the manifold contains it.]  The anchored
MAP refine also improves 45.9 → 14.4.  Probes, with a step-budget
control (800 vs 3000 z-descent steps, which matters for M1's sharper
landscape): at quiet noise (1e-4) M1-60k is better or equal — fano W1
0.0139 vs 0.0263, semicircle 0.094 vs 0.216 (2.3x), triplet mildly
worse (0.061 vs 0.049); consistency and dips stay clean everywhere
(negw <= 2.9e-3).  BUT at noise 1e-3 the fano band genuinely regresses
(chi2 4.4 vs 1.0, W1 0.102 vs 0.028, budget-independent) — and
M1-6k did NOT show this (0.0215), so it is a training-length effect
under the widened prior, not the map per se.  Open item for the
promotion decision: seed/length replicates (is 60k past a sweet spot
for this prior mix?), an eval-checkpointed rerun, or proceed to the
data arm (whose labels and curriculum change the mix anyway) with the
fano@1e-3 number tracked as a regression canary.

**Bank v2 in the dispersive map: better labels the frozen loss cannot
price** (*measured*, 2026-11-23, β = 8, 6 truths per family; the label
protocol gained `residue_mode="dispersive"` — same ladder, homotopy,
cleanup and dip bound, fit in b = d − (Σd)a, |d| ≤ 0.8; analytic Jacobian
checked against finite differences to 8e-10; the complex path stays
bit-identical to v1).  Versus v1-protocol labels at the same truths,
median label W₁ (physical): gauss 0.021 → 0.009, gapped 0.030 → 0.011,
semicircle 0.0073 → 0.0029, box 0.0063 → 0.0042, pseudogap 0.020 → 0.006,
with fewer poles (6–9 vs 9–10) and dips ≤ 1e-3.  The gain comes from
dispersive-dominant poles (a → 0, b finite), exactly as the 2026-09-01
diagnosis predicted — and that is the conflict: the frozen pole loss
compares poles through t = b/a, and the label load Σ b²/a (what one
matched-position, t = 0 prediction pays) is median 45–6800, max 1e6 for
gauss/gapped/semicircle/pseudogap, against the M1 prior's median 0.21,
max 30.  A bound |b| ≤ T·a inside the dispersive fit (hinge block +
exact projection) keeps the load ≤ 10 at T = 5–15 and holds v2 quality
for gapped/semicircle/box, but the bounded fit lands in poor optima on
pseudogap (W₁ 0.036–0.08, residual 5e-5 vs v1's 0.020 at 7e-8).
**Decision**: the step-2 data arm uses v1-protocol labels rebuilt at
β = 8 — reachable by the M1 decoder (|b| ≤ a lies inside the dispersive
space), priceable by the frozen loss (t ≤ 1), and already judged; their
W₁ sits 2–10× below today's smooth-case reconstruction errors.  Using
the dispersive space for labels is a follow-up arm that needs either a
loss feature that prices dispersive-dominant poles or a bounded fit seeded
from the v1 solution (feasible inside the bound, so never worse than v1).

**SOS residue sampler, prototyped against the current priors**
(*measured*, 2026-09-01, `scripts/step2/proto_sos_prior.py`,
`out/step2/sos_prior_proto.*`; motivation: the prior's residue
constraints should not silently bias coverage of the physical set — a
nonneg univariate polynomial is a sum of two squares, so at fixed pole
locations P = q1²+q2² parameterises the ENTIRE positive set: positivity
exact by construction, Σb = 0 automatic, no certificate, no shrink
bias; residues r_p = −2i P(z_p)/Q'(z_p), verified).  Matched
position/width draws at β = 8, 300 truths per sampler: step-1 map
max|b|/a med/p95 0.21/0.50, never > 1, dips 1%; M1 rescaled-t
dispersive 0.77/3.30, 40% exercise |b| > a, dips 13% (all within the
0.5% bound) at 2.2 certificate calls/truth (the shrink-loop cost and
bias, now quantified); SOS 2.98/41.5, 80% exercise |b| > a, dips
EXACTLY zero, fastest — and produces qualitatively new physical shapes
neither prior reaches (smooth two-hump backgrounds, one-sided sharp
band edges).  Two measured limitations: (i) 36% of pole arrangements
rejected outright (40 tries never yield all-positive weights — an
arrangement-coverage bias of its own); (ii) the induced weight
structure collapses (effective pole count med 2.3 vs 4.6 — random
numerators concentrate mass, losing the designed envelopes).  Verdict:
a valuable third STREAM, not a replacement.  Queued design that fixes
both at once, pending its own prototype: **SOS-perturbed motif truths**
— P = (1−λ)·P_motif + λ·(q1²+q2²): convex combinations of nonneg
numerators stay nonneg, so exact positivity survives while the motif
envelope and arrangement coverage are inherited at small λ; λ becomes
the fairness dial, A/B'd like every distribution change.

**Reconstruction gallery, nine truths x two models x two stages**
(*measured*, 2026-09-01, `out/step2/recon_gallery_nz*.{json,png}`;
amortized ~2 ms forward pass + z-descent 3000, no scans).  Standing
picture at quiet noise (1e-4), z-descent W₁ s1_metric vs M1-60k: M1
wins the smooth world ~2x (gauss pair 0.183/0.096, semicircle
0.216/0.094, box 0.222/0.111) and the fano band (0.026/0.014), ties
pseudogap (0.21 both — neither map helps its dip structure; the data
arm owns it), cedes ~20% on the discrete probes (triplet 0.049/0.061,
ladder 0.052/0.062).  At loud noise (1e-3) s1 leads most pole probes
(fano 0.028/0.102 — the known 60k regression) while M1 still renders
semicircle/box mildly better AND visibly comb-free where s1 produces
violent spike combs at the chi2 floor.  The amortized stage is rough
everywhere (W₁ 0.09–0.55, chi2 up to 4e3 at quiet noise) and slightly
worse for M1 — the pipeline's quality lives in z-descent onward, as
designed.  Two pipeline-level artifacts caught and measured: (i) the
gapped-pair reconstructions spike 4x above truth AT the chi2 floor
(null-space behaviour at 1e-3, both models); (ii) the lorentz-pair
probe sticks at chi2 ~12 under the default z-prior weight 1.0 for BOTH
models, seed-independent, and releasing to 0.2 reaches chi2 1.7 / W₁
halved — the encoder mis-centres this simple input and the anchor pins
the descent there; prior-weight sensitivity of z-descent joins the
diagnostics watchlist.

**The mis-centering mechanism, isolated: a mirror-symmetry attractor in
the encoder** (*measured*, 2026-09-01, s1_metric — i.e., pre-existing
step-1 behaviour, not a map-arm effect).  Amortized chi2 on two-sided
Lorentzian pairs grows MONOTONICALLY as the side weights balance: 6.6
(a = 80/20) → 18 (65/35) → 36 (55/45) → 97 (50/50), while for the
balanced cases best-of-64 posterior samples barely help (97 → 53: the
posterior fails to COVER, not merely to centre) yet a one-sided pair is
rescued by sampling (30 → 8.7).  The smoking gun: the proposal's mirror
asymmetry is shrunk toward zero by exactly the amount that predicts the
failure — truth/proposal asymmetry 0.540/0.540 at 80/20 (no shrink),
0.331/0.306, 0.253/0.163 (−36%), 0.227/0.104 (−54%) at 50/50.  The
encoder snaps near-balanced two-sided inputs toward the PH-symmetric
submanifold — an amortization prior plausibly trained in by the pole
prior's 30% exactly-mirrored draws.  This retro-explains the banked
step-1 observations (encoder "mis-centred more than multimodal", probe
z-descent floors 6–15, amortized proposal seeding bad basins) and
matters for real data: the Holstein is exactly a near-symmetric-but-
not cell (PH-asymmetric at 4-sigma).  Candidate fixes, each its own
A/B: training side — soften the exact-mirror subpopulation (symmetry-
breaking jitter on the ph draws); inference side — adaptive z-prior
weight or best-of-K seeding when the input is near tau-symmetric.
2026-09-01*).  (i) *Rank-ladder shape verdicts*
(`physics.consolidate.ladder_verdict`, judged in
`out/step2/ladder_verdicts.*`): the discrete-vs-continuum dichotomy of
the noiseless ladders (crash-to-floor vs geometric decay; cf. Zhang &
Gull Fig. 2) applied to the banked dual-output chi2(n) traces.  Arrival
SPEED is not a discriminator (measured overlap: the triplet crash
arrives x9.8 while a semicircle's geometric rungs run x10-40); cliff
DOMINANCE plus the post-arrival shelf is.  Verdicts: asym triplet
**pole-closed at n=3** (cliff share 76%, shelf 4); fano band
**ambiguous** (a resolved band at the comb/smooth boundary -- the
honest answer); **Holstein: continuum-like** (arrives at the
calibration floor still improving x4.9, no shelf, cliff share 36%) --
independent support for the step-2 thesis that the cell carries smooth
spectral content the pole vocabulary lacks.  (ii) *Capacity vs beta*
(`out/step2/capacity_vs_beta.*`, ladders to n=20, 160 labels): with the
cap lifted, 45% (beta=8) to 70% (beta>=12) of smooth labels select
n>12, BUT the cost of capping at 12 is a median 4-7x on residuals
sitting at 1e-7 -- invisible at any training noise -- and the worst
n=12 residual stays <=1e-5 everywhere except one gauss_mixture draw at
beta=20 (1.5e-4, marginally above the 3e-5 quietest noise).  Design
number: **P_max=12 stands through beta=20**; the thin beta=20 tail says
re-run this sweep before any beta-extension beyond 20.  Caveat on
record: the verdict is at the data-residual level; the extra rungs were
separately measured to keep buying A-space W1, so the rendering-level
capacity question reopens if step-2 training shows W1 saturation.

Bank numbers, pinned (also asserted by the unit tests): median
Matsubara residual **1.0e-7** — the roadmap's 1e-4–1e-6 truncation
estimate, beaten; per-family med/p95: gauss 1.6e-7/1.0e-6, gapped
1.4e-7/9.4e-6, semicircle 6.1e-8/2.8e-7, box 6.9e-8/2.7e-7, pseudogap
1.5e-7/8.3e-7.  G-space through the training forward map: **98.8% of
entries below the quietest training noise (3e-5)**; the 15 exceptions
are 12 floor-limited truths at β ≲ 6.5 (features AT the resolution
floor; resid honestly recorded, up to 6.5e-2) plus 3 high-β
amplification cases ≤ 4.8e-5.  W₁(label, truth) median 0.023
(per-family 0.0096–0.049).  Label rank: median n = 10 at cap 12; rank
ladders decay geometrically through the noise band and the supported
rank grows as tolerance tightens — the roadmap's rank-vs-SNR unit test.
97.9% of labels carry a dip (asymmetry is load-bearing for smooth
shapes), every one ≤ the 5e-3 bound; minimum dimensionless width sits
exactly at the 1.05 × 0.1π floor in every family.  Floor-limited
regime documented: truths with features at the floor at β ≈ 2–3 are
representable only to ~1e-3–1e-2 with ~1.4× sub-resolution overshoot —
measured to be representability, not protocol (invariant under block
weight, mollification scale, and extra rank) — and the per-entry
``resid`` records exactly how far the output space can follow each
truth; training can weight by it.

**The user's resolution of the whole capacity question: the scan's top
rung IS the prior-started optimization, and reporting was the only
gap** (*verified*: the pre-refit rung reproduces the prior-start curve
— fano χ² 0.879 / W₁ 0.0118 vs 0.876 / 0.0113).  The good full-capacity
rendering was computed inside every anchored scan and discarded by the
minimal-form selection.  RankScanTrace now exposes it (`full_fit`,
`full_chi2`): the standing n_max = 8 protocol returns BOTH named
outputs from one scan — the LR-certified minimal skeleton and the
prior-started full rendering — no tempered selection, no second dial,
no extra compute.  The tempered machinery remains a diagnostic (its
unique contribution stays the mid-capacity fano n = 11 point).

**Ladder length settled: n_max = 8 stays the default** (*measured*,
n_max = 16 vs 8 at matched anchor weights — the "isn't a fitted descent
more prudent than a cold drop?" question, answered empirically NO at
every tested w).  The long fitted descent degrades the minimal forms
(triplet W₁ 0.023 → 0.056, fano 0.024 → 0.033; Holstein unchanged,
n = 6 in-band) even with the anchor force on: at the low rungs the
anchor is a 16-pole spectrum that few-pole fits cannot represent, so
its gradient pulls toward coverage compromises rather than the crisp
arrangement, and every extra fitted rung is a diffusion step among
floor-equivalent basins.  The unfitted drop-reduction cannot corrupt
what it keeps — the winning principle at all w, not only w → 0.
Standing protocol: the anchored scan with n_max = 8, unfitted drop
init, w ≈ 0.5–1, pure-χ² LR.  The 16-rung ladder and the tempered
selection remain diagnostic instruments (the fano n = 11 result shows
what they can find), not the default path.

**The tempered-prior scan** (*measured*, S(n) = χ²_tot + λ₀k(n) +
w·c(n), c(n) from posterior sampling, ladder to 16, λ₀ never scaled,
n_LR always reported).  The prior's measured count beliefs: fano
n ≈ 11, triplet n ≈ 10, Holstein n ≈ 13.  Headline: **fano at w = 16
selects n = 11 and reaches W₁ 0.0077 — the best reconstruction of the
campaign, better than BOTH limits of the interpolation** (minimal form
0.024–0.035, full 16-pole 0.011): the tempered middle found a point
neither endpoint contained.  The triplet behaves as registered — large
w climbs to n = 10 and degrades toward the 16-pole value (0.045), the
counter-case proving the dial is informative, not uniformly helpful.
The Holstein climbs 6 → 7 → 13 while staying in the calibration band
until w = 64 overreaches (χ² 0.32, anchor force distorting fits).
Negative finding, recorded: at w = 0 on the 16-rung ladder the
triplet's minimal form came back at 0.056, NOT the short-ladder 0.023 —
initialization alone carries the arrangement only a few rungs; over a
13-stage drop-descent the anchor FORCE is needed to maintain it.
Practical recipe: arrangement force floored at w_arr = max(w, 0.5)
(the measured corner where decoupling the two dials genuinely matters);
crossover w* (where selection departs n_LR, here between 4 and 16 for
all three cases) joins the truth-free per-dataset diagnostics.

**Anchored-scan weight sweep: no sharp optimum, and that is the
result** (*measured*, w = 0.1…5 on all three cases).  Certified counts
are w-INVARIANT (5/3/6 at every weight — the pure-χ² LR rule delivers
exactly the intended robustness); probe W₁ is a broad plateau (fano
0.0222–0.0245, triplet 0.0230–0.0251 — ±5% over 50× in w, opposite
gentle trends); the only meaningful gradient is real-data calibration:
the Holstein's certified fit moves from 1.7σ below the perfect-fit band
at w = 0.1 into the band at w = 2 (χ² 0.148).  Default w = 1; the
per-dataset tuner, when wanted, is truth-free: choose w to centre the
certified fit in the calibration band.

**The w = 0 control resolves the phase comparison: the cold merge was
the confound** (*measured*, user-demanded fairness check).  Pure χ²
refit of the UNMERGED 16-pole z-descent configuration reproduces the
"manifold-refit" probe results exactly (fano 0.875/0.0113, triplet
0.860/0.0444) — the penalty contributed nothing on in-vocabulary data
at any weight.  The free scans' spiky, W₁-poor answers came from the
scan's cold-merge destroying the arrangement BEFORE any fitting; the
prior's information lives in the starting arrangement, and plain
likelihood descent preserves it.  Ledger correction: the manifold
penalty's real value is confined to out-of-vocabulary data (Holstein,
where w measurably arbitrates noise absorption); on in-vocabulary data
the operative principle is *don't merge before you fit*.  Queued scan
upgrade from this finding: build count candidates by drop-lightest
pruning of the refitted 16-pole configuration instead of cold-merging
the raw decode.

**Scope decision for step 2** (user directive 2026-08-26): step 2
changes ONE thing — MPM-labeled smooth families in the training data —
judged against the banked targets (manifold floor 24.0, canary 0.438,
w-sensitivity flag, seven probes green).  The latent-basin/menu
unification (auxiliary class heads, structured latent) is explicitly
DEFERRED to its own arm on top of the step-2 checkpoint (heads on/off
A/B at fixed data), per the one-variable-per-step discipline.  Also
recorded from the sweep: initialization is itself a prior — at w → 0
the prior-started refit does NOT recover the free scan's result (it
finds a better-likelihood basin near its start: fano 0.874 vs the
scan's 0.935) — the starting point, not the penalty, carries most of
the prior's influence on in-vocabulary data.

**Fine weight sweep of the prior-started refit** (*measured*, w =
0.05…2 on all three cases): in-vocabulary data makes the weight
irrelevant — probe W₁ flat within noise across 40× in w (fano
0.0104–0.0119, triplet 0.0429–0.0439), every fit at floor χ² with
penalty ≤ 10⁻³; from the prior-started point the χ² floor for
in-distribution data lies ON the manifold and the pull is a safety
rail, not a dial.  On the Holstein the weight matters exactly as the
out-of-vocabulary hypothesis predicts: small w lets the 16-pole refit
absorb noise (χ² 0.042 at w = 0.05, ~2.2σ below the perfect-fit band;
manifold distance 0.17), and even the count-gated answers sit below the
band until w ≥ 0.5 (gated: n = 6 at 0.067–0.069 for w ≤ 0.2; n = 6 at
0.132/0.136 for w = 0.5/1.0 — inside the band; n = 7 at 0.103 at
w = 2).  Consequences: (i) default w ≈ 0.5–2 for real data, anything
for synthetic; (ii) w-sensitivity is itself a *vocabulary-coverage
diagnostic* — a flat plateau certifies in-vocabulary data, w-dependence
flags that the manifold is being outvoted; (iii) the certified Holstein
count is w-stable at 6–7 with the n = 6/7 split sitting inside the
calibration band — report both, per the tie policy.

**The prior-started phase (user design) supersedes the free-started
one** (*measured*, decisive).  Same objective and budget, opposite
starting point: begin AT the z-descent decode (penalty starts ~0) and
let whitened χ² pull away with the manifold as a continuous prior
force.  Direction of approach decides the basin: on fano the
prior-started fit reaches **floor χ² 0.877, manifold distance 0.0008,
W₁ 0.011 — template-quality with zero template knowledge**, the best
assumption-free reconstruction the pipeline has produced (free scan
0.054, free-started 0.045, best template 0.012); triplet W₁ 0.043 vs
free 0.062.  Canonical per-dataset order is now: z-descent →
**prior-started manifold refit** → count-gated scan → menu; the
free-started variant retires to a diagnostic.  On the Holstein the
ungated 16-pole phase overshoots (χ² 0.054–0.16 vs target 0.175 — no
count control), and the count-gated scan from that basin gives the
best-certified real-data answer yet: **n = 7 at χ² 0.103**.
Calibration honesty: the perfect-fit χ² distribution under this
correlation structure is 0.175 ± 0.061 (only ~17 effective independent
dof of 80), so 0.103 sits ~1.2σ below expectation — mild residual
overfit at worst, not the clean pathology the ungated 0.054 was.  The
real-data fit gap is effectively CLOSED by inference improvements
(20× → 3.2× → 2.2× → ~0.6×, within calibration uncertainty); what
remains for step 2 is expressiveness (manifold floor 24.0), the canary,
and the structured/physics description of the 7-pole answer.

**Manifold-regularized refits: the harshness dial, built and measured**
(`refine.manifold_refit`).  Free-pole refits with an A-space pull toward
the decoder manifold's projection (anchor z re-projected each round):
w → ∞ recovers z-descent, w → 0 the free refit.  Contract verified on
the probes (manifold distance monotone in w; χ² pinned at floor; W₁
improves 10–15% over the free fit — the hierarchy stands: declared
structure > manifold prior > no prior, so the dial upgrades the
*fallback*, not the menu).  On the Holstein it beats the design goal:
starting from free(6) at 0.559, the regularized refits reach **χ²
0.38–0.45** (ratio 2.2–2.6× vs 3.2×) while pulling 3–5× closer to the
manifold — the manifold guide *accelerates* likelihood descent (also
seen at short budgets in the unit test) rather than trading against it.
Real-data ratio now 2.2×.  Operational lesson, learned twice in one
day: the encoder must always see the dataset's FULL native grid (a
one-token 80-vs-81 change re-collapsed the scan); fits see the
statistical slices — `refine_latent`'s internal split is now the
mandated convention for every entry point.

**Step-1 gate closed with the boundary proof** (*measured*, Holstein
cell).  z-descent with the prior anchor released (λ = 0, 10k steps)
reaches χ² 23.96 — **137× the calibrated target**: the decoder
manifold's expressiveness floor on this data, free of regularisation or
optimiser excuses.  Free poles off-manifold reach 0.56 (3.2×).  The gap
between those two numbers IS the vocabulary step-1 training lacks (no
smooth families, no MPM labels) — step 2's training must pull the
manifold floor down toward the free floor; both numbers are now the
gate.  [*Re-measured 2026-11-24*: the 23.96 is NOT an expressiveness floor.  The
10,000-step descent was still falling when it stopped (19.4 at 40,000
steps), and Levenberg–Marquardt on the same objective reaches the noise
target, χ² 0.169 = 0.96× target, in 662 iterations.  The s1_metric
manifold contains a noise-level fit of this cell; it lies far from the
prior (|z| = 212 against ≈ 5 for a typical draw) and first-order descent
does not get there.  See "Solver and calibration re-measurement" in the
step-2 roadmap.]  Two more verdicts from the final menu: (i) the cell is NOT
PH-symmetric — established by the *paired* per-bin test (the correct
one: mirrored slices share bins and are positively correlated, so the
naive independent-slices test *understates* the significance):
τ-array verified exact (0…8 inclusive, uniform, mirror sums = β to
0 ulp), per-slice-pair t-statistic max |t| = 4.1, rms 2.5,
symmetry-hypothesis χ² = 6.5/dof — so the
asymmetric free(6) winner reflects real data asymmetry, every mirrored
entry (ladder AND the new mirrored_peaks, best 4.7) is floored by the
symmetric-model bound, and the earlier "robust mirrored_ladder(3)" was
the best *symmetric* story told to a broken scan; (ii) the winner's
sharp sub-thermal spikes are width-unidentifiable at χ² ≈ floor —
honest-resolution display remains the convention for such features.

**Real-data checkpoint A** (*measured*, half-filled Holstein β = 8,
Ω = 1.0, λ = 0.325, 400 bins, arm-C and arm-A models): the menu selects
**mirrored_ladder(3)** — PH symmetry detected from the bins by model
comparison, no declaration — and the result is *doubly robust*: identical
winner and parameters to 3 digits (ε₀ = 1.29, Ω = 1.45, γ at the width
floor, r = 0.94) under both models AND under two very different
whiteners (LW ρ = 0.012 vs structured-target ρ = 0.45, ξ = 3.05).  Open
gaps, banked as step-2 targets: (i) winner χ² sits ~20× above the
calibrated perfect-fit target with either whitener — the real spectrum
is richer than any current menu entry; (ii) the free-pole scan collapses
on real data (best fit 3 orders above the 4-parameter template,
non-monotone in n) — the amortized proposal seeds a bad basin the warm
descent cannot escape; template-informed scan starts are the queued fix.
Whitening: `covariance(method="structured")` shrinks toward the physical
exp(−|Δτ|/ξ) target (Schäfer–Strimmer intensity); on synthetic
exp-correlated bins it calibrates the perfect-fit target near 1 where
LW sat at ~0.24.  Bin autocorrelation ruled out as an explanation of the
gap (*measured*: blocking curve flat, Sokal τ_int = 0.53 ≈ white,
g = 1.07, n_eff = 375/400 — the bins are pre-blocked past their
correlation time), so the ~19–20× χ² excess is genuine spectral
mismatch and step 2 owns all of it.

**The rank scan closes the count question** (*measured*, built after the
comb-aliasing discovery).  `rank_scan` replaces the fixed point's local,
path-dependent count decision with a global one: warm-started descent in
n (each candidate = the fitted (n+1)-parent reduced by merge-closest or
drop-lightest, plus jittered restarts; χ² keeps the best basin per n),
selected by the sequential likelihood ratio (a step to n+1 must buy
≥ 9.5 in total whitened χ² — the pruner's doctrine applied to globally
optimised fits).  On the ladder probes from the composite model's output
it recovers the pole count on *all* cases: 3-rung → 3 (escaping the
6-pole alias the fixed point could never leave), mirrored → 4 with no
symmetry knowledge, and 4-rung → 3 — correctly: the faintest rung's
evidence is 6.6 < 9.5 at this noise realisation, so "the data does not
demand it" is the statistically right answer.  Within-count positions
are recovered to the information limit only — several χ²-equivalent
minima within ±0.3–0.5 exist for faint rungs (measured), so no χ²-based
method can pin them tighter; dominant poles are sharp.  The χ²(n) curve
in the trace is the honesty artifact: a sharp elbow = data-determined
count, a shelf = report the ambiguity.  (Mirror-constrained scan is
guarded off pending a correct folded descent; the plain scan needs no
symmetry declaration.)

**Noise-convergence study** (*measured*, full pipeline over σ = 3×10⁻³
… 10⁻⁵ on the ladder probes): the three error classes separate exactly
as the analyticity argument predicts.  *Counts* staircase up fast
(evidence ∝ 1/σ²): the 4-rung flips 3 → 4 at σ = 3×10⁻⁵, as computed in
advance from its 7.1-vs-9.5 evidence at 10⁻⁴.  *Spectra* converge then
plateau at W₁ ≈ 0.03 — not to zero at any tested noise.  *Faint-rung
positions* stagnate at ~0.3 across three decades — the logarithmic wall
drawn in data.  New phenomenon at extreme SNR: mild count *overshoot*
(4/3, 5/4 at σ ≤ 3×10⁻⁵) — at that precision the within-count position
slop of the refit basins is itself worth ≫ 9.5 in χ², so the LR test
honestly demands an extra pole to absorb it; an optimization/amortization
precision limit, not physics.  Together these are the quantitative case
for structured-template priors at selection time: only shared-parameter
structure can cross the log wall.

**Structured templates cross the log wall** (*measured*,
`physics/templates.py`).  Physics hypotheses fit as competing models —
a ladder is 4 parameters (base, spacing, shared width, decay) for *any*
rung count — selected against free poles by an ascending-complexity LR
walk (Wilson–Hilferty χ² quantiles; equal complexity decided by fit
alone).  On the probes: 3-rung recovered to **four digits**
(ε₀ = 0.4999, Ω = 0.9996, γ = 0.300, r = 0.5504; W₁ 2×10⁻⁴); the
4-rung's faint rung — individually below the 9.5 evidence threshold —
is recovered *through the template* (ladder(4) vs ladder(3) is an
equal-k contest: 0.906 vs 153, decisive) with all parameters to 3–4
digits; the mirrored case selects mirrored_ladder(2) exactly
(1.198, 1.001, 0.299, 0.454 vs 1.2, 1.0, 0.3, 0.45), i.e. symmetry
*detected by model comparison*, no declaration needed; and the
extreme-SNR count overshoot vanishes (templates leave no slop for an
absorber pole).  Free poles stay on the menu as the assumption-free
fallback, so the floor is the plain rank scan; a template only wins
when the data is consistent with it AND it is more parsimonious.  The
selection trace (winner, χ², margins over runners-up) is part of the
output — the prior's entire contribution is declared, per dataset.

**The three error laws, measured with guide lines and no fits**
(template sweep, σ = 3×10⁻³ … 10⁻⁵): *detection* evidence rides slope
−2 over four decades (the 3-rung's faint-rung evidence crosses the
χ²₄ threshold at σ ≈ 2×10⁻³, exactly where its noisiest-point selection
degrades — the panels interlock); *identification* error (template W₁
and rung positions) rides slope +1 down to W₁ ~ 10⁻⁴–10⁻⁵; free-pole
*resolution* sits on the 1/ln(1/σ) wall (flat at ~0.3–0.8 position
error across all noise) while the template track descends linearly
through it — three to four orders of magnitude below at the clean end.
Templates also recover full structure at σ = 3×10⁻³ where the free scan
supports only half the poles (ladder(4) at W₁ 0.005 from data the
likelihood reads as two bumps).  At fixed data quality, the declared
prior is the resolution.

Residual (ii) resolves the same way — and exposes what pretraining
quality is *for* (*measured*).  The consolidation fixed point run from
the pole-loss model's amortized output recovers the **exact minimal
pole set** on all three probes: 16 → 1 / 2 / 2 poles with the true
(ε, γ, a) to 3–4 digits, b collapsed to ≤ 2×10⁻⁴, χ² ≈ 0.88–0.92.
From the A-space model's messier output the same procedure stalls at
5–7 poles: the main features are right, but the single true pole stays
split into two just outside the π/β merge resolution (separation 0.404
vs threshold 0.393) plus ~1%-weight dust — a χ²-equivalent local fixed
point the data cannot distinguish from the truth.  Consolidation is a
*local* fixed point: the pretraining basin decides which representation
is recovered, not just how accurate the spectrum is.  That is the
measured argument for pole-first pretraining.
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

**Since 2026-11-23 the judges run through `spevae eval`** (docs/EVAL.md):
one case registry, one scorer for every method, MaxEnt and DEAC in every
scorecard, 8 noise draws per case, hierarchical bootstrap intervals and
paired comparisons. It reproduces the banked judges (204 of 212 checks;
every χ², count, stream and Holstein number bit for bit; the 8 drifts are
the 2026-09-01 map judge banked in the previous environment).

**First full scorecard, s1_metric vs M1-60k** (*measured*, 2026-11-23,
`out/eval/s1_vs_M1_full/`; the scan pipeline on the first draw pair only):

- **Versus MaxEnt (exact noise covariance, flat default model)**, scan
  pipeline: spevae's W₁ is 2–550× lower on all nine pole-closed cases at
  both noise levels, every interval excluding 1 (minimal form 2.1–550×,
  full rendering 2.1–55×; Lorentzian pair 0.001 vs 0.49 at 1e-4). MaxEnt is
  better on the smooth truths — gapped pair 9.2× (1e-4) / 2.6× (1e-3),
  Gaussian pair 3.6× (1e-4), pseudogap 2.4× / 1.9×, semicircle 1.3× (1e-4);
  box splits (MaxEnt 1.1× at 1e-4, spevae 1.3× at 1e-3). This is the step-2
  target stated as a number: close the smooth gap to MaxEnt without
  giving up the pole-closed advantage.
- **The M1 fano regression is a z-descent-stage effect.** Under zd3000
  M1/s1 on the Fano band at 1e-3 is 3.55× [3.48, 3.62] over 8 draws; under
  the scan pipeline it is 1.11 [0.99, 1.23] (minimal) and 1.02 (full). With
  the scan pipeline M1 vs s1 is a wash (scan_full 10 better / 10 worse of
  28 cells); M1's distinct gains are the Holstein manifold floor
  (23.96 → 3.83; step-capped descents, see the 2026-11-24
  re-measurement) and smooth truths under z-descent (semicircle zd3000
  0.216 → 0.094 at 1e-4).
- **The amortized pass is far from the data**: χ² 50–6000 per point at
  1e-4 for both models; the inference pipeline (≈60 s per dataset for the
  scan) does the work. This amortization gap is the measured case for a
  stronger posterior model.
- **Holstein**: every spevae scan output, MaxEnt (χ²/target 0.39) and DEAC
  (0.96) fit the data within noise, yet spevae returns sharp phonon-ladder
  peaks while MaxEnt and DEAC return broad humps. The data do not decide
  between them; the prior does. The training prior is therefore the
  quantity to get right for real data.
- **Noise-draw intervals are narrow** (errors are systematic per
  checkpoint); recipe claims need training-seed replicates, which the suite
  takes as repeated `--model TAG=...`.

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
