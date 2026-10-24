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
24.0 → down materially (THE gate); legacy canary 0.438 → down (its
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
gap (6.7x) is what the smooth-family data arm now owns.  The anchored
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
gate.  Two more verdicts from the final menu: (i) the cell is NOT
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
