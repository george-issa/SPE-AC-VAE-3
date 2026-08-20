# Design

Why this pipeline is built the way it is, what it inherits from the first
attempt (`../SPE-AC-VAE`), and where it deliberately departs from it.

---

## 1. The problem

Determinant quantum Monte Carlo returns an imaginary-time Green's function on a
grid $\tau_i = i\,\Delta\tau$, $i = 0 \ldots \beta/\Delta\tau$, with correlated
statistical noise. The physics of interest lives on the real-frequency axis:

$$G(\tau) \;=\; \int_{-\infty}^{\infty}\! d\omega\; A(\omega)\,
\frac{e^{-\tau\omega}}{1+e^{-\beta\omega}},
\qquad A(\omega)\ge 0,\quad \int A\,d\omega = 1 .$$

Inverting this is ill-posed: the kernel's singular values fall exponentially, so
spectral features that differ visibly on the real axis can produce $G(\tau)$
that differ by less than the Monte Carlo noise. Any method must therefore supply
information the data does not contain. The question is only *where that
information comes from and whether it is stated honestly*.

Three answers exist in the literature. **Maximum entropy** supplies it as a
default model plus an entropic prior. **Stochastic methods** (SAC, and
`SmoQyDEAC.jl`) supply it as a sampling measure over configurations. **Learned
methods** supply it as a training distribution. This package is the third kind,
with as much of the prior as possible moved out of the training distribution and
into *hard structural constraints*, because a constraint that holds by
construction cannot be argued with and does not need a control experiment.

---

## 2. Representation: a simple pole expansion

$$g(z) = \sum_{p=1}^{P} \frac{r_p}{z-z_p},\qquad
z_p = \varepsilon_p - i\gamma_p,\qquad r_p = a_p + i b_p$$

$$A(\omega) = -\frac1\pi \operatorname{Im} g(\omega+i0^+)
= \frac1\pi \sum_p \frac{a_p\gamma_p - b_p(\omega-\varepsilon_p)}
{(\omega-\varepsilon_p)^2+\gamma_p^2}$$

Consequences that matter in practice:

- **No frequency grid.** The output is $4P$ numbers. $A(\omega)$ can be
  evaluated anywhere, at any resolution, after training — no rebinning, no
  interpolation, no grid-dependent artefacts.
- **Compact.** $P \sim 10$–$20$ numbers versus $10^3$ grid points, which is a
  far better-conditioned thing to ask a decoder to produce.
- **Well-founded.** Pole representations are the modern high-accuracy approach to
  this problem: minimal pole representations converge systematically with data
  quality ([Zhang & Gull, PRB 110, 035154 (2024)][mpm]), stochastic pole expansion
  resolves sharp and smooth features together ([Huang & Liang, PRB 108, 235143
  (2023)][spx]), and barycentric/AAA rational fits do the same very cheaply
  ([Huang, PRB 111, 125139 (2025)][aaa]).  The same work carries a warning this
  package inherits deliberately: real-axis error grows exponentially with the
  Matsubara noise level, so the *recoverable* pole count is set by the data, not
  by ambition — which is why $P$ is an upper bound softened by `softmax` weights
  and the reported number is the participation ratio, not $P$.

### The residue-mode choice

`residue_mode="complex"` — the default — carries the full physics constraint
set this project specifies:

| requirement | how |
|---|---|
| causality | $\gamma_p > 0 \Rightarrow$ all poles in the lower half plane |
| $\operatorname{Re} r_p > 0$ (weight interpretability) | `softmax` weights $a_p$ |
| $\sum_p \operatorname{Re} r_p = 1$ (× measured norm) | `softmax` + norm rescaling |
| $\sum_p \operatorname{Im} r_p = 0$, individually $\neq 0$ | $b_p = a_p(t_p - \langle t\rangle_a)$, $t = \tanh(u)/2$ |
| bounded asymmetry | $\|b_p\| \le a_p$ from the same map |
| $A \sim c_2/\omega^2$ | follows from $\sum_p b_p = 0$ |
| width floor $\gamma_p \ge 0.1\pi/\beta$ | additive floor inside the `softplus` |

The nonzero $b_p$ are what buy **asymmetric (Fano-like) line shapes** — real
spectra are "almost Lorentzian, but not exactly", and a symmetric-only basis
pays for every asymmetry by stacking extra poles.  The price is that pointwise
positivity stops being a per-term property: a single pole with $b_p\neq0$ has

$$\min_\omega A_p = \frac{a_p - \sqrt{a_p^2+b_p^2}}{2\pi\gamma_p} < 0,$$

diverging as $\gamma_p \to 0$, and positivity of the total relies on cross-pole
cancellation. **This was the dominant pathology of the first attempt**, where a
systematic audit found every divergent run had $\min_p\gamma_p < 0.06$ while
every healthy run had $\gamma_p > 0.23$ — and no term in the loss could see it,
because a near-delta pole still produces a perfectly smooth $G(\tau)$.

This package keeps the asymmetry and closes the pathology from four sides:

1. the **width floor** removes the $\gamma \to 0$ divergence outright (with the
   optional `width_barrier` as a soft standoff above it);
2. the **$|b_p| \le a_p$ bound** ties every dispersive part to the weight of
   its own pole, so a weightless pole is structurally inert;
3. a **positivity penalty** in both spaces during training — spectral
   $\int\mathrm{ReLU}(-A)^2$ on the quadrature, and the complete-monotonicity
   penalties on $G, G'', G''''$ in data space (Stage 2);
4. an **exact, grid-free certificate at inference**
   (`spevae.physics.positivity`): $A \ge 0$ on all of $\mathbb{R}$ reduces to a
   real-root count of the degree-$2P{-}2$ numerator polynomial, decided from
   the pole parameters by a companion matrix; where $A$ does dip negative, the
   dip is located and its total weight integrated in closed form.  The verdict
   ships with every reported spectrum — positivity is *checked and stated*,
   never assumed.

With $b_p \equiv 0$ (`residue_mode="real"`) the expansion degenerates to a
**positive mixture of Lorentzians**: $A \ge 0$ pointwise by construction — the
discrete analogue of the observation that respecting the Nevanlinna analytic
structure yields intrinsically positive and normalised spectra
([Fei, Yeh & Gull, PRL 126, 056402 (2021)][nevanlinna]).  That mode remains one
flag away for correlators where the structural guarantee outranks asymmetry;
the measured trade on matched A/B runs: complex buys relative $L^2$
(0.277 vs 0.323 overall, 0.280 vs 0.375 on gapped-only training) at a
$\min_\omega A \sim -10^{-3}$ cost that the certificate quantifies per result.

---

## 3. The dimensionless reduction

Substituting $x = \tau/\beta \in [0,1]$ and $\nu = \beta\omega$:

$$G(x) = \int d\nu\; \tilde A(\nu)\, K(x,\nu),
\qquad K(x,\nu) = \frac{1}{e^{x\nu}+e^{(x-1)\nu}},
\qquad \tilde A(\nu) = A(\nu/\beta)/\beta .$$

**$\beta$ has vanished from the kernel.** The inverse problem is one problem, not
a family indexed by temperature. Three things follow.

1. **The input-dimension problem dissolves.** $L_\tau = \beta/\Delta\tau + 1$
   changes with the simulation, but $x \in [0,1]$ does not. Combined with an
   attention encoder over time slices, one set of weights accepts any grid. The
   first attempt used a fixed-width convolutional stack and therefore needed a
   separate model per cell — fatal for a package other people run on their own
   data.
2. **The width floor becomes a constant.** The finest resolvable structure is set
   by the first Matsubara frequency $\omega_0 = \pi/\beta$, i.e.
   $\tilde\nu_0 = \pi$. So $\tilde\gamma_{\min} = 0.1\pi$ is one number at every
   temperature, rather than a $\beta$-dependent hyperparameter to re-tune.
3. **Training transfers across $\beta$.** A gapped spectrum at $\beta = 5$ and
   one at $\beta = 20$ are the same learning problem in $\nu$.

$\beta$ still enters as *conditioning*: it does not change the map, but it does
change the prior (bandwidth measured in units of temperature), so it is fed to
the encoder and it sets the physical frequency cutoff.

---

## 4. Exact 1-homogeneity: "zero maps to zero"

Analytic continuation is a **linear** inverse problem, so the estimator should be
positively homogeneous: $A[cG] = c\,A[G]$. The package enforces this exactly.

The spectral norm is read off the data by the fermionic sum rule,
$N = G(0)+G(\beta) = \int A\,d\omega$. The encoder sees only $\hat G = G/N$ and
is therefore *exactly* scale-invariant; the decoder multiplies the residues by
$N$. Hence

$$A[cG] = c\,A[G] \quad\text{for all } c, \qquad\text{in particular}\qquad A[0]=0,$$

identically in the weights, at initialisation and out of distribution. Note the
structure: the *poles* are scale-invariant and the *residues* are
1-homogeneous — rescaling $G$ changes how much spectral weight there is, not
where it sits. `tests/test_model.py` asserts both.

This is also why $\sum_p a_p = 1$ and "$G=0 \Rightarrow A=0$" are not in
conflict, which they would be if the sum rule were hardwired to the constant 1:
the *normalised* residues sum to one, the physical ones sum to the measured
norm, and that norm is 1 for a fermionic single-particle $G$.

---

## 5. The forward map

The reconstruction loss is only as good as the map that produces $G$ from the
poles. Requirement: accuracy well below the DQMC bin-mean noise
($\sim 2.5\times10^{-5}$), *uniformly over the pole configurations the decoder
can emit*.

### What was wrong with the tangent-substituted rule

The natural rule for a Lorentzian is $\nu = \varepsilon+\gamma\tan(\pi t/2)$,
which flattens the pole's own measure and covers $\mathbb{R}$ exactly. The first
attempt used it with 256 Gauss–Legendre nodes. It is adapted to the *pole* but
not to the *kernel*: the Fermi factor turns over on a scale $\Delta\nu\sim1$ at
$\nu=0$, which in $t$ coordinates is a boundary layer of width
$2\gamma/\pi\varepsilon^2$ pinned at $|t|\to1$. Resolving it needs

$$n_{\rm nodes} \gtrsim \pi^{3/2}\,|\varepsilon|^{3/2}/\gamma^{1/2},$$

which measured convergence reproduces: ~225 nodes at $\beta\omega=8$, but
$\sim4\times10^4$ at $\beta\omega=250$. Accuracy therefore degrades silently as
$\beta$ or the bandwidth grows. Measured here: with $|\varepsilon| \lesssim 24$
and 64 nodes the error was $2\times10^{-5}$ — already at the noise floor — and
it gets worse from there.

### What is used instead

`SpectralQuadrature` is a fixed, batch-shared rule with two panels:

- **Core** — trapezoid on a uniform grid over $[-\nu_c,\nu_c]$, spacing $h$. By
  Poisson summation the error for an integrand analytic in a strip of half-width
  $\gamma$ is $2e^{-2\pi\gamma/h}$: spectral accuracy, *uniform in where the
  feature sits*. At $h = \tilde\gamma_{\min}/3.5$ that is $5\times10^{-10}$.
- **Tails** — $\nu = \pm\nu_c/u$, $u\in(0,1]$, Gauss–Legendre in $u$. The
  Jacobian $\nu_c/u^2$ exactly cancels the $1/\nu^2$ decay, so the fat Lorentzian
  tails are handled *exactly* rather than truncated (a plain uniform grid would
  need $\nu_c\sim10^6$).

Measured against adaptive quadrature: $10^{-9}$–$10^{-12}$ across
$|\varepsilon|$ up to 200 and $\gamma$ from the floor to $5\times10^3$, and
$\sim10^{-10}$ in the worst case with every pole pinned at the width floor. The
dependence on $h$ is exponential, as predicted — 0.2 gives $10^{-5}$, 0.09 gives
$10^{-10}$ — so `nodes_per_width` is not a knob to economise on.

Bonus properties: nodes are shared, so the whole map is one matrix product with
$O(BM + ML)$ memory instead of $O(BPNL)$; and the *same* rule integrates pole
expansions, Gaussian mixtures and arbitrary curves, so training targets and model
predictions are never compared across two different quadratures.

The first attempt needed an elaborate five-moment Matsubara subtraction in
float64 to patch the $\tau=0$ endpoint. None of that is required here.

---

## 6. Architecture

**Encoder** — a transformer over imaginary-time tokens. Each token carries the
standardised $\hat G$, a signed-log channel, the local noise level, and
continuous positional features of $x$; a `CLS` token carries $\log\beta$,
$\log L_\tau$, the curve's own location and scale, and the leading Matsubara
coefficients $g(i\nu_n)$ (the representation every classical method works in,
handed over for free). Ragged lengths are handled by a padding mask.

One implementation detail turned out to be decisive and is worth recording.
Within a batch the grid, its Fourier features, $\beta$ and the noise scale are
*identical across samples*; only $\hat G$ varies. Fed raw, the constant channels
dominate the embedding magnitude and the per-sample signal survives as a ~2%
perturbation — which after a few normalised layers left the posterior mean
essentially constant, i.e. **zero active units from step one**, with no KL
pressure needed to explain it. Per-sample standardisation along $\tau$, a
signed-log channel, scale-matched auxiliaries and a `LayerNorm` on both
embeddings fixed it: on a fixed-batch overfit test, active units went 0 → 8 and
the spectral $W_1$ error went 1.88 → 0.43.

**Decoder** — $P$ learned *pole queries* refined by self-attention and
conditioned on $z$ (a DETR-style set decoder). The poles are a set: all $P!$
relabellings give the same $A(\omega)$, so an MLP emitting $4P$ ordered numbers
must break that symmetry arbitrarily and gradients from different samples fight
over the same output slot. Position-head biases start spread across the band, so
the queries do not start degenerate.

**Constraint layer** — every guarantee in §2, applied on the last line, with
`softmax` for the weights (smooth everywhere, unlike $|a|/\sum|a|$, whose kink at
zero has zero subgradient and permanently traps a pole that reaches zero weight),
`softplus` + floor for the widths, and a `sinh` position map with the bound
imposed by clamping the *pre-activation* so the interior stays exactly
$\beta$-independent.

`P` is an upper bound, not a target; report
`effective_pole_count` (participation ratio $1/\sum_p a_p^2$), which cannot be
inflated by a dusting of near-zero weights.

---

## 7. Training

**Stage 1 — amortised supervised pretraining** produces the shipped model: one
set of weights that continues anybody's $G(\tau)$ at any $(\beta,\Delta\tau)$ and
noise level in a single forward pass. This is a deliberate inversion of the first
attempt, which fitted a fresh model per dataset by minimising $\chi^2$ — a
procedure with no prior, so nothing prevents it from landing on whichever of the
many $\chi^2\approx1$ spectra the optimiser reaches first. That failure was
observed and documented there: a matched-compute campaign found $\chi^2\approx1$
runs that recovered the wrong spectrum, and concluded that "$\chi^2 \approx 1$
implies correct $A(\omega)$" is false.

Supervision is the only stage at which the ill-posedness can actually be beaten:
on real data no loss can separate two spectra that give the same $G(\tau)$ to
within noise, so the ability to tell them apart has to be learned beforehand,
from cases where the answer was known. This is the same rationale as
[Fournier *et al.*, PRL 124, 056401 (2020)][fournier], who found a supervised
network matches maximum entropy at low noise and beats it substantially at high
noise, three orders of magnitude faster.

The generator is infinite, so there is no fixed epoch and no overfitting to a
finite training set: longer runs and larger models buy accuracy monotonically
until the network saturates the information in the data.  Measured, on a
genuinely frozen held-out stream (the evaluation batches are pinned by
`fixed_scalars`, so these numbers are comparable across checkpoints — the
predecessor's eval stream drifted with training, and $\chi^2$ once "improved"
3.5× between two evaluations of the same weights): over a 27-minute
`fast`-preset run of 6000 steps, every metric improved at every one of twelve
evaluations — $W_1$: 1.05 → 0.35, relative $L^2$: 0.38 → 0.19, geometric-mean
$\chi^2$: 785 → 71 — with active units steady at 11.75/12 (no collapse).

### Measured: the pole-count sweep

Three runs at $P \in \{8, 16, 24\}$, identical seed, preset and frozen eval
stream (`scripts/pole_count_report.py`, `scripts/plot_pole_count.py`,
`scripts/plot_pole_count_spectra.py`):

```
P    params       W1    l2rel   chi2geo   eff mean   eff p90
 8  812,188   0.2859   0.1867     303.8       6.36      7.45
16  812,988   0.2734   0.1592     306.8      13.63     15.39
24  813,788   0.2787   0.1598     300.2      19.10     22.76
```

Two directions, two different answers:

- **Up (16 → 24): free.** Every accuracy metric is tied within noise; the
  eval trajectories overlap point by point from step 500 to the end. Surplus
  poles spread the same spectral weight more finely (effective count grows
  with the budget at roughly constant ~80–85% utilisation) instead of
  misbehaving — "prediction gets equally good or better with more poles"
  holds as *equally good*, at this training budget and noise level.
- **Down (16 → 8): starvation, and it is shape-selective.** $W_1$ (peak
  positions) stays within ~4%, but relative $L^2$ (shapes and heights) is 17%
  worse and the gap *widens* over training — eight poles can place features
  but run out of basis functions to refine them, exactly the deficit the
  first attempt hit with 3 Lorentzians against two Gaussians. On a
  PH-symmetric coupling-resolved Holstein cell ($\lambda{=}0.35$, $n{=}1$,
  $\Omega{=}1$, $\beta{=}10$), starvation also shows as **symmetry
  violation**: the model does not hardwire $A(\omega) = A(-\omega)$, so the
  integrated asymmetry is a learned-quality diagnostic — 0.54 at $P=8$ versus
  0.21–0.25 at $P \ge 16$ (shipped DEAC reference: 0.065). Too few poles
  cannot tile two symmetric bands consistently.

One caution: refinement quality at fixed $P$ (a single MAP restart of a
non-convex optimisation) is *not* a pole-count measurement; conclusions about
$P$ belong to the amortised, frozen-stream comparison above.

### Measured: the ripple (pole-comb) experiment

The visible artefact of a fast checkpoint is a comb of narrow Lorentzians
tiling what should be a smooth band. Anatomy: ripple lives at the sub-thermal
scale, which is the kernel's null space — $\chi^2$ *cannot* see it (that is
the ill-posedness), and it cancels in $W_1$'s CDF; only the $L^2$ term sees
it, and it carried weight 0.3 against $W_1$'s 3.0. It is not a representation
limit (two poles reproduce a two-Lorentzian truth exactly) and not the
constraint layer.

Because the losses cannot report what they cannot see, `spectral_ripple`
measures it directly — sub-thermal oscillation content
$\int|A - \tilde A_\pi|/\int A$, logged next to the target's own value in
every evaluation (sharp families carry *real* sub-thermal structure; the
artefact is the excess).

Five 3000-step arms at fixed seed, preset and frozen eval stream:

```
arm                      W1agg    l2agg   rippleAgg
D control               0.4275   0.2159      0.1809
A  (w1 1.5, l2 1.0)     0.4941   0.2200      0.1317
A' (w1 3.0, l2 1.0)     0.4380   0.2053      0.1337
B  width barrier 5e-3   0.4380   0.2226      0.1864
C  sparsity 0.03        0.4492   0.2312      0.1960
```

$L^2$ pressure is the lever that works: arm A′ cuts ripple 27% and improves
aggregate $L^2$ at control-level $W_1$ and $\chi^2$, so ``l2 = 1.0`` is now
the default. The width barrier and the weight-entropy sparsity term did
nothing at the tested weights — recorded as negative results so they are not
re-tried blindly.

Confirmed at matched full budget (6000 steps, same frozen stream): ripple
0.135 vs 0.178 (−24%), relative $L^2$ 0.147 vs 0.159, $W_1$ and $\chi^2$ at
parity. The residual comb is training time — the trajectories were still
improving monotonically at cutoff — and the ensemble mean (the default
reported spectrum) averages a further fraction away.  A footnote for anyone
timing runs: suppressing the spurious per-batch MPS `irfft` warning nearly
*doubled* training throughput (832 s vs 1622 s for identical work) — the
warning machinery and log I/O were a real cost, not just noise.

**Stage 2 — per-dataset fine-tuning** (`spevae.training.finetune`) adapts the
*weights* to one measured cell — its noise level, correlation structure and
spectral family — with no ground truth: the self-supervision is the calibrated
per-bin $\chi^2$, the free-bits KL, the consistency term applied to what bins
really are (independent noise realisations of one spectrum), and the positivity
structure in both spaces.  Three design points, each anchored to a measured
failure:

- **`train="encoder_only"` by default.** The first attempt measured what a free
  $\chi^2$ fine-tune does to the pole budget Stage 1 installs: pole usage went
  2.30 → 11.08 on identical inputs — surplus poles wake up to fit noise, which
  is exactly the noise-budget warning from the minimal-pole literature
  ([Zhang & Gull][mpm]).  Freezing the decoder keeps the latent → pole map and
  its learned sparsity; the encoder adapts to the cell.  `"all"` and
  `"decoder_only"` remain available and are recorded in the history.
- **A two-phase divergence guard.**  Real bins contain outliers, and one bad
  minibatch early in fine-tuning can throw a pretrained model somewhere it
  never returns from.  A batch whose loss is non-finite or grossly out of
  distribution — relative to an EMA of *accepted* losses, so a bad batch cannot
  inflate its own threshold — is skipped before `backward()`; a non-finite
  gradient norm is skipped after; a full epoch of skips stops the run and
  restores the best checkpoint.
- **The complete-monotonicity penalties.**  For $A \ge 0$ the fermionic kernel
  makes $G$, $G''$ and $G''''$ all non-negative; the variance-weighted stencil
  penalties (Ben Cohen-Stead's formulation, ported from the first attempt)
  charge violations in data space, where the complex-residue mode is otherwise
  free to buy $\chi^2$ with unphysical dips.

**Stage 3 — per-input latent refinement** (`spevae.refine`) moves only $z$ with
the weights frozen: the cheap, always-safe last mile for a single spectrum,
stopping at the *calibrated* noise floor rather than at 1.  The three stages
compose: pretrain once, fine-tune per dataset, refine per input.

### The objective

| term | purpose |
|---|---|
| $\sqrt{\chi^2}$ | data fit, covariance-whitened |
| $W_1(A_{\rm pred}, A_{\rm true})$ | **peak position** |
| relative $L^2$ | peak heights and shape |
| KL with free bits | latent regularisation without collapse |
| symmetric-KL consistency | noise invariance |
| $\int \mathrm{ReLU}(-A)^2$ | positivity pressure (complex mode) |

Three choices deserve their reasons.

*Wasserstein over $L^2$.* $L^2$ is **flat in peak position** once two peaks stop
overlapping — moving a predicted peak from $\omega=3$ toward a true peak at
$\omega=1$ changes nothing until they touch. The first attempt hit this directly:
a documented series of experiments in which no setting of a smoothness
regulariser could move a misplaced peak, because smoothness penalises curvature,
not location, and the $\chi^2$ landscape is genuinely flat over a window of peak
positions. $W_1 = \int|F_{\rm pred}-F_{\rm true}|\,d\nu$ grows *linearly* with
displacement, so its gradient always points at the right answer. It is exact and
cheap on the quadrature grid. Reported in physical $\omega$ (i.e. $W_1/\beta$),
so it reads directly as a typical peak-placement error and so that cold cells do
not dominate the average.

*$\sqrt{\chi^2}$ rather than $\chi^2$ — or $\log(1+\chi^2)$.* The raw value
spans five orders of magnitude between initialisation ($\sim10^5$) and
convergence; used directly it swamps every other term early.  All the candidate
transforms are monotone (same minimiser), so only the gradient scale differs.
`log1p` was the first choice and **over-corrects**: at $\chi^2 \sim 4000$ its
gradient $1/(1+\chi^2)$ leaves the reconstruction contributing $3\times10^{-4}$
— nothing — and training proceeds on the spectral terms alone.  The A/B run
settled it: `sqrt` reached $W_1 = 0.594$ where `log1p` reached $0.723$ under
identical budgets, and $\sqrt{\chi^2}$ is the residual *norm*, whose gradient
$1/2\sqrt{\chi^2}$ compresses without silencing.

*Free bits, not KL annealing.* Posterior collapse occurs when the decoder can fit
the data mean from a constant code, making the encoder's marginal contribution
worth less than its KL cost. Free bits exempts the first $\lambda$ nats per
dimension, removing the incentive, at the cost of one hyperparameter rather than
a schedule. The first attempt documented collapse as its most important negative
result — $\chi^2$ and self-consistency both *improve* at collapse while the model
stops doing per-sample inference at all — and never implemented a fix.

### Noise handling

Training noise is correlated in $\tau$ with a sampled correlation length, an
independent "nugget" fraction, and an amplitude spanning $3\times10^{-5}$ to
$3\times10^{-2}$. Two consequences worth recording:

- The nugget is not a numerical convenience. A purely smooth correlation model is
  nearly rank-deficient, and whitening with its inverse demands an essentially
  exact fit along directions carrying no information — which inflated $\chi^2$ to
  $10^8$ here before it was added. Real bin noise always has an independent
  component.
- On real data, $R$ is estimated by **Ledoit–Wolf shrinkage of the correlation
  matrix** (not of the covariance: the per-slice variances are well estimated and
  should be left alone). The historically common alternative — truncating the
  eigendecomposition at 99% of the trace — discards exactly the directions
  encoding high-frequency spectral structure; on one real cell in the first
  attempt it kept 4 of 100 directions and the resulting spectra were blind to
  satellite features at $\omega\approx\pm3$.
- Under shrinkage the $\chi^2$ value at a perfect fit is **not 1** but
  $\frac1L\sum_k \lambda_k/(\rho\mu+(1-\rho)\lambda_k) \le 1$. Comparing against
  1 instead is how a "$\chi^2<1$ anomaly" gets manufactured — the first attempt
  carried one as an open blocker for months. `chi2_target` returns the correct
  value for any whitener, `expected_chi2` the closed form for plain Ledoit–Wolf.

### Measured on real data — and it is worse than expected

Reading one cell of the site-Holstein cube ($n=1.0$, $\Omega=1.0$, $\beta=10$;
100 bins, 101 time slices) through `spevae.data.read_holstein_cube`:

```
sum rule  : G(0)+G(beta) = 1.000000   (conventions confirmed)
LW rho    : 0.0069  ->  chi2 at a perfect fit = 0.1116
bins/slices: 100/100   *** N_bins <= L_tau ***
```

**The calibrated target is 0.11, not 1.** Two effects compound:

1. With $N_{\rm bins} \approx L_\tau$ the sample correlation is effectively
   rank-deficient. Thinning the $\tau$ grid on the *same* bins recovers the
   calibration monotonically — target $0.11 \to 0.19 \to 0.32 \to 0.54$ as the
   bins/slices ratio goes $1 \to 2 \to 4 \to 10$ — which identifies the ratio,
   not the estimator, as the binding constraint.
2. Ledoit–Wolf under-shrinks here *by design*: its intensity minimises distance
   to an identity target, and a strongly correlated $R$ is far from the identity,
   so $\rho$ comes out at $7\times10^{-3}$ and the near-null directions survive.

An eigenvalue floor (`min_eigenvalue`) bounds the whitener's amplification by
$1/\sqrt{\rm floor}$ — measured $11.3 \to 4.4$ at a floor of 0.05, which stops a
handful of null directions from dominating the gradient. It does **not** restore
$\mathbb{E}[\chi^2]=1$, and nothing can: no reweighting recovers information the
bins do not contain.

The consequence is worth stating plainly, because it bears directly on the
predecessor project's open blocker. Its Hubbard runs converged to
$\chi^2 \in [0.86, 0.88]$ and one site-Holstein cell to $0.627$, read as
"suspiciously below the noise floor of 1". Against a properly calibrated target
of order $0.1$–$0.3$ those runs were not below the floor at all — they were
*above* it, i.e. still underfitting. The anomaly was in the reference value, not
in the runs. **Actionable:** report $\chi^2/\chi^2_{\rm target}$, and either run
more bins or use fewer $\tau$ slices than bins.

### Noise invariance, stated in the objective

Every batch carries two independent noise draws of the same clean $G$. The
consistency term penalises the symmetric KL between their posteriors, which says
directly: *the answer is a property of the physics, not of the random seed*. It
also gives the latent space a reason to represent signal and discard noise —
the mechanism by which $\beta$-VAEs on Kondo spectra recovered $\log T_K$ and the
particle–hole asymmetry as individual latent coordinates
([Miles *et al.*, PRB 104, 235111 (2021)][kondo]).

### Training-set bias

A supervised network inherits the prior of what it was trained on; this is the
documented failure mode of the whole approach
([Zhang *et al.*, PRR 4, 043082 (2022)][bias]). A network trained on single
Gaussians returns a single Gaussian for a gapped superconductor, with an
excellent $\chi^2$. The response is breadth: nine qualitatively different
families (smooth multi-peak, gapped, correlated metal with quasiparticle peak and
Hubbard bands, phonon satellites, semicircle, BCS coherence peaks, box, Lorentzian
mixtures, and band-limited random positive fields), wide parameter ranges, and
per-family metrics in every evaluation — because an aggregate number hides
exactly the failure that matters.

---

## 8. Benchmarking against MaxEnt and SmoQyDEAC

`spevae.benchmark` runs three solvers through one interface, one forward map for
scoring, and one set of metrics:

| method | how | prior / regulariser | characteristic failure |
|---|---|---|---|
| **MaxEnt** | vendored [`ana_cont`][anacont] | default model + entropy | broadens sharp structure |
| **SmoQyDEAC** | [`SmoQyDEAC.jl`][deac] via a Julia subprocess | none (stochastic search) | visibly rough spectra |
| **spevae** | this package | learned training distribution | limited by that distribution |

The three fail *differently*, which is the point. Two methods that share a bias
agreeing tells you nothing; agreement across methods with unrelated failure modes
is real evidence that a feature is in the data rather than in a prior.

### The conventions genuinely match

All three define the same kernel — verified in `tests/test_benchmark.py`, not
assumed:

$$\underbrace{\frac{1}{e^{x\nu}+e^{(x-1)\nu}}}_{\text{spevae}}
=\underbrace{\frac{e^{-\tau\omega}}{1+e^{-\omega}}\bigg|_{\tau\to\tau/\beta,\;\omega\to\beta\omega}}_{\texttt{ana\_cont}}
=\underbrace{\frac{1}{e^{\omega\tau}+e^{-\omega(\beta-\tau)}}}_{\texttt{SmoQyDEAC}}$$

A sign or normalisation mismatch here produces plausible-looking but
incomparable spectra and nothing downstream would catch it.

### Scoring is done once, not three times

Each solver's own $\chi^2$ uses its own kernel discretisation and its own
normalisation, so those numbers are not comparable. Every result is instead
back-transformed through *this package's* forward map and scored against the
same whitened covariance. Two consequences:

- A disagreement between a solver's internal $\chi^2$ and ours is a convention
  bug, caught before it becomes a figure.
- The back-transform is truncated at whatever $\omega$ grid a method returned.
  For a grid solver that is correct — the grid *is* its answer. For a pole
  expansion it is not: the $1/\omega^2$ tail beyond the window lands almost
  entirely at $\tau\in\{0,\beta\}$ where $K\to1$, worth $\sim10^{-2}$ for broad
  poles on a $\pm20$ grid. `spevae` therefore supplies its own exact
  reconstruction rather than being charged for being untruncated.

### Peak counts need a resolution scale

A differential-evolution continuation of a two-peak spectrum that MaxEnt renders
with 3 maxima carried **52**. Raw peak counts measure how rough a solver is, not
whether the physics agrees. `compare(..., beta=...)` smooths every spectrum to
$\pi/\beta$ — the resolution limit — before counting peaks and matching them.
Aggregate distances stay unsmoothed, because roughness is a real property of an
answer and should not be hidden.

### Measured

**Synthetic, two Lorentzians, known answer** ($\beta=10$, 800 bins, 81 slices;
spevae = the 27-minute `fast`-preset checkpoint, complex residues, scored
through its own exact forward map):

```
method           L2 rel     Linf       W1   peak dw   peak dh   chi2/target
spevae           0.4978   0.2148   0.3187    0.0500    0.3393      4554
spevae-refined   0.5586   0.3511   0.1589    0.1700    0.1350       197
maxent           0.2184   0.1184   0.2363    0.0350    0.0921      17.5
deac             0.3861   0.3706   0.3416    0.1100    0.0694      54.1
```

All four rows agree on the structure (2 peaks, no gap, at resolution
$\pi/\beta$). The division of labour: MaxEnt is smoothest and wins
$L^2$/$L^\infty$; DEAC gets heights right; the amortised spevae pass (0.2 s)
places peaks well ($\Delta\omega = 0.05$) but underfits at this training
budget, and **latent refinement takes the best $W_1$ of any method** (0.159)
while pulling $\chi^2$/target from 4554 to 197. A note on bookkeeping: the
previous edition of this table charged spevae through a truncating grid
back-transform the package itself calls unfair; this one is scored through the
exact forward map, like every number the benchmark now emits.

**Real half-filled Hubbard** ($U=6$, $\beta=10$, $L=4$; 800 bins, 201 slices,
calibrated target 0.390): the same checkpoint, zero per-dataset work, returns a
Mott-gapped spectrum with Hubbard bands centred near $\pm U/2$, certified
$A \ge 0$ with genuinely complex residues, effective poles 12.7/16, and
$\chi^2 = 194$ — the honest signature of an amortised model that has not yet
been fine-tuned to this cell. This dataset is also what surfaced the
pinned-slice trap: PH symmetry fixes $G(0) = G(\beta) = 1/2$ to machine
precision ($\sigma \sim 10^{-17}$), and dividing by that "noise" produced
$\chi^2 \sim 10^{27}$ until pinned slices were excluded from the whitened
block (`MeasuredGreens.statistical_slices`) and reported as an
exact-constraint deviation instead.

**Real site-Holstein cell** ($n{=}1$, $\Omega{=}1$, $\beta{=}10$), which ships a
SmoQyDEAC reference (calibrated target 0.112):

```
method            norm      min A       chi2   target    ratio
spevae          1.0009  -3.51e-05    169.371   0.1116  1517.8
spevae-refined  0.9999  -2.09e-05      0.155   0.1116     1.39
maxent          0.9999  +1.71e-07      0.026   0.1116     0.23
deac            1.0000  +6.49e-05      0.072   0.1116     0.64
deac-ref        1.0000  +6.25e-05      0.109   0.1116     0.98
```

**The shipped reference sits at χ²/target = 0.98** — exactly on the calibrated
noise floor. Against the naive target of 1 it would have looked like a 9×
*overfit*, which is precisely the spurious "$\chi^2 < 1$ anomaly" the
predecessor project carried as an open blocker. This is independent confirmation
that `chi2_target` is the right reference value and 1 is not.

The two spevae rows are the amortised/refined split in action: a phonon cell
is far from the fast checkpoint's sweet spot (ratio 1518 in 0.3 s), and latent
refinement — weights frozen, prior intact — walks it to **1.39, essentially at
the noise floor**, with $\min A$ at the $10^{-5}$ level and agreement with the
reference ($W_1 = 0.095$) comparable to MaxEnt's (0.061) and DEAC's (0.054).

MaxEnt at 0.23 is genuinely fitting below the floor here: it has ~800 grid
degrees of freedom against 100 time slices whose covariance is barely
determined, and `chi2kink` does not protect against that when the whitener is
this ill-conditioned. Worth knowing before quoting a MaxEnt $\chi^2$ as evidence.

### Measured: the noise-regime map

Eight ground-truth spectra (regenerated from the v1 library's formulas
through the verified forward map — the library's own bins were measured
inconsistent with their truths at up to ~500σ and are unusable for truth
recovery), evaluated at two noise levels against MaxEnt on identical bins,
best of refined/consolidated per dataset, $W_1$:

```
                        s=1e-4, 1000 bins        s=1e-2, 100 bins
dataset                  ours    MaxEnt           ours    MaxEnt
gaussian_double         0.034     0.006          0.080     0.143
gapped_mott             0.055     0.004          0.097     0.108
gapped_semicircle       0.042     0.030          0.051     0.275
pseudogap               0.029     0.007          0.061     0.166
three_peak_hubbard      0.070     0.004          0.100     0.370
asymmetric_three_peak   0.088     0.006          0.172     0.291
fano                    0.038     0.008          0.043     0.154
two_close_peaks         0.008     0.002          0.017     0.038

W1 wins                   0/8                      8/8  (1.1x - 5.4x)
l2 wins                   0/8                      5/8
```

The crossover the supervised-learning literature predicts (Fournier *et
al.*), measured on this package's own consistent data: at pristine noise
MaxEnt is the right tool and this package is honest about it; at realistic
DQMC noise the learned prior carries — most decisively on hard band edges
(semicircle, 5.4×) and asymmetric line shapes (Fano, 3.6×, certified
positive).  Where we lag at low noise, the diagnostics say why: Gaussian
tails sit above the pole-ansatz floor ($\chi^2/{\rm tgt}$ 11–43 there,
1.5–2.3 on pole-friendly shapes), and the fast-tier (55-minute) checkpoint
under-resolves sharp multi-peak interiors — the `full` preset is the known
lever.

**Full-preset model, same battery** (β-universal: frozen-stream $W_1$
0.245 on β ∈ [2,20] vs the fast tier's 0.561 — flat across the range at a
~5% cost on the fast tier's home window; trained 25.7k steps total through
a kill-and-warm-restart, EMA final): the regime map is preserved and
sharpened — high-noise $W_1$ 8/8 vs MaxEnt with the largest margin grown to
6.9× (semicircle, refined 0.040 vs 0.275), and the two prior refinement
stalls (gapped_mott) healed (low-noise refined 0.055 → 0.025, χ²/tgt 1.85).
Per-dataset, the full model beats the fast-24k on 4/8 (low) and 5/8 (high) —
its real purchase is uniformity over β, not the β=10 benchmark row.

Protocol note from the same campaigns: consolidation's data-refit is a
*low-noise* instrument — with $\chi^2$ anchored to little information it can
degrade a good refined start (measured repeatedly at $s=10^{-2}$).  At high
noise, report the refined (prior-carried) spectrum, and consolidate with the
Matsubara polish only.

### Measured: which sharp features are data-demanded (parametric superresolution)

The :math:`e^{-\pi^2/\beta\Delta\omega}` suppression bounds *generic*
structure recovery, but a **parametric** feature carried by a few pole
parameters can be pinned far below :math:`\pi/\beta` when the SNR is high --
the way a line centre is located to a fraction of its width.  Whether that
happened is decidable per feature: ``sharpness_verdict`` refits the same pole
set at matched budgets with and without widths floored at the resolution
scale, and reads the likelihood ratio.  (Its own unit test caught a bug that
had silently weakened the smeared arm -- explicit floors are hypotheses and
are now honored exactly; the earlier ad-hoc verdicts were retracted.)

Measured on the regenerated truths at :math:`s=10^{-4}`, 1000 bins:

```
gapped_semicircle  sharp 1.39  smeared 3.60   n*dchi2 = +443  DATA-DEMANDED
pseudogap          sharp 2.27  smeared 2.21   n*dchi2 =  -12  undecidable
                   A(0): 0.082 under BOTH hypotheses (truth: 0)
```

The discrimination is physical: hard band edges are high-contrast features
that feed many Matsubara points coherently -- the data pins them below the
naive resolution.  The pseudogap cusp's sub-resolution content is
low-amplitude, so its depth beyond the measurable floor (:math:`A(0)
\approx 0.08` at :math:`\beta = 10`) is the prior's assertion.  The shipped
convention follows: the honest-resolution curve is the default display, and
the verdict promotes specific sharp features to data-backed claims where the
ratio says so.

### Measured: the per-feature ledger, and the SNR-aware display scale

``feature_ledger`` runs both hypothesis tests per live pole (existence:
remove-and-refit; sharpness: floor *this pole's* width at :math:`\pi/\beta`,
free the rest -- per-pole floors in the refit make this possible), at matched
budgets against the converged baseline.  On the two contested truths:

```
gapped_semicircle (11 poles): 5 sharp DATA-DEMANDED -- and every one is an
  edge pole (inner edges +-1: eps -1.39/-1.09/+1.40; outer edges +-4:
  eps -3.58/+3.53; n*dchi2 +18 to +375) -- while all 6 band-interior poles
  come back sharpness-undecidable (-12 to -0.4).  The ledger separates the
  band-top ripple from the genuine discontinuities pole by pole, with no
  human judgement in the loop.
pseudogap (10 poles): zero sharp-demanded (8 resolution-broad,
  2 undecidable) -- all visible structure is envelope, as the global
  verdict predicted.
```

Display scale: :math:`\pi/\beta` is the SNR-independent safe default, but the
effective resolution improves logarithmically with precision,
:math:`\Delta\omega_{\rm eff} \approx \pi^2/(\beta \ln \mathrm{SNR})`
(from the suppression law) -- 0.09 at this dataset's SNR, so a
:math:`\pi/2\beta` display is still conservative there.  The convention:
show the finest scale the noise floor justifies, with the ledger labelling
which visible features are measurements and which are the prior's assertions.

### The mirror pair positivity theorem, and its measured price

For a PH-symmetric mirror pair $(\pm\varepsilon, \gamma, a/2, \pm b/2)$ the
numerator polynomial of the pair's spectral density collapses to the even
quadratic

$$\mathcal{N}(\omega) = (a\gamma - b\varepsilon)\,\omega^2
  + (\varepsilon^2+\gamma^2)(a\gamma + b\varepsilon),$$

so the pair is non-negative on **all** of $\mathbb{R}$ **iff**
$|b| \le a\gamma/\varepsilon$ — exact and closed-form.  Imposing the smooth
sufficient bound $|b| \le a\gamma/\sqrt{\gamma^2+\varepsilon^2}$ inside the
mirror-constrained refit (`pair_bound=True`) makes every pair — hence the
total — **certified $A \ge 0$ by construction**, with the gap floor coming
out flat and strictly positive.

The price is measured, not hidden: on the gIssa Holstein cell the
$\chi^2$-optimal fit genuinely wants per-pair asymmetries far beyond the
bound ($|b| \sim 0.5a$ against a cap of a few percent of $a$ at
$\varepsilon/\gamma \sim 3$), staying globally positive through *cross-pair*
cancellation instead — per-pair positivity is sufficient but not necessary.
Measured at $\pi/\beta$: unconstrained mirror refit reaches
$\chi^2/{\rm tgt} = 0.77$ with a $-1.3\times10^{-3}$ dip; the pair-bounded
one reaches $12.6$ ($\chi^2$/dof $\approx 2.4$ in absolute terms) with a
flat $[+4\times10^{-4}, +2\times10^{-3}]$ gap floor.  Penalty-based
positivity sits in between and barely moves the dip across five decades of
weight — the dip is $\chi^2$-preferred, not a tuning accident.  All three
modes ship; the certificate and the $\chi^2$ ratio quantify the trade per
dataset, and the physics judgement of which to report belongs to the user.

## 9. What is *not* claimed

- **$\chi^2 \approx 1$ does not imply a correct $A(\omega)$.** It means the fit is
  consistent with the data at the noise level. Nothing more.
- **Ensemble spread is an uncertainty only if the encoder is alive.** A collapsed
  posterior gives a beautifully tight ensemble that means nothing. `active_units`
  is computed with every result and belongs in every table.
- **Agreement with MaxEnt or DEAC is agreement between methods**, not proof of
  correctness — none of them is ground truth. Disagreement is the more
  informative signal: it localises where the inversion is genuinely
  underdetermined.

---

## 10. Attribution

The pole/residue output ansatz with hardwired sum rule and zero-sum tail, the
tangent-substituted Gauss–Legendre forward map, the covariance-whitened $\chi^2$
calibrated to 1, the variance-weighted $G/G''/G''''$ negativity penalties, and
the two-stage pretrain/fine-tune protocol are due to **Benjamin Cohen-Stead**
(UT-Knoxville), via the first attempt in `../SPE-AC-VAE`. This package keeps
the ansatz, the calibration idea and the penalties, and replaces the
quadrature, the encoder, the decoder, the loss and the training protocol for
the reasons documented above.

---

## References

[anacont]: https://doi.org/10.1016/j.cpc.2022.108519
[fournier]: https://link.aps.org/doi/10.1103/PhysRevLett.124.056401
[nevanlinna]: https://link.aps.org/doi/10.1103/PhysRevLett.126.056402
[mpm]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.035154
[pes]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.075151
[spx]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.235143
[aaa]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.125139
[kondo]: https://link.aps.org/doi/10.1103/PhysRevB.104.235111
[bias]: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.043082
[yoon]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.245101
[ae]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.115119
[deac]: https://scipost.org/SciPostPhysCodeb.39

**Machine learning for analytic continuation**

- R. Fournier, L. Wang, O. V. Yazyev, Q. Wu, *Artificial Neural Network Approach
  to the Analytic Continuation Problem*, [PRL **124**, 056401 (2020)][fournier].
  Supervised ANN; matches MaxEnt at low noise, substantially better at high
  noise, ~1000× faster.
- H. Yoon, J.-H. Sim, M. J. Han, *Analytic continuation via domain-knowledge free
  machine learning*, [PRB **98**, 245101 (2018)][yoon].
- M. Kliczkowski, L. Keyes, S. Roy, T. Paiva, M. Randeria, N. Trivedi,
  M. M. Maśka, *Autoencoder-based analytic continuation method for strongly
  correlated quantum systems*, [PRB **110**, 115119 (2024)][ae]
  (Editors' Suggestion). The closest published prior work: a **VAE-type
  encoder** (its supplement is explicit — a `DenseVar(μ, σ)` bottleneck citing
  Kingma–Welling) feeding a *fixed, non-trainable kernel-matrix decoder* on an
  $\omega$ grid of 250–1000 points, trained by data-space $\chi^2$ + moment
  regularisers at one fixed $\beta$ with $M = 40$ $\tau$-points. Equals or
  beats MaxEnt, most clearly at high noise. This package differs in the
  representation (a learned decoder emitting a grid-free pole set with the
  constraints hardwired), variable $(\beta, \Delta\tau)$, the explicit ELBO
  with free bits, and latent-ensemble uncertainty — to our knowledge no
  published work combines a VAE with a pole-expansion decoder.
- R. Zhang, M. E. Merkel, S. Beck, C. Ederer, *Training biases in machine
  learning for the analytic continuation of quantum many-body Green's
  functions*, [PRR **4**, 043082 (2022)][bias]. **The** cautionary reference on
  prior inheritance; also the source of the MC-dropout uncertainty idea.
- C. Miles, M. R. Carbone, E. J. Sturm, D. Lu, A. Weichselbaum, K. Barros,
  R. M. Konik, *Machine learning of Kondo physics using variational
  autoencoders and symbolic regression*, [PRB **104**, 235111 (2021)][kondo].
  VAE latent coordinates align with the Kondo temperature and particle–hole
  asymmetry.

**Pole representations and analytic structure**

- J. Fei, C.-N. Yeh, E. Gull, *Nevanlinna Analytical Continuation*,
  [PRL **126**, 056402 (2021)][nevanlinna]. Respecting the analytic structure
  gives intrinsically positive, normalised spectra.
- L. Zhang, E. Gull, *Minimal pole representation and controlled analytic
  continuation of Matsubara response functions*,
  [PRB **110**, 035154 (2024)][mpm]. Noise sets a hard budget on the
  recoverable pole count; codes: MiniPole / PronyAC (Green-Phys).
- L. Huang, S. Liang, *Stochastic pole expansion method for analytic
  continuation of the Green's function*, [PRB **108**, 235143 (2023)][spx].
- L. Huang, *Barycentric rational function approximation made simple*,
  [PRB **111**, 125139 (2025)][aaa].
- Z. Huang, E. Gull, L. Lin, *Robust analytic continuation of Green's functions
  via projection, pole estimation, and semidefinite relaxation*,
  [PRB **107**, 075151 (2023)][pes]. "Denoise onto the causal manifold, then
  fit poles" — the amortised encoder here is the learned version of that
  projection.

**Benchmarks**

- J. C. Neuhaus, N. Nichols, D. Banerjee, B. Cohen-Stead, T. A. Maier,
  A. Del Maestro, S. Johnston, *SmoQyDEAC.jl*,
  [SciPost Phys. Codebases **39** (2024)][deac].
- J. Kaufmann and K. Held, *ana_cont: Python package for analytic
  continuation*, [Comput. Phys. Commun. **282**, 108519 (2023)][anacont].
  Vendored under `src/spevae/_vendor/ana_cont` (MIT); see its `VENDORED.md`.
