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
  quality ([Ying, PRB 110, 035154 (2024)][ying]), stochastic pole expansion
  resolves sharp and smooth features together ([Huang, PRB 108, 235143
  (2023)][spx]), and barycentric/AAA rational fits do the same very cheaply
  ([Huang *et al.*, PRB 111, 125139 (2025)][aaa]).

### The residue-mode choice

With $b_p \equiv 0$ (`residue_mode="real"`, the default) the expansion is a
**positive mixture of Lorentzians**. That single choice delivers, simultaneously
and by construction:

| requirement | how |
|---|---|
| causality | $\gamma_p > 0 \Rightarrow$ all poles in the lower half plane |
| $A(\omega) \ge 0$ pointwise | positive combination of positive Lorentzians |
| $\int A = 1$ | $\sum_p a_p = 1$ |
| $\sum_p \operatorname{Im} r_p = 0$ | trivially, $b\equiv 0$ |
| $A \sim c_2/\omega^2$, $c_2>0$ | follows from $\sum_p b_p = 0$ |

This is the discrete analogue of the observation that respecting the Nevanlinna
analytic structure yields intrinsically positive and normalised spectra
([Fei, Yeh & Gull, PRL 126, 056402 (2021)][nevanlinna]).

`residue_mode="complex"` frees $b_p$ subject to $\sum_p b_p = 0$ and
$|b_p| \le a_p$. It buys asymmetric (Fano-like) line shapes at the cost of
pointwise positivity: a single pole with $b_p\neq0$ has

$$\min_\omega A_p = \frac{a_p - \sqrt{a_p^2+b_p^2}}{2\pi\gamma_p} < 0,$$

diverging as $\gamma_p \to 0$. Positivity of the total then relies on cross-pole
cancellation, which $\sum_p b_p = 0$ enforces only in the integral, not
pointwise. **This was the dominant pathology of the first attempt**, where a
systematic audit found every divergent run had $\min_p\gamma_p < 0.06$ while
every healthy run had $\gamma_p > 0.23$ — and no term in the loss could see it,
because a near-delta pole still produces a perfectly smooth $G(\tau)$. Real
residues remove the mechanism rather than penalising the symptom. Use the
complex mode only for correlators that are genuinely not positive-definite.

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
until the network saturates the information in the data.

**Stage 2 — optional per-dataset fine-tuning** on the user's own bins and
covariance, using $\chi^2$ alone. Useful when a dataset sits outside the training
prior; it should be anchored to the pretrained weights, because unanchored
$\chi^2$ minimisation is exactly the procedure Stage 1 exists to replace.

### The objective

| term | purpose |
|---|---|
| $\log(1+\chi^2)$ | data fit, covariance-whitened |
| $W_1(A_{\rm pred}, A_{\rm true})$ | **peak position** |
| relative $L^2$ | peak heights and shape |
| KL with free bits | latent regularisation without collapse |
| symmetric-KL consistency | noise invariance |

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

*$\log(1+\chi^2)$ rather than $\chi^2$.* The raw value spans five orders of
magnitude between initialisation ($\sim10^5$) and convergence ($\sim1$); used
directly it swamps every other term early and contributes nothing later. `log1p`
is monotone — same minimiser — and its gradient $1/(1+\chi^2)$ is exactly the
self-scaling needed.

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
([Rigo & Mitchell, PRB 104, 235111 (2021)][kondo]).

### Training-set bias

A supervised network inherits the prior of what it was trained on; this is the
documented failure mode of the whole approach
([Sim *et al.*, PRR 4, 043082 (2022)][bias]). A network trained on single
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

**Synthetic, two Lorentzians, known answer** ($\beta=10$, 800 bins, 81 slices):

```
method       L2 rel     Linf       W1   peak dw   peak dh   chi2/target
spevae       0.4922   0.2420   0.3020    0.1450    0.3887      9447
maxent       0.2184   0.1184   0.2363    0.0350    0.0921        17.5
deac         0.3829   0.4446   0.2969    0.0100    0.0589        58.3
```

Exactly the expected division of labour: DEAC places peaks best (Δω = 0.01) and
gets heights right (6%); MaxEnt is smoothest and wins on $L^2$/$L^\infty$;
`spevae` at this checkpoint is a 9-minute smoke model and is honestly the worst
of the three, with a $\chi^2$ ratio saying so out loud.

**Real site-Holstein cell** ($n{=}1$, $\Omega{=}1$, $\beta{=}10$), which ships a
SmoQyDEAC reference:

```
method        norm      min A       chi2   target    ratio
maxent      0.9999  +1.71e-07      0.005   0.0867     0.05
deac        1.0000  +2.34e-04      0.073   0.0867     0.85
deac-ref    1.0000  +6.25e-05      0.086   0.0867     0.99
```

**The shipped reference sits at χ²/target = 0.99** — exactly on the calibrated
noise floor. Against the naive target of 1 it would have looked like a 12×
*overfit*, which is precisely the spurious "$\chi^2 < 1$ anomaly" the
predecessor project carried as an open blocker. This is independent confirmation
that `chi2_target` is the right reference value and 1 is not.

MaxEnt at 0.05 is genuinely fitting below the floor here: it has ~800 grid
degrees of freedom against 100 time slices whose covariance is barely
determined, and `chi2kink` does not protect against that when the whitener is
this ill-conditioned. Worth knowing before quoting a MaxEnt $\chi^2$ as evidence.

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
calibrated to 1, and the two-stage pretrain/fine-tune protocol are due to
**Benjamin Cohen-Stead** (UT-Knoxville), via the first attempt in
`../SPE-AC-VAE`. This package keeps the ansatz and the calibration idea and
replaces the quadrature, the encoder, the decoder, the loss and the training
protocol for the reasons documented above.

---

## References

[anacont]: https://doi.org/10.1016/j.cpc.2022.108519
[fournier]: https://link.aps.org/doi/10.1103/PhysRevLett.124.056401
[nevanlinna]: https://link.aps.org/doi/10.1103/PhysRevLett.126.056402
[ying]: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.035154
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
- *Autoencoder-based analytic continuation method for strongly correlated quantum
  systems*, [PRB **110**, 115119 (2024)][ae]. Unsupervised autoencoder; recovers
  sharp features MaxEnt loses.
- J.-H. Sim *et al.*, *Training biases in machine learning for the analytic
  continuation of quantum many-body Green's functions*,
  [PRR **4**, 043082 (2022)][bias]. **The** cautionary reference on prior
  inheritance.
- C. Rigo, A. K. Mitchell, *Machine learning of Kondo physics using variational
  autoencoders and symbolic regression*, [PRB **104**, 235111 (2021)][kondo].
  $\beta$-VAE latent coordinates align with $\log T_K$ and particle–hole
  asymmetry.

**Pole representations and analytic structure**

- J. Fei, C.-N. Yeh, E. Gull, *Nevanlinna Analytical Continuation*,
  [PRL **126**, 056402 (2021)][nevanlinna]. Respecting the analytic structure
  gives intrinsically positive, normalised spectra.
- L. Ying, *Minimal pole representation and controlled analytic continuation of
  Matsubara response functions*, [PRB **110**, 035154 (2024)][ying].
- L. Huang, *Stochastic pole expansion method for analytic continuation of the
  Green's function*, [PRB **108**, 235143 (2023)][spx].
- L. Huang *et al.*, *Barycentric rational function approximation made simple*,
  [PRB **111**, 125139 (2025)][aaa].

**Benchmarks**

- J. C. Neuhaus, N. Nichols, D. Banerjee, B. Cohen-Stead, T. A. Maier,
  A. Del Maestro, S. Johnston, *SmoQyDEAC.jl*,
  [SciPost Phys. Codebases **39** (2024)][deac].
- J. Kaufmann and K. Held, *ana_cont: Python package for analytic
  continuation*, [Comput. Phys. Commun. **282**, 108519 (2023)][anacont].
  Vendored under `src/spevae/_vendor/ana_cont` (MIT); see its `VENDORED.md`.
