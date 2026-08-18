# spevae

Analytic continuation of noisy imaginary-time Green's functions to real-frequency
spectral functions, using a physics-constrained variational autoencoder whose
decoder emits a **simple pole expansion** rather than a frequency grid.

```
G(τ)  from DQMC   ──▶   encoder   ──▶   z   ──▶   decoder   ──▶   {ε_p, γ_p, a_p}
   any β, any Δτ                                                        │
                                                                        ▼
                                                     A(ω) = Σ_p a_p · Lorentzian(ε_p, γ_p)
                                                     causal · positive · normalised · 1/ω²
```

## What is guaranteed by construction

Not by a penalty, not by training — these hold identically for every parameter
value, at initialisation and out of distribution, and are asserted in
`tests/test_model.py`:

| property | mechanism |
|---|---|
| **causality** — all poles in the lower half plane | `γ = γ_min + softplus(·)` |
| **positive weights**, $a_p > 0$ | `softmax` |
| **sum rule**, $\sum_p a_p = 1$ | `softmax` normalisation |
| **zero-sum tail**, $\sum_p \operatorname{Im}r_p = 0$ | identically zero in the default mode |
| **$A(\omega) \ge 0$ pointwise** | positive mixture of Lorentzians |
| **$A \sim c_2/\omega^2$**, $c_2 > 0$ | follows from $\sum_p b_p = 0$ |
| **$G \equiv 0 \Rightarrow A \equiv 0$** | residues scaled by the measured norm |
| **$A[cG] = c\,A[G]$** exactly | encoder sees $G/N$; decoder rescales by $N$ |
| **width floor** $\gamma \ge 0.1\,\pi/\beta$ | the Matsubara resolution limit |

## Why one model covers every (β, Δτ)

With $x=\tau/\beta$ and $\nu=\beta\omega$ the fermionic kernel becomes

$$K(x,\nu)=\frac{1}{e^{x\nu}+e^{(x-1)\nu}},$$

which contains no $\beta$ at all. The inverse problem is a single problem, not a
family indexed by temperature, so one set of weights serves every simulation
cell. Combined with an attention encoder over time slices, the number of time
slices is just a sequence length. See [`docs/DESIGN.md`](docs/DESIGN.md) §3.

## Install

```bash
conda env create -f environment.yml
conda activate spevae
pip install -e ".[dev]"
pytest            # 132 tests: physics, symmetries, calibration, benchmark
```

## Use

```python
import torch
from spevae.models import SPEVAE

model = SPEVAE(latent_dim=8, num_poles=16, omega_c=20.0)
# ... train, or load a checkpoint ...

G    = torch.tensor(gtau)            # (n_bins, L_tau), tau = 0 … beta inclusive
out  = model.continue_(G, beta=10.0, omega=torch.linspace(-8, 8, 2001),
                       n_samples=200)

out["A"]      # (n_bins, 2001) spectral function
out["A_std"]  # ensemble spread — where the data stops determining the answer
out["poles"]  # grid-free: evaluate A anywhere afterwards, at any resolution
```

`out["poles"]` is the real output. `A(ω)` on a grid is a convenience;
`out["poles"].spectral(any_omega)` re-evaluates at arbitrary resolution with no
network call.

### Train

```bash
python scripts/smoke_train.py --steps 6000 --batch-size 48   # ~10 min, CPU
```

Training data is generated on the fly from nine spectral families across wide
ranges of β, Δτ, bandwidth and noise, so there is no fixed epoch and no
overfitting to a finite training set — longer runs and larger models buy accuracy
monotonically.

## Benchmarking

```bash
python scripts/benchmark.py --synthetic --checkpoint out/model.pt         # known answer
python scripts/benchmark.py --holstein 19 1 5 --checkpoint out/model.pt   # bundled cube
python scripts/benchmark.py --bins mybins.txt --beta 8.0 --figure cmp.png # your data
```

`--bins` takes a whitespace-delimited `(N_bins × L_tau)` file with
`τ = 0 … β` inclusive. The sign convention is **detected**, not assumed — the
sum rule $G(0)+G(\beta)=1$ is satisfied by exactly one of the two signs — and
the run aborts rather than producing a confident, wrong spectrum if neither
works.

Three solvers, one interface, one forward map for scoring, one set of metrics:

- **MaxEnt** — [`ana_cont`][anacont], vendored under `src/spevae/_vendor` (MIT).
  Nothing to install.
- **SmoQyDEAC** — [`SmoQyDEAC.jl`][deac] through a Julia subprocess. Needs Julia
  with `SmoQyDEAC` installed; point `$SPEVAE_JULIA_PROJECT` at the environment
  (default `@deac`).
- **spevae** — this package.

All three define the *same* kernel — there is a test asserting it, because a
sign or normalisation mismatch produces plausible-looking but incomparable
spectra that nothing downstream would catch. Every result is scored through one
forward map against one whitened covariance, so the $\chi^2$ column means the
same thing in every row.

The three fail differently — MaxEnt broadens sharp structure, DEAC returns rough
spectra, this package is bounded by its training distribution — which is exactly
why agreeing matters. Where all three agree, the feature is in the data; where
they disagree, the inversion is underdetermined *there*, and the disagreement
localises it.

## Layout

```
src/spevae/
├── physics/            exact, network-free, verified to ~1e-9
│   ├── kernel.py       dimensionless fermionic kernel
│   ├── poles.py        SPE algebra: A(ω), tails, moments, diagnostics
│   └── forward.py      quadrature + poles → G(τ), Matsubara transform
├── models/
│   ├── constraints.py  every guarantee in the table above
│   ├── encoder.py      transformer over τ-slices; any grid, any β
│   ├── decoder.py      DETR-style pole-set decoder
│   └── vae.py          assembly, homogeneity, inference API
├── data/               spectral families, DQMC-like noise, real-DQMC readers
├── losses.py           χ², Wasserstein, free-bits KL, noise consistency
├── metrics.py          peaks, heights, gaps — what a CM reader checks
├── training/           Stage 1 pretraining
├── benchmark/          MaxEnt + SmoQyDEAC + spevae, one comparator
└── _vendor/ana_cont    vendored MaxEnt solver (MIT)
tests/                  physics invariants, symmetries, calibration, benchmark
docs/DESIGN.md          rationale, measurements, literature
scripts/                smoke_train.py, benchmark.py
```

[anacont]: https://doi.org/10.1016/j.cpc.2022.108519
[deac]: https://scipost.org/SciPostPhysCodeb.39

## Reading the results honestly

Three things the package computes and reports, and which any write-up should
carry:

- **`chi2` ≈ 1 does not mean the spectrum is right.** It means the fit is
  consistent with the data at the noise level. The kernel maps a large family of
  spectra to indistinguishable $G(\tau)$; that is the whole content of the
  ill-posedness.
- **`active_units`** — a collapsed encoder produces a beautifully tight ensemble
  that means nothing. Report it next to any uncertainty band.
- **`effective_poles`** — the participation ratio $1/\sum_p a_p^2$, i.e. how many
  poles the fit actually spends. `num_poles` is an upper bound, not a result.

Under Ledoit–Wolf shrinkage the perfect-fit $\chi^2$ is **below 1**;
`spevae.data.chi2_target` returns the correct value for any whitener, and the
benchmark table reports the ratio. Comparing against 1 instead is how a spurious
"$\chi^2 < 1$ anomaly" appears — on a real site-Holstein cell the calibrated
target is **0.087**, and the SmoQyDEAC reference shipped with that cell sits at
0.99 of it.

## Status

Implemented and verified: physics core, model, constraint layer, synthetic data
generator, losses, metrics, Stage-1 training, real-DQMC readers, and the
three-way benchmark harness with plotting. 132 tests. See `docs/DESIGN.md` for what is
measured and how.

Not yet wired up: Stage-2 per-dataset fine-tuning, plotting, and readers for the
SmoQy simulation directories and the coupling-resolved Holstein cube.

## Attribution

The pole/residue ansatz with hardwired sum rule and zero-sum tail, the
covariance-whitened $\chi^2$ calibrated to 1, and the two-stage protocol are due
to **Benjamin Cohen-Stead** (UT-Knoxville), via the predecessor project in
`../SPE-AC-VAE`. See `docs/DESIGN.md` §10 and the reference list there.

## Licence

MIT.
