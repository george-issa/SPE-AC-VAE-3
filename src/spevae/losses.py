r"""Training objectives.

Reconstruction (:math:`\chi^2`)
-------------------------------
:func:`chi2` whitens with the factorised covariance :math:`C = D R D`, so a
perfect fit gives :math:`\mathbb{E}[\chi^2] = 1` (or the shrinkage-corrected
value from :func:`~spevae.data.noise.expected_chi2` when :math:`R` has been
regularised).  It is the only term available on real data, where no ground truth
exists.

**It is also not sufficient.**  :math:`\chi^2 \approx 1` says the fit is
consistent with the data at the noise level; it does not say the spectrum is
right, because the kernel maps a large family of spectra to indistinguishable
:math:`G(\tau)`.  That is the whole content of the ill-posedness, and it is why
this module also provides supervised terms for pretraining, where the answer is
known.

Spectral distance (Wasserstein)
-------------------------------
:func:`spectral_w1` is the primary supervised term.  For two normalised positive
measures the 1-Wasserstein distance is
:math:`W_1 = \int |F_{\rm pred}(\nu) - F_{\rm true}(\nu)|\,d\nu` with :math:`F`
the cumulative distribution -- exact, cheap on the quadrature grid, and
differentiable.

Why not plain :math:`L^2` on :math:`A(\omega)`?  Because :math:`L^2` is *flat in
peak position* once two peaks stop overlapping: moving a predicted peak from
:math:`\omega=3` to :math:`\omega=2` changes nothing until it reaches the true
peak at :math:`\omega=1`.  The predecessor project hit this directly -- a
documented set of experiments in which no amount of smoothness regularisation
moved a misplaced peak, because smoothness penalises curvature, not location.
:math:`W_1` grows *linearly* with displacement and therefore always points
downhill towards the right position.  :math:`L^2` is still useful for peak
heights and shape, so the two are combined rather than substituted.

KL with free bits
-----------------
:func:`kl_divergence` implements the free-bits objective (Kingma et al., 2016):
each latent dimension is exempt from the KL penalty below a floor
:math:`\lambda`.  Posterior collapse happens when the decoder can fit the data
mean from a constant code, so that the encoder's marginal contribution is worth
less than its KL cost; free bits removes the incentive by making the first
:math:`\lambda` nats per dimension free.  A collapsed run can still show an
excellent :math:`\chi^2`, so this is a structural fix rather than something to
detect after the fact -- see :func:`~spevae.models.vae.active_units`.

Noise consistency
-----------------
:func:`noise_consistency` penalises the disagreement between the posteriors of
two independent noise realisations of the same underlying spectrum.  This states
directly, in the objective, the invariance the user cares about: *the answer is a
property of the physics, not of the random seed*.  It also gives the latent
space a reason to represent the signal and discard the noise, which is exactly
the "meaningful latent representation" goal -- the same mechanism by which
:math:`\beta`-VAEs on Kondo spectra recovered :math:`\log T_K` and the
particle--hole asymmetry as individual latent coordinates
(Miles *et al.*, *PRB* **104**, 235111 (2021)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .physics.forward import SpectralQuadrature

__all__ = [
    "floatify",
    "chi2",
    "weighted_mse",
    "spectral_w1",
    "spectral_l2",
    "spectral_multiscale_l2",
    "spectral_positivity",
    "spectral_ripple",
    "weight_entropy",
    "kl_divergence",
    "noise_consistency",
    "width_barrier",
    "LossWeights",
    "pretrain_loss",
    "negativity_stencil_weights",
    "negativity_penalties",
]


def floatify(parts: dict) -> dict[str, float]:
    """Convert a loss-parts dict of 0-d tensors to Python floats.

    Call this only when the numbers are about to be logged or plotted: each
    conversion synchronises with the accelerator.
    """
    return {k: (float(v) if torch.is_tensor(v) else v) for k, v in parts.items()}


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #

def chi2(
    G_pred: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    n_independent: int | None = None,
    slices: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Covariance-whitened reconstruction loss, calibrated to 1 at the noise floor.

    .. math::
        \chi^2 = \frac{1}{L}\,
        \big\| R^{-1/2} D^{-1} (G_{\rm pred} - G_{\rm obs}) \big\|^2 ,
        \qquad C = D R D .

    Parameters
    ----------
    G_pred, G_obs, sigma
        ``(B, L)``.
    inv_sqrt_R
        ``(L, L)`` whitener for the correlation matrix.  ``None`` treats the
        slices as independent, which *overstates* the number of constraints when
        the noise is correlated -- acceptable only when that is genuinely true.
    n_independent
        Use only the leading ``n_independent`` time slices.  On an
        endpoint-inclusive grid of *measured* bins, :math:`\tau=\beta` is
        linearly dependent on :math:`\tau=0` through the sum rule, making the
        full covariance singular; pass ``L-1`` there, **together with a whitener
        computed on that leading block** (:mod:`spevae.data.real` does).  A
        whitener of any other size raises, because slicing
        :math:`R^{-1/2}` is not the whitener of the sliced block.  Synthetic
        batches need neither: their noise is full-rank over all ``L`` slices.
    slices
        Explicit column indices to score -- what
        :meth:`spevae.data.MeasuredGreens.covariance` returns as ``"slices"``.
        Takes precedence over ``n_independent``, and is the correct choice on
        measured data, where exactly-pinned slices (PH-symmetric endpoints)
        must be excluded rather than divided by their zero variance.
    """
    if slices is not None:
        if inv_sqrt_R is not None and inv_sqrt_R.shape[-1] != slices.numel():
            raise ValueError(
                f"inv_sqrt_R is {tuple(inv_sqrt_R.shape)} but {slices.numel()} "
                f"slices were selected; the whitener must be computed on "
                f"exactly those columns.")
        d = (G_pred[..., slices] - G_obs[..., slices]) / sigma[..., slices]
        if inv_sqrt_R is not None:
            d = d @ inv_sqrt_R.transpose(-1, -2)
        return (d * d).sum(-1).mean() / d.shape[-1]

    d = (G_pred - G_obs) / sigma
    if n_independent is not None:
        d = d[:, :n_independent]
        if inv_sqrt_R is not None and inv_sqrt_R.shape[-1] != n_independent:
            # Slicing a whitener computed for the full grid is NOT the whitener
            # of the sliced block: (R^{-1/2})[:n,:n] != (R[:n,:n])^{-1/2}, and
            # the mismatch silently mis-calibrates chi^2 by up to a few percent.
            # Compute inv_sqrt_R on the block you score (data/real.py does).
            raise ValueError(
                f"inv_sqrt_R is {tuple(inv_sqrt_R.shape)} but n_independent="
                f"{n_independent}; pass a whitener computed for exactly the "
                f"leading {n_independent}x{n_independent} block."
            )
    if inv_sqrt_R is not None:
        d = d @ inv_sqrt_R.transpose(-1, -2)
    return (d * d).sum(-1).mean() / d.shape[-1]


def weighted_mse(G_pred: torch.Tensor, G_obs: torch.Tensor,
                 sigma: torch.Tensor | None = None) -> torch.Tensor:
    """Plain (optionally inverse-variance weighted) mean squared error on ``G``."""
    d = G_pred - G_obs
    if sigma is not None:
        d = d / sigma
    return (d * d).mean()


# --------------------------------------------------------------------------- #
# Spectral distances
# --------------------------------------------------------------------------- #

def _cdf(A: torch.Tensor, quad: SpectralQuadrature) -> torch.Tensor:
    r"""Normalised cumulative :math:`\int_{-\infty}^{\nu} A / \int A`, on the core panel.

    Two points of care:

    * The reciprocal tail panels are not ordered in :math:`\nu`, so the running
      integral is formed on the uniform core -- which carries essentially all of
      the weight for any spectrum inside the frequency cutoff.
    * The cumulative is normalised to end at 1.  :math:`W_1` is a metric between
      measures *of equal mass*; the predicted mass is the data's own
      :math:`G(0)+G(\beta)` and so differs from the target's by the noise on
      those two slices.  Left unnormalised, that :math:`\sim 10^{-4}` mismatch
      would contribute :math:`\sim 10^{-4}\cdot 2\nu_c` to the loss -- comparable
      to a converged :math:`W_1`, i.e. pure noise dominating the signal.  The
      overall scale is enforced exactly elsewhere, so this term is free to
      compare shapes only.
    """
    core = A[:, : quad.n_core] * quad.weights[: quad.n_core]
    return torch.cumsum(core, dim=-1) / core.sum(-1, keepdim=True).clamp_min(1e-30)


def spectral_w1(A_pred: torch.Tensor, A_true: torch.Tensor,
                quad: SpectralQuadrature, beta: float | None = None) -> torch.Tensor:
    r"""1-Wasserstein distance between two spectra.

    .. math:: W_1 = \int \big|F_{\rm pred}(\nu) - F_{\rm true}(\nu)\big|\, d\nu

    Linear in peak displacement, so its gradient points towards the correct
    position even when predicted and true peaks do not overlap at all.

    Pass ``beta`` to get the answer in **physical** frequency, :math:`W_1/\beta`.
    Always do so when averaging across cells: in dimensionless units the same
    physical error counts ten times more at :math:`\beta=20` than at
    :math:`\beta=2`, which would quietly turn the objective into "get the cold
    cells right".  It also makes the number directly readable as a typical
    peak-placement error in units of :math:`\omega`.
    """
    diff = (_cdf(A_pred, quad) - _cdf(A_true, quad)).abs()
    w1 = (diff * quad.weights[: quad.n_core]).sum(-1).mean()
    return w1 if beta is None else w1 / float(beta)


def spectral_l2(A_pred: torch.Tensor, A_true: torch.Tensor,
                quad: SpectralQuadrature, relative: bool = False) -> torch.Tensor:
    r""":math:`\int |A_{\rm pred} - A_{\rm true}|^2 d\nu` -- sensitive to peak *heights*.

    ``relative=True`` divides by :math:`\int A_{\rm true}^2 d\nu`, which makes
    the value dimensionless and comparable across :math:`\beta` (the absolute
    form carries a factor :math:`1/\beta`, since :math:`\tilde A = A/\beta`).
    """
    d = A_pred - A_true
    num = quad.integrate(d * d)
    if not relative:
        return num.mean()
    return (num / quad.integrate(A_true * A_true).clamp_min(1e-30)).mean()


def spectral_multiscale_l2(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    quad: SpectralQuadrature,
    scales: tuple[float, ...] = (1.0, 4.0, 16.0),
) -> torch.Tensor:
    r"""Mean :math:`L^2` between progressively broadened versions of both spectra.

    Comparing only at full resolution makes the loss dominated by the sharpest
    features, where a pole expansion and a Gaussian target can never agree
    exactly.  Broadening both sides equally before comparing adds a term that
    sees the coarse envelope -- where the two *should* agree -- so the gradient
    is informative about overall shape as well as fine structure.

    ``scales`` are broadening widths as multiples of the quadrature spacing.
    """
    from .data.spectra import broaden

    n = quad.n_core
    p, t = A_pred[:, :n], A_true[:, :n]
    w = quad.weights[:n]
    total = A_pred.new_zeros(())
    for s in scales:
        ps = p if s <= 1.0 else broaden(p, quad.h, s * quad.h)
        ts = t if s <= 1.0 else broaden(t, quad.h, s * quad.h)
        d = ps - ts
        total = total + ((d * d) * w).sum(-1).mean()
    return total / len(scales)


# --------------------------------------------------------------------------- #
# Latent-space terms
# --------------------------------------------------------------------------- #

def spectral_ripple(A: torch.Tensor, quad: SpectralQuadrature,
                    sigma: float = 3.141592653589793) -> torch.Tensor:
    r"""Sub-thermal oscillation content: :math:`\int |A - \tilde A_\sigma| d\nu / \int A d\nu`.

    :math:`\tilde A_\sigma` is :math:`A` Gaussian-broadened at the first
    Matsubara scale :math:`\sigma = \pi` (dimensionless), the finest structure
    :math:`\tau`-data at temperature :math:`1/\beta` can resolve.  A smooth
    band scores near zero; a comb of narrow poles tiling that band scores its
    oscillation fraction.  This is the *diagnostic* for the pole-comb artefact
    ("ripple"): structure at this scale is almost invisible to :math:`\chi^2`
    (the kernel suppresses it exponentially -- that is the ill-posedness) and
    largely cancels in the :math:`W_1` CDF, so it must be measured directly or
    it will not be seen at all.
    """
    from .data.spectra import broaden

    n = quad.n_core
    core = A[:, :n]
    smooth = broaden(core, quad.h, sigma)
    w = quad.weights[:n]
    num = ((core - smooth).abs() * w).sum(-1)
    den = (core.abs() * w).sum(-1).clamp_min(1e-30)
    return (num / den).mean()


def weight_entropy(a: torch.Tensor) -> torch.Tensor:
    r"""Shannon entropy of the unit-normalised pole weights, batch-averaged.

    :math:`H = -\sum_p \hat a_p \ln \hat a_p` with :math:`\hat a = a/\sum a`.
    As a penalty this is mild pressure toward spending *few* poles -- the
    minimal-pole philosophy (Zhang & Gull) as a soft prior rather than a hard
    truncation.  :math:`H` ranges from 0 (one pole) to :math:`\ln P`.
    """
    a_unit = a / a.sum(-1, keepdim=True).clamp_min(1e-30)
    return -(a_unit * a_unit.clamp_min(1e-12).log()).sum(-1).mean()


def spectral_positivity(A_pred: torch.Tensor, quad: SpectralQuadrature,
                        relative: bool = True) -> torch.Tensor:
    r"""Penalty on negative excursions, :math:`\int \mathrm{ReLU}(-A)^2 d\nu`.

    Needed **only** in ``residue_mode="complex"`` (the default), where a pole
    with :math:`b_p \neq 0` has :math:`\min_\omega A_p < 0` and positivity of
    the total relies on cross-pole cancellation.  In the real-residue mode the
    integrand is identically zero, and the callers skip it there.

    It is a penalty, not a guarantee -- that is the trade the complex mode makes,
    and the honest way to report it is to print :math:`\min_\omega A` next to
    any spectrum rather than to claim positivity.  ``relative=True`` divides by
    :math:`\int A^2`, making the value dimensionless and comparable across
    :math:`\beta`.
    """
    neg = torch.relu(-A_pred)
    num = quad.integrate(neg * neg)
    if not relative:
        return num.mean()
    return (num / quad.integrate(A_pred * A_pred).clamp_min(1e-30)).mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor,
                  free_bits: float = 0.0) -> torch.Tensor:
    r"""KL to a standard normal prior, with a per-dimension free-bits floor.

    .. math::
        \mathcal{L}_{\rm KL} = \sum_u \max\!\big(\lambda,\;
        \tfrac{1}{B}\sum_i \mathrm{KL}\big[q(z_u\mid x_i)\,\|\,p(z_u)\big]\big)

    With :math:`\lambda > 0` a dimension carrying less than :math:`\lambda` nats
    contributes a constant, so there is no gradient pushing it further towards
    the prior: the encoder can hold information for free up to that budget.  This
    is the standard structural remedy for posterior collapse, and it costs one
    hyperparameter rather than a schedule.
    """
    logvar = logvar.clamp(-20.0, 20.0)
    per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)     # (B, Lz)
    per_dim = per_dim.mean(dim=0)                                  # (Lz,)
    if free_bits > 0.0:
        per_dim = torch.clamp(per_dim, min=float(free_bits))
    return per_dim.sum()


def noise_consistency(
    mu_a: torch.Tensor,
    logvar_a: torch.Tensor,
    mu_b: torch.Tensor,
    logvar_b: torch.Tensor,
) -> torch.Tensor:
    r"""Symmetric KL between the posteriors of two noise realisations of one spectrum.

    Zero exactly when the encoder gives both realisations the same posterior,
    i.e. when it has learned to read the physics and ignore the seed.  Symmetric
    KL rather than :math:`\|\mu_a-\mu_b\|^2` because it also matches the
    *widths*: two realisations should agree on how uncertain they are, not just
    on the central value.
    """
    la = logvar_a.clamp(-20.0, 20.0)
    lb = logvar_b.clamp(-20.0, 20.0)
    va, vb = la.exp(), lb.exp()
    d2 = (mu_a - mu_b).pow(2)
    kl_ab = 0.5 * ((va + d2) / vb + lb - la - 1.0)
    kl_ba = 0.5 * ((vb + d2) / va + la - lb - 1.0)
    return (kl_ab + kl_ba).sum(-1).mean()


def width_barrier(gamma: torch.Tensor, gamma_min: float) -> torch.Tensor:
    r"""Optional soft repulsion from the width floor, :math:`-\langle\log(\gamma-\gamma_{\min})\rangle`.

    Off by default: the floor is already a hard constraint and, with real
    residues, a pole at the floor produces a tall but *positive* Lorentzian
    rather than the divergent, sign-changing feature that complex residues would
    give.  Provided for the complex-residue mode, where a pole approaching the
    real axis does drive :math:`\min_\omega A_p \sim -1/\gamma`.
    """
    return -torch.log((gamma - gamma_min).clamp_min(1e-8)).mean()


# --------------------------------------------------------------------------- #
# Combined pretraining objective
# --------------------------------------------------------------------------- #

@dataclass
class LossWeights:
    r"""Weights for :func:`pretrain_loss`.

    Defaults are chosen so that every active term contributes at a comparable
    scale at initialisation, on the synthetic generator's default ranges.  The
    two that matter most:

    * ``w1`` carries peak *position*, which is the quantity classical methods
      get wrong and the one users notice.
    * ``l2`` carries peak *heights and smoothness* -- it is the only term that
      can see the pole-comb ("ripple") artefact, which lives in the kernel's
      null space (invisible to ``chi2``) and cancels in the ``w1`` CDF.  The
      controlled A/B sweep measured: raising it 0.3 -> 1.0 at fixed ``w1``
      cut the ripple metric 27% and improved aggregate ``l2`` with no
      :math:`W_1` or :math:`\chi^2` cost, while a width barrier (5e-3) and a
      weight-entropy sparsity term (0.03) did nothing at those weights.
    * ``chi2`` keeps the model honest against the data it will actually see at
      inference; it is the only term that survives into fine-tuning.

    ``chi2_transform`` compresses the reconstruction term's five-order dynamic
    range.  ``"sqrt"`` (default) uses the residual *norm*, whose gradient scales
    as :math:`1/2\sqrt{\chi^2}`; ``"log1p"`` scales as :math:`1/(1+\chi^2)`,
    which turned out to over-correct -- at :math:`\chi^2 \sim 4000` it leaves the
    reconstruction contributing a gradient of :math:`3\times10^{-4}`, i.e.
    nothing at all, and training proceeds on the spectral terms alone.
    ``"raw"`` applies no transform and lets it dominate.

    ``positivity`` is only meaningful in ``residue_mode="complex"`` (the
    default); in the real mode the penalty is identically zero and is skipped.
    """

    chi2: float = 1.0
    w1: float = 3.0
    l2: float = 1.0                # raised from 0.3 by the ripple experiment
    kl: float = 1.0
    free_bits: float = 0.05
    consistency: float = 0.1
    barrier: float = 0.0
    positivity: float = 1.0
    sparsity: float = 0.0          # weight-entropy penalty (minimal-pole prior)
    multiscale: float = 0.0        # broadened-envelope L2 alongside the plain one
    multiscale_scales: tuple = (1.0, 8.0, 32.0)
    chi2_transform: str = "sqrt"


def pretrain_loss(
    result,
    batch,
    model,
    weights: LossWeights,
    alt_posterior: tuple | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Supervised Stage-1 objective.

    Parameters
    ----------
    result
        :class:`~spevae.models.vae.ACResult` for ``batch.G``.
    batch
        :class:`~spevae.data.synthetic.TrainingBatch`.
    model
        The :class:`~spevae.models.vae.SPEVAE`, for the width floor.
    alt_posterior
        Optional ``(mu, logvar)`` for ``batch.G_alt``; enables the consistency
        term.  Deliberately the *posterior*, not a full result: the term only
        reads mu and logvar, so decoding the second realisation and pushing it
        through the forward map -- which an earlier version did -- was about a
        quarter of every training step spent on tensors nothing consumed.

    Returns
    -------
    (total, parts)
        ``parts`` holds detached scalar **tensors** for logging; pass them
        through :func:`floatify` when you actually need Python numbers.  ``chi2`` in particular is
        the diagnostic to watch: it should approach 1 from above, and *not* go
        below the shrinkage-corrected floor.
    """
    quad = batch.quad
    A_pred = result.poles.spectral(quad.nodes.to(result.poles.eps.dtype))

    # Use every slice.  On *measured* bins tau = beta is linearly dependent on
    # tau = 0 through the sum rule and must be dropped (data/real.py does), but
    # the synthetic generator draws noise independently at all L slices, so
    # G(0) + G(beta) != 1 realisation by realisation and the full covariance is
    # regular.  Dropping a slice here both discards a genuine constraint and
    # de-calibrates chi^2, because batch.inv_sqrt_R whitens the full grid.
    raw_chi2 = chi2(result.G_recon, batch.G, batch.sigma,
                    inv_sqrt_R=batch.inv_sqrt_R)
    # chi^2 spans five orders of magnitude between initialisation and the
    # ansatz floor, so it needs compressing -- but not as hard as log1p, which
    # was the first attempt and silenced the term entirely once chi^2 was in the
    # thousands.  All the options here are monotone, so the minimiser is the
    # same; only the gradient scale differs.
    if weights.chi2_transform == "sqrt":
        L_chi2 = torch.sqrt(raw_chi2.clamp_min(1e-12))
    elif weights.chi2_transform == "log1p":
        L_chi2 = torch.log1p(raw_chi2)
    elif weights.chi2_transform == "raw":
        L_chi2 = raw_chi2
    else:
        raise ValueError(f"unknown chi2_transform {weights.chi2_transform!r}")
    L_w1 = spectral_w1(A_pred, batch.A_true, quad, beta=batch.beta)
    L_l2 = spectral_l2(A_pred, batch.A_true, quad, relative=True)
    L_kl = kl_divergence(result.mu, result.logvar, free_bits=weights.free_bits)

    total = (weights.chi2 * L_chi2 + weights.w1 * L_w1
             + weights.l2 * L_l2 + weights.kl * L_kl)
    # Detached *tensors*, not Python floats.  Every float(tensor) is a device
    # synchronisation that also drains the queued compute; there are eight of
    # them here and they run on every step, so materialising this dict eagerly
    # cost more than the physics.  The caller converts only when it logs -- see
    # `floatify`.
    parts = {"chi2": raw_chi2.detach(), "log_chi2": L_chi2.detach(),
             "w1_omega": L_w1.detach(), "l2_rel": L_l2.detach(),
             "kl": L_kl.detach()}

    if alt_posterior is not None and weights.consistency > 0.0:
        L_cons = noise_consistency(result.mu, result.logvar,
                                   alt_posterior[0], alt_posterior[1])
        total = total + weights.consistency * L_cons
        parts["consistency"] = L_cons.detach()

    # In real mode A is a positive Lorentzian mixture and the integrand of the
    # positivity penalty is identically zero -- skip the (B, M)-sized no-op.
    if weights.positivity > 0.0 and getattr(model, "residue_mode", "complex") != "real":
        L_pos = spectral_positivity(A_pred, quad)
        total = total + weights.positivity * L_pos
        parts["positivity"] = L_pos.detach()
    parts["min_A"] = A_pred.min().detach()

    if weights.barrier > 0.0:
        L_bar = width_barrier(result.poles.gamma, model.gamma_min)
        total = total + weights.barrier * L_bar
        parts["barrier"] = L_bar.detach()

    if weights.sparsity > 0.0:
        L_sp = weight_entropy(result.poles.a)
        total = total + weights.sparsity * L_sp
        parts["sparsity"] = L_sp.detach()

    if weights.multiscale > 0.0:
        L_ms = spectral_multiscale_l2(A_pred, batch.A_true, quad,
                                      scales=weights.multiscale_scales)
        total = total + weights.multiscale * L_ms
        parts["multiscale"] = L_ms.detach()

    parts["total"] = total.detach()
    return total, parts


# --------------------------------------------------------------------------- #
# Data-space positivity: the complete-monotonicity penalties (Stage 2)
# --------------------------------------------------------------------------- #

def negativity_stencil_weights(var0: torch.Tensor) -> dict[str, torch.Tensor]:
    r"""Inverse-variance weights for the :math:`G, G'', G''''` penalties.

    ``var0 = diag(C)`` is the per-slice, per-bin noise variance.  The stencil
    variances propagate it through the finite-difference coefficients
    (:math:`1,4,1` and :math:`1,16,36,16,1` are the squared stencil weights).
    A stencil whose variance is *numerically* zero -- sum-rule-pinned endpoints
    in particle-hole-symmetric bins, where the algorithm fixes
    :math:`G(0) = G(\beta) = 1/2` to machine precision (measured variance
    :math:`\sim 10^{-33}`, not an exact 0) -- gets weight zero: dividing by it
    would hand two slices a :math:`10^{33}` weight, and a noiseless point's
    positivity is already exact.  The threshold is therefore relative to the
    largest variance, not a comparison with literal zero.

    Due to Benjamin Cohen-Stead's reference formulation (see README).
    """
    v = var0.double()
    floor = 1e-16 * float(v.max())

    def inv(x: torch.Tensor) -> torch.Tensor:
        ok = x > floor
        return torch.where(ok, 1.0 / torch.where(ok, x, torch.ones_like(x)),
                           torch.zeros_like(x))

    var2 = v[:-2] + 4.0 * v[1:-1] + v[2:]
    var4 = v[:-4] + 16.0 * v[1:-3] + 36.0 * v[2:-2] + 16.0 * v[3:-1] + v[4:]
    return {"inv_var0": inv(v).to(var0.dtype),
            "inv_var2": inv(var2).to(var0.dtype),
            "inv_var4": inv(var4).to(var0.dtype)}


def negativity_penalties(
    G_recon: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    r"""Variance-weighted penalties on :math:`G<0`, :math:`G''<0`, :math:`G''''<0`.

    For a non-negative :math:`A`, the fermionic kernel makes :math:`G(\tau)`
    completely monotone on :math:`(0, \beta)`: :math:`G \ge 0` and every even
    :math:`\tau`-derivative is non-negative, since
    :math:`\partial_\tau^{2k} G = \int A(\omega)\,\omega^{2k} K(\tau,\omega)
    \,d\omega \ge 0`.  These are *data-space necessary conditions* of
    positivity -- exactly the structure a :math:`\chi^2`-only fine-tune is free
    to violate in the complex-residue mode, and the natural Stage-2 partner of
    the spectral-side penalty :func:`spectral_positivity`.

    .. math::
        \mathcal{L}_0 = \Big\langle \tfrac{1}{L}\textstyle\sum_\tau
            \mathrm{ReLU}(-G)^2 / \mathrm{var}_0 \Big\rangle_B, \qquad
        \mathcal{L}_2, \mathcal{L}_4 \text{ likewise on the stencils.}

    ``weights`` comes from :func:`negativity_stencil_weights`.
    """
    L = G_recon.shape[-1]
    neg0 = torch.relu(-G_recon)
    out = {"neg_G": (neg0.pow(2) * weights["inv_var0"]).sum(-1).mean() / L}
    if L >= 3:
        d2 = G_recon[:, 2:] - 2.0 * G_recon[:, 1:-1] + G_recon[:, :-2]
        out["neg_G2"] = (torch.relu(-d2).pow(2)
                         * weights["inv_var2"]).sum(-1).mean() / (L - 2)
    if L >= 5:
        d4 = (G_recon[:, 4:] - 4.0 * G_recon[:, 3:-1] + 6.0 * G_recon[:, 2:-2]
              - 4.0 * G_recon[:, 1:-3] + G_recon[:, :-4])
        out["neg_G4"] = (torch.relu(-d4).pow(2)
                         * weights["inv_var4"]).sum(-1).mean() / (L - 4)
    return out
