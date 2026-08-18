r"""Stage 2: per-dataset refinement, in the latent space only.

The problem this solves, and the one it refuses to create
---------------------------------------------------------
A pretrained network answers in one forward pass.  Two things can then be
wrong with the answer:

* **Amortisation gap.**  The encoder is a single function trained to be good on
  average.  For any particular :math:`G(\tau)` there is generally a nearby code
  that fits *that* dataset better.
* **Distribution shift.**  If the data's physics is not well covered by the
  training families, the amortised answer is an extrapolation.

The obvious fix -- keep training the network on this dataset's :math:`\chi^2` --
is the wrong one, and its failure is documented rather than hypothetical.  On an
unknown dataset :math:`\chi^2` is the only available loss, and :math:`\chi^2`
cannot separate the many spectra that fit the data to within noise; that *is*
the ill-posedness.  The predecessor project's matched-compute campaign found
:math:`\chi^2 \approx 1` runs that recovered visibly wrong spectra.  Fine-tuning
the weights on :math:`\chi^2` therefore spends the prior that pretraining bought
and lands back at unregularised fitting.

So this stage freezes every weight and optimises only :math:`z`:

.. math::
    \hat z \;=\; \arg\min_z\;
    \tfrac12 \big\| R^{-1/2} D^{-1}\big(G - \mathcal{F}[\mathrm{dec}(z)]\big)\big\|^2
    \;+\; \tfrac{\lambda}{2}\,\|z\|^2 .

That is MAP estimation in the latent space: the first term is the Gaussian
log-likelihood of the data, the second is the model's own :math:`\mathcal N(0,I)`
prior, and :math:`\lambda = 1` makes it exactly the posterior mode.  The answer
is always something the decoder can produce, so **every structural guarantee
survives untouched** -- causality, positivity, the sum rule, the tail, the width
floor -- and the search cannot wander off the manifold of spectra the model was
taught are plausible.  What changes is only *which* of those spectra is chosen,
and now it is chosen using this dataset rather than an average over the training
distribution.

Stopping
--------
At the *calibrated* noise floor, not at :math:`\chi^2 = 1`.  On real DQMC data
the perfect-fit target can be far below 1 (0.087 on one site-Holstein cell), and
driving :math:`\chi^2` to 1 there would mean stopping while still badly
underfit.  Refinement halts when :math:`\chi^2/\chi^2_{\rm target}` reaches
``target_ratio``, which by default is 1.

Diagnostics that matter
-----------------------
:attr:`RefineResult.z_shift` reports how far the refined code moved from the
encoder's posterior mean, in units of the posterior standard deviation.  A small
shift means the amortised answer was already good.  A shift of many sigma means
either a real amortisation gap or that the data is outside the training
distribution -- and in the second case the right response is to add a spectral
family and retrain, not to trust a code the prior considers unlikely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .data.real import MeasuredGreens
from .models.vae import SPEVAE

__all__ = ["RefineConfig", "RefineResult", "refine_latent"]


@dataclass
class RefineConfig:
    r"""Settings for :func:`refine_latent`.

    Attributes
    ----------
    steps
        Maximum optimisation steps.  Cheap: each is one decode plus one matrix
        product, with no encoder pass and no parameter gradients, so a few
        thousand costs seconds.  Always check :attr:`RefineResult.converged`
        and the :math:`\chi^2` ratio afterwards -- an unconverged refinement is
        an intermediate answer, not a result, and the number of steps needed
        depends on how well-conditioned the trained decoder's :math:`z \to A`
        map is.
    lr
        Adam step size on :math:`z`.  The latent is :math:`\mathcal N(0,I)`-scaled
        by construction, so this is in units where 1 is an enormous move.
    prior_weight
        :math:`\lambda`.  ``1.0`` is the model's own prior and gives the exact
        MAP estimate.  Lower it only if you have a specific reason to trust the
        data over the prior, and say so when reporting -- it is the knob that
        trades away the regularisation this stage exists to preserve.
    target_ratio
        Stop once :math:`\chi^2/\chi^2_{\rm target}` falls to this.  ``1.0`` is
        the noise floor; going below it is fitting noise.
    min_eigenvalue
        Eigenvalue floor for the correlation estimate; raise it when
        :math:`N_{\rm bins} \lesssim L_\tau`.
    per_bin
        Refine every bin separately (each gets its own :math:`z`) instead of
        refining the bin mean.  The per-bin spread is then a data-driven
        uncertainty rather than a latent-prior one.
    """

    steps: int = 2000
    lr: float = 0.05
    prior_weight: float = 1.0
    target_ratio: float = 1.0
    min_eigenvalue: float = 0.0
    per_bin: bool = False
    patience: int = 200
    log_every: int = 0


@dataclass
class RefineResult:
    """What refinement produced, next to what it started from."""

    z: torch.Tensor                     # (B, Lz) refined code
    z_init: torch.Tensor                # (B, Lz) encoder posterior mean
    z_sigma: torch.Tensor               # (B, Lz) encoder posterior sd
    poles: object                       # PoleSet, physical units
    G_recon: torch.Tensor               # (B, L)
    chi2: float                         # final, per degree of freedom
    chi2_init: float                    # before refinement
    chi2_target: float
    history: list = field(default_factory=list)
    beta: float = 0.0
    steps_run: int = 0
    converged: bool = False
    data_term: float = 0.0              # 0.5 * n * chi2 at the solution
    prior_term: float = 0.0             # 0.5 * lambda * ||z||^2 at the solution

    @property
    def chi2_ratio(self) -> float:
        return self.chi2 / self.chi2_target if self.chi2_target > 0 else float("nan")

    @property
    def prior_is_binding(self) -> bool:
        r"""Whether the prior actually constrains the answer.

        MAP balances a data term against a prior term.  When the model fits
        badly the data term can exceed the prior by orders of magnitude, and the
        result is then effectively pure :math:`\chi^2` minimisation -- still
        confined to the decoder's manifold and still obeying every structural
        constraint, but *not* meaningfully regularised by the prior.  That is a
        statement about the pretrained model, not about this stage, and it is
        worth knowing before quoting the answer.
        """
        return self.data_term < 100.0 * max(self.prior_term, 1e-12)

    @property
    def z_shift(self) -> torch.Tensor:
        r"""``|z - mu| / sigma`` per latent dimension: the amortisation gap in sigma."""
        return (self.z - self.z_init).abs() / self.z_sigma.clamp_min(1e-6)

    def spectral(self, omega: torch.Tensor) -> torch.Tensor:
        return self.poles.spectral(omega.to(self.z.dtype))

    def summary(self) -> str:
        s = self.z_shift
        return (
            f"latent refinement ({self.steps_run} steps"
            f"{', converged' if self.converged else ''})\n"
            f"  chi2      : {self.chi2_init:.4g} -> {self.chi2:.4g}"
            f"   (target {self.chi2_target:.4g}, ratio {self.chi2_ratio:.3g})\n"
            f"  z shift   : mean {float(s.mean()):.2f} sigma, "
            f"max {float(s.max()):.2f} sigma\n"
            f"  gamma     : {float(self.poles.gamma.min()):.4g} .. "
            f"{float(self.poles.gamma.max()):.4g}\n"
            f"  weights   : sum {float(self.poles.a.sum(-1).mean()):.6f}, "
            f"min {float(self.poles.a.min()):.3g}\n"
            f"  MAP terms : data {self.data_term:.4g} vs prior {self.prior_term:.4g}"
            f"  -> prior {'binding' if self.prior_is_binding else 'NOT binding'}"
        )


@torch.enable_grad()
def refine_latent(
    model: SPEVAE,
    mg: MeasuredGreens,
    cfg: RefineConfig | None = None,
) -> RefineResult:
    r"""Refine the latent code against one dataset, with the weights frozen.

    Parameters
    ----------
    model
        A trained :class:`~spevae.models.vae.SPEVAE`.  Left untouched: its
        parameters' ``requires_grad`` flags are restored on exit, and no
        optimiser ever sees them.
    mg
        The measured Green's function.
    cfg
        See :class:`RefineConfig`.

    Returns
    -------
    RefineResult
    """
    cfg = cfg or RefineConfig()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    cov = mg.covariance(cfg.min_eigenvalue)
    n = cov["n_independent"]
    inv_sqrt_R = cov["inv_sqrt_R"].to(device=device, dtype=dtype)

    # Two different sigmas, and mixing them up is a shape error at best and a
    # silent miscalibration at worst.  The encoder needs one value per time
    # slice (length L); the chi^2 uses only the linearly-independent block
    # (length n = L-1 on an endpoint-inclusive grid, since tau = beta is fixed
    # by the sum rule).
    scale = 1.0 if cfg.per_bin else 1.0 / math.sqrt(mg.n_bins)
    G = (mg.bins if cfg.per_bin else mg.mean.unsqueeze(0)).to(device=device,
                                                              dtype=dtype)
    sigma_full = (mg.sigma * scale).to(device=device, dtype=dtype)
    sigma_full = sigma_full.unsqueeze(0).expand(G.shape[0], -1)
    sigma_fit = (cov["sigma"] * scale).to(device=device, dtype=dtype)
    x = mg.x.to(device=device, dtype=dtype)

    # Freeze everything.  The point of this stage is that the prior does not move.
    was_training = model.training
    saved = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved:
        p.requires_grad_(False)
    model.eval()

    try:
        with torch.no_grad():
            mu, logvar, norm = model.encode(
                G, x, mg.beta, sigma=sigma_full,
                includes_endpoints=mg.includes_endpoints)
            z_sigma = torch.exp(0.5 * logvar.clamp(-20.0, 20.0))
        z = mu.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=cfg.lr)
        K = model.kernel_matrix(mg.beta, x)
        fwd = model.forward_map(mg.beta, device, dtype)

        def chi2_of(z_):
            poles = model.decode(z_, norm, mg.beta)
            G_rec = fwd(poles, x, kernel_matrix=K)
            d = ((G_rec - G)[:, :n] / sigma_fit) @ inv_sqrt_R.transpose(-1, -2)
            return (d * d).sum(-1).mean() / n, poles, G_rec

        with torch.no_grad():
            chi2_init = float(chi2_of(z)[0])

        best = (float("inf"), z.detach().clone())
        history, stall, converged, step = [], 0, False, 0
        for step in range(1, cfg.steps + 1):
            chi2, _, _ = chi2_of(z)
            # MAP objective: half the chi^2 *sum* (not the per-dof average) plus
            # half the squared code norm.  Using the average here would silently
            # reweight data against prior by a factor of n.
            loss = 0.5 * n * chi2 + 0.5 * cfg.prior_weight * (z * z).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            val = float(chi2.detach())
            history.append(val)
            if val < best[0] - 1e-9:
                best = (val, z.detach().clone())
                stall = 0
            else:
                stall += 1
            if cfg.log_every and step % cfg.log_every == 0:
                print(f"  refine {step:4d}  chi2 {val:.5g}  "
                      f"ratio {val / cov['chi2_target']:.4g}")
            if val <= cfg.target_ratio * cov["chi2_target"]:
                converged = True
                break
            if stall >= cfg.patience:
                break

        z_best = best[1].requires_grad_(False)
        with torch.no_grad():
            chi2_final, poles, G_rec = chi2_of(z_best)
            data_term = float(0.5 * n * chi2_final)
            prior_term = float(0.5 * cfg.prior_weight * (z_best * z_best).sum(-1).mean())
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)
        if was_training:
            model.train()

    return RefineResult(
        z=z_best.detach(), z_init=mu.detach(), z_sigma=z_sigma.detach(),
        poles=poles.detach().to_physical(mg.beta), G_recon=G_rec.detach(),
        chi2=float(chi2_final), chi2_init=chi2_init,
        chi2_target=cov["chi2_target"], history=history, beta=float(mg.beta),
        steps_run=step, converged=converged,
        data_term=data_term, prior_term=prior_term,
    )
