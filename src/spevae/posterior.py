r"""The posterior, as the tool reports it: one band, its evidence, its warnings.

What a user gets from :func:`continue_posterior`
------------------------------------------------
1. **Samples that fit the data.**  Flow samples are refined against the exact
   likelihood by least squares, each against its own redraw of the noise
   (a bootstrap of the bins when bins are given, the noise model otherwise);
   a raw sample that already fits the measured data within noise is kept as
   drawn; a refined sample whose misfit is implausible for a posterior draw
   is rejected.  The band is the spread of what remains.
2. **A statement of the prior.**  The band is the posterior *under the
   training prior of the checkpoint*, and says so.  It is not a property of
   the data alone -- no continuation is.
3. **How much of the answer the prior supplied.**  For each frequency window
   the report sets the posterior interval of the spectral weight next to the
   **prior-free bracket**: the smallest and largest weight any non-negative,
   normalised spectrum can put in that window while every whitened residual
   stays within :data:`BOX_SIGMA` standard deviations.  That bracket uses the
   kernel, the data and the error bars, and nothing learned.  The ratio of
   the two widths (the *share*) is how much of the bracket the posterior
   still spans; one minus it is the localisation the prior supplied.  It is
   reported as a number, not as a warning: measured on in-prior truths the
   posterior is correct at shares of 0.1-0.2, because hard-edged window
   weights are something the kernel never determines well.
4. **Warnings**, each tied to something measured on this dataset: too few
   samples survive the accept step, the samples sit at the model's pole-slot
   cap, the inputs fall outside the range the checkpoint was trained on.

None of this makes the answer right when the truth lies outside the prior.
It makes visible *where* the answer rests on the prior, which is the part a
physicist has to judge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .physics import PoleGreens, SpectralQuadrature
from .physics.poles import PoleSet

__all__ = ["PosteriorReport", "continue_posterior", "prior_free_bounds", "BOX_SIGMA",
           "DEFAULT_WINDOWS"]

#: Half-width, in standard deviations, of the per-residual box that defines
#: "consistent with the data" for the prior-free bracket.  With 65 residuals
#: the truth satisfies it with probability 0.97.
BOX_SIGMA = 3.5
#: Default frequency windows (physical units) for weights and brackets.
DEFAULT_WINDOWS = ((-float("inf"), -3.0), (-3.0, -1.5), (-1.5, -0.5), (-0.5, 0.5),
                   (0.5, 1.5), (1.5, 3.0), (3.0, float("inf")))


def prior_free_bounds(G, x, beta, sigma, inv_sqrt_R=None, windows=DEFAULT_WINDOWS,
                      box_sigma: float = BOX_SIGMA, norm: float | None = None,
                      omega_max: float = 20.0) -> list[dict]:
    r"""Smallest and largest spectral weight per window allowed by the data alone.

    Two linear programs per window over a non-negative measure on the
    quadrature nodes, :math:`\sum_m a_m = N`, with every whitened residual of
    :math:`G - \sum_m a_m K(x, \nu_m)` inside :math:`\pm` ``box_sigma``.  No
    prior, no smoothness, no pole structure: any positive spectrum the data
    cannot exclude is admitted, which is why the brackets are wide -- they
    are the honest statement of what the data alone determine.

    Returns one dict per window: ``lo``, ``hi`` (fractions of the total
    weight), ``window`` and ``feasible``.
    """
    from scipy.optimize import linprog

    G = torch.as_tensor(G, dtype=torch.float64).reshape(-1)
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    s = torch.as_tensor(sigma, dtype=torch.float64).reshape(-1)
    quad = SpectralQuadrature.for_beta(float(beta), dtype=torch.float64)
    nu = quad.nodes
    use = (nu.abs() <= omega_max * beta)
    K = PoleGreens(quad).kernel_matrix(x)[use]                   # (M, L)
    om = (nu[use] / beta).numpy()
    Wm = torch.eye(G.numel(), dtype=torch.float64) if inv_sqrt_R is None \
        else torch.as_tensor(inv_sqrt_R, dtype=torch.float64)
    # whitened residual r = W ((G - a K) / sigma);  |r_i| <= box_sigma
    B = (Wm @ (K / s).T).numpy()                                 # (L, M): r = d - B a
    d = (Wm @ (G / s)).numpy()
    N = float(G[0] + G[-1]) if norm is None else float(norm)
    A_ub = np.vstack([B, -B])
    b_ub = np.concatenate([d + box_sigma, box_sigma - d])
    A_eq, b_eq = np.ones((1, om.size)), np.array([N])
    out = []
    for lo_w, hi_w in windows:
        c = ((om >= lo_w) & (om < hi_w)).astype(float)
        res = {"window": (float(lo_w), float(hi_w)), "feasible": True}
        for key, sign in (("lo", 1.0), ("hi", -1.0)):
            r = linprog(sign * c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=(0, None), method="highs")
            if r.status != 0:
                res["feasible"] = False
                res[key] = float("nan")
            else:
                res[key] = float(sign * r.fun / N)
        out.append(res)
    return out


@dataclass
class PosteriorReport:
    """What :func:`continue_posterior` returns."""

    poles: PoleSet                       # accepted samples, physical units
    omega: np.ndarray
    band: dict                           # q05, q16, q50, q84, q95, mean on omega
    chi2: np.ndarray                     # accepted samples, per point, vs the measured data
    chi2_target: float
    n_samples: int
    n_accepted: int
    n_kept_raw: int                      # accepted samples that needed no refinement
    windows: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    prior: str = ""
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        acc = self.n_accepted / max(self.n_samples, 1)
        lines = [
            f"posterior from {self.n_accepted} of {self.n_samples} samples "
            f"({acc:.0%} accepted, {self.n_kept_raw} needed no refinement, "
            f"{self.diagnostics.get('samples_given_a_pole', 0)} were given an extra pole)",
            f"  fit      : chi2 / target median "
            f"{np.median(self.chi2) / self.chi2_target:.2f} "
            f"[{np.quantile(self.chi2, 0.05) / self.chi2_target:.2f}, "
            f"{np.quantile(self.chi2, 0.95) / self.chi2_target:.2f}]",
            f"  prior    : {self.prior}",
        ]
        if self.windows:
            lines.append("  weight per window: posterior 5-95%  |  allowed by the data alone  "
                         "|  share of the bracket")
            for w in self.windows:
                lo, hi = w["window"]
                name = f"[{lo:g}, {hi:g})".replace("inf", "∞")
                lines.append(f"    {name:14s} {w['q05']:.3f}-{w['q95']:.3f}  |  "
                             f"{w['lo']:.3f}-{w['hi']:.3f}  |  {w['share']:.2f}")
            sh = [w["share"] for w in self.windows if w["share"] == w["share"]]
            if sh:
                lines.append(f"  the posterior spans a median {np.median(sh):.0%} of what the data alone "
                             "allow per window; the rest of the localisation is the prior's")
        for msg in self.warnings:
            lines.append(f"  WARNING  : {msg}")
        if not self.warnings:
            lines.append("  warnings : none")
        return "\n".join(lines)


def continue_posterior(flow, G, x, beta: float, sigma, inv_sqrt_R=None, bins=None,
                       n_samples: int = 200, omega=None, chi2_target: float = 1.0,
                       windows=DEFAULT_WINDOWS, bounds: bool = True, accept_sigma: float = 3.0,
                       lam: float = 1.0, generator: torch.Generator | None = None,
                       x_fit=None, G_fit=None, sigma_fit=None,
                       max_births: int = 2, sigma_floor: float | None = None) -> PosteriorReport:
    r"""Posterior over spectra for one dataset, with its evidence and warnings.

    Parameters
    ----------
    flow
        A trained :class:`~spevae.models.flow.FlowPosterior`.
    G, x, sigma
        The measured :math:`G(\tau)`, :math:`x = \tau/\beta` and the standard
        errors of the mean, on the full grid (what the encoder sees).
    inv_sqrt_R
        Whitener of the noise correlation on the fitted slices (identity if
        omitted).
    bins
        ``(N_bins, L_fit)`` raw bins on the fitted slices.  When given, each
        sample is refined against a bootstrap mean of the bins -- the noise
        the measurement actually shows -- instead of a draw from the noise
        model.
    x_fit, G_fit, sigma_fit
        The fitted slices when they differ from the encoder's grid (real
        cells drop exactly pinned slices); default: the full grid.
    chi2_target
        The per-point :math:`\chi^2` a perfect fit scores under this whitener
        (1 for synthetic data; :meth:`MeasuredGreens.covariance` reports it
        for real cells).
    accept_sigma
        A sample is accepted when its misfit is within this many standard
        deviations (:math:`\sqrt{2/L}` per point) of the target.
    sigma_floor
        A systematic-error floor, relative to :math:`\max|G|`, added in
        quadrature to every error bar (same correlation structure).  DQMC
        data carry a Trotter error of order :math:`\Delta\tau^2` that the
        statistical error bars do not contain; below about 1e-4 relative it
        is usually the larger of the two, and fitting to the statistical
        error alone then fits the discretisation.  Off by default; the
        report warns when the error bars are that small and no floor is set.
    max_births
        A sample that cannot reach the data is given up to this many extra
        poles, one at a time, each placed where a sample that does fit has
        one, and refitted (the refinement itself never changes a sample's
        pole count).  Measured on held-out truths: the share of samples that
        fit rises from 0.68 to 0.88 at noise 3e-5 and to 1.00 above 3e-4.
    """
    from .models.flow import add_pole, refine_samples

    g = generator or torch.Generator().manual_seed(0)
    G = torch.as_tensor(G, dtype=torch.float32).reshape(1, -1)
    x = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
    sigma = torch.as_tensor(sigma, dtype=torch.float32).reshape(1, -1)
    xf = x if x_fit is None else torch.as_tensor(x_fit, dtype=torch.float32).reshape(-1)
    Gf = G if G_fit is None else torch.as_tensor(G_fit, dtype=torch.float32).reshape(1, -1)
    sf = sigma if sigma_fit is None else torch.as_tensor(sigma_fit, dtype=torch.float32).reshape(1, -1)
    L = xf.numel()
    rel_stat = float((sigma / G.abs().amax()).median())           # as measured, before any floor
    if sigma_floor:
        fl = float(sigma_floor) * float(G.abs().amax())
        sigma, sf = torch.sqrt(sigma ** 2 + fl ** 2), torch.sqrt(sf ** 2 + fl ** 2)
    W = torch.eye(L) if inv_sqrt_R is None else torch.as_tensor(inv_sqrt_R, dtype=torch.float32)
    S = int(n_samples)
    quad = SpectralQuadrature.for_beta(float(beta))
    fwd = PoleGreens(quad)

    def chi2_of(ps):
        d = ((fwd(ps, xf) - Gf) / sf) @ W.T
        return (d * d).mean(-1)

    z, norm = flow.sample_coords(G, x, beta, sigma=sigma, n_samples=S, generator=g,
                                 inv_sqrt_R=W, x_fit=xf, G_fit=Gf, sigma_fit=sf)
    raw = flow.coords.decode(z, norm=norm, nu_max=flow.nu_max(beta))
    c_raw = chi2_of(raw)
    band_edge = chi2_target * (1.0 + accept_sigma * math.sqrt(2.0 / L))
    # each sample's own redraw of the data: a bootstrap of the bins when there are bins,
    # a draw from the noise model otherwise
    if bins is not None:
        bins = torch.as_tensor(bins, dtype=torch.float32)
        idx = torch.randint(bins.shape[0], (S, bins.shape[0]), generator=g)
        target = bins[idx].mean(1)
    else:
        xi = torch.randn(S, L, generator=g)
        target = Gf + sf * (xi @ torch.linalg.inv(W).T)
    ps, _, u = refine_samples(flow, z, norm, beta, xf, target, sf.expand(S, -1), W, lam,
                              anneal_from=1e-2, return_coords=True)
    # a raw sample that already fits the measured data is a posterior draw as it stands
    fits_raw = c_raw <= chi2_target * (1.0 + 2.0 * math.sqrt(2.0 / L))
    ps = PoleSet(*(torch.where(fits_raw.unsqueeze(-1), getattr(raw, k).to(getattr(ps, k).dtype),
                               getattr(ps, k)) for k in ("eps", "gamma", "a", "b")))
    c2 = chi2_of(ps)
    # births: a sample that cannot reach the data gets one more pole, borrowed from a
    # sample that fits, and a second fit from where the first one ended
    zc = torch.cat([u, z[..., 4:5]], -1)
    n_born = torch.zeros(S, dtype=torch.long)
    for _ in range(int(max_births)):
        bad = c2 > band_edge
        if not bool(bad.any()) or bool(bad.all()) and S < 2:
            break
        good = ~bad
        pool = PoleSet(*(getattr(ps, k)[good if bool(good.any()) else slice(None)]
                         for k in ("eps", "gamma", "a", "b")))
        z_new, born = add_pole(flow, zc, pool, generator=g)
        did = born & bad
        if not bool(did.any()):
            break
        ps2, _, u2 = refine_samples(flow, z_new[did], norm[did], beta, xf, target[did],
                                    sf.expand(S, -1)[did], W, lam, lm_iters=60, return_coords=True)
        d2 = ((fwd(ps2, xf) - Gf) / sf) @ W.T
        c_new = (d2 * d2).mean(-1)
        better = c_new < c2[did]
        rows = torch.nonzero(did).flatten()[better]
        for k in ("eps", "gamma", "a", "b"):
            getattr(ps, k)[rows] = getattr(ps2, k)[better].to(getattr(ps, k).dtype)
        zc[rows] = torch.cat([u2, z_new[did][..., 4:5]], -1)[better]
        c2[rows] = c_new[better]
        n_born[rows] += 1
    acc = c2 <= band_edge
    warnings = []
    if not bool(acc.any()):
        warnings.append("NO sample fits the data within noise; the band below is the unfitted "
                        "ensemble and should not be used. The data are outside what this "
                        "checkpoint can represent.")
        acc = torch.ones_like(acc)
    frac = float(acc.float().mean())
    if 0.0 < frac < 0.5:
        warnings.append(f"only {frac:.0%} of samples could be brought onto the data; the rest "
                        "were rejected. The data sit at the edge of the training prior.")
    sel = PoleSet(*(getattr(ps, k)[acc] for k in ("eps", "gamma", "a", "b")))
    n_pres = (sel.a != 0).sum(-1)
    if float(n_pres.float().median()) >= flow.P:
        warnings.append(f"the accepted samples use all {flow.P} pole slots; the spectrum may need "
                        "more structure than the model can place.")
    recipe = getattr(flow, "meta", {}).get("recipe", {}) or {}
    if rel_stat < 1e-4 and not sigma_floor:
        warnings.append(f"statistical error bars are {rel_stat:.0e} relative. Systematic errors "
                        "(the Trotter error of DQMC, of order dtau^2) are usually larger than "
                        "that and are not in the error bars; consider sigma_floor.")
    rel = float((sigma / G.abs().amax()).median())
    lo_n, hi_n = recipe.get("noise_lo", 3e-5), recipe.get("noise_hi", 3e-2)
    if not lo_n * 0.5 <= rel <= hi_n * 2.0:
        warnings.append(f"relative error bar {rel:.1e} is outside the range this checkpoint was "
                        f"trained on ({lo_n:g} to {hi_n:g}).")
    betas = recipe.get("beta_choices") or [8.0]
    if not min(betas) * 0.45 <= float(beta) <= max(betas) * 2.2:
        warnings.append(f"beta = {beta:g} is outside the range this checkpoint was trained on "
                        f"({min(betas):g} to {max(betas):g}).")

    omega = np.linspace(-8.0, 8.0, 801) if omega is None else np.asarray(omega, float)
    nu = torch.tensor(omega * beta, dtype=torch.float64)
    p64 = PoleSet(*(getattr(sel, k).double() for k in ("eps", "gamma", "a", "b")))
    A = (p64.spectral(nu) * beta).numpy()
    q = np.quantile(A, [0.05, 0.16, 0.5, 0.84, 0.95], axis=0)
    band = dict(zip(("q05", "q16", "q50", "q84", "q95"), q), mean=A.mean(0))

    win = []
    if windows:
        # exact window weights of each sample from the pole CDF
        eps, gam, a, b = (getattr(p64, k) for k in ("eps", "gamma", "a", "b"))

        def cdf(w):
            if math.isinf(w):
                return torch.zeros(eps.shape[0], dtype=torch.float64) if w < 0 else a.sum(-1)
            u = w * beta - eps
            return (a * (torch.atan(u / gam) / math.pi + 0.5)
                    - b * torch.log(u * u + gam * gam) / (2.0 * math.pi)).sum(-1)

        tot = a.sum(-1)
        pf = (prior_free_bounds(Gf[0], xf, beta, sf[0], W, windows,
                                norm=float(G[0, 0] + G[0, -1]))
              if bounds else [None] * len(windows))
        for (lo_w, hi_w), bnd in zip(windows, pf):
            wgt = ((cdf(hi_w) - cdf(lo_w)) / tot).numpy()
            row = {"window": (float(lo_w), float(hi_w)), "mean": float(wgt.mean()),
                   "q05": float(np.quantile(wgt, 0.05)), "q95": float(np.quantile(wgt, 0.95))}
            if bnd is not None and bnd["feasible"]:
                width = max(bnd["hi"] - bnd["lo"], 1e-12)
                row.update(lo=bnd["lo"], hi=bnd["hi"],
                           share=float((row["q95"] - row["q05"]) / width))
                # a posterior interval that leaves the bracket contradicts the data
                if row["q95"] < bnd["lo"] - 1e-6 or row["q05"] > bnd["hi"] + 1e-6:
                    warnings.append(f"the posterior weight in [{lo_w:g}, {hi_w:g}) lies outside "
                                    "what the data allow; do not use this window.")
            else:
                row.update(lo=float("nan"), hi=float("nan"), share=float("nan"))
            win.append(row)

    name = getattr(flow, "meta", {}).get("recipe", {}).get("name") if isinstance(
        getattr(flow, "meta", {}).get("recipe", None), dict) else None
    prior = ("the training prior of this checkpoint"
             + (f" ({name})" if name else "")
             + ": mixtures of pole features, smooth bands and exactly solvable models; "
               "positive, normalised, causal spectra with widths above the resolution floor")
    return PosteriorReport(
        poles=PoleSet(sel.eps / beta, sel.gamma / beta, sel.a, sel.b), omega=omega, band=band,
        chi2=c2[acc].numpy(), chi2_target=float(chi2_target), n_samples=S,
        n_accepted=int(acc.sum()), n_kept_raw=int((acc & fits_raw).sum()), windows=win,
        warnings=warnings, prior=prior,
        diagnostics={"accepted_fraction": frac, "raw_chi2_median": float(c_raw.median()),
                     "samples_given_a_pole": int((n_born[acc] > 0).sum()),
                     "n_poles_median": float(n_pres.float().median()),
                     "relative_error_bar": rel_stat, "sigma_floor": sigma_floor,
                     "bootstrap_redraw": bins is not None})
