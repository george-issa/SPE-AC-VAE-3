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
4. **Narrow lines, flagged.**  Every pole has a width of at least the
   trained floor.  When a tenth or more of the weight sits at it, the fit is
   repeated with the floor lowered, and the window intervals span both
   readings of those poles -- as fitted, and as sharp lines -- because the
   data do not fix the shape of a line that narrow.
5. **Warnings**, each tied to something measured on this dataset: too few
   samples survive the accept step, the samples sit at the model's pole-slot
   cap, the inputs fall outside the range the checkpoint was trained on.

Measured limits (TRAINING.md (xxii)).  On draws of the prior and on exact
lattice spectra the window intervals hold the truth at their stated rate.
On exactly diagonalised line spectra they hold it at 0.52 / 0.86 (for
0.50 / 0.90) where narrow lines are flagged and at 0.29 / 0.64 where the
lines are too dense to be flagged; for a spectrum that may be discrete the
prior-free bracket is the statement to use -- it held in every window.

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
           "DEFAULT_WINDOWS", "BAND_TARGETS", "band_levels", "fit_band_table",
           "weight_at_width_floor", "FLOOR_TRIGGER", "FLOOR_FACTOR", "FLOOR_WITHIN",
           "central_interval"]


#: A pole counts as sitting at the width floor when its half-width is within this factor of it.
FLOOR_WITHIN = 1.25


def weight_at_width_floor(a: torch.Tensor, gamma: torch.Tensor, gamma_min: float,
                          within: float = FLOOR_WITHIN) -> float:
    """Share of spectral weight on poles sitting at the model's minimum width.

    ``a, gamma`` are ``(samples, poles)`` in dimensionless units; the result
    is the median over samples of the weight on poles with
    ``gamma <= within * gamma_min``.  A pole is pinned at the floor only when
    the data want a feature narrower than the model can draw: measured on
    the exactly solvable models it is 0.00 on every lattice spectrum (100
    runs over five settings) and 0.15-0.4 or more on isolated sharp lines --
    the spectra on which a method without a width floor is sharper.
    """
    a = a.double()
    tot = a.sum(-1).clamp_min(1e-300)
    return float(((a * (gamma.double() <= within * gamma_min)).sum(-1) / tot).median())

#: Coverages the reported bands are stated at.
BAND_TARGETS = (0.5, 0.68, 0.9, 0.95)

#: Share of weight at the width floor (:func:`weight_at_width_floor`) from which the report
#: says the data favour narrower lines, and ``floor_factor="auto"`` lowers the fit's floor.
FLOOR_TRIGGER = 0.1
#: What ``floor_factor="auto"`` divides the trained width floor by.
FLOOR_FACTOR = 4.0


def band_levels(table: dict | None, rel_noise: float, beta: float) -> dict:
    r"""Central level of the sample ensemble that holds the truth with each target probability.

    The samples are over-dispersed: measured on held-out draws of the training
    prior, the central 90% of their spectral weight in a frequency window
    holds the truth about 93-96% of the time.  A band table, fitted and
    tested on such draws (:func:`fit_band_table`,
    ``scripts/step3/fit_band_calibration.py``) and shipped beside the
    checkpoint, says which central level to quote for each target, per
    stratum of relative error bar (and beta).  Without a table the levels
    are the targets themselves.

    The table is for *window weights*, the quantity it is fitted and tested
    on.  It is not applied to the pointwise band of :math:`A(\omega)`:
    measured on exact lattice spectra, plain sample quantiles hold
    :math:`A(\omega)` at 0.90-0.99 of frequencies at the 90% level, and the
    table's narrower levels fall to 0.83 at beta = 16.
    """
    if not table:
        return {t: t for t in BAND_TARGETS}
    i = int(np.searchsorted(np.asarray(table.get("noise_edges", []), float), rel_noise, side="right"))
    k = int(np.searchsorted(np.asarray(table.get("beta_edges", []), float), beta, side="right"))
    return {float(t): float(v) for t, v in zip(table["targets"], table["levels"][i][k])}


def central_interval(w, level: float) -> tuple[float, float]:
    """Sample quantiles between which one more draw falls with probability ``level``.

    Among ``n`` exchangeable draws a further one lands between two order
    statistics ``k`` ranks apart with probability ``k / (n + 1)``, whereas
    the central ``level`` of the sample quantiles spans ``level * (n - 1)``
    ranks.  So the quantiles are taken at the central
    ``level * (n + 1) / (n - 1)`` -- the convention the band table is fitted
    in (ranks among ``n + 1`` slots).  With 200 samples the 90% interval is
    the central 90.9%; with 30 it is the central 96.2%; below
    ``(1 + level) / (1 - level)`` samples (19 at 90%) the level cannot be
    reached and the interval is the range of the samples.
    """
    w = np.asarray(w, float)
    n = w.shape[0]
    c = min(float(level) * (n + 1) / max(n - 1, 1), 1.0)
    return float(np.quantile(w, 0.5 - c / 2.0)), float(np.quantile(w, 0.5 + c / 2.0))


def fit_band_table(need, rel_noise, beta, noise_edges=(), beta_edges=(), targets=BAND_TARGETS,
                   min_truths: int = 100) -> dict:
    """Band table from held-out truths.

    ``need`` is ``(n_truths, n_stats)``: for every truth and statistic, the
    central level of its sample ensemble at which the truth is just covered,
    ``|2 PIT - 1|``.  The level to quote for a target coverage ``t`` is the
    ``t``-quantile of ``need`` in the stratum; a stratum with fewer than
    ``min_truths`` truths takes the pooled value.
    """
    need, rel_noise, beta = np.asarray(need, float), np.asarray(rel_noise, float), np.asarray(beta, float)
    pooled = [float(np.quantile(need, t)) for t in targets]
    ni = np.searchsorted(np.asarray(noise_edges, float), rel_noise, side="right")
    bi = np.searchsorted(np.asarray(beta_edges, float), beta, side="right")
    levels, counts = [], []
    for i in range(len(noise_edges) + 1):
        levels.append([])
        counts.append([])
        for k in range(len(beta_edges) + 1):
            sel = (ni == i) & (bi == k)
            counts[-1].append(int(sel.sum()))
            levels[-1].append([float(np.quantile(need[sel], t)) for t in targets]
                              if sel.sum() >= min_truths else list(pooled))
    return {"targets": [float(t) for t in targets], "noise_edges": [float(v) for v in noise_edges],
            "beta_edges": [float(v) for v in beta_edges], "levels": levels, "pooled": pooled,
            "truths": counts}

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
    band: dict                           # q05, q16, q25, q50, q75, q84, q95, mean on omega:
    #                                      plain sample quantiles of A(omega), pointwise
    #                                      (the band table applies to `windows` only)
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
        lev = self.diagnostics.get("band_levels") or {}
        if self.diagnostics.get("band_calibrated"):
            lines.append(f"  intervals: window weights calibrated on held-out draws of that prior (the 90% "
                         f"interval is the central {lev.get(0.9, 0.9):.0%} of the samples); the A(omega) "
                         "band is plain sample quantiles")
        elif self.diagnostics.get("band_table_withheld"):
            lines.append("  intervals: plain sample quantiles; the checkpoint's band table is not applied "
                         "when the data favour narrow lines (it was calibrated on spectra without them)")
        else:
            lines.append("  intervals: plain sample quantiles, not calibrated (measured conservative: a "
                         "90% window interval holds the truth about 93-96% of the time)")
        if self.windows:
            if self.diagnostics.get("line_shape_readings"):
                lines.append("  the window intervals span both readings of the narrow poles: as fitted, "
                             "and as sharp lines")
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
                       lam: float = 10.0, generator: torch.Generator | None = None,
                       x_fit=None, G_fit=None, sigma_fit=None,
                       max_births: int = 2, sigma_floor: float | None = None,
                       count_model=None, count_floor: float = 0.02,
                       band_calibration="auto", floor_factor="auto") -> PosteriorReport:
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
    lam
        How firmly each sample is held at the flow's own proposal while it is
        fitted to the data (weight of :math:`\tfrac12\lambda\|u-u_k\|^2` in
        the flow's normalised pole coordinates).  The data fix a few
        directions; this decides who sets the rest, the flow or the
        least-squares path.  Measured at 1, 10 and 100: 10 lowers the error
        on exactly solvable models by about a third and leaves the misfit
        calibrated; at 100 the samples no longer fit the data as well as
        the truth does (data-fit rank 0.29).
    sigma_floor
        A systematic-error floor, relative to :math:`\max|G|`, added in
        quadrature to every error bar (same correlation structure).  DQMC
        data carry a Trotter error of order :math:`\Delta\tau^2` that the
        statistical error bars do not contain; below about 1e-4 relative it
        is usually the larger of the two, and fitting to the statistical
        error alone then fits the discretisation.  Off by default; the
        report warns when the error bars are that small and no floor is set.
    band_calibration
        The band table that turns sample quantiles of the window weights into
        intervals of stated coverage (:func:`band_levels`).  ``"auto"`` uses
        the one shipped with the checkpoint
        (``flow.meta["band_calibration"]``) if there is one; ``None`` reports
        plain sample quantiles.  The samples (``poles``) and the pointwise
        band of :math:`A(\omega)` are never changed by it.
    count_model
        A :class:`~spevae.models.count.CountPosterior`.  When given, the
        accepted samples are resampled so that their pole counts follow the
        classifier's :math:`p(n \mid G)` instead of the flow's own, which
        barely depends on the data (``count_floor`` mixes in a uniform share
        to bound the weights).  The effective sample size is reported in
        ``diagnostics``.
    floor_factor
        The fit's width floor, as a divisor of the trained one
        (:class:`~spevae.models.flow.LoweredFloor`).  ``"auto"`` (default)
        fits as trained first and, when :func:`weight_at_width_floor` of the
        accepted samples reaches :data:`FLOOR_TRIGGER` -- the data favour
        lines narrower than the model draws -- fits the same proposals
        again, against the same noise redraws, with the floor divided by
        :data:`FLOOR_FACTOR`, and reports that fit.  ``1`` always fits with
        the floor the flow was trained with; a number above 1 always lets
        the fit narrow a pole down to the trained floor divided by it, on a
        quadrature that much finer.  The flow's proposals are the same in
        every case.  Measured (TRAINING.md (xxii)): the second fit is run
        on none of 100 exact lattice spectra and on 10% of held-out prior
        draws, where it changes neither error nor coverage; on exact line
        spectra it is run on 4 in 10 and lowers the error on those by
        4-20%, at four to five times the fitting time.

        Whenever a tenth or more of the weight is on poles no wider than
        :data:`FLOOR_WITHIN` trained floors, the window intervals span both
        readings of those poles -- as fitted, and as sharp lines at their
        positions -- and the band table is not applied: the data do not fix
        the shape of a line that narrow, and the weight of a window with
        such a line near its edge depends on it.
    max_births
        A sample that cannot reach the data is given up to this many extra
        poles, one at a time, each placed where a sample that does fit has
        one, and refitted (the refinement itself never changes a sample's
        pole count).  Measured on held-out truths: the share of samples that
        fit rises from 0.68 to 0.88 at noise 3e-5 and to 1.00 above 3e-4.
    """
    from .models.flow import LoweredFloor, add_pole, refine_samples

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
    fwd0 = PoleGreens(SpectralQuadrature.for_beta(float(beta), gamma_min=flow.coords.gamma_min))

    def chi2_with(fwd, ps):
        d = ((fwd(ps, xf) - Gf) / sf) @ W.T
        return (d * d).mean(-1)

    z, norm = flow.sample_coords(G, x, beta, sigma=sigma, n_samples=S, generator=g,
                                 inv_sqrt_R=W, x_fit=xf, G_fit=Gf, sigma_fit=sf)
    raw = flow.coords.decode(z, norm=norm, nu_max=flow.nu_max(beta))
    c_raw = chi2_with(fwd0, raw)
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
    # a raw sample that already fits the measured data is a posterior draw as it stands
    fits_raw = c_raw <= chi2_target * (1.0 + 2.0 * math.sqrt(2.0 / L))

    def fit(model, z_start):
        """The fit stage in the coordinates of ``model``: the flow, or a LoweredFloor on it."""
        fwd = fwd0 if model is flow else PoleGreens(
            SpectralQuadrature.for_beta(float(beta), gamma_min=model.coords.gamma_min))
        ps, _, u = refine_samples(model, z_start, norm, beta, xf, target, sf.expand(S, -1), W, lam,
                                  anneal_from=1e-2, return_coords=True)
        ps = PoleSet(*(torch.where(fits_raw.unsqueeze(-1), getattr(raw, k).to(getattr(ps, k).dtype),
                                   getattr(ps, k)) for k in ("eps", "gamma", "a", "b")))
        c2 = chi2_with(fwd, ps)
        # births: a sample that cannot reach the data gets one more pole, borrowed from a
        # sample that fits, and a second fit from where the first one ended
        zc = torch.cat([u, z_start[..., 4:5]], -1)
        n_born = torch.zeros(S, dtype=torch.long)
        for _ in range(int(max_births)):
            bad = c2 > band_edge
            if not bool(bad.any()) or bool(bad.all()) and S < 2:
                break
            good = ~bad
            pool = PoleSet(*(getattr(ps, k)[good if bool(good.any()) else slice(None)]
                             for k in ("eps", "gamma", "a", "b")))
            z_new, born = add_pole(model, zc, pool, generator=g)
            did = born & bad
            if not bool(did.any()):
                break
            ps2, _, u2 = refine_samples(model, z_new[did], norm[did], beta, xf, target[did],
                                        sf.expand(S, -1)[did], W, lam, lm_iters=60, return_coords=True)
            c_new = chi2_with(fwd, ps2)
            better = c_new < c2[did]
            rows = torch.nonzero(did).flatten()[better]
            for k in ("eps", "gamma", "a", "b"):
                getattr(ps, k)[rows] = getattr(ps2, k)[better].to(getattr(ps, k).dtype)
            zc[rows] = torch.cat([u2, z_new[did][..., 4:5]], -1)[better]
            c2[rows] = c_new[better]
            n_born[rows] += 1
        return ps, c2, n_born

    def floor_share(ps, c2, gamma_min):
        ok = c2 <= band_edge
        ok = ok if bool(ok.any()) else torch.ones_like(ok)
        return weight_at_width_floor(ps.a[ok], ps.gamma[ok], gamma_min), float(ok.float().mean())

    auto = isinstance(floor_factor, str)
    if auto and floor_factor != "auto":
        raise ValueError(f"floor_factor={floor_factor!r}; expected a number >= 1 or 'auto'")
    gmin_trained = flow.coords.gamma_min
    factor, floor_diag = 1.0, {}
    if auto or float(floor_factor) == 1.0:
        ps, c2, n_born = fit(flow, z)
        share, frac_trained = floor_share(ps, c2, gmin_trained)
        if auto and share >= FLOOR_TRIGGER:
            factor = FLOOR_FACTOR
            floor_diag = {"weight_at_trained_width_floor": share,
                          "accepted_fraction_trained_floor": frac_trained,
                          "chi2_median_trained_floor": float(c2.median())}
    else:
        factor = float(floor_factor)
    if factor != 1.0:
        low = LoweredFloor(flow, factor)
        ps, c2, n_born = fit(low, low.recode(z))
    gmin_fit = gmin_trained / factor
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
    chi2_sel = c2[acc]
    count_diag = {}
    if count_model is not None:
        from .models.count import resample_by_count
        p_n = count_model.probs(G, x, beta, sigma=sigma)[0]
        before = float(n_pres.float().mean())
        take, count_diag = resample_by_count(p_n, n_pres, generator=g, floor=count_floor)
        sel = PoleSet(*(getattr(sel, k)[take] for k in ("eps", "gamma", "a", "b")))
        n_pres, chi2_sel = n_pres[take], chi2_sel[take]
        count_diag.update(count_mean_before=before, count_mean_after=float(n_pres.float().mean()))
        if count_diag["count_mass_covered"] < 0.5:
            warnings.append(f"the samples contain pole counts holding only "
                            f"{count_diag['count_mass_covered']:.0%} of the count posterior; "
                            "draw more samples.")
        elif count_diag["count_ess"] < 8:
            warnings.append(f"only {count_diag['count_ess']:.0f} effective samples after matching "
                            "the pole-count posterior; draw more samples.")
    if float(n_pres.float().median()) >= flow.P:
        warnings.append(f"the accepted samples use all {flow.P} pole slots; the spectrum may need "
                        "more structure than the model can place.")
    at_floor = weight_at_width_floor(sel.a, sel.gamma, gmin_fit)
    narrower = ("the data favour lines narrower than the model can draw. Positions and weights of "
                "those lines are usable; their widths and tails are the model's, and a method "
                "without a width floor (MaxEnt) may be sharper here.")
    if factor != 1.0:
        a64 = sel.a.double()
        floor_diag["weight_below_trained_width_floor"] = float(
            ((a64 * (sel.gamma.double() < gmin_trained)).sum(-1) / a64.sum(-1).clamp_min(1e-300)).median())
        if floor_diag.get("weight_at_trained_width_floor") is not None:
            warnings.append(f"{floor_diag['weight_at_trained_width_floor']:.0%} of the spectral weight sat "
                            f"on poles at the model's minimum width, so the samples were fitted again "
                            f"with that minimum divided by {factor:g} ({gmin_fit / float(beta):.3g} at "
                            "this beta). The flow was trained on nothing that narrow: those lines are "
                            "placed by the fit to the data, not by the learned prior, and their shape "
                            "is not determined.")
        if at_floor >= FLOOR_TRIGGER:
            warnings.append(f"{at_floor:.0%} of the spectral weight still sits at the lowered minimum "
                            f"width ({gmin_fit / float(beta):.3g} at this beta): " + narrower)
    elif at_floor >= FLOOR_TRIGGER:
        warnings.append(f"{at_floor:.0%} of the spectral weight sits on poles at the model's minimum "
                        f"width ({gmin_fit / float(beta):.3g} at this beta): " + narrower
                        + " floor_factor='auto' fits again with a lower minimum.")
    # narrow lines: a tenth or more of the weight on poles no wider than the trained floor.  The
    # data do not fix the shape of such a line, and the weight of a window with one near its edge
    # depends on it, so the window intervals below span both readings: as fitted, and as sharp lines
    narrow_share = weight_at_width_floor(sel.a, sel.gamma, gmin_trained)
    line_shape = (narrow_share >= FLOOR_TRIGGER
                  or floor_diag.get("weight_at_trained_width_floor", 0.0) >= FLOOR_TRIGGER)
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
    table = (getattr(flow, "meta", {}).get("band_calibration") if isinstance(band_calibration, str)
             else band_calibration)
    # the table is fitted and tested on draws of the prior, where the samples are over-dispersed;
    # on line spectra they are not (TRAINING.md (xxii)), so it is withheld when lines are flagged
    withheld = bool(table) and (factor != 1.0 or line_shape)
    if withheld:
        table = None
    lev = band_levels(table, rel_stat, float(beta))
    n_sel = int(sel.a.shape[0])
    if lev[0.9] * (n_sel + 1) > n_sel - 1:
        warnings.append(f"only {n_sel} samples to form the intervals from: the 90% window intervals "
                        "are the range of the samples and hold less than that. Draw more samples.")
    q = np.quantile(A, [0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95], axis=0)
    band = dict(zip(("q05", "q16", "q25", "q50", "q75", "q84", "q95"), q), mean=A.mean(0))

    win = []
    if windows:
        # exact window weights of each sample from the pole CDF
        eps, gam, a, b = (getattr(p64, k) for k in ("eps", "gamma", "a", "b"))

        narrow = gam <= FLOOR_WITHIN * gmin_trained

        def cdf(w, sharp=False):
            if math.isinf(w):
                return torch.zeros(eps.shape[0], dtype=torch.float64) if w < 0 else a.sum(-1)
            u = w * beta - eps
            absorptive = torch.atan(u / gam) / math.pi + 0.5
            if sharp:                                  # a narrow pole read as a line at its position
                absorptive = torch.where(narrow, (u > 0).to(absorptive.dtype), absorptive)
            return (a * absorptive - b * torch.log(u * u + gam * gam) / (2.0 * math.pi)).sum(-1)

        tot = a.sum(-1)
        pf = (prior_free_bounds(Gf[0], xf, beta, sf[0], W, windows,
                                norm=float(G[0, 0] + G[0, -1]))
              if bounds else [None] * len(windows))
        for (lo_w, hi_w), bnd in zip(windows, pf):
            wgt = ((cdf(hi_w) - cdf(lo_w)) / tot).numpy()
            (q05, q95), (q25, q75) = central_interval(wgt, lev[0.9]), central_interval(wgt, lev[0.5])
            row = {"window": (float(lo_w), float(hi_w)), "mean": float(wgt.mean()),
                   "q05": q05, "q95": q95, "q25": q25, "q75": q75}
            if line_shape:
                ws = ((cdf(hi_w, True) - cdf(lo_w, True)) / tot).numpy()
                (s05, s95), (s25, s75) = central_interval(ws, lev[0.9]), central_interval(ws, lev[0.5])
                row.update(as_fitted=(q05, q95), sharp_lines=(s05, s95), q05=min(q05, s05),
                           q95=max(q95, s95), q25=min(q25, s25), q75=max(q75, s75))
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
        chi2=chi2_sel.numpy(), chi2_target=float(chi2_target), n_samples=S,
        n_accepted=int(acc.sum()), n_kept_raw=int((acc & fits_raw).sum()), windows=win,
        warnings=warnings, prior=prior,
        diagnostics={"accepted_fraction": frac, "raw_chi2_median": float(c_raw.median()),
                     "samples_given_a_pole": int((n_born[acc] > 0).sum()),
                     "n_poles_median": float(n_pres.float().median()),
                     "weight_at_width_floor": at_floor, "floor_factor": factor, **floor_diag,
                     "relative_error_bar": rel_stat, "sigma_floor": sigma_floor,
                     "bootstrap_redraw": bins is not None, "band_calibrated": bool(table),
                     "band_table_withheld": withheld, "line_shape_readings": bool(line_shape),
                     "weight_on_narrow_poles": narrow_share,
                     "band_levels": lev, **count_diag})
