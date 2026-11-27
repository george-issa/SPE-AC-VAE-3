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

Measured limits (TRAINING.md (xxii), (xxiii)).  On draws of the prior and
on exact lattice spectra the window intervals hold the truth at their stated
rate.  On exactly diagonalised line spectra they hold it at 0.54 / 0.92 (for
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
from .physics.poles import PoleSet, line_density

__all__ = ["PosteriorReport", "continue_posterior", "prior_free_bounds", "BOX_SIGMA",
           "DEFAULT_WINDOWS", "BAND_TARGETS", "band_levels", "fit_band_table",
           "weight_at_width_floor", "FLOOR_TRIGGER", "FLOOR_FACTOR", "FLOOR_WITHIN", "FLOOR_FOLLOW",
           "central_interval", "continue_measured", "DISCRETE_DRAWS", "BOUNDS_TIME_LIMIT",
           "LINE_FIT", "LINES_VIABLE", "LINE_ROUNDS", "PEAKS_FIT_BELOW"]


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
#: Line positions drawn per pole and sample for the discrete reading of ``continue_posterior``.
DISCRETE_DRAWS = 16
#: When narrow lines are flagged, ``floor_factor="auto"`` can fit the poles the flow pinned at
#: its width floor as zero-width lines (delta functions): it does when the spectrum is stated to
#: be discrete, and when peaks of the minimum width reach the data for fewer than
#: :data:`PEAKS_FIT_BELOW` of the samples (the data themselves ask for something narrower).
#: ``False`` never does, which is the report judged in TRAINING.md (xxii)-(xxx): narrow poles
#: as Lorentzians of a quarter of the floor.
LINE_FIT = True
#: Share of samples below which the fit at the trained width floor counts as rejected by the data.
PEAKS_FIT_BELOW = 0.5
#: Share of the flow's proposals that must settle on the data with zero-width lines for that
#: fit to be the one reported; below it the lowered floor is fitted as well.
LINES_VIABLE = 0.25
#: How many times the line fit turns poles it has narrowed below the trained floor into lines.
LINE_ROUNDS = 2
#: In a fit with a lowered floor, poles the flow pinned at its trained floor start pinned at the
#: lowered one (``LoweredFloor.recode(follow_within=FLOOR_WITHIN)``).  ``False`` starts every
#: pole at the width the flow gave it, which is what TRAINING.md (xxii) judged.
FLOOR_FOLLOW = True


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
#: Seconds one linear program of :func:`prior_free_bounds` may take before its window is given up.
BOUNDS_TIME_LIMIT = 30.0
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

    No bound on the support either: beyond the quadrature (``|omega| <=
    omega_max``) the measure may sit on log-spaced frequencies out to
    ``1e4 * omega_max`` and at :math:`\pm\infty`, where it reaches only
    :math:`G(0)` or :math:`G(\beta)`.  Without those nodes a spectrum with
    weight outside the quadrature is excluded by construction: measured on a
    lattice spectrum with a Lorentzian tail at beta = 1 (7% of its weight
    beyond ``omega_max``), the exact weight of a window lay outside its
    bracket (TRAINING.md (xxviii)).

    Returns one dict per window: ``lo``, ``hi`` (fractions of the total
    weight), ``window`` and ``feasible``.  ``feasible`` is false, and the
    bounds NaN, when a program has no solution or is not solved within
    :data:`BOUNDS_TIME_LIMIT` seconds (seen on one spectrum in 44 at
    beta = 20 with error bars of 1e-5 and 401 time points, where a program
    otherwise takes four seconds and that one ran for over ten minutes).
    """
    from scipy.optimize import linprog

    from .physics.kernel import fermi_kernel

    G = torch.as_tensor(G, dtype=torch.float64).reshape(-1)
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    s = torch.as_tensor(sigma, dtype=torch.float64).reshape(-1)
    quad = SpectralQuadrature.for_beta(float(beta), dtype=torch.float64)
    nu = quad.nodes
    use = (nu.abs() <= omega_max * beta)
    K = PoleGreens(quad).kernel_matrix(x)[use]                   # (M, L)
    far = omega_max * torch.logspace(0.0, 4.0, 49, dtype=torch.float64)[1:]
    far = torch.cat([-far.flip(0), far])
    at_inf = torch.stack([(x == 1.0), (x == 0.0)]).to(K.dtype)   # weight at -inf, at +inf
    K = torch.cat([K, fermi_kernel(x.reshape(1, -1), (far * beta).reshape(-1, 1)), at_inf])
    big = float(np.finfo(float).max)                             # +-inf, inside the open-ended windows
    om = np.concatenate([(nu[use] / beta).numpy(), far.numpy(), [-big, big]])
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
                        bounds=(0, None), method="highs", options={"time_limit": BOUNDS_TIME_LIMIT})
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

    poles: PoleSet                       # accepted samples, physical units; gamma == 0 marks a
    #                                      zero-width line (a delta function of weight a)
    omega: np.ndarray
    band: dict                           # q05, q16, q25, q50, q75, q84, q95, mean, lines on omega.
    #                                      The quantiles are plain sample quantiles of A(omega),
    #                                      pointwise (the band table applies to `windows` only),
    #                                      of everything but the zero-width lines, which have no
    #                                      height.  `lines` is the mean weight those lines put on
    #                                      each grid node (zero without them); `mean` is the
    #                                      posterior-mean density with that weight included, so
    #                                      its integral over the grid is the total weight.
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
            f"{self.diagnostics.get('samples_given_a_pole', 0)} were given an extra pole"
            + (f", {self.diagnostics['samples_restarted']} were restarted from a sample that fits"
               if self.diagnostics.get("samples_restarted") else "") + ")",
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
                         + ("to a spectrum stated to be discrete" if self.diagnostics.get("discrete_reading")
                            else "when the data favour narrow lines")
                         + " (it was calibrated on spectra without them)")
        else:
            lines.append("  intervals: plain sample quantiles, not calibrated (measured conservative: a "
                         "90% window interval holds the truth about 93-96% of the time)")
        if self.windows:
            if self.diagnostics.get("discrete_reading"):
                lines.append("  the spectrum was stated to be discrete: "
                             + (f"{self.diagnostics.get('weight_on_lines', 0.0):.0%} of the weight is on "
                                "zero-width lines, and " if self.diagnostics.get("line_fit") else "")
                             + "the window intervals span both readings of every other pole, as fitted and "
                               "as one sharp line within its half-width")
            elif self.diagnostics.get("line_fit"):
                lines.append(f"  {self.diagnostics.get('weight_on_lines', 0.0):.0%} of the weight is on "
                             "zero-width lines, fitted because peaks of the model's minimum width do not "
                             "reach the data")
            elif self.diagnostics.get("line_shape_readings"):
                lines.append("  the window intervals span both readings of the narrow poles: as fitted, "
                             "and as sharp lines")
            lines.append("  weight per window: posterior 5-95%  |  allowed by the data alone  "
                         "|  share of the bracket")
            for w in self.windows:
                lo, hi = w["window"]
                name = f"[{lo:g}, {hi:g})".replace("inf", "∞")
                lines.append(f"    {name:14s} {w['q05']:.3f}-{w['q95']:.3f}  |  "
                             + (f"{w['lo']:.3f}-{w['hi']:.3f}  |  {w['share']:.2f}" if w["lo"] == w["lo"]
                                else "not computed"))
            sh = [w["share"] for w in self.windows if w["share"] == w["share"]]
            if sh:
                lines.append(f"  the posterior spans a median {np.median(sh):.0%} of what the data alone "
                             "allow per window; the rest of the localisation is the prior's")
        for msg in self.warnings:
            lines.append(f"  WARNING  : {msg}")
        if not self.warnings:
            lines.append("  warnings : none")
        return "\n".join(lines)

    def line_cells(self, n: int = 240, omega_max: float | None = None):
        """The zero-width lines for drawing: ``(centres, weight)`` of ``n`` equal cells over
        ``[-omega_max, omega_max]`` (default: the grid), the mean weight the lines put in each."""
        w = np.asarray(self.omega)
        lim = float(np.abs(w).max()) if omega_max is None else float(omega_max)
        edges = np.linspace(-lim, lim, int(n) + 1)
        cells = np.histogram(w, bins=edges, weights=np.asarray(self.band.get("lines", np.zeros_like(w))))[0]
        return 0.5 * (edges[1:] + edges[:-1]), cells

    def save(self, path) -> None:
        """Write the report to ``path`` (.npz) and ``path + ".json"``.

        The .npz holds the frequency grid, the pointwise band (``A_lines``:
        the weight of the zero-width lines per grid node), the accepted
        pole sets (one row per sample, physical units; a half-width of zero
        marks a zero-width line) and their misfits; the
        .json holds the window table, the warnings, the diagnostics and the
        text of :meth:`summary`.
        """
        import json

        p = self.poles
        np.savez(str(path), omega=self.omega, **{f"A_{k}": np.asarray(v) for k, v in self.band.items()},
                 poles_omega=p.eps.double().numpy(), poles_gamma=p.gamma.double().numpy(),
                 residues_re=p.a.double().numpy(), residues_im=p.b.double().numpy(),
                 chi2=self.chi2, chi2_target=self.chi2_target)
        plain = (int, float, str, bool, dict, list, tuple, type(None))
        with open(str(path) + ".json", "w") as fh:
            json.dump({"n_samples": self.n_samples, "n_accepted": self.n_accepted,
                       "chi2_over_target_median": float(np.median(self.chi2) / self.chi2_target),
                       "prior": self.prior, "warnings": self.warnings, "windows": self.windows,
                       "diagnostics": {k: v for k, v in self.diagnostics.items() if isinstance(v, plain)},
                       "summary": self.summary()}, fh, indent=1, default=float)

    def plot(self, path, title: str | None = None, omega_max: float | None = None) -> None:
        """Draw the report: the band of A(omega), the misfit of the samples, the window weights."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        blue, ink, pale = "#2a78d6", "#0b0b0b", "#d9d8d3"
        w, b = np.asarray(self.omega), self.band
        has_lines = bool(np.any(np.asarray(b.get("lines", 0.0)) > 0))
        if omega_max is None:                      # the range holding all but 1% of the mean weight
            c = np.cumsum(b["mean"])
            c = c / max(c[-1], 1e-300)
            omega_max = 1.25 * max(abs(w[np.searchsorted(c, 0.005)]),
                                   abs(w[min(np.searchsorted(c, 0.995), len(w) - 1)]), 1e-6)
        m = np.abs(w) <= omega_max
        st = slice(None, None, max(1, int(m.sum() // 3000)))
        f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True,
                                       gridspec_kw={"width_ratios": [1.6, 0.8, 1.3]})
        if has_lines:
            # zero-width lines have no height: the band is that of the rest of the spectrum, and
            # the lines are drawn as the mean weight they put in each of 240 cells (right-hand axis)
            centres, cells = self.line_cells(240, omega_max)
            ar = a0.twinx()
            on = cells > 1e-4 * max(float(cells.max()), 1e-300)
            ar.vlines(centres[on], 0.0, cells[on], color=ink, lw=1.6)
            ar.set_ylim(0.0, 1.15 * max(float(cells.max()), 1e-6))
            ar.set_ylabel("weight of the zero-width lines, per cell")
            ar.spines["top"].set_visible(False)
            a0.plot([], [], color=ink, lw=1.6, label="zero-width lines (right axis)")
        a0.fill_between(w[m][st], b["q05"][m][st], b["q95"][m][st], color=blue, alpha=0.25, lw=0,
                        label="90% band (plain sample quantiles)" + (" of the rest" if has_lines else ""))
        a0.plot(w[m][st], b["q50"][m][st], color=blue, lw=1.5, label="median")
        top = max(float(b["q95"][m].max()), 1e-6)
        a0.set_ylim(-0.03 * top, 1.15 * top)
        a0.set_xlabel(r"$\omega$")
        a0.set_ylabel(r"$A(\omega)$")
        a0.set_title(f"spectrum; {self.n_accepted} of {self.n_samples} samples fit the data", fontsize=9.5)
        a0.legend(fontsize=8, frameon=False, loc="upper left")
        r = self.chi2 / self.chi2_target
        a1.hist(r, bins=np.linspace(min(0.6, r.min()), max(1.6, r.max()), 28), color=blue, alpha=0.75)
        a1.axvline(1.0, color=ink, lw=1.2)
        a1.set_xlabel(r"$\chi^2$ / value a perfect fit scores")
        a1.set_ylabel("accepted samples")
        a1.set_title(f"misfit of each sample (median {np.median(r):.2f}); line = perfect fit", fontsize=9.5)
        y = np.arange(len(self.windows))[::-1]
        for yy, row in zip(y, self.windows):
            if row["lo"] == row["lo"]:
                a2.plot([row["lo"], row["hi"]], [yy, yy], color=pale, lw=9, solid_capstyle="butt")
            a2.plot([row["q05"], row["q95"]], [yy, yy], color=blue, lw=9, alpha=0.45, solid_capstyle="butt")
            a2.plot([row["q25"], row["q75"]], [yy, yy], color=blue, lw=9, solid_capstyle="butt")
        a2.set_yticks(y, [f"[{lo:g}, {hi:g})".replace("inf", "∞") for lo, hi in
                          (row["window"] for row in self.windows)], fontsize=8.5)
        a2.set_xlabel("spectral weight in the frequency window")
        a2.plot([], [], color=pale, lw=7, label="allowed by the data alone")
        a2.plot([], [], color=blue, lw=7, alpha=0.45, label="posterior 90% interval")
        a2.plot([], [], color=blue, lw=7, label="posterior 50% interval")
        a2.legend(fontsize=8, frameon=False, loc="best")
        d = self.diagnostics
        kind = ("calibrated intervals" if d.get("band_calibrated") else
                "zero-width lines fitted" if d.get("line_fit") else
                "both readings of the narrow lines" if d.get("line_shape_readings") else "plain sample quantiles")
        a2.set_title(f"weight per window ({kind})", fontsize=9.5)
        for ax in (a0, a1, a2):
            ax.grid(color="#e4e3df", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        n = len(self.warnings)
        f.suptitle((title + "  |  " if title else "") + ("no warnings" if n == 0 else
                   f"{n} warning{'s' if n > 1 else ''}: see the printed report"), fontsize=10.5)
        f.savefig(str(path), dpi=150)
        plt.close(f)


def continue_posterior(flow, G, x, beta: float, sigma, inv_sqrt_R=None, bins=None,
                       n_samples: int = 200, omega=None, chi2_target: float = 1.0,
                       windows=DEFAULT_WINDOWS, bounds: bool = True, accept_sigma: float = 3.0,
                       lam: float = 10.0, generator: torch.Generator | None = None,
                       x_fit=None, G_fit=None, sigma_fit=None,
                       max_births: int = 2, sigma_floor: float | None = None,
                       count_model=None, count_floor: float = 0.02,
                       band_calibration="auto", floor_factor="auto",
                       discrete: bool = False) -> PosteriorReport:
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
        every case; in the second fit a pole the flow pinned at its floor
        starts pinned at the lowered one (:data:`FLOOR_FOLLOW`).  Measured
        (TRAINING.md (xxii), (xxiii)): the second fit is run on none of 100
        exact lattice spectra and on 10% of held-out prior draws, where it
        changes neither error nor coverage; on exact line spectra it is run
        on 4 in 10 and lowers the error on those by 18% on average, at four
        to five times the fitting time.

        Whenever a tenth or more of the weight is on poles no wider than
        :data:`FLOOR_WITHIN` trained floors, the window intervals span both
        readings of those poles -- as fitted, and as sharp lines at their
        positions -- and the band table is not applied: the data do not fix
        the shape of a line that narrow, and the weight of a window with
        such a line near its edge depends on it.

        *Zero-width lines.*  A Lorentzian of any width has tails, and at
        error bars near 1e-5 the data see them: a sharp line then cannot
        be fitted with one, half the samples are rejected, and the tails
        are most of the error.  So when the narrow poles are flagged and
        the fit at the trained floor reaches the data for fewer than
        :data:`PEAKS_FIT_BELOW` of the samples -- the data themselves ask
        for something narrower -- the poles the flow pinned at its floor
        are fitted as zero-width lines instead (delta functions: a position
        and a weight; ``gamma == 0`` in the reported poles), with the other
        poles free to narrow to the lowered floor; a pole the fit then
        narrows below the trained floor is tried as a line too.  If at least
        :data:`LINES_VIABLE` of the flow's proposals settle on the data that
        way, the rest are started again from one that did and that fit is
        the report; otherwise the lowered floor is fitted as before.  The
        same is done whenever ``discrete`` is set.  Window intervals of a
        report with lines are never narrower than the relative error bar of
        the data.  Measured (TRAINING.md (xxxi)) on exact line spectra at
        beta = 8, error bars 1e-5: W1 0.105 -> 0.042 against MaxEnt's 0.068.
    discrete
        State that the spectrum is a set of discrete lines (an atom, a
        molecule, a finite cluster).  Poles the flow pins at its width
        floor are then fitted as zero-width lines whenever narrow lines are
        flagged (see ``floor_factor``), whether or not the data could tell
        a line from a narrow peak; the other samples are fitted as always.
        The window intervals span both readings of every pole that is not
        a fitted line -- as
        fitted, and as one sharp line somewhere within one half-width of
        its centre (drawn uniformly, :data:`DISCRETE_DRAWS` times per
        sample) -- and the band table is not applied.  In the second
        reading a pole that straddles a window edge puts all of its weight
        on one side; that reading is not checked against the data, so its
        interval is cut to the range the data alone allow when that range
        is computed (``bounds``).  Without this statement a spectrum of
        many close lines is indistinguishable from a continuous one in the
        data, the prior favours the continuous one, and the intervals are
        too narrow for the lines, with nothing to flag.  Measured
        (TRAINING.md (xxx)) on exact line spectra the report does not
        flag: the stated 50% / 90% intervals hold 0.34 / 0.62 without the
        statement and 0.67 / 0.95 with it (0.29 / 0.61 and 0.70 / 0.94 at
        settings not used to choose the reading), at 2.6 times the width;
        the 50% interval is conservative.  On a continuous spectrum the
        statement is wrong and costs intervals ten times wider than
        needed.  The band of :math:`A(\omega)` is not changed.
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
        d = (fwd(ps, xf) - Gf) / sf
        d = d @ W.T.to(d.dtype)
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

    def fit(model, z_start, lines=None):
        """The fit stage in the coordinates of ``model``: the flow, or a LoweredFloor on it.
        With ``lines`` (samples, poles), those poles are fitted as zero-width lines, in float64."""
        fwd = fwd0 if model is flow else PoleGreens(
            SpectralQuadrature.for_beta(float(beta), gamma_min=model.coords.gamma_min))
        nrm, sig = norm, sf.expand(S, -1)
        if lines is not None:
            z_start, nrm, lines = z_start.double(), norm.double(), lines.clone()
        ps, _, u = refine_samples(model, z_start, nrm, beta, xf, target, sig, W, lam,
                                  anneal_from=1e-2, return_coords=True, lines=lines)
        if lines is None:
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
            if lines is None:
                z_new, born = add_pole(model, zc, pool, generator=g)
                ln_new = None
            else:
                z_new, born, ln_new = add_pole(model, zc, pool, generator=g, lines=lines)
            did = born & bad
            if not bool(did.any()):
                break
            ps2, _, u2 = refine_samples(model, z_new[did], nrm[did], beta, xf, target[did], sig[did], W, lam,
                                        lm_iters=60, return_coords=True,
                                        lines=None if ln_new is None else ln_new[did])
            c_new = chi2_with(fwd, ps2)
            better = c_new < c2[did]
            rows = torch.nonzero(did).flatten()[better]
            for k in ("eps", "gamma", "a", "b"):
                getattr(ps, k)[rows] = getattr(ps2, k)[better].to(getattr(ps, k).dtype)
            zc[rows] = torch.cat([u2, z_new[did][..., 4:5]], -1)[better]
            c2[rows] = c_new[better]
            n_born[rows] += 1
            if ln_new is not None:
                lines[rows] = ln_new[rows]
        if lines is None:
            return ps, c2, n_born
        # a pole the fit has narrowed below the trained floor is tried as a line as well; the
        # change is kept for a sample when it then fits the data, or fits them better than before
        for _ in range(int(LINE_ROUNDS)):
            more = (ps.gamma > 0) & (ps.gamma <= FLOOR_WITHIN * gmin_trained) & (ps.a != 0)
            did = more.any(-1)
            if not bool(did.any()):
                break
            ln_new = lines | more
            ps2, _, u2 = refine_samples(model, zc[did], nrm[did], beta, xf, target[did], sig[did], W, lam,
                                        lm_iters=60, return_coords=True, lines=ln_new[did])
            c_new = chi2_with(fwd, ps2)
            better = (c_new <= band_edge) | (c_new < c2[did])
            rows = torch.nonzero(did).flatten()[better]
            if rows.numel() == 0:
                break
            for k in ("eps", "gamma", "a", "b"):
                getattr(ps, k)[rows] = getattr(ps2, k)[better]
            zc[rows, :, :4] = u2[better]
            c2[rows] = c_new[better]
            lines[rows] = ln_new[rows]

        def settled(p, c):
            """Fits the data, with no pole left that is narrower than the trained floor yet not a line."""
            return (c <= band_edge) & ~((p.gamma > 0) & (p.gamma <= FLOOR_WITHIN * gmin_trained)
                                        & (p.a != 0)).any(-1)
        good = settled(ps, c2)
        info = {"fraction": float(good.float().mean()), "restarted": torch.zeros(S, dtype=torch.bool)}
        # a sample the fit left elsewhere starts again from one that settled (its poles and which
        # of them are lines) and is fitted to its own redraw of the data
        bad = ~good
        if info["fraction"] >= LINES_VIABLE and bool(bad.any()):
            src = torch.nonzero(good).flatten()
            src = src[torch.randint(src.numel(), (int(bad.sum()),), generator=g)]
            ps2, _, u2 = refine_samples(model, zc[src], nrm[bad], beta, xf, target[bad], sig[bad], W, lam,
                                        lm_iters=60, return_coords=True, lines=lines[src])
            c_new = chi2_with(fwd, ps2)
            took = settled(ps2, c_new)
            rows = torch.nonzero(bad).flatten()[took]
            for k in ("eps", "gamma", "a", "b"):
                getattr(ps, k)[rows] = getattr(ps2, k)[took]
            c2[rows] = c_new[took]
            n_born[rows] = n_born[src][took]
            info["restarted"][rows] = True
            good = settled(ps, c2)
        info["settled"] = good
        return ps, c2, n_born, info

    def floor_share(ps, c2, gamma_min):
        ok = c2 <= band_edge
        ok = ok if bool(ok.any()) else torch.ones_like(ok)
        return weight_at_width_floor(ps.a[ok], ps.gamma[ok], gamma_min), float(ok.float().mean())

    auto = isinstance(floor_factor, str)
    if auto and floor_factor != "auto":
        raise ValueError(f"floor_factor={floor_factor!r}; expected a number >= 1 or 'auto'")
    gmin_trained = flow.coords.gamma_min
    factor, floor_diag, line_diag, wide, usable = 1.0, {}, {}, None, None
    kept_raw = fits_raw.clone()                     # samples reported exactly as the flow drew them
    flagged = False
    if auto or float(floor_factor) == 1.0:
        ps, c2, n_born = fit(flow, z)
        share, frac_trained = floor_share(ps, c2, gmin_trained)
        frac_peaks = float((c2 <= band_edge).float().mean())    # 0 when no sample fits at this floor
        if auto and share >= FLOOR_TRIGGER:
            flagged, factor = True, FLOOR_FACTOR
            floor_diag = {"weight_at_trained_width_floor": share,
                          "accepted_fraction_trained_floor": frac_trained,
                          "chi2_median_trained_floor": float(c2.median())}
    else:
        factor = float(floor_factor)
    if factor != 1.0:
        low = LoweredFloor(flow, factor)
        z_low = low.recode(z, follow_within=FLOOR_WITHIN if FLOOR_FOLLOW else None)
        if flagged and LINE_FIT and (discrete or frac_peaks < PEAKS_FIT_BELOW):
            # the samples put weight on poles at the model's minimum width, and either the user
            # states that the spectrum is a set of lines or peaks of that width do not reach the
            # data.  A Lorentzian of any width has tails, and at small error bars a sharp line
            # cannot be fitted with one: fit the same proposals, against the same redraws, with
            # the poles the flow pinned at its floor as zero-width lines (the other poles may
            # narrow to the lowered floor)
            ok1 = c2 <= band_edge
            wide = PoleSet(*(getattr(ps, k)[ok1].double() for k in ("eps", "gamma", "a", "b")))
            pinned = (raw.gamma <= FLOOR_WITHIN * gmin_trained) & (raw.a != 0)
            ps_l, c2_l, born_l, info = fit(low, z_low, lines=pinned)
            ok_l = info["settled"]
            line_diag = {"accepted_fraction_as_lines": info["fraction"],
                         "samples_restarted": int((info["restarted"] & ok_l).sum()),
                         "samples_fitting_at_trained_floor": int(ok1.sum())}
            kept_raw = kept_raw & ~ok_l
            if info["fraction"] >= LINES_VIABLE:
                ps, c2, n_born, usable = ps_l, c2_l, born_l, ok_l
            else:                           # lines do not reach the data: the lowered floor, as before,
                ps, c2, n_born = fit(low, z_low)        # and the line fit for the samples it settled
                line_diag["accepted_fraction_lowered_floor"] = float((c2 <= band_edge).float().mean())
                pick = ok_l.unsqueeze(-1)
                ps = PoleSet(*(torch.where(pick, getattr(ps_l, k), getattr(ps, k).double())
                               for k in ("eps", "gamma", "a", "b")))
                c2 = torch.where(ok_l, c2_l, c2.to(c2_l.dtype))
                n_born = torch.where(ok_l, born_l, n_born)
                line_diag["samples_mixed"] = bool(ok_l.any())
        else:
            ps, c2, n_born = fit(low, z_low)
    gmin_fit = gmin_trained / factor
    acc = (c2 <= band_edge) if usable is None else usable
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
    is_line = sel.gamma == 0
    on_lines = float(((sel.a.double() * is_line).sum(-1) / sel.a.double().sum(-1).clamp_min(1e-300)).median())
    at_floor = weight_at_width_floor(torch.where(is_line, torch.zeros_like(sel.a), sel.a), sel.gamma, gmin_fit)
    narrower = ("the data favour lines narrower than the model can draw. Positions and weights of "
                "those lines are usable; their widths and tails are the model's, and a method "
                "without a width floor (MaxEnt) may be sharper here.")
    if factor != 1.0:
        a64 = sel.a.double()
        floor_diag["weight_below_trained_width_floor"] = float(
            ((a64 * (sel.gamma.double() < gmin_trained)).sum(-1) / a64.sum(-1).clamp_min(1e-300)).median())
        if bool(is_line.any()):
            n1 = line_diag["samples_fitting_at_trained_floor"]
            msg = (f"{floor_diag['weight_at_trained_width_floor']:.0%} of the spectral weight sat on poles at "
                   f"the model's minimum width ({gmin_trained / float(beta):.3g} at this beta)")
            if n1 < PEAKS_FIT_BELOW * S:
                msg += (f", and peaks of that width fit the data for {n1} of {S} samples only: the data ask "
                        "for narrower features, so those poles were fitted as zero-width lines "
                        f"({line_diag['accepted_fraction_as_lines']:.0%} of the samples fit the data that way).")
            else:
                msg += (". The spectrum was stated to be discrete, so those poles were fitted as zero-width "
                        f"lines ({line_diag['accepted_fraction_as_lines']:.0%} of the samples fit the data "
                        f"that way); peaks of the minimum width fit as well ({n1} of {S} samples), so the "
                        "data alone would not tell a line from a narrow peak here.")
            msg += (" The flow was trained on nothing that narrow: the positions and weights of those lines "
                    "are set by the fit to the data.")
            if line_diag.get("samples_mixed"):
                msg += (f" With the minimum width divided by {factor:g} instead, "
                        f"{line_diag['accepted_fraction_lowered_floor']:.0%} of the samples fit; a sample "
                        "is reported with lines where that settled and with narrow peaks otherwise.")
            warnings.append(msg)
        elif floor_diag.get("weight_at_trained_width_floor") is not None:
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
    A = (p64.spectral(nu) * beta).numpy()                  # everything but the zero-width lines
    A_lines = (line_density(p64.eps, p64.gamma, p64.a, nu) * beta).numpy().mean(0)
    table = (getattr(flow, "meta", {}).get("band_calibration") if isinstance(band_calibration, str)
             else band_calibration)
    # the table is fitted and tested on draws of the prior, where the samples are over-dispersed;
    # on line spectra they are not (TRAINING.md (xxii)), so it is withheld when lines are flagged
    withheld = bool(table) and (factor != 1.0 or line_shape or bool(discrete))
    if withheld:
        table = None
    lev = band_levels(table, rel_stat, float(beta))
    n_sel = int(sel.a.shape[0])
    if lev[0.9] * (n_sel + 1) > n_sel - 1:
        warnings.append(f"only {n_sel} samples to form the intervals from: the 90% window intervals "
                        "are the range of the samples and hold less than that. Draw more samples.")
    q = np.quantile(A, [0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95], axis=0)
    cell = np.gradient(omega) if omega.size > 1 else np.ones_like(omega)
    cell[[0, -1]] *= 0.5 if omega.size > 1 else 1.0        # trapezoid cells, as line_density uses
    band = dict(zip(("q05", "q16", "q25", "q50", "q75", "q84", "q95"), q), mean=A.mean(0) + A_lines,
                lines=A_lines * cell)

    win = []
    if windows:
        # exact window weights of each sample from the pole CDF
        eps, gam, a, b = (getattr(p64, k) for k in ("eps", "gamma", "a", "b"))

        def weights(p, lo_w, hi_w, sharp=False):
            """Share of each sample's weight in [lo_w, hi_w); with ``sharp``, a pole no wider than
            the trained floor is read as a line at its position (a zero-width line is one anyway)."""
            line = p.gamma == 0
            step = (line | (p.gamma <= FLOOR_WITHIN * gmin_trained)) if sharp else line
            gs = torch.where(line, torch.ones_like(p.gamma), p.gamma)

            def cdf(w):
                if math.isinf(w):
                    return torch.zeros(p.eps.shape[0], dtype=torch.float64) if w < 0 else p.a.sum(-1)
                u = w * beta - p.eps
                absorptive = torch.where(step, (u > 0).to(p.a.dtype), torch.atan(u / gs) / math.pi + 0.5)
                disp = torch.where(line, torch.zeros_like(u), torch.log(u * u + gs * gs))
                return (p.a * absorptive - p.b * disp / (2.0 * math.pi)).sum(-1)
            return ((cdf(hi_w) - cdf(lo_w)) / p.a.sum(-1)).numpy()

        # when the reported samples hold fitted lines, the fit at the trained floor is the other
        # reading of the same features, where it reaches the data for at least half the samples
        other = wide if (wide is not None and bool(is_line.any()) and not discrete
                         and 2 * wide.eps.shape[0] >= S) else None

        if discrete:
            # every pole as one sharp line, uniformly within one half-width of its centre: all
            # of its weight on one side of a window edge in any one draw.  Chosen on the exact
            # line spectra of five settings against wider and heavier-tailed placements and
            # against two or three lines per pole, scored on six others (TRAINING.md (xxx))
            t = 2.0 * torch.rand((DISCRETE_DRAWS,) + tuple(eps.shape), generator=g, dtype=torch.float64) - 1.0
            pos = eps + gam * t                                    # (draws, samples, poles)

            def cdf_lines(w):
                if math.isinf(w):
                    one = torch.zeros(eps.shape[0], dtype=torch.float64) if w < 0 else a.sum(-1)
                    return one.expand(DISCRETE_DRAWS, -1)
                u = w * beta - eps
                disp = torch.where(gam == 0, torch.zeros_like(u), torch.log(u * u + gam * gam))
                return (a * (w * beta > pos).to(a.dtype) - b * disp / (2.0 * math.pi)).sum(-1)

        tot = a.sum(-1)
        # samples made of lines can agree on a window's weight exactly (no line in it: zero), but
        # weight below the error bar of the data is invisible to them: no interval is narrower
        pad = rel_stat if bool(is_line.any()) else 0.0
        pf = (prior_free_bounds(Gf[0], xf, beta, sf[0], W, windows,
                                norm=float(G[0, 0] + G[0, -1]))
              if bounds else [None] * len(windows))
        for (lo_w, hi_w), bnd in zip(windows, pf):
            wgt = weights(p64, lo_w, hi_w)
            (q05, q95), (q25, q75) = central_interval(wgt, lev[0.9]), central_interval(wgt, lev[0.5])
            row = {"window": (float(lo_w), float(hi_w)), "mean": float(wgt.mean()),
                   "q05": q05, "q95": q95, "q25": q25, "q75": q75}
            if line_shape:
                ws = weights(p64, lo_w, hi_w, sharp=True)
                (s05, s95), (s25, s75) = central_interval(ws, lev[0.9]), central_interval(ws, lev[0.5])
                row.update(as_fitted=(q05, q95), sharp_lines=(s05, s95), q05=min(q05, s05),
                           q95=max(q95, s95), q25=min(q25, s25), q75=max(q75, s75))
                if other is not None:
                    wo = weights(other, lo_w, hi_w)
                    (o05, o95), (o25, o75) = central_interval(wo, lev[0.9]), central_interval(wo, lev[0.5])
                    row.update(minimum_width_peaks=(o05, o95), q05=min(row["q05"], o05),
                               q95=max(row["q95"], o95), q25=min(row["q25"], o25), q75=max(row["q75"], o75))
            if discrete:
                wd = ((cdf_lines(hi_w) - cdf_lines(lo_w)) / tot).reshape(-1).numpy()
                (d05, d95), (d25, d75) = central_interval(wd, lev[0.9]), central_interval(wd, lev[0.5])
                if bnd is not None and bnd["feasible"]:        # the line placements were not fitted
                    d05, d25, d75, d95 = (min(max(v, bnd["lo"]), bnd["hi"]) for v in (d05, d25, d75, d95))
                row.update(as_fitted=(q05, q95), discrete_lines=(d05, d95), q05=min(row["q05"], d05),
                           q95=max(row["q95"], d95), q25=min(row["q25"], d25), q75=max(row["q75"], d75))
            if pad:
                row.update(q05=row["q05"] - pad, q95=row["q95"] + pad, q25=row["q25"] - pad,
                           q75=row["q75"] + pad)
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
        unsolved = sum(b is not None and not b["feasible"] for b in pf)
        if unsolved:
            warnings.append(f"the range the data alone allow could not be computed for {unsolved} of "
                            f"{len(win)} windows (the linear program had no solution or did not finish); "
                            "the posterior intervals of those windows are not checked against it.")

    name = getattr(flow, "meta", {}).get("recipe", {}).get("name") if isinstance(
        getattr(flow, "meta", {}).get("recipe", None), dict) else None
    prior = ("the training prior of this checkpoint"
             + (f" ({name})" if name else "")
             + ": mixtures of pole features, smooth bands and exactly solvable models; "
               "positive, normalised, causal spectra with widths above the resolution floor")
    return PosteriorReport(
        poles=PoleSet(sel.eps / beta, sel.gamma / beta, sel.a, sel.b), omega=omega, band=band,
        chi2=chi2_sel.numpy(), chi2_target=float(chi2_target), n_samples=S,
        n_accepted=int(acc.sum()), n_kept_raw=int((acc & kept_raw).sum()), windows=win,
        warnings=warnings, prior=prior,
        diagnostics={"accepted_fraction": frac, "raw_chi2_median": float(c_raw.median()),
                     "samples_given_a_pole": int((n_born[acc] > 0).sum()),
                     "samples_restarted": line_diag.get("samples_restarted", 0),
                     "n_poles_median": float(n_pres.float().median()),
                     "weight_at_width_floor": at_floor, "floor_factor": factor,
                     "floor_follow": bool(FLOOR_FOLLOW and factor != 1.0), **floor_diag,
                     "line_fit": bool(is_line.any()), "weight_on_lines": on_lines, **line_diag,
                     "window_interval_floor": rel_stat if bool(is_line.any()) else 0.0,
                     "relative_error_bar": rel_stat, "sigma_floor": sigma_floor,
                     "bootstrap_redraw": bins is not None, "band_calibrated": bool(table),
                     "band_table_withheld": withheld, "line_shape_readings": bool(line_shape),
                     "discrete_reading": bool(discrete),
                     "weight_on_narrow_poles": narrow_share,
                     "band_levels": lev, **count_diag})


def continue_measured(flow, mg, min_eigenvalue: float = 1e-10, cov_method: str = "structured",
                      **kw) -> PosteriorReport:
    r""":func:`continue_posterior` for measured bins.

    ``mg`` is a :class:`spevae.data.real.MeasuredGreens` (bins of
    :math:`G(\tau)` as read by ``spevae.data``).  The covariance of the bin
    mean is estimated from the bins (``cov_method``, eigenvalues of the
    correlation floored at ``min_eigenvalue``), slices pinned by a symmetry
    (zero variance) are left out of the fit, and the samples are redrawn
    against bootstrap resamples of the bins.  Further keyword arguments go to
    :func:`continue_posterior`.
    """
    cov = mg.covariance(min_eigenvalue, method=cov_method)
    sl, n = cov["slices"], mg.n_bins
    return continue_posterior(flow, mg.mean, mg.x, mg.beta, mg.sigma_encoder / math.sqrt(n),
                              cov["inv_sqrt_R"], bins=mg.bins[:, sl], chi2_target=cov["chi2_target"],
                              x_fit=mg.x[sl], G_fit=mg.mean[sl], sigma_fit=cov["sigma"] / math.sqrt(n), **kw)
