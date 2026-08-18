r"""Figures for a three-way comparison.

Deliberately three panels, because the three questions a reader has are
different and a single overlay answers only the first:

1. **The spectra.**  Where do the methods agree, and where do they not?
2. **The disagreement band.**  Plotted as a shaded envelope across methods,
   because the *location* of the disagreement is the result -- it marks where the
   inversion is underdetermined.  An aggregate distance number throws that away.
3. **The reconstruction residuals**, in units of the data's own error.  A
   spectrum that does not reproduce :math:`G(\tau)` is not a candidate answer,
   however plausible it looks, and this is the panel where that shows.

``matplotlib`` is an optional dependency (``pip install spevae[plot]``); the
import is deferred so the rest of the harness runs without it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_comparison"]

#: Colour-blind-safe qualitative palette (Okabe--Ito), fixed per method so the
#: same method keeps its colour across every figure in a paper.
_COLOURS = {
    "spevae": "#0072B2",
    "maxent": "#D55E00",
    "deac": "#009E73",
    "deac-ref": "#CC79A7",
    "truth": "#000000",
}
_FALLBACK = ["#E69F00", "#56B4E9", "#F0E442", "#999999"]


def _colour(name: str, i: int) -> str:
    return _COLOURS.get(name, _FALLBACK[i % len(_FALLBACK)])


def plot_comparison(
    results,
    mg=None,
    truth: tuple[np.ndarray, np.ndarray] | None = None,
    omega_range: tuple[float, float] | None = None,
    show_ensemble: bool = True,
    title: str | None = None,
    path: str | None = None,
):
    r"""Overlay spectra, the cross-method spread, and reconstruction residuals.

    Parameters
    ----------
    results
        A list of :class:`~spevae.benchmark.result.ContinuationResult`.
    mg
        The :class:`~spevae.data.real.MeasuredGreens` they continue.  Required
        for the residual panel; omit it to draw only the spectra.
    truth
        ``(omega, A)`` when there is an exact answer.
    show_ensemble
        Shade each method's own uncertainty band (``A_std``) where it reports
        one.  Note these are not commensurable across methods -- DEAC's is a
        bin-to-bin spread, spevae's a latent ensemble -- so they are drawn
        faintly and the cross-method band is what carries the argument.
    path
        Save here instead of returning an interactive figure.

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.pyplot as plt

    n_panels = 3 if mg is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(8.0, 2.7 * n_panels),
                             sharex=False, constrained_layout=True)
    ax_A, ax_band = axes[0], axes[1]

    lo = min(float(r.omega[0]) for r in results)
    hi = max(float(r.omega[-1]) for r in results)
    if omega_range is not None:
        lo, hi = omega_range
    grid = np.linspace(lo, hi, 2001)

    if truth is not None:
        ax_A.plot(truth[0], truth[1], color=_COLOURS["truth"], lw=2.2,
                  ls="--", label="exact", zorder=5)

    curves = []
    for i, r in enumerate(results):
        c = _colour(r.method, i)
        ax_A.plot(r.omega, r.A, color=c, lw=1.6, label=r.method)
        if show_ensemble and r.A_std is not None:
            ax_A.fill_between(r.omega, r.A - r.A_std, r.A + r.A_std,
                              color=c, alpha=0.15, lw=0)
        curves.append(r.on_grid(grid))

    ax_A.set_ylabel(r"$A(\omega)$")
    ax_A.set_xlim(lo, hi)
    ax_A.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax_A.legend(frameon=False, ncol=len(results) + 1, fontsize=9)
    if title:
        ax_A.set_title(title, fontsize=11)

    # Cross-method spread: the honest uncertainty on a real dataset.
    stack = np.stack(curves)
    ax_band.fill_between(grid, stack.min(0), stack.max(0), color="0.75",
                         alpha=0.7, lw=0, label="spread across methods")
    ax_band.plot(grid, stack.mean(0), color="0.25", lw=1.4, label="mean")
    if truth is not None:
        ax_band.plot(truth[0], truth[1], color=_COLOURS["truth"], lw=1.6,
                     ls="--", label="exact")
    ax_band.set_ylabel(r"$A(\omega)$")
    ax_band.set_xlabel(r"$\omega$")
    ax_band.set_xlim(lo, hi)
    ax_band.legend(frameon=False, fontsize=9)
    ax_band.set_title("where the methods disagree, the data does not decide",
                      fontsize=9, loc="left", color="0.35")

    if mg is not None:
        ax_r = axes[2]
        tau = mg.tau.double().numpy()
        sem = mg.sigma_mean.double().numpy()
        for i, r in enumerate(results):
            if r.G_recon is None:
                continue
            ax_r.plot(tau, (r.G_recon - mg.mean.double().numpy()) / sem,
                      color=_colour(r.method, i), lw=1.3, label=r.method)
        # Symmetric log: a method sitting at 1 sigma and one at 300 sigma have to
        # be readable in the same panel, and a linear axis shows only the second.
        ax_r.set_yscale("symlog", linthresh=1.0)
        for k in (-1, 1):
            ax_r.axhline(k, color="0.6", lw=0.8, ls=":")
        ax_r.axhline(0.0, color="0.3", lw=0.8)
        if mg.includes_endpoints:
            # tau = beta is linearly dependent on tau = 0 through the sum rule,
            # so it is excluded from the covariance and from every fit; a
            # residual there is an extrapolation, not a fit failure.
            span = 0.012 * (tau[-1] - tau[0])
            ax_r.axvspan(tau[-1] - span, tau[-1] + span, color="0.85",
                         zorder=0, lw=0)
            ax_r.annotate("not fitted", xy=(tau[-1], 0), xytext=(-6, 6),
                          textcoords="offset points", ha="right", va="bottom",
                          fontsize=8, color="0.45")
        ax_r.set_xlabel(r"$\tau$")
        ax_r.set_ylabel(r"$(G_{\rm rec}-G)/\sigma_{\bar G}$")
        ax_r.set_title("reconstruction residuals in units of the data error "
                       "(symlog; dotted = 1 sigma)", fontsize=9, loc="left",
                       color="0.35")
        ax_r.legend(frameon=False, ncol=len(results), fontsize=9)

    if path:
        fig.savefig(path, dpi=150)
    return fig, axes
