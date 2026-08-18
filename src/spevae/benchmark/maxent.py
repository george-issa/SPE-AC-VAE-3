r"""Maximum-entropy analytic continuation, via the vendored `ana_cont`.

MaxEnt is the field's default method, so it is the baseline any new approach has
to be shown against.  It regularises the inversion by maximising the entropy of
:math:`A` relative to a default model :math:`D`,

.. math::
    Q_\alpha[A] = \tfrac12\chi^2[A] - \alpha S[A],
    \qquad S[A] = -\!\int\! d\omega\, A\log\frac{A}{D},

and picks :math:`\alpha` by the ``chi2kink`` heuristic.  Two properties of that
construction matter when reading a comparison table:

* **The default model is a prior**, exactly as this package's training
  distribution is.  A flat :math:`D` is the honest choice when nothing is known,
  and it is what is used here, but "MaxEnt has no prior" is not a defensible
  reading of the result.
* **The entropy term penalises structure**, so MaxEnt systematically broadens
  sharp features and can wash out high-frequency weight.  Disagreement with a
  method that does not share that bias is informative rather than a defect on
  either side.
* **The output window is part of the answer.**  A flat default model on a
  bounded :math:`\omega` range has entropy to gain by pushing weight towards the
  edges, and the result is a visible upturn at :math:`|\omega| \to
  \omega_{\max}` -- clearly present in the package's own benchmark figure at
  :math:`\omega = +10`.  It is an artefact of the window, not physics.  Widen
  ``omega_max`` past anywhere the data could place weight, and treat the extreme
  edges of a MaxEnt spectrum as unreliable.

Covariance
----------
``use_full_covariance=True`` passes the full :math:`(L,L)` matrix, which is more
informative but inverts badly when :math:`N_{\rm bins} \lesssim L_\tau` -- the
regime where :class:`~spevae.data.real.MeasuredGreens` reports a
:math:`\chi^2` target far below 1.  The diagonal path is the stabler, more
common choice.  The default here follows the data: full covariance when there
are comfortably more bins than time slices, diagonal otherwise, with the choice
recorded in ``meta`` so a table can say which was used.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from ..data.real import MeasuredGreens
from .result import ContinuationResult, _Timer

__all__ = ["maxent_available", "run_maxent"]


def maxent_available() -> bool:
    """Whether the vendored solver imports (it needs only numpy/scipy)."""
    try:
        import spevae._vendor.ana_cont.continuation  # noqa: F401
        return True
    except Exception:
        return False


def run_maxent(
    mg: MeasuredGreens,
    omega: np.ndarray | None = None,
    default_model: np.ndarray | None = None,
    use_full_covariance: bool | None = None,
    alpha_start: float = 1e12,
    alpha_end: float = 1e-2,
    preblur: bool = False,
    blur_width: float = 0.5,
    min_eigenvalue: float = 0.0,
    n_omega: int = 1001,
    omega_max: float = 15.0,
    verbose: bool = False,
) -> ContinuationResult:
    r"""Continue ``mg`` with maximum entropy.

    Parameters
    ----------
    mg
        The measured Green's function.  MaxEnt continues the **bin mean** with
        the standard error of the mean, which is the method's normal usage.
    omega
        Output grid; defaults to ``linspace(-omega_max, omega_max, n_omega)``.
    default_model
        Prior :math:`D(\omega)`, normalised internally.  ``None`` gives a flat
        model -- the choice that assumes least, and the one to keep unless a
        specific prior is being tested deliberately.
    use_full_covariance
        ``None`` decides from the data: full when ``n_bins > 2 * n_independent``,
        diagonal otherwise.

    Returns
    -------
    ContinuationResult
        With ``A`` normalised to the sum rule and the reconstruction scored
        through this package's forward map, not ana_cont's.
    """
    import spevae._vendor.ana_cont.continuation as cont

    if omega is None:
        omega = np.linspace(-omega_max, omega_max, n_omega)
    omega = np.asarray(omega, dtype=float)

    cov = mg.covariance(min_eigenvalue)
    n = cov["n_independent"]
    tau = mg.tau.double().numpy()[:n]
    G = mg.mean.double().numpy()[:n]
    sem = (cov["sigma"].double().numpy() / np.sqrt(mg.n_bins))

    if use_full_covariance is None:
        use_full_covariance = mg.n_bins > 2 * n

    if default_model is None:
        model = np.ones_like(omega)
    else:
        model = np.asarray(default_model, dtype=float).copy()
        model = np.clip(model, 1e-12, None)
    model /= np.trapezoid(model, omega)

    problem = cont.AnalyticContinuationProblem(
        im_axis=tau, re_axis=omega, im_data=G,
        kernel_mode="time_fermionic", beta=mg.beta,
    )
    kwargs = dict(method="maxent_svd", alpha_determination="chi2kink",
                  optimizer="newton", model=model, interactive=False,
                  alpha_start=alpha_start, alpha_end=alpha_end,
                  preblur=preblur, blur_width=blur_width)
    if use_full_covariance:
        # The *same* regularised correlation the package whitens with, so MaxEnt
        # and spevae are held to one noise model.  Taken directly rather than by
        # inverting the whitener: that round trip loses precision precisely when
        # R is near-singular, which is when it matters.
        R = cov["R"].double().numpy()
        kwargs["cov"] = (sem.reshape(-1, 1) * R * sem.reshape(1, -1))
    else:
        kwargs["stdev"] = sem

    # ana_cont narrates its alpha scan to stdout, which drowns a sweep. The
    # captured text is kept in `meta` rather than discarded -- the alpha/chi2
    # trace is how you check that chi2kink found a real kink and not an endpoint.
    log = io.StringIO()
    with _Timer() as t:
        if verbose:
            sol, _ = problem.solve(**kwargs)
        else:
            with contextlib.redirect_stdout(log):
                sol, _ = problem.solve(**kwargs)

    A = np.asarray(sol.A_opt, dtype=float)
    res = ContinuationResult(
        method="maxent", omega=omega, A=A, runtime=t.elapsed,
        meta={"solver": "ana_cont maxent_svd (chi2kink)",
              "full_covariance": bool(use_full_covariance),
              "internal_chi2": float(sol.chi2),
              "alpha_range": (alpha_start, alpha_end),
              "preblur": preblur,
              "default_model": "flat" if default_model is None else "supplied",
              "n_bins": mg.n_bins, "n_independent": n,
              "alpha_scan": log.getvalue()},
    )
    return res.attach_reconstruction(mg, min_eigenvalue, against="mean")
