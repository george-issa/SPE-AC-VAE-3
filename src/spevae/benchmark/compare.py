r"""Three-way comparison: spevae vs maximum entropy vs SmoQyDEAC.

What a comparison table can and cannot say
------------------------------------------
None of the three methods is ground truth.  On synthetic data there *is* a
truth, and :func:`compare` scores against it.  On real DQMC data there is not,
and the table is an **agreement test between methods**, which supports exactly
two kinds of statement:

* Where all three agree, the feature is determined by the data rather than by
  any one method's prior.  That is the strongest claim available without an
  exact solution.
* Where they disagree, the inversion is underdetermined *there*, and the
  disagreement localises it.  This is the more informative case, and it is the
  one an aggregate distance number destroys.

The three methods fail differently, which is what makes the test worth running:
MaxEnt broadens sharp structure through its entropy term, DEAC is stochastic and
returns visibly rough spectra, and this package is limited by its training
distribution and by the Lorentzian ansatz.  Two methods sharing a bias would
agree for the wrong reason.

Every row carries :math:`\chi^2/\chi^2_{\rm target}` and :math:`\min_\omega A`,
so a spectrum that does not reproduce the data, or that goes negative, cannot be
mistaken for a good answer on the strength of a small :math:`L^2`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..data.real import MeasuredGreens
from ..metrics import find_peaks, gap_metrics, smooth_to_resolution, spectral_report
from .result import ContinuationResult

__all__ = ["run_spevae", "compare", "agreement_matrix", "format_table"]


def run_spevae(
    model,
    mg: MeasuredGreens,
    omega: np.ndarray | None = None,
    n_samples: int = 200,
    min_eigenvalue: float = 0.0,
    n_omega: int = 1001,
    omega_max: float = 15.0,
    use_bin_mean: bool = True,
    refine: "RefineConfig | None" = None,
) -> ContinuationResult:
    r"""Continue ``mg`` with a trained :class:`~spevae.models.vae.SPEVAE`.

    ``refine`` runs :func:`~spevae.refine.refine_latent` on top of the amortised
    pass and reports *that* spectrum instead, labelled ``spevae-refined``.  Both
    rows belong in a table: the gap between them is the amortisation gap, and
    ``meta['prior_binding']`` says whether the refinement was genuinely MAP or
    effectively unregularised chi^2 fitting inside the decoder's manifold.

    ``use_bin_mean=True`` continues the averaged curve, which is what MaxEnt and
    DEAC do, so the comparison is like for like.  Setting it ``False`` continues
    every bin separately and averages the resulting spectra -- a different
    estimator, and the one that exercises the per-sample inference this
    architecture is for.  ``A_std`` is the latent-ensemble spread in the first
    case and the bin-to-bin spread in the second; they mean different things and
    ``meta['spread']`` records which.
    """
    from .result import _Timer

    if omega is None:
        omega = np.linspace(-omega_max, omega_max, n_omega)
    omega = np.asarray(omega, dtype=float)
    w = torch.as_tensor(omega, dtype=torch.float32)

    G = (mg.mean.unsqueeze(0) if use_bin_mean else mg.bins).float()
    sigma = ((mg.sigma_mean if use_bin_mean else mg.sigma)
             .float().unsqueeze(0).expand(G.shape[0], -1).contiguous())

    with _Timer() as t:
        out = model.continue_(G, beta=mg.beta, tau=mg.tau.float(), sigma=sigma,
                              omega=w, n_samples=n_samples)
    A = out["A"].double().numpy()
    if use_bin_mean:
        A_mean = A[0]
        A_std = out["A_std"][0].double().numpy() if "A_std" in out else None
        spread = "latent ensemble"
    else:
        A_mean = A.mean(0)
        A_std = A.std(0)
        spread = "bin to bin"

    poles = out["poles"]
    # Exact certificate on the reported sample: A >= 0 decided from the pole
    # parameters, with the negative excursion quantified when there is one.
    from ..physics.positivity import positivity_certificate
    cert = positivity_certificate(poles[0:1])[0]
    res = ContinuationResult(
        method="spevae", omega=omega, A=A_mean, A_std=A_std, runtime=t.elapsed,
        meta={"solver": "spevae SPE-VAE",
              "certified_positive": bool(cert.is_positive),
              "min_A_exact": float(cert.min_A),
              "negative_weight": float(cert.negative_weight),
              "n_samples": n_samples, "spread": spread,
              "num_poles": int(poles.shape[1]),
              # Participation ratio on the *unit-sum* weights: poles.a has been
              # scaled by the measured norm N, and 1/sum(a^2) on those is off by
              # N^2 -- the estimator must be 0-homogeneous to mean "how many".
              "effective_poles": float(
                  1.0 / ((poles.a[0] / poles.a[0].sum()) ** 2).sum()),
              "gamma_min": float(poles.gamma.min()),
              "gamma_max": float(poles.gamma.max()),
              "input": "bin mean" if use_bin_mean else "per bin"},
    )
    # The pole expansion is analytic with 1/omega^2 tails, so the grid
    # back-transform would charge it for weight that lies outside the reported
    # window -- an error of order 1e-2 at tau = 0 for broad poles.  Score it
    # against its own exact forward map instead; the reported spectrum is
    # unchanged, only the chi^2 bookkeeping is made fair.
    G_exact = out["G_recon"].double().numpy()
    if G_exact.ndim > 1:
        G_exact = G_exact.mean(0)

    if refine is not None:
        from ..refine import refine_latent
        with _Timer() as t2:
            rr = refine_latent(model, mg, refine)
        A_ref = rr.spectral(torch.as_tensor(omega, dtype=torch.float32))
        A_ref = A_ref.double().numpy()
        res = ContinuationResult(
            method="spevae-refined", omega=omega,
            A=A_ref[0] if A_ref.ndim > 1 else A_ref,
            A_std=A_ref.std(0) if A_ref.ndim > 1 and A_ref.shape[0] > 1 else None,
            runtime=t.elapsed + t2.elapsed,
            meta={**res.meta, "solver": "spevae + latent refinement",
                  "refine_steps": rr.steps_run, "refine_converged": rr.converged,
                  "chi2_before_refine": rr.chi2_init,
                  "z_shift_max_sigma": float(rr.z_shift.max()),
                  "prior_binding": rr.prior_is_binding,
                  "spread": "latent MAP point estimate"},
        )
        G_exact = rr.G_recon.double().numpy()
        if G_exact.ndim > 1:
            G_exact = G_exact.mean(0)

    return res.attach_reconstruction(mg, min_eigenvalue, against="mean",
                                     G_recon=G_exact)


def _common_grid(results: list[ContinuationResult], n: int = 2001) -> np.ndarray:
    lo = max(float(r.omega[0]) for r in results)
    hi = min(float(r.omega[-1]) for r in results)
    if hi <= lo:
        raise ValueError("the methods' frequency grids do not overlap")
    return np.linspace(lo, hi, n)


def agreement_matrix(results: list[ContinuationResult],
                     n: int = 2001) -> tuple[list[str], np.ndarray]:
    r"""Pairwise :math:`L^2` distance between methods on a common grid.

    Symmetric, zero on the diagonal.  Read it as a map of *where the methods
    stop agreeing*, not as a ranking -- without ground truth, being far from the
    others is not evidence of being wrong.
    """
    grid = _common_grid(results, n)
    curves = [r.on_grid(grid) for r in results]
    k = len(results)
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = float(np.sqrt(np.trapezoid((curves[i] - curves[j]) ** 2, grid)))
            M[i, j] = M[j, i] = d
    return [r.method for r in results], M


def compare(
    results: list[ContinuationResult],
    truth: tuple[np.ndarray, np.ndarray] | None = None,
    reference: str | None = None,
    n_grid: int = 2001,
    beta: float | None = None,
) -> dict:
    r"""Score a set of continuations against each other and, if available, a truth.

    Parameters
    ----------
    results
        One :class:`ContinuationResult` per method.
    truth
        ``(omega, A)`` exact spectrum, when there is one (synthetic tests only).
    reference
        Method name to score the others against when there is no truth.  Scoring
        *against* a method is a statement about agreement, never accuracy; the
        returned dict labels it as such.
    beta
        Inverse temperature.  When given, peak counts and matched-peak errors are
        evaluated after smoothing every spectrum to the resolution limit
        :math:`\pi/\beta`, so the table compares physics rather than how rough
        each solver's output is.  Aggregate distances stay unsmoothed.

    Returns
    -------
    dict
        ``per_method`` (one row of diagnostics each), ``vs_truth`` or
        ``vs_reference`` when applicable, and ``agreement`` (labels, matrix).
    """
    resolution = 0.0 if beta is None else math.pi / float(beta)
    out: dict = {"per_method": {}, "agreement": None, "resolution": resolution}
    for r in results:
        peaks = find_peaks(r.A, r.omega, resolution=resolution)
        gap = gap_metrics(smooth_to_resolution(r.A, r.omega, resolution), r.omega)
        out["per_method"][r.method] = {
            "norm": r.norm, "min_A": r.min_A,
            "chi2": r.chi2, "chi2_target": r.chi2_target,
            "chi2_ratio": r.chi2_ratio, "runtime_s": r.runtime,
            "n_peaks": len(peaks),
            "peaks": [(round(p.omega, 3), round(p.height, 4)) for p in peaks[:6]],
            "gapped": gap["gapped"], "gap": gap["gap"],
            **{f"meta.{k}": v for k, v in r.meta.items()},
        }

    labels, M = agreement_matrix(results, n_grid)
    out["agreement"] = {"labels": labels, "l2": M}

    if truth is not None:
        w_t, A_t = np.asarray(truth[0], float), np.asarray(truth[1], float)
        grid = _common_grid(results, n_grid)
        A_ref = np.interp(grid, w_t, A_t, left=0.0, right=0.0)
        out["vs_truth"] = {
            r.method: spectral_report(r.on_grid(grid), A_ref, grid,
                                      resolution=resolution)
            for r in results}
    elif reference is not None:
        ref = next((r for r in results if r.method == reference), None)
        if ref is None:
            raise ValueError(f"reference {reference!r} is not among the results")
        grid = _common_grid(results, n_grid)
        A_ref = ref.on_grid(grid)
        out["vs_reference"] = {
            "reference": reference,
            "caveat": "agreement between analytic-continuation methods, "
                      "not accuracy: the reference is not ground truth",
            "scores": {
                r.method: spectral_report(r.on_grid(grid), A_ref, grid,
                                          resolution=resolution)
                for r in results if r.method != reference},
        }
    return out


def format_table(cmp: dict) -> str:
    """Render :func:`compare`'s output as fixed-width text."""
    lines = []
    hdr = (f"{'method':10s} {'norm':>7s} {'min A':>10s} {'chi2':>10s} "
           f"{'target':>8s} {'ratio':>8s} {'peaks':>6s} {'gap':>7s} {'time/s':>8s}")
    lines += [hdr, "-" * len(hdr)]
    for m, d in cmp["per_method"].items():
        lines.append(
            f"{m:10s} {d['norm']:7.4f} {d['min_A']:+10.2e} {d['chi2']:10.3f} "
            f"{d['chi2_target']:8.4f} {d['chi2_ratio']:8.2f} {d['n_peaks']:6d} "
            f"{d['gap']:7.3f} {d['runtime_s']:8.1f}")

    warned = [m for m, d in cmp["per_method"].items()
              if d.get("meta.prior_binding") is False]
    if warned:
        lines += ["",
                  f"WARNING: {', '.join(warned)} -- the data term dominated the "
                  f"prior by >100x, so refinement was effectively chi^2 fitting",
                  "         inside the decoder's manifold, not MAP.  chi^2 does not "
                  "determine the spectrum; expect it to move away from",
                  "         the truth, not towards it.  The fix is a better-trained "
                  "model, not more refinement steps."]
    if cmp.get("resolution"):
        lines.append(f"(peak/gap columns evaluated at resolution "
                     f"pi/beta = {cmp['resolution']:.3f})")
    if cmp.get("agreement"):
        labels, M = cmp["agreement"]["labels"], cmp["agreement"]["l2"]
        lines += ["", "pairwise L2 between methods (no method is ground truth):",
                  "           " + " ".join(f"{l:>9s}" for l in labels)]
        for i, l in enumerate(labels):
            lines.append(f"  {l:9s}" + " ".join(f"{M[i, j]:9.4f}"
                                                for j in range(len(labels))))

    for key, title in (("vs_truth", "vs exact truth"),
                       ("vs_reference", "vs reference")):
        block = cmp.get(key)
        if not block:
            continue
        scores = block if key == "vs_truth" else block["scores"]
        if key == "vs_reference":
            lines += ["", f"{title} = {block['reference']}  ({block['caveat']})"]
        else:
            lines += ["", title]
        sub = (f"{'method':10s} {'L2 rel':>8s} {'Linf':>8s} {'W1':>8s} "
               f"{'peak dw':>9s} {'peak dh':>9s} {'gap err':>8s}")
        lines += [sub, "-" * len(sub)]
        for m, s in scores.items():
            lines.append(
                f"{m:10s} {s['l2_rel']:8.4f} {s['linf']:8.4f} {s['w1']:8.4f} "
                f"{s['peak_d_omega']:9.4f} {s['peak_rel_d_height']:9.4f} "
                f"{s['gap_error']:8.4f}")
    return "\n".join(lines)
