r"""Continuation methods behind one interface, and the one scorer they share.

The suite judges *methods*, not checkpoints.  A method takes a dataset and
returns either a pole set or a spectrum on a frequency grid; everything
else -- :math:`W_1` against the truth, :math:`\chi^2` against the data,
negative weight, pole count -- is computed here, by the same code for
every method.  A method's own internal :math:`\chi^2` is kept alongside for
the acceptance check, never used for ranking.

One forward map for everything: spectra (pole or grid) are evaluated on
the quadrature nodes of :meth:`SpectralQuadrature.for_beta` and pushed
through the same kernel matrix that produced the truths' clean data.
"""

from __future__ import annotations

import contextlib
import io
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..losses import spectral_w1
from ..physics import PoleGreens, SpectralQuadrature, certify
from ..physics.poles import PoleSet
from .cases import Instance

__all__ = ["Output", "METHODS", "make_method", "score_synthetic", "score_real",
           "plot_grid", "spectrum_on_grid"]

#: Physical-frequency grid used for every stored spectrum (figures only).
PLOT_OMEGA = np.linspace(-8.0, 8.0, 641)


@dataclass
class Output:
    """What a method returns for one dataset.

    Exactly one of ``poles`` (dimensionless, at the dataset's beta) or
    ``(omega, A)`` (physical units, any grid) is set.
    """

    method: str
    poles: PoleSet | None = None
    omega: np.ndarray | None = None
    A: np.ndarray | None = None
    internal_chi2: float = float("nan")
    runtime: float = float("nan")
    extras: dict = field(default_factory=dict)
    #: further answers from the same run, scored as their own rows
    #: (the scan pipeline's full rendering next to its minimal form)
    alternates: dict = field(default_factory=dict)


# ------------------------------------------------------------- scoring --- #
def _A_nodes(out: Output, quad: SpectralQuadrature, beta: float) -> torch.Tensor:
    """The method's spectrum on the quadrature nodes, dimensionless, (1, M)."""
    if out.poles is not None:
        if out.poles.eps.dtype == torch.float64:
            return out.poles.spectral(quad.nodes.to(torch.float64)).float()
        return out.poles.spectral(quad.nodes)
    w = (quad.nodes / beta).double().numpy()
    A_phys = np.interp(w, out.omega, out.A, left=0.0, right=0.0)
    return torch.tensor(A_phys / beta, dtype=torch.float32).unsqueeze(0)


def _G_recon(out: Output, A_nodes: torch.Tensor, quad, x) -> torch.Tensor:
    if out.poles is not None:
        return PoleGreens(quad)(out.poles, x)       # runs in the poles' own dtype
    K = PoleGreens(quad).kernel_matrix(x)
    return (A_nodes * quad.weights) @ K


def _neg_weight(out: Output, A_nodes: torch.Tensor, quad, beta) -> float:
    if out.poles is not None:
        p = out.poles.to_physical(beta)
        rep = certify(p.eps[0].tolist(), p.gamma[0].tolist(), p.a[0].tolist(),
                      p.b[0].tolist(), atol=1e-8)
        return 0.0 if rep.is_positive else float(rep.negative_weight)
    return float((A_nodes.clamp(max=0.0).abs() * quad.weights).sum())


def _n_poles(out: Output, thresh: float = 1e-3) -> int | None:
    if out.poles is None:
        return None
    a = out.poles.a[0]
    return int((a / a.sum() > thresh).sum())


def spectrum_on_grid(out: Output, beta: float, omega: np.ndarray = PLOT_OMEGA) -> np.ndarray:
    """Physical A(omega) on a uniform grid, for figures."""
    if out.poles is not None:
        nu = torch.tensor(omega * beta, dtype=torch.float32)
        return (out.poles.spectral(nu)[0] * beta).double().numpy()
    return np.interp(omega, out.omega, out.A, left=0.0, right=0.0)


def plot_grid() -> np.ndarray:
    return PLOT_OMEGA


def score_synthetic(out: Output, inst: Instance) -> tuple[dict, torch.Tensor]:
    """Metrics for one synthetic instance; also returns A on the nodes."""
    tr = inst.truth
    A = _A_nodes(out, tr.quad, tr.beta)
    G = _G_recon(out, A, tr.quad, tr.x)
    d = ((G - inst.G) / inst.sigma) @ inst.isr.T
    rec = {
        "w1": float(spectral_w1(A, tr.A, tr.quad, beta=tr.beta)),
        "chi2": float((d * d).sum(-1).mean() / inst.G.shape[-1]),
        "chi2_internal": out.internal_chi2,
        "neg_weight": _neg_weight(out, A, tr.quad, tr.beta),
        "n_poles": _n_poles(out),
        "runtime": out.runtime,
    }
    rec.update(out.extras)
    return rec, A


def score_real(out: Output, mg, quad: SpectralQuadrature) -> dict:
    """chi^2 against a measured cell under the canonical real-data noise model.

    Same convention as :func:`refine_latent`: structured covariance with
    eigenvalue floor 1e-10, statistical slices only, standard error of the
    mean, divided by the number of independent slices.
    """
    cov = mg.covariance(1e-10, method="structured")
    sl, n = cov["slices"], cov["n_independent"]
    A = _A_nodes(out, quad, mg.beta)
    x = mg.x.float()
    G = _G_recon(out, A, quad, x).double()
    sigma = cov["sigma"].double() / math.sqrt(mg.n_bins)
    d = ((G[:, sl] - mg.mean.double()[sl]) / sigma) @ cov["inv_sqrt_R"].double().T
    chi2 = float((d * d).sum(-1).mean() / n)
    rec = {"chi2": chi2, "chi2_target": float(cov["chi2_target"]),
           "ratio": chi2 / float(cov["chi2_target"]),
           "chi2_internal": out.internal_chi2,
           "neg_weight": _neg_weight(out, A, quad, mg.beta),
           "n_poles": _n_poles(out), "runtime": out.runtime}
    rec.update(out.extras)
    return rec


# ------------------------------------------------------------- methods --- #
class SpevaeAmortized:
    """One deterministic forward pass: the network's own answer."""

    needs_model = True

    def __init__(self, model):
        self.model, self.name = model, "amortized"

    def synthetic(self, inst: Instance) -> Output:
        tr = inst.truth
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(inst.G, tr.x, tr.beta, sigma=inst.sigma, deterministic=True)
        return Output(self.name, poles=out.poles.detach(),
                      runtime=time.perf_counter() - t0)

    def real(self, mg) -> Output:
        from ..refine import RefineConfig, refine_latent
        t0 = time.perf_counter()
        rr = refine_latent(self.model, mg, RefineConfig(
            cov_method="structured", min_eigenvalue=1e-10, steps=0))
        return Output(self.name, poles=rr.poles.to_dimensionless(mg.beta),
                      internal_chi2=float(rr.chi2), runtime=time.perf_counter() - t0)


class SpevaeZDescent:
    """MAP descent in the latent (weights frozen, prior anchor 1).

    ``zd<N>`` is Adam for N steps (the banked protocols); ``zdlm`` minimises
    the same objective by Levenberg--Marquardt to a stationary point (or, on
    a real cell, to the calibrated noise floor) and records how it ended.
    """

    needs_model = True

    def __init__(self, model, steps: int, solver: str = "adam"):
        self.model, self.steps, self.solver = model, int(steps), solver
        self.name = "zdlm" if solver == "lm" else f"zd{int(steps)}"

    def _extras(self, r) -> dict:
        ex = {"chi2_amortized": float(r.chi2_init), "steps_run": int(r.steps_run)}
        if self.solver == "lm":
            ex.update(converged=bool(r.converged), stationary=bool(r.stationary))
        return ex

    def synthetic(self, inst: Instance) -> Output:
        from ..refine import z_descend
        tr = inst.truth
        t0 = time.perf_counter()
        zr = z_descend(self.model, inst.G[0], tr.x, tr.beta, sigma=inst.sigma[0],
                       inv_sqrt_R=inst.isr, steps=self.steps, solver=self.solver)
        return Output(self.name, poles=zr.poles.to_dimensionless(tr.beta),
                      internal_chi2=float(zr.chi2), runtime=time.perf_counter() - t0,
                      extras=self._extras(zr))

    def real(self, mg) -> Output:
        from ..refine import RefineConfig, refine_latent
        t0 = time.perf_counter()
        rr = refine_latent(self.model, mg, RefineConfig(
            cov_method="structured", min_eigenvalue=1e-10,
            steps=self.steps if self.solver == "adam" else 1, solver=self.solver))
        return Output(self.name, poles=rr.poles.to_dimensionless(mg.beta),
                      internal_chi2=float(rr.chi2), runtime=time.perf_counter() - t0,
                      extras=self._extras(rr))


class SpevaeScan:
    """The certified pipeline: z-descent (MAP on real data) -> anchored rank scan.

    Returns the LR-selected minimal form as the main answer and the
    prior-started full-capacity rendering as the alternate ``scan_full``
    -- the dual output of TRAINING.md.  Settings are the banked protocol:
    z-descent 1500 and anchor weight 0.5 on synthetic data; MAP 3000 and
    anchor weight 1.0 on real cells; pre-refit 3000, initial reduce "drop".
    """

    needs_model = True

    def __init__(self, model, z_steps=1500, w_synth=0.5, w_real=1.0):
        self.model, self.name = model, "scan_minimal"
        self.z_steps, self.w_synth, self.w_real = z_steps, w_synth, w_real

    def _scan(self, poles_phys, beta, x, G, sigma, isr, w, quad):
        from ..physics.consolidate import rank_scan
        A_anchor = poles_phys.to_dimensionless(beta).spectral(quad.nodes.to(torch.float64))
        sel, tr = rank_scan(poles_phys, beta, x[None], G.reshape(1, -1), sigma.reshape(1, -1),
                            inv_sqrt_R=isr, anchor_A=A_anchor, anchor_weight=w,
                            pre_refit_steps=3000, initial_reduce="drop")
        extras = {"selected_n": int(tr.selected_n), "n_LR": tr.n_LR,
                  "scan_ns": list(tr.ns), "scan_chi2s": [float(c) for c in tr.chi2s]}
        full = Output("scan_full", poles=tr.full_fit.to_dimensionless(beta),
                      internal_chi2=float(tr.full_chi2))
        return Output(self.name, poles=sel.to_dimensionless(beta),
                      internal_chi2=float(tr.chi2_selected), extras=extras,
                      alternates={"scan_full": full})

    def synthetic(self, inst: Instance) -> Output:
        from ..refine import z_descend
        tr = inst.truth
        t0 = time.perf_counter()
        zr = z_descend(self.model, inst.G, tr.x, tr.beta, sigma=inst.sigma,
                       inv_sqrt_R=inst.isr, steps=self.z_steps)
        out = self._scan(zr.poles, tr.beta, tr.x, inst.G, inst.sigma, inst.isr,
                         self.w_synth, tr.quad)
        out.runtime = time.perf_counter() - t0
        return out

    def real(self, mg) -> Output:
        from ..refine import RefineConfig, refine_latent
        t0 = time.perf_counter()
        quad = SpectralQuadrature.for_beta(mg.beta)
        rr = refine_latent(self.model, mg, RefineConfig(
            cov_method="structured", min_eigenvalue=1e-10, steps=3000))
        cov = mg.covariance(1e-10, method="structured")
        sl = mg.statistical_slices
        out = self._scan(rr.poles, mg.beta, mg.x[sl], mg.mean[sl], mg.sigma_mean[sl],
                         cov["inv_sqrt_R"], self.w_real, quad)
        out.runtime = time.perf_counter() - t0
        return out


def _ana_cont_maxent(tau, G, beta, stdev=None, cov=None, omega=None):
    """Bare ana_cont MaxEnt call with the package's standard settings."""
    import spevae._vendor.ana_cont.continuation as cont
    omega = np.linspace(-15.0, 15.0, 1001) if omega is None else omega
    model = np.ones_like(omega)
    model /= np.trapezoid(model, omega)
    problem = cont.AnalyticContinuationProblem(
        im_axis=tau, re_axis=omega, im_data=G, kernel_mode="time_fermionic", beta=beta)
    kw = dict(method="maxent_svd", alpha_determination="chi2kink", optimizer="newton",
              model=model, interactive=False, alpha_start=1e12, alpha_end=1e-2,
              preblur=False, blur_width=0.5)
    if cov is not None:
        kw["cov"] = cov
    else:
        kw["stdev"] = stdev
    with contextlib.redirect_stdout(io.StringIO()):
        sol, _ = problem.solve(**kw)
    return omega, np.asarray(sol.A_opt, float), float(sol.chi2)


class MaxEnt:
    """Maximum entropy (vendored ana_cont, flat default model, chi2-kink alpha).

    Synthetic cases get the *exact* noise model the data were drawn with
    (full covariance sigma_i R_ij sigma_j), the same information spevae's
    whitener uses.  Real cells go through :func:`run_maxent` unchanged.
    """

    needs_model = False
    name = "maxent"

    def synthetic(self, inst: Instance) -> Output:
        tr = inst.truth
        tau = (tr.x * tr.beta).double().numpy()
        G = inst.G[0].double().numpy()
        s = inst.sigma[0].double().numpy()
        cov = s[:, None] * inst.R.double().numpy() * s[None, :]
        t0 = time.perf_counter()
        omega, A, chi2 = _ana_cont_maxent(tau, G, tr.beta, cov=cov)
        return Output(self.name, omega=omega, A=A, internal_chi2=chi2,
                      runtime=time.perf_counter() - t0)

    def real(self, mg) -> Output:
        from ..benchmark import run_maxent
        res = run_maxent(mg, min_eigenvalue=1e-10)
        return Output(self.name, omega=res.omega, A=res.A,
                      internal_chi2=float(res.meta["internal_chi2"]), runtime=res.runtime,
                      extras={"full_covariance": res.meta["full_covariance"]})


class DEAC:
    """SmoQyDEAC through its Julia driver (needs bins: real cells only in v1)."""

    needs_model = False
    name = "deac"

    def __init__(self, **kw):
        self.kw = kw

    def synthetic(self, inst: Instance) -> Output:
        raise NotImplementedError("DEAC needs raw bins; synthetic bins are a v1.1 item")

    def real(self, mg) -> Output:
        from ..benchmark import run_deac
        res = run_deac(mg, min_eigenvalue=1e-10, **self.kw)
        return Output(self.name, omega=res.omega, A=res.A, runtime=res.runtime,
                      extras={k: res.meta[k] for k in ("num_bins", "runs_per_bin",
                                                       "generations") if k in res.meta})


class ShippedReference:
    """The SmoQyDEAC continuation shipped with the real cell (a comparator)."""

    needs_model = False
    name = "deac_shipped"

    def synthetic(self, inst):
        raise NotImplementedError

    def real(self, mg) -> Output:
        if mg.reference is None:
            raise ValueError("this cell ships no reference continuation")
        w, A = (np.asarray(v, float) for v in mg.reference)
        return Output(self.name, omega=w, A=A)


METHODS = {"amortized", "zd800", "zd1500", "zd3000", "zdlm", "scan_minimal", "maxent",
           "deac", "deac_shipped"}


def make_method(spec: str, model=None):
    """Build a method from its spec string, e.g. ``"zd800"`` or ``"maxent"``."""
    if spec == "amortized":
        return SpevaeAmortized(model)
    if spec == "scan_minimal":
        return SpevaeScan(model)
    if spec == "zdlm":
        return SpevaeZDescent(model, 0, solver="lm")
    if spec.startswith("zd"):
        return SpevaeZDescent(model, int(spec[2:]))
    if spec == "maxent":
        return MaxEnt()
    if spec == "deac":
        return DEAC()
    if spec == "deac_shipped":
        return ShippedReference()
    raise ValueError(f"unknown method {spec!r}")
