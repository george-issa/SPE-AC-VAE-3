r"""Structured spectral templates: physics hypotheses as competing models.

The rank scan's only prior is "fewest poles".  The ambiguities that survive
it -- comb-spacing aliasing, within-count position slop, faint rungs below
the per-pole evidence threshold -- are exactly the ones physical *structure*
crushes: a ladder is not m independent poles (4m parameters) but one object
with a base, a spacing, a shared width and a decay (4 parameters for any m).
Fitting such templates alongside free pole sets and selecting by the same
likelihood-ratio doctrine turns "the prior beats the log wall" into a
declared, per-dataset model comparison: a template can only *win* when the
data is fully consistent with it AND it is more parsimonious; free poles
remain on the menu as the assumption-free fallback, so the floor is the
plain rank scan.

Selection walks candidates in ascending parameter count: a more complex
model displaces a simpler one only if it improves the total whitened
:math:`\chi^2` by more than the 95% quantile of :math:`\chi^2_{\Delta k}`
(Wilson--Hilferty).  Equal parameter counts (e.g. ladder(3) vs ladder(4) --
both 4 parameters) are decided by fit alone, which is how a template can
carry a faint rung that free poles could not individually justify.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .forward import PoleGreens, SpectralQuadrature
from .poles import PoleSet

__all__ = ["TemplateFit", "fit_ladder", "ladder_starts", "select_model",
           "chi2_quantile_95"]


def chi2_quantile_95(d: int) -> float:
    """Wilson-Hilferty approximation of the chi^2_d 95% point."""
    if d <= 0:
        return 0.0
    z = 1.6449                                  # Phi^-1(0.95)
    return d * (1.0 - 2.0 / (9.0 * d) + z * math.sqrt(2.0 / (9.0 * d))) ** 3


@dataclass
class TemplateFit:
    """One fitted hypothesis, comparable across the whole menu."""

    name: str                 #: "ladder", "mirrored_ladder", "free"
    m: int                    #: structural multiplicity (rungs; poles for free)
    k_params: int             #: free parameters the fit spent
    chi2: float               #: whitened per-dof chi^2 after the fit
    poles: PoleSet            #: physical units
    params: dict              #: named template parameters (physical units)

    def __str__(self) -> str:
        ps = ", ".join(f"{k}={v:.4g}" for k, v in self.params.items())
        return (f"{self.name}(m={self.m}): chi2 {self.chi2:.4g}, "
                f"k={self.k_params} [{ps}]")


def _ladder_poles(eps0, Om, gam, r, m, mirrored, beta):
    """Assemble the dimensionless PoleSet of a (possibly mirrored) ladder."""
    ks = torch.arange(m, dtype=eps0.dtype)
    eps = eps0 + ks * Om
    w = torch.softmax(ks * torch.log(r.clamp_min(1e-6)), dim=-1)  # r^k normed
    g = gam.expand(m)
    if mirrored:
        eps = torch.cat([eps, -eps])
        g = torch.cat([g, g])
        w = 0.5 * torch.cat([w, w])
    return PoleSet(eps=(eps * beta).unsqueeze(0), gamma=(g * beta).unsqueeze(0),
                   a=w.unsqueeze(0), b=torch.zeros_like(w).unsqueeze(0))


def fit_ladder(
    beta: float,
    x: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    m: int = 3,
    mirrored: bool = False,
    starts: list[tuple[float, float]] | None = None,
    steps: int = 1500,
    lr: float = 0.02,
    gamma_floor: float | None = None,
) -> TemplateFit:
    r"""Fit an m-rung (optionally mirrored) ladder template to the data.

    Four parameters whatever m is: base position :math:`\varepsilon_0`,
    spacing :math:`\Omega`, shared width :math:`\gamma`, decay ratio
    :math:`r`; weights are :math:`r^k` normalised, so the sum rule is exact
    and the spectrum is non-negative by construction.  ``starts`` is a list
    of :math:`(\varepsilon_0, \Omega)` initialisations (see
    :func:`ladder_starts`); the best final :math:`\chi^2` wins.
    """
    quad = SpectralQuadrature.for_beta(float(beta), dtype=torch.float64)
    fwd = PoleGreens(quad)
    xd = x.reshape(-1).double()
    K = fwd.kernel_matrix(xd)
    G = G_obs.reshape(1, -1).double()
    s = sigma.reshape(1, -1).double()
    W = None if inv_sqrt_R is None else inv_sqrt_R.double()
    floor = (0.1 * math.pi / beta) if gamma_floor is None else float(gamma_floor)

    def chi2_of(ps):
        Gp = fwd(ps, xd, kernel_matrix=K)
        d = (Gp - G) / s
        if W is not None:
            d = d @ W.transpose(-1, -2)
        return (d * d).sum(-1).mean() / d.shape[-1]

    if starts is None:
        starts = [(0.5, 1.0)]
    best = None
    for e0_init, om_init in starts:
        u_e = torch.tensor(float(e0_init), dtype=torch.float64,
                           requires_grad=True)
        u_o = torch.log(torch.expm1(torch.tensor(
            max(float(om_init) - 0.05, 1e-3), dtype=torch.float64)))
        u_o = u_o.clone().requires_grad_(True)
        u_g = torch.log(torch.expm1(torch.tensor(0.25, dtype=torch.float64)))
        u_g = u_g.clone().requires_grad_(True)
        u_r = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([u_e, u_o, u_g, u_r], lr=lr)

        def assemble():
            Om = 0.05 + torch.nn.functional.softplus(u_o)
            gam = floor + torch.nn.functional.softplus(u_g)
            r = torch.sigmoid(u_r)
            return _ladder_poles(u_e, Om, gam, r, m, mirrored, beta), Om, gam, r

        best_local, best_state = float("inf"), None
        for _ in range(steps):
            ps, Om, gam, r = assemble()
            loss = chi2_of(ps)
            v = float(loss.detach())
            if v < best_local:
                best_local = v
                best_state = (float(u_e), float(Om), float(gam), float(r))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if best is None or best_local < best[0]:
            best = (best_local, best_state)

    c, (e0, Om, gam, r) = best
    with torch.no_grad():
        ps = _ladder_poles(torch.tensor(e0, dtype=torch.float64),
                           torch.tensor(Om, dtype=torch.float64),
                           torch.tensor(gam, dtype=torch.float64),
                           torch.tensor(r, dtype=torch.float64),
                           m, mirrored, beta)
    poles_phys = PoleSet(eps=ps.eps / beta, gamma=ps.gamma / beta,
                         a=ps.a, b=ps.b)
    return TemplateFit(name="mirrored_ladder" if mirrored else "ladder",
                       m=m, k_params=4, chi2=float(c), poles=poles_phys,
                       params=dict(eps0=e0, Omega=Om, gamma=gam, decay=r))


def ladder_starts(poles: PoleSet, mirrored: bool = False,
                  weight_floor: float = 0.03) -> list[tuple[float, float]]:
    """(eps0, Omega) initialisations proposed by an amortized pole set.

    Base candidates: the heaviest pole and the lowest significant pole.
    Spacing candidates: the median gap of the significant poles and its
    x0.5 / x2 aliases -- the multi-scale starts are what let the fit jump
    between rival comb interpretations.
    """
    eps = poles.eps.reshape(-1)
    a = poles.a.reshape(-1)
    if mirrored:
        keep = eps > 0
        eps, a = eps[keep], a[keep]
    sig = a >= weight_floor * float(a.sum())
    if sig.sum() < 1:
        sig = a >= 0.0
    es = torch.sort(eps[sig]).values
    heavy = float(eps[int(a.argmax())]) if not mirrored or eps.numel() else 0.0
    bases = {round(float(es[0]), 3), round(heavy, 3)}
    if es.numel() >= 2:
        gaps = (es[1:] - es[:-1])
        om0 = float(gaps.median())
    else:
        om0 = 1.0
    oms = {round(max(om0, 0.1), 3), round(max(0.5 * om0, 0.1), 3),
           round(max(2.0 * om0, 0.1), 3)}
    # the gap between the two heaviest poles is often the cleanest spacing
    # estimate when faint stacks contaminate the median
    if eps.numel() >= 2:
        top2 = torch.topk(a, k=2).indices
        om_h = abs(float(eps[top2[0]] - eps[top2[1]]))
        if om_h > 0.1:
            oms.add(round(om_h, 3))
    return [(b, o) for b in bases for o in oms]


def select_model(candidates: list[TemplateFit], L_eff: int) -> TemplateFit:
    """Ascending-k walk: complexity must pay the chi^2 quantile for its
    extra parameters; equal complexity is decided by fit alone."""
    order = sorted(candidates, key=lambda c: (c.k_params, c.chi2))
    current = order[0]
    for cand in order[1:]:
        dk = cand.k_params - current.k_params
        gain = (current.chi2 - cand.chi2) * L_eff
        if gain > chi2_quantile_95(dk):
            current = cand
    return current
