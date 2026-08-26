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
           "fit_peaks", "peak_starts", "fit_edged_band", "fit_mirrored_peaks",
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


def select_model(candidates: list[TemplateFit], L_eff: int,
                 repolish=None, borderline=(0.5, 2.0)) -> TemplateFit:
    """Ascending-k walk: complexity must pay the chi^2 quantile for its
    extra parameters; equal complexity is decided by fit alone.

    ``repolish``: optional ``TemplateFit -> TemplateFit`` returning the same
    hypothesis refit at floor quality (longer, finer).  When a displacement
    decision's gain lands within ``borderline`` times its threshold, both
    competitors are re-polished before ruling -- optimiser orbit noise is a
    few units of total chi^2, the same order as the margin exactly when the
    call is close (the rank scan's measured rule, applied to the menu).
    A candidate whose polish does not improve chi^2 keeps its original fit.
    """
    cands = sorted(candidates, key=lambda c: (c.k_params, c.chi2))
    polished: set[int] = set()

    def _polish(i):
        if repolish is None or i in polished:
            return
        better = repolish(cands[i])
        if better is not None and better.chi2 < cands[i].chi2:
            cands[i] = better
        polished.add(i)

    cur = 0
    for i in range(1, len(cands)):
        dk = cands[i].k_params - cands[cur].k_params
        q = chi2_quantile_95(dk)
        gain = (cands[cur].chi2 - cands[i].chi2) * L_eff
        lo, hi = (borderline[0] * q, borderline[1] * q) if dk > 0 else (0.0, 5.0)
        if lo < gain < max(hi, lo + 1e-9):
            _polish(cur)
            _polish(i)
            gain = (cands[cur].chi2 - cands[i].chi2) * L_eff
        if gain > q:
            cur = i
    return cands[cur]


def fit_peaks(
    beta: float,
    x: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    n: int = 3,
    starts: list[list[float]] | None = None,
    steps: int = 2000,
    lr: float = 0.02,
    gamma_floor: float | None = None,
    asym: bool = False,
    t_starts: list[list[float]] | None = None,
) -> TemplateFit:
    r"""n independent symmetric peaks: the humblest structured hypothesis.

    ``asym=True`` frees one asymmetry per peak with the exact structural
    constraints of the decoder itself -- :math:`b = a(t - \langle t
    \rangle_a)`, :math:`t = \tanh(u)/2` -- so :math:`\sum b = 0` holds
    identically and :math:`|b| \le a`.  Costs :math:`n - 1` extra effective
    parameters (k = 4n - 2): a symmetric-only basis pays for every real
    asymmetry by stacking extra poles, and this entry is what lets the menu
    say so.

    Positions, widths and (normalised) weights are free but the asymmetries
    are pinned to zero -- k = 3n - 1 against the free fit's 4n.  This is the
    entry whose absence the irregular-triplet test exposed: same fit as free
    poles whenever the truth is symmetric, strictly cheaper.
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
        starts = [list(torch.linspace(-2.0, 2.0, n))]
    # rival basins are the rule, not the exception: jittered restarts around
    # every provided start, chi^2 keeps the best (seeded, deterministic)
    rng = torch.Generator().manual_seed(0)
    expanded = []
    for e_init in starts:
        e0 = torch.tensor([float(v) for v in e_init], dtype=torch.float64)
        expanded.append(e0)
        for _ in range(3):
            expanded.append(e0 + 0.3 * (2.0 * torch.rand(n, generator=rng,
                            dtype=torch.float64) - 1.0))
    best = None
    for e_init in expanded:
        best_local, best_state = float("inf"), None
        # Asymmetry basins are separated: a symmetric fit absorbs real
        # asymmetry into shifted positions (measured), so neither gradient
        # descent nor a linear b-solve escapes once geometry has settled.
        # Multi-start over t sign patterns, jointly optimised from step 0.
        if asym:
            gen_t = torch.Generator().manual_seed(1)
            t_inits = [torch.zeros(n, dtype=torch.float64)]
            for _ in range(3):
                sgn = torch.where(torch.rand(n, generator=gen_t) < 0.5,
                                  -1.0, 1.0).to(torch.float64)
                t_inits.append(0.25 * sgn)
            for tv in (t_starts or []):
                t_inits.append(torch.tensor([float(v) for v in tv],
                                            dtype=torch.float64))
        else:
            t_inits = [torch.zeros(n, dtype=torch.float64)]
        e_init0 = e_init.clone()
        for t0 in t_inits:
            u_e = e_init0.clone().requires_grad_(True)
            u_g = torch.log(torch.expm1(torch.full((n,), 0.25,
                            dtype=torch.float64))).clone().requires_grad_(True)
            u_w = torch.zeros(n, dtype=torch.float64, requires_grad=True)
            u_t = torch.atanh((2.0 * t0).clamp(-0.98, 0.98)).clone(
                ).requires_grad_(True)
            params = [u_e, u_g, u_w] + ([u_t] if asym else [])
            opt = torch.optim.Adam(params, lr=lr)
            for it in range(steps):
                g = floor + torch.nn.functional.softplus(u_g)
                w = torch.softmax(u_w, dim=-1)
                if asym:
                    t = 0.5 * torch.tanh(u_t)
                    bb = w * (t - (w * t).sum())
                else:
                    bb = torch.zeros(n, dtype=torch.float64)
                ps = PoleSet(eps=(u_e * beta).unsqueeze(0),
                             gamma=(g * beta).unsqueeze(0),
                             a=w.unsqueeze(0), b=bb.unsqueeze(0))
                loss = chi2_of(ps)
                v = float(loss.detach())
                if v < best_local:
                    best_local = v
                    best_state = (u_e.detach().clone(), g.detach().clone(),
                                  w.detach().clone(), bb.detach().clone())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        if best is None or best_local < best[0]:
            best = (best_local, best_state)
    c, (e, g, w, bb) = best
    poles = PoleSet(eps=e.unsqueeze(0), gamma=g.unsqueeze(0),
                    a=w.unsqueeze(0), b=bb.unsqueeze(0))
    pr = {f"eps{i}": float(e[i]) for i in range(n)}
    if asym:
        pr.update({f"b{i}": float(bb[i]) for i in range(n)})
    return TemplateFit(name="peaks_asym" if asym else "peaks", m=n,
                       k_params=(4 * n - 2) if asym else (3 * n - 1),
                       chi2=float(c), poles=poles, params=pr)


def peak_starts(poles: PoleSet, n: int) -> list[list[float]]:
    """Position inits: the n heaviest amortized poles, sorted."""
    a = poles.a.reshape(-1)
    eps = poles.eps.reshape(-1)
    idx = torch.topk(a, k=min(n, a.numel())).indices
    base = sorted(float(eps[i]) for i in idx)
    while len(base) < n:
        base.append(base[-1] + 1.0)
    return [base]


def fit_edged_band(
    beta: float,
    x: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    m: int = 5,
    starts: list[tuple[float, float]] | None = None,
    steps: int = 2500,
    lr: float = 0.02,
    gamma_floor: float | None = None,
) -> TemplateFit:
    r"""A band with sharp edges: m poles spanning [c-W, c+W], narrow at the
    edges and broad in the body, with one weight-tilt parameter.

    k = 5 whatever m is (centre, half-width, body width, edge width, tilt);
    m is representation multiplicity -- a resolution choice, not physics.
    """
    quad = SpectralQuadrature.for_beta(float(beta), dtype=torch.float64)
    fwd = PoleGreens(quad)
    xd = x.reshape(-1).double()
    K = fwd.kernel_matrix(xd)
    G = G_obs.reshape(1, -1).double()
    s = sigma.reshape(1, -1).double()
    Wm = None if inv_sqrt_R is None else inv_sqrt_R.double()
    floor = (0.1 * math.pi / beta) if gamma_floor is None else float(gamma_floor)

    def chi2_of(ps):
        Gp = fwd(ps, xd, kernel_matrix=K)
        d = (Gp - G) / s
        if Wm is not None:
            d = d @ Wm.transpose(-1, -2)
        return (d * d).sum(-1).mean() / d.shape[-1]

    ks = torch.linspace(-1.0, 1.0, m, dtype=torch.float64)
    if starts is None:
        starts = [(0.0, 1.5)]
    best = None
    for c0, w0 in starts:
        u_c = torch.tensor(float(c0), dtype=torch.float64, requires_grad=True)
        u_W = torch.log(torch.expm1(torch.tensor(max(float(w0), 0.2),
                        dtype=torch.float64))).clone().requires_grad_(True)
        u_gb = torch.log(torch.expm1(torch.tensor(0.4, dtype=torch.float64)))
        u_gb = u_gb.clone().requires_grad_(True)
        u_ge = torch.log(torch.expm1(torch.tensor(0.1, dtype=torch.float64)))
        u_ge = u_ge.clone().requires_grad_(True)
        u_t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([u_c, u_W, u_gb, u_ge, u_t], lr=lr)
        best_local, best_state = float("inf"), None
        for _ in range(steps):
            Wb = 0.1 + torch.nn.functional.softplus(u_W)
            gb = floor + torch.nn.functional.softplus(u_gb)
            ge = floor + torch.nn.functional.softplus(u_ge)
            gam = torch.cat([ge[None], gb.expand(m - 2), ge[None]])
            eps = u_c + ks * Wb
            w = torch.softmax(u_t * ks, dim=-1)
            ps = PoleSet(eps=(eps * beta).unsqueeze(0),
                         gamma=(gam * beta).unsqueeze(0),
                         a=w.unsqueeze(0),
                         b=torch.zeros(1, m, dtype=torch.float64))
            loss = chi2_of(ps)
            v = float(loss.detach())
            if v < best_local:
                best_local = v
                best_state = (float(u_c), float(Wb), float(gb), float(ge),
                              float(u_t))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if best is None or best_local < best[0]:
            best = (best_local, best_state)
    c, (cc, Wb, gb, ge, t) = best
    with torch.no_grad():
        gam = torch.cat([torch.tensor([ge]), torch.full((m - 2,), gb),
                         torch.tensor([ge])]).double()
        eps = cc + ks * Wb
        w = torch.softmax(torch.tensor(t, dtype=torch.float64) * ks, dim=-1)
    poles = PoleSet(eps=eps.unsqueeze(0), gamma=gam.unsqueeze(0),
                    a=w.unsqueeze(0), b=torch.zeros(1, m, dtype=torch.float64))
    return TemplateFit(name="edged_band", m=m, k_params=5, chi2=float(c),
                       poles=poles,
                       params=dict(centre=cc, half_width=Wb, gamma_body=gb,
                                   gamma_edge=ge, tilt=t))


def fit_mirrored_peaks(
    beta: float,
    x: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    n: int = 3,
    starts: list[list[float]] | None = None,
    steps: int = 2500,
    lr: float = 0.02,
    gamma_floor: float | None = None,
) -> TemplateFit:
    r"""n free mirror pairs: particle-hole symmetry as the only structure.

    The entry the Holstein data asked for (checkpoint A, 2026-08-25): its
    free-pole answer matched the mirrored ladder's count (3 pairs) but
    rejected the ladder's shared-width/geometric-decay constraints at 6x in
    chi^2.  Here each pair has its own position, width and weight --
    A(w) = A(-w) exact, k = 3n - 1, nothing else assumed.
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
        starts = [list(torch.linspace(0.5, 2.5, n))]
    rng = torch.Generator().manual_seed(0)
    expanded = []
    for e_init in starts:
        e0 = torch.tensor([abs(float(v)) for v in e_init], dtype=torch.float64)
        expanded.append(e0)
        for _ in range(3):
            expanded.append((e0 + 0.3 * (2.0 * torch.rand(n, generator=rng,
                            dtype=torch.float64) - 1.0)).abs().clamp_min(0.02))
    best = None
    for e_init in expanded:
        u_e = e_init.clone().requires_grad_(True)
        u_g = torch.log(torch.expm1(torch.full((n,), 0.25,
                        dtype=torch.float64))).clone().requires_grad_(True)
        u_w = torch.zeros(n, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([u_e, u_g, u_w], lr=lr)
        best_local, best_state = float("inf"), None
        for _ in range(steps):
            e = u_e.abs()                        # positions live on w > 0
            g = floor + torch.nn.functional.softplus(u_g)
            w = torch.softmax(u_w, dim=-1)
            ps = PoleSet(
                eps=(torch.cat([e, -e]) * beta).unsqueeze(0),
                gamma=(torch.cat([g, g]) * beta).unsqueeze(0),
                a=(0.5 * torch.cat([w, w])).unsqueeze(0),
                b=torch.zeros(1, 2 * n, dtype=torch.float64))
            loss = chi2_of(ps)
            v = float(loss.detach())
            if v < best_local:
                best_local = v
                best_state = (e.detach().clone(), g.detach().clone(),
                              w.detach().clone())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if best is None or best_local < best[0]:
            best = (best_local, best_state)
    c, (e, g, w) = best
    poles = PoleSet(eps=torch.cat([e, -e]).unsqueeze(0),
                    gamma=torch.cat([g, g]).unsqueeze(0),
                    a=(0.5 * torch.cat([w, w])).unsqueeze(0),
                    b=torch.zeros(1, 2 * n, dtype=torch.float64))
    pr = {f"eps{i}": float(e[i]) for i in range(n)}
    pr.update({f"w{i}": float(w[i]) for i in range(n)})
    return TemplateFit(name="mirrored_peaks", m=n, k_params=3 * n - 1,
                       chi2=float(c), poles=poles, params=pr)
