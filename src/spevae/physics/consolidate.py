r"""Pole consolidation: a minimal-pole-style output from a many-pole fit.

The trained decoder tiles spectral weight over more poles than the physics
needs, because merging poles is a *flat direction of* :math:`\chi^2`: a
ten-pole tiling of a band and a two-pole representation of the same envelope
produce :math:`G(\tau)` identical to within noise.  Nothing in training or
refinement pays for consolidation -- so it is done here, explicitly, as an
exactly-controlled post-process:

1. **Cluster** poles whose positions lie within the data's own resolution
   scale (the first Matsubara frequency :math:`\pi/\beta` by default) of the
   cluster's running weighted centroid.  Structure finer than that scale is
   not resolvable from :math:`\tau` data at temperature :math:`1/\beta` --
   merging inside it discards nothing the data ever knew.
2. **Merge** each cluster conservatively: :math:`a = \sum a_i` and
   :math:`b = \sum b_i` (both sum rules preserved *exactly*, since clusters
   partition the set), position at the weighted centroid, width from the
   members' mean width plus their weighted position spread.
3. **Polish** (optional, default on): re-fit the merged parameters to the
   *original* pole set's Green's function on the Matsubara points
   :math:`z_n = i(2n{+}1)\rho` (:math:`\rho` = the resolution scale) --
   closed-form evaluations, no :math:`\omega` grid anywhere -- then restore
   the sum rules exactly.

The claim "this changed nothing the data can see" is *measured*, not
assumed: the report carries the relative Matsubara mismatch, and callers
(CLI, tests) additionally push both sets through the exact forward map and
compare in units of the data noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .poles import PoleSet

__all__ = ["ConsolidationReport", "consolidate_poles", "refit_poles_to_greens"]


@dataclass
class ConsolidationReport:
    """What consolidation did to one pole set."""

    n_in: int                  #: poles carrying weight before
    n_out: int                 #: poles after merging
    dropped_weight: float      #: weight fraction of discarded ~zero poles
    resolution: float          #: the merge scale used (units of the poles)
    matsubara_err_merged: float    #: rel. G-mismatch after step 2
    matsubara_err_polished: float  #: ... after step 3 (== merged if no polish)

    def __str__(self) -> str:
        return (f"{self.n_in} -> {self.n_out} poles at resolution "
                f"{self.resolution:g}; Matsubara mismatch "
                f"{self.matsubara_err_merged:.2e} -> "
                f"{self.matsubara_err_polished:.2e}")


def _matsubara_greens(eps, gamma, a, b, z):
    """F(z_n) = sum_p r_p / (z_n - z_p) for complex z (grid-free)."""
    zp = torch.complex(eps, -gamma)
    r = torch.complex(a, b)
    return (r.unsqueeze(-1) / (z.unsqueeze(0) - zp.unsqueeze(-1))).sum(0)


def _rel_err(F, F_ref):
    return float(((F - F_ref).abs().square().sum()
                  / F_ref.abs().square().sum().clamp_min(1e-300)).sqrt())


def _cluster(eps: torch.Tensor, a: torch.Tensor, resolution: float) -> list[list[int]]:
    """Greedy 1-D clustering against the running weighted centroid."""
    order = torch.argsort(eps).tolist()
    clusters: list[list[int]] = []
    cent, wsum = None, 0.0
    for i in order:
        e, w = float(eps[i]), max(float(a[i]), 1e-300)
        if cent is not None and abs(e - cent) <= resolution:
            clusters[-1].append(i)
            cent = (cent * wsum + e * w) / (wsum + w)
            wsum += w
        else:
            clusters.append([i])
            cent, wsum = e, w
    return clusters


def _polish(eps, gamma, a, b, F_ref, z, gamma_floor, steps, lr):
    """Re-fit merged poles to the original Matsubara data, constraints kept.

    Tracks the best-seen parameters: Adam's normalised steps random-walk the
    parameters even at negligible loss (an exact merge would otherwise be
    *degraded* by its own polish), so the returned state is the best visited,
    never merely the last.
    """
    A_tot, B_tot = a.sum(), b.sum()
    ref_scale = F_ref.abs().square().sum().clamp_min(1e-300)
    u_e = eps.clone().requires_grad_(True)
    # inverse softplus for the width offset above the floor
    g0 = (gamma - gamma_floor).clamp_min(1e-6)
    u_g = torch.log(torch.expm1(g0)).requires_grad_(True)
    u_a = torch.log(a.clamp_min(1e-12)).requires_grad_(True)
    u_b = b.clone().requires_grad_(True)
    opt = torch.optim.Adam([u_e, u_g, u_a, u_b], lr=lr)

    def state():
        g = gamma_floor + torch.nn.functional.softplus(u_g)
        return u_e.detach().clone(), g.detach().clone(), \
            torch.exp(u_a).detach().clone(), u_b.detach().clone()

    best = state()
    with torch.no_grad():
        F0 = _matsubara_greens(*best, z)
        best_loss = float((F0 - F_ref).abs().square().sum())
    if best_loss / float(ref_scale) < 1e-14:
        steps = 0                          # already exact; nothing to polish

    for _ in range(steps):
        cand = state()                     # the params this loss belongs to
        g = gamma_floor + torch.nn.functional.softplus(u_g)
        aa = torch.exp(u_a)
        F = _matsubara_greens(u_e, g, aa, u_b, z)
        loss = (F - F_ref).abs().square().sum()
        val = float(loss.detach())
        if val < best_loss:
            best_loss, best = val, cand
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        u_e_, g, aa, bb = (t.clone() for t in best)
        u_e = u_e_
        # restore the sum rules exactly; the polish moves them only slightly,
        # so these are eps-scale corrections, but exact is exact
        aa = aa * (A_tot / aa.sum().clamp_min(1e-300))
        bb = bb + (B_tot - bb.sum()) * aa / aa.sum()
        # keep the per-pole asymmetry bound, then re-restore the b sum
        bb = torch.clamp(bb, -aa, aa)
        bb = bb + (B_tot - bb.sum()) * aa / aa.sum()
        bb = torch.clamp(bb, -aa, aa)
    return u_e.detach(), g.detach(), aa.detach(), bb.detach()


def consolidate_poles(
    poles: PoleSet,
    resolution: float,
    weight_floor: float = 1e-4,
    polish_steps: int = 300,
    polish_lr: float = 0.02,
    n_matsubara: int = 64,
    gamma_floor: float | None = None,
) -> tuple[PoleSet, list[ConsolidationReport]]:
    r"""Merge sub-resolution pole structure; return the minimal-style set.

    Parameters
    ----------
    poles
        ``(B, P)`` pole set, any units; ``resolution`` must be in the same
        units (physical poles with ``resolution = pi/beta`` is the default
        pipeline use).
    weight_floor
        Poles below this fraction of the total weight are absorbed into the
        merge bookkeeping but cannot seed a cluster; their (tiny) weight is
        reported as ``dropped_weight`` if they sit isolated.
    polish_steps
        0 disables the Matsubara re-fit.
    n_matsubara
        Fit points :math:`z_n = i(2n{+}1)\,\rho`.  At
        :math:`\rho = \pi/\beta` these are exactly the fermionic Matsubara
        frequencies -- the complete information content of :math:`G(\tau)`.

    Returns
    -------
    (PoleSet, list[ConsolidationReport])
        Batched pole set padded with zero-weight entries to the largest
        cluster count (a zero-weight pole is spectrally inert and pruned by
        the positivity certificate); one report per sample.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    B, P = poles.shape
    eps = poles.eps.detach().double()
    gam = poles.gamma.detach().double()
    a = poles.a.detach().double()
    b = poles.b.detach().double()
    z = 1j * resolution * torch.arange(1, 2 * n_matsubara, 2,
                                       dtype=torch.float64)

    out, reports = [], []
    for i in range(B):
        total = float(a[i].sum())
        keep = a[i] > weight_floor * max(total, 1e-300)
        dropped = float(a[i][~keep].sum()) / max(total, 1e-300)
        e_i, g_i, a_i, b_i = eps[i][keep], gam[i][keep], a[i][keep], b[i][keep]
        # the dropped poles' b must not break the zero-sum rule
        b_i = b_i + float(b[i][~keep].sum()) * a_i / a_i.sum()

        F_ref = _matsubara_greens(eps[i], gam[i], a[i], b[i], z)

        clusters = _cluster(e_i, a_i, resolution)
        e_m, g_m, a_m, b_m = [], [], [], []
        for cl in clusters:
            idx = torch.tensor(cl)
            ac = a_i[idx]
            A = ac.sum()
            e_new = float((ac * e_i[idx]).sum() / A)
            mad = float((ac * (e_i[idx] - e_new).abs()).sum() / A)
            e_m.append(e_new)
            g_m.append(float((ac * g_i[idx]).sum() / A) + mad)
            a_m.append(float(A))
            b_m.append(float(b_i[idx].sum()))
        e_m = torch.tensor(e_m, dtype=torch.float64)
        g_m = torch.tensor(g_m, dtype=torch.float64)
        a_m = torch.tensor(a_m, dtype=torch.float64)
        b_m = torch.tensor(b_m, dtype=torch.float64)

        err_merged = _rel_err(_matsubara_greens(e_m, g_m, a_m, b_m, z), F_ref)
        err_pol = err_merged
        if polish_steps > 0 and e_m.numel() > 0:
            floor = float(gam[i][keep].min()) if gamma_floor is None \
                else float(gamma_floor)
            e_m, g_m, a_m, b_m = _polish(e_m, g_m, a_m, b_m, F_ref, z,
                                         floor, polish_steps, polish_lr)
            err_pol = _rel_err(_matsubara_greens(e_m, g_m, a_m, b_m, z), F_ref)

        out.append((e_m, g_m, a_m, b_m))
        reports.append(ConsolidationReport(
            n_in=int(keep.sum()), n_out=int(e_m.numel()),
            dropped_weight=dropped, resolution=float(resolution),
            matsubara_err_merged=err_merged, matsubara_err_polished=err_pol))

    K = max(o[0].numel() for o in out)
    dtype = poles.eps.dtype

    def pad(vals, fill=0.0, width=1.0):
        t = torch.full((B, K), fill, dtype=torch.float64)
        for j, v in enumerate(vals):
            t[j, : v.numel()] = v
        return t.to(dtype)

    merged = PoleSet(
        eps=pad([o[0] for o in out]),
        gamma=pad([o[1] for o in out], fill=1.0),   # inert: zero weight below
        a=pad([o[2] for o in out]),
        b=pad([o[3] for o in out]),
    )
    return merged, reports


def refit_poles_to_greens(
    poles: PoleSet,
    beta: float,
    x: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    slices: torch.Tensor | None = None,
    gamma_floor: float | None = None,
    steps: int = 2000,
    lr: float = 0.02,
) -> tuple[PoleSet, float, float]:
    r"""Refit a (physical-unit) pole set directly to measured :math:`G(\tau)`.

    The missing last step of consolidation.  Merging at the thermal scale is
    **not** automatically :math:`\chi^2`-neutral -- the information limit is
    set by the noise, not by the temperature, and good data resolves some
    sub-thermal structure -- so the merged poles are re-optimised against the
    whitened data itself, through the exact forward map, with every
    constraint held *structurally* (the decoder's own parametrisation):
    :math:`\gamma = \gamma_{\min} + \mathrm{softplus}`,
    :math:`a = A_{\rm tot}\,\mathrm{softmax}(u)` (sum rule exact),
    :math:`b = a\,(t - \langle t\rangle_a)`, :math:`t = \tanh(u)/2`
    (:math:`\sum b = 0` exact, :math:`|b| \le a`).

    Returns ``(refit_poles, chi2_before, chi2_after)`` with poles in the
    input's physical units.  Zero-weight padding poles are dropped first.
    """
    from .forward import PoleGreens, SpectralQuadrature

    live = poles.a[0] > 0
    eps0 = (poles.eps[0][live] * beta).double()          # dimensionless
    gam0 = (poles.gamma[0][live] * beta).double()
    a0 = poles.a[0][live].double()
    b0 = poles.b[0][live].double()
    A_tot = a0.sum()
    # The width floor is the *physics* resolution limit (the decoder's own
    # 0.1 pi), never an accident of the input: a merged pole is broader than
    # its members by construction, and flooring at the merged minimum would
    # lock the refit above the widths the data actually wants.
    floor = 0.1 * math.pi if gamma_floor is None else float(gamma_floor) * float(beta)
    floor = min(floor, float(gam0.min()) * 0.999)

    quad = SpectralQuadrature.for_beta(float(beta), dtype=torch.float64)
    fwd = PoleGreens(quad)
    xd = x.double()
    K = fwd.kernel_matrix(xd)
    G = G_obs.reshape(1, -1).double()
    s = sigma.reshape(1, -1).double()
    W = None if inv_sqrt_R is None else inv_sqrt_R.double()

    def chi2_of(ps_dimless):
        Gp = fwd(ps_dimless, xd, kernel_matrix=K)
        d = (Gp - G) / s
        if slices is not None:
            d = d[:, slices]
        if W is not None:
            d = d @ W.transpose(-1, -2)
        return (d * d).sum(-1).mean() / d.shape[-1]

    u_e = eps0.clone().requires_grad_(True)
    u_g = torch.log(torch.expm1((gam0 - floor).clamp_min(1e-6))).requires_grad_(True)
    u_a = torch.log(a0.clamp_min(1e-12)).requires_grad_(True)
    t0 = (b0 / a0.clamp_min(1e-12)).clamp(-0.49, 0.49)
    u_b = torch.atanh(2.0 * t0).requires_grad_(True)

    def assemble():
        g = floor + torch.nn.functional.softplus(u_g)
        a = A_tot * torch.softmax(u_a, dim=-1)
        t = 0.5 * torch.tanh(u_b)
        b = a * (t - (a * t).sum() / A_tot)
        return PoleSet(eps=u_e.unsqueeze(0), gamma=g.unsqueeze(0),
                       a=a.unsqueeze(0), b=b.unsqueeze(0))

    with torch.no_grad():
        chi2_before = float(chi2_of(assemble()))
    best_loss, best = chi2_before, None
    opt = torch.optim.Adam([u_e, u_g, u_a, u_b], lr=lr)
    for _ in range(steps):
        ps = assemble()
        loss = chi2_of(ps)
        val = float(loss.detach())
        if val < best_loss:
            best_loss = val
            best = PoleSet(*(t.detach().clone()
                             for t in (ps.eps, ps.gamma, ps.a, ps.b)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if best is None:
        best = PoleSet(*(t.detach() for t in
                         (lambda p: (p.eps, p.gamma, p.a, p.b))(assemble())))
    out = PoleSet(eps=(best.eps / beta).to(poles.eps.dtype),
                  gamma=(best.gamma / beta).to(poles.eps.dtype),
                  a=best.a.to(poles.eps.dtype),
                  b=best.b.to(poles.eps.dtype))
    return out, chi2_before, best_loss
