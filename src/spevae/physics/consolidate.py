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

__all__ = ["ConsolidationReport", "consolidate_poles",
           "refit_poles_to_greens", "minimal_pole_fixed_point",
           "FixedPointTrace", "rank_scan", "RankScanTrace",
           "pole_significance_table", "sharpness_verdict",
           "feature_ledger"]


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
    ph_symmetric: bool = False,
    fold_tolerance: float = 0.0,
    positivity_weight: float = 0.0,
    pair_bound: bool = False,
    anchor_A: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
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

    Parameters beyond the plain fit
    -------------------------------
    ph_symmetric
        Mirror-constrained refit for particle-hole-symmetric data: the input
        is folded onto the positive half-line (:math:`\varepsilon \to
        |\varepsilon|`, :math:`b \to \mathrm{sign}(\varepsilon)\,b`), half
        the poles are optimised, and the full set is their exact mirror
        expansion :math:`(\pm\varepsilon, \gamma, a/2, \pm b/2)`.
        :math:`A(\omega) = A(-\omega)` and :math:`\sum b = 0` then hold
        *identically*, and the free parameter count halves.
    fold_tolerance
        Physical-frequency scale for merging folded half-poles that land on
        top of each other (a :math:`\pm\varepsilon` pair from the
        unconstrained stage folds to two coincident half-poles).  0 keeps
        every fold separate.
    positivity_weight
        Weight of :math:`\int \mathrm{ReLU}(-A)^2 / \int A^2` on the
        quadrature -- the same penalty training uses, previously absent from
        this stage, which is exactly where in-gap negative dips came from.
    pair_bound
        Mirror mode only: **structural** positivity.  A mirror pair's
        numerator polynomial is the even quadratic
        :math:`(a\gamma - b\varepsilon)\,\omega^2 +
        (\varepsilon^2+\gamma^2)(a\gamma + b\varepsilon)`, so the pair is
        non-negative on all of :math:`\mathbb{R}` **iff**
        :math:`|b| \le a\gamma/\varepsilon` -- exact and closed-form.  The
        smooth sufficient bound
        :math:`|b| \le a\gamma/\sqrt{\gamma^2+\varepsilon^2}` is imposed in
        the parametrisation, making every pair -- hence the total --
        certified :math:`A \ge 0` *by construction*, with asymmetry retained
        up to the bound.  This is the guarantee a penalty cannot give.

    Returns ``(refit_poles, chi2_before, chi2_after)`` with poles in the
    input's physical units; the reported values are pure :math:`\chi^2`
    (the positivity penalty steers, it is not smuggled into the score).
    Zero-weight padding poles are dropped first.
    """
    from .forward import PoleGreens, SpectralQuadrature

    live = poles.a[0] > 0
    eps0 = (poles.eps[0][live] * beta).double()          # dimensionless
    gam0 = (poles.gamma[0][live] * beta).double()
    a0 = poles.a[0][live].double()
    b0 = poles.b[0][live].double()
    A_tot = a0.sum()

    if ph_symmetric:
        # Fold onto the positive half-line.  sign(0) = 0 zeroes b for an
        # on-axis pole, which the mirror expansion would cancel anyway.
        sgn = torch.sign(eps0)
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        eps0, b0 = eps0.abs(), sgn * b0
        if fold_tolerance > 0.0:
            tol = float(fold_tolerance) * float(beta)
            clusters = _cluster(eps0, a0, tol)
            e_m, g_m, a_m, b_m = [], [], [], []
            for cl in clusters:
                idx = torch.tensor(cl)
                ac = a0[idx]
                A = ac.sum()
                e_new = float((ac * eps0[idx]).sum() / A)
                mad = float((ac * (eps0[idx] - e_new).abs()).sum() / A)
                e_m.append(e_new)
                g_m.append(float((ac * gam0[idx]).sum() / A) + mad)
                a_m.append(float(A))
                b_m.append(float(b0[idx].sum()))
            eps0 = torch.tensor(e_m, dtype=torch.float64)
            gam0 = torch.tensor(g_m, dtype=torch.float64)
            a0 = torch.tensor(a_m, dtype=torch.float64)
            b0 = torch.tensor(b_m, dtype=torch.float64)
    # Default floor: the physics resolution limit (the decoder's own 0.1 pi),
    # capped below the input's own minimum so a merged start is never locked
    # above the widths the data wants.  An EXPLICIT gamma_floor is honored
    # exactly and never capped -- it is a hypothesis (e.g. the smeared arm of
    # a sharpness test), and silently weakening it once turned that test into
    # a comparison of two identical constraints.  A tensor gamma_floor gives a
    # PER-POLE floor (physical units, length = live pole count), which is what
    # per-feature hypothesis tests need: constrain one pole, free the rest.
    if gamma_floor is None:
        floor = min(0.1 * math.pi, float(gam0.min()) * 0.999)
    elif torch.is_tensor(gamma_floor):
        floor = (gamma_floor.double() * float(beta)).clamp_min(1e-6)
    else:
        floor = float(gamma_floor) * float(beta)

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

    wq = quad.weights
    nodes = quad.nodes

    A_anc = None if anchor_A is None else anchor_A.reshape(1, -1).double()

    def objective_of(ps_dimless):
        loss = chi2_of(ps_dimless)
        if positivity_weight > 0.0:
            A = ps_dimless.spectral(nodes)
            neg = torch.relu(-A)
            pen = ((neg * neg) * wq).sum() / ((A * A) * wq).sum().clamp_min(1e-300)
            loss = loss + positivity_weight * pen
        if anchor_weight > 0.0 and A_anc is not None:
            # manifold prior force: chi2 stays the reported/selected number,
            # the anchor term only shapes the descent (prior never votes in
            # the LR count test)
            A = ps_dimless.spectral(nodes)
            num = (((A - A_anc) ** 2) * wq).sum(-1)
            den = ((A_anc ** 2) * wq).sum(-1).clamp_min(1e-300)
            loss = loss + anchor_weight * (num / den).mean()
        return loss

    u_e = eps0.clone().requires_grad_(True)
    u_g = torch.log(torch.expm1((gam0 - floor).clamp_min(1e-6))).requires_grad_(True)
    u_a = torch.log(a0.clamp_min(1e-12)).requires_grad_(True)
    t0 = (b0 / a0.clamp_min(1e-12)).clamp(-0.49, 0.49)
    u_b = torch.atanh(2.0 * t0).requires_grad_(True)

    def assemble():
        g = floor + torch.nn.functional.softplus(u_g)
        a = A_tot * torch.softmax(u_a, dim=-1)
        t = 0.5 * torch.tanh(u_b)
        if ph_symmetric:
            # exact mirror expansion; no b-centering needed -- the mirror IS
            # the zero-sum mechanism, and centering would steal a half-set DOF
            if pair_bound:
                # |b| <= a*gamma/sqrt(gamma^2+eps^2) <= min(a, a*gamma/|eps|):
                # every pair non-negative everywhere, structurally
                b = a * (g / torch.sqrt(g * g + u_e * u_e)) * torch.tanh(u_b)
            else:
                b = a * t
            return PoleSet(
                eps=torch.cat([u_e, -u_e]).unsqueeze(0),
                gamma=torch.cat([g, g]).unsqueeze(0),
                a=0.5 * torch.cat([a, a]).unsqueeze(0),
                b=0.5 * torch.cat([b, -b]).unsqueeze(0),
            )
        b = a * (t - (a * t).sum() / A_tot)
        return PoleSet(eps=u_e.unsqueeze(0), gamma=g.unsqueeze(0),
                       a=a.unsqueeze(0), b=b.unsqueeze(0))

    with torch.no_grad():
        first = assemble()
        chi2_before = float(chi2_of(first))
        best_obj = float(objective_of(first))
    best = PoleSet(*(t.detach().clone()
                     for t in (first.eps, first.gamma, first.a, first.b)))
    opt = torch.optim.Adam([u_e, u_g, u_a, u_b], lr=lr)
    for _ in range(steps):
        ps = assemble()
        loss = objective_of(ps)
        val = float(loss.detach())
        if val < best_obj:
            best_obj = val
            best = PoleSet(*(t.detach().clone()
                             for t in (ps.eps, ps.gamma, ps.a, ps.b)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        chi2_after = float(chi2_of(best))
    out = PoleSet(eps=(best.eps / beta).to(poles.eps.dtype),
                  gamma=(best.gamma / beta).to(poles.eps.dtype),
                  a=best.a.to(poles.eps.dtype),
                  b=best.b.to(poles.eps.dtype))
    return out, chi2_before, chi2_after


@dataclass
class FixedPointTrace:
    """What one fixed-point consolidation did, iteration by iteration."""

    events: list                  #: strings: "merge 16->9", "prune eps=+1.2 (dchi2 0.3)", ...
    n_iterations: int
    n_final: int
    chi2_final: float             #: per-dof, whitened
    pruned_total: int
    merged_total: int

    def __str__(self) -> str:
        return (f"fixed point in {self.n_iterations} iterations: "
                f"{self.n_final} poles, chi2 {self.chi2_final:.4g} "
                f"({self.merged_total} merged, {self.pruned_total} pruned)\n  "
                + "\n  ".join(self.events))


def minimal_pole_fixed_point(
    poles: PoleSet,
    beta: float,
    x: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    slices: torch.Tensor | None = None,
    resolution: float | None = None,
    refit_steps: int = 6000,
    trial_steps: int = 1500,
    prune_quantile: float = 9.5,
    positivity_weight: float = 1.0,
    max_iterations: int = 10,
) -> tuple[PoleSet, FixedPointTrace]:
    r"""Merge / refit / prune to a fixed point: the operational minimal pole set.

    The loop alternates three moves until a full pass changes nothing:

    * **merge** poles within ``resolution`` (default :math:`\pi/\beta`) of a
      weighted centroid -- redundancy removal, weight conserved exactly;
    * **refit** the survivors to the data through the exact forward map,
      every constraint structural;
    * **prune** by likelihood ratio: remove one pole (weakest first), refit
      briefly, and accept the removal iff the *total* whitened
      :math:`\chi^2` rises by less than ``prune_quantile`` (default 9.5,
      the 95% point of :math:`\chi^2_4` -- one pole = four parameters).
      A pole whose deletion the data cannot detect at that level is
      unsupported structure.

    Merging asks "are these two distinguishable from one?"; pruning asks
    "is this one distinguishable from nothing?".  Both are required: a merge
    can never remove an isolated spurious pole, and a prune can never tidy a
    comb of significant but redundant ones.

    Returns the fixed-point :class:`PoleSet` (physical units, batch of 1)
    and a :class:`FixedPointTrace` documenting every accepted move.
    """
    res = float(resolution) if resolution is not None else math.pi / float(beta)
    n_dof = int(slices.numel()) if slices is not None else int(x.numel())

    def live_count(ps):
        return int((ps.a[0] > 1e-6 * float(ps.a[0].sum())).sum())

    def refit(ps, steps):
        return refit_poles_to_greens(
            ps, beta, x, G_obs, sigma, inv_sqrt_R=inv_sqrt_R, slices=slices,
            steps=steps, positivity_weight=positivity_weight)

    def drop(ps, j):
        keep = [i for i in range(ps.shape[1]) if i != j
                and float(ps.a[0][i]) > 0]
        return PoleSet(eps=ps.eps[:, keep], gamma=ps.gamma[:, keep],
                       a=ps.a[:, keep], b=ps.b[:, keep])

    events: list[str] = []
    merged_total = pruned_total = 0

    current, _, chi2_cur = refit(poles, refit_steps)
    events.append(f"initial refit: {live_count(current)} poles, "
                  f"chi2 {chi2_cur:.4g}")

    for it in range(1, max_iterations + 1):
        changed = False

        # -- merge phase ------------------------------------------------- #
        n_before = live_count(current)
        merged, _ = consolidate_poles(current, res, polish_steps=0)
        if live_count(merged) < n_before:
            merged_total += n_before - live_count(merged)
            current, _, chi2_cur = refit(merged, refit_steps)
            events.append(f"iter {it}: merge {n_before} -> "
                          f"{live_count(current)}, chi2 {chi2_cur:.4g}")
            changed = True

        # -- prune phase (weakest first, rescan after every success) ------ #
        while live_count(current) > 1:
            live = [i for i in range(current.shape[1])
                    if float(current.a[0][i]) > 0]
            order = sorted(live, key=lambda i: float(current.a[0][i]))
            accepted = False
            for j in order:
                cand, _, chi2_new = refit(drop(current, j), trial_steps)
                d_total = n_dof * (chi2_new - chi2_cur)
                # Gray zone: the short trial's statistic is inflated by
                # under-convergence (measured 3-30% relative), so a
                # near-threshold verdict gets the full budget before it
                # counts.  Far-from-threshold verdicts (the common case by
                # orders of magnitude) keep the cheap trial.
                if prune_quantile / 4 < d_total < 4 * prune_quantile \
                        and trial_steps < refit_steps:
                    cand, _, chi2_new = refit(drop(current, j), refit_steps)
                    d_total = n_dof * (chi2_new - chi2_cur)
                if d_total < prune_quantile:
                    events.append(
                        f"iter {it}: prune eps={float(current.eps[0][j]):+.3f} "
                        f"(a={float(current.a[0][j]):.4f}, "
                        f"n*dchi2={d_total:+.1f})")
                    current, chi2_cur = cand, chi2_new
                    pruned_total += 1
                    accepted = changed = True
                    break
            if not accepted:
                break

        if not changed:
            break

    # Final refit, iterated to convergence: a fixed budget was measured to
    # leave up to ~0.9 chi2/dof on the table (semicircle: 2.68 -> 1.76 with
    # more steps), which then contaminates any comparison made against the
    # fixed point.  Chunks continue until the relative improvement stalls.
    current, _, chi2_cur = refit(current, refit_steps)
    for _ in range(5):
        cand, _, c2 = refit(current, refit_steps // 2)
        if c2 >= chi2_cur * (1.0 - 1e-2):
            if c2 < chi2_cur:
                current, chi2_cur = cand, c2
            break
        events.append(f"convergence polish: chi2 {chi2_cur:.4g} -> {c2:.4g}")
        current, chi2_cur = cand, c2
    trace = FixedPointTrace(events=events, n_iterations=it,
                            n_final=live_count(current),
                            chi2_final=chi2_cur,
                            pruned_total=pruned_total,
                            merged_total=merged_total)
    return current, trace


def _smoothed_A(ps: PoleSet, w: torch.Tensor, res: float) -> torch.Tensor:
    """A(w) Gaussian-broadened at the resolution scale, on a uniform grid."""
    A = ps.spectral(w)[0]
    h = float(w[1] - w[0])
    n = int(4 * res / h) + 1
    k = torch.arange(-n, n + 1, dtype=A.dtype) * h
    ker = torch.exp(-0.5 * (k / res) ** 2)
    ker = ker / ker.sum()
    return torch.nn.functional.conv1d(
        A.reshape(1, 1, -1), ker.reshape(1, 1, -1), padding=n).reshape(-1)


def pole_significance_table(
    poles: PoleSet,
    beta: float,
    x: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    slices: torch.Tensor | None = None,
    resolution: float | None = None,
    trial_budgets: tuple[int, ...] = (1500, 6000),
    positivity_weight: float = 1.0,
) -> list[dict]:
    r"""Per-pole evidence: what removing each pole costs, two ways.

    For every live pole: remove it, refit (at each trial budget), and report

    * ``n_dchi2``: the likelihood-ratio statistic :math:`n\,\Delta\chi^2`
      per budget -- comparing budgets measures the *pruner bias*: an
      under-converged trial refit inflates the statistic and over-protects
      poles;
    * ``dA_res``: the relative change of the resolution-broadened spectrum,
      :math:`\int|\tilde A_{\rm without} - \tilde A_{\rm with}| /
      \int \tilde A` -- whether the pole matters *at the honest reporting
      scale*, or is sub-resolution decoration the ansatz uses to shape an
      envelope.
    """
    res = float(resolution) if resolution is not None else math.pi / float(beta)
    n_dof = int(slices.numel()) if slices is not None else int(x.numel())
    span = float(poles.eps[0].abs().max()) + 5.0 * res + \
        3.0 * float(poles.gamma[0].max())
    w = torch.linspace(-span, span, 4001, dtype=torch.float64)

    base, _, chi2_base = refit_poles_to_greens(
        poles, beta, x, G_obs, sigma, inv_sqrt_R=inv_sqrt_R, slices=slices,
        steps=max(trial_budgets), positivity_weight=positivity_weight)
    A_base = _smoothed_A(base.to(torch.float64), w, res)
    norm = float(torch.trapezoid(A_base.abs(), w))

    live = [i for i in range(base.shape[1]) if float(base.a[0][i]) > 1e-6]
    rows = []
    for j in live:
        keep = [i for i in live if i != j]
        cand0 = PoleSet(eps=base.eps[:, keep], gamma=base.gamma[:, keep],
                        a=base.a[:, keep], b=base.b[:, keep])
        row = {"eps": float(base.eps[0][j]), "gamma": float(base.gamma[0][j]),
               "a": float(base.a[0][j]), "n_dchi2": {}}
        for budget in trial_budgets:
            cand, _, c = refit_poles_to_greens(
                cand0, beta, x, G_obs, sigma, inv_sqrt_R=inv_sqrt_R,
                slices=slices, steps=budget,
                positivity_weight=positivity_weight)
            row["n_dchi2"][budget] = n_dof * (c - chi2_base)
            if budget == max(trial_budgets):
                A_c = _smoothed_A(cand.to(torch.float64), w, res)
                row["dA_res"] = float(torch.trapezoid((A_c - A_base).abs(), w)) / norm
        rows.append(row)
    return rows


def sharpness_verdict(
    poles: PoleSet,
    beta: float,
    x: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    slices: torch.Tensor | None = None,
    resolution: float | None = None,
    steps: int = 8000,
    threshold: float = 9.5,
    positivity_weight: float = 1.0,
) -> dict:
    r"""Is the sub-resolution sharpness of a pole set data-demanded?

    Likelihood-ratio test at matched optimisation budgets: the given set is
    refit twice from the same start -- once unconstrained ("sharp"), once
    with every width floored at the resolution scale ("smeared", the most
    resolution-limited configuration with the same pole structure).  The
    statistic :math:`n\,\Delta\chi^2 = n(\chi^2_{
m smeared} -
    \chi^2_{
m sharp})` decides:

    * above ``threshold``: the sharp structure is **data-demanded** -- a
      parametric feature pinned below the naive :math:`\pi/eta` limit, the
      way a line centre is located to a fraction of its width;
    * within :math:`\pm`\ ``threshold``: **undecidable** -- a measured flat
      direction of :math:`\chi^2`; whatever sharpness the reported curve
      shows there is the prior's assertion, and should be labelled as such.

    Matched budgets matter: the asymmetric version of this test produced a
    reversed verdict from convergence artifacts alone.
    """
    res = float(resolution) if resolution is not None else math.pi / float(beta)
    n_dof = int(slices.numel()) if slices is not None else int(x.numel())
    sharp, _, c_sharp = refit_poles_to_greens(
        poles, beta, x, G_obs, sigma, inv_sqrt_R=inv_sqrt_R, slices=slices,
        steps=steps, positivity_weight=positivity_weight)
    smeared, _, c_sm = refit_poles_to_greens(
        poles, beta, x, G_obs, sigma, inv_sqrt_R=inv_sqrt_R, slices=slices,
        steps=steps, positivity_weight=positivity_weight, gamma_floor=res)
    d = n_dof * (c_sm - c_sharp)
    return {
        "chi2_sharp": c_sharp, "chi2_smeared": c_sm, "n_dchi2": d,
        "data_demanded": bool(d > threshold),
        "verdict": ("sharp structure DATA-DEMANDED" if d > threshold else
                    "sharpness undecidable at this beta (prior assertion)"),
        "min_gamma_sharp": float(sharp.gamma[0][sharp.a[0] > 1e-6].min()),
        "resolution": res,
        "sharp": sharp, "smeared": smeared,
    }


def feature_ledger(
    poles: PoleSet,
    beta: float,
    x: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    slices: torch.Tensor | None = None,
    resolution: float | None = None,
    steps: int = 6000,
    threshold: float = 9.5,
    positivity_weight: float = 1.0,
) -> list[dict]:
    r"""Per-feature hypothesis tests: the evidence ledger of a pole set.

    For every live pole, two matched-budget likelihood-ratio tests against
    the converged baseline:

    * **existence** -- remove the pole, refit everything else:
      :math:`n\,\Delta\chi^2_{\rm exist}`.  Below ``threshold``: the pole is
      unsupported (the pruner's question, answered per feature).
    * **sharpness** -- only for poles narrower than the resolution: floor
      *this pole's* width at the resolution scale, free everything else:
      :math:`n\,\Delta\chi^2_{\rm sharp}`.  Above ``threshold``: the
      sub-resolution width is data-demanded for this specific feature
      (parametric superresolution); below: the sharpness there is the
      prior's assertion.

    The verdict per pole is one of ``"spurious"``, ``"sharp (data-demanded)"``,
    ``"sharpness undecidable"``, or ``"resolution-broad"`` (no sharpness
    question arises).
    """
    res = float(resolution) if resolution is not None else math.pi / float(beta)
    n_dof = int(slices.numel()) if slices is not None else int(x.numel())

    base, _, chi2_base = refit_poles_to_greens(
        poles, beta, x, G_obs, sigma, inv_sqrt_R=inv_sqrt_R, slices=slices,
        steps=steps, positivity_weight=positivity_weight)
    live = [i for i in range(base.shape[1]) if float(base.a[0][i]) > 1e-6]

    def sub(keep):
        return PoleSet(eps=base.eps[:, keep], gamma=base.gamma[:, keep],
                       a=base.a[:, keep], b=base.b[:, keep])

    rows = []
    for j in live:
        row = {"eps": float(base.eps[0][j]), "gamma": float(base.gamma[0][j]),
               "a": float(base.a[0][j]), "b": float(base.b[0][j])}
        _, _, c_wo = refit_poles_to_greens(
            sub([i for i in live if i != j]), beta, x, G_obs, sigma,
            inv_sqrt_R=inv_sqrt_R, slices=slices, steps=steps,
            positivity_weight=positivity_weight)
        row["n_dchi2_exist"] = n_dof * (c_wo - chi2_base)

        if row["gamma"] < res:
            floors = base.gamma[0][live].clone().double() * 0.0 + 1e-9
            floors[live.index(j)] = res
            _, _, c_sm = refit_poles_to_greens(
                sub(live), beta, x, G_obs, sigma,
                inv_sqrt_R=inv_sqrt_R, slices=slices, steps=steps,
                positivity_weight=positivity_weight, gamma_floor=floors)
            row["n_dchi2_sharp"] = n_dof * (c_sm - chi2_base)
        else:
            row["n_dchi2_sharp"] = None

        if row["n_dchi2_exist"] < threshold:
            row["verdict"] = "spurious"
        elif row["n_dchi2_sharp"] is None:
            row["verdict"] = "resolution-broad"
        elif row["n_dchi2_sharp"] > threshold:
            row["verdict"] = "sharp (data-demanded)"
        else:
            row["verdict"] = "sharpness undecidable"
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Rank scan: the global pole-count decision
# --------------------------------------------------------------------------- #

@dataclass
class RankScanTrace:
    """The chi^2(n) curve and the selection it led to.

    A sharp elbow means the pole count is data-determined; a flat shelf
    means the data genuinely cannot distinguish the counts on it -- print
    the curve, it is the honesty artifact.
    """

    ns: list                      #: candidate pole counts, ascending
    chi2s: list                   #: whitened per-dof chi^2 after full refit, per n
    fits_by_n: dict               #: n -> fitted PoleSet (physical units)
    selected_n: int
    chi2_selected: float
    chi2_best: float
    tolerance: float
    n_LR: int | None = None       #: pure-statistics count (always reported)
    capacity_w: float = 0.0       #: temper weight used in tempered selection
    full_fit: object = None       #: the pre-refit full-capacity configuration
    full_chi2: float | None = None  #: its whitened chi2 -- the prior-started
                                    #: rendering, reported alongside the
                                    #: minimal form (user insight 2026-08-26)

    def __str__(self) -> str:
        curve = "  ".join(f"{n}:{c:.3g}" for n, c in zip(self.ns, self.chi2s))
        return (f"rank scan [{curve}] -> n={self.selected_n} "
                f"(chi2 {self.chi2_selected:.4g}, best {self.chi2_best:.4g}, "
                f"tol x{self.tolerance:g})")


@dataclass
class LadderVerdict:
    """Shape classification of a chi^2(n) rank ladder.

    The noiseless dichotomy is exact (measured on the step-2 label bank
    and in Zhang & Gull's Fig. 2): a pole-closed spectrum CRASHES to the
    floor at its true count and shelves there; a continuum decays
    geometrically, every rung buying a steady factor.  Noise blurs it --
    both kinds stop at the calibrated floor -- so the verdict here reads
    the *arrival*: a cliff followed by a flat shelf is pole-closed-
    consistent, a steady multiplicative approach that is still improving
    when it enters the floor, with no shelf, is continuum-consistent.
    This is a descriptive diagnostic (calibrated on the three banked
    dual-output cases), not a certificate; the LR count remains the
    statistics.
    """

    verdict: str          #: "pole-closed" | "continuum-like" | "ambiguous"
    n_arrival: int        #: first count whose chi^2 enters the floor band
    r_arrival: float      #: improvement factor of the arriving step
    shelf: int            #: rungs that stay inside the floor band after it
    cliff_share: float    #: largest single-step share of the pre-arrival drop
    floor: float          #: the floor level used (calibrated target or best)

    def __str__(self) -> str:
        return (f"{self.verdict} (arrive n={self.n_arrival} x{self.r_arrival:.1f}, "
                f"shelf {self.shelf}, cliff share {self.cliff_share:.0%})")


def ladder_verdict(ns, chi2s, floor_target: float | None = None,
                   band: float = 2.0) -> LadderVerdict:
    """Classify a rank ladder's shape; see :class:`LadderVerdict`.

    ``floor_target`` is the calibrated perfect-fit chi^2 when known (the
    honest floor for noisy data); otherwise the curve's own minimum.
    ``band`` scales the floor into the arrival band.  Thresholds are
    calibrated on the banked exemplars: the asym triplet (pole-closed
    truth: arrival x9.8 after a x1300 cliff, shelf 4), the Holstein
    (arrival x4.9 still in a steady multiplicative regime, shelf <= 1),
    and the fano band (a resolved band at the comb/smooth boundary --
    genuinely ambiguous, and classified as such).
    """
    ns = list(ns)
    c = [float(v) for v in chi2s]
    floor = float(floor_target) if floor_target is not None else min(c)
    level = band * floor
    inside = [i for i, v in enumerate(c) if v <= level]
    if not inside:
        return LadderVerdict("ambiguous", ns[int(min(range(len(c)),
                             key=lambda i: c[i]))], 1.0, 0, 0.0, floor)
    k = inside[0]
    r_arr = c[k - 1] / c[k] if k > 0 else float("inf")
    # A shelf rung sits inside the floor band WITHOUT continuing to improve
    # materially -- on noisy data chi^2 hovers at the floor; a noiseless
    # curve that keeps dropping below a synthetic floor has not shelved.
    shelf = 0
    for i in range(k + 1, len(c)):
        if c[i] <= level and c[i] > c[k] / band**2:
            shelf += 1
        else:
            break
    drops = [math.log(max(c[i] / c[i + 1], 1.0)) for i in range(k)]
    total = sum(drops)
    cliff = max(drops) / total if total > 0 else 1.0 if k == 0 else 0.0
    # "steady": the arriving step's rate is consistent with the rung before
    # it -- a curve still in its multiplicative regime.  Deliberately
    # relative: a continuum converging at a steep 20x/rung is still a
    # continuum, which is also why arrival SPEED is not a discriminator
    # (measured overlap: the triplet crash arrives x9.8, a semicircle's
    # geometric rungs run x10-40); cliff DOMINANCE is the robust one.
    steady = (len(drops) >= 2 and drops[-1] > 0 and drops[-2] > 0
              and max(drops[-1], drops[-2]) / min(drops[-1], drops[-2]) <= 3.0)
    if shelf >= 2 and cliff >= 0.6:
        v = "pole-closed"
    elif shelf <= 1 and steady and cliff < 0.6:
        v = "continuum-like"
    else:
        v = "ambiguous"
    return LadderVerdict(v, ns[k], float(r_arr), shelf, float(cliff), floor)


def _merge_closest_pair(eps, gam, a, b):
    """Merge the closest pair (positions in units of the narrower width)."""
    P = eps.numel()
    d = (eps.unsqueeze(0) - eps.unsqueeze(1)).abs()
    scale = torch.minimum(gam.unsqueeze(0), gam.unsqueeze(1)).clamp_min(1e-12)
    d = d / scale + torch.eye(P, dtype=eps.dtype) * 1e30
    i, j = divmod(int(d.argmin()), P)
    keep = [k for k in range(P) if k not in (i, j)]
    w = (a[i] + a[j]).clamp_min(1e-30)
    eps_m = (a[i] * eps[i] + a[j] * eps[j]) / w
    gam_m = (a[i] * gam[i] + a[j] * gam[j]) / w
    return (torch.cat([eps[keep], eps_m[None]]),
            torch.cat([gam[keep], gam_m[None]]),
            torch.cat([a[keep], w[None]]),
            torch.cat([b[keep], (b[i] + b[j])[None]]))


def chi2_quantile_95_local(k: int) -> float:
    """Per-parameter LR price consistent with the chi^2_4 = 9.49 pole rule."""
    return 9.4877 / 4.0 * k


def rank_scan(
    poles: PoleSet,
    beta: float,
    x: torch.Tensor,
    G_obs: torch.Tensor,
    sigma: torch.Tensor,
    inv_sqrt_R: torch.Tensor | None = None,
    slices: torch.Tensor | None = None,
    n_max: int = 8,
    refit_steps: int = 2000,
    polish_steps: int = 2000,
    lr: float = 0.02,
    tolerance: float = 1.5,     # retained in the trace for context only
    prune_quantile: float = 9.5,
    restarts: int = 2,
    jitter: float = 0.4,
    seed: int = 0,
    gamma_floor: float | None = None,
    ph_symmetric: bool = False,
    anchor_A: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
    pre_refit_steps: int = 0,
    initial_reduce: str = "merge",
    capacity_c: list | None = None,
) -> tuple[PoleSet, RankScanTrace]:
    r"""Global pole-count selection: refit at every count, pick the elbow.

    The fixed point (:func:`minimal_pole_fixed_point`) is a *local* search:
    it prunes one pole at a time, and a wrong-count basin whose poles must
    reorganise wholesale under a removal blocks every individual step (the
    measured comb-aliasing trap).  The scan makes the one discrete decision
    -- how many poles -- globally instead: hierarchical weighted merging of
    the amortized set supplies an n-pole initialisation for every
    n = 1..n_max, each is refit *in full* against the whitened data (its own
    basin, not a pruning remnant), and the smallest n whose refit chi^2 lies
    within ``tolerance`` of the best over all n is selected -- the parsimony
    elbow, i.e. the likelihood-ratio doctrine applied to the whole curve
    rather than to one removal at a time.  This is the noisy-tau-data
    counterpart of the SVD-rank decision in the minimal pole method of
    Zhang & Gull: the noise floor, not the parameterisation, sets how many
    poles the data supports.

    Returns the selected, polished pole set (physical units) and the
    :class:`RankScanTrace` carrying the full chi^2(n) curve.
    """
    if ph_symmetric:
        raise NotImplementedError(
            "mirror-constrained rank scan is not validated yet (the folded "
            "descent duplicates half-poles); run the plain scan -- it "
            "recovers mirrored counts without symmetry knowledge -- or the "
            "mirror-constrained refit/fixed point directly")
    eps = poles.eps.reshape(-1).double().clone()
    gam = poles.gamma.reshape(-1).double().clone()
    a = poles.a.reshape(-1).double().clone()
    b = poles.b.reshape(-1).double().clone()
    live = a > 1e-9 * float(a.sum())
    eps, gam, a, b = eps[live], gam[live], a[live], b[live]

    if pre_refit_steps > 0:
        # refit the FULL live configuration before any reduction: the cold
        # merge was measured to destroy the proposal's arrangement (the
        # w = 0 control, 2026-08-26); candidates should inherit a fitted
        # configuration, not wreckage
        dt0 = poles.eps.dtype
        full = PoleSet(eps=eps[None].to(dt0), gamma=gam[None].to(dt0),
                       a=a[None].to(dt0), b=b[None].to(dt0))
        fitted0, _c0_full, _c_full = refit_poles_to_greens(
            full, beta, x, G_obs, sigma, inv_sqrt_R=inv_sqrt_R, slices=slices,
            gamma_floor=gamma_floor, steps=pre_refit_steps, lr=lr,
            ph_symmetric=ph_symmetric, anchor_A=anchor_A,
            anchor_weight=anchor_weight)
        eps = fitted0.eps.reshape(-1).double()
        gam = fitted0.gamma.reshape(-1).double()
        a = fitted0.a.reshape(-1).double()
        b = fitted0.b.reshape(-1).double()

    # cold-merge the amortized set down to n_max, then descend in n with a
    # WARM start: each n-pole candidate is the fitted (n+1)-pole result with
    # its closest pair merged.  Every fit therefore starts from a
    # data-consistent parent -- backward elimination with full refits --
    # which is what carries a candidate across the half-spacing moves a
    # cold merge-tree init cannot make.
    dt = poles.eps.dtype
    while eps.numel() > n_max:
        if initial_reduce == "drop":
            k = int(a.argmin())
            keep = [i for i in range(a.numel()) if i != k]
            scale = float(a.sum() / a[keep].sum().clamp_min(1e-30))
            eps, gam, a, b = eps[keep], gam[keep], a[keep] * scale, b[keep]
        else:
            eps, gam, a, b = _merge_closest_pair(eps, gam, a, b)

    ns: list[int] = []
    fits: dict[int, tuple[PoleSet, float]] = {}

    rng = torch.Generator().manual_seed(seed)
    drop = None
    while True:
        n = eps.numel()
        # Multi-start: refit landscapes at fixed n have rival basins of their
        # own (the measured 3-pole merger alias sits 0.35 chi^2/dof above the
        # truth basin).  Two deterministic starts -- the parent with its
        # closest pair merged, and the parent with its lightest pole dropped
        # (the right move when the parent is truth-plus-dust) -- plus
        # jittered restarts; chi^2 arbitrates, and at fixed n that
        # comparison IS statistically decisive, unlike across n.
        span = (eps.max() - eps.min()).clamp_min(1.0)
        spacing = span / max(n, 1)
        starts = [(eps, gam, a, b)]
        if drop is not None:
            starts.append(drop)
        for r in range(restarts):
            e0 = eps + jitter * spacing * (
                2.0 * torch.rand(eps.shape, generator=rng, dtype=eps.dtype) - 1.0)
            starts.append((e0, gam, a, b))
        best_fit, best_c = None, float("inf")
        for e0, g0, a0, b0 in starts:
            cand = PoleSet(eps=e0[None].to(dt), gamma=g0[None].to(dt),
                           a=a0[None].to(dt), b=b0[None].to(dt))
            fitted, _, c_end = refit_poles_to_greens(
                cand, beta, x, G_obs, sigma, inv_sqrt_R=inv_sqrt_R,
                slices=slices, gamma_floor=gamma_floor, steps=refit_steps,
                lr=lr, ph_symmetric=ph_symmetric,
                fold_tolerance=0.1 if ph_symmetric else 0.0,
                anchor_A=anchor_A, anchor_weight=anchor_weight)
            if float(c_end) < best_c:
                best_fit, best_c = fitted, float(c_end)
        fitted, c_end = best_fit, best_c
        n = int(fitted.a.shape[-1])          # ph refit expands to the full set
        ns.append(n)
        fits[n] = (fitted, float(c_end))
        if n <= (2 if ph_symmetric else 1):
            break
        eps = fitted.eps.reshape(-1).double()
        gam = fitted.gamma.reshape(-1).double()
        a = fitted.a.reshape(-1).double()
        b = fitted.b.reshape(-1).double()
        if ph_symmetric:
            # descend in the folded half-space: the expanded output is exact
            # mirror pairs, so the positive half (weights doubled) is a
            # complete representation; one half-space merge removes one
            # mirror PAIR from the full set, and the refit re-folds anyway.
            half = eps > 0
            eps, gam, a, b = eps[half], gam[half], 2.0 * a[half], 2.0 * b[half]
        if a.numel() > 1 and not ph_symmetric:
            k = int(a.argmin())
            keep = [i for i in range(a.numel()) if i != k]
            scale = float(a.sum() / a[keep].sum().clamp_min(1e-30))
            drop = (eps[keep], gam[keep], a[keep] * scale, b[keep])
        else:
            drop = None
        eps, gam, a, b = _merge_closest_pair(eps, gam, a, b)
    ns = sorted(ns)

    chi2s = [fits[n][1] for n in ns]
    best = min(chi2s)
    # Sequential likelihood ratio, smallest n first: accept n unless the
    # step to n+1 buys more than ``prune_quantile`` in TOTAL whitened chi^2
    # (the same chi^2_4 doctrine the pruner uses, applied to globally
    # optimised fits).  A relative band on per-dof chi^2 is too crude here:
    # a 2-pole merger can sit within 1.3x of the floor while the third pole
    # is still statistically demanded.
    L_eff = int(G_obs.reshape(-1).numel())

    def _repolish(idx):
        n_ = ns[idx]
        refit, _, c_ = refit_poles_to_greens(
            fits[n_][0], beta, x, G_obs, sigma, inv_sqrt_R=inv_sqrt_R,
            slices=slices, gamma_floor=gamma_floor, steps=polish_steps,
            lr=0.5 * lr, ph_symmetric=ph_symmetric,
            anchor_A=anchor_A, anchor_weight=anchor_weight)
        if float(c_) < chi2s[idx]:
            fits[n_] = (refit, float(c_))
            chi2s[idx] = float(c_)

    polished: set[int] = set()
    sel_lr = ns[-1]
    for k, n in enumerate(ns[:-1]):
        gain = (chi2s[k] - min(chi2s[k + 1:])) * L_eff
        # Borderline decisions are settled on floor-quality fits: Adam's
        # step-size orbit costs a few units of total chi^2, comparable to
        # the LR margin exactly when it matters.  Polish both competitors
        # at the finer rate before ruling; clear-cut gaps skip this.
        if 0.5 * prune_quantile < gain < 2.0 * prune_quantile:
            rival = k + 1 + int(torch.tensor(chi2s[k + 1:]).argmin())
            for idx in (k, rival):
                if ns[idx] not in polished:
                    _repolish(idx)
                    polished.add(ns[idx])
            gain = (chi2s[k] - min(chi2s[k + 1:])) * L_eff
        if gain < prune_quantile:
            sel_lr = n
            break
    sel = sel_lr
    if capacity_c is not None and anchor_weight > 0.0:
        # tempered selection: S(n) = chi2_tot + lambda0 k(n) + w c(n).
        # The LR price (lambda0 per parameter, from the same quantile
        # doctrine) is NEVER scaled; w scales only the prior's capacity
        # belief.  n_LR stays reported alongside.
        lam0 = chi2_quantile_95_local(1)
        S = []
        for k, n in enumerate(ns):
            kk = 4 * n - 2
            c_n = capacity_c[n - 1] if n - 1 < len(capacity_c) else 1e9
            S.append(chi2s[k] * L_eff + lam0 * kk + anchor_weight * c_n)
        sel = ns[int(min(range(len(S)), key=lambda i: S[i]))]
    selected, _, c_sel = refit_poles_to_greens(
        fits[sel][0], beta, x, G_obs, sigma, inv_sqrt_R=inv_sqrt_R,
        slices=slices, gamma_floor=gamma_floor, steps=polish_steps, lr=0.5 * lr,
        ph_symmetric=ph_symmetric, anchor_A=anchor_A,
        anchor_weight=anchor_weight)
    trace = RankScanTrace(ns=ns, chi2s=chi2s,
                          fits_by_n={n: fits[n][0] for n in ns},
                          n_LR=sel_lr, capacity_w=float(anchor_weight),
                          selected_n=sel, chi2_selected=float(c_sel),
                          chi2_best=best, tolerance=tolerance,
                          full_fit=(fitted0 if pre_refit_steps > 0 else None),
                          full_chi2=(float(_c_full) if pre_refit_steps > 0
                                     else None))
    return selected, trace
