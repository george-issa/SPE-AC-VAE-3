r"""A conditional flow-matching posterior over pole sets.

The VAE maps ``G -> z -> one pole set``.  This model learns the whole
conditional distribution :math:`p(\text{poles}\mid G)` and *samples* it:
different noise draws give different pole sets, each consistent with the data
in proportion to how often the training prior produces it.  The training
prior plays the role of the Bayesian prior; the noise added to ``G`` in
training plays the role of the likelihood.

Pole tokens
-----------
A spectrum is a set of up to ``P`` poles; each pole is one *token* carrying
four unconstrained coordinates

=====  ==========================  ==========================================
 u_e    asinh(eps / pi)            position, compressed far from the origin
 u_g    log(gamma - gamma_min)     width above the resolution floor
 u_a    log a  (>= A_DUST)         weight; softmax over tokens gives sum a = 1
 u_d    atanh(d / D)               dispersive amplitude, b = d - (sum d) a
=====  ==========================  ==========================================

plus a fifth *presence* channel (+1 a pole, -1 an empty slot).  Every decoded
sample satisfies causality, the width floor, positive normalised weights and
the zero-sum asymmetry by construction -- the same dispersive residue map as
the M1 decoder.  Empty slots (beyond a truth's pole count) sit at the mean
coordinates with presence -1 and decode to exactly zero weight; a first
version that marked them by log-weight -14 put them ~8 standard deviations
out and let them dominate the loss (measured: loss ~27 vs ~2).
The token set is unordered: the network uses no position information, so
permuting poles permutes its output.

Flow matching
-------------
Training pairs a noise draw ``x0`` with the label ``x1`` (Hungarian-matched
token to token, which straightens the paths), blends ``x_t = (1-t) x0 + t x1``
and regresses the velocity ``x1 - x0``.  Sampling integrates the learned
velocity from fresh noise with Heun steps.  The network is a transformer over
the pole tokens (self-attention: poles coordinate) that cross-attends to the
:class:`GreensEncoder`'s per-slice outputs (what the data says).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
from torch import nn

from ..physics.poles import DISPERSIVE_SCALE, PoleSet
from .constraints import DEFAULT_GAMMA_MIN, spectral_norm_from_greens
from .encoder import GreensEncoder

__all__ = ["FlowPosterior", "PoleCoords", "save_flow", "load_flow", "refine_samples",
           "add_pole", "A_DUST"]

#: Log-weight of a dust token (an empty slot).  exp(-14) ~ 8e-7.
A_DUST = -14.0
_D_CLIP = 0.995


class PoleCoords(nn.Module):
    """Pole sets <-> standardised token coordinates (dimensionless units)."""

    def __init__(self, gamma_min: float = DEFAULT_GAMMA_MIN, D: float = DISPERSIVE_SCALE):
        super().__init__()
        self.gamma_min = float(gamma_min)
        self.D = float(D)
        self.register_buffer("mean", torch.zeros(4))
        self.register_buffer("std", torch.ones(4))

    @staticmethod
    def gauge(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        r"""The ``c`` minimising :math:`\max_p |b_p + c a_p|` per row.

        ``d = b + c a`` is the dispersive amplitude that reproduces ``b`` (the
        map fixes d only up to ``c a`` because ``sum a = 1``); this choice keeps
        every |d_p| furthest inside the bound.  Exact: the optimum of the
        convex piecewise-linear max sits at a pairwise crossing.
        """
        num = b.unsqueeze(2) + b.unsqueeze(1)                     # (B, P, P)
        den = a.unsqueeze(2) + a.unsqueeze(1)
        ok = valid.unsqueeze(2) & valid.unsqueeze(1) & (den > 0)
        cand = torch.where(ok, -num / den.clamp_min(1e-30), torch.zeros_like(num))
        cand = torch.cat([cand.flatten(1), torch.zeros_like(a[:, :1])], 1)  # (B, K)
        val = (b.unsqueeze(1) + cand.unsqueeze(2) * a.unsqueeze(1)).abs()   # (B, K, P)
        val = torch.where(valid.unsqueeze(1), val, torch.zeros_like(val)).amax(-1)
        return cand.gather(1, val.argmin(1, keepdim=True)).squeeze(1)

    def raw(self, labels: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Label dict (dimensionless, padded) -> raw coordinates (B, P, 4), valid mask."""
        a, b = labels["a"], labels["b"]
        valid = a > 0
        s = a.sum(-1, keepdim=True).clamp_min(1e-30)
        a, b = a / s, b / s
        c = self.gauge(a, b, valid)
        d = (b + c.unsqueeze(-1) * a) / self.D
        u_e = torch.asinh(labels["eps"] / math.pi)
        u_g = torch.log((labels["gamma"] - self.gamma_min).clamp_min(1e-6))
        u_a = torch.log(a.clamp_min(1e-30)).clamp_min(A_DUST)
        u_d = torch.atanh(d.clamp(-_D_CLIP, _D_CLIP))
        dust = torch.stack([torch.zeros_like(a),
                            torch.full_like(a, math.log(1.0 - self.gamma_min)),
                            torch.full_like(a, A_DUST), torch.zeros_like(a)], -1)
        u = torch.stack([u_e, u_g, u_a, u_d], -1)
        return torch.where(valid.unsqueeze(-1), u, dust), valid

    def encode(self, labels: dict) -> torch.Tensor:
        """-> (B, P, 5): standardised (u_e, u_g, u_a, u_d) and presence +-1."""
        u, valid = self.raw(labels)
        x = (u - self.mean) / self.std
        x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
        pres = valid.to(x.dtype) * 2.0 - 1.0
        return torch.cat([x, pres.unsqueeze(-1)], -1)

    @torch.no_grad()
    def fit(self, labels_list: list[dict]) -> None:
        """Set the standardisation from real (non-dust) tokens of some labels."""
        us = []
        for lab in labels_list:
            u, valid = self.raw(lab)
            us.append(u[valid])
        u = torch.cat(us, 0)
        self.mean.copy_(u.mean(0))
        self.std.copy_(u.std(0).clamp_min(1e-3))

    def decode(self, x: torch.Tensor, norm: torch.Tensor | None = None,
               nu_max: float | None = None) -> PoleSet:
        """Coordinates (B, P, 5) -> dimensionless :class:`PoleSet`; absent
        slots come out as zero-weight padding."""
        u = x[..., :4] * self.std + self.mean
        present = x[..., 4] > 0
        none = ~present.any(-1, keepdim=True)            # keep at least one pole
        top = torch.zeros_like(present).scatter_(-1, x[..., 4].argmax(-1, keepdim=True), True)
        present = present | (none & top)
        eps = math.pi * torch.sinh(u[..., 0].clamp(-12.0, 12.0))
        if nu_max is not None:
            eps = eps.clamp(-nu_max, nu_max)
        gamma = self.gamma_min + torch.exp(u[..., 1].clamp(-12.0, 8.0))
        logit = torch.where(present, u[..., 2], torch.full_like(u[..., 2], -float("inf")))
        a = torch.softmax(logit, dim=-1)
        d = torch.where(present, self.D * torch.tanh(u[..., 3]), torch.zeros_like(u[..., 3]))
        b = d - d.sum(-1, keepdim=True) * a
        if norm is not None:
            n = norm.reshape(-1, 1)
            a, b = a * n, b * n
        return PoleSet(eps=eps, gamma=gamma, a=a, b=b)


def _time_features(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    ang = t.reshape(-1, 1) * f * 1000.0
    return torch.cat([ang.sin(), ang.cos()], -1)


class FlowPosterior(nn.Module):
    """Conditional flow-matching sampler of pole sets given G(tau)."""

    def __init__(self, num_poles: int = 12, d_model: int = 96, enc_layers: int = 2,
                 flow_layers: int = 4, n_heads: int = 8, d_ff: int = 512,
                 n_matsubara: int = 24, omega_c: float = 20.0,
                 gamma_min: float = DEFAULT_GAMMA_MIN,
                 dispersive_scale: float = DISPERSIVE_SCALE,
                 feedback: bool = False) -> None:
        super().__init__()
        self.config = dict(num_poles=num_poles, d_model=d_model, enc_layers=enc_layers,
                           flow_layers=flow_layers, n_heads=n_heads, d_ff=d_ff,
                           n_matsubara=n_matsubara, omega_c=omega_c, gamma_min=gamma_min,
                           dispersive_scale=dispersive_scale, feedback=bool(feedback))
        self.feedback = bool(feedback)
        #: Scale of the base noise at sampling time (1 = as trained).  A plain
        #: attribute, not a weight: a per-checkpoint calibration constant.
        self.base_scale = 1.0
        self.P = int(num_poles)
        self.omega_c = float(omega_c)
        self.coords = PoleCoords(gamma_min, dispersive_scale)
        self.encoder = GreensEncoder(latent_dim=2, d_model=d_model, n_layers=enc_layers,
                                     n_heads=n_heads, d_ff=d_ff, n_matsubara=n_matsubara)
        self.in_proj = nn.Linear(5, d_model)
        self.t_mlp = nn.Sequential(nn.Linear(64, d_model), nn.GELU(),
                                   nn.Linear(d_model, d_model))
        self.ctx_proj = nn.Linear(d_model, d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=d_ff, dropout=0.0,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.flow = nn.TransformerDecoder(layer, num_layers=flow_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 5))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)
        if self.feedback:
            # Simulator feedback (cf. Holzschuh & Thuerey, arXiv:2410.22573).  The
            # amortized flow never learns to use the data's precision: its raw samples
            # miss the data by 10^2-10^5 in chi^2 at low noise (measured), because the
            # information sits in the 4th-5th digit of G.  With feedback the velocity
            # network is shown, at every step, what the exact likelihood says about its
            # own current estimate -- one damped Gauss-Newton step in token coordinates
            # and the misfit, both in units of the error bar -- and learns how far to
            # follow it.  Zero-initialised: training starts from the plain flow.
            self.fb_proj = nn.Linear(4, d_model)
            self.fb_cond = nn.Linear(2, d_model)
            for m in (self.fb_proj, self.fb_cond):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        object.__setattr__(self, "_fb_cache", {})

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def nu_max(self, beta: float) -> float:
        return float(beta) * self.omega_c

    # -- conditioning ------------------------------------------------------- #
    def context(self, G: torch.Tensor, x: torch.Tensor, beta: float,
                sigma: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                includes_endpoints: bool = True):
        """``G -> (memory, keep, norm)``, scale stripped as in the VAE."""
        norm = spectral_norm_from_greens(G, includes_endpoints)
        denom = torch.where(norm.abs() > 1e-12, norm, torch.ones_like(norm))
        g_hat = G / denom.reshape(-1, 1)
        s_hat = None if sigma is None else sigma / denom.reshape(-1, 1).abs()
        h, keep = self.encoder.tokens(g_hat, x, sigma_hat=s_hat, beta=beta, mask=mask)
        return h, keep, norm

    def velocity(self, xt: torch.Tensor, t: torch.Tensor, memory: torch.Tensor,
                 keep: torch.Tensor, fb=None) -> torch.Tensor:
        h = self.in_proj(xt)
        cond = self.t_mlp(_time_features(t)) + self.ctx_proj(memory[:, 0])
        if fb is not None:
            step, logchi = fb
            h = h + self.fb_proj(step)
            cond = cond + self.fb_cond(torch.stack([logchi, torch.ones_like(logchi)], -1))
        h = h + cond.unsqueeze(1)
        h = self.flow(h, memory, memory_key_padding_mask=~keep)
        return self.out(h)

    # -- simulator feedback ------------------------------------------------- #
    def feedback_data(self, G, x, beta, sigma, norm, inv_sqrt_R=None, x_fit=None,
                      G_fit=None, sigma_fit=None) -> dict:
        """Everything :meth:`gn_feedback` needs about one batch of data (CPU, float64).

        ``G, sigma, norm`` are per row; ``*_fit`` name the fitted slices when
        they differ from the encoder's grid; ``inv_sqrt_R`` defaults to the
        identity.
        """
        from ..physics import PoleGreens, SpectralQuadrature
        dd = torch.float64
        key = ("fwd", round(float(beta), 9))
        if key not in self._fb_cache:
            self._fb_cache[key] = PoleGreens(SpectralQuadrature.for_beta(float(beta), dtype=dd))
        fwd = self._fb_cache[key]
        cpu = lambda v: v.detach().cpu().to(dd)        # two steps: MPS has no float64
        xf = cpu(x if x_fit is None else x_fit).reshape(-1)
        L = xf.numel()
        return {"fwd": fwd, "x": xf, "K": fwd.kernel_matrix(xf),
                "G": cpu(G if G_fit is None else G_fit).reshape(-1, L),
                "sigma": cpu(sigma if sigma_fit is None else sigma_fit).reshape(-1, L),
                "W": torch.eye(L, dtype=dd) if inv_sqrt_R is None else cpu(inv_sqrt_R),
                "norm": cpu(norm).reshape(-1), "nu_max": self.nu_max(beta)}

    @torch.no_grad()
    def gn_feedback(self, x_hat: torch.Tensor, data: dict):
        r"""What the exact likelihood says about an estimate ``x_hat`` (N, P, 5).

        Returns ``(step, logchi)``: one damped Gauss--Newton step of
        :math:`\tfrac12\chi^2` in token coordinates (``(N, P, 4)``, zero on
        absent tokens, clipped to :math:`\pm 4` prior standard deviations) and
        :math:`\log_{10}\chi^2/6` per point.  Float64 on the CPU: at relative
        noise 1e-5 the residual is not representable in float32.
        """
        from ..physics import rowwise_jacobian
        dd = torch.float64
        if "coords" not in self._fb_cache:                   # a CPU copy of the coordinate map
            c = PoleCoords(self.coords.gamma_min, self.coords.D)
            c.mean.copy_(self.coords.mean.detach().cpu())
            c.std.copy_(self.coords.std.detach().cpu())
            self._fb_cache["coords"] = c
        c = self._fb_cache["coords"]
        fwd, x, K, Wd = data["fwd"], data["x"], data["K"], data["W"]
        zc = x_hat.detach().cpu().to(dd)
        N, P, _ = zc.shape
        step = torch.zeros(N, P, 4, dtype=dd)
        logchi = torch.zeros(N, dtype=dd)
        for i0 in range(0, N, 256):
            rows = slice(i0, min(i0 + 256, N))
            u, pres = zc[rows, :, :4], zc[rows, :, 4:5]
            n = u.shape[0]
            present = zc[rows, :, 4] > 0
            top = torch.zeros_like(present).scatter_(-1, zc[rows, :, 4].argmax(-1, keepdim=True), True)
            present = present | (~present.any(-1, keepdim=True) & top)
            live = present.to(dd).unsqueeze(-1).expand(-1, -1, 4).reshape(n, -1)
            with torch.enable_grad():
                ug = u.clone().requires_grad_(True)
                ps = c.decode(torch.cat([ug, pres], -1), norm=data["norm"][rows],
                              nu_max=data["nu_max"])
                th = torch.stack([ps.eps, ps.gamma, ps.a, ps.b], -1).reshape(n, -1)
                Jt = rowwise_jacobian(th, ug)
            G, dG = fwd.jacobian(ps.detach(), x, K)
            sig = data["sigma"][rows]
            r = ((G - data["G"][rows]) / sig) @ Wd.T
            J = Wd @ ((dG.reshape(n, -1, 4 * P) @ Jt) / sig.unsqueeze(-1))
            A = J.transpose(1, 2) @ J
            g = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1) * live
            D = torch.diagonal(A, dim1=1, dim2=2).clamp_min(1e-12) * live
            M = (A * live.unsqueeze(1) * live.unsqueeze(2)
                 + torch.diag_embed(1e-2 * D + 1e-3 * live + (1.0 - live)))
            sol = torch.linalg.solve(M, -g.unsqueeze(-1)).squeeze(-1)
            step[rows] = torch.nan_to_num(sol).reshape(n, P, 4).clamp(-4.0, 4.0)
            logchi[rows] = torch.log10((r * r).mean(-1).clamp(1e-3, 1e12))
        return (step.to(x_hat.device, x_hat.dtype), (logchi / 6.0).to(x_hat.device, x_hat.dtype))

    def _velocity_fb(self, z, t, mem, keep, data):
        """Velocity with feedback: a first pass without it gives the estimate the
        likelihood is asked about."""
        v0 = self.velocity(z, t, mem, keep)
        if not self.feedback or data is None:
            return v0
        x_hat = z + (1.0 - t).reshape(-1, 1, 1) * v0
        return self.velocity(z, t, mem, keep, self.gn_feedback(x_hat, data))

    # -- training ----------------------------------------------------------- #
    def loss(self, batch, ot_pairing: bool = True,
             generator: torch.Generator | None = None) -> torch.Tensor:
        lab = batch.poles_true
        x1 = self.coords.encode({k: lab[k] for k in ("eps", "gamma", "a", "b")})
        B, P, _ = x1.shape
        if P < self.P:                                      # pad with dust slots
            pad = self.coords.encode({
                "eps": torch.zeros(B, self.P - P, device=x1.device),
                "gamma": torch.ones(B, self.P - P, device=x1.device),
                "a": torch.zeros(B, self.P - P, device=x1.device),
                "b": torch.zeros(B, self.P - P, device=x1.device)})
            x1 = torch.cat([x1, pad], 1)
        x0 = torch.randn(x1.shape, generator=generator, device="cpu").to(x1.device)
        if ot_pairing:
            x1 = _hungarian_pair(x0, x1)
        t = torch.rand(B, generator=generator, device="cpu").to(x1.device)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        mem, keep, norm = self.context(batch.G, batch.x, batch.beta, sigma=batch.sigma)
        v = self.velocity(xt, t, mem, keep)
        loss = ((v - (x1 - x0)) ** 2).mean()
        if not self.feedback:
            return loss
        # second pass: the same regression, now told what the exact likelihood says
        # about the first pass's estimate of x1 (no gradient through the feedback)
        data = self.feedback_data(batch.G, batch.x, batch.beta, batch.sigma, norm,
                                  inv_sqrt_R=batch.inv_sqrt_R)
        x_hat = (xt + (1 - t)[:, None, None] * v).detach()
        v2 = self.velocity(xt, t, mem, keep, self.gn_feedback(x_hat, data))
        return 0.5 * loss + ((v2 - (x1 - x0)) ** 2).mean()

    # -- sampling ----------------------------------------------------------- #
    @torch.no_grad()
    def sample(self, G: torch.Tensor, x: torch.Tensor, beta: float,
               sigma: torch.Tensor | None = None, n_samples: int = 64,
               steps: int = 25, generator: torch.Generator | None = None,
               includes_endpoints: bool = True, mask: torch.Tensor | None = None,
               **fit) -> PoleSet:
        """``n_samples`` posterior pole sets per input row, dimensionless.

        Returns a :class:`PoleSet` of batch ``B * n_samples`` (row-major: the
        samples of input 0 first).  Heun integration, ``2 * steps`` network
        evaluations.  ``fit`` is passed to :meth:`sample_coords`.
        """
        z, norm = self.sample_coords(G, x, beta, sigma=sigma, n_samples=n_samples,
                                     steps=steps, generator=generator,
                                     includes_endpoints=includes_endpoints, mask=mask, **fit)
        return self.coords.decode(z, norm=norm, nu_max=self.nu_max(beta))

    @torch.no_grad()
    def sample_coords(self, G: torch.Tensor, x: torch.Tensor, beta: float,
                      sigma: torch.Tensor | None = None, n_samples: int = 64,
                      steps: int = 25, generator: torch.Generator | None = None,
                      includes_endpoints: bool = True, mask: torch.Tensor | None = None,
                      inv_sqrt_R: torch.Tensor | None = None, x_fit=None, G_fit=None,
                      sigma_fit=None):
        """Like :meth:`sample` but returns ``(coords (B*S, P, 5), norm (B*S,))``.

        A model built with ``feedback=True`` also needs the likelihood's side
        of the data: ``inv_sqrt_R`` (identity if omitted) and, when the fit
        uses fewer slices than the encoder sees, ``x_fit, G_fit, sigma_fit``.
        """
        mem, keep, norm = self.context(G, x, beta, sigma=sigma, mask=mask,
                                       includes_endpoints=includes_endpoints)
        S = int(n_samples)
        data = None
        if self.feedback:
            if sigma is None and sigma_fit is None:
                raise ValueError("a feedback flow needs the error bars")
            rep_ = lambda v: None if v is None else v.reshape(G.shape[0], -1).repeat_interleave(S, 0)
            data = self.feedback_data(rep_(G), x, beta, rep_(sigma), norm.repeat_interleave(S, 0),
                                      inv_sqrt_R=inv_sqrt_R, x_fit=x_fit, G_fit=rep_(G_fit),
                                      sigma_fit=rep_(sigma_fit))
        mem = mem.repeat_interleave(S, 0)
        keep = keep.repeat_interleave(S, 0)
        norm = norm.repeat_interleave(S, 0)
        z = float(self.base_scale) * torch.randn(mem.shape[0], self.P, 5, generator=generator,
                                                 device="cpu").to(mem.device, mem.dtype)
        ts = torch.linspace(0.0, 1.0, steps + 1, device=mem.device, dtype=mem.dtype)
        for k in range(steps):
            t0, t1 = ts[k], ts[k + 1]
            v0 = self._velocity_fb(z, t0.expand(z.shape[0]), mem, keep, data)
            zp = z + (t1 - t0) * v0
            v1 = self._velocity_fb(zp, t1.expand(z.shape[0]), mem, keep, data)
            z = z + 0.5 * (t1 - t0) * (v0 + v1)
        return z, norm


def refine_samples(flow: FlowPosterior, z0: torch.Tensor, norm: torch.Tensor, beta: float,
                   x_fit: torch.Tensor, G_fit: torch.Tensor, sigma_fit: torch.Tensor,
                   inv_sqrt_R: torch.Tensor, lam: float, steps: int = 300, lr: float = 0.05,
                   perturb: bool = False, generator: torch.Generator | None = None,
                   anneal: float | None = None, lr_final: float | None = None,
                   anneal_from: float | None = None, solver: str = "lm",
                   lm_iters: int = 40, lm_stages: int = 6, lm_stage_iters: int = 6,
                   return_coords: bool = False):
    r"""Sample-then-refine: pull each flow sample onto the data.

    Each sample's token coordinates ``u`` minimise

    .. math:: \tfrac12 \chi^2_{\rm sum}(u;\,\tilde G) + \tfrac{\lambda}{2}
              \lVert u - u_k \rVert^2

    -- the exact whitened likelihood plus a Gaussian tether of precision
    ``lam`` to the sample's own start ``u_k`` (present tokens only; the
    presence pattern is kept), so the flow supplies *where* solutions may be
    and the likelihood supplies the data's precision.  With ``perturb`` each
    sample fits its own noise redraw :math:`\tilde G = G + \sigma \odot R^{1/2}
    \xi` (randomize-then-optimize), so the spread also reflects what the
    noise allows.  Samples are independent, so all of them run as one batch;
    each keeps the step of its own lowest objective.

    ``anneal`` (e.g. 30) starts the fit against error bars inflated by that
    factor and shrinks them geometrically to the true sigma over the first 70%
    of the steps -- a continuation that lets each sample settle into the basin
    nearest its own start before the full precision of the data bites;
    ``lr_final`` decays the step size along a cosine.  Measured motivation: at
    300 constant steps, samples at noise 1e-4 stayed at chi^2 ~175.
    ``anneal_from`` sets the start in ABSOLUTE terms instead: each row's
    error bars start at ``anneal_from`` (its median sigma scaled up to it,
    never down), so one setting serves every noise level -- the fit always
    begins where the posterior is broad.

    ``solver`` picks how the objective is minimised.  It is a nonlinear
    least-squares problem (L whitened residuals plus the tether), so the
    default ``"lm"`` is Levenberg--Marquardt with the exact pole Jacobian in
    float64 (:mod:`spevae.physics.lsq`): the same continuation in the error
    bars (``lm_stages`` stages of ``lm_stage_iters`` iterations from the
    ``anneal`` / ``anneal_from`` start), then up to ``lm_iters`` iterations at
    the data's own precision.  ``"adam"`` is the original first-order loop
    (``steps``, ``lr``, ``lr_final``), kept because the judgements banked
    before 2026-11-24 used it.  Measured on identical samples, targets and
    objective at noise 3e-5: Adam (3000 steps) ends at median chi^2 5.2 per
    point with 72% of samples above 2; LM at 1.2 with 26%, about 10x faster.
    Both keep each sample's pole count fixed -- what LM leaves unconverged is
    mostly samples with fewer poles than the truth.

    ``G_fit, sigma_fit`` are ``(N, L)`` (already repeated per sample),
    ``x_fit`` ``(L,)``, ``inv_sqrt_R`` ``(L, L)``.  Returns
    ``(PoleSet, chi2 per point (N,) against the UNperturbed data)``, plus the
    refined token coordinates ``(N, P, 4)`` when ``return_coords`` is set
    (what a second pass -- e.g. after a pole birth -- starts from).
    """
    from ..physics import PoleGreens, SpectralQuadrature
    quad = SpectralQuadrature.for_beta(beta).to(z0.device)
    fwd = PoleGreens(quad)
    dt = z0.dtype
    x_fit = x_fit.to(dt)
    K = fwd.kernel_matrix(x_fit)
    W = inv_sqrt_R.to(dt)
    target = G_fit.to(dt)
    if perturb:
        R_half = torch.linalg.inv(W)                       # symmetric sqrt of R
        xi = torch.randn(target.shape, generator=generator, device="cpu").to(dt)
        target = target + sigma_fit.to(dt) * (xi @ R_half.T)
    pres = z0[..., 4:5]
    keep = (pres > 0).to(dt)
    u0 = z0[..., :4].detach()
    u = u0.clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    nu_max = flow.nu_max(beta)
    best = torch.full((z0.shape[0],), float("inf"), dtype=dt)
    best_u = u0.clone()

    def objective(uu, data):
        ps = flow.coords.decode(torch.cat([uu, pres], -1), norm=norm, nu_max=nu_max)
        d = ((fwd(ps, x_fit, kernel_matrix=K) - data) / sigma_fit.to(dt)) @ W.T
        c2 = (d * d).sum(-1)
        prox = (keep * (uu - u0) ** 2).sum((-1, -2))
        return 0.5 * c2 + 0.5 * lam * prox, c2

    sig_true = sigma_fit.to(dt)
    t_anneal = max(int(0.7 * steps), 1)
    if anneal_from is not None:
        start = (float(anneal_from) / sig_true.median(-1).values).clamp_min(1.0).unsqueeze(-1)
    elif anneal is not None:
        start = torch.full((sig_true.shape[0], 1), float(anneal), dtype=dt)
    else:
        start = None
    if solver not in ("lm", "adam"):
        raise ValueError(f"solver={solver!r}; expected 'lm' or 'adam'")
    if solver == "lm":
        best_u = _refine_lm(flow, z0, norm, beta, x_fit, target, sig_true, W, lam, start,
                            lm_iters, lm_stages, lm_stage_iters).to(dt)
        steps = 0                                       # skip the first-order loop below
    for k in range(steps):
        fac = 1.0 if start is None or k >= t_anneal else start ** (1.0 - k / t_anneal)
        if lr_final is not None:
            for gp in opt.param_groups:
                gp["lr"] = lr_final + 0.5 * (lr - lr_final) * (1 + math.cos(math.pi * k / steps))
        sigma_fit = sig_true * fac
        obj, _ = objective(u, target)
        if start is None or k >= t_anneal:   # only full-precision objectives compete
            with torch.no_grad():
                better = obj < best
                best = torch.where(better, obj.detach(), best)
                best_u[better] = u.detach()[better]
        opt.zero_grad(set_to_none=True)
        obj.sum().backward()
        opt.step()
    sigma_fit = sig_true
    with torch.no_grad():
        if solver == "adam":
            obj, _ = objective(u, target)
            better = obj < best
            best_u[better] = u.detach()[better]
        ps = flow.coords.decode(torch.cat([best_u, pres], -1), norm=norm, nu_max=nu_max)
        d = ((fwd(ps, x_fit, kernel_matrix=K) - G_fit.to(dt)) / sigma_fit.to(dt)) @ W.T
        chi2 = (d * d).mean(-1)
    if return_coords:
        return ps, chi2, best_u.detach()
    return ps, chi2


@torch.no_grad()
def add_pole(flow: FlowPosterior, z: torch.Tensor, donors: PoleSet,
             generator: torch.Generator | None = None, weight: float = 0.03) -> tuple:
    """Switch on one dormant pole slot in each row of ``z`` (a *birth*).

    The refinement keeps each sample's pole count, so a sample the flow gave
    too few poles can never reach the data.  A birth places a new, light
    pole at the position and width of a pole borrowed at random from
    ``donors`` -- pole sets that DO fit the same data -- and leaves the fit to
    move it.  Rows with no dormant slot are returned unchanged.

    ``z`` is ``(N, P, 5)`` token coordinates (refined coordinates with their
    presence channel); ``donors`` a dimensionless :class:`PoleSet` whose
    non-zero-weight poles form the pool.  Returns ``(z_new, born)`` with
    ``born`` an ``(N,)`` bool mask.
    """
    c = flow.coords
    z = z.clone()
    live = donors.a != 0
    pool_e, pool_g = donors.eps[live], donors.gamma[live]
    present = z[..., 4] > 0
    born = ~present.all(-1) & (pool_e.numel() > 0)
    if not bool(born.any()):
        return z, born
    N = z.shape[0]
    pick = torch.randint(pool_e.numel(), (N,), generator=generator)
    u_e = torch.asinh(pool_e[pick].to(z.dtype) / math.pi)
    u_g = torch.log((pool_g[pick].to(z.dtype) - c.gamma_min).clamp_min(1e-6))
    # log-weight relative to the row's present poles: softmax share ~ `weight`
    raw_a = z[..., 2] * c.std[2] + c.mean[2]
    lse = torch.logsumexp(torch.where(present, raw_a, torch.full_like(raw_a, -float("inf"))), -1)
    u_a = lse + math.log(weight / (1.0 - weight))
    new = torch.stack([(u_e - c.mean[0]) / c.std[0], (u_g - c.mean[1]) / c.std[1],
                       (u_a - c.mean[2]) / c.std[2], (0.0 - c.mean[3]) / c.std[3] + 0 * u_a,
                       torch.ones_like(u_a)], -1)                       # (N, 5)
    slot = torch.where(present, torch.full_like(z[..., 4], -float("inf")), z[..., 4]).argmax(-1)
    rows = torch.nonzero(born).flatten()
    z[rows, slot[rows]] = new[rows]
    return z, born


def _refine_lm(flow: FlowPosterior, z0: torch.Tensor, norm: torch.Tensor, beta: float,
               x_fit: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor,
               W: torch.Tensor, lam: float, start: torch.Tensor | None, iters: int,
               stages: int, stage_iters: int, chunk: int = 256) -> torch.Tensor:
    """Levenberg--Marquardt on the refinement objective; returns ``u`` ``(N, P, 4)``.

    Residuals per sample: the whitened data misfit (L) and ``sqrt(lam)`` times
    the offset of each present token from its start (4 P).  Float64, exact
    Jacobian (pole derivatives in closed form, coordinate map by autograd),
    rows in chunks so the per-node derivative tensor stays bounded.
    """
    from ..physics import PoleGreens, SpectralQuadrature, lm_solve, rowwise_jacobian
    dd = torch.float64
    quad = SpectralQuadrature.for_beta(beta, dtype=dd).to(z0.device)
    fwd = PoleGreens(quad)
    x = x_fit.to(dd)
    K = fwd.kernel_matrix(x)
    Wd = W.to(dd)
    nu_max = flow.nu_max(beta)
    root = math.sqrt(float(lam))
    N, P, _ = z0.shape
    out = torch.empty(N, P, 4, dtype=dd, device=z0.device)
    for i0 in range(0, N, chunk):
        rows = slice(i0, min(i0 + chunk, N))
        zc = z0[rows].detach().to(dd)
        u0, pres = zc[..., :4], zc[..., 4:5]
        n = u0.shape[0]
        nrm, tgt, sig = norm[rows].to(dd), target[rows].to(dd), sigma[rows].to(dd)
        present = zc[..., 4] > 0                                    # exactly decode()'s rule
        top = torch.zeros_like(present).scatter_(-1, zc[..., 4].argmax(-1, keepdim=True), True)
        present = present | (~present.any(-1, keepdim=True) & top)
        keep = present.to(dd).unsqueeze(-1)
        live = keep.expand(-1, -1, 4).reshape(n, -1)
        tether_jac = root * torch.diag_embed(live)

        def fun(uf, jac, fac):
            u = uf.reshape(n, P, 4)
            rt = root * (keep * (u - u0)).reshape(n, -1)
            if not jac:
                with torch.no_grad():
                    ps = flow.coords.decode(torch.cat([u, pres], -1), norm=nrm, nu_max=nu_max)
                    rd = ((fwd(ps, x, kernel_matrix=K) - tgt) / (sig * fac)) @ Wd.T
                return torch.cat([rd, rt], -1), None
            with torch.enable_grad():
                ug = u.detach().requires_grad_(True)
                ps = flow.coords.decode(torch.cat([ug, pres], -1), norm=nrm, nu_max=nu_max)
                th = torch.stack([ps.eps, ps.gamma, ps.a, ps.b], -1).reshape(n, -1)
                Jt = rowwise_jacobian(th, ug)                       # (n, 4P, 4P)
            G, dG = fwd.jacobian(ps.detach(), x, K)
            rd = ((G - tgt) / (sig * fac)) @ Wd.T
            Jd = Wd @ ((dG.reshape(n, -1, 4 * P) @ Jt) / (sig * fac).unsqueeze(-1))
            return torch.cat([rd, rt], -1), torch.cat([Jd, tether_jac], 1)

        u = u0.reshape(n, -1)
        if start is not None:
            st = start[rows].to(dd)
            for k in range(int(stages)):
                fac = st ** (1.0 - k / stages)
                u, _, _ = lm_solve(lambda uf, jac: fun(uf, jac, fac), u,
                                   max_iter=stage_iters, live=live)
        one = torch.ones(n, 1, dtype=dd, device=z0.device)
        u, _, _ = lm_solve(lambda uf, jac: fun(uf, jac, one), u, max_iter=iters, live=live)
        out[rows] = u.reshape(n, P, 4)
    return out


def _hungarian_pair(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Permute each row's label tokens to the noise tokens' optimal matching."""
    from scipy.optimize import linear_sum_assignment
    c = torch.cdist(x0.detach().float().cpu(), x1.detach().float().cpu()).pow(2).numpy()
    perm = np.stack([linear_sum_assignment(ci)[1] for ci in c])
    idx = torch.as_tensor(perm, device=x1.device).unsqueeze(-1).expand_as(x1)
    return x1.gather(1, idx)


def save_flow(model: FlowPosterior, path: str, **meta) -> str:
    torch.save({"format": "flow-1", "config": dict(model.config),
                "state_dict": model.state_dict(), "meta": meta}, path)
    return path


def load_flow(path: str, map_location="cpu") -> FlowPosterior:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if ck.get("format") != "flow-1":
        raise ValueError(f"{path} is not a flow-posterior checkpoint")
    m = FlowPosterior(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.meta = ck.get("meta", {})
    # the band table fitted for this checkpoint, if it ships with one (posterior.band_levels)
    side = str(path) + ".bands.json"
    if os.path.exists(side):
        with open(side) as fh:
            m.meta["band_calibration"] = json.load(fh)
    return m.eval()
