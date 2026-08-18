r"""Synthetic spectral functions: a deliberately broad training prior.

Why breadth is the design goal
------------------------------
A supervised analytic-continuation network inherits the prior of whatever it was
trained on.  This is not a subtlety -- it is *the* documented failure mode of the
approach (Rigo & Mitchell, "Machine learning of Kondo physics...", *PRB* **104**,
235111 (2021); and explicitly Sim et al., "Training biases in machine learning
for the analytic continuation of quantum many-body Green's functions",
*PRR* **4**, 043082 (2022)).  A network trained on single Gaussians will return
a single Gaussian for a gapped superconductor, and its :math:`\chi^2` will look
excellent while doing it, because the kernel washes out precisely the structure
that distinguishes them.

The response here is a library of qualitatively different families -- smooth and
sharp, gapped and gapless, symmetric and not -- sampled together, plus wide
ranges on every parameter.  A family the user's physics needs but the prior
lacks is a *silent* failure, so the families are enumerated explicitly and are
easy to extend: add a function, register it, retrain.

Resolution floor
----------------
Every spectrum is broadened so that no feature is narrower than the resolution
limit :math:`\tilde\gamma_{\min} \sim 0.1\pi` set by the first Matsubara
frequency.  Two independent reasons, which happily agree:

* **Physics.** Structure finer than :math:`\pi/\beta` is not recoverable from
  data on :math:`\tau \in [0,\beta]`; training a network to reproduce it teaches
  it to hallucinate.
* **Numerics.** The forward quadrature integrates a feature of width
  :math:`\gamma` with error :math:`2e^{-2\pi\gamma/h}`, so an unbroadened band
  edge would make the target :math:`(A, G)` pairs mutually inconsistent -- the
  network would be fit against a :math:`G` that its own :math:`A` does not
  produce.

Units
-----
Families are parameterised in **physical** :math:`\omega` because that is where
the physics ranges are meaningful (a Hubbard gap is :math:`U`, not
:math:`\beta U`), then evaluated on the dimensionless quadrature nodes with
:math:`\nu = \beta\omega`.  Weights are dimensionless and unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..physics.forward import SpectralQuadrature

__all__ = [
    "SpectrumBatch",
    "FAMILIES",
    "sample_spectra",
    "broaden",
]


@dataclass
class SpectrumBatch:
    r"""A batch of spectral functions sampled on quadrature nodes.

    Attributes
    ----------
    A
        ``(B, M)`` dimensionless :math:`\tilde A(\nu)`, normalised so that
        ``quad.integrate(A) == 1``.
    family
        Per-sample family name, for stratified reporting.
    beta
        Inverse temperature the batch was generated at.
    poles
        Exact pole labels ``(eps, gamma, a)`` in dimensionless units when the
        family is a Lorentzian mixture, else ``None``.  Parameter-space
        supervision is only meaningful when the target genuinely *is* a pole
        expansion; against a Gaussian target the "true" poles do not exist and
        fitting to them would be fitting to an arbitrary choice.
    """

    A: torch.Tensor
    family: list[str]
    beta: float
    poles: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _u(rng, shape, lo, hi, device, dtype):
    return lo + (hi - lo) * torch.rand(shape, generator=rng, device=device, dtype=dtype)


def _logu(rng, shape, lo, hi, device, dtype):
    return torch.exp(_u(rng, shape, math.log(lo), math.log(hi), device, dtype))


def _dirichlet(rng, B, n, device, dtype, alpha=1.0):
    """Dirichlet weights via normalised Gamma draws (torch.Generator-friendly)."""
    e = -torch.log(torch.rand(B, n, generator=rng, device=device, dtype=dtype).clamp_min(1e-12))
    if alpha != 1.0:
        e = e.pow(1.0 / alpha)
    return e / e.sum(-1, keepdim=True)


def _gauss(nu, mu, sigma):
    """(B, C) params, (M,) nodes -> (B, M) normalised Gaussian mixture component sum."""
    z = (nu.reshape(1, 1, -1) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    return torch.exp(-0.5 * z * z) / (sigma.unsqueeze(-1) * math.sqrt(2.0 * math.pi))


def _lorentz(nu, mu, gamma):
    d = nu.reshape(1, 1, -1) - mu.unsqueeze(-1)
    g = gamma.unsqueeze(-1)
    return (g / math.pi) / (d * d + g * g)


def broaden(A_core: torch.Tensor, h: float, sigma: float) -> torch.Tensor:
    r"""Gaussian-broaden a spectrum sampled on the uniform core grid.

    FFT convolution on the uniform panel.  ``sigma`` is in the same
    (dimensionless) units as ``h``.  Used to enforce the resolution floor on
    families with genuinely sharp features -- band edges, BCS coherence peaks,
    boxes -- which would otherwise be integrated to only first order.
    """
    if sigma <= 0:
        return A_core
    M = A_core.shape[-1]
    n = 2 * M                                            # zero-pad: no wraparound
    k = 2.0 * math.pi * torch.fft.rfftfreq(n, d=h, device=A_core.device,
                                           dtype=A_core.dtype)
    ker = torch.exp(-0.5 * (sigma * k) ** 2)
    pad = torch.nn.functional.pad(A_core, (0, n - M))
    # dim=-1 on both transforms: without it torch infers the axis and emits a
    # resize warning on every call, which buries a training log in noise.
    conv = torch.fft.irfft(torch.fft.rfft(pad, dim=-1) * ker, n=n, dim=-1)
    return conv[..., :M]


# --------------------------------------------------------------------------- #
# families -- each returns (B, M) unnormalised A on the *dimensionless* nodes
# --------------------------------------------------------------------------- #

def _f_gauss_mixture(rng, B, nu, beta, W, dev, dt, crng=None, **kw):
    """One to four Gaussian peaks: the generic smooth multi-peak spectrum."""
    C = int(torch.randint(1, 5, (1,), generator=crng).item())
    mu = _u(rng, (B, C), -1.0, 1.0, dev, dt) * W.unsqueeze(-1)
    sig = _logu(rng, (B, C), 0.05, 0.6, dev, dt) * W.unsqueeze(-1)
    w = _dirichlet(rng, B, C, dev, dt, alpha=0.8)
    return (w.unsqueeze(-1) * _gauss(nu, mu * beta, sig * beta)).sum(1) * beta, None


def _f_lorentz_mixture(rng, B, nu, beta, W, dev, dt, crng=None,
                       gamma_min=0.1 * math.pi, **kw):
    """Lorentzian mixture -- the only family with exact pole labels."""
    C = int(torch.randint(1, 5, (1,), generator=crng).item())
    mu = _u(rng, (B, C), -1.0, 1.0, dev, dt) * W.unsqueeze(-1)
    gam = _logu(rng, (B, C), 0.03, 0.5, dev, dt) * W.unsqueeze(-1)
    w = _dirichlet(rng, B, C, dev, dt, alpha=0.8)
    eps_d = mu * beta
    gam_d = (gam * beta).clamp_min(gamma_min)
    A = (w.unsqueeze(-1) * _lorentz(nu, eps_d, gam_d)).sum(1)
    return A, (eps_d, gam_d, w)


def _f_gapped(rng, B, nu, beta, W, dev, dt, crng=None, **kw):
    """Two lobes with a hard gap: Mott insulator / band insulator shape."""
    gap = _u(rng, (B, 1), 0.1, 0.6, dev, dt) * W.unsqueeze(-1)
    off = _u(rng, (B, 1), 0.15, 0.45, dev, dt) * W.unsqueeze(-1)
    sig = _logu(rng, (B, 1), 0.05, 0.25, dev, dt) * W.unsqueeze(-1)
    mu = torch.cat([-(gap + off), gap + off], dim=1)
    sig = torch.cat([sig, sig], dim=1)
    # Asymmetric weights so the family is not accidentally particle-hole
    # symmetric: half-filling is a special case, not the generic one.
    w0 = _u(rng, (B, 1), 0.25, 0.75, dev, dt)
    w = torch.cat([w0, 1.0 - w0], dim=1)
    return (w.unsqueeze(-1) * _gauss(nu, mu * beta, sig * beta)).sum(1) * beta, None


def _f_metal_hubbard(rng, B, nu, beta, W, dev, dt, crng=None, **kw):
    """Quasiparticle peak plus upper/lower Hubbard bands: the correlated metal."""
    U = _u(rng, (B, 1), 0.3, 1.0, dev, dt) * W.unsqueeze(-1)
    qp_pos = _u(rng, (B, 1), -0.08, 0.08, dev, dt) * W.unsqueeze(-1)
    qp_w = _logu(rng, (B, 1), 0.02, 0.12, dev, dt) * W.unsqueeze(-1)
    hb_w = _logu(rng, (B, 1), 0.10, 0.35, dev, dt) * W.unsqueeze(-1)
    z = _u(rng, (B, 1), 0.05, 0.7, dev, dt)               # quasiparticle weight
    asym = _u(rng, (B, 1), 0.3, 0.7, dev, dt)
    mu = torch.cat([qp_pos, -0.5 * U, 0.5 * U], dim=1)
    sig = torch.cat([qp_w, hb_w, hb_w], dim=1)
    w = torch.cat([z, (1 - z) * asym, (1 - z) * (1 - asym)], dim=1)
    return (w.unsqueeze(-1) * _gauss(nu, mu * beta, sig * beta)).sum(1) * beta, None


def _f_phonon(rng, B, nu, beta, W, dev, dt, crng=None, **kw):
    """Broad continuum with narrow satellites at multiples of a phonon frequency."""
    Om = _u(rng, (B, 1), 0.05, 0.3, dev, dt) * W.unsqueeze(-1)
    cont_w = _logu(rng, (B, 1), 0.2, 0.6, dev, dt) * W.unsqueeze(-1)
    sat_w = _logu(rng, (B, 1), 0.02, 0.08, dev, dt) * W.unsqueeze(-1)
    mus = [torch.zeros_like(Om)]
    sigs = [cont_w]
    for k in (1, 2, 3):
        mus += [k * Om, -k * Om]
        sigs += [sat_w, sat_w]
    mu = torch.cat(mus, dim=1)
    sig = torch.cat(sigs, dim=1)
    w = _dirichlet(rng, B, mu.shape[1], dev, dt, alpha=0.6)
    w = w + 0.3 * torch.nn.functional.one_hot(
        torch.zeros(B, dtype=torch.long, device=dev), mu.shape[1]).to(dt)
    w = w / w.sum(-1, keepdim=True)
    return (w.unsqueeze(-1) * _gauss(nu, mu * beta, sig * beta)).sum(1) * beta, None


def _f_semicircle(rng, B, nu, beta, W, dev, dt, crng=None, **kw):
    """Bethe-lattice DOS: square-root band edges, the classic smooth-with-edges case."""
    D = _u(rng, (B, 1), 0.4, 1.0, dev, dt) * W.unsqueeze(-1) * beta
    shift = _u(rng, (B, 1), -0.15, 0.15, dev, dt) * W.unsqueeze(-1) * beta
    d = nu.reshape(1, -1) - shift
    A = (2.0 / (math.pi * D**2)) * torch.sqrt((D**2 - d**2).clamp_min(0.0))
    return A, None


def _f_bcs(rng, B, nu, beta, W, dev, dt, crng=None, **kw):
    """BCS density of states: a hard gap with divergent coherence peaks."""
    D = _u(rng, (B, 1), 0.05, 0.35, dev, dt) * W.unsqueeze(-1) * beta
    Wc = _u(rng, (B, 1), 0.6, 1.0, dev, dt) * W.unsqueeze(-1) * beta
    d = nu.reshape(1, -1).abs()
    inside = (d * d - D * D).clamp_min(1e-9)
    A = torch.where((d > D) & (d < Wc), d / torch.sqrt(inside), torch.zeros_like(d))
    return A, None


def _f_box(rng, B, nu, beta, W, dev, dt, crng=None, **kw):
    """Flat band of finite width: maximally featureless, a useful negative control."""
    lo = _u(rng, (B, 1), -1.0, -0.1, dev, dt) * W.unsqueeze(-1) * beta
    hi = _u(rng, (B, 1), 0.1, 1.0, dev, dt) * W.unsqueeze(-1) * beta
    d = nu.reshape(1, -1)
    return ((d > lo) & (d < hi)).to(dt), None


def _f_random_smooth(rng, B, nu, beta, W, dev, dt, crng=None, n_modes=6, **kw):
    """exp of a band-limited random field: positive by construction, no family bias.

    The point of this one is to cover shapes nobody thought to enumerate.  A
    prior made only of named physical shapes is still a narrow prior.
    """
    scale = (W.unsqueeze(-1) * beta)
    k = _logu(rng, (B, n_modes), 0.3, 3.0, dev, dt) / scale
    ph = _u(rng, (B, n_modes), 0.0, 2 * math.pi, dev, dt)
    amp = _u(rng, (B, n_modes), 0.0, 1.2, dev, dt)
    d = nu.reshape(1, 1, -1)
    field = (amp.unsqueeze(-1) * torch.sin(k.unsqueeze(-1) * d + ph.unsqueeze(-1))).sum(1)
    envelope = torch.exp(-0.5 * (nu.reshape(1, -1) / scale) ** 2)
    return torch.exp(field.clamp(-8.0, 8.0)) * envelope, None


#: Registry.  ``needs_broadening`` marks families with genuinely sharp features
#: that must be smoothed to the resolution floor before they are integrable.
FAMILIES: dict[str, tuple] = {
    "gauss_mixture": (_f_gauss_mixture, False),
    "lorentz_mixture": (_f_lorentz_mixture, False),
    "gapped": (_f_gapped, False),
    "metal_hubbard": (_f_metal_hubbard, False),
    "phonon": (_f_phonon, False),
    "semicircle": (_f_semicircle, True),
    "bcs": (_f_bcs, True),
    "box": (_f_box, True),
    "random_smooth": (_f_random_smooth, False),
}


def sample_spectra(
    quad: SpectralQuadrature,
    beta: float,
    batch_size: int,
    rng: torch.Generator,
    families: list[str] | None = None,
    omega_bandwidth: tuple[float, float] = (1.0, 12.0),
    gamma_min: float = 0.1 * math.pi,
    broadening: float | None = None,
    device=None,
    dtype=None,
    crng: torch.Generator | None = None,
) -> SpectrumBatch:
    r"""Draw a batch of spectra, one family per batch, normalised to unit weight.

    Parameters
    ----------
    quad
        Quadrature whose nodes the spectrum is sampled on; also fixes the
        core spacing used for broadening.
    beta
        Inverse temperature; sets :math:`\nu = \beta\omega`.
    families
        Subset of :data:`FAMILIES` to draw from.  One family is chosen per call
        so that a batch is homogeneous, which keeps stratified metrics simple.
    omega_bandwidth
        Range for the per-sample physical bandwidth :math:`W`.  Sampling this
        widely is what stops the model from learning "the band edge is always
        at 4".
    broadening
        Gaussian broadening applied to sharp families, in dimensionless units.
        Defaults to ``gamma_min``, i.e. exactly the resolution floor.

    Returns
    -------
    SpectrumBatch
    """
    device = device or quad.nodes.device
    dtype = dtype or quad.nodes.dtype
    # Discrete choices are drawn on the CPU: on an accelerator each one costs a
    # device synchronisation, and there are several per batch.
    crng = crng if crng is not None else torch.Generator().manual_seed(
        int(torch.randint(2**31 - 1, (1,), generator=rng, device=device).item()))
    names = families or list(FAMILIES)
    idx = int(torch.randint(len(names), (1,), generator=crng).item())
    name = names[idx]
    fn, sharp = FAMILIES[name]

    nu = quad.nodes.to(device=device, dtype=dtype)
    W = _u(rng, (batch_size,), *omega_bandwidth, device, dtype)
    A, poles = fn(rng, batch_size, nu, float(beta), W, device, dtype,
                  crng=crng, gamma_min=gamma_min)

    if sharp:
        sigma = gamma_min if broadening is None else float(broadening)
        core = broaden(A[:, : quad.n_core].contiguous(), quad.h, sigma)
        A = torch.cat([core.clamp_min(0.0), A[:, quad.n_core:]], dim=-1)

    A = A.clamp_min(0.0)
    total = quad.integrate(A).clamp_min(1e-30)
    A = A / total.reshape(-1, 1)
    if poles is not None:
        eps_d, gam_d, w = poles
        poles = (eps_d, gam_d, w / total.reshape(-1, 1))

    return SpectrumBatch(A=A, family=[name] * batch_size, beta=float(beta),
                         poles=poles, meta={"bandwidth": W})
