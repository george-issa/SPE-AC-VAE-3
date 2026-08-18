r"""What is achievable, as distinct from what was achieved.

A trained model's :math:`\chi^2` is only interpretable against the best value the
*representation* can reach on the same data.  Those are different numbers, and
on this package's default ansatz they are very different: a positive mixture of
Lorentzians has :math:`c_2 = \tfrac1\pi\sum_p a_p\gamma_p \ge \gamma_{\min}/\pi`,
so its :math:`1/\omega^2` tail cannot be made small, and the resulting misplaced
spectral weight -- about :math:`0.2/(\beta W)`, or :math:`3\times10^{-3}` at
:math:`\beta=10`, :math:`W=6` -- sits two orders of magnitude above the DQMC
bin-mean noise.

:func:`ansatz_chi2_floor` measures that floor directly: it throws away the
network and optimises the pole parameters themselves against the same whitened
:math:`\chi^2`.  Whatever it returns is the best any model with this ansatz could
do on this dataset.  Reporting a training :math:`\chi^2` without it invites two
wrong conclusions -- that the model is underfit when it is at the ansatz limit,
or that :math:`\chi^2 \to 1` is a reachable goal when it is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .data.real import MeasuredGreens
from .models.constraints import DEFAULT_GAMMA_MIN, PoleConstraints
from .physics.forward import PoleGreens, SpectralQuadrature

__all__ = ["AnsatzFloor", "ansatz_chi2_floor", "tail_weight_bound"]


def tail_weight_bound(gamma_min: float, beta: float, bandwidth: float) -> float:
    r"""Spectral weight a real-residue expansion *must* place beyond the band.

    .. math::
        \int_{|\omega| > W} A \, d\omega \;\simeq\; \frac{2c_2}{W}
        \;\ge\; \frac{2\gamma_{\min}^{\rm dimensionless}}{\pi\,\beta\,W}

    A hard lower bound, not an estimate: it follows from :math:`a_p \ge 0`,
    :math:`\sum_p a_p = 1` and :math:`\gamma_p \ge \gamma_{\min}` alone.  Compare
    it against the data's relative noise to see whether the ansatz can reach the
    noise floor at all.
    """
    return 2.0 * gamma_min / (math.pi * beta * bandwidth)


@dataclass
class AnsatzFloor:
    chi2: float
    chi2_target: float
    num_poles: int
    residue_mode: str
    gamma_min: float
    steps: int

    @property
    def ratio(self) -> float:
        return self.chi2 / self.chi2_target if self.chi2_target > 0 else float("nan")

    @property
    def reachable(self) -> bool:
        """Whether the noise floor is attainable by this ansatz at all."""
        return self.ratio < 2.0

    def summary(self) -> str:
        return (f"ansatz floor ({self.residue_mode} residues, P={self.num_poles}, "
                f"{self.steps} steps)\n"
                f"  best chi2 : {self.chi2:.4g}   target {self.chi2_target:.4g}   "
                f"ratio {self.ratio:.4g}\n"
                f"  verdict   : noise floor "
                f"{'reachable' if self.reachable else 'NOT reachable with this ansatz'}")


@torch.enable_grad()
def ansatz_chi2_floor(
    mg: MeasuredGreens,
    num_poles: int = 20,
    residue_mode: str = "real",
    gamma_min: float = DEFAULT_GAMMA_MIN,
    omega_c: float = 20.0,
    steps: int = 3000,
    lr: float = 0.03,
    min_eigenvalue: float = 0.0,
    quad_nodes_per_width: float = 3.5,
    seed: int = 0,
) -> AnsatzFloor:
    r"""Best :math:`\chi^2` the pole ansatz can reach on ``mg``, network aside.

    Optimises :math:`(\varepsilon_p, \gamma_p, a_p[, b_p])` directly through the
    same constraint layer and the same forward map the model uses, so the answer
    is the representation's limit under exactly the constraints the model
    obeys -- not a looser bound from an unconstrained fit.

    Cheap (seconds), and worth running once per dataset before reading any
    training curve.
    """
    torch.manual_seed(seed)
    dtype = torch.float64
    beta = float(mg.beta)

    quad = SpectralQuadrature(nu_c=beta * omega_c,
                              h=gamma_min / quad_nodes_per_width, dtype=dtype)
    fwd = PoleGreens(quad)
    x = mg.x.to(dtype)
    K = fwd.kernel_matrix(x)

    cov = mg.covariance(min_eigenvalue)
    n = cov["n_independent"]
    W = cov["inv_sqrt_R"].to(dtype)
    sigma = (cov["sigma"] / math.sqrt(mg.n_bins)).to(dtype)
    G_obs = mg.mean.to(dtype).unsqueeze(0)

    con = PoleConstraints(residue_mode=residue_mode, gamma_min=gamma_min,
                          nu_max=0.8 * beta * omega_c)
    n_heads = 4 if residue_mode == "complex" else 3
    u = [torch.zeros(1, num_poles, dtype=dtype, requires_grad=True)
         for _ in range(n_heads)]
    with torch.no_grad():
        # Spread the starting positions: identical poles receive identical
        # gradients and never separate.
        u[0].copy_(torch.linspace(-3, 3, num_poles, dtype=dtype).unsqueeze(0))
    opt = torch.optim.Adam(u, lr=lr)

    best = float("inf")
    for _ in range(steps):
        poles = con(u[0], u[1], u[2], u[3] if n_heads == 4 else None)
        G = fwd(poles, x, kernel_matrix=K)
        d = ((G - G_obs)[:, :n] / sigma) @ W.transpose(-1, -2)
        chi2 = (d * d).sum(-1).mean() / n
        best = min(best, float(chi2))
        opt.zero_grad(set_to_none=True)
        chi2.backward()
        opt.step()

    return AnsatzFloor(chi2=best, chi2_target=cov["chi2_target"],
                       num_poles=num_poles, residue_mode=residue_mode,
                       gamma_min=gamma_min, steps=steps)
