r"""Result figures: the pole set itself, and the latent space behind it.

The spectrum plot every solver produces is deliberately *not* the centrepiece
here.  This package's result is the pole set, and the two figures that carry
information no :math:`A(\omega)` curve shows are:

* the **pole/residue planes** -- poles :math:`z_p = \varepsilon_p - i\gamma_p`
  and residues :math:`r_p = a_p + i b_p` side by side with one colour per pole
  index, so the pairing in :math:`A = -\tfrac{1}{\pi}\sum \mathrm{Im}[r_p /
  (\omega - z_p)]` is visible, and a latent ensemble appears as per-pole
  clouds whose spread *is* the uncertainty of each feature;
* the **latent diagnostics** -- the posterior means over a dataset against the
  prior, plus per-dimension variance against the active-unit threshold.  A
  collapsed latent produces beautifully tight, meaningless ensembles, and this
  is the picture that shows it before it misleads anyone.

Matplotlib is imported inside the functions: plotting stays an optional
dependency of an otherwise headless package.
"""

from __future__ import annotations

import numpy as np
import torch

from .physics.poles import PoleSet

__all__ = ["plot_pole_residue_plane", "plot_latent_space"]

# Okabe-Ito cycle, matching benchmark/plot.py.
_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
            "#56B4E9", "#F0E442", "#2C2C2C"]


def _c(i: int) -> str:
    return _COLOURS[i % len(_COLOURS)]


def plot_pole_residue_plane(
    poles: PoleSet,
    bold: PoleSet | None = None,
    path: str | None = None,
    title: str = "",
    weight_floor: float = 1e-3,
):
    r"""Poles (left) and residues (right) in their complex planes.

    Parameters
    ----------
    poles
        ``(B, P)`` pole sets -- an ensemble (latent samples or per-bin results)
        drawn as translucent per-pole clouds.  Pass physical-unit poles
        (:meth:`PoleSet.to_physical`) unless dimensionless is what you mean.
    bold
        A single ``(1, P)`` set drawn opaque with labels (typically the
        posterior-mean decode).  Defaults to the ensemble mean in each plane.
    weight_floor
        Poles with mean weight below this fraction of the total are drawn
        hollow: present in the parameter count, absent from the physics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = poles.eps.detach().cpu().numpy()
    gam = poles.gamma.detach().cpu().numpy()
    a = poles.a.detach().cpu().numpy()
    b = poles.b.detach().cpu().numpy()
    P = eps.shape[1]
    total = max(float(a.mean(0).sum()), 1e-300)

    if bold is not None:
        be, bg = bold.eps.detach().cpu().numpy()[0], bold.gamma.detach().cpu().numpy()[0]
        ba, bb = bold.a.detach().cpu().numpy()[0], bold.b.detach().cpu().numpy()[0]
    else:
        be, bg = eps.mean(0), gam.mean(0)
        ba, bb = a.mean(0), b.mean(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for p in range(P):
        active = ba[p] > weight_floor * total
        col = _c(p)
        if eps.shape[0] > 1:
            ax1.scatter(eps[:, p], -gam[:, p], s=8, alpha=0.18, color=col,
                        linewidths=0)
            ax2.scatter(a[:, p], b[:, p], s=8, alpha=0.18, color=col,
                        linewidths=0)
        size = 40 + 400 * ba[p] / total
        face = col if active else "none"
        ax1.scatter([be[p]], [-bg[p]], s=size, color=face, edgecolors=col,
                    zorder=3)
        ax2.scatter([ba[p]], [bb[p]], s=size, color=face, edgecolors=col,
                    zorder=3)
        if active:
            ax1.annotate(f"{p + 1}", (be[p], -bg[p]),
                         textcoords="offset points", xytext=(6, 5), fontsize=9,
                         color=col)
            ax2.annotate(f"{p + 1}", (ba[p], bb[p]),
                         textcoords="offset points", xytext=(6, 5), fontsize=9,
                         color=col)

    ax1.axhline(0.0, color="0.75", lw=0.8)
    ax1.set_xlabel(r"$\mathrm{Re}\,z_p = \varepsilon_p$")
    ax1.set_ylabel(r"$\mathrm{Im}\,z_p = -\gamma_p$")
    ax1.set_title("poles (lower half-plane = causal)")
    ax1.set_ylim(top=max(0.05 * gam.max(), 0.0))

    ax2.axhline(0.0, color="0.75", lw=0.8)
    ax2.axvline(0.0, color="0.75", lw=0.8)
    ax2.set_xlabel(r"$\mathrm{Re}\,r_p = a_p$  (weight)")
    ax2.set_ylabel(r"$\mathrm{Im}\,r_p = b_p$  (asymmetry)")
    ax2.set_title(r"residues ($\sum a_p = N$, $\sum b_p = 0$)")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    return fig


def plot_latent_space(
    mu: torch.Tensor,
    path: str | None = None,
    threshold: float = 1e-2,
    title: str = "",
    color_by: np.ndarray | None = None,
):
    r"""Posterior means against the prior, and per-dimension activity.

    Left: :math:`\mu(x)` over the dataset, projected onto its top two
    principal components when the latent has more than two dimensions (an
    isotropic :math:`\mathcal{N}(0, I)` prior projects to an isotropic unit
    Gaussian under any orthonormal projection, so the 1:math:`\sigma`/2:math:`\sigma`
    prior rings stay meaningful).  Right: :math:`\mathrm{Var}_x[\mu_u]` per
    dimension against the active-unit threshold -- the collapse diagnostic.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = mu.detach().cpu().numpy()
    D = m.shape[1]
    var = m.var(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    if D >= 2:
        centred = m - m.mean(0, keepdims=True)
        # PCA via SVD; explained variance labels the axes honestly.
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        proj = centred @ vt[:2].T + m.mean(0, keepdims=True) @ vt[:2].T
        ev = (s**2) / max((s**2).sum(), 1e-300)
        xlab = f"PC1 ({ev[0]:.0%} var)" if D > 2 else r"$\mu_1$"
        ylab = f"PC2 ({ev[1]:.0%} var)" if D > 2 else r"$\mu_2$"
        sc = ax1.scatter(proj[:, 0], proj[:, 1], s=10, alpha=0.5,
                         c=(color_by if color_by is not None else _COLOURS[0]),
                         cmap="viridis" if color_by is not None else None,
                         linewidths=0)
        if color_by is not None:
            fig.colorbar(sc, ax=ax1, shrink=0.85)
        for r_sig, ls in ((1.0, "-"), (2.0, "--")):
            t = np.linspace(0, 2 * np.pi, 200)
            ax1.plot(r_sig * np.cos(t), r_sig * np.sin(t), ls, color="0.6",
                     lw=0.9)
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)
        ax1.set_title(r"posterior means vs the $\mathcal{N}(0,I)$ prior")
        ax1.set_aspect("equal", adjustable="datalim")
    else:
        ax1.hist(m[:, 0], bins=40, color=_COLOURS[0], alpha=0.8)
        ax1.set_xlabel(r"$\mu_1$")
        ax1.set_title("posterior means")

    active = var > threshold
    ax2.bar(np.arange(D), var,
            color=[_COLOURS[0] if act else "0.8" for act in active])
    ax2.axhline(threshold, color=_COLOURS[1], lw=1.0,
                label=f"active-unit threshold ({threshold:g})")
    ax2.set_yscale("log")
    ax2.set_xlabel("latent dimension")
    ax2.set_ylabel(r"$\mathrm{Var}_x[\mu_u]$")
    ax2.set_title(f"active units: {int(active.sum())}/{D}")
    ax2.legend(frameon=False, fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    return fig
