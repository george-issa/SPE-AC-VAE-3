"""Reconstruction gallery: s1_metric vs M1-60k across nine test spectra.

Rows of truths spanning both worlds -- four pole-closed probes (isolated
pair, asym triplet, fano band, phonon-like ladder) and the five smooth
families -- each reconstructed by both 60k models at two stages: the
AMORTIZED forward pass (encoder posterior mean -> decoder, the ~2 ms
inference output) and z-descent from it (3000 steps).  One panel per
truth; amortized dotted, z-descent solid, W1 (physical) in the legend.

Usage: recon_gallery.py <noise>   (e.g. 0.001)
Artifacts: out/step2/recon_gallery_nz{noise}.{json,png}
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spevae import load_model
from spevae.data.label_bank import GAMMA_MIN, SmoothTruth
from spevae.data.noise import NoiseModel, sample_correlated_noise
from spevae.losses import spectral_w1
from spevae.physics import PoleGreens, SpectralQuadrature
from spevae.physics.poles import PoleSet
from spevae.refine import z_descend

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                     "axes.labelsize": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})

OUT = Path("out/step2")
BETA = 8.0
NTAU = 64
NZ = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
STEPS = 3000
MODELS = {"s1_metric": "out/step1/s1_metric.pt",
          "M1_60k": "out/step2/map_M1_60k.pt"}
COLORS = {"s1_metric": "#4269d0", "M1_60k": "#3ca951"}

quad = SpectralQuadrature.for_beta(BETA)
x = torch.linspace(0.0, 1.0, NTAU + 1)
K = PoleGreens(quad).kernel_matrix(x)


def pole_probe(eps, gam, a, t):
    a_, t_ = np.array(a, float), np.array(t, float)
    b = a_ * (t_ - (a_ * t_).sum() / a_.sum())
    ps = PoleSet(eps=torch.tensor([eps]) * BETA,
                 gamma=torch.tensor([gam]) * BETA,
                 a=torch.tensor([a]).float(),
                 b=torch.tensor([b.tolist()]).float())
    return ps.spectral(quad.nodes), PoleGreens(quad)(ps, x)


def smooth_probe(family, params):
    tr = SmoothTruth(family, BETA, params)
    A = (torch.tensor(tr.A(quad.nodes.numpy() / BETA)) / BETA).float()
    return A.unsqueeze(0), ((A * quad.weights) @ K).unsqueeze(0)


ks = np.linspace(-1, 1, 5)
TRUTHS = {
    "lorentz pair": pole_probe([-1.8, 1.4], [0.4, 0.6], [0.55, 0.45],
                               [0.2, -0.2]),
    "asym triplet": pole_probe([0.4, 1.1, 2.6], [0.35, 0.25, 0.45],
                               [0.5, 0.3, 0.2], [0.35, -0.4, -0.25]),
    "fano band": pole_probe((0.3 + ks * 1.2).tolist(), [0.5] * 5, [0.2] * 5,
                            np.clip(0.7 * ks, -0.45, 0.45).tolist()),
    "ladder": pole_probe([0.5, 1.6, 2.7, 3.8], [0.35] * 4,
                         [0.45, 0.28, 0.17, 0.10], [0.0] * 4),
    "gauss pair": smooth_probe("gauss_mixture",
                               {"weights": [0.6, 0.4], "mus": [-1.2, 1.8],
                                "sigmas": [0.5, 0.9], "W": 4.0}),
    "gapped pair": smooth_probe("gapped_pair",
                                {"weights": [0.45, 0.55], "mus": [-2.2, 2.2],
                                 "sigmas": [0.4, 0.4], "W": 4.0}),
    "semicircle": smooth_probe("semicircle",
                               {"centre": 0.3, "radius": 4.0,
                                "sigma_res": GAMMA_MIN / BETA, "W": 4.0}),
    "box": smooth_probe("box", {"lo": -3.0, "hi": 4.0,
                                "sigma_res": GAMMA_MIN / BETA, "W": 4.0}),
    "pseudogap": smooth_probe("pseudogap",
                              {"weights": [1.2062, -0.2062], "mus": [0.1, 0.1],
                               "sigmas": [2.0, 0.4851], "W": 4.0}),
}

models = {tag: load_model(p) for tag, p in MODELS.items()}
report, recons = {}, {}
for name, (A_true, G_clean) in TRUTHS.items():
    g = torch.Generator().manual_seed(7)
    n, sigma, R = sample_correlated_noise(G_clean, x, NoiseModel(scale=NZ,
                                                                 xi=0.1), g)
    ev, V = torch.linalg.eigh(R)
    isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    report[name] = {}
    for tag, model in models.items():
        # the ~2 ms output: encoder posterior mean -> decoder, no descent
        with torch.no_grad():
            mu, _, norm = model.encode(G_clean + n, x, BETA, sigma=sigma)
            A_am = model.decode(mu, norm, BETA).spectral(quad.nodes)
        zr = z_descend(model, G_clean + n, x, BETA, sigma=sigma,
                       inv_sqrt_R=isr, steps=STEPS)
        A = zr.poles.to_dimensionless(BETA).spectral(quad.nodes)
        report[name][tag] = {
            "chi2_amort": float(zr.chi2_init),
            "w1_amort": float(spectral_w1(A_am, A_true, quad, beta=BETA)),
            "chi2_zd": float(zr.chi2),
            "w1_zd": float(spectral_w1(A, A_true, quad, beta=BETA))}
        recons[(name, tag, "amort")] = A_am[0].detach()
        recons[(name, tag, "zd")] = A[0].detach()
    r = report[name]
    print(f"{name:13s}: " + "   ".join(
        f"{t}: amort {r[t]['chi2_amort']:.3g}/{r[t]['w1_amort']:.4f} "
        f"zd {r[t]['chi2_zd']:.3g}/{r[t]['w1_zd']:.4f}"
        for t in MODELS), flush=True)

stem = f"recon_gallery_nz{NZ:g}"
with open(OUT / f"{stem}.json", "w") as fh:
    json.dump(report, fh, indent=1)

fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.2), constrained_layout=True)
wgrid = (quad.nodes / BETA).numpy()
core = np.abs(wgrid) < 8.0
for ax, name in zip(axes.ravel(), TRUTHS):
    A_true = TRUTHS[name][0][0].numpy()
    ax.plot(wgrid[core], BETA * A_true[core], color="0.15", lw=1.7,
            label="truth")
    for tag in MODELS:
        r = report[name][tag]
        ax.plot(wgrid[core], BETA * recons[(name, tag, "amort")].numpy()[core],
                color=COLORS[tag], lw=0.9, ls=":", alpha=0.7,
                label=f"{tag} amortized (W1 {r['w1_amort']:.3f})")
        ax.plot(wgrid[core], BETA * recons[(name, tag, "zd")].numpy()[core],
                color=COLORS[tag], lw=1.4, ls="--",
                label=f"{tag} z-descent (W1 {r['w1_zd']:.3f})")
    ax.set_title(name, fontsize=9)
    ax.legend(frameon=False, fontsize=6)
for ax in axes[-1]:
    ax.set_xlabel(r"$\omega$")
fig.suptitle(f"Reconstruction gallery at noise {NZ:g}, beta = 8: "
             f"s1_metric (step-1 map) vs M1-60k (dispersive map); "
             f"amortized ~2 ms output (dotted) + z-descent {STEPS} steps "
             f"(dashed)", fontsize=10)
fig.savefig(OUT / f"{stem}.png", dpi=150)
print(f"-> {OUT}/{stem}.png, .json")
