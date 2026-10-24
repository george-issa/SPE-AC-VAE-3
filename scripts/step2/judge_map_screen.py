"""Screen judge for the residue-map arm: ctrl vs M0 vs M1 at 6k steps.

Common fixed probes for every model (the arms' own eval streams differ by
construction, so log metrics cannot compare them): two pole-closed step-1
probes (asym triplet, fano band) and one smooth out-of-vocabulary truth
(semicircle), each at two noise levels, with PAIRED noise realisations.

Per (model, probe, noise): z-descent chi2, W1(recon, truth) in physical
units, certified negative weight of the reconstruction (does the widened
channel dip?), and the paired-noise consistency W1 between the two
reconstructions -- the direct measurement of noise exciting the dispersive
channel.  Artifacts: out/step2/map_screen6k.{json,png}.
"""
import json
import math
import os
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
from spevae.physics import PoleGreens, SpectralQuadrature, certify
from spevae.physics.poles import PoleSet
from spevae.refine import z_descend

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                     "axes.labelsize": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})

OUT = Path("out/step2")
BETA = 8.0
NTAU = 64
# default: the 6k screens; override with argv pairs "tag=path" plus an
# optional leading output stem, e.g.
#   judge_map_screen.py map_judge60k s1_metric=out/step1/s1_metric.pt \
#       M1_60k=out/step2/map_M1_60k.pt
STEM = "map_screen6k"
MODELS = {"ctrl": "map_ctrl6k.pt", "M0": "map_M0_6k.pt", "M1": "map_M1_6k.pt"}
if len(sys.argv) > 1:
    STEM = sys.argv[1]
    MODELS = dict(kv.split("=", 1) for kv in sys.argv[2:])
NOISES = (1e-3, 1e-4)


def mk_pole_probe(eps, gam, a, t):
    a_, t_ = np.array(a), np.array(t)
    b = a_ * (t_ - (a_ * t_).sum() / a_.sum())
    ps = PoleSet(eps=torch.tensor([eps]) * BETA, gamma=torch.tensor([gam]) * BETA,
                 a=torch.tensor([a]).double().float(),
                 b=torch.tensor([b.tolist()]).float())
    return ps


quad = SpectralQuadrature.for_beta(BETA)
x = torch.linspace(0.0, 1.0, NTAU + 1)

PROBES = {}
for name, ps in (
    ("asym triplet", mk_pole_probe([0.4, 1.1, 2.6], [0.35, 0.25, 0.45],
                                   [0.5, 0.3, 0.2], [0.35, -0.4, -0.25])),
    ("fano band", mk_pole_probe((0.3 + np.linspace(-1, 1, 5) * 1.2).tolist(),
                                [0.5] * 5, [0.2] * 5,
                                np.clip(0.7 * np.linspace(-1, 1, 5),
                                        -0.45, 0.45).tolist())),
):
    A_true = ps.spectral(quad.nodes)
    G_clean = PoleGreens(quad)(ps, x)
    PROBES[name] = (A_true, G_clean)

semi = SmoothTruth("semicircle", BETA, {"centre": 0.3, "radius": 4.0,
                                        "sigma_res": GAMMA_MIN / BETA, "W": 4.0})
A_semi = (torch.tensor(semi.A(quad.nodes.numpy() / BETA)) / BETA).float()
K = PoleGreens(quad).kernel_matrix(x)
G_semi = (A_semi * quad.weights) @ K
PROBES["semicircle"] = (A_semi.unsqueeze(0), G_semi.unsqueeze(0))

report = {}
recons = {}
for tag, fname in MODELS.items():
    model = load_model(fname if "/" in str(fname) else OUT / fname)
    report[tag] = {}
    for pname, (A_true, G_clean) in PROBES.items():
        for nz in NOISES:
            pair = []
            for k in (0, 1):
                g = torch.Generator().manual_seed(7 + k)
                n, sigma, R = sample_correlated_noise(
                    G_clean, x, NoiseModel(scale=nz, xi=0.1), g)
                ev, V = torch.linalg.eigh(R)
                isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
                zr = z_descend(model, G_clean + n, x, BETA, sigma=sigma,
                               inv_sqrt_R=isr, steps=int(os.environ.get("ZSTEPS", 800)))
                pair.append(zr)
            # z_descend returns PHYSICAL-unit poles; the probes and the
            # quadrature live in dimensionless nu = beta*omega
            A0 = pair[0].poles.to_dimensionless(BETA).spectral(quad.nodes)
            A1 = pair[1].poles.to_dimensionless(BETA).spectral(quad.nodes)
            w1 = float(spectral_w1(A0, A_true, quad, beta=BETA))
            cons = float(spectral_w1(A0, A1, quad, beta=BETA))
            p = pair[0].poles      # physical units; certify is unit-consistent
            rep = certify(p.eps[0].tolist(), p.gamma[0].tolist(),
                          p.a[0].tolist(), p.b[0].tolist(), atol=1e-8)
            negw = 0.0 if rep.is_positive else float(rep.negative_weight)
            bsat = float((p.b.abs() / p.a.clamp_min(1e-9)).max())
            report[tag][f"{pname}@{nz:g}"] = {
                "chi2": float(pair[0].chi2), "w1": w1, "consistency_w1": cons,
                "neg_weight": negw, "max_b_over_a": bsat}
            if nz == NOISES[0]:
                recons[(tag, pname)] = A0[0].detach()
        print(f"{tag:5s} {pname:13s}: " + "  ".join(
            f"nz={nz:g}: chi2={report[tag][f'{pname}@{nz:g}']['chi2']:.3g} "
            f"W1={report[tag][f'{pname}@{nz:g}']['w1']:.4f} "
            f"cons={report[tag][f'{pname}@{nz:g}']['consistency_w1']:.4f} "
            f"negw={report[tag][f'{pname}@{nz:g}']['neg_weight']:.1e}"
            for nz in NOISES), flush=True)

with open(OUT / f"{STEM}.json", "w") as fh:
    json.dump(report, fh, indent=1)

# figure: probes x models overlays at the louder noise
fig, axes = plt.subplots(len(PROBES), len(MODELS),
                         figsize=(3.6 * len(MODELS), 2.4 * len(PROBES)),
                         constrained_layout=True)
wgrid = (quad.nodes / BETA).numpy()
core = np.abs(wgrid) < 8.0
for i, pname in enumerate(PROBES):
    A_true = PROBES[pname][0][0].numpy()
    for j, tag in enumerate(MODELS):
        ax = axes[i, j]
        ax.plot(wgrid[core], BETA * A_true[core], color="0.15", lw=1.5,
                label="truth")
        ax.plot(wgrid[core], BETA * recons[(tag, pname)].numpy()[core],
                color="#4269d0", lw=1.1, ls="--", label="z-descent")
        r = report[tag][f"{pname}@{NOISES[0]:g}"]
        ax.set_title(f"{tag} / {pname}: W1 {r['w1']:.3f}, "
                     f"cons {r['consistency_w1']:.3f}", fontsize=8)
        if i == 0 and j == 0:
            ax.legend(frameon=False, fontsize=7)
        if i == len(PROBES) - 1:
            ax.set_xlabel(r"$\omega$")
fig.suptitle(f"Residue-map screen at 6k steps (noise {NOISES[0]:g}): "
             "ctrl (step-1 map) vs M0 (dispersive decoder) vs M1 (+ widened prior)",
             fontsize=10)
fig.savefig(OUT / f"{STEM}.png", dpi=150)
print(f"-> {OUT}/{STEM}.png, .json")
