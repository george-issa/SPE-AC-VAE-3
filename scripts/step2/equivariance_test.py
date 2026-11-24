"""Inference-time PH group averaging (arXiv:2609.27132) on the banked checkpoints.

The symmetry: K(beta - tau, -omega) = K(tau, omega), so G(beta - tau) is exactly
the data of A(-omega).  f = the trained model; F = MirrorSymmetrized(f) =
1/2 f(G) U 1/2 M f(PG) -- exactly equivariant, no retraining.  Each input is
paired with its EXACT reflection (same noise realisation, sigma and whitener
reflected), so f(G) vs f(PG) differ only through the model's own
non-equivariance.

Parts (run separately, one process each):
  probes   13 truths (step-1 pole probes + the 9-truth gallery) x noise
           1e-4 / 1e-3: amortized and z-descent (3000 steps) for f on G,
           f on PG, and F; the equivariance gap W1(f(G), M f(PG)).
  streams  frozen pole stream + legacy canary, each also MIRRORED, through
           evaluate() for f and F.
  holstein MAP refine (3k) and the manifold floor (anchor released, 10k
           steps) for f and F -- the step-2 gate.

Usage: equivariance_test.py <tag> <ckpt> <part>
Artifacts: out/step2/equivariance/<tag>_<part>.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch

from spevae import load_model
from spevae.data.label_bank import GAMMA_MIN, SmoothTruth
from spevae.data.noise import NoiseModel, sample_correlated_noise
from spevae.losses import spectral_w1
from spevae.models.equivariant import MirrorSymmetrized, mirror_poles, reflect_tau
from spevae.physics import PoleGreens, SpectralQuadrature
from spevae.physics.poles import PoleSet
from spevae.refine import RefineConfig, refine_latent, z_descend

torch.set_num_threads(2)
TAG, CKPT, PART = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = Path("out/step2/equivariance")
OUT.mkdir(parents=True, exist_ok=True)
BETA, NTAU, STEPS = 8.0, 64, 3000
quad = SpectralQuadrature.for_beta(BETA)
x = torch.linspace(0.0, 1.0, NTAU + 1)
K = PoleGreens(quad).kernel_matrix(x)


def node_mirror(q):
    """Index permutation with nodes[perm] == -nodes (the rule is symmetric but unsorted)."""
    o = torch.argsort(q.nodes)
    perm = torch.empty_like(o)
    perm[o] = o.flip(0)
    assert torch.allclose(q.nodes[perm], -q.nodes, atol=1e-4), "nodes not symmetric"
    assert torch.allclose(q.weights[perm], q.weights, rtol=1e-6), "weights not symmetric"
    return perm


PERM = node_mirror(quad)
f = load_model(CKPT)
F = MirrorSymmetrized(f)


def flipA(A):                      # A(nu) -> A(-nu) on the quadrature nodes
    return A[..., PERM]


def w1(A, At):
    return float(spectral_w1(A, At, quad, beta=BETA))


def A_of(ps_dimless):
    return ps_dimless.spectral(quad.nodes)


# ---------------------------------------------------------------- truths --- #
def pole_probe(eps, gam, a, t):
    a_, t_ = np.array(a, float), np.array(t, float)
    b = a_ * (t_ - (a_ * t_).sum() / a_.sum())
    ps = PoleSet(eps=torch.tensor([eps]) * BETA, gamma=torch.tensor([gam]) * BETA,
                 a=torch.tensor([a]).float(), b=torch.tensor([b.tolist()]).float())
    return ps.spectral(quad.nodes), PoleGreens(quad)(ps, x)


def smooth_probe(family, params):
    tr = SmoothTruth(family, BETA, params)
    A = (torch.tensor(tr.A(quad.nodes.numpy() / BETA)) / BETA).float()
    return A.unsqueeze(0), ((A * quad.weights) @ K).unsqueeze(0)


def ladder_t(e0, Om, m, g, d):
    e = [e0 + k * Om for k in range(m)]
    a = [d ** k for k in range(m)]
    return e, [g] * m, [v / sum(a) for v in a], [0.0] * m


ks = np.linspace(-1, 1, 5)
TRUTHS = {
    # step-1 probe set (judge_arm.py / asym_judge.py)
    "3-rung ladder": pole_probe(*ladder_t(0.5, 1.0, 3, 0.30, 0.55)),
    "4-rung ladder": pole_probe(*ladder_t(-0.4, 0.8, 4, 0.25, 0.60)),
    "edged band": pole_probe((0.3 + ks * 1.4).tolist(), [0.12] + [0.45] * 3 + [0.12],
                             [0.2] * 5, [0.0] * 5),
    "irregular triplet": pole_probe([0.4, 1.1, 2.6], [0.35, 0.25, 0.45],
                                    [0.5, 0.3, 0.2], [0.0] * 3),
    "asym triplet": pole_probe([0.4, 1.1, 2.6], [0.35, 0.25, 0.45],
                               [0.5, 0.3, 0.2], [0.35, -0.4, -0.25]),
    "fano band": pole_probe((0.3 + ks * 1.2).tolist(), [0.5] * 5, [0.2] * 5,
                            np.clip(0.7 * ks, -0.45, 0.45).tolist()),
    # gallery (recon_gallery.py)
    "lorentz pair": pole_probe([-1.8, 1.4], [0.4, 0.6], [0.55, 0.45], [0.2, -0.2]),
    "gauss pair": smooth_probe("gauss_mixture", {"weights": [0.6, 0.4], "mus": [-1.2, 1.8],
                                                 "sigmas": [0.5, 0.9], "W": 4.0}),
    "gapped pair": smooth_probe("gapped_pair", {"weights": [0.45, 0.55], "mus": [-2.2, 2.2],
                                                "sigmas": [0.4, 0.4], "W": 4.0}),
    "semicircle": smooth_probe("semicircle", {"centre": 0.3, "radius": 4.0,
                                              "sigma_res": GAMMA_MIN / BETA, "W": 4.0}),
    "box": smooth_probe("box", {"lo": -3.0, "hi": 4.0,
                                "sigma_res": GAMMA_MIN / BETA, "W": 4.0}),
    "pseudogap": smooth_probe("pseudogap", {"weights": [1.2062, -0.2062], "mus": [0.1, 0.1],
                                            "sigmas": [2.0, 0.4851], "W": 4.0}),
}


def run_probes():
    rep = {}
    for nz in (1e-4, 1e-3):
        for name, (A_true, G_clean) in TRUTHS.items():
            n, sigma, R = sample_correlated_noise(G_clean, x, NoiseModel(scale=nz, xi=0.1),
                                                  torch.Generator().manual_seed(7))
            G = (G_clean + n).reshape(1, -1)
            sigma = sigma.reshape(1, -1)
            ev, V = torch.linalg.eigh(R)
            isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
            Gr, sr, isr_r = reflect_tau(G), reflect_tau(sigma), isr.flip(0).flip(1)
            At, Atr = A_true, flipA(A_true)
            row = {}
            with torch.no_grad():
                Af = A_of(f(G, x, BETA, sigma=sigma, deterministic=True).poles)
                Afr = A_of(f(Gr, x, BETA, sigma=sr, deterministic=True).poles)
                AF = A_of(F(G, x, BETA, sigma=sigma, deterministic=True).poles)
                AFr = A_of(F(Gr, x, BETA, sigma=sr, deterministic=True).poles)
            row["amort"] = dict(w1_f=w1(Af, At), w1_f_mirror=w1(Afr, Atr), w1_F=w1(AF, At),
                                gap_f=w1(Af, flipA(Afr)),
                                F_exactness=float((AFr - flipA(AF)).abs().max()))
            zf = z_descend(f, G[0], x, BETA, sigma=sigma[0], inv_sqrt_R=isr, steps=STEPS)
            zfr = z_descend(f, Gr[0], x, BETA, sigma=sr[0], inv_sqrt_R=isr_r, steps=STEPS)
            zF = z_descend(F, G[0], x, BETA, sigma=sigma[0], inv_sqrt_R=isr, steps=STEPS)
            Azf = A_of(zf.poles.to_dimensionless(BETA))
            Azfr = A_of(zfr.poles.to_dimensionless(BETA))
            AzF = A_of(zF.poles.to_dimensionless(BETA))
            row["zd"] = dict(w1_f=w1(Azf, At), w1_f_mirror=w1(Azfr, Atr), w1_F=w1(AzF, At),
                             chi2_f=zf.chi2, chi2_f_mirror=zfr.chi2, chi2_F=zF.chi2,
                             chi2_init_f=zf.chi2_init, chi2_init_F=zF.chi2_init,
                             gap_f=w1(Azf, flipA(Azfr)),
                             minA_f=float(Azf.min()), minA_F=float(AzF.min()))
            row["A"] = dict(truth=(At[0] * BETA).tolist(),
                            amort_f=(Af[0] * BETA).tolist(),
                            amort_f_mirror_back=(flipA(Afr)[0] * BETA).tolist(),
                            amort_F=(AF[0] * BETA).tolist(),
                            zd_f=(Azf[0] * BETA).tolist(),
                            zd_f_mirror_back=(flipA(Azfr)[0] * BETA).tolist(),
                            zd_F=(AzF[0] * BETA).tolist())
            rep[f"{name}@{nz:g}"] = row
            a, z = row["amort"], row["zd"]
            print(f"[{name:<17}@{nz:g}] amort W1 f {a['w1_f']:.4f} f(PG) {a['w1_f_mirror']:.4f}"
                  f" F {a['w1_F']:.4f} gap {a['gap_f']:.4f} | zd W1 f {z['w1_f']:.4f}"
                  f" f(PG) {z['w1_f_mirror']:.4f} F {z['w1_F']:.4f} gap {z['gap_f']:.4f}"
                  f" | chi2 f {z['chi2_f']:.3g} f(PG) {z['chi2_f_mirror']:.3g}"
                  f" F {z['chi2_F']:.3g}", flush=True)
    rep["nodes"] = (quad.nodes / BETA).tolist()
    return rep


# --------------------------------------------------------------- streams --- #
class Mirrored:
    """Wrap a generator: every batch is replaced by its exact tau-reflection."""

    def __init__(self, gen):
        self.gen = gen

    def fixed_scalars(self, seed):
        return self.gen.fixed_scalars(seed)

    def batch(self, n, rng):
        b = self.gen.batch(n, rng)
        assert torch.allclose(b.x.flip(-1), 1 - b.x, atol=1e-6)
        b.G, b.G_alt, b.G_clean = b.G.flip(-1), b.G_alt.flip(-1), b.G_clean.flip(-1)
        b.sigma = b.sigma.flip(-1)
        b.inv_sqrt_R = b.inv_sqrt_R.flip(0).flip(1)
        b.A_true = b.A_true[..., node_mirror(b.quad)]
        return b


def run_streams():
    from spevae.data.pole_prior import PolePriorGenerator
    from spevae.data.synthetic import SyntheticGenerator
    from spevae.training import evaluate
    rep = {}
    for sname, mk in [("pole", lambda: PolePriorGenerator(beta_range=(8.0, 8.0))),
                      ("legacy", lambda: SyntheticGenerator(beta_range=(8.0, 8.0)))]:
        for orient in ("orig", "mirrored"):
            for mname, m in (("f", f), ("F", F)):
                gen = mk() if orient == "orig" else Mirrored(mk())
                r = evaluate(m, gen, n_batches=16)
                rep[f"{sname}/{orient}/{mname}"] = {k: float(v) for k, v in r.items()}
                print(f"[{sname:6s} {orient:8s} {mname}] W1 {r['w1_omega']:.4f}  "
                      f"l2 {r['l2_rel']:.4f}  chi2geo {r['chi2_geo']:.3g}  "
                      f"ripple {r['ripple']:.4f}", flush=True)
    return rep


# -------------------------------------------------------------- holstein --- #
def run_holstein():
    from spevae.data.readers import read_gissa_cube
    CUBE = ("/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/"
            "gIssa/for_george_ac.jld2")
    mg = read_gissa_cube(CUBE, beta=8.0, omega=1.0, n=1.0, lam=0.325)
    rep = {}
    for mname, m in (("f", f), ("F", F)):
        rr = refine_latent(m, mg, RefineConfig(cov_method="structured",
                                               min_eigenvalue=1e-10, steps=3000))
        rb = refine_latent(m, mg, RefineConfig(cov_method="structured", min_eigenvalue=1e-10,
                                               steps=10000, prior_weight=0.0, patience=3000))
        w = torch.linspace(-8, 8, 1601)
        rep[mname] = dict(chi2_init=rr.chi2_init, chi2_map=rr.chi2, floor=rb.chi2,
                          target=rb.chi2_target, ratio=rb.chi2 / rb.chi2_target,
                          A_map=rr.poles.spectral(w).reshape(-1).tolist(),
                          A_floor=rb.poles.spectral(w).reshape(-1).tolist(),
                          omega=w.tolist())
        print(f"[Holstein {mname}] amort chi2 {rr.chi2_init:.1f}  MAP {rr.chi2:.2f}  "
              f"floor {rb.chi2:.3f} ({rb.chi2 / rb.chi2_target:.1f}x target "
              f"{rb.chi2_target:.3f})", flush=True)
    return rep


rep = {"probes": run_probes, "streams": run_streams, "holstein": run_holstein}[PART]()
rep["ckpt"] = CKPT
json.dump(rep, open(OUT / f"{TAG}_{PART}.json", "w"))
print("EQUIV_DONE")
