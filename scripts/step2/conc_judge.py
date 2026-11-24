"""Concentration screen judge: does the network's pole STRUCTURE follow the labels?

For each checkpoint (tag=path on argv):
  count    512 spectra from the M1 training prior (beta 8, dispersive): the
           network's effective pole count vs the label's, binned, correlation,
           mean |gap|; spectral W1 / chi2 on the same spectra (amortized).
  fano     the in-distribution fano probe (5 poles, noise 1e-4, seed 7):
           network pole sets amortized and after z-descent (3000).
Artifacts: out/step2/conc/<tag>_count.json, then conc_fig.py draws the figure.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import torch

from spevae import load_model
from spevae.data.noise import NoiseModel, sample_correlated_noise
from spevae.data.pole_prior import PolePriorGenerator
from spevae.losses import chi2 as chi2_fn, spectral_w1
from spevae.physics.forward import PoleGreens, SpectralQuadrature
from spevae.physics.poles import PoleSet, effective_pole_count
from spevae.refine import z_descend

torch.set_num_threads(3)
OUT = Path("out/step2/conc")
BETA = 8.0
BINS = ((1, 2.5), (2.5, 4.5), (4.5, 7), (7, 13))


def count_part(f):
    gen = PolePriorGenerator(beta_range=(BETA, BETA), residue_mode="dispersive")
    rng = torch.Generator().manual_seed(3)
    nt, ne, w1s, c2s = [], [], [], []
    with gen.fixed_scalars(3):
        batches = [gen.batch(64, rng) for _ in range(8)]
    for b in batches:
        with torch.no_grad():
            out = f(b.G, b.x, b.beta, sigma=b.sigma, deterministic=True)
        A = out.poles.spectral(b.quad.nodes)
        ne += effective_pole_count(out.poles.a).tolist()
        nt += effective_pole_count(b.poles_true["a"]).tolist()
        for i in range(A.shape[0]):
            w1s.append(float(spectral_w1(A[i:i + 1], b.A_true[i:i + 1], b.quad, beta=b.beta)))
        c2s.append(float(chi2_fn(out.G_recon, b.G, b.sigma, inv_sqrt_R=b.inv_sqrt_R)))
    nt, ne = np.array(nt), np.array(ne)
    r = dict(label_neff=nt.tolist(), net_neff=ne.tolist(),
             corr=float(np.corrcoef(nt, ne)[0, 1]), gap=float(np.abs(ne - nt).mean()),
             w1=float(np.mean(w1s)), chi2_geo=float(np.exp(np.mean(np.log(c2s)))), bins={})
    for lo, hi in BINS:
        m = (nt >= lo) & (nt < hi)
        r["bins"][f"{lo}-{hi}"] = dict(n=int(m.sum()), label=float(nt[m].mean()),
                                       net=float(ne[m].mean()), net_sd=float(ne[m].std()))
    return r


def fano_part(f):
    quad = SpectralQuadrature.for_beta(BETA)
    x = torch.linspace(0, 1, 65)
    ks = np.linspace(-1, 1, 5)
    e, a = 0.3 + ks * 1.2, np.full(5, 0.2)
    t = np.clip(0.7 * ks, -0.45, 0.45)
    b = a * (t - (a * t).sum())
    truth = PoleSet(torch.tensor(np.array([e * BETA])).float(), torch.full((1, 5), 0.5 * BETA),
                    torch.tensor(np.array([a])).float(), torch.tensor(np.array([b])).float())
    Gc = PoleGreens(quad)(truth, x)
    n, sig, R = sample_correlated_noise(Gc, x, NoiseModel(scale=1e-4, xi=0.1),
                                        torch.Generator().manual_seed(7))
    G = Gc + n
    ev, V = torch.linalg.eigh(R)
    isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    At = truth.spectral(quad.nodes)
    w = torch.linspace(-4, 4.5, 1200)
    res = {}
    with torch.no_grad():
        p_am = f(G, x, BETA, sigma=sig, deterministic=True).poles.to_physical(BETA)
    p_zd = z_descend(f, G[0], x, BETA, sigma=sig[0], inv_sqrt_R=isr, steps=3000).poles
    for stage, p in (("amortized", p_am), ("z_descent", p_zd)):
        o = torch.argsort(p.eps[0])
        res[stage] = dict(
            neff=float(effective_pole_count(p.a)[0]), n_big=int((p.a[0] > 0.02).sum()),
            w1=float(spectral_w1(p.to_dimensionless(BETA).spectral(quad.nodes), At, quad, beta=BETA)),
            eps=p.eps[0][o].tolist(), gamma=p.gamma[0][o].tolist(), a=p.a[0][o].tolist(),
            A=p.spectral(w)[0].tolist())
    res["omega"] = w.tolist()
    res["A_truth"] = truth.to_physical(BETA).spectral(w)[0].tolist()
    res["label"] = dict(eps=e.tolist(), a=a.tolist())
    return res


for kv in sys.argv[1:]:
    tag, path = kv.split("=", 1)
    f = load_model(path).eval()
    rep = {"ckpt": path, "count": count_part(f), "fano": fano_part(f)}
    c, fz = rep["count"], rep["fano"]
    print(f"[{tag}] count corr {c['corr']:.2f}  mean|gap| {c['gap']:.2f}  stream W1 {c['w1']:.4f}  "
          f"chi2geo {c['chi2_geo']:.3g} | " + "  ".join(
              f"{k}: {v['net']:.1f}" for k, v in c["bins"].items()) +
          f" | fano amort {fz['amortized']['neff']:.1f} poles W1 {fz['amortized']['w1']:.3f}, "
          f"zd {fz['z_descent']['neff']:.1f} poles W1 {fz['z_descent']['w1']:.3f}", flush=True)
    json.dump(rep, open(OUT / f"{tag}_count.json", "w"))
print("CONC_JUDGE_DONE")
