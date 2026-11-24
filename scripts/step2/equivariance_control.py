"""Attribution control for the group-averaging gains: symmetry or capacity?

MirrorSymmetrized's reachable set {1/2 dec(z1) U 1/2 M dec(z2)} is ALSO a
two-component mixture (2x latent, 2x poles).  The control is the identical
construction without the mirror, {1/2 dec(z1) U 1/2 dec(z2)} -- same capacity,
same optimiser, same budgets -- so any gap between the two is the symmetry's.
A swap-symmetric start (z1 = z2) would never separate, so the control starts
z2 = mu + s * sd * xi (posterior-scaled jitter, three seeds); for scale, the
mirrored model's own start differs by |mu(G) - mu(PG)|, reported alongside.

Parts:  holstein  MAP (3k) + manifold floor (10k, anchor released), 3 seeds
        probes    z-descent 3000 on the 12 probe truths at 1e-4 and 1e-3, seed 0
Usage: equivariance_control.py <tag> <ckpt> <part>
Artifacts: out/step2/equivariance/<tag>_control_<part>.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch

from spevae import load_model
from spevae.models.equivariant import MirrorSymmetrized, union_half

torch.set_num_threads(2)
TAG, CKPT, PART = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = Path("out/step2/equivariance")


class Mixture2(MirrorSymmetrized):
    """Same two-branch capacity as MirrorSymmetrized, NO mirror (the control)."""

    def __init__(self, base, seed, scale=1.0):
        super().__init__(base)
        self.seed, self.scale = int(seed), float(scale)
        self.start_sep = None

    def encode(self, G, x, beta, sigma=None, mask=None, includes_endpoints=True,
               norm=None):
        mu, logvar, norm = self.base.encode(G, x, beta, sigma=sigma, mask=mask,
                                            includes_endpoints=includes_endpoints,
                                            norm=norm)
        g = torch.Generator().manual_seed(self.seed)
        xi = torch.randn(mu.shape, generator=g).to(mu)
        mu2 = mu + self.scale * torch.exp(0.5 * logvar) * xi
        self.start_sep = float((mu2 - mu).norm())
        return torch.cat([mu, mu2], -1), torch.cat([logvar, logvar], -1), norm

    def decode(self, z, norm, beta):
        Lz, B = self.base.latent_dim, z.shape[0]
        p = self.base.decode(torch.cat([z[:, :Lz], z[:, Lz:]], 0),
                             torch.cat([norm, norm], 0), beta)
        return union_half(p[:B], p[B:])


f = load_model(CKPT)
rep = {"ckpt": CKPT}

if PART == "holstein":
    from spevae.data.readers import read_gissa_cube
    from spevae.models.equivariant import reflect_tau
    from spevae.refine import RefineConfig, refine_latent
    CUBE = ("/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/"
            "gIssa/for_george_ac.jld2")
    mg = read_gissa_cube(CUBE, beta=8.0, omega=1.0, n=1.0, lam=0.325)
    # scale reference: how far apart the mirrored model's two codes start
    with torch.no_grad():
        Fm = MirrorSymmetrized(f)
        G = mg.mean.unsqueeze(0).to(torch.float32)
        s = mg.sigma_encoder.unsqueeze(0).to(torch.float32) / mg.n_bins ** 0.5
        mu, _, _ = Fm.encode(G, mg.x.to(torch.float32), mg.beta, sigma=s)
        Lz = f.latent_dim
        rep["mirror_start_sep"] = float((mu[:, :Lz] - mu[:, Lz:]).norm())
    print(f"mirrored model start separation |mu(G)-mu(PG)| = {rep['mirror_start_sep']:.3f}")
    for seed in (0, 1, 2):
        m = Mixture2(f, seed)
        rr = refine_latent(m, mg, RefineConfig(cov_method="structured",
                                               min_eigenvalue=1e-10, steps=3000))
        rb = refine_latent(m, mg, RefineConfig(cov_method="structured", min_eigenvalue=1e-10,
                                               steps=10000, prior_weight=0.0, patience=3000))
        rep[f"seed{seed}"] = dict(start_sep=m.start_sep, chi2_init=rr.chi2_init,
                                  chi2_map=rr.chi2, floor=rb.chi2, target=rb.chi2_target)
        print(f"[Holstein mixture2 seed {seed}] start sep {m.start_sep:.3f}  amort chi2 "
              f"{rr.chi2_init:.1f}  MAP {rr.chi2:.2f}  floor {rb.chi2:.3f} "
              f"({rb.chi2 / rb.chi2_target:.1f}x)", flush=True)

elif PART == "probes":
    sys.argv = [sys.argv[0], TAG, CKPT, "none"]
    # reuse the exact truths/noise from the main experiment without running it
    src = Path("scripts/step2/equivariance_test.py").read_text().split("def run_probes")[0]
    ns = {"__name__": "eqt_defs"}
    exec(compile(src, "equivariance_test_defs", "exec"), ns)
    from spevae.data.noise import NoiseModel, sample_correlated_noise
    from spevae.refine import z_descend
    m = Mixture2(f, 0)
    for nz in (1e-4, 1e-3):
        for name, (A_true, G_clean) in ns["TRUTHS"].items():
            n, sigma, R = sample_correlated_noise(G_clean, ns["x"], NoiseModel(scale=nz, xi=0.1),
                                                  torch.Generator().manual_seed(7))
            ev, V = torch.linalg.eigh(R)
            isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
            zr = z_descend(m, (G_clean + n).reshape(-1), ns["x"], ns["BETA"],
                           sigma=sigma.reshape(-1), inv_sqrt_R=isr, steps=ns["STEPS"])
            A = ns["A_of"](zr.poles.to_dimensionless(ns["BETA"]))
            rep[f"{name}@{nz:g}"] = dict(w1=ns["w1"](A, A_true), chi2=zr.chi2)
            print(f"[{name:<17}@{nz:g}] mixture2 zd W1 {rep[f'{name}@{nz:g}']['w1']:.4f} "
                  f"chi2 {zr.chi2:.3g}", flush=True)

json.dump(rep, open(OUT / f"{TAG}_control_{PART}.json", "w"))
print("CONTROL_DONE")
