"""Latent-landscape experiment: is the posterior multimodal in arrangement
space, does some z decode near truth, and does z-space chi^2 descent reach it?"""
import sys, json, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise
import numpy as np

beta, ntau, nz = 8.0, 64, 1e-4
def mk(eps, gam, a, t):
    a_, t_ = np.array(a), np.array(t)
    b = a_ * (t_ - (a_ * t_).sum() / a_.sum())
    return (torch.tensor([eps])*beta, torch.tensor([gam])*beta,
            torch.tensor([a]), torch.tensor([b.tolist()]))
ks = np.linspace(-1, 1, 5)
PROBES = [("fano band", *mk((0.3+ks*1.2).tolist(),[0.5]*5,[0.2]*5,
                            np.clip(0.7*ks,-0.45,0.45).tolist())),
          ("asym triplet", *mk([0.4,1.1,2.6],[0.35,0.25,0.45],[0.5,0.3,0.2],
                               [0.35,-0.4,-0.25]))]
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0, 1.0, ntau+1)
fwd = PoleGreens(quad)
K = fwd.kernel_matrix(x)
model = load_model("out/step1/s1_metric.pt")
ART = {"nodes": (quad.nodes/beta).tolist(), "probes": []}
for name, te, tg, ta, tb in PROBES:
    truth = PoleSet(eps=te, gamma=tg, a=ta, b=tb)
    Gc = PoleGreens(quad)(truth, x)
    n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=nz, xi=0.1),
                                          torch.Generator().manual_seed(7))
    G = Gc + n
    At = truth.spectral(quad.nodes)
    with torch.no_grad():
        mu, logvar, norm = model.encode(G, x, beta, sigma=sigma)
    std = (0.5 * logvar).exp()
    def chi2_of(ps):
        Gp = fwd(ps, x, kernel_matrix=K)
        d = (Gp - G) / sigma
        return float((d * d).mean())
    # (1) posterior sampling: multimodality + best-sample quality
    gen = torch.Generator().manual_seed(0)
    rows = []
    with torch.no_grad():
        for i in range(64):
            z = mu + std * torch.randn(mu.shape, generator=gen)
            ps = model.decode(z, norm, beta)
            rows.append((chi2_of(ps),
                         float(spectral_w1(ps.spectral(quad.nodes), At, quad,
                                           beta=beta))))
        ps_mean = model.decode(mu, norm, beta)
        mean_row = (chi2_of(ps_mean),
                    float(spectral_w1(ps_mean.spectral(quad.nodes), At, quad,
                                      beta=beta)))
    chi2s = [r[0] for r in rows]; w1s = [r[1] for r in rows]
    # (2) z-space chi^2 descent from the posterior mean and from best sample
    def z_descend(z0, steps=800, lr=0.05):
        z = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        best_c, best_z = float("inf"), z0.clone()
        for _ in range(steps):
            ps = model.decode(z, norm, beta)
            Gp = fwd(ps, x, kernel_matrix=K)
            loss = (((Gp - G) / sigma) ** 2).mean()
            v = float(loss.detach())
            if v < best_c:
                best_c, best_z = v, z.detach().clone()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            ps = model.decode(best_z, norm, beta)
            return best_c, float(spectral_w1(ps.spectral(quad.nodes), At, quad,
                                             beta=beta)), ps
    zbest = None
    with torch.no_grad():
        i_best = int(np.argmin(chi2s))
    gen2 = torch.Generator().manual_seed(0)
    zs = [mu + std * torch.randn(mu.shape, generator=gen2) for _ in range(64)]
    c_mean, w_mean, _ = z_descend(mu)
    c_samp, w_samp, ps_s = z_descend(zs[i_best])
    print(f"[{name}]")
    print(f"  posterior mean : chi2 {mean_row[0]:9.1f}  W1 {mean_row[1]:.4f}")
    print(f"  64 samples     : chi2 median {np.median(chi2s):9.1f}  "
          f"best {min(chi2s):9.1f} | W1 median {np.median(w1s):.4f}  "
          f"best {min(w1s):.4f}  spread {np.std(w1s):.4f}")
    print(f"  z-descent(mean): chi2 {c_mean:9.1f}  W1 {w_mean:.4f}")
    print(f"  z-descent(best): chi2 {c_samp:9.1f}  W1 {w_samp:.4f}", flush=True)
    ART["probes"].append(dict(name=name, truth_A=(At[0]*beta).tolist(),
                              mean=mean_row, samples=rows,
                              zdesc_mean=[c_mean, w_mean],
                              zdesc_best=[c_samp, w_samp],
                              A_zdesc=(ps_s.spectral(quad.nodes)[0]*beta).tolist()))
json.dump(ART, open("out/step1/latent_landscape.json", "w"))
print("LATENT_DONE")
