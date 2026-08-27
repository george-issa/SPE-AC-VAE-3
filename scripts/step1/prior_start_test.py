"""The properly in-between phase: START at the z-descent decode, descend
whitened chi^2 with the manifold penalty as a continuous prior force."""
import sys, json, torch
sys.path.insert(0, "src")
import numpy as np
from spevae import load_model
from spevae.refine import z_descend, refine_latent, RefineConfig, manifold_refit
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise
from spevae.data.readers import read_gissa_cube

beta = 8.0
quad = SpectralQuadrature.for_beta(beta)
model = load_model("out/step1/s1_metric.pt")
WEIGHTS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
old = json.load(open("out/step1/manifold_test.json"))
ART = {"nodes": (quad.nodes/beta).tolist(), "cases": []}

def sweep(name, zr_poles, zr_z, norm, G, x, sigma, isr, truth):
    def w1_of(ps):
        if truth is None: return None
        return float(spectral_w1(
            ps.to_dimensionless(beta).spectral(quad.nodes.to(torch.float64)).float(),
            truth.spectral(quad.nodes), quad, beta=beta))
    prev = next(c for c in old["cases"] if c["name"] == name)
    case = dict(name=name, truth_A=prev.get("truth_A"),
                zd=dict(chi2=None, w1=w1_of(zr_poles),
                        A=(zr_poles.to_dimensionless(beta).spectral(
                            quad.nodes.to(torch.float64))[0]*beta).tolist()),
                free=prev["free"], free_started=prev["sweeps"], sweeps=[])
    for w in WEIGHTS:
        pf, info = manifold_refit(model, zr_poles, zr_z, norm, G, x, beta,
                                  sigma, inv_sqrt_R=isr, weight=w, rounds=4,
                                  fit_steps=1000, proj_steps=300)
        w1v = w1_of(pf)
        print(f"[{name}] w={w:>4}: chi2 {info['chi2']:.3f}  "
              f"penalty {info['penalty']:.4f}  w1 {w1v}", flush=True)
        case["sweeps"].append(dict(weight=w, chi2=info["chi2"],
                                   penalty=info["penalty"], w1=w1v,
                                   A=(pf.to_dimensionless(beta).spectral(
                                       quad.nodes.to(torch.float64))[0]*beta).tolist()))
    ART["cases"].append(case)

def mk(eps, gam, a, t):
    a_, t_ = np.array(a), np.array(t)
    b = a_ * (t_ - (a_ * t_).sum() / a_.sum())
    return PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                   a=torch.tensor([a]), b=torch.tensor([b.tolist()]))
ks = np.linspace(-1, 1, 5)
for name, truth in [("fano band", mk((0.3+ks*1.2).tolist(), [0.5]*5, [0.2]*5,
                                     np.clip(0.7*ks, -0.45, 0.45).tolist())),
                    ("asym triplet", mk([0.4,1.1,2.6],[0.35,0.25,0.45],
                                        [0.5,0.3,0.2],[0.35,-0.4,-0.25]))]:
    x = torch.linspace(0.0, 1.0, 65)
    Gc = PoleGreens(quad)(truth, x)
    n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=1e-4, xi=0.1),
                                          torch.Generator().manual_seed(7))
    G = Gc + n
    ev, V = torch.linalg.eigh(R)
    isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    zr = z_descend(model, G, x, beta, sigma=sigma, inv_sqrt_R=isr, steps=1500)
    with torch.no_grad():
        _, _, norm = model.encode(G.reshape(1,-1).float(), x.float(), beta,
                                  sigma=sigma.reshape(1,-1).float())
    sweep(name, zr.poles, zr.z, norm, G, x, sigma, isr, truth)

CUBE = "/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/gIssa/for_george_ac.jld2"
mg = read_gissa_cube(CUBE, beta=8.0, omega=1.0, n=1.0, lam=0.325)
cov = mg.covariance(1e-10, method="structured")
sl = mg.statistical_slices
rr = refine_latent(model, mg, RefineConfig(cov_method="structured",
                                           min_eigenvalue=1e-10, steps=3000))
with torch.no_grad():
    _, _, norm = model.encode(mg.mean.unsqueeze(0).float(), mg.x.float(), beta,
                              sigma=mg.sigma_encoder.unsqueeze(0).float())
from spevae.physics.consolidate import rank_scan
G_h, x_h, s_h, isr_h = mg.mean[sl], mg.x[sl], mg.sigma_mean[sl], cov["inv_sqrt_R"]
case = dict(name="Holstein", truth_A=None,
            zd=dict(chi2=rr.chi2, w1=None,
                    A=(rr.poles.to_dimensionless(beta).spectral(
                        quad.nodes.to(torch.float64))[0]*beta).tolist()),
            free=next(c for c in old["cases"] if c["name"] == "Holstein")["free"],
            sweeps=[])
for w in WEIGHTS:
    pf, info = manifold_refit(model, rr.poles, rr.z, norm, G_h, x_h, beta, s_h,
                              inv_sqrt_R=isr_h, weight=w, rounds=4,
                              fit_steps=1000, proj_steps=300)
    gs, gt = rank_scan(pf, beta, x_h[None], G_h.unsqueeze(0), s_h.unsqueeze(0),
                       inv_sqrt_R=isr_h)
    print(f"[Holstein] w={w:>5}: refit chi2 {info['chi2']:.3f} "
          f"pen {info['penalty']:.4f} | gated n={gt.selected_n} "
          f"chi2 {gt.chi2_selected:.3f}", flush=True)
    case["sweeps"].append(dict(weight=w, chi2=info["chi2"],
                               penalty=info["penalty"], w1=None,
                               gated_n=gt.selected_n,
                               gated_chi2=gt.chi2_selected,
                               A=(pf.to_dimensionless(beta).spectral(
                                   quad.nodes.to(torch.float64))[0]*beta).tolist(),
                               A_gated=(gs.to_dimensionless(beta).spectral(
                                   quad.nodes.to(torch.float64))[0]*beta).tolist()))
ART["cases"].append(case)
ART["holstein_target"] = float(cov["chi2_target"])
json.dump(ART, open("out/step1/wsweep.json", "w"))
print("PRIOR_START_DONE")
