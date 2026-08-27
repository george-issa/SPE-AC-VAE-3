"""Manifold-regularized refits on Holstein + fano + asym triplet, w-sweep."""
import sys, json, torch
sys.path.insert(0, "src")
import numpy as np
from spevae import load_model
from spevae.refine import z_descend, refine_latent, RefineConfig, manifold_refit
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise
from spevae.data.readers import read_gissa_cube

beta = 8.0
quad = SpectralQuadrature.for_beta(beta)
model = load_model("out/step1/s1_metric.pt")
WEIGHTS = [0.5, 2.0, 8.0]
ART = {"nodes": (quad.nodes/beta).tolist(), "cases": []}

def run_case(name, G, x, sigma, isr, sig_enc, truth=None, slices=None):
    zr = z_descend(model, G, x, beta, sigma=sig_enc, inv_sqrt_R=isr,
                   slices=slices, steps=1500)
    with torch.no_grad():
        _, _, norm = model.encode(G.reshape(1,-1).float(), x.float(), beta,
                                  sigma=sig_enc.reshape(1,-1).float())
    free, tr = rank_scan(zr.poles, beta, x.reshape(1,-1), G.reshape(1,-1),
                         sigma.reshape(1,-1), inv_sqrt_R=isr)
    def w1_of(ps):
        if truth is None: return None
        return float(spectral_w1(ps.to_dimensionless(beta).spectral(quad.nodes.double()),
                                 truth.spectral(quad.nodes).double(), quad.double()
                                 if hasattr(quad, "double") else quad, beta=beta)) \
            if False else float(spectral_w1(
                ps.to_dimensionless(beta).spectral(quad.nodes.to(torch.float64)).float(),
                truth.spectral(quad.nodes), quad, beta=beta))
    case = dict(name=name,
                zd=dict(chi2=zr.chi2, w1=w1_of(zr.poles),
                        A=(zr.poles.to_dimensionless(beta).spectral(
                            quad.nodes.to(torch.float64))[0]*beta).tolist()),
                free=dict(chi2=tr.chi2_selected, n=tr.selected_n, w1=w1_of(free),
                          A=(free.to_dimensionless(beta).spectral(
                              quad.nodes.to(torch.float64))[0]*beta).tolist()),
                truth_A=None if truth is None else (truth.spectral(quad.nodes)[0]*beta).tolist(),
                sweeps=[])
    print(f"[{name}] zd chi2 {zr.chi2:.2f} w1 {case['zd']['w1']} | "
          f"free({tr.selected_n}) chi2 {tr.chi2_selected:.3f} w1 {case['free']['w1']}",
          flush=True)
    for w in WEIGHTS:
        pf, info = manifold_refit(model, free, zr.z, norm, G, x, beta, sigma,
                                  inv_sqrt_R=isr, weight=w, rounds=4,
                                  fit_steps=1000, proj_steps=300)
        w1v = w1_of(pf)
        print(f"  w={w:>4}: chi2 {info['chi2']:.3f}  penalty {info['penalty']:.4f}"
              f"  w1 {w1v}", flush=True)
        case["sweeps"].append(dict(weight=w, chi2=info["chi2"],
                                   penalty=info["penalty"], w1=w1v,
                                   A=(pf.to_dimensionless(beta).spectral(
                                       quad.nodes.to(torch.float64))[0]*beta).tolist()))
    ART["cases"].append(case)

# synthetic probes
def mk(eps, gam, a, t):
    a_, t_ = np.array(a), np.array(t)
    b = a_ * (t_ - (a_ * t_).sum() / a_.sum())
    return PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                   a=torch.tensor([a]), b=torch.tensor([b.tolist()]))
ks = np.linspace(-1, 1, 5)
for name, truth in []:
    x = torch.linspace(0.0, 1.0, 65)
    Gc = PoleGreens(quad)(truth, x)
    n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=1e-4, xi=0.1),
                                          torch.Generator().manual_seed(7))
    G = Gc + n
    ev, V = torch.linalg.eigh(R)
    isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    run_case(name, G, x, sigma, isr, sigma, truth=truth)

# Holstein -- via refine_latent (full-grid encoder input, the healed path)
CUBE = "/Users/gissa/Documents/AC/SPE-AC-VAE-old/Data/datasets/real/gIssa/for_george_ac.jld2"
mg = read_gissa_cube(CUBE, beta=8.0, omega=1.0, n=1.0, lam=0.325)
cov = mg.covariance(1e-10, method="structured")
sl = mg.statistical_slices
rr = refine_latent(model, mg, RefineConfig(cov_method="structured",
                                           min_eigenvalue=1e-10, steps=3000))
print(f"[Holstein] refine_latent chi2 {rr.chi2_init:.1f} -> {rr.chi2:.2f}", flush=True)
with torch.no_grad():
    _, _, norm = model.encode(mg.mean.unsqueeze(0).float(), mg.x.float(), beta,
                              sigma=mg.sigma_encoder.unsqueeze(0).float())
G_h = mg.mean[sl]; x_h = mg.x[sl]; sig_h = mg.sigma_mean[sl]
isr_h = cov["inv_sqrt_R"]
free, tr = rank_scan(rr.poles, beta, x_h[None], G_h.unsqueeze(0),
                     sig_h.unsqueeze(0), inv_sqrt_R=isr_h)
print(f"[Holstein] scan: {tr}", flush=True)
case = dict(name="Holstein", truth_A=None,
            zd=dict(chi2=rr.chi2, w1=None,
                    A=(rr.poles.to_dimensionless(beta).spectral(
                        quad.nodes.to(torch.float64))[0]*beta).tolist()),
            free=dict(chi2=tr.chi2_selected, n=tr.selected_n, w1=None,
                      A=(free.to_dimensionless(beta).spectral(
                          quad.nodes.to(torch.float64))[0]*beta).tolist()),
            sweeps=[])
for w in WEIGHTS:
    pf, info = manifold_refit(model, free, rr.z, norm, G_h, x_h, beta, sig_h,
                              inv_sqrt_R=isr_h, weight=w, rounds=4,
                              fit_steps=1000, proj_steps=300)
    print(f"  w={w:>4}: chi2 {info['chi2']:.3f}  penalty {info['penalty']:.4f}",
          flush=True)
    case["sweeps"].append(dict(weight=w, chi2=info["chi2"],
                               penalty=info["penalty"], w1=None,
                               A=(pf.to_dimensionless(beta).spectral(
                                   quad.nodes.to(torch.float64))[0]*beta).tolist()))
ART["cases"].append(case)
ART["holstein_target"] = float(cov["chi2_target"])
old = json.load(open("out/step1/manifold_test.json"))
ART["cases"] = [c for c in old["cases"] if c["name"] != "Holstein"] + ART["cases"]
json.dump(ART, open("out/step1/manifold_test.json", "w"))
print("MANIFOLD_DONE")
