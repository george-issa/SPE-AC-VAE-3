import sys, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise
beta, ntau, noise = 8.0, 64, 1e-4
s0 = 1 + 0.45
e3 = [-2.2, -1.2, 1.2, 2.2]; g3 = [0.3]*4
a3 = [0.45/s0/2, 1/s0/2, 1/s0/2, 0.45/s0/2]
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0,1.0,ntau+1)
truth = PoleSet(eps=torch.tensor([e3])*beta, gamma=torch.tensor([g3])*beta,
                a=torch.tensor([a3]), b=torch.zeros(1,4))
Gc = PoleGreens(quad)(truth, x)
n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=noise, xi=0.1),
                                      torch.Generator().manual_seed(7))
G = Gc + n
ev, V = torch.linalg.eigh(R)
isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
model = load_model("out/microsweep/id_ladder_full.pt")
with torch.no_grad():
    out = model(G, x, beta, sigma=sigma, deterministic=True)
sel, tr = rank_scan(out.poles.detach().to_physical(beta), beta, x[None], G,
                    sigma, inv_sqrt_R=isr, ph_symmetric=True)
w1 = float(spectral_w1(sel.to_dimensionless(beta).spectral(quad.nodes),
                       truth.spectral(quad.nodes), quad, beta=beta))
print(f"[mirrored+PH] true 4  {tr}")
print(f"  W1 {w1:.4f}")
for i in torch.argsort(sel.eps[0]):
    print(f"    eps {float(sel.eps[0,i]):+.4f}  gam {float(sel.gamma[0,i]):.4f}  "
          f"a {float(sel.a[0,i]):.4f}  b {float(sel.b[0,i]):+.5f}")
