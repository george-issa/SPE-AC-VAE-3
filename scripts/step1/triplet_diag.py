import sys, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.physics.templates import fit_peaks, peak_starts
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise
beta, ntau, nz = 8.0, 64, 1e-4
eps, gam, w = [0.4, 1.1, 2.6], [0.35, 0.25, 0.45], [0.5, 0.3, 0.2]
truth = PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                a=torch.tensor([w]), b=torch.zeros(1, 3))
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0, 1.0, ntau+1)
Gc = PoleGreens(quad)(truth, x)
for ck in ["out/step1/s1_fixes.pt", "out/microsweep/id_ladder_full.pt"]:
    model = load_model(ck)
    for seed in (7, 11):
        n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=nz, xi=0.1),
                                              torch.Generator().manual_seed(seed))
        G = Gc + n
        ev, V = torch.linalg.eigh(R)
        isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
        with torch.no_grad():
            out = model(G, x, beta, sigma=sigma, deterministic=True)
        amort = out.poles.detach().to_physical(beta)
        free, tr = rank_scan(amort, beta, x[None], G, sigma, inv_sqrt_R=isr)
        pk = fit_peaks(beta, x, G, sigma, inv_sqrt_R=isr, n=3,
                       starts=peak_starts(amort, 3))
        w1f = float(spectral_w1(free.to_dimensionless(beta).spectral(quad.nodes),
                                truth.spectral(quad.nodes), quad, beta=beta))
        w1p = float(spectral_w1(pk.poles.to_dimensionless(beta).spectral(quad.nodes),
                                truth.spectral(quad.nodes), quad, beta=beta))
        print(f"{ck.split('/')[-1]:<22} seed {seed}: scan n={tr.selected_n} "
              f"chi2 {tr.chi2_selected:.3f} W1 {w1f:.4f} | peaks(3) chi2 {pk.chi2:.3f} "
              f"W1 {w1p:.4f}", flush=True)
print("DIAG_DONE")
