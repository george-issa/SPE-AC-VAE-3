"""Model-menu gate: templates + free rank scan, selected by ascending-k LR."""
import sys, torch
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.physics.templates import (TemplateFit, fit_ladder, ladder_starts,
                                      select_model)
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise

beta, ntau = 8.0, 64
def ladder(e0, Om, m, g, d):
    e = [e0 + k*Om for k in range(m)]; a = [d**k for k in range(m)]; s = sum(a)
    return e, [g]*m, [v/s for v in a]
e1,g1,a1 = ladder(0.5,1.0,3,0.30,0.55)
e2,g2,a2 = ladder(-0.4,0.8,4,0.25,0.60)
em,gm,am = ladder(1.2,1.0,2,0.30,0.45)
e3 = [-v for v in reversed(em)]+em; g3 = list(reversed(gm))+gm
a3 = [v/2 for v in reversed(am)]+[v/2 for v in am]
CASES = [("mirrored", e3,g3,a3)]
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0,1.0,ntau+1)
model = load_model("out/microsweep/id_ladder_full.pt")
for nz in [1e-4, 3e-5]:
    print(f"===== noise {nz:.0e} =====")
    for name, eps, gam, w in CASES:
        truth = PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                        a=torch.tensor([w]), b=torch.zeros(1,len(eps)))
        Gc = PoleGreens(quad)(truth, x)
        n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=nz, xi=0.1),
                                              torch.Generator().manual_seed(7))
        G = Gc + n
        ev, V = torch.linalg.eigh(R)
        isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
        sig_net = sigma.clamp_min(3e-5 * float(Gc.abs().max()))
        with torch.no_grad():
            out = model(G, x, beta, sigma=sig_net, deterministic=True)
        amort = out.poles.detach().to_physical(beta)
        menu = []
        st = ladder_starts(amort)
        st_m = ladder_starts(amort, mirrored=True)
        for m in (2, 3, 4, 5):
            menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m, starts=st))
        for m in (1, 2, 3):
            menu.append(fit_ladder(beta, x, G, sigma, inv_sqrt_R=isr, m=m,
                                   mirrored=True, starts=st_m, steps=4000))
        free, tr = rank_scan(amort, beta, x[None], G, sigma, inv_sqrt_R=isr)
        menu.append(TemplateFit(name="free", m=tr.selected_n,
                                k_params=4*tr.selected_n, chi2=tr.chi2_selected,
                                poles=free, params={}))
        winner = select_model(menu, L_eff=ntau+1)
        As = winner.poles.to_dimensionless(beta).spectral(quad.nodes)
        w1 = float(spectral_w1(As, truth.spectral(quad.nodes), quad, beta=beta))
        print(f"[{name:<9}] winner: {winner}  W1 {w1:.4f}")
        losers = ", ".join(f"{c.name}({c.m}):{c.chi2:.3g}" for c in
                           sorted(menu, key=lambda c: c.chi2)[:4] if c is not winner)
        print(f"{'':<12}runners-up: {losers}")
