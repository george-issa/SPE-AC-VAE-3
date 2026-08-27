"""Spectra + chi^2(n) insets across noise levels for the three probes."""
import sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import rank_scan
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#3A3F45"})
beta, ntau = 8.0, 64
def ladder(e0, Om, m, g, d):
    e = [e0 + k*Om for k in range(m)]; a = [d**k for k in range(m)]; s = sum(a)
    return e, [g]*m, [v/s for v in a]
e1,g1,a1 = ladder(0.5,1.0,3,0.30,0.55)
e2,g2,a2 = ladder(-0.4,0.8,4,0.25,0.60)
em,gm,am = ladder(1.2,1.0,2,0.30,0.45)
e3 = [-v for v in reversed(em)]+em; g3 = list(reversed(gm))+gm
a3 = [v/2 for v in reversed(am)]+[v/2 for v in am]
CASES = [("3-rung", e1,g1,a1), ("4-rung", e2,g2,a2), ("mirrored", e3,g3,a3)]
NOISES = [3e-3, 3e-4, 1e-4, 3e-5]
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0,1.0,ntau+1)
nodes = quad.nodes; om = (nodes/beta).numpy()
model = load_model("out/microsweep/id_ladder_full.pt")
fig, axes = plt.subplots(3, 4, figsize=(12.4, 8.2), constrained_layout=True,
                         sharex="row", sharey="row")
for row, (name, eps, gam, w) in enumerate(CASES):
    truth = PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                    a=torch.tensor([w]), b=torch.zeros(1,len(eps)))
    Gc = PoleGreens(quad)(truth, x)
    At = truth.spectral(nodes)
    for col, nz in enumerate(NOISES):
        n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=nz, xi=0.1),
                                              torch.Generator().manual_seed(7))
        G = Gc + n
        ev, V = torch.linalg.eigh(R)
        isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
        sig_net = sigma.clamp_min(3e-5 * float(Gc.abs().max()))
        with torch.no_grad():
            out = model(G, x, beta, sigma=sig_net, deterministic=True)
        sel, tr = rank_scan(out.poles.detach().to_physical(beta), beta, x[None],
                            G, sigma, inv_sqrt_R=isr)
        As = sel.to_dimensionless(beta).spectral(nodes)
        w1 = float(spectral_w1(As, At, quad, beta=beta))
        ax = axes[row][col]
        ax.plot(om, (At[0]*beta).numpy(), color="#1D242C", lw=1.6, zorder=2)
        ax.plot(om, (As[0]*beta).numpy(), color="#B8451F", lw=1.2, ls="--", zorder=3)
        ax.scatter(eps, [0.008]*len(eps), s=70, marker="*", facecolors="none",
                   edgecolors="#111", linewidths=0.9, zorder=5, clip_on=False)
        pe = sel.eps[0].numpy()
        Ah = sel.to_dimensionless(beta).spectral(
            (torch.tensor(pe)*beta).to(torch.get_default_dtype()))
        ax.vlines(pe, 0, Ah[0].numpy()*beta, color="#B8451F", lw=0.7, alpha=0.6, zorder=1)
        ax.set_xlim(-4.4, 4.4)
        ok = "=" if tr.selected_n == len(eps) else ("<" if tr.selected_n < len(eps) else ">")
        ax.text(0.02, 0.97, f"n {tr.selected_n} {ok} {len(eps)}\nW$_1$ {w1:.3f}",
                transform=ax.transAxes, va="top", fontsize=7, family="monospace")
        if row == 0: ax.set_title(f"$\\sigma$ = {nz:.0e}", pad=4)
        if col == 0: ax.set_ylabel(f"{name}\n$A(\\omega)$")
        if row == 2: ax.set_xlabel("$\\omega$")
        # chi^2(n) inset
        ins = ax.inset_axes([0.66, 0.55, 0.32, 0.4])
        ins.plot(tr.ns, tr.chi2s, "-o", color="#0E7C86", ms=2, lw=0.9)
        ins.scatter([tr.selected_n], [tr.chi2s[tr.ns.index(tr.selected_n)]],
                    s=40, facecolors="none", edgecolors="#B8451F", linewidths=1.1)
        ins.axvline(len(eps), color="#1D242C", lw=0.6, ls="--", alpha=0.5)
        ins.set_yscale("log")
        ins.tick_params(labelsize=5.5, length=1.5, pad=1)
        ins.set_xticks([2, 4, 6, 8])
        ins.set_title("$\\chi^2(n)$", fontsize=6, pad=1)
fig.suptitle("Selected reconstructions and $\\chi^2(n)$ decision curves across noise "
             "(circled = selected count, dashed line = true count)", fontsize=10)
fig.savefig(sys.argv[1], dpi=190)
print("figure ->", sys.argv[1])
