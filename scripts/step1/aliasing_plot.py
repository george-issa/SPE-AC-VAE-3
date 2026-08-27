"""The gate figure: consolidated reconstructions (composite model, strict
parsimony) with pole structure vs truth -- makes the comb aliasing visible."""
import sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, "src")
from spevae import load_model
from spevae.physics.poles import PoleSet
from spevae.physics.forward import SpectralQuadrature, PoleGreens
from spevae.physics.consolidate import minimal_pole_fixed_point
from spevae.losses import spectral_w1
from spevae.data.noise import NoiseModel, sample_correlated_noise

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#3A3F45"})
beta, ntau, noise = 8.0, 64, 1e-4
def ladder(e0, Om, m, g, d):
    e = [e0 + k*Om for k in range(m)]; a = [d**k for k in range(m)]; s = sum(a)
    return e, [g]*m, [v/s for v in a]
e1,g1,a1 = ladder(0.5,1.0,3,0.30,0.55)
e2,g2,a2 = ladder(-0.4,0.8,4,0.25,0.60)
em,gm,am = ladder(1.2,1.0,2,0.30,0.45)
e3 = [-v for v in reversed(em)]+em; g3 = list(reversed(gm))+gm
a3 = [v/2 for v in reversed(am)]+[v/2 for v in am]
CASES = [("3-rung ladder — aliased", e1,g1,a1, "the data equally supports a denser comb"),
         ("4-rung ladder — exact", e2,g2,a2, ""),
         ("mirrored ladder — hedged", e3,g3,a3, "symmetry unknown to the model")]
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0,1.0,ntau+1)
nodes = quad.nodes; om = (nodes/beta).numpy()
model = load_model("out/microsweep/id_ladder_full.pt")
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7), constrained_layout=True)
for ax, (name, eps, gam, w, note) in zip(axes, CASES):
    truth = PoleSet(eps=torch.tensor([eps])*beta, gamma=torch.tensor([gam])*beta,
                    a=torch.tensor([w]), b=torch.zeros(1,len(eps)))
    Gc = PoleGreens(quad)(truth, x)
    n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=noise, xi=0.1),
                                          torch.Generator().manual_seed(7))
    G = Gc + n
    ev, V = torch.linalg.eigh(R)
    isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    with torch.no_grad():
        out = model(G, x, beta, sigma=sigma, deterministic=True)
    pc, tr = minimal_pole_fixed_point(out.poles.detach().to_physical(beta), beta,
                                      x[None], G, sigma, inv_sqrt_R=isr,
                                      prune_quantile=40.0)
    Ac = pc.to_dimensionless(beta).spectral(nodes)
    At = truth.spectral(nodes)
    w1 = float(spectral_w1(Ac, At, quad, beta=beta))
    ax.plot(om, (At[0]*beta).numpy(), color="#1D242C", lw=1.9, zorder=2)
    ax.plot(om, (Ac[0]*beta).numpy(), color="#B8451F", lw=1.4, ls="--", zorder=3)
    # truth poles: stars on the axis; found poles: stems up to the curve
    ax.scatter(eps, [0.015]*len(eps), s=140, marker="*", facecolors="none",
               edgecolors="#111", linewidths=1.2, zorder=5, clip_on=False)
    pe = pc.eps[0].numpy(); pa = pc.a[0].numpy()
    Ah = pc.to_dimensionless(beta).spectral(
        (torch.tensor(pe)*beta).to(torch.get_default_dtype()))
    ax.vlines(pe, 0, Ah[0].numpy()*beta, color="#B8451F", lw=0.9, alpha=0.65, zorder=1)
    ax.scatter(pe, Ah[0].numpy()*beta, s=10 + 70*pa/max(pa.max(),1e-9),
               color="#B8451F", edgecolors="#5A2410", linewidths=0.4, zorder=4)
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_xlim(-4.6, 4.6); ax.set_title(name, pad=5); ax.set_xlabel("$\\omega$")
    ax.text(0.02, 0.97, f"{pc.a.shape[-1]} poles (true {len(eps)})\n"
            f"W$_1$ {w1:.4f}\n$\\chi^2$ {tr.chi2_final:.2f}",
            transform=ax.transAxes, va="top", fontsize=7.5, family="monospace")
    ax.text(0.98, 0.03, note, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=7, style="italic", color="#5A6572")
axes[0].set_ylabel("$A(\\omega)$")
handles = [Line2D([], [], color="#1D242C", lw=1.9, label="truth $A(\\omega)$"),
           Line2D([], [], color="#B8451F", lw=1.4, ls="--", label="consolidated $A(\\omega)$"),
           Line2D([], [], marker="*", ls="", mfc="none", mec="#111", ms=10, label="true poles"),
           Line2D([], [], marker="o", ls="", mfc="#B8451F", mec="#5A2410", ms=5, label="found poles")]
axes[1].legend(handles=handles, fontsize=7, frameon=False, loc="upper right",
                borderpad=0.2, handletextpad=0.4, labelspacing=0.3)
fig.suptitle("Composite model + strict parsimony (q = 40): structure exact where the data "
             "identifies it; elsewhere parsimony visibly trades fit for simplicity", fontsize=10)
fig.savefig(sys.argv[1], dpi=200)
print("figure ->", sys.argv[1])
