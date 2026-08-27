"""Rank-scan gate figure: reconstructions (top) and chi^2(n) curves (bottom)."""
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
CASES = [("3-rung ladder", e1,g1,a1), ("4-rung ladder", e2,g2,a2),
         ("mirrored ladder", e3,g3,a3)]
quad = SpectralQuadrature.for_beta(beta)
x = torch.linspace(0.0,1.0,ntau+1)
nodes = quad.nodes; om = (nodes/beta).numpy()
model = load_model("out/microsweep/id_ladder_full.pt")
fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.4), constrained_layout=True,
                         gridspec_kw={"height_ratios": [1.5, 1.0]})
for col, (name, eps, gam, w) in enumerate(CASES):
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
    sel, tr = rank_scan(out.poles.detach().to_physical(beta), beta, x[None], G,
                        sigma, inv_sqrt_R=isr)
    As = sel.to_dimensionless(beta).spectral(nodes)
    At = truth.spectral(nodes)
    w1 = float(spectral_w1(As, At, quad, beta=beta))
    ax = axes[0][col]
    ax.plot(om, (At[0]*beta).numpy(), color="#1D242C", lw=1.9, zorder=2)
    ax.plot(om, (As[0]*beta).numpy(), color="#B8451F", lw=1.4, ls="--", zorder=3)
    ax.scatter(eps, [0.012]*len(eps), s=130, marker="*", facecolors="none",
               edgecolors="#111", linewidths=1.2, zorder=5, clip_on=False)
    pe = sel.eps[0].numpy(); pa = sel.a[0].numpy()
    Ah = sel.to_dimensionless(beta).spectral(
        (torch.tensor(pe)*beta).to(torch.get_default_dtype()))
    ax.vlines(pe, 0, Ah[0].numpy()*beta, color="#B8451F", lw=0.9, alpha=0.65, zorder=1)
    ax.scatter(pe, Ah[0].numpy()*beta, s=10 + 70*pa/max(pa.max(),1e-9),
               color="#B8451F", edgecolors="#5A2410", linewidths=0.4, zorder=4)
    ax.axhline(0, color="#999", lw=0.4)
    ax.set_xlim(-4.6, 4.6); ax.set_title(name, pad=5)
    ax.text(0.02, 0.97, f"n = {tr.selected_n} (true {len(eps)})\n"
            f"W$_1$ {w1:.4f}\n$\\chi^2$ {tr.chi2_selected:.2f}",
            transform=ax.transAxes, va="top", fontsize=7.5, family="monospace")
    if col == 0: ax.set_ylabel("$A(\\omega)$")
    ax2 = axes[1][col]
    ax2.plot(tr.ns, tr.chi2s, "-o", color="#0E7C86", ms=3.5, lw=1.2)
    ax2.axhline(min(tr.chi2s), color="#999", lw=0.6, ls=":")
    ax2.scatter([tr.selected_n], [tr.chi2s[tr.ns.index(tr.selected_n)]],
                s=90, marker="o", facecolors="none", edgecolors="#B8451F",
                linewidths=1.6, zorder=5)
    ax2.axvline(len(eps), color="#1D242C", lw=0.8, ls="--", alpha=0.5)
    ax2.set_yscale("log"); ax2.set_xlabel("pole count $n$")
    ax2.set_xticks(tr.ns)
    if col == 0: ax2.set_ylabel("$\\chi^2(n)$ per dof")
handles = [Line2D([], [], color="#1D242C", lw=1.9, label="truth"),
           Line2D([], [], color="#B8451F", lw=1.4, ls="--", label="rank-scan selection"),
           Line2D([], [], marker="*", ls="", mfc="none", mec="#111", ms=10, label="true poles"),
           Line2D([], [], marker="o", ls="", mfc="none", mec="#B8451F", ms=8,
                  label="selected $n$")]
axes[0][1].legend(handles=handles[:3], fontsize=7, frameon=False, loc="upper right",
                  borderpad=0.2, handletextpad=0.4)
axes[1][1].legend(handles=[handles[3],
                  Line2D([], [], color="#1D242C", lw=0.8, ls="--", alpha=0.5, label="true count")],
                  fontsize=7, frameon=False, loc="upper right")
fig.suptitle("Rank scan on the ladder probes: reconstruction (top) and the $\\chi^2(n)$ "
             "decision curve (bottom, sequential LR at 9.5)", fontsize=10)
fig.savefig(sys.argv[1], dpi=200)
print("figure ->", sys.argv[1])
