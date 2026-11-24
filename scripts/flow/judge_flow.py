"""Go/no-go judge for the conditional flow posterior vs the VAE control.

Both models were trained on the same distribution (M1-recipe pole prior,
beta = 8, 6k steps, batch 128).  Four tests:

A  calibration on held-out prior draws (in distribution): simulation-based
   calibration ranks of the truth among posterior samples for 9 spectral
   statistics, interval coverage at 50% / 90%, pointwise band coverage of A(w)
B  data consistency: chi^2 of posterior samples vs the truth's own chi^2
C  two-sided pairs as the side weights balance (the VAE's mirror attractor):
   best-of-64 chi^2 and whether samples recover the side weight
D  the 14 frozen suite truths and the real Holstein cell: posterior bands

    python scripts/flow/judge_flow.py out/flow/F_6k.pt out/step2/equiv_ctrl6k.pt out/flow/judge_F6k
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spevae import load_model
from spevae.data.pole_prior import PolePriorGenerator
from spevae.evaluation import cases as C
from spevae.models.flow import load_flow
from spevae.physics import PoleGreens, SpectralQuadrature
from spevae.physics.poles import PoleSet

BETA = 8.0
S = 100                                   # posterior samples per input
W = np.linspace(-30.0, 30.0, 4001)        # physical omega grid for statistics
DW = W[1] - W[0]
EDGES = [-3.0, -1.5, -0.5, 0.5, 1.5, 3.0]  # 7 windows
STATS = [f"w[{a},{b}]" for a, b in zip([-30] + EDGES, EDGES + [30])] + ["<w>", "peak"]

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
COL = {"flow": "#2a78d6", "vae": "#eb6834"}
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True,
                     "grid.color": "#e4e3df", "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def A_phys(ps: PoleSet, beta=BETA, w=W) -> torch.Tensor:
    nu = torch.tensor(w * beta, dtype=torch.float64)
    p = PoleSet(*(getattr(ps, k).double() for k in ("eps", "gamma", "a", "b")))
    return p.spectral(nu) * beta                               # (N, len(w))


def stats_of(A: torch.Tensor) -> np.ndarray:
    A = A.numpy()
    cum = np.cumsum(A, 1) * DW
    out = []
    lo = -30.0
    for e in EDGES + [30.0]:
        i0, i1 = np.searchsorted(W, lo), np.searchsorted(W, e)
        out.append(cum[:, min(i1, len(W) - 1)] - (cum[:, i0] if i0 > 0 else 0))
        lo = e
    core = np.abs(W) <= 10
    out.append((A[:, core] * W[core]).sum(1) * DW / np.maximum(A[:, core].sum(1) * DW, 1e-12))
    out.append(W[core][A[:, core].argmax(1)])
    return np.stack(out, 1)                                    # (N, 9)


def w1(A: np.ndarray, At: np.ndarray) -> float:
    return float(np.abs(np.cumsum(A - At) * DW).sum() * DW)


def chi2(ps: PoleSet, quad, x, G, sigma, isr) -> torch.Tensor:
    G_rec = PoleGreens(quad)(PoleSet(*(getattr(ps, k).float() for k in
                                       ("eps", "gamma", "a", "b"))), x)
    d = ((G_rec - G) / sigma) @ isr.T
    return (d * d).mean(-1)


def vae_samples(m, G, x, beta, sigma, n, gen):
    with torch.no_grad():
        mu, logvar, norm = m.encode(G, x, beta, sigma=sigma)
        mu, logvar = mu.repeat_interleave(n, 0), logvar.repeat_interleave(n, 0)
        z = mu + torch.randn(mu.shape, generator=gen) * torch.exp(0.5 * logvar.clamp(-20, 20))
        return m.decode(z, norm.repeat_interleave(n, 0), beta)


def vae_mean(m, G, x, beta, sigma):
    with torch.no_grad():
        return m(G, x, beta, sigma=sigma, deterministic=True).poles


# ---------------------------------------------------------------- A + B --- #
def test_calibration(flow, vae, M=256, bs=64):
    gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive",
                             beta_range=(BETA, BETA))
    g = torch.Generator().manual_seed(31337)
    out = {k: {"ranks": [], "cover50": [], "cover90": [], "band90": [], "band_sig": [],
               "chi2_med": [], "w1_mean": [], "w1_sample_med": []} for k in ("flow", "vae")}
    out["truth_chi2"], out["vae_amortized_w1"] = [], []
    with gen.fixed_scalars(31337):
        for k in range(M // bs):
            b = gen.batch(bs, torch.Generator().manual_seed(4242 + k))
            truth = PoleSet(b.poles_true["eps"], b.poles_true["gamma"],
                            b.poles_true["a"], b.poles_true["b"])
            At = A_phys(truth)
            st = stats_of(At)
            out["truth_chi2"] += chi2(truth, b.quad, b.x, b.G, b.sigma, b.inv_sqrt_R).tolist()
            Aam = A_phys(vae_mean(vae, b.G, b.x, BETA, b.sigma)).numpy()
            out["vae_amortized_w1"] += [w1(Aam[i], At[i].numpy()) for i in range(bs)]
            for name in ("flow", "vae"):
                if name == "flow":
                    ps = flow.sample(b.G, b.x, BETA, sigma=b.sigma, n_samples=S, generator=g)
                else:
                    ps = vae_samples(vae, b.G, b.x, BETA, b.sigma, S, g)
                A = A_phys(ps)                                     # (bs*S, W)
                ss = stats_of(A).reshape(bs, S, -1)
                c2 = chi2(ps, b.quad, b.x, b.G.repeat_interleave(S, 0),
                          b.sigma.repeat_interleave(S, 0), b.inv_sqrt_R).reshape(bs, S)
                A = A.reshape(bs, S, -1).numpy()
                core = np.abs(W) <= 8
                for i in range(bs):
                    r = (ss[i] < st[i]).sum(0) + np.random.default_rng(i).integers(
                        0, (ss[i] == st[i]).sum(0) + 1)
                    out[name]["ranks"].append(r.tolist())
                    q = np.quantile(ss[i], [0.25, 0.75, 0.05, 0.95], axis=0)
                    out[name]["cover50"].append(((st[i] >= q[0]) & (st[i] <= q[1])).tolist())
                    out[name]["cover90"].append(((st[i] >= q[2]) & (st[i] <= q[3])).tolist())
                    lo, hi = np.quantile(A[i][:, core], [0.05, 0.95], axis=0)
                    ti = At[i].numpy()[core]
                    inside = (ti >= lo - 1e-9) & (ti <= hi + 1e-9)
                    sig = ti > 0.05 * ti.max()
                    out[name]["band90"].append(float(inside.mean()))
                    out[name]["band_sig"].append(float(inside[sig].mean()))
                    out[name]["chi2_med"].append(float(c2[i].median()))
                    out[name]["w1_mean"].append(w1(A[i].mean(0), At[i].numpy()))
                    out[name]["w1_sample_med"].append(float(np.median(
                        [w1(A[i][j], At[i].numpy()) for j in range(0, S, 5)])))
            print(f"  calibration batch {k + 1}/{M // bs}", flush=True)
    summ = {}
    for name in ("flow", "vae"):
        R = np.array(out[name]["ranks"]) / S                       # (M, 9) in [0, 1]
        ks = [float(np.max(np.abs(np.sort(R[:, j]) - (np.arange(1, len(R) + 1) / len(R)))))
              for j in range(R.shape[1])]
        summ[name] = {
            "ks_per_stat": dict(zip(STATS, ks)),
            "ks_max": max(ks), "ks_crit_5pct": 1.36 / math.sqrt(len(R)),
            "cover50": float(np.mean(out[name]["cover50"])),
            "cover90": float(np.mean(out[name]["cover90"])),
            "band90_pointwise": float(np.mean(out[name]["band90"])),
            "band90_signal": float(np.mean(out[name]["band_sig"])),
            "sample_chi2_median": float(np.median(out[name]["chi2_med"])),
            "w1_posterior_mean": float(np.median(out[name]["w1_mean"])),
            "w1_single_sample": float(np.median(out[name]["w1_sample_med"])),
        }
    summ["truth_chi2_median"] = float(np.median(out["truth_chi2"]))
    summ["vae_amortized_w1"] = float(np.median(out["vae_amortized_w1"]))
    return summ, out


def fig_sbc(out, summ, path):
    fig, axes = plt.subplots(2, 9, figsize=(16, 4.2), sharey="row", constrained_layout=True)
    for row, name in enumerate(("flow", "vae")):
        R = np.array(out[name]["ranks"])
        for j, s in enumerate(STATS):
            ax = axes[row, j]
            ax.hist(R[:, j], bins=10, range=(0, S + 1), color=COL[name], rwidth=0.9)
            ax.axhline(len(R) / 10, color=INK2, lw=0.9, ls="--")
            ax.set_title(s if row == 0 else "", fontsize=7.5)
            ax.set_xticks([])
        axes[row, 0].set_ylabel(f"{name}\nrank count")
    fig.suptitle("Simulation-based calibration on 256 held-out prior draws: rank of the truth "
                 f"among {S} posterior samples (flat = calibrated; dashed = uniform)\n"
                 f"max KS: flow {summ['flow']['ks_max']:.3f}, VAE {summ['vae']['ks_max']:.3f} "
                 f"(5% critical {summ['flow']['ks_crit_5pct']:.3f})", fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- C --- #
def test_pairs(flow, vae):
    quad = SpectralQuadrature.for_beta(BETA)
    x = torch.linspace(0.0, 1.0, 65)
    g = torch.Generator().manual_seed(99)
    rows, curves = [], {}
    for wl in (0.8, 0.65, 0.55, 0.5):
        truth = PoleSet(eps=torch.tensor([[-1.6, 1.4]]) * BETA, gamma=torch.tensor([[0.45, 0.5]]) * BETA,
                        a=torch.tensor([[wl, 1 - wl]]), b=torch.zeros(1, 2))
        Gc = PoleGreens(quad)(truth, x)
        from spevae.data.noise import NoiseModel, sample_correlated_noise
        n, sigma, R = sample_correlated_noise(Gc, x, NoiseModel(scale=1e-4, xi=0.1),
                                              torch.Generator().manual_seed(7))
        G = (Gc + n).reshape(1, -1)
        sigma = sigma.reshape(1, -1)
        ev, V = torch.linalg.eigh(R)
        isr = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
        row = {"w_left": wl}
        am = vae_mean(vae, G, x, BETA, sigma)
        row["vae_amortized_chi2"] = float(chi2(am, quad, x, G, sigma, isr)[0])
        for name in ("flow", "vae"):
            ps = (flow.sample(G, x, BETA, sigma=sigma, n_samples=64, generator=g) if name == "flow"
                  else vae_samples(vae, G, x, BETA, sigma, 64, g))
            c2 = chi2(ps, quad, x, G.expand(64, -1), sigma.expand(64, -1), isr)
            A = A_phys(ps).numpy()
            left = A[:, W < 0].sum(1) * DW / np.maximum(A.sum(1) * DW, 1e-12)
            row[f"{name}_best64_chi2"] = float(c2.min())
            row[f"{name}_median_chi2"] = float(c2.median())
            row[f"{name}_left_weight_q05_q50_q95"] = np.quantile(left, [0.05, 0.5, 0.95]).tolist()
            row[f"{name}_frac_within_0.05"] = float((np.abs(left - wl) < 0.05).mean())
            curves[(wl, name)] = (np.quantile(A, [0.05, 0.5, 0.95], axis=0))
        curves[(wl, "truth")] = A_phys(truth).numpy()[0]
        rows.append(row)
        print(f"  pair {wl:.2f}: VAE amort chi2 {row['vae_amortized_chi2']:.3g}, best64 "
              f"VAE {row['vae_best64_chi2']:.3g} flow {row['flow_best64_chi2']:.3g}; "
              f"left weight flow {row['flow_left_weight_q05_q50_q95']}", flush=True)
    return rows, curves


def fig_pairs(rows, curves, path):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2), sharex=True, constrained_layout=True)
    core = np.abs(W) <= 4
    for j, r in enumerate(rows):
        wl = r["w_left"]
        for i, name in enumerate(("flow", "vae")):
            ax = axes[i, j]
            q = curves[(wl, name)]
            ax.fill_between(W[core], q[0][core], q[2][core], color=COL[name], alpha=0.25, lw=0,
                            label="90% band")
            ax.plot(W[core], q[1][core], color=COL[name], lw=1.3, label="median")
            ax.plot(W[core], curves[(wl, "truth")][core], color=INK, lw=1.6, ls="--", label="truth")
            ax.set_title(f"{name}: left weight {wl:.2f}, best-of-64 chi2 "
                         f"{r[f'{name}_best64_chi2']:.2g}", fontsize=8)
        axes[1, j].set_xlabel(r"$\omega$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Two-sided Lorentzian pairs, noise 1e-4: does the posterior cover the side "
                 "weights as they balance?", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- D --- #
def test_suite(flow, vae, noise=1e-3):
    g = torch.Generator().manual_seed(5)
    rows, curves = {}, {}
    for name in C.CASES:
        inst = C.draw(C.realize(name), noise, 7)
        tr = inst.truth
        ps = flow.sample(inst.G, tr.x, BETA, sigma=inst.sigma, n_samples=S, generator=g)
        A = A_phys(ps).numpy()
        At = C.truth_on_grid(name, W)
        c2 = chi2(ps, tr.quad, tr.x, inst.G.expand(S, -1), inst.sigma.expand(S, -1), inst.isr)
        lo, hi = np.quantile(A, [0.05, 0.95], axis=0)
        core = np.abs(W) <= 8
        am = A_phys(vae_mean(vae, inst.G, tr.x, BETA, inst.sigma)).numpy()[0]
        rows[name] = {"flow_w1_mean": w1(A.mean(0), At), "vae_amortized_w1": w1(am, At),
                      "flow_band90_cover": float(((At >= lo - 1e-9) & (At <= hi + 1e-9))[core].mean()),
                      "flow_chi2_median": float(c2.median()), "flow_chi2_best": float(c2.min())}
        curves[name] = (np.quantile(A, [0.05, 0.5, 0.95], axis=0), At, am)
    return rows, curves


def fig_suite(rows, curves, path, noise):
    names = list(rows)
    fig, axes = plt.subplots(3, 5, figsize=(15, 7.5), constrained_layout=True)
    core = np.abs(W) <= 6
    for ax, n in zip(axes.flat, names):
        q, At, am = curves[n]
        ax.fill_between(W[core], q[0][core], q[2][core], color=COL["flow"], alpha=0.25, lw=0)
        ax.plot(W[core], q[1][core], color=COL["flow"], lw=1.2, label="flow median + 90% band")
        ax.plot(W[core], am[core], color=COL["vae"], lw=1.0, label="VAE amortized")
        ax.plot(W[core], At[core], color=INK, lw=1.5, ls="--", label="truth")
        r = rows[n]
        ax.set_title(f"{n}: W1 flow {r['flow_w1_mean']:.3f} / VAE {r['vae_amortized_w1']:.3f}",
                     fontsize=8)
        ax.tick_params(labelleft=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3)
    fig.suptitle(f"Frozen suite truths at noise {noise:g} (seed 7): flow posterior vs VAE "
                 "one-pass output; smooth truths are outside both models' training prior",
                 fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def test_holstein(flow, vae):
    import math as _m
    mg = C.holstein_cell()
    cov = mg.covariance(1e-10, method="structured")
    sl = cov["slices"]
    G = mg.mean.float().unsqueeze(0)
    sig_enc = (mg.sigma_encoder / _m.sqrt(mg.n_bins)).float().unsqueeze(0)
    x = mg.x.float()
    ps = flow.sample(G, x, mg.beta, sigma=sig_enc, n_samples=200,
                     generator=torch.Generator().manual_seed(3))
    quad = SpectralQuadrature.for_beta(mg.beta)
    G_rec = PoleGreens(quad)(ps, x).double()
    s = cov["sigma"].double() / _m.sqrt(mg.n_bins)
    d = ((G_rec[:, sl] - mg.mean.double()[sl]) / s) @ cov["inv_sqrt_R"].double().T
    c2 = (d * d).sum(-1) / cov["n_independent"]
    A = A_phys(ps, beta=mg.beta).numpy()
    return {"chi2_over_target_q05_q50_q95": (np.quantile(c2.numpy(), [0.05, 0.5, 0.95])
                                             / cov["chi2_target"]).tolist(),
            "best_chi2_over_target": float(c2.min() / cov["chi2_target"])}, \
        (np.quantile(A, [0.05, 0.5, 0.95], axis=0), mg.reference)


def fig_holstein(res, curves, path):
    q, ref = curves
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    core = np.abs(W) <= 6
    ax.fill_between(W[core], q[0][core], q[2][core], color=COL["flow"], alpha=0.25, lw=0,
                    label="flow 90% band")
    ax.plot(W[core], q[1][core], color=COL["flow"], lw=1.3, label="flow median")
    if ref is not None:
        w, A = (np.asarray(v, float) for v in ref)
        m = np.abs(w) <= 6
        ax.plot(w[m], A[m], color=INK, lw=1.1, label="SmoQyDEAC (shipped)")
    r = res["chi2_over_target_q05_q50_q95"]
    ax.set_title(f"Holstein cell: flow posterior (200 samples), chi2/target median {r[1]:.3g} "
                 f"[{r[0]:.3g}, {r[2]:.3g}]", fontsize=8.5)
    ax.set_xlabel(r"$\omega$")
    ax.legend(fontsize=7.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    flow_path, vae_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    flow, vae = load_flow(flow_path).eval(), load_model(vae_path).eval()
    report = {"flow": flow_path, "vae": vae_path, "flow_params": flow.num_parameters(),
              "vae_params": vae.num_parameters(), "samples_per_input": S}
    print("A+B calibration / data consistency ...", flush=True)
    summ, raw = test_calibration(flow, vae)
    report["calibration"] = summ
    fig_sbc(raw, summ, out / "sbc.png")
    print(json.dumps(summ, indent=1), flush=True)
    print("C two-sided pairs ...", flush=True)
    rows, curves = test_pairs(flow, vae)
    report["pairs"] = rows
    fig_pairs(rows, curves, out / "pairs.png")
    for nz in (1e-3, 1e-4):
        print(f"D suite at {nz:g} ...", flush=True)
        srows, scurves = test_suite(flow, vae, nz)
        report[f"suite@{nz:g}"] = srows
        fig_suite(srows, scurves, out / f"suite_{nz:g}.png", nz)
    print("D Holstein ...", flush=True)
    hres, hcurves = test_holstein(flow, vae)
    report["holstein"] = hres
    fig_holstein(hres, hcurves, out / "holstein.png")
    with open(out / "judge.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"-> {out}/judge.json + figures")


if __name__ == "__main__":
    main()
