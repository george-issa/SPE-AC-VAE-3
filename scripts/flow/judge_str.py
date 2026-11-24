"""Judge sample-then-refine (models/flow.refine_samples).

Phase "scan": per noise level (1e-2, 1e-3, 1e-4), 32 held-out prior draws x 32
samples, the raw flow vs refine at tether lam in {1, 30, 300, 3000}, with and
without a per-sample noise redraw.  Metrics: sample chi^2 (median, against the
unperturbed data), 50% / 90% interval coverage of 9 spectral statistics,
pointwise 90% band coverage of A(w), W1 of the posterior mean.  Calibrated =
coverage at its nominal level at EVERY noise level with chi^2 near the truth's.

Phase "final LAM PERTURB": the chosen setting on 128 mixed-noise prior draws
(SBC ranks), the two-sided pairs, the 14 suite truths and the Holstein cell,
next to the raw flow and the VAE control.

    python scripts/flow/judge_str.py scan  out/flow/F_6k.pt out/flow/str_F6k
    python scripts/flow/judge_str.py final out/flow/F_6k.pt out/flow/str_F6k 30 1
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/flow")

import numpy as np
import torch

import judge_flow as J
from spevae import load_model
from spevae.data.pole_prior import PolePriorGenerator
from spevae.evaluation import cases as C
from spevae.models.flow import load_flow, refine_samples
from spevae.physics.poles import PoleSet

BETA = 8.0
#: refinement schedule of the judgements banked 2026-11-23/24 (chosen by the
#: budget scan, see TRAINING.md): 3000 Adam steps, cosine lr 0.05 -> 0.005, error
#: bars annealed from an absolute 1e-2 down to the data's own.  Pinned to Adam so
#: this script keeps reproducing those numbers.
REFINE = dict(steps=3000, lr_final=0.005, anneal_from=1e-2, solver="adam")
#: the least-squares solver that replaced it (2026-11-24): same continuation,
#: Levenberg-Marquardt instead of Adam -- see models/flow.refine_samples
REFINE_LM = dict(anneal_from=1e-2, solver="lm")


def posterior(flow, G, x, sigma, isr, S, lam, perturb, g, x_fit=None, G_fit=None,
              s_fit=None):
    """Flow samples, optionally refined.  G/sigma for the encoder; *_fit for the
    likelihood (defaults: the same grid)."""
    z, norm = flow.sample_coords(G, x, BETA if x_fit is None else BETA, sigma=sigma,
                                 n_samples=S, generator=g)
    if lam is None:
        return flow.coords.decode(z, norm=norm, nu_max=flow.nu_max(BETA))
    xf = x if x_fit is None else x_fit
    Gf = (G if G_fit is None else G_fit).repeat_interleave(S, 0)
    sf = (sigma if s_fit is None else s_fit).repeat_interleave(S, 0)
    ps, _ = refine_samples(flow, z, norm, BETA, xf, Gf, sf, isr, lam, perturb=perturb,
                           generator=g, **REFINE)
    return ps


def metrics(ps, b, S, truth):
    At = J.A_phys(truth)
    st = J.stats_of(At)
    B = b.G.shape[0]
    A = J.A_phys(ps)
    ss = J.stats_of(A).reshape(B, S, -1)
    c2 = J.chi2(ps, b.quad, b.x, b.G.repeat_interleave(S, 0),
                b.sigma.repeat_interleave(S, 0), b.inv_sqrt_R).reshape(B, S)
    A = A.reshape(B, S, -1).numpy()
    q = np.quantile(ss, [0.25, 0.75, 0.05, 0.95], axis=1)
    cov50 = float(((st >= q[0]) & (st <= q[1])).mean())
    cov90 = float(((st >= q[2]) & (st <= q[3])).mean())
    core = np.abs(J.W) <= 8
    lo, hi = np.quantile(A[:, :, core], [0.05, 0.95], axis=1)
    band = float(((At.numpy()[:, core] >= lo - 1e-9) & (At.numpy()[:, core] <= hi + 1e-9)).mean())
    w1 = float(np.median([J.w1(A[i].mean(0), At[i].numpy()) for i in range(B)]))
    ranks = ((ss < st[:, None, :]).sum(1)).tolist()
    return {"chi2_median": float(c2.median()), "cover50": cov50, "cover90": cov90,
            "band90": band, "w1_mean": w1}, ranks


def scan(flow_path, out):
    flow = load_flow(flow_path).eval()
    S, M = 32, 32
    configs = [("raw", None, False)] + [(f"lam{l:g}{'+p' if p else ''}", l, p)
                                        for l in (1.0, 30.0, 300.0, 3000.0) for p in (False, True)]
    rows = []
    for nz in (1e-2, 1e-3, 1e-4):
        gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive",
                                 beta_range=(BETA, BETA), dtau_choices=(0.125,),
                                 noise_scale_range=(nz, nz))
        with gen.fixed_scalars(4040):
            b = gen.batch(M, torch.Generator().manual_seed(4041))
        truth = PoleSet(b.poles_true["eps"], b.poles_true["gamma"], b.poles_true["a"],
                        b.poles_true["b"])
        tc2 = float(J.chi2(truth, b.quad, b.x, b.G, b.sigma, b.inv_sqrt_R).median())
        for name, lam, pert in configs:
            g = torch.Generator().manual_seed(17)
            ps = posterior(flow, b.G, b.x, b.sigma, b.inv_sqrt_R, S, lam, pert, g)
            m, _ = metrics(ps, b, S, truth)
            m.update(noise=nz, config=name, truth_chi2=tc2)
            rows.append(m)
            print(f"{nz:7.0e} {name:12s} chi2 {m['chi2_median']:9.3g} (truth {tc2:.2f})  "
                  f"cov50 {m['cover50']:.2f} cov90 {m['cover90']:.2f} band90 {m['band90']:.2f}  "
                  f"W1 {m['w1_mean']:.4f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scan.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    fig_scan(rows, out / "scan.png")


def fig_scan(rows, path):
    import matplotlib.pyplot as plt
    configs = list(dict.fromkeys(r["config"] for r in rows))
    noises = sorted({r["noise"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, nz in enumerate(noises):
        rr = [next(r for r in rows if r["config"] == c and r["noise"] == nz) for c in configs]
        xs = np.arange(len(configs)) + (k - 1) * 0.22
        axes[0].plot(xs, [r["cover90"] for r in rr], "o", color=cols[k], label=f"noise {nz:g}")
        axes[1].plot(xs, [r["band90"] for r in rr], "o", color=cols[k])
        axes[2].plot(xs, [r["chi2_median"] for r in rr], "o", color=cols[k])
    for ax, t in zip(axes, ["90% interval coverage (9 statistics)", "pointwise 90% band coverage",
                            "median sample chi2 (truth ~1)"]):
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=60, fontsize=7)
        ax.set_title(t)
    axes[0].axhline(0.9, color="#52514e", ls="--", lw=0.9)
    axes[1].axhline(0.9, color="#52514e", ls="--", lw=0.9)
    axes[2].set_yscale("log")
    axes[2].axhline(1.0, color="#52514e", ls="--", lw=0.9)
    axes[0].legend(fontsize=7)
    fig.suptitle("Sample-then-refine: tether strength lam (+p = per-sample noise redraw) "
                 "vs calibration, per noise level", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def final(flow_path, out, lam, pert, vae_path="out/step2/equiv_ctrl6k.pt"):
    flow = load_flow(flow_path).eval()
    vae = load_model(vae_path).eval()
    S, M, bs = 32, 96, 32
    res = {"lam": lam, "perturb": pert}
    # SBC on mixed-noise prior draws (the training distribution)
    gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive",
                             beta_range=(BETA, BETA))
    ranks = {"str": [], "raw": [], "vae": []}
    agg = {k: [] for k in ranks}
    with gen.fixed_scalars(5050):
        for k in range(M // bs):
            b = gen.batch(bs, torch.Generator().manual_seed(5051 + k))
            truth = PoleSet(b.poles_true["eps"], b.poles_true["gamma"], b.poles_true["a"],
                            b.poles_true["b"])
            g = torch.Generator().manual_seed(100 + k)
            for name in ranks:
                if name == "vae":
                    ps = J.vae_samples(vae, b.G, b.x, BETA, b.sigma, S, g)
                else:
                    ps = posterior(flow, b.G, b.x, b.sigma, b.inv_sqrt_R, S,
                                   lam if name == "str" else None, pert, g)
                m, r = metrics(ps, b, S, truth)
                ranks[name] += r
                agg[name].append(m)
            print(f"  SBC batch {k + 1}/{M // bs}", flush=True)
    res["sbc"] = {}
    for name in ranks:
        R = np.array(ranks[name]) / S
        ks = [float(np.max(np.abs(np.sort(R[:, j]) - np.arange(1, len(R) + 1) / len(R))))
              for j in range(R.shape[1])]
        res["sbc"][name] = {"ks_max": max(ks), "ks_crit_5pct": 1.36 / math.sqrt(len(R)),
                            **{k: float(np.mean([m[k] for m in agg[name]]))
                               for k in ("cover50", "cover90", "band90", "w1_mean")},
                            "chi2_median": float(np.median([m["chi2_median"] for m in agg[name]]))}
        print(name, res["sbc"][name], flush=True)
    fig_ranks(ranks, S, res["sbc"], out / "final_sbc.png")
    # suite truths + Holstein with bands
    for nz in (1e-3, 1e-4):
        rows, curves = {}, {}
        for case in C.CASES:
            inst = C.draw(C.realize(case), nz, 7)
            tr = inst.truth
            g = torch.Generator().manual_seed(9)
            ps = posterior(flow, inst.G, tr.x, inst.sigma, inst.isr, S, lam, pert, g)
            A = J.A_phys(ps).numpy()
            At = C.truth_on_grid(case, J.W)
            c2 = J.chi2(ps, tr.quad, tr.x, inst.G.expand(S, -1), inst.sigma.expand(S, -1), inst.isr)
            lo, hi = np.quantile(A, [0.05, 0.95], axis=0)
            core = np.abs(J.W) <= 8
            am = J.A_phys(J.vae_mean(vae, inst.G, tr.x, BETA, inst.sigma)).numpy()[0]
            rows[case] = {"str_w1_mean": J.w1(A.mean(0), At), "vae_amortized_w1": J.w1(am, At),
                          "band90_cover": float(((At >= lo - 1e-9) & (At <= hi + 1e-9))[core].mean()),
                          "chi2_median": float(c2.median())}
            curves[case] = (np.quantile(A, [0.05, 0.5, 0.95], axis=0), At, am)
        res[f"suite@{nz:g}"] = rows
        J.fig_suite({k: {"flow_w1_mean": v["str_w1_mean"], "vae_amortized_w1": v["vae_amortized_w1"]}
                     for k, v in rows.items()}, curves, out / f"final_suite_{nz:g}.png", nz)
        print(f"suite@{nz:g}", json.dumps(rows)[:400], flush=True)
    mg = C.holstein_cell()
    cov = mg.covariance(1e-10, method="structured")
    sl = cov["slices"]
    Gfull = mg.mean.float().unsqueeze(0)
    s_enc = (mg.sigma_encoder / math.sqrt(mg.n_bins)).float().unsqueeze(0)
    s_fit = (cov["sigma"] / math.sqrt(mg.n_bins)).float().unsqueeze(0)
    g = torch.Generator().manual_seed(3)
    z, norm = flow.sample_coords(Gfull, mg.x.float(), mg.beta, sigma=s_enc, n_samples=200,
                                 generator=g)
    ps, c2 = refine_samples(flow, z, norm, mg.beta, mg.x.float()[sl],
                            Gfull[:, sl].expand(200, -1), s_fit.expand(200, -1),
                            cov["inv_sqrt_R"].float(), lam, perturb=pert, generator=g,
                            **REFINE)
    A = J.A_phys(ps, beta=mg.beta).numpy()
    res["holstein"] = {"chi2_over_target_q05_q50_q95":
                       (np.quantile(c2.numpy(), [0.05, 0.5, 0.95]) / cov["chi2_target"]).tolist()}
    J.fig_holstein(res["holstein"], (np.quantile(A, [0.05, 0.5, 0.95], axis=0), mg.reference),
                   out / "final_holstein.png")
    print("holstein", res["holstein"], flush=True)
    with open(out / "final.json", "w") as fh:
        json.dump(res, fh, indent=1)


def fig_ranks(ranks, S, summ, path):
    import matplotlib.pyplot as plt
    names = list(ranks)
    fig, axes = plt.subplots(len(names), 9, figsize=(16, 2.1 * len(names)), sharey=True,
                             constrained_layout=True)
    cols = {"str": "#1baf7a", "raw": "#2a78d6", "vae": "#eb6834"}
    for i, n in enumerate(names):
        R = np.array(ranks[n])
        for j, st in enumerate(J.STATS):
            ax = axes[i, j]
            ax.hist(R[:, j], bins=8, range=(0, S + 1), color=cols[n], rwidth=0.9)
            ax.axhline(len(R) / 8, color="#52514e", ls="--", lw=0.9)
            ax.set_xticks([])
            if i == 0:
                ax.set_title(st, fontsize=7.5)
        axes[i, 0].set_ylabel(f"{n}\nKS {summ[n]['ks_max']:.2f}")
    fig.suptitle("SBC ranks, 96 mixed-noise prior draws: refined flow (str), raw flow, VAE "
                 "(flat = calibrated)", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(6)
    mode, flow_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    if mode == "scan":
        scan(flow_path, out)
    else:
        final(flow_path, out, float(sys.argv[4]), bool(int(sys.argv[5])),
              *(sys.argv[6:7]))
