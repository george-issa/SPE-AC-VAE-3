"""Is the refinement's non-convergence at low noise an optimiser problem?

`refine_samples` minimises, per flow sample,

    F(u) = 1/2 chi^2_sum(u; G~) + lam/2 |u - u_k|^2          (G~ = redrawn data)

with 3000 Adam steps; at relative noise 3e-5 the samples end at chi^2 ~ 5-15
per point.  F is a nonlinear LEAST-SQUARES problem (L whitened residuals + the
tether, 48 unknowns), so this script solves the SAME problem -- same flow
samples, same redrawn targets, same tether, same fixed pole count -- with a
batched Levenberg-Marquardt iteration (analytic Jacobian of the pole spectrum,
float64, the same error-bar continuation) and compares against the Adam result
stored by `stratified_sbc.py` (arm `str`, identical seeds).

    python scripts/flow/lm_refine_test.py out/flow/F_60k.pt out/flow/review_20261124
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/flow")

import numpy as np
import torch

import stratified_sbc as SS

LEVELS = [0, 1, 3]                       # indices into SS.NOISES: 3e-5, 1e-4, 1e-3
DT = torch.float64


def lm_refine(flow, z0, norm, beta, x, target, sigma, W, lam, anneal_from=1e-2,
              stages=6, iters_stage=6, iters_final=40):
    """Batched Levenberg-Marquardt on F.  Returns (PoleSet, u, F, iterations)."""
    from spevae.physics import PoleGreens, SpectralQuadrature
    from spevae.physics.poles import PoleSet
    quad = SpectralQuadrature.for_beta(beta, dtype=DT)
    nodes, wts = quad.nodes, quad.weights
    Kw = wts[:, None] * PoleGreens(quad).kernel_matrix(x.to(DT))          # (M, L)
    c = flow.coords
    mean, std, gmin, D = c.mean.to(DT), c.std.to(DT), c.gamma_min, c.D
    nu_max = flow.nu_max(beta)
    pres = z0[..., 4]
    present = pres > 0
    none = ~present.any(-1, keepdim=True)
    top = torch.zeros_like(present).scatter_(-1, pres.argmax(-1, keepdim=True), True)
    present = present | (none & top)                                      # (N, P), as in decode
    keep = present.to(DT)
    u0 = z0[..., :4].to(DT)
    nrm = norm.to(DT).reshape(-1, 1)
    N, P = present.shape
    W, target, sig = W.to(DT), target.to(DT), sigma.to(DT)

    def theta(u):
        r = u * std + mean
        eps = (math.pi * torch.sinh(r[..., 0].clamp(-12.0, 12.0))).clamp(-nu_max, nu_max)
        gam = gmin + torch.exp(r[..., 1].clamp(-12.0, 8.0))
        a = torch.softmax(torch.where(present, r[..., 2], torch.full_like(r[..., 2], -float("inf"))), -1)
        d = torch.where(present, D * torch.tanh(r[..., 3]), torch.zeros_like(r[..., 3]))
        b = d - d.sum(-1, keepdim=True) * a
        return torch.stack([eps, gam, a * nrm, b * nrm], -1)              # (N, P, 4)

    def greens(th, jac=False):
        eps, gam, a, b = (th[..., k].unsqueeze(-1) for k in range(4))    # (N, P, 1)
        dl = nodes.reshape(1, 1, -1) - eps                                # (N, P, M)
        den = dl * dl + gam * gam
        num = a * gam - b * dl
        G = ((num / den).sum(1) / math.pi) @ Kw                           # (N, L)
        if not jac:
            return G, None
        dA = torch.stack([(b / den + 2.0 * num * dl / den ** 2),          # d/d eps
                          (a / den - 2.0 * num * gam / den ** 2),         # d/d gamma
                          gam / den,                                      # d/d a
                          -dl / den], 2) / math.pi                        # d/d b   -> (N, P, 4, M)
        return G, (dA.reshape(N, P * 4, -1) @ Kw).transpose(1, 2)         # (N, L, 4P)

    def residual(u, fac, jac=False):
        th = theta(u)
        G, JG = greens(th, jac)
        rd = ((G - target) / (sig * fac)) @ W.T                           # (N, L)
        rt = math.sqrt(lam) * (keep.unsqueeze(-1) * (u - u0)).reshape(N, -1)
        if not jac:
            return rd, rt, None
        # d theta / d u, per sample: (4P, 4P) through softmax / gauge
        uu = u.detach().requires_grad_(True)
        tt = theta(uu).reshape(N, -1)
        Jt = torch.stack([torch.autograd.grad(tt[:, j].sum(), uu, retain_graph=True)[0].reshape(N, -1)
                          for j in range(tt.shape[1])], 1)                # (N, 4P, 4P)
        Jd = ((JG @ Jt) / (sig * fac).unsqueeze(-1)).transpose(1, 2) @ W.T  # (N, 4P, L)
        return rd, rt, Jd.transpose(1, 2)                                 # (N, L, 4P)

    med = sig.median(-1).values
    start = (anneal_from / med).clamp_min(1.0)
    sched = [start ** (1.0 - k / stages) for k in range(stages) for _ in range(iters_stage)]
    sched += [torch.ones_like(start)] * iters_final
    u = u0.clone()
    mu = torch.full((N,), 1e-2, dtype=DT)
    kd = keep.unsqueeze(-1).expand(-1, -1, 4).reshape(N, -1)              # (N, 4P) live coordinates
    eye_dead = torch.diag_embed(1.0 - kd)
    best_F = torch.full((N,), float("inf"), dtype=DT)
    best_u = u0.clone()
    for it, f in enumerate(sched):
        fac = f.unsqueeze(-1)
        with torch.enable_grad():
            rd, rt, Jd = residual(u, fac, jac=True)
        rd, rt, Jd = rd.detach(), rt.detach(), Jd.detach()
        F = 0.5 * ((rd * rd).sum(-1) + (rt * rt).sum(-1))
        A = Jd.transpose(1, 2) @ Jd + lam * torch.diag_embed(kd)
        g = (Jd.transpose(1, 2) @ rd.unsqueeze(-1)).squeeze(-1) + math.sqrt(lam) * rt
        dA = torch.diagonal(A, dim1=1, dim2=2).clamp_min(1e-12)
        step = torch.linalg.solve(A + mu.reshape(-1, 1, 1) * torch.diag_embed(dA * kd) + eye_dead,
                                  -(g * kd).unsqueeze(-1)).squeeze(-1)
        un = u + step.reshape(N, P, 4)
        rd2, rt2, _ = residual(un, fac)
        F2 = 0.5 * ((rd2 * rd2).sum(-1) + (rt2 * rt2).sum(-1))
        ok = torch.isfinite(F2) & (F2 < F)
        u = torch.where(ok.reshape(-1, 1, 1), un, u)
        mu = torch.where(ok, mu / 3.0, mu * 4.0).clamp(1e-10, 1e10)
        if float(f.max()) == 1.0:                                         # full-precision stage
            Fc = torch.where(ok, F2, F)
            better = Fc < best_F
            best_F = torch.where(better, Fc, best_F)
            best_u[better] = u[better]
    th = theta(best_u)
    ps = PoleSet(eps=th[..., 0], gamma=th[..., 1], a=th[..., 2] * keep, b=th[..., 3] * keep)
    return ps, best_u, best_F, len(sched)


def main():
    import judge_flow as J
    import judge_str as JS
    from spevae.data.pole_prior import PolePriorGenerator
    from spevae.models.flow import load_flow
    from spevae.physics.poles import PoleSet

    flow_path, out = sys.argv[1], Path(sys.argv[2])
    torch.set_num_threads(6)
    flow = load_flow(flow_path).eval()
    M, S = SS.M, SS.S
    adam = []
    for p in sorted(out.glob("stratified_sbc.shard*.json")):
        adam += [r for r in json.load(open(p)) if r["arm"] == "str"]
    rows = []
    for i in LEVELS:
        nz = SS.NOISES[i]
        gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive",
                                 beta_range=(SS.BETA, SS.BETA), dtau_choices=(SS.DTAUS[i % len(SS.DTAUS)],),
                                 noise_scale_range=(nz, nz))
        with gen.fixed_scalars(7000 + i):
            b = gen.batch(M, torch.Generator().manual_seed(7100 + i))
        truth = PoleSet(b.poles_true["eps"], b.poles_true["gamma"], b.poles_true["a"], b.poles_true["b"])
        At = J.A_phys(truth)
        st = J.stats_of(At)
        # the same flow samples and the same redrawn targets as the stored Adam run
        g = torch.Generator().manual_seed(40 + i)
        z, norm = flow.sample_coords(b.G, b.x, SS.BETA, sigma=b.sigma, n_samples=S, generator=g)
        Gr, sr = b.G.repeat_interleave(S, 0), b.sigma.repeat_interleave(S, 0)
        xi = torch.randn(Gr.shape, generator=g)
        target = Gr + sr * (xi @ torch.linalg.inv(b.inv_sqrt_R).T)
        t0 = time.time()
        ps, _, _, n_it = lm_refine(flow, z, norm, SS.BETA, b.x, target, sr, b.inv_sqrt_R, 1.0)
        secs = time.time() - t0
        c2 = J.chi2(ps, b.quad, b.x, Gr, sr, b.inv_sqrt_R).reshape(M, S).numpy()
        c2own = J.chi2(ps, b.quad, b.x, target, sr, b.inv_sqrt_R).reshape(M, S).numpy()
        A = J.A_phys(ps)
        ss = J.stats_of(A).reshape(M, S, -1)
        less, ties = (ss < st[:, None, :]).sum(1), (ss == st[:, None, :]).sum(1)
        pit = (less + np.random.default_rng(900 + i).random(less.shape) * (ties + 1)) / (S + 1)
        Am = A.reshape(M, S, -1).numpy().mean(1)
        w1 = [J.w1(Am[k], At[k].numpy()) for k in range(M)]
        ra = next(r for r in adam if r["noise"] == nz)
        ca = np.array(ra["chi2_per_sample"])
        pa = np.array(ra["pit"])
        row = {"noise": nz, "n_tau": int(b.x.numel()), "lm_iterations": n_it,
               "lm": {"seconds": secs, "chi2_vs_data_q50_q90": np.quantile(c2, [.5, .9]).tolist(),
                      "frac_chi2_above_2": float((c2 > 2).mean()),
                      "chi2_vs_own_target_q50_q90": np.quantile(c2own, [.5, .9]).tolist(),
                      "pit_cov50": float(np.mean((pit > .25) & (pit < .75))),
                      "pit_cov90": float(np.mean((pit > .05) & (pit < .95))),
                      "w1_median": float(np.median(w1)), "chi2_per_sample": c2.tolist()},
               "adam": {"seconds": ra["seconds"], "chi2_vs_data_q50_q90": np.quantile(ca, [.5, .9]).tolist(),
                        "frac_chi2_above_2": float((ca > 2).mean()),
                        "pit_cov50": float(np.mean((pa > .25) & (pa < .75))),
                        "pit_cov90": float(np.mean((pa > .05) & (pa < .95))),
                        "w1_median": float(np.median(ra["w1_per_truth"])), "chi2_per_sample": ca.tolist()},
               "n_present": (ps.a.reshape(M, S, -1) > 0).sum(-1).tolist(),
               "n_true": b.poles_true["n_true"].tolist()}
        rows.append(row)
        for k in ("adam", "lm"):
            r = row[k]
            print(f"noise {nz:7.0e} {k:5s} chi2 vs data median {r['chi2_vs_data_q50_q90'][0]:7.3f} "
                  f"q90 {r['chi2_vs_data_q50_q90'][1]:8.3f}  frac>2 {r['frac_chi2_above_2']:.2f}  "
                  f"PIT cov50 {r['pit_cov50']:.2f} cov90 {r['pit_cov90']:.2f}  W1 {r['w1_median']:.4f}  "
                  f"({r['seconds']:.0f}s)", flush=True)
        with open(out / "lm_refine_test.json", "w") as fh:
            json.dump(rows, fh)
    fig(rows, out / "lm_refine_test.png")


def fig(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"adam": "#1baf7a", "lm": "#eb6834"}
    LAB = {"adam": "Adam, 3000 steps (current)", "lm": "Levenberg-Marquardt"}
    fig_, axes = plt.subplots(1, len(rows) + 1, figsize=(4.6 * (len(rows) + 1), 3.8), constrained_layout=True)
    for ax, r in zip(axes, rows):
        for k in ("adam", "lm"):
            c = np.sort(np.array(r[k]["chi2_per_sample"]).ravel())
            ax.plot(c, (np.arange(len(c)) + 0.5) / len(c), color=COL[k], lw=1.6,
                    label=f"{LAB[k]}: median {r[k]['chi2_vs_data_q50_q90'][0]:.2f}, {r[k]['seconds']:.0f}s")
        ax.axvline(1.0, color="#52514e", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("sample $\\chi^2$ per point vs the measured data")
        ax.set_ylabel("fraction of samples below")
        ax.set_title(f"noise {r['noise']:g} ({r['n_tau']} slices)", fontsize=9.5)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax = axes[-1]
    xs = np.arange(len(rows))
    for j, k in enumerate(("adam", "lm")):
        ax.bar(xs + (j - 0.5) * 0.36, [r[k]["w1_median"] for r in rows], 0.34, color=COL[k], label=LAB[k])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['noise']:g}" for r in rows])
    ax.set_xlabel("relative noise level")
    ax.set_title("W$_1$ of the posterior mean (median over truths)", fontsize=9.5)
    ax.set_ylim(0, 1.3 * max(r[k]["w1_median"] for r in rows for k in ("adam", "lm")))
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for ax in axes:
        ax.grid(color="#d9d8d4", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig_.suptitle("Same flow samples, same redrawn targets, same objective: Adam vs a least-squares solver "
                  f"({SS.M} truths x {SS.S} samples per level; Adam timed with 4 threads under load, "
                  "115-140 s standalone; LM with 6 threads)", fontsize=10)
    fig_.savefig(path, dpi=150)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
