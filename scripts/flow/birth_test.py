"""Does letting the refinement ADD a pole rescue the samples that cannot fit?

The refinement keeps each sample's pole count.  After least squares, what is
left above the accept threshold is mostly samples with fewer poles than the
truth.  Here every rejected sample gets up to `ROUNDS` births: a light pole at
the position and width of a pole borrowed from a sample that fits the same
data (`models/flow.add_pole`), then a second least-squares pass from the
refined coordinates against the same redrawn target.

Same held-out truths, flow samples and redrawn targets as `stratified_sbc.py`
(small design); reported with and without births: accepted fraction, PIT
coverage and W1 of the accepted ensemble.

    python scripts/flow/birth_test.py out/flow/F_60k.pt out/flow/review_20261124
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

LEVELS = [0, 1, 2, 3]            # 3e-5, 1e-4, 3e-4, 1e-3
ROUNDS = 2


def main():
    import judge_flow as J
    from spevae.models.flow import add_pole, load_flow, refine_samples
    from spevae.physics.poles import PoleSet

    flow_path, out = sys.argv[1], Path(sys.argv[2])
    torch.set_num_threads(6)
    flow = load_flow(flow_path).eval()
    M, S = SS.M, SS.S
    rows = []
    for i in LEVELS:
        nz = SS.NOISES[i]
        b = SS.batch_for(i)
        truth = PoleSet(b.poles_true["eps"], b.poles_true["gamma"], b.poles_true["a"], b.poles_true["b"])
        At = J.A_phys(truth)
        st = J.stats_of(At)
        L = b.x.numel()
        edge = 1.0 + 3.0 * math.sqrt(2.0 / L)
        g = torch.Generator().manual_seed(40 + i)
        z, norm = flow.sample_coords(b.G, b.x, SS.BETA, sigma=b.sigma, n_samples=S, generator=g)
        Gr, sr = b.G.repeat_interleave(S, 0), b.sigma.repeat_interleave(S, 0)
        xi = torch.randn(Gr.shape, generator=g)
        target = Gr + sr * (xi @ torch.linalg.inv(b.inv_sqrt_R).T)
        t0 = time.time()
        ps, _, u = refine_samples(flow, z, norm, SS.BETA, b.x, target, sr, b.inv_sqrt_R, 1.0,
                                  anneal_from=1e-2, return_coords=True)
        chi = lambda p: J.chi2(p, b.quad, b.x, Gr, sr, b.inv_sqrt_R)
        c2 = chi(ps)
        zc = torch.cat([u, z[..., 4:5]], -1)                 # refined coordinates + presence
        state = {"none": (PoleSet(*(getattr(ps, k).clone() for k in ("eps", "gamma", "a", "b"))),
                          c2.clone())}
        n_births = torch.zeros(M * S, dtype=torch.long)
        for _ in range(ROUNDS):
            bad = c2 > edge
            if not bool(bad.any()):
                break
            z_new = zc.clone()
            did = torch.zeros(M * S, dtype=torch.bool)
            for m in range(M):                               # donors: samples of the SAME truth that fit
                sl = slice(m * S, (m + 1) * S)
                rows_bad = bad[sl]
                if not bool(rows_bad.any()):
                    continue
                good = ~rows_bad
                pool = PoleSet(*(getattr(ps, k)[sl][good if bool(good.any()) else slice(None)]
                                 for k in ("eps", "gamma", "a", "b")))
                zz, born = add_pole(flow, zc[sl], pool, generator=g)
                take = born & rows_bad
                z_new[sl][take] = zz[take]
                did[sl] = take
            if not bool(did.any()):
                break
            ps2, _, u2 = refine_samples(flow, z_new[did], norm[did], SS.BETA, b.x, target[did], sr[did],
                                        b.inv_sqrt_R, 1.0, lm_iters=60, return_coords=True)
            c_new = J.chi2(ps2, b.quad, b.x, Gr[did], sr[did], b.inv_sqrt_R)
            better = c_new < c2[did]                         # keep a birth only if it fits better
            idx = torch.nonzero(did).flatten()[better]
            for k in ("eps", "gamma", "a", "b"):
                getattr(ps, k)[idx] = getattr(ps2, k)[better].to(getattr(ps, k).dtype)
            zc[idx] = torch.cat([u2, z_new[did][..., 4:5]], -1)[better]
            c2[idx] = c_new[better]
            n_births[idx] += 1
        state["births"] = (ps, c2)
        secs = time.time() - t0
        row = {"noise": nz, "n_tau": int(L), "seconds": secs, "births_per_sample": float(n_births.float().mean())}
        for name, (p, c) in state.items():
            acc = (c <= edge).reshape(M, S).numpy()
            none = ~acc.any(1)
            acc[none] = True
            A = J.A_phys(p)
            ss = J.stats_of(A).reshape(M, S, -1)
            n_acc = acc.sum(1)[:, None]
            less = ((ss < st[:, None, :]) & acc[:, :, None]).sum(1)
            U = np.random.default_rng(900 + i).random(less.shape)
            pit = (less + U) / (n_acc + 1)
            Am = (A.reshape(M, S, -1).numpy() * acc[:, :, None]).sum(1) / n_acc
            w1 = [J.w1(Am[k], At[k].numpy()) for k in range(M)]
            n_pres = (p.a.reshape(M, S, -1) != 0).sum(-1).numpy()
            nt = b.poles_true["n_true"].numpy()[:, None]
            row[name] = {"frac_accepted": float((c <= edge).float().mean()),
                         "truths_with_no_accepted_sample": int(none.sum()),
                         "pit_cov50": float(np.mean((pit > .25) & (pit < .75))),
                         "pit_cov90": float(np.mean((pit > .05) & (pit < .95))),
                         "w1_median": float(np.median(w1)),
                         "chi2_median_all": float(c.median()),
                         "frac_fewer_poles_than_truth": float((n_pres < nt).mean()),
                         "p_count_correct": float((n_pres == nt).mean())}
            r = row[name]
            print(f"noise {nz:7.0e} {name:7s} accepted {r['frac_accepted']:.2f}  PIT {r['pit_cov50']:.2f}/"
                  f"{r['pit_cov90']:.2f}  W1 {r['w1_median']:.4f}  chi2(all) {r['chi2_median_all']:.2f}  "
                  f"too few poles {r['frac_fewer_poles_than_truth']:.2f}  count right "
                  f"{r['p_count_correct']:.2f}", flush=True)
        rows.append(row)
    with open(out / "birth_test.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    fig(rows, out / "birth_test.png")


def fig(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"none": "#eb6834", "births": "#0f8a8a"}
    LAB = {"none": "refine + accept (no births)", "births": f"refine + up to {ROUNDS} births + accept"}
    f, axes = plt.subplots(1, 4, figsize=(17, 3.6), constrained_layout=True)
    x = np.arange(len(rows))
    for ax, key, title in zip(axes, ("frac_accepted", "w1_median", "pit_cov50", "p_count_correct"),
                              ("fraction of samples that fit the data", "W$_1$ of the posterior mean (median)",
                               "50% interval coverage (calibrated = 0.5)", "samples with the true pole count")):
        for j, k in enumerate(("none", "births")):
            ax.bar(x + (j - 0.5) * 0.36, [r[k][key] for r in rows], 0.34, color=COL[k], label=LAB[k])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['noise']:g}" for r in rows])
        ax.set_xlabel("relative noise level")
        ax.set_title(title, fontsize=9.5)
        if key == "pit_cov50":
            ax.axhline(0.5, color="#52514e", ls="--", lw=0.9)
        ax.grid(color="#d9d8d4", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    f.suptitle(f"Pole births in the refinement ({SS.M} truths x {SS.S} samples per level; same flow samples "
               "and redrawn targets in both arms)", fontsize=10.5)
    f.savefig(path, dpi=150)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
