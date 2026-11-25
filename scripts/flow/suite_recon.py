"""Flow-posterior reconstructions of the 14 suite truths, with the current pipeline.

For each truth on the banked noise draw (seed 7): 64 flow samples, refined
with the least-squares solver against per-sample noise redraws, samples whose
chi^2 against the measured data is implausible for a posterior draw rejected
(above 1 + 3 sqrt(2/L) per point).  Plotted: the 90% band and median of the
accepted samples, the truth, and -- on the same draw, scored by the same W1 on
the same grid -- the VAE's least-squares latent MAP (`zdlm`) and MaxEnt.

    python scripts/flow/suite_recon.py out/flow/F_60k.pt out/step2/data_arm/D24_60k.pt \
        out/flow/review_20261124 [tether=10]

The banked run in out/flow/review_20261124 used tether 1.
"""
import json
import math
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/flow")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import judge_flow as J

TETHER = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0

BETA, S, SEED = 8.0, 64, 7
FLOW, VAE, ME, INK = "#2a78d6", "#eb6834", "#7d7c78", "#0b0b0b"


def run(flow, vae, noise):
    from spevae.evaluation import cases as C
    from spevae.evaluation import methods as M
    from spevae.models.flow import refine_samples
    rows, curves = {}, {}
    for case in C.CASES:
        inst = C.draw(C.realize(case), noise, SEED)
        tr = inst.truth
        L = tr.x.numel()
        At = C.truth_on_grid(case, J.W)
        t0 = time.time()
        g = torch.Generator().manual_seed(9)
        z, norm = flow.sample_coords(inst.G, tr.x, BETA, sigma=inst.sigma, n_samples=S, generator=g)
        ps, c2 = refine_samples(flow, z, norm, BETA, tr.x, inst.G.expand(S, -1),
                                inst.sigma.expand(S, -1), inst.isr, TETHER, perturb=True,
                                generator=g, anneal_from=1e-2)
        t_flow = time.time() - t0
        acc = (c2 <= 1.0 + 3.0 * math.sqrt(2.0 / L)).numpy()
        flagged = not acc.any()
        if flagged:
            acc[:] = True
        A = J.A_phys(ps).numpy()[acc]
        q = np.quantile(A, [0.05, 0.5, 0.95], axis=0)
        core = np.abs(J.W) <= 8
        zd = M.make_method("zdlm", vae).synthetic(inst)
        A_zd = J.A_phys(zd.poles).numpy()[0]
        me = M.make_method("maxent").synthetic(inst)
        A_me = np.interp(J.W, me.omega, me.A, left=0.0, right=0.0)
        zrec, _ = M.score_synthetic(zd, inst)
        mrec, _ = M.score_synthetic(me, inst)
        rows[case] = {
            "flow_w1_mean": J.w1(A.mean(0), At), "flow_w1_median": J.w1(q[1], At),
            "vae_zdlm_w1": J.w1(A_zd, At), "maxent_w1": J.w1(A_me, At),
            "flow_chi2_median": float(np.median(c2.numpy()[acc])),
            "vae_zdlm_chi2": zrec["chi2"], "maxent_chi2": mrec["chi2"],
            "flow_frac_accepted": float(acc.mean()), "flow_no_sample_fits": bool(flagged),
            "flow_band90_cover": float(((At >= q[0] - 1e-9) & (At <= q[2] + 1e-9))[core].mean()),
            "flow_poles_median": float(np.median((ps.a.numpy() > 0).sum(-1)[acc])),
            "flow_seconds": t_flow, "vae_seconds": zd.runtime, "maxent_seconds": me.runtime}
        curves[case] = (q, At, A_zd, A_me)
        r = rows[case]
        print(f"{noise:g} {case:17s} W1 flow {r['flow_w1_mean']:.4f}  VAE {r['vae_zdlm_w1']:.4f}  "
              f"MaxEnt {r['maxent_w1']:.4f} | flow chi2 {r['flow_chi2_median']:.2f}, accepted "
              f"{r['flow_frac_accepted']:.0%}, band covers {r['flow_band90_cover']:.0%}", flush=True)
    return rows, curves


def fig(rows, curves, path, noise):
    names = list(rows)
    f, axes = plt.subplots(3, 5, figsize=(17, 9.2), constrained_layout=True)
    core = np.abs(J.W) <= 6
    for ax, n in zip(axes.flat, names):
        q, At, A_zd, A_me = curves[n]
        ax.fill_between(J.W[core], q[0][core], q[2][core], color=FLOW, alpha=0.25, lw=0,
                        label="flow posterior, 90% band")
        ax.plot(J.W[core], q[1][core], color=FLOW, lw=1.4, label="flow posterior, median")
        ax.plot(J.W[core], A_zd[core], color=VAE, lw=1.0, label="VAE latent MAP (D24, least squares)")
        ax.plot(J.W[core], A_me[core], color=ME, lw=1.0, label="MaxEnt")
        ax.plot(J.W[core], At[core], color=INK, lw=1.5, ls="--", label="truth")
        r = rows[n]
        vals = {"flow": r["flow_w1_mean"], "VAE": r["vae_zdlm_w1"], "MaxEnt": r["maxent_w1"]}
        best = min(vals, key=vals.get)
        ax.set_title(n + "\n" + "   ".join(
            (f"[{k} {v:.3f}]" if k == best else f"{k} {v:.3f}") for k, v in vals.items()), fontsize=8.5)
        ax.set_ylim(bottom=min(0.0, float(q[0][core].min())) - 0.01)
        ax.tick_params(labelleft=False)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    f.legend(h, l, loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    wins = {k: sum(1 for r in rows.values()
                   if min(r["flow_w1_mean"], r["vae_zdlm_w1"], r["maxent_w1"]) == r[c])
            for k, c in (("flow", "flow_w1_mean"), ("VAE", "vae_zdlm_w1"), ("MaxEnt", "maxent_w1"))}
    f.suptitle(f"Suite truths at noise {noise:g}, banked draw (seed {SEED}): refined flow posterior "
               f"({S} samples; least-squares refine + redraw + accept/reject) next to the VAE and MaxEnt\n"
               f"Panel titles give W$_1$ to the truth, lowest in brackets: flow {wins['flow']}, "
               f"VAE {wins['VAE']}, MaxEnt {wins['MaxEnt']} of {len(rows)}", fontsize=10.5)
    f.savefig(path, dpi=150)
    plt.close(f)


def main():
    from spevae import load_model
    from spevae.models.flow import load_flow
    torch.set_num_threads(6)
    flow, vae, out = load_flow(sys.argv[1]).eval(), load_model(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    res = {"flow": sys.argv[1], "vae": sys.argv[2], "samples": S, "seed": SEED}
    for nz in (1e-4, 1e-3):
        rows, curves = run(flow, vae, nz)
        res[f"noise_{nz:g}"] = rows
        fig(rows, curves, out / f"suite_recon_{nz:g}.png", nz)
    with open(out / "suite_recon.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"-> {out}/suite_recon.json + suite_recon_0.0001.png, suite_recon_0.001.png")


if __name__ == "__main__":
    main()
