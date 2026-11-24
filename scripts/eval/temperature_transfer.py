"""Do the beta = 8 checkpoints work at other temperatures?

Every checkpoint so far was trained at a single inverse temperature (beta = 8).
The kernel is beta-free in dimensionless variables, so nothing in the code
prevents running at another beta -- but the same physical spectrum occupies a
different region of the dimensionless prior there.  This script takes suite
truths (fixed in PHYSICAL units), builds them at several beta with the suite's
time step (dtau = 0.125), and scores three methods on the same noise draws:

  zdlm      VAE latent MAP by least squares (the D24 checkpoint)
  flow      refined flow posterior mean, 32 samples (F_60k)
  maxent    ana_cont MaxEnt with the exact noise covariance

    python scripts/eval/temperature_transfer.py out/eval/temperature_transfer \
        out/step2/data_arm/D24_60k.pt out/flow/F_60k.pt
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import torch

BETAS = (4.0, 8.0, 16.0)
CASES = ("lorentz pair", "3-rung ladder", "fano band", "gauss pair", "semicircle")
NOISE = 1e-3
SEEDS = (7, 8, 9, 10)
S = 32
COL = {"zdlm": "#eb6834", "flow": "#1baf7a", "maxent": "#52514e"}
LAB = {"zdlm": "VAE + least-squares MAP (D24)", "flow": "refined flow posterior mean (F_60k)",
       "maxent": "MaxEnt"}


def main():
    from spevae import load_model
    from spevae.evaluation import cases as C
    from spevae.evaluation import methods as M
    from spevae.losses import spectral_w1
    from spevae.models.flow import load_flow, refine_samples
    from spevae.physics import PoleGreens

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    vae, flow = load_model(sys.argv[2]), load_flow(sys.argv[3]).eval()
    rows = []
    for beta in BETAS:
        for case in CASES:
            tr = C.realize(case, beta=beta, ntau=int(round(beta / 0.125)))
            for seed in SEEDS:
                inst = C.draw(tr, NOISE, seed)
                for name in ("zdlm", "maxent", "flow"):
                    t0 = time.time()
                    if name == "flow":
                        g = torch.Generator().manual_seed(100 + seed)
                        z, norm = flow.sample_coords(inst.G, tr.x, beta, sigma=inst.sigma,
                                                     n_samples=S, generator=g)
                        ps, c2 = refine_samples(flow, z, norm, beta, tr.x, inst.G.expand(S, -1),
                                                inst.sigma.expand(S, -1), inst.isr, 1.0,
                                                perturb=True, generator=g, anneal_from=1e-2)
                        A = ps.spectral(tr.quad.nodes).mean(0, keepdim=True)
                        d = ((PoleGreens(tr.quad)(ps, tr.x) - inst.G) / inst.sigma) @ inst.isr.T
                        rec = {"w1": float(spectral_w1(A, tr.A, tr.quad, beta=beta)),
                               "chi2": float((d * d).mean(-1).median())}
                    else:
                        o = M.make_method(name, vae).synthetic(inst)
                        rec, _ = M.score_synthetic(o, inst)
                    rows.append({"beta": beta, "case": case, "seed": seed, "method": name,
                                 "w1": rec["w1"], "chi2": rec["chi2"], "n_tau": int(tr.x.numel()),
                                 "seconds": time.time() - t0})
            line = "  ".join(f"{m} {np.mean([r['w1'] for r in rows if r['beta'] == beta and r['case'] == case and r['method'] == m]):.3f}"
                             for m in ("zdlm", "flow", "maxent"))
            print(f"beta {beta:4.0f} {case:14s} W1: {line}", flush=True)
    with open(out / "temperature_transfer.json", "w") as fh:
        json.dump({"noise": NOISE, "seeds": SEEDS, "samples": S, "rows": rows}, fh)
    fig(rows, out / "temperature_transfer.png")


def fig(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_, axes = plt.subplots(1, len(CASES), figsize=(3.5 * len(CASES), 3.4), constrained_layout=True,
                              sharey=True)
    for ax, case in zip(axes, CASES):
        for m in ("zdlm", "flow", "maxent"):
            mu = [np.mean([r["w1"] for r in rows if r["beta"] == b and r["case"] == case and r["method"] == m])
                  for b in BETAS]
            lo = [np.min([r["w1"] for r in rows if r["beta"] == b and r["case"] == case and r["method"] == m])
                  for b in BETAS]
            hi = [np.max([r["w1"] for r in rows if r["beta"] == b and r["case"] == case and r["method"] == m])
                  for b in BETAS]
            ax.errorbar(BETAS, mu, yerr=[np.subtract(mu, lo), np.subtract(hi, mu)], fmt="o-",
                        color=COL[m], lw=1.3, ms=4, capsize=2, label=LAB[m])
        ax.axvline(8.0, color="#b9b8b4", ls=":", lw=1.0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(BETAS)
        ax.set_xticklabels([f"{b:g}" for b in BETAS])
        ax.set_xlabel(r"$\beta$")
        ax.set_title(case, fontsize=9.5)
        ax.grid(color="#d9d8d4", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("W$_1$ to the truth\n(mean, min-max over 4 draws)", fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    fig_.legend(h, l, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    fig_.suptitle(f"Same physical spectra at three temperatures, noise {NOISE:g}: models trained at "
                  r"$\beta = 8$ only (dotted line)", fontsize=10.5)
    fig_.savefig(path, dpi=150)
    print(f"-> {path}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--fig":          # re-render from the stored rows
        d = Path(sys.argv[1])
        fig(json.load(open(d / "temperature_transfer.json"))["rows"], d / "temperature_transfer.png")
    else:
        main()
