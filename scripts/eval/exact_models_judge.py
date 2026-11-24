"""Known-answer judge: exactly solvable models through the current pipeline.

Truths come from `spevae.data.exact_models` -- lattice Green's functions with
a lifetime (continuum) and exactly diagonalised finite models (delta spectra,
G(tau) in closed form).  None of them is in any current checkpoint's training
set.  Each gets the suite's noise model (correlated, xi = 0.1) on a
beta / dtau grid and is continued three ways on the same draw:

  flow      refined flow posterior (least-squares refine + redraw + accept)
  vae       VAE latent MAP by least squares (zdlm)
  maxent    ana_cont MaxEnt with the exact covariance

Score: W1 between cumulative spectral weights on a fine grid (exact step
functions for the delta spectra).

    python scripts/eval/exact_models_judge.py out/eval/exact_models \
        out/flow/F_60k.pt out/step2/data_arm/D24_60k.pt [noise=1e-3] [beta=8]
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

S, DTAU, SEED, PER_FAMILY = 64, 0.125, 7, 2
WG = np.linspace(-30.0, 30.0, 60001)
DW = WG[1] - WG[0]
FLOW, VAE, ME, INK = "#2a78d6", "#eb6834", "#7d7c78", "#0b0b0b"


def w1_cdf(cdf_a, cdf_b):
    return float(np.abs(cdf_a - cdf_b).sum() * DW)


def a_phys(ps, beta):
    nu = torch.tensor(WG * beta, dtype=torch.float64)
    p = type(ps)(*(getattr(ps, k).double() for k in ("eps", "gamma", "a", "b")))
    return (p.spectral(nu) * beta).numpy()


def make_truths(beta):
    from spevae.data import exact_models as X
    from spevae.evaluation.cases import Truth
    from spevae.physics import PoleGreens, SpectralQuadrature
    quad = SpectralQuadrature.for_beta(beta)
    x = torch.linspace(0.0, 1.0, int(round(beta / DTAU)) + 1)
    K = PoleGreens(quad).kernel_matrix(x)
    nu = quad.nodes.double().numpy()
    out = []
    rng = torch.Generator().manual_seed(2026)
    for fam in X.EXACT_CONTINUUM:
        for k in range(PER_FAMILY):
            tr = X.sample_exact_truth(fam, beta, rng)
            A = torch.tensor(tr.A(nu / beta) / beta, dtype=torch.float32)
            Gc = ((A * quad.weights) @ K).unsqueeze(0)
            Aw = tr.A(WG)
            out.append({"family": fam, "kind": "continuum", "params": tr.params, "A_grid": Aw,
                        "cdf": np.cumsum(Aw) * DW,
                        "truth": Truth(None, beta, x, quad, A.unsqueeze(0), Gc, None)})
    for fam in X.EXACT_DISCRETE:
        for k in range(PER_FAMILY):
            sp = X.sample_discrete_spectrum(fam, beta, rng)
            Gc = torch.tensor(sp.greens(x.double().numpy()), dtype=torch.float32).unsqueeze(0)
            out.append({"family": fam, "kind": "discrete", "params": sp.params, "spectrum": sp,
                        "cdf": sp.cdf(WG),
                        "truth": Truth(None, beta, x, quad, torch.zeros(1, quad.nodes.numel()), Gc, None)})
    return out


def run(flow, vae, noise, beta):
    from spevae.evaluation import cases as C
    from spevae.evaluation import methods as M
    from spevae.models.flow import refine_samples
    rows = []
    for item in make_truths(beta):
        inst = C.draw(item["truth"], noise, SEED)
        tr = inst.truth
        L = tr.x.numel()
        t0 = time.time()
        g = torch.Generator().manual_seed(9)
        z, norm = flow.sample_coords(inst.G, tr.x, beta, sigma=inst.sigma, n_samples=S, generator=g,
                                     inv_sqrt_R=inst.isr)
        ps, c2 = refine_samples(flow, z, norm, beta, tr.x, inst.G.expand(S, -1),
                                inst.sigma.expand(S, -1), inst.isr, 1.0, perturb=True,
                                generator=g, anneal_from=1e-2)
        acc = (c2 <= 1.0 + 3.0 * math.sqrt(2.0 / L)).numpy()
        none_fit = not acc.any()
        if none_fit:
            acc[:] = True
        A = a_phys(ps, beta)[acc]
        q = np.quantile(A, [0.05, 0.5, 0.95], axis=0)
        zd = M.make_method("zdlm", vae).synthetic(inst)
        A_zd = a_phys(zd.poles, beta)[0]
        me = M.make_method("maxent").synthetic(inst)
        A_me = np.interp(WG, me.omega, me.A, left=0.0, right=0.0)
        zc = M.score_synthetic(zd, inst)[0]["chi2"]
        mc = M.score_synthetic(me, inst)[0]["chi2"]
        cdf = lambda a: np.cumsum(a) * DW
        row = {"family": item["family"], "kind": item["kind"], "params": item["params"],
               "w1": {"flow": w1_cdf(cdf(A.mean(0)), item["cdf"]),
                      "vae": w1_cdf(cdf(A_zd), item["cdf"]),
                      "maxent": w1_cdf(cdf(A_me), item["cdf"])},
               "chi2": {"flow": float(np.median(c2.numpy()[acc])), "vae": zc, "maxent": mc},
               "flow_frac_accepted": float(acc.mean()), "flow_no_sample_fits": bool(none_fit),
               "seconds_flow": time.time() - t0}
        core = np.abs(WG) <= 8
        if item["kind"] == "continuum":
            At = item["A_grid"]
            row["flow_band90_cover"] = float(((At >= q[0] - 1e-9) & (At <= q[2] + 1e-9))[core].mean())
        rows.append((row, (q, A_zd, A_me, item)))
        w = row["w1"]
        print(f"{item['kind']:9s} {item['family']:16s} W1 flow {w['flow']:.4f} VAE {w['vae']:.4f} "
              f"MaxEnt {w['maxent']:.4f} | chi2 flow {row['chi2']['flow']:.2f} VAE {zc:.2f} MaxEnt {mc:.2f}"
              f" | accepted {row['flow_frac_accepted']:.0%}", flush=True)
    return rows


def fig(rows, kind, path, noise, beta):
    sel = [r for r in rows if r[0]["kind"] == kind]
    ncol = 5 if kind == "continuum" else 4
    nrow = math.ceil(len(sel) / ncol)
    f, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.9 * nrow), constrained_layout=True)
    for ax, (row, (q, A_zd, A_me, item)) in zip(axes.flat, sel):
        if kind == "continuum":
            lim = min(8.0, 1.25 * (abs(item["params"]["mu"]) + item["params"]["W"]) + 0.5)
        else:
            sp = item["spectrum"]
            lim = min(12.0, 1.4 * float(np.abs(sp.energies[sp.weights > 1e-3]).max()) + 0.8)
        m = np.abs(WG) <= lim
        st = slice(None, None, 8)
        ax.fill_between(WG[m][st], q[0][m][st], q[2][m][st], color=FLOW, alpha=0.25, lw=0,
                        label="flow posterior, 90% band")
        ax.plot(WG[m][st], q[1][m][st], color=FLOW, lw=1.3, label="flow posterior, median")
        ax.plot(WG[m][st], A_zd[m][st], color=VAE, lw=1.0, label="VAE latent MAP")
        ax.plot(WG[m][st], A_me[m][st], color=ME, lw=1.0, label="MaxEnt")
        if kind == "continuum":
            ax.plot(WG[m][st], item["A_grid"][m][st], color=INK, lw=1.5, ls="--", label="exact")
            top = 1.6 * float(item["A_grid"][m].max())
        else:
            sp = item["spectrum"]
            top = 1.3 * float(max(q[2][m].max(), A_zd[m].max()))
            big = sp.weights > 2e-3
            ax.vlines(sp.energies[big], 0, sp.weights[big] / sp.weights.max() * 0.92 * top,
                      color=INK, lw=1.6, label="exact lines (height = weight)")
        ax.set_ylim(-0.03 * top, top)
        vals = {"flow": row["w1"]["flow"], "VAE": row["w1"]["vae"], "MaxEnt": row["w1"]["maxent"]}
        best = min(vals, key=vals.get)
        ax.set_title(item["family"] + "\n" + "   ".join(
            (f"[{k} {v:.3f}]" if k == best else f"{k} {v:.3f}") for k, v in vals.items()), fontsize=8.5)
        ax.tick_params(labelleft=False)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for ax in axes.flat[len(sel):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    f.legend(h, l, loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    wins = {k: sum(1 for r, _ in sel if min(r["w1"], key=r["w1"].get) == c)
            for k, c in (("flow", "flow"), ("VAE", "vae"), ("MaxEnt", "maxent"))}
    what = ("exact lattice models with a lifetime" if kind == "continuum"
            else "exactly diagonalised models (delta spectra)")
    f.suptitle(f"Known answers, {what}: beta = {beta:g}, noise {noise:g}; none of these is in any "
               f"checkpoint's training set\nPanel titles give W$_1$ to the exact answer, lowest in "
               f"brackets: flow {wins['flow']}, VAE {wins['VAE']}, MaxEnt {wins['MaxEnt']} of {len(sel)}",
               fontsize=10.5)
    f.savefig(path, dpi=140)
    plt.close(f)


def main():
    from spevae import load_model
    from spevae.models.flow import load_flow
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    noise = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
    beta = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
    torch.set_num_threads(6)
    flow, vae = load_flow(sys.argv[2]).eval(), load_model(sys.argv[3])
    rows = run(flow, vae, noise, beta)
    tag = f"b{beta:g}_n{noise:g}"
    for kind in ("continuum", "discrete"):
        fig(rows, kind, out / f"exact_{kind}_{tag}.png", noise, beta)
    summ = {}
    for kind in ("continuum", "discrete"):
        sel = [r for r, _ in rows if r["kind"] == kind]
        summ[kind] = {
            "n": len(sel),
            "wins": {m: sum(1 for r in sel if min(r["w1"], key=r["w1"].get) == m)
                     for m in ("flow", "vae", "maxent")},
            "w1_geomean": {m: float(np.exp(np.mean([np.log(r["w1"][m]) for r in sel])))
                           for m in ("flow", "vae", "maxent")},
            "chi2_median": {m: float(np.median([r["chi2"][m] for r in sel]))
                            for m in ("flow", "vae", "maxent")},
            "flow_frac_accepted_min_median": [float(min(r["flow_frac_accepted"] for r in sel)),
                                              float(np.median([r["flow_frac_accepted"] for r in sel]))],
            "flow_no_sample_fits": int(sum(r["flow_no_sample_fits"] for r in sel))}
    with open(out / f"exact_models_{tag}.json", "w") as fh:
        json.dump({"noise": noise, "beta": beta, "samples": S, "summary": summ,
                   "rows": [r for r, _ in rows]}, fh, indent=1)
    print(json.dumps(summ, indent=1))
    print(f"-> {out}/exact_models_{tag}.json + exact_continuum_{tag}.png, exact_discrete_{tag}.png")


if __name__ == "__main__":
    main()
