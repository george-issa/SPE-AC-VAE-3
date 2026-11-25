"""Is the excess width of the reported posterior carried by the pole count?

For held-out truths of the step-3 prior that ARE pole sets (so the true count
is defined), the reported posterior is split by each sample's pole count
relative to the truth's, and the PIT coverage of the truth's window weights
is recomputed inside each group:

  all        every accepted sample (what is reported),
  right      samples with the true count            (an oracle no user has),
  near       samples within one pole of it,
  more       samples with at least two poles more.

If `right` is calibrated and `all` is not, the excess is the flow's spread
over pole counts.  Also recorded: how the count distribution compares with
the truth's, and the spread (standard deviation of the window weights) of
each group relative to `all`.

    python scripts/step3/count_diagnostic.py out/step3/F3_6k.pt out/step3/bank/label_bank_trusted.pt \
        out/step3/count_diag_F3_6k [shard nshards]
    python scripts/step3/count_diagnostic.py fig out/step3/count_diag_F3_6k
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/step3")
warnings.filterwarnings("ignore")

import numpy as np
import torch

import calibration as CAL

CELLS = [(8.9e-5, 8.0, 0.125), (3.8e-4, 20.0, 0.2), (1.6e-3, 2.0, 0.05), (7.0e-3, 8.0, 0.25),
         (1.8e-4, 4.0, 0.125), (3.4e-3, 12.0, 0.1)]
M, S = 32, 96
GROUPS = ("all", "right", "near", "more")


def run(flow_path, bank, out, shard, nshards):
    from spevae.data.pole_prior import PolePriorGenerator
    from spevae.models.flow import load_flow
    from spevae.physics.poles import PoleSet
    from spevae.posterior import DEFAULT_WINDOWS, continue_posterior

    torch.set_num_threads(4)
    flow = load_flow(flow_path).eval()
    rows = []
    for i, (nz, beta, dtau) in enumerate(CELLS):
        if i % nshards != shard:
            continue
        gen = PolePriorGenerator(omega_c=20.0, P_max=12, residue_mode="dispersive", smooth_bank=bank,
                                 p_smooth=0.0, p_exact=0.0, beta_choices=(beta,),
                                 noise_scale_range=(nz, nz), dtau_choices=(dtau,))
        with gen.fixed_scalars(41000 + i):
            b = gen.batch(M, torch.Generator().manual_seed(42000 + i))
        nodes = (b.quad.nodes.double() / beta).numpy()
        wq = b.quad.weights.double().numpy()
        A_true = b.A_true.double().numpy()
        tot = (A_true * wq).sum(-1)
        w_true = np.stack([(A_true * wq * ((nodes >= lo) & (nodes < hi))).sum(-1) / tot
                           for lo, hi in DEFAULT_WINDOWS], -1)
        n_true = b.poles_true["n_true"].numpy()
        t0 = time.time()
        rec = {"noise": nz, "beta": beta, "n_tau": int(b.x.numel()), "w_true": w_true.tolist(),
               "n_true": n_true.tolist(), "samples": []}
        for m in range(M):
            rep = continue_posterior(flow, b.G[m], b.x, beta, b.sigma[m], b.inv_sqrt_R, n_samples=S,
                                     bounds=False, generator=torch.Generator().manual_seed(7000 + 97 * i + m))
            dim = PoleSet(rep.poles.eps * beta, rep.poles.gamma * beta, rep.poles.a, rep.poles.b)
            rec["samples"].append({"w": CAL.window_weights_poles(dim, beta, DEFAULT_WINDOWS).tolist(),
                                   "n": (rep.poles.a != 0).sum(-1).tolist(), "accepted": rep.n_accepted})
        rec["seconds"] = time.time() - t0
        rows.append(rec)
        print(f"noise {nz:7.1e} beta {beta:4.0f}: {rec['seconds']:.0f}s", flush=True)
        with open(out / f"count.shard{shard}.json", "w") as fh:
            json.dump(rows, fh)


def summarise(out):
    rows = []
    for p in sorted(out.glob("count.shard*.json")):
        rows += json.load(open(p))
    if len(rows) != len(CELLS):
        raise SystemExit(f"{out}: {len(rows)} of {len(CELLS)} cells present -- still running?")
    rows.sort(key=lambda r: r["noise"])
    rng = np.random.default_rng(5)
    pit = {g: [] for g in GROUPS}
    rel_sd = {g: [] for g in GROUPS}
    share = {g: [] for g in GROUPS}
    dn = []
    cells = []
    for r in rows:
        wt, nt = np.array(r["w_true"]), np.array(r["n_true"])
        cp = {g: [] for g in GROUPS}
        for m, a in enumerate(r["samples"]):
            w, n = np.array(a["w"]), np.array(a["n"])
            d = n - nt[m]
            dn += d.tolist()
            sel = {"all": np.ones(len(w), bool), "right": d == 0, "near": np.abs(d) <= 1, "more": d >= 2}
            sd_all = w.std(0)
            for g in GROUPS:
                k = int(sel[g].sum())
                share[g].append(k / len(w))
                if k < 8:                       # too few samples for a rank
                    continue
                ws = w[sel[g]]
                p = ((ws < wt[m]).sum(0) + rng.random(wt.shape[1])) / (k + 1)
                pit[g].append(p)
                cp[g].append(p)
                rel_sd[g].append(float(np.median(ws.std(0) / np.maximum(sd_all, 1e-12))))
        cells.append({"noise": r["noise"], "beta": r["beta"],
                      **{g: ({"truths": len(cp[g]),
                              "cov50": float(np.mean((np.array(cp[g]) > .25) & (np.array(cp[g]) < .75))),
                              "cov90": float(np.mean((np.array(cp[g]) > .05) & (np.array(cp[g]) < .95)))}
                             if cp[g] else None) for g in GROUPS}})
    res = {"design": {"cells": CELLS, "truths_per_cell": M, "samples": S}, "cells": cells, "groups": {},
           "count_minus_true": {str(k): float(np.mean(np.array(dn) == k)) for k in range(-6, 9)},
           "count_minus_true_mean": float(np.mean(dn))}
    for g in GROUPS:
        p = np.array(pit[g])
        in50, in90 = (p > .25) & (p < .75), (p > .05) & (p < .95)
        n = len(p)
        res["groups"][g] = {"truths_with_enough_samples": n, "share_of_samples": float(np.mean(share[g])),
                            "cov50": float(in50.mean()), "cov50_se": float(in50.mean(1).std() / np.sqrt(n)),
                            "cov90": float(in90.mean()), "cov90_se": float(in90.mean(1).std() / np.sqrt(n)),
                            "spread_relative_to_all": float(np.median(rel_sd[g]))}
    with open(out / "count_diagnostic.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def fig(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = summarise(out)
    COL = {"all": "#eb6834", "right": "#0f8a8a", "near": "#2a78d6", "more": "#7d7c78"}
    LAB = {"all": "all accepted samples (reported)", "right": "samples with the true pole count",
           "near": "within one pole of it", "more": "two or more poles too many"}
    f, axes = plt.subplots(1, 3, figsize=(15.5, 3.9), constrained_layout=True)
    ax = axes[0]
    ks = [int(k) for k in res["count_minus_true"]]
    ax.bar(ks, [res["count_minus_true"][str(k)] for k in ks], color="#2a78d6", width=0.8)
    ax.axvline(0, color="#0b0b0b", lw=1.0)
    ax.set_xlabel("pole count of a sample minus the truth's")
    ax.set_title(f"reported samples by pole count (mean excess {res['count_minus_true_mean']:+.1f} poles)", fontsize=9.5)
    x = np.arange(len(GROUPS))
    for ax, key, nom in ((axes[1], "cov50", 0.5), (axes[2], "cov90", 0.9)):
        g = res["groups"]
        ax.bar(x, [g[k][key] for k in GROUPS], yerr=[g[k][key + "_se"] for k in GROUPS], capsize=3,
               color=[COL[k] for k in GROUPS], width=0.62)
        ax.axhline(nom, color="#52514e", ls="--", lw=0.9)
        for xi, k in zip(x, GROUPS):
            ax.text(xi, g[k][key] + g[k][key + "_se"] + 0.015, f"{g[k][key]:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{LAB[k]}\n{g[k]['share_of_samples']:.0%} of samples, spread x{g[k]['spread_relative_to_all']:.2f}"
                            for k in GROUPS], fontsize=7)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{int(nom * 100)}% interval coverage inside each group", fontsize=9.5)
    for ax in axes:
        ax.grid(color="#e4e3df", lw=0.6, axis="y")
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    n = res["groups"]["all"]["truths_with_enough_samples"]
    f.suptitle(f"Is the excess width carried by the pole count?  {n} held-out pole-set truths, {S} samples each, "
               f"{len(CELLS)} noise / beta cells", fontsize=10.5)
    f.savefig(out / "count_diagnostic.png", dpi=150)
    print(json.dumps({k: res[k] for k in ("groups", "count_minus_true_mean")}, indent=1))
    print(f"-> {out}/count_diagnostic.json + .png")


if __name__ == "__main__":
    if sys.argv[1] == "fig":
        fig(Path(sys.argv[2]))
    else:
        o = Path(sys.argv[3])
        o.mkdir(parents=True, exist_ok=True)
        run(sys.argv[1], sys.argv[2], o, int(sys.argv[4]) if len(sys.argv) > 4 else 0,
            int(sys.argv[5]) if len(sys.argv) > 5 else 1)
