"""What the pole model could say about exact line spectra, against what the flow says.

The flow is trained on the exactly diagonalised models as broadened lines:
every line a Lorentzian of the floor's width or more, the training data
computed from that broadened spectrum.  The lattice models are trained the
other way round: data from the exact spectrum, target the canonical pole
label fitted to it (`label_bank.extract_smooth_label`).  This runs the same
label protocol on the exact lines of the known-answer judge -- noise-free
data, the trained width floor -- and scores the label like a continuation:
W1 to the exact cumulative weight, next to the flow's and MaxEnt's score on
noisy data of the same spectrum (read from the judge's output).

The label is what the model class can represent when it is told everything;
no posterior from noisy data can beat it on average, so the gap between the
label and the flow bounds what a prior that knew line spectra could gain.

    python scripts/step3/line_label_probe.py out/step3/line_labels out/eval/exact_models_F3_60k_s1 [noise=0.001]

Writes line_labels.{json,png}.
"""
import json
import math
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/eval")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import exact_models_judge as J

BETAS = (2.0, 8.0, 16.0)
FITS = 1e-5            # a label reproduces the exact G(tau) when its largest deviation is below this
LABEL, FLOW, ME, INK = "#0b3d91", J.FLOW, J.ME, J.INK


def gm(v):
    return float(np.exp(np.mean(np.log(v)))) if len(v) else float("nan")


def main():
    from spevae.data.label_bank import extract_smooth_label
    from spevae.physics import PoleGreens, PoleSet, SpectralQuadrature
    out, judged = Path(sys.argv[1]), Path(sys.argv[2])
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for beta in BETAS:
        ref = [r for r in json.load(open(judged / f"exact_models_b{beta:g}_n{noise:g}_lam10.json"))["rows"]
               if r["kind"] == "discrete"]
        items = [t for t in J.make_truths(beta) if t["kind"] == "discrete"]
        fwd = PoleGreens(SpectralQuadrature.for_beta(beta, dtype=torch.float64))
        x = np.linspace(0.0, 1.0, int(round(beta / J.DTAU)) + 1)
        nu = torch.tensor(J.WG * beta, dtype=torch.float64)
        rows = []
        for item, r0 in zip(items, ref):
            assert item["family"] == r0["family"]
            sp, t0 = item["spectrum"], time.time()
            lab = extract_smooth_label(sp, P_max=12, residue_mode="dispersive")
            ps = PoleSet(*(torch.tensor(np.asarray(v), dtype=torch.float64).unsqueeze(0)
                           for v in (lab.eps * beta, lab.gamma * beta, lab.a, lab.b)))
            A = (ps.spectral(nu) * beta)[0].numpy()
            gap = float(np.abs(fwd(ps, torch.tensor(x))[0].numpy() - sp.greens(x)).max())
            rows.append({"family": item["family"], "poles": int(len(lab.eps)), "gap": gap,
                         "negative_weight": float(lab.neg_weight),
                         "min_width_over_floor": float(lab.gamma.min() * beta / (0.1 * math.pi)),
                         "w1": {"label": J.w1_cdf(np.cumsum(A) * J.DW, item["cdf"]), "flow": r0["w1"]["flow"],
                                "maxent": r0["w1"]["maxent"]}})
            r = rows[-1]
            print(f"beta {beta:g} {item['family']:16s} label {r['poles']:2d} poles, max |G_label - G_exact| {gap:.1e} "
                  f"| W1 label {r['w1']['label']:.3f} flow {r['w1']['flow']:.3f} MaxEnt {r['w1']['maxent']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        fit = [r for r in rows if r["gap"] < FITS]
        res.append({"beta": beta, "noise": noise, "rows": rows,
                    "w1_geomean": {k: gm([r["w1"][k] for r in rows]) for k in ("label", "flow", "maxent")},
                    "label_fits": len(fit),
                    "w1_geomean_where_label_fits": {k: gm([r["w1"][k] for r in fit]) for k in ("label", "flow", "maxent")}})
    with open(out / "line_labels.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    f, axes = plt.subplots(1, len(res), figsize=(5.6 * len(res), 4.8), constrained_layout=True)
    for ax, s in zip(axes, res):
        xs = np.arange(len(s["rows"]))
        fits = np.array([r["gap"] < FITS for r in s["rows"]])
        lab = np.array([r["w1"]["label"] for r in s["rows"]])
        ax.plot(xs[fits], lab[fits], "o", ms=7, color=LABEL, label="pole label, reproduces the exact G")
        ax.plot(xs[~fits], lab[~fits], "o", ms=7, mfc="white", color=LABEL,
                label="pole label, limited by the width floor")
        ax.plot(xs, [r["w1"]["flow"] for r in s["rows"]], "s", ms=5, color=FLOW, label=f"flow, noise {s['noise']:g}")
        ax.plot(xs, [r["w1"]["maxent"] for r in s["rows"]], "_", ms=11, mew=1.8, color=ME,
                label=f"MaxEnt, noise {s['noise']:g}")
        ax.set_yscale("log")
        ax.set_xticks(xs, [r["family"].replace("hubbard_", "hub_").replace("anderson_", "and_") for r in s["rows"]],
                      rotation=90, fontsize=7)
        g, gf = s["w1_geomean"], s["w1_geomean_where_label_fits"]
        ax.set_title(f"beta = {s['beta']:g}: label {g['label']:.3f}, flow {g['flow']:.3f}, MaxEnt {g['maxent']:.3f}\n"
                     f"where the label fits ({s['label_fits']} of {len(s['rows'])}): {gf['label']:.3f}, "
                     f"{gf['flow']:.3f}, {gf['maxent']:.3f}", fontsize=9)
        ax.set_ylabel("W$_1$ to the exact answer")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    f.legend(h, l, loc="outside lower center", ncol=4, fontsize=8.5, frameon=False)
    f.suptitle("Exact line spectra: the canonical pole label of the noise-free data (what the model class can "
               "represent) against the flow and MaxEnt on noisy data", fontsize=10.5)
    f.savefig(out / "line_labels.png", dpi=150)
    for s in res:
        g, gf = s["w1_geomean"], s["w1_geomean_where_label_fits"]
        print(f"== beta {s['beta']:g}: label {g['label']:.3f} flow {g['flow']:.3f} MaxEnt {g['maxent']:.3f} | where the "
              f"label fits ({s['label_fits']}/{len(s['rows'])}): {gf['label']:.3f} {gf['flow']:.3f} {gf['maxent']:.3f}")
    print(f"-> {out}/line_labels.{{json,png}}")


if __name__ == "__main__":
    main()
