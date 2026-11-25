"""What the width floor is, on two exact line spectra.

Every pole of the model has half-width gamma >= 0.1 pi / beta (a tenth of the
first Matsubara frequency).  A true line is narrower than that, so the model
draws it as a peak of at least that width.  For two exactly diagonalised
spectra of the known-answer set this draws, on the same noisy data,

  - the exact lines,
  - the exact lines given the minimum width (plain Lorentzians, exact
    positions and weights): what the floor alone costs,
  - the refined flow posterior and MaxEnt,

and the cumulative weights, whose area between curves is W1.

    python scripts/step3/width_floor_figure.py out/step3/width_floor out/step3/F3_60k_s1.pt

Writes width_floor.{json,png}.
"""
import json
import math
import sys
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

# (beta, noise, index among the discrete truths, half-range of the spectrum panel)
CASES = ((8.0, 1e-4, 0, 2.2, "upper left"), (2.0, 1e-3, 5, 6.0, "upper center"))
FLOW, ME, INK, FLOOR = J.FLOW, J.ME, J.INK, "#b3261e"
WG, DW = J.WG, J.DW


def one(flow, beta, noise, k, lam=10.0):
    from spevae.evaluation import cases as C
    from spevae.evaluation import methods as M
    from spevae.models.flow import refine_samples
    from spevae.posterior import weight_at_width_floor
    item = [t for t in J.make_truths(beta) if t["kind"] == "discrete"][k]
    inst = C.draw(item["truth"], noise, J.SEED)
    tr = inst.truth
    L = tr.x.numel()
    g = torch.Generator().manual_seed(9)
    z, norm = flow.sample_coords(inst.G, tr.x, beta, sigma=inst.sigma, n_samples=J.S, generator=g,
                                 inv_sqrt_R=inst.isr)
    ps, c2 = refine_samples(flow, z, norm, beta, tr.x, inst.G.expand(J.S, -1), inst.sigma.expand(J.S, -1),
                            inst.isr, lam, perturb=True, generator=g, anneal_from=1e-2)
    acc = (c2 <= 1.0 + 3.0 * math.sqrt(2.0 / L)).numpy()
    if not acc.any():
        acc[:] = True
    A = J.a_phys(ps, beta)[acc]
    me = M.make_method("maxent").synthetic(inst)
    A_me = np.interp(WG, me.omega, me.A, left=0.0, right=0.0)
    sp = item["spectrum"]
    gmin = flow.coords.gamma_min / beta
    A_floor = sp.broadened(WG, gmin)
    cdf = lambda a: np.cumsum(a) * DW
    curves = {"flow": A.mean(0), "maxent": A_me, "floor": A_floor}
    w1 = {k_: J.w1_cdf(cdf(v), item["cdf"]) for k_, v in curves.items()}
    # how much of each W1 comes from weight placed more than 1 away from every exact line
    big = sp.weights > 2e-3
    far = np.abs(WG[:, None] - sp.energies[big][None, :]).min(1) > 1.0
    w1_far = {k_: float((np.abs(cdf(v) - item["cdf"]) * far).sum() * DW) for k_, v in curves.items()}
    acc_t = torch.as_tensor(acc)
    return {"item": item, "beta": beta, "noise": noise, "gamma_min": gmin, "w1": w1, "w1_far": w1_far,
            "share_at_floor": weight_at_width_floor(ps.a[acc_t], ps.gamma[acc_t], flow.coords.gamma_min),
            "q": np.quantile(A, [0.05, 0.5, 0.95], axis=0), "mean": A.mean(0), "A_me": A_me,
            "A_floor": A_floor, "accepted": float(acc.mean())}


def main():
    from spevae.models.flow import load_flow
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    flow = load_flow(sys.argv[2]).eval()
    torch.set_num_threads(4)
    res = [one(flow, *c[:3]) for c in CASES]

    f, axes = plt.subplots(len(res), 2, figsize=(14, 4.0 * len(res)), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.25, 1]})
    cdf = lambda a: np.cumsum(a) * DW
    for (a0, a1), r, c in zip(axes, res, CASES):
        sp, half = r["item"]["spectrum"], c[3]
        big = sp.weights > 2e-3
        mid = float((sp.energies[big] * sp.weights[big]).sum() / sp.weights[big].sum())
        m = np.abs(WG - mid) <= half
        st = slice(None, None, 2)
        top = 1.15 * float(max(r["q"][2][m].max(), r["A_floor"][m].max(), r["A_me"][m].max()))
        a0.fill_between(WG[m][st], r["q"][0][m][st], r["q"][2][m][st], color=FLOW, alpha=0.25, lw=0,
                        label="flow posterior, 90% band")
        a0.plot(WG[m][st], r["q"][1][m][st], color=FLOW, lw=1.5, label="flow posterior, median")
        a0.plot(WG[m][st], r["A_me"][m][st], color=ME, lw=1.2, label="MaxEnt")
        a0.plot(WG[m][st], r["A_floor"][m][st], color=FLOOR, lw=1.2, ls="--",
                label="exact lines drawn at the minimum width")
        a0.vlines(sp.energies[big], 0, sp.weights[big] / sp.weights.max() * 0.95 * top, color=INK, lw=1.8,
                  label="exact lines (height = weight)")
        a0.set_ylim(-0.03 * top, top)
        a0.set_xlabel(r"$\omega$")
        a0.set_ylabel(r"$A(\omega)$")
        a0.set_title(f"{r['item']['family']}, beta = {r['beta']:g}, error bar {r['noise']:g}\nminimum half-width "
                     f"0.1$\\pi$/beta = {r['gamma_min']:.3f}; {r['share_at_floor']:.0%} of the flow's weight "
                     "is on poles at that minimum", fontsize=9.5)
        a0.legend(fontsize=7.5, frameon=False, loc=c[4])
        m1 = np.abs(WG - mid) <= 2.5 * half
        a1.step(WG[m1], r["item"]["cdf"][m1], where="post", color=INK, lw=1.6, label="exact")
        a1.plot(WG[m1][st], cdf(r["mean"])[m1][st], color=FLOW, lw=1.4, label="flow posterior mean")
        a1.plot(WG[m1][st], cdf(r["A_me"])[m1][st], color=ME, lw=1.2, label="MaxEnt")
        a1.plot(WG[m1][st], cdf(r["A_floor"])[m1][st], color=FLOOR, lw=1.2, ls="--",
                label="exact lines at the minimum width")
        a1.set_xlabel(r"$\omega$")
        a1.set_ylabel("cumulative spectral weight")
        w, wf = r["w1"], r["w1_far"]
        a1.set_title("W$_1$ = area between a curve and the exact steps\n"
                     f"flow {w['flow']:.3f}, MaxEnt {w['maxent']:.3f}, minimum-width lines {w['floor']:.3f}\n"
                     f"(part from weight more than 1 away from every line: {wf['flow']:.3f}, "
                     f"{wf['maxent']:.3f}, {wf['floor']:.3f})", fontsize=9.5)
        a1.legend(fontsize=7.5, frameon=False, loc="upper left")
        for ax in (a0, a1):
            ax.grid(color="#e4e3df", lw=0.6)
            ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    f.suptitle("The width floor on two exact line spectra: no peak of the model is narrower than "
               "0.1$\\pi$/beta (red dashed = the exact lines drawn that wide)", fontsize=10.5)
    f.savefig(out / "width_floor.png", dpi=150)
    with open(out / "width_floor.json", "w") as fh:
        json.dump([{"family": r["item"]["family"], "params": r["item"]["params"], "beta": r["beta"],
                    "noise": r["noise"], "gamma_min": r["gamma_min"], "w1": r["w1"], "w1_far": r["w1_far"],
                    "share_at_floor": r["share_at_floor"], "accepted": r["accepted"]} for r in res],
                  fh, indent=1, default=float)
    for r in res:
        print(r["item"]["family"], r["beta"], r["noise"], "W1", {k: round(v, 3) for k, v in r["w1"].items()},
              "far", {k: round(v, 3) for k, v in r["w1_far"].items()},
              "share at floor", round(r["share_at_floor"], 2), "accepted", r["accepted"])
    print(f"-> {out}/width_floor.{{json,png}}")


if __name__ == "__main__":
    main()
