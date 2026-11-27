"""Reconstructions of spectra with a known answer, as a user of the package gets them.

Every spectrum of the known-answer judge (20 exact lattice spectra, 12 exactly
diagonalised line spectra; `scripts/eval/exact_models_judge.py`) is given the
judge's noise and continued with `spevae.posterior.continue_posterior` at its
defaults -- the call a user makes -- and with MaxEnt on the same data.  The
curves are stored, so the figures can be redrawn without recomputing.

    python scripts/figures/known_answers.py run OUT FLOW BETA NOISE [samples=200] [stated]
    python scripts/figures/known_answers.py fig OUT DEST

`run` writes OUT/known_b<beta>_n<noise>.{json,npz}; with `stated` it continues
the line spectra only, with the spectrum stated to be discrete
(`discrete=True`, what `spevae continue --discrete` does), and writes
OUT/known_stated_b<beta>_n<noise>.{json,npz}.  `fig` draws every setting
found in OUT into DEST as lattice_beta<beta>_errorbar<noise>.png,
lines_beta<beta>_errorbar<noise>.png and
lines_stated_discrete_beta<beta>_errorbar<noise>.png and prints the scores.

Error (W1) is the area between the reconstructed and the exact cumulative
spectral weight, in units of omega; for the posterior it is taken on the mean
of the accepted samples.
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

MODEL, ME, INK = "#2a78d6", "#7d7c78", "#0b0b0b"
NAMES = {"tb_chain": "chain (1D)", "tb_square": "square lattice", "tb_triangular": "triangular lattice",
         "tb_honeycomb": "honeycomb lattice", "tb_bethe": "Bethe lattice", "tb_hypercubic": "hypercubic, d = inf",
         "ssh_chain": "SSH chain", "bcs": "BCS superconductor", "resonant_level": "resonant level",
         "falicov_kimball": "Falicov-Kimball", "hubbard_atom": "Hubbard atom", "hubbard_dimer": "Hubbard dimer",
         "hubbard_trimer": "Hubbard trimer", "hubbard_ring4": "Hubbard 4-site ring",
         "anderson_cluster": "Anderson cluster", "holstein_atom": "Holstein atom"}
KINDS = (("continuum", "lattice"), ("discrete", "lines"))


def half_range(item):
    if item["kind"] == "continuum":
        return min(8.0, 1.25 * (abs(item["params"]["mu"]) + item["params"]["W"]) + 0.5)
    sp = item["spectrum"]
    return min(12.0, 1.4 * float(np.abs(sp.energies[sp.weights > 1e-3]).max()) + 0.8)


def run(out, flow_path, beta, noise, samples, stated=False):
    from spevae.evaluation import cases as C
    from spevae.evaluation import methods as M
    from spevae.models.flow import load_flow
    from spevae.posterior import continue_posterior
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    flow = load_flow(flow_path).eval()
    torch.set_num_threads(4)
    rows, curves = [], {}
    for k, item in enumerate(J.make_truths(beta)):
        if stated and item["kind"] != "discrete":
            continue
        inst = C.draw(item["truth"], noise, J.SEED)
        t0 = time.time()
        rep = continue_posterior(flow, inst.G, inst.truth.x, beta, inst.sigma, inst.isr, n_samples=samples,
                                 omega=J.WG, bounds=False, discrete=stated,
                                 generator=torch.Generator().manual_seed(9))
        secs = time.time() - t0
        me = M.make_method("maxent").synthetic(inst)
        A_me = np.interp(J.WG, me.omega, me.A, left=0.0, right=0.0)
        cdf = lambda a: np.cumsum(a) * J.DW
        d = rep.diagnostics
        row = {"k": k, "family": item["family"], "kind": item["kind"], "params": item["params"],
               "zero_width_lines": bool(d.get("line_fit")),
               "w1": {"model": J.w1_cdf(cdf(rep.band["mean"]), item["cdf"]),
                      "maxent": J.w1_cdf(cdf(A_me), item["cdf"])},
               "narrow_lines_flagged": bool(d.get("line_shape_readings")),
               "floor_factor": float(d.get("floor_factor", 1.0)),
               "accepted": rep.n_accepted / max(rep.n_samples, 1),
               "chi2_over_target": float(np.median(rep.chi2) / rep.chi2_target),
               "warnings": list(rep.warnings), "seconds": secs, "half_range": half_range(item)}
        m = np.abs(J.WG) <= row["half_range"]
        st = max(1, int(m.sum() // 3000))
        curves[f"w{k}"] = J.WG[m][::st]
        curves[f"q{k}"] = np.stack([rep.band[q][m][::st] for q in ("q05", "q50", "q95")]).astype(np.float32)
        curves[f"me{k}"] = A_me[m][::st].astype(np.float32)
        # the model's zero-width lines, if it fitted any: mean weight per cell (they have no height)
        curves[f"ln{k}"] = np.stack(rep.line_cells(400, row["half_range"]))
        if item["kind"] == "continuum":
            curves[f"ex{k}"] = item["A_grid"][m][::st].astype(np.float32)
        else:
            sp = item["spectrum"]
            curves[f"ex{k}"] = np.stack([sp.energies, sp.weights])
        rows.append(row)
        print(f"{item['kind']:9s} {item['family']:16s} error model {row['w1']['model']:.4f} MaxEnt "
              f"{row['w1']['maxent']:.4f} | accepted {row['accepted']:.0%}"
              f"{' | narrow lines flagged' if row['narrow_lines_flagged'] else ''} | {secs:.0f}s", flush=True)
    meta = getattr(flow, "meta", {}) or {}
    tag = f"known_{'stated_' if stated else ''}b{beta:g}_n{noise:g}"
    with open(out / f"{tag}.json", "w") as fh:
        json.dump({"checkpoint": str(flow_path), "model": (meta.get("recipe") or {}).get("name"),
                   "beta": beta, "noise": noise, "samples": samples, "dtau": J.DTAU, "stated": bool(stated),
                   "made": time.strftime("%Y-%m-%d"), "rows": rows}, fh, indent=1, default=float)
    np.savez_compressed(out / f"{tag}.npz", **curves)
    print(f"-> {out}/{tag}.json + .npz")


def gmean(v):
    return float(np.exp(np.mean(np.log(v))))


def draw(res, cur, kind, path):
    idx = [i for i, r in enumerate(res["rows"]) if r["kind"] == kind]
    ncol = 5 if kind == "continuum" else 4
    nrow = math.ceil(len(idx) / ncol)
    f, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.0 * nrow), constrained_layout=True)
    for ax, i in zip(axes.flat, idx):
        r = res["rows"][i]
        k = r.get("k", i)                               # the spectrum's index in the stored curves
        w, q, me, ex = cur[f"w{k}"], cur[f"q{k}"], cur[f"me{k}"], cur[f"ex{k}"]
        ln = cur[f"ln{k}"] if f"ln{k}" in cur else np.zeros((2, 1))
        ax.fill_between(w, q[0], q[2], color=MODEL, alpha=0.25, lw=0, label="this model, 90% band")
        ax.plot(w, q[1], color=MODEL, lw=1.4, label="this model, median")
        ax.plot(w, me, color=ME, lw=1.0, label="MaxEnt")
        if kind == "continuum":
            ax.plot(w, ex, color=INK, lw=1.4, ls="--", label="exact")
            top = 1.6 * float(ex.max())
        else:
            # a sharp line's band can tower over everything else: cap the axis at 4x the median's peak
            top = 1.3 * float(max(min(q[2].max(), 4.0 * q[1].max()), me.max()))
            big = ex[1] > 2e-3
            scale = 0.92 * top / ex[1].max()             # one scale for the exact lines and the model's
            on = ln[1] > 2e-3
            ax.vlines(ln[0][on], 0, np.minimum(ln[1][on] * scale, top), color=MODEL, lw=4.0, alpha=0.55,
                      label="this model, zero-width lines (same scale)")
            ax.vlines(ex[0][big], 0, ex[1][big] * scale, color=INK, lw=1.3,
                      label="exact lines (height proportional to weight)")
        ax.set_ylim(-0.03 * top, top)
        a, b = r["w1"]["model"], r["w1"]["maxent"]
        ax.set_title(NAMES.get(r["family"], r["family"])
                     + (" - zero-width lines fitted" if r.get("zero_width_lines") else
                        " - narrow lines flagged" if r["narrow_lines_flagged"] else "")
                     + f"\nerror: this model {a:.3f}, MaxEnt {b:.3f}", fontsize=8.5,
                     color=INK if a <= b else "#b3261e")
        ax.tick_params(labelsize=7)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for ax in axes.flat[len(idx):]:
        ax.axis("off")
    for ax in axes[-1] if nrow > 1 else axes:
        ax.set_xlabel(r"$\omega$", fontsize=9)
    for ax in (axes[:, 0] if nrow > 1 else axes[:1]):
        ax.set_ylabel(r"$A(\omega)$", fontsize=9)
    h, l = axes.flat[0].get_legend_handles_labels()
    f.legend(h, l, loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    sel = [res["rows"][i] for i in idx]
    wins = sum(r["w1"]["model"] <= r["w1"]["maxent"] for r in sel)
    gm, ge = gmean([r["w1"]["model"] for r in sel]), gmean([r["w1"]["maxent"] for r in sel])
    what = ("exact lattice spectra with a lifetime" if kind == "continuum"
            else "exactly diagonalised models (line spectra)"
            + (", stated to be discrete" if res.get("stated") else ""))
    f.suptitle(f"Known answers, {what}: beta = {res['beta']:g}, relative error bar {res['noise']:g}, "
               f"dtau = {res['dtau']:g}\nerror = area between reconstructed and exact cumulative weight "
               f"(lower is better); this model {gm:.3f}, MaxEnt {ge:.3f} on average; this model is the lower "
               f"on {wins} of {len(sel)} (titles in red where it is not)", fontsize=10.5)
    f.text(0.995, 0.004, f"model {res.get('model') or Path(res['checkpoint']).stem}, {res['samples']} samples, "
           f"drawn {res['made']}", ha="right", va="bottom", fontsize=6.5, color="#7d7c78")
    f.savefig(path, dpi=140)
    plt.close(f)
    return {"n": len(sel), "model": gm, "maxent": ge, "model_lower": wins,
            "flagged": sum(r["narrow_lines_flagged"] for r in sel),
            "with_lines": sum(bool(r.get("zero_width_lines")) for r in sel)}


def fig(out, dest):
    out, dest = Path(out), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    summ = {}
    for p in sorted(out.glob("known_b*_n*.json")) + sorted(out.glob("known_stated_b*_n*.json")):
        res = json.load(open(p))
        cur = np.load(p.with_suffix(".npz"))
        for kind, word in (KINDS[1:] if res.get("stated") else KINDS):
            name = (f"{word}{'_stated_discrete' if res.get('stated') else ''}_beta{res['beta']:g}"
                    f"_errorbar{res['noise']:g}.png")
            s = draw(res, cur, kind, dest / name)
            summ[name] = {"beta": res["beta"], "noise": res["noise"], "model": res.get("model"),
                          "made": res["made"], **s}
            print(f"{name:40s} model {s['model']:.3f}  MaxEnt {s['maxent']:.3f}  lower on "
                  f"{s['model_lower']}/{s['n']}  flagged {s['flagged']}")
    with open(out / "known_summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    return summ


if __name__ == "__main__":
    if sys.argv[1] == "run":
        rest = [a for a in sys.argv[6:] if a != "stated"]
        run(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]),
            int(rest[0]) if rest else 200, "stated" in sys.argv[6:])
    elif sys.argv[1] == "fig":
        fig(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
