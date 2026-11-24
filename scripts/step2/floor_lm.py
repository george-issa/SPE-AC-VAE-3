"""The Holstein "manifold floor" and MAP fit, converged.

The banked floor (`evaluation/suite._run_floor`, `floor_curve.py`) is an Adam
descent in the latent with the prior off, capped at 10,000 steps.  This script
puts three numbers side by side for each checkpoint:

  adam10k   the banked protocol, with its `converged` flag and step count
  lm floor  the same objective (prior off) by Levenberg-Marquardt, stopping at
            the calibrated noise target
  lm MAP    prior on (lambda = 1), from the encoder mean and from four
            perturbed starts (mean + 0.5 N(0, I)) -- the MAP landscape has
            several stationary points, so one start is not the answer

and records |z| at each solution: how far outside the model's own N(0, I)
prior the real cell has to be placed to be fitted.

    python scripts/step2/floor_lm.py out/step2/floor_lm \
        s1_metric=out/step1/s1_metric.pt M1=out/step2/map_M1_60k.pt \
        ctrl60=out/step2/data_arm/ctrl_60k.pt D24=out/step2/data_arm/D24_60k.pt
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

INK, MUTED, GRID = "#1f1e1c", "#52514e", "#d9d8d4"
COL = {"adam": "#eda100", "lm": "#eb6834", "map": "#2a78d6"}
KW = dict(cov_method="structured", min_eigenvalue=1e-10)


def judge(tag, path):
    from spevae import load_model
    from spevae.evaluation.cases import holstein_cell
    from spevae.refine import RefineConfig, refine_latent
    mg = holstein_cell()
    m = load_model(path)
    out = {"tag": tag, "path": path}

    def rec(r, seconds):
        return {"chi2": float(r.chi2), "ratio": float(r.chi2_ratio), "steps": int(r.steps_run),
                "converged": bool(r.converged), "stationary": bool(r.stationary),
                "z_norm": float(r.z.norm()), "data_term": float(r.data_term),
                "prior_term": float(r.prior_term), "seconds": seconds}

    t0 = time.time()
    r = refine_latent(m, mg, RefineConfig(steps=10000, prior_weight=0.0, patience=3000, **KW))
    out["adam10k_floor"] = rec(r, time.time() - t0)
    out["target"], out["latent_dim"] = float(r.chi2_target), int(r.z.shape[-1])
    out["z_init_norm"] = float(r.z_init.norm())
    t0 = time.time()
    r = refine_latent(m, mg, RefineConfig(steps=3000, **KW))
    out["adam3000_map"] = rec(r, time.time() - t0)
    t0 = time.time()
    r = refine_latent(m, mg, RefineConfig(solver="lm", prior_weight=0.0, lm_iters=3000, **KW))
    out["lm_floor"] = rec(r, time.time() - t0)
    out["lm_map"] = []
    orig = m.encode
    for seed in (None, 1, 2, 3, 4):
        if seed is not None:
            def enc(*a, _s=seed, **k):
                mu, lv, n = orig(*a, **k)
                g = torch.Generator().manual_seed(_s)
                return mu + 0.5 * torch.randn(mu.shape, generator=g), lv, n
            m.encode = enc
        t0 = time.time()
        r = refine_latent(m, mg, RefineConfig(solver="lm", prior_weight=1.0, lm_iters=1500,
                                              lm_ftol=1e-10, **KW))
        m.encode = orig
        out["lm_map"].append(dict(rec(r, time.time() - t0),
                                  start="encoder mean" if seed is None else f"mean + 0.5 N #{seed}"))
    f, a, mp = out["lm_floor"], out["adam10k_floor"], out["lm_map"]
    print(f"{tag:10s} adam10k floor {a['ratio']:7.2f}x (converged {a['converged']}) | lm floor "
          f"{f['ratio']:.2f}x in {f['steps']} it, |z| {f['z_norm']:.0f} | lm MAP from the mean "
          f"{mp[0]['ratio']:.2f}x ({'at target' if mp[0]['converged'] else 'stationary' if mp[0]['stationary'] else 'budget'}), "
          f"5 starts {min(x['ratio'] for x in mp):.2f}-{max(x['ratio'] for x in mp):.2f}x, "
          f"MAP objective {min(x['data_term'] + x['prior_term'] for x in mp):.0f}-"
          f"{max(x['data_term'] + x['prior_term'] for x in mp):.0f}", flush=True)
    return out


def fig(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = [r["tag"] for r in res]
    y = np.arange(len(res))
    fig_, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15.5, 3.6), constrained_layout=True)
    a0.plot([r["adam10k_floor"]["ratio"] for r in res], y - 0.12, "o", color=COL["adam"], ms=7,
            label="Adam, 10,000 steps (banked floor; none converged)")
    a0.plot([r["lm_floor"]["ratio"] for r in res], y + 0.12, "o", color=COL["lm"], ms=7,
            label="Levenberg-Marquardt (stops at the target)")
    for k, r in enumerate(res):
        a0.annotate(f"{r['lm_floor']['steps']} it", (r["lm_floor"]["ratio"], k + 0.12),
                    xytext=(-6, 7), textcoords="offset points", fontsize=7.5, color=COL["lm"], ha="right")
    a0.axvline(1.0, color=MUTED, ls="--", lw=0.9)
    a0.set_xscale("log")
    a0.set_title("prior off: best $\\chi^2$ / target the latent reaches\n(dashed = noise target)", fontsize=9.5)
    a0.legend(fontsize=7.5, frameon=False, loc="lower right")
    for k, r in enumerate(res):
        a1.plot(r["adam3000_map"]["ratio"], k - 0.15, "o", color=COL["adam"], ms=7,
                label="Adam, 3000 steps (banked MAP)" if k == 0 else None)
        xs = [x["ratio"] for x in r["lm_map"]]
        a1.plot(xs[1:], [k + 0.15] * (len(xs) - 1), "o", color=COL["map"], ms=5, mfc="none",
                label="LM, perturbed starts" if k == 0 else None)
        a1.plot(xs[0], k + 0.15, "o", color=COL["map"], ms=7,
                label="LM, from the encoder mean" if k == 0 else None)
    a1.axvline(1.0, color=MUTED, ls="--", lw=0.9)
    a1.set_xscale("log")
    a1.set_title("prior on (MAP): $\\chi^2$ / target at the solution\n(stationary points depend on the start)",
                 fontsize=9.5)
    a1.legend(fontsize=7.5, frameon=False, loc="lower right")
    for k, r in enumerate(res):
        a2.plot(r["lm_floor"]["z_norm"], k - 0.15, "o", color=COL["lm"], ms=7,
                label="at the prior-off fit" if k == 0 else None)
        zs = [x["z_norm"] for x in r["lm_map"]]
        a2.plot(zs[1:], [k + 0.15] * (len(zs) - 1), "o", color=COL["map"], ms=5, mfc="none")
        a2.plot(zs[0], k + 0.15, "o", color=COL["map"], ms=7,
                label="at the MAP solutions" if k == 0 else None)
    a2.axvline(res[0]["latent_dim"] ** 0.5, color=MUTED, ls="--", lw=0.9)
    a2.annotate("typical prior draw", (res[0]["latent_dim"] ** 0.5, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -3), textcoords="offset points", fontsize=7.5, color=MUTED, va="top")
    a2.set_xscale("log")
    a2.set_title("$|z|$ at the solution: how far outside its own prior\nthe model must go to fit the real cell",
                 fontsize=9.5)
    a2.legend(fontsize=7.5, frameon=False, loc="lower right")
    for ax in (a0, a1, a2):
        ax.set_yticks(y)
        ax.set_yticklabels(tags)
        ax.invert_yaxis()
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig_.suptitle("Holstein cell: the latent fit, converged.  Every checkpoint's manifold contains a "
                  "noise-level fit; what separates them is how far from the prior it lies", fontsize=10.5)
    fig_.savefig(path, dpi=150)


if __name__ == "__main__":
    torch.set_num_threads(2)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    res = [judge(*a.split("=", 1)) for a in sys.argv[2:]]
    with open(out / "floor_lm.json", "w") as fh:
        json.dump(res, fh, indent=1)
    fig(res, out / "floor_lm.png")
    print(f"-> {out}/floor_lm.json + .png")
