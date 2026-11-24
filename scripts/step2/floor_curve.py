"""The real-data auditor along training: Holstein fit vs step, per arm.

For every snapshot of each run: amortized chi2, MAP chi2 (z-descent 3000,
prior anchor 1) and the manifold floor (prior released, 10k steps; the
descent stops once chi2 reaches the target, so "floor = target" means "the
decoder manifold reaches the noise level").  Torch pinned to 2 threads per
worker, as in spevae eval.

    python scripts/step2/floor_curve.py out/step2/data_arm ctrl_60k D24_60k
"""
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "src")


def job(path):
    import torch
    torch.set_num_threads(2)
    from spevae import load_model
    from spevae.evaluation.cases import holstein_cell
    from spevae.refine import RefineConfig, refine_latent
    mg = holstein_cell()
    m = load_model(path)
    rr = refine_latent(m, mg, RefineConfig(cov_method="structured", min_eigenvalue=1e-10,
                                           steps=3000))
    rb = refine_latent(m, mg, RefineConfig(cov_method="structured", min_eigenvalue=1e-10,
                                           steps=10000, prior_weight=0.0, patience=3000))
    return path, {"chi2_init": float(rr.chi2_init), "chi2_map": float(rr.chi2),
                  "floor": float(rb.chi2), "target": float(rb.chi2_target)}


def main():
    d = Path(sys.argv[1])
    runs = sys.argv[2:]
    paths = []
    for r in runs:
        snaps = sorted(d.glob(f"{r}.step*.pt"),
                       key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)))
        paths += [str(p) for p in snaps] + [str(d / f"{r}.pt")]
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(job, paths))
    out = {}
    for r in runs:
        rows = []
        for p, v in res.items():
            name = Path(p).name
            if not name.startswith(r + "."):
                continue
            m = re.search(r"step(\d+)", name)
            rows.append({"step": int(m.group(1)) if m else "final", **v})
        rows.sort(key=lambda x: (x["step"] == "final", x["step"] if x["step"] != "final" else 0))
        out[r] = rows
        for x in rows:
            print(f"{r:10s} step {str(x['step']):>6s}: amortized {x['chi2_init']:9.1f}  "
                  f"MAP {x['chi2_map']:8.3f}  floor {x['floor']:8.3f}  "
                  f"(target {x['target']:.3f})", flush=True)
    with open(d / "floor_curve.json", "w") as fh:
        json.dump(out, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for k, r in enumerate(runs):
        rows = [x for x in out[r] if x["step"] != "final"]
        st = [x["step"] for x in rows]
        axes[0].plot(st, [x["chi2_map"] / x["target"] for x in rows], "-o", color=cols[k],
                     ms=4, label=r)
        axes[1].plot(st, [x["floor"] / x["target"] for x in rows], "-o", color=cols[k], ms=4,
                     label=r)
    for ax, t in zip(axes, ["MAP (z-descent 3000, prior anchored)",
                            "manifold floor (prior released)"]):
        ax.set_yscale("log")
        ax.axhline(1.0, color="#52514e", ls="--", lw=0.9)
        ax.set_xlabel("training step")
        ax.set_ylabel(r"$\chi^2/\chi^2_{\rm target}$ on the Holstein cell")
        ax.set_title(t)
        ax.legend(fontsize=8)
    fig.suptitle("Real-data auditor along training (dashed: the noise level)", fontsize=10)
    fig.savefig(d / "floor_curve.png", dpi=150)
    print(f"-> {d}/floor_curve.json, .png")


if __name__ == "__main__":
    main()
