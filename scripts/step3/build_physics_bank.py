"""Step-3 label bank: smooth families + exactly solvable lattice models, on a beta grid.

For every beta in the grid and every family -- the five smooth families of
step 2 and the ten exact lattice families of `data/exact_models.py` -- draw
`--n-per-family` truths and extract the canonical pole label (the step-2
protocol, unchanged).  Each worker also returns

  * the truth's exact spectrum on the quadrature nodes the pole prior uses at
    that beta (written to `<bank>.nodes.pt`; the generator reads it instead of
    re-evaluating lattice sums during training), and
  * the label's data gap max|G_label - G_truth| on a 65-point grid.

Entries are kept ("trusted") when the gap is below `--gap-max` and the
label's negative weight is within the dip tolerance, as in step 2.

    python scripts/step3/build_physics_bank.py --out out/step3/bank --n-per-family 64 --workers 14
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch

BETAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0)
_QUAD = {}


def _quad(beta):
    if beta not in _QUAD:
        from spevae.data.pole_prior import PolePriorGenerator
        if "gen" not in _QUAD:
            _QUAD["gen"] = PolePriorGenerator(dtype=torch.float64)
        _QUAD[beta] = _QUAD["gen"].quadrature(beta)
    return _QUAD[beta]


def entry(job):
    """Label + exact spectrum on the nodes + data gap, for one truth."""
    import warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    from spevae.data.label_bank import extract_smooth_label
    from spevae.physics import PoleGreens, PoleSet
    truth, P_max, mode = job
    t0 = time.time()
    lab = extract_smooth_label(truth, P_max=P_max, residue_mode=mode)
    beta = truth.beta
    quad = _quad(beta)
    nu = quad.nodes.double().numpy()
    A = torch.tensor(truth.A(nu / beta) / beta, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)
    fwd = PoleGreens(quad)
    K = fwd.kernel_matrix(x)
    G_truth = (A * quad.weights.double()) @ K
    ps = PoleSet(eps=torch.tensor(lab.eps).double().unsqueeze(0) * beta,
                 gamma=torch.tensor(lab.gamma).double().unsqueeze(0) * beta,
                 a=torch.tensor(lab.a).double().unsqueeze(0),
                 b=torch.tensor(lab.b).double().unsqueeze(0))
    gap = float((fwd(ps, x, kernel_matrix=K)[0] - G_truth).abs().max())
    return lab, A.float().numpy(), gap, time.time() - t0


def main():
    from spevae.data.exact_models import EXACT_CONTINUUM
    from spevae.data.label_bank import SMOOTH_FAMILIES, LabelBank, draw_bank_truths

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/step3/bank"))
    ap.add_argument("--n-per-family", type=int, default=64)
    ap.add_argument("--betas", type=float, nargs="+", default=list(BETAS))
    ap.add_argument("--families", nargs="+", default=list(SMOOTH_FAMILIES + EXACT_CONTINUUM))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--p-max", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--gap-max", type=float, default=3e-5)
    ap.add_argument("--residue-mode", default="complex")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    truths = draw_bank_truths(args.n_per_family, args.seed, families=tuple(args.families),
                              beta_choices=tuple(args.betas))
    jobs = [(t, args.p_max, args.residue_mode) for t in truths]
    print(f"{len(jobs)} truths: {len(args.families)} families x {len(args.betas)} betas x "
          f"{args.n_per_family}; {args.workers} workers", flush=True)
    t0 = time.time()
    results = []
    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(entry, jobs, chunksize=4)):
            results.append(r)
            if (i + 1) % max(1, len(jobs) // 20) == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    P = args.p_max
    gap = np.array([r[2] for r in results])
    negw = np.array([r[0].neg_weight for r in results])
    keep = np.where((gap < args.gap_max) & (negw <= 5e-3))[0]

    def bank_of(idx, meta):
        n = len(idx)
        out = {k: torch.zeros(n, P) for k in ("eps", "gamma", "a", "b")}
        out["gamma"] += 1.0
        n_true = torch.zeros(n, dtype=torch.long)
        for row, i in enumerate(idx):
            lab = results[i][0]
            m = lab.n
            n_true[row] = m
            for k, v in (("eps", lab.eps), ("gamma", lab.gamma), ("a", lab.a), ("b", lab.b)):
                out[k][row, :m] = torch.tensor(v).float()
        return LabelBank(
            P_max=P, beta=torch.tensor([truths[i].beta for i in idx]),
            eps=out["eps"], gamma=out["gamma"], a=out["a"], b=out["b"], n_true=n_true,
            resid=torch.tensor([results[i][0].resid for i in idx]).float(),
            neg_weight=torch.tensor([results[i][0].neg_weight for i in idx]).float(),
            family=[truths[i].family for i in idx], params=[truths[i].params for i in idx],
            ladders=[results[i][0].ladder for i in idx], meta=meta)

    meta = {"seed": args.seed, "betas": list(args.betas), "families": list(args.families),
            "n_per_family": args.n_per_family, "residue_mode": args.residue_mode,
            "gap_max": args.gap_max, "kept": int(len(keep)), "of": len(jobs)}
    bank_of(range(len(jobs)), dict(meta, trusted=False)).save(args.out / "label_bank.pt")
    trusted = bank_of(list(map(int, keep)), dict(meta, trusted=True))
    trusted.save(args.out / "label_bank_trusted.pt")

    # exact spectra on the generator's quadrature nodes, per beta, indexed by bank row
    nodes = {}
    for row, i in enumerate(map(int, keep)):
        key = round(float(truths[i].beta), 6)
        d = nodes.setdefault(key, {"index": {}, "A": []})
        d["index"][row] = len(d["A"])
        d["A"].append(torch.from_numpy(results[i][1]))
    for d in nodes.values():
        d["A"] = torch.stack(d["A"])
    torch.save(nodes, str(args.out / "label_bank_trusted.pt") + ".nodes.pt")

    fams = list(dict.fromkeys(t.family for t in truths))
    report = {"meta": meta, "wall_seconds": time.time() - t0, "per_family": {}, "per_beta": {}}
    for f in fams:
        ix = [i for i, t in enumerate(truths) if t.family == f]
        report["per_family"][f] = {
            "kept": int(np.isin(ix, keep).sum()), "of": len(ix),
            "gap_median": float(np.median(gap[ix])), "gap_p95": float(np.quantile(gap[ix], 0.95)),
            "n_poles_median": float(np.median([results[i][0].n for i in ix])),
            "seconds_median": float(np.median([results[i][3] for i in ix]))}
    for b in args.betas:
        ix = [i for i, t in enumerate(truths) if t.beta == b]
        report["per_beta"][f"{b:g}"] = {"kept": int(np.isin(ix, keep).sum()), "of": len(ix),
                                        "gap_p95": float(np.quantile(gap[ix], 0.95))}
    with open(args.out / "bank_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({"kept": len(keep), "of": len(jobs), "wall_s": round(time.time() - t0),
                      "per_family_kept": {f: f"{v['kept']}/{v['of']}"
                                          for f, v in report["per_family"].items()}}, indent=1))
    print(f"-> {args.out}/label_bank.pt, label_bank_trusted.pt (+ .nodes.pt), bank_report.json")


if __name__ == "__main__":
    main()
