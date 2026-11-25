"""A copy of the step-3 label bank with some families removed (for a leave-out arm).

    python scripts/step3/leave_out_bank.py out/step3/bank/label_bank_trusted.pt \
        out/step3/bank_leave_out tb_honeycomb bcs ...

Writes <out>/label_bank_trusted.pt and its .nodes.pt sidecar (the truths'
exact spectra on the quadrature nodes, re-indexed), and prints what is left.
"""
import collections
import sys
from pathlib import Path

import torch


def main():
    src, out, drop = sys.argv[1], Path(sys.argv[2]), set(sys.argv[3:])
    out.mkdir(parents=True, exist_ok=True)
    b = torch.load(src, weights_only=False)
    fam = b["family"]
    missing = drop - set(fam)
    if missing:
        raise SystemExit(f"not in the bank: {sorted(missing)}")
    keep = [i for i, f in enumerate(fam) if f not in drop]
    new_of = {old: new for new, old in enumerate(keep)}
    ix = torch.tensor(keep)
    nb = {}
    for k, v in b.items():
        if torch.is_tensor(v) and v.shape[:1] == (len(fam),):
            nb[k] = v[ix].clone()
        elif isinstance(v, list) and len(v) == len(fam):
            nb[k] = [v[i] for i in keep]
        else:
            nb[k] = v
    nb["meta"] = {**b["meta"], "left_out": sorted(drop), "source": src}
    torch.save(nb, out / "label_bank_trusted.pt")
    nodes = torch.load(src + ".nodes.pt", weights_only=False)
    nn = {}
    for beta, pre in nodes.items():
        pairs = sorted((new_of[j], k) for j, k in pre["index"].items() if j in new_of)
        nn[beta] = {"index": {j: n for n, (j, _) in enumerate(pairs)},
                    "A": pre["A"][torch.tensor([k for _, k in pairs])].clone()}
    torch.save(nn, str(out / "label_bank_trusted.pt") + ".nodes.pt")
    print(f"kept {len(keep)} of {len(fam)} labels; left out {sorted(drop)}")
    print(dict(collections.Counter(nb["family"])))


if __name__ == "__main__":
    main()
