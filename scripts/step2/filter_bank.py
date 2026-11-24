"""Keep the bank labels that reproduce their truth below the quietest training noise.

A label whose noise-free data gap max|G_label - G_truth| reaches the quietest
training noise (3e-5) teaches a target the data contradicts at low noise.
Measured on bank_b8: 1.7% of gauss_mixture, 1.2% of gapped_pair, 0.3% of
pseudogap, none of semicircle/box -- the floor-limited tail visible in the
overlay figure's worst-residual column.

    python scripts/step2/filter_bank.py out/step2/bank_b8
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from spevae.data.label_bank import LabelBank

GAP_MAX = 3e-5

d = Path(sys.argv[1])
bank = LabelBank.load(d / "label_bank.pt")
judge = np.load(d / "label_bank_judge.npz")
keep = np.where((judge["g_gap"] < GAP_MAX) & (bank.neg_weight.numpy() <= 5e-3))[0]
ix = list(map(int, keep))
sub = LabelBank(
    P_max=bank.P_max, beta=bank.beta[keep], eps=bank.eps[keep], gamma=bank.gamma[keep],
    a=bank.a[keep], b=bank.b[keep], n_true=bank.n_true[keep], resid=bank.resid[keep],
    neg_weight=bank.neg_weight[keep], family=[bank.family[i] for i in ix],
    params=[bank.params[i] for i in ix], ladders=[bank.ladders[i] for i in ix],
    meta={**bank.meta, "filter": {"g_gap_max": GAP_MAX, "neg_weight_max": 5e-3,
                                  "source": str(d / "label_bank.pt"),
                                  "kept": len(ix), "of": len(bank)}})
sub.save(d / "label_bank_trusted.pt")
per = {f: (sum(1 for i in ix if bank.family[i] == f), bank.family.count(f))
       for f in dict.fromkeys(bank.family)}
print(json.dumps({"kept": len(ix), "of": len(bank), "per_family": per}))
