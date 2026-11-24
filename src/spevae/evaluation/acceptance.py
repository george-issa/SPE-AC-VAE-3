r"""Acceptance: the suite must reproduce the banked numbers before it judges.

Runs the ``accept`` tier (the banked judges' exact protocols on the banked
noise draws) on the two reference checkpoints and compares every number
against the JSON artifacts it was banked in.  Tolerance classes:

- ``exact`` (rtol 1e-9): pipeline outputs banked in this environment
  (torch 2.12, 2 threads) -- chi^2 values, selected pole counts, streams,
  every Holstein number including the manifold floor -- must reproduce
  bit for bit;
- ``w1`` (rtol 1e-4): W1 values.  The judges evaluated spectra on float32
  nodes (most) or float64 nodes (dual output) and the suite evaluates the
  same poles in one convention; measured differences are <= 6e-5;
- ``old_env`` (rtol 2e-3): the map-screen judge was banked on 2026-09-01
  under the previous Python/torch environment.  Values beyond tolerance
  there are reported as DRIFT, not FAIL: they measure how strongly a
  pipeline amplifies float-level differences (thread count was ruled out --
  z-descent is identical at 1, 2, 4 and 8 threads).

Banked artifacts live in ``out/`` and are not versioned, so this runs only
on the machine that holds them.
"""

from __future__ import annotations

import json
import math
import os

__all__ = ["REFERENCE_MODELS", "check", "run_acceptance"]

REFERENCE_MODELS = {"s1_metric": "out/step1/s1_metric.pt",
                    "M1_60k": "out/step2/map_M1_60k.pt"}
RTOL = {"exact": 1e-9, "w1": 1e-4, "old_env": 2e-3}
J8 = ("3-rung ladder", "4-rung ladder", "edged band", "irregular triplet", "asym triplet",
      "fano band", "lorentz pair", "gauss pair", "gapped pair", "semicircle", "box",
      "pseudogap")


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _expected(root="out"):
    """(model, label, selector, field, banked value, tolerance class, source)."""
    rows = []
    for tag in REFERENCE_MODELS:
        mj = f"{root}/step2/m1judge/{tag}_judge.json"
        if os.path.exists(mj):
            d = _load(mj)
            rows += [(tag, "pole stream W1", ("streams", "pole"), "w1_omega",
                      d["stream"]["w1_omega"], "exact", mj),
                     (tag, "legacy canary W1", ("streams", "canary"), "w1_omega",
                      d["canary"]["w1_omega"], "exact", mj)]
        mp = f"{root}/step2/map_judge60k.json"
        if os.path.exists(mp):
            d = _load(mp)[tag]
            for key, v in d.items():
                case, nz = key.rsplit("@", 1)
                sel = ("synthetic", "zd800", case, float(nz), 7)
                rows += [(tag, f"zd800 {key} W1", sel, "w1", v["w1"], "old_env", mp),
                         (tag, f"zd800 {key} chi2", sel, "chi2_internal", v["chi2"],
                          "old_env", mp),
                         (tag, f"zd800 {key} consistency", sel, "consistency_w1",
                          v["consistency_w1"], "old_env", mp)]
        ep = f"{root}/step2/equivariance/{tag}_probes.json"
        if os.path.exists(ep):
            d = _load(ep)
            for case in J8:
                for nz in (1e-4, 1e-3):
                    r = d[f"{case}@{nz:g}"]
                    rows += [(tag, f"amortized {case}@{nz:g} W1",
                              ("synthetic", "amortized", case, nz, 7), "w1",
                              r["amort"]["w1_f"], "w1", ep),
                             (tag, f"zd3000 {case}@{nz:g} W1",
                              ("synthetic", "zd3000", case, nz, 7), "w1",
                              r["zd"]["w1_f"], "w1", ep),
                             (tag, f"zd3000 {case}@{nz:g} chi2",
                              ("synthetic", "zd3000", case, nz, 7), "chi2_internal",
                              r["zd"]["chi2_f"], "exact", ep)]
        eh = f"{root}/step2/equivariance/{tag}_holstein.json"
        if os.path.exists(eh):
            f = _load(eh)["f"]
            rows += [(tag, "Holstein amortized chi2", ("real", "amortized"), "chi2_internal",
                      f["chi2_init"], "exact", eh),
                     (tag, "Holstein MAP chi2 (zd3000)", ("real", "zd3000"), "chi2_internal",
                      f["chi2_map"], "exact", eh),
                     (tag, "Holstein manifold floor", ("floor",), "floor", f["floor"],
                      "exact", eh)]
        dj = f"{root}/step2/m1judge/{tag}_dual.json"
        if os.path.exists(dj):
            for c in _load(dj)["cases"]:
                if c["name"] == "Holstein":
                    rows += [(tag, "Holstein certified chi2", ("real", "scan_minimal"),
                              "chi2_internal", c["minimal"]["chi2"], "exact", dj),
                             (tag, "Holstein certified n", ("real", "scan_minimal"),
                              "selected_n", c["minimal"]["n"], "exact", dj),
                             (tag, "Holstein full-rendering chi2", ("real", "scan_full"),
                              "chi2_internal", c["full"]["chi2"], "exact", dj)]
                else:
                    sel = ("synthetic", "scan_minimal", c["name"], 1e-4, 7)
                    rows += [(tag, f"certified {c['name']} n", sel, "selected_n",
                              c["minimal"]["n"], "exact", dj),
                             (tag, f"certified {c['name']} chi2", sel, "chi2_internal",
                              c["minimal"]["chi2"], "exact", dj),
                             (tag, f"certified {c['name']} W1", sel, "w1",
                              c["minimal"]["w1"], "w1", dj),
                             (tag, f"full rendering {c['name']} W1",
                              ("synthetic", "scan_full", c["name"], 1e-4, 7), "w1",
                              c["full"]["w1"], "w1", dj)]
    return rows


def _find(records, tag, sel):
    kind = sel[0]
    for r in records:
        if r.get("kind") != kind or r.get("model") != tag:
            continue
        if kind == "streams":
            return r[sel[1]]
        if kind == "floor":
            return r
        if kind == "real" and r["method"] == sel[1]:
            return r
        if (kind == "synthetic" and r["method"] == sel[1] and r["case"] == sel[2]
                and math.isclose(r["noise"], sel[3]) and r["seed"] == sel[4]):
            return r
    return None


def check(card: dict, root: str = "out") -> list[dict]:
    out = []
    for tag, label, sel, field, want, tol, src in _expected(root):
        r = _find(card["records"], tag, sel)
        got = None if r is None else r.get(field)
        if got is None:
            status, rel = "MISSING", float("nan")
        else:
            rel = abs(got - want) / max(abs(want), 1e-300)
            status = ("PASS" if rel <= RTOL[tol] else
                      "DRIFT" if tol == "old_env" else "FAIL")
        out.append({"model": tag, "check": label, "banked": want, "got": got,
                    "rel_diff": rel, "tolerance": tol, "status": status, "source": src})
    return out


def run_acceptance(out_dir: str, workers: int | None = None, root: str = "out") -> list[dict]:
    from .report import write_report
    from .suite import run_suite
    models = [(t, p) for t, p in REFERENCE_MODELS.items() if os.path.exists(p)]
    card = run_suite(models, tier="accept", workers=workers, skip=())
    write_report(card, out_dir)
    rows = check(card, root)
    with open(os.path.join(out_dir, "acceptance.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows
