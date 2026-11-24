"""Reproduce the banked judge numbers with `spevae eval` (run from the repo root).

    python scripts/eval/acceptance.py [OUT_DIR] [WORKERS]
"""
import sys
from collections import Counter

sys.path.insert(0, "src")

from spevae.evaluation.acceptance import run_acceptance


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "out/eval/acceptance"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = run_acceptance(out, workers)
    for r in rows:
        if r["status"] != "PASS":
            print(f"{r['status']:7s} {r['model']:9s} {r['check']:45s} banked {r['banked']!r:>22} "
                  f"got {r['got']!r:>22} rel {r['rel_diff']:.2e} ({r['tolerance']})")
    c = Counter(r["status"] for r in rows)
    print(f"acceptance: {c.get('PASS', 0)} pass, {c.get('FAIL', 0)} fail, "
          f"{c.get('DRIFT', 0)} drift (old environment), {c.get('MISSING', 0)} missing "
          f"of {len(rows)} -> {out}/acceptance.json")


if __name__ == "__main__":       # worker processes re-import this file (spawn)
    main()
