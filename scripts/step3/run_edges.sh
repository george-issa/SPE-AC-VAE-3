#!/bin/sh
# Known answers at the edges of the trained range, through the posterior report.
# The five standard settings of the judges sit in the middle of it (beta 2-16,
# error bars 1e-4 to 1e-2, dtau 0.125); these are the corners a user can reach:
# the hottest and coldest temperatures, the smallest error bar, the finest and
# coarsest time step, and the two extreme combinations.
#
#   sh scripts/step3/run_edges.sh out/step3/F3L_60k_s1.pt out/step3/edges_F3L_60k_s1
#
# Writes OUT/floor_option_b<beta>_n<noise>[_dt<dtau>].json per setting (the
# judge of floor_option_judge.py, MaxEnt on every spectrum) and, through
# edges_summary.py, OUT/edges.{json,png}.
set -u
FLOW=$1
OUT=$2
mkdir -p "$OUT"
for st in "1 1e-3 0.05" "20 1e-3 0.125" "8 1e-5 0.125" "8 1e-3 0.02" "8 1e-3 0.5" "20 1e-5 0.05" "1 1e-2 0.1"; do
    set -- $st
    JUDGE_DTAU=$3 python scripts/step3/floor_option_judge.py "$OUT" "$FLOW" $1 $2 2 \
        > "$OUT/run_b$1_n$2_dt$3.log" 2>&1 &
done
wait
echo "$(date +%H:%M:%S) edges judged"
[ -f scripts/step3/edges_summary.py ] && python scripts/step3/edges_summary.py "$OUT"
