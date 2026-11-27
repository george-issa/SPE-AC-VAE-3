#!/bin/sh
# The posterior report with zero-width lines, on the known answers of all twelve settings (the
# five standard ones and the seven at the edges of the trained range): at its defaults on every
# spectrum, and with the spectrum stated to be discrete on the line spectra, against the stored
# results of the report before and of MaxEnt.
#
#   sh scripts/step3/run_line_fit.sh out/step3/F3L_60k_s1.pt out/step3/line_fit_F3L_60k_s1 \
#       out/step3/floor_option_F3L_60k_s1_table out/step3/edges_F3L_60k_s1 [always=DIR]
#
# Six processes at a time, three threads each.  Writes OUT/line_fit[_stated]_b<beta>_n<noise>
# [_dt<dtau>].json per setting and, through line_fit_judge.py fig, OUT/line_fit.{json,png}.
set -u
FLOW=$1
OUT=$2
shift 2
mkdir -p "$OUT"
export FLOW OUT
for arm in "" stated; do
    printf '%s\n' "8 1e-5 0.125" "20 1e-5 0.05" "8 1e-4 0.125" "8 1e-3 0.125" "8 1e-2 0.125" "2 1e-3 0.125" \
        "16 1e-3 0.125" "1 1e-3 0.05" "20 1e-3 0.125" "8 1e-3 0.02" "8 1e-3 0.5" "1 1e-2 0.1" |
        ARM=$arm xargs -P 6 -L 1 sh -c 'JUDGE_DTAU=$3 python scripts/step3/line_fit_judge.py run "$OUT" "$FLOW" \
            $1 $2 $ARM > "$OUT/run_${ARM:-default}_b$1_n$2_dt$3.log" 2>&1' _
    echo "$(date +%H:%M:%S) judged ${arm:-at the defaults}"
done
python scripts/step3/line_fit_judge.py fig "$OUT" "$@"
