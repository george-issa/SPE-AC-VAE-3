#!/bin/sh
# Step-3 flow with the exactly diagonalised (line) models taught the way the lattice models are:
# exact data from the lines, the canonical pole label as target (bank built by
# build_physics_bank.py --families <line families> --base <step-3 bank>).  The recipe of
# run_full.sh otherwise, except the mixture: 0.45 from the bank (which now holds line labels),
# 0.10 broadened lines (was 0.35 / 0.15).
#
#   sh scripts/step3/run_lines.sh "1" [STEPS=60000]
#
# Writes out/step3/F3L_<k>k_s<seed>.pt, out/step3/cal_F3L_<k>k_s<seed>/ (truths from this
# recipe's prior), out/eval/exact_models_F3L_<k>k_s<seed>/ and out/step3/floor_option_F3L_<k>k_s<seed>/.
set -u
STEPS=${2:-60000}
B=out/step3/bank_lines/label_bank_merged.pt
VAE=out/step2/data_arm/D24_60k.pt
export PYTHONPATH=src
for SEED in $1; do
    N=F3L_$((STEPS / 1000))k_s$SEED
    python scripts/flow/train_flow.py --out out/step3/$N.pt --steps $STEPS --seed $SEED --device mps \
        --smooth-bank $B --p-smooth 0.45 --p-exact 0.10 \
        --beta-choices 1 1.5 2 3 4 6 8 12 16 20 --noise-range 1e-5 3e-2 \
        --dtau-choices 0.02 0.05 0.1 0.125 0.2 0.25 0.5 --name step3-lines-$((STEPS / 1000))k-s$SEED \
        > out/step3/$N.log 2>&1 || { echo "training $N failed"; continue; }
    echo "$(date +%H:%M:%S) trained $N"
    for s in 0 1 2; do
        CAL_P_SMOOTH=0.45 CAL_P_EXACT=0.10 python scripts/step3/calibration.py out/step3/$N.pt $B \
            out/step3/cal_$N $s 3 > out/step3/cal_$N.shard$s.log 2>&1 &
    done
    wait
    python scripts/step3/calibration.py fig out/step3/cal_$N > out/step3/cal_$N.fig.log 2>&1
    for setting in "1e-3 8" "1e-4 8" "1e-2 8" "1e-3 2" "1e-3 16"; do
        python scripts/eval/exact_models_judge.py out/eval/exact_models_$N out/step3/$N.pt $VAE $setting \
            >> out/eval/exact_models_$N.log 2>&1
    done
    echo "$(date +%H:%M:%S) judged $N on the known answers"
    mkdir -p out/step3/floor_option_$N
    for setting in "8 1e-4" "8 1e-3" "8 1e-2" "2 1e-3" "16 1e-3"; do
        python scripts/step3/floor_option_judge.py out/step3/floor_option_$N out/step3/$N.pt $setting 2 \
            > out/step3/floor_option_$N/run_$(echo $setting | tr ' ' '_').log 2>&1 &
    done
    wait
    python scripts/step3/floor_option_judge.py fig out/step3/floor_option_$N > out/step3/floor_option_$N/fig.log 2>&1
    echo "$(date +%H:%M:%S) judged $N through the report"
done
