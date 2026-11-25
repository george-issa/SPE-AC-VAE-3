#!/bin/sh
# Full-length step-3 flow, one or more seeds in sequence.  Every model is judged
# as soon as it exists -- calibration on the step-3 prior, then the known-answer
# models at five (noise, beta) settings -- while the next seed trains.
#
#   sh scripts/step3/run_full.sh "1 2 3" [STEPS=60000]
#
# Writes out/step3/F3_<k>k_s<seed>.pt, out/step3/cal_F3_<k>k_s<seed>/ and
# out/eval/exact_models_F3_<k>k_s<seed>/.
set -u
STEPS=${2:-60000}
B=out/step3/bank/label_bank_trusted.pt
VAE=out/step2/data_arm/D24_60k.pt
export PYTHONPATH=src

judge() {
    N=$1
    for s in 0 1 2; do
        python scripts/step3/calibration.py out/step3/$N.pt $B out/step3/cal_$N $s 3 \
            > out/step3/cal_$N.shard$s.log 2>&1 &
    done
    wait
    python scripts/step3/calibration.py fig out/step3/cal_$N > out/step3/cal_$N.fig.log 2>&1
    for setting in "1e-3 8" "1e-4 8" "1e-2 8" "1e-3 2" "1e-3 16"; do
        python scripts/eval/exact_models_judge.py out/eval/exact_models_$N out/step3/$N.pt $VAE $setting \
            >> out/eval/exact_models_$N.log 2>&1
    done
}

for SEED in $1; do
    N=F3_$((STEPS / 1000))k_s$SEED
    python scripts/flow/train_flow.py --out out/step3/$N.pt --steps $STEPS --seed $SEED --device mps \
        --smooth-bank $B --p-smooth 0.35 --p-exact 0.15 \
        --beta-choices 1 1.5 2 3 4 6 8 12 16 20 --noise-range 1e-5 3e-2 \
        --dtau-choices 0.02 0.05 0.1 0.125 0.2 0.25 0.5 --name step3-$((STEPS / 1000))k-s$SEED \
        > out/step3/$N.log 2>&1 || { echo "training $N failed"; continue; }
    echo "$(date +%H:%M:%S) trained $N"
    judge $N &
done
wait
echo "$(date +%H:%M:%S) all done"
