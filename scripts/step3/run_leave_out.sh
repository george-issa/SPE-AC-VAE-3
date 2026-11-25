#!/bin/sh
# Leave-families-out arm of the full-length step-3 flow: the recipe of run_full.sh, seed 1,
# with five lattice families removed from the label bank and three exactly diagonalised
# families removed from the generator.  Judged on the known-answer models at the five
# settings of run_full.sh; the left-out families are then models the flow never saw.
#
#   sh scripts/step3/run_leave_out.sh [STEPS=60000]
#
# Writes out/step3/bank_leave_out/, out/step3/F3_<k>k_lo.pt and out/eval/exact_models_F3_<k>k_lo/.
set -u
STEPS=${1:-60000}
N=F3_$((STEPS / 1000))k_lo
B=out/step3/bank_leave_out/label_bank_trusted.pt
VAE=out/step2/data_arm/D24_60k.pt
export PYTHONPATH=src
python scripts/step3/leave_out_bank.py out/step3/bank/label_bank_trusted.pt out/step3/bank_leave_out \
    tb_honeycomb tb_square ssh_chain bcs falicov_kimball > out/step3/$N.bank.log 2>&1 || exit 1
python scripts/flow/train_flow.py --out out/step3/$N.pt --steps $STEPS --seed 1 --device mps \
    --smooth-bank $B --p-smooth 0.35 --p-exact 0.15 \
    --exact-families hubbard_atom hubbard_trimer anderson_cluster \
    --beta-choices 1 1.5 2 3 4 6 8 12 16 20 --noise-range 1e-5 3e-2 \
    --dtau-choices 0.02 0.05 0.1 0.125 0.2 0.25 0.5 --name step3-$((STEPS / 1000))k-leave-out \
    > out/step3/$N.log 2>&1 || { echo "training $N failed"; exit 1; }
echo "$(date +%H:%M:%S) trained $N"
for setting in "1e-3 8" "1e-4 8" "1e-2 8" "1e-3 2" "1e-3 16"; do
    python scripts/eval/exact_models_judge.py out/eval/exact_models_$N out/step3/$N.pt $VAE $setting \
        >> out/eval/exact_models_$N.log 2>&1
done
echo "$(date +%H:%M:%S) judged $N"
