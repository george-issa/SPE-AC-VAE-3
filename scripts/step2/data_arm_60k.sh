#!/bin/zsh
# Step-2 data arm promoted to the judged 60k tier (build-order item 3).
# D24 (M1 recipe + trusted smooth bank at p_smooth 0.3, latent 24) won the 6k
# screen (out/eval/step2_screen6k/).  The control is the M1 recipe retrained
# with the same code, so both runs carry snapshots every 5k steps: the 6k
# screen found the Holstein floor AT target for every 6k model while M1-60k
# sits at 3.83, so the auditor is read along each run, not only at the end.
cd "$(dirname "$0")/../.."
export PYTHONPATH=src OMP_NUM_THREADS=2
OUT=out/step2/data_arm
mkdir -p $OUT
FLAGS=(--preset fast --steps 60000 --beta-min 8 --beta-max 8 --data poles --pole-metric width
       --residue-mode dispersive --prior-residue-mode dispersive --latent-dim 24
       --pole-eps1 0.1 --seed 1 --device mps --keep-snapshots)
python -u -m spevae.cli pretrain $FLAGS --out $OUT/ctrl_60k.pt > $OUT/ctrl_60k.log 2>&1 &
python -u -m spevae.cli pretrain $FLAGS --smooth-bank out/step2/bank_b8/label_bank_trusted.pt \
    --p-smooth 0.3 --out $OUT/D24_60k.pt > $OUT/D24_60k.log 2>&1 &
wait
echo DATA_ARM_60K_FINISHED
