#!/bin/zsh
# Step-2 data arm, 6k screen (build-order item 3, screened before the 60k tier).
#
# One variable: the training data gains smooth label-bank truths (v1-protocol
# labels rebuilt at beta = 8 and filtered to data gap < 3e-5, out/step2/bank_b8/
# label_bank_trusted.pt) at p_smooth = 0.3.  The latent
# width is the in-step factorial (24 / 32 / 48).  Everything else is the M1
# recipe.  Controls: the M1 recipe at 6k, seeds 1 and 2, already trained
# (out/step2/equiv_ctrl6k.pt, out/step2/conc/ctrl_s2_6k.pt).
#
# Judged afterwards with:  spevae eval --tier fast (+ floor) -- see TRAINING.md.
cd "$(dirname "$0")/../.."
export PYTHONPATH=src OMP_NUM_THREADS=2
OUT=out/step2/data_arm
mkdir -p $OUT
BANK=out/step2/bank_b8/label_bank_trusted.pt
FLAGS=(--preset fast --steps 6000 --beta-min 8 --beta-max 8 --data poles --pole-metric width
       --residue-mode dispersive --prior-residue-mode dispersive
       --pole-eps1 0.1 --seed 1 --device mps
       --smooth-bank $BANK --p-smooth 0.3)
for L in 24 32 48; do
  python -u -m spevae.cli pretrain $FLAGS --latent-dim $L --out $OUT/D${L}_6k.pt \
      > $OUT/D${L}_6k.log 2>&1 &
done
wait
echo DATA_ARM_SCREEN_FINISHED
