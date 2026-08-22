#!/bin/bash
#SBATCH -A <YOUR_ALLOCATION>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 3
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -J spevae-identity-grid
# Ladder identity grid: architecture x final-OT-eps x seed (12 runs, 4/node).
# Judge afterwards with scripts/microsweep_report.py analogs + ladder probes.
module load pytorch
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=src
i=0
for arch in fast full; do
  for eps in 0.1 0.01; do
    for seed in 1 2 3; do
      steps=$([ $arch = fast ] && echo 60000 || echo 30000)
      srun -N1 -n1 --gpus-per-task=1 --exact \
        python -u -m spevae.cli pretrain --preset $arch --steps $steps \
          --batch-size 128 --seed $seed --device cuda \
          --data poles --motifs ladder --pole-eps1 $eps \
          --beta-min 8 --beta-max 8 --dtau 0.125 \
          --out out/grid/id_${arch}_e${eps}_s${seed}.pt \
          > out/grid/id_${arch}_e${eps}_s${seed}.log 2>&1 &
      i=$((i+1))
    done
  done
done
wait
echo "GRID COMPLETE: $i runs"
