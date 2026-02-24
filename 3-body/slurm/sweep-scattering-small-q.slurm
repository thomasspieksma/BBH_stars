#!/bin/bash
#SBATCH --job-name=three-body-small-q
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=96

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 20 (q, e) pairs for small mass ratios
q_array=(0.0005 0.0005 0.001 0.001 0.001 0.002 0.002 0.002 0.005 0.005 0.005 0.01 0.01 0.01 0.02 0.02 0.02 0.05 0.05 0.05)
e_array=(0.3    0.6    0.3   0.6   0.9   0.3   0.6   0.9   0.3   0.6   0.9   0.3  0.6  0.9  0.3  0.6  0.9  0.3  0.6  0.9)

q=${q_array[$SLURM_ARRAY_TASK_ID]}
e=${e_array[$SLURM_ARRAY_TASK_ID]}

echo "Running job $SLURM_ARRAY_TASK_ID with q=$q, e=$e and $OMP_NUM_THREADS threads"

srun --cpu-bind=cores ./scattering $q $e
