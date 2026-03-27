#!/bin/bash
#SBATCH --job-name=three-body-small-q
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=96

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

q_array=(0.02 0.02 0.02 0.05 0.05 0.05)
e_array=(0.1  0.5  0.7  0.1  0.5  0.7)

q=${q_array[$SLURM_ARRAY_TASK_ID]}
e=${e_array[$SLURM_ARRAY_TASK_ID]}

echo "Running job $SLURM_ARRAY_TASK_ID with q=$q, e=$e and $OMP_NUM_THREADS threads"
srun --cpu-bind=cores ./scattering $q $e
