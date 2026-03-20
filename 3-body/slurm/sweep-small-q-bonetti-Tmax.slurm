#!/bin/bash
#SBATCH --job-name=small-q-bonetti-Tmax
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --time=167:59:00
#SBATCH --cpus-per-task=96

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

q_array=(0.0001 0.0002 0.0005)
e_array=(0.6    0.6    0.6)

q=${q_array[$SLURM_ARRAY_TASK_ID]}
e=${e_array[$SLURM_ARRAY_TASK_ID]}

echo "Running job $SLURM_ARRAY_TASK_ID with q=$q, e=$e and $OMP_NUM_THREADS threads"

srun --cpu-bind=cores ./scattering $q $e
