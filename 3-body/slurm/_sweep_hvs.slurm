#!/bin/bash
#SBATCH --job-name=hvs
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=96

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

q_array=(1e-3 1e-2)
e=0.9

q=${q_array[$SLURM_ARRAY_TASK_ID]}

echo "Running ./scattering_hvs $q $e with $OMP_NUM_THREADS threads"
srun --cpu-bind=cores ./scattering_hvs $q $e
