#!/bin/bash
#SBATCH --job-name=bonetti
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=96

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

q_array=(1e-4 5e-4 1e-3 3e-3 5e-3 1e-2 5e-2 1e-1 1)
e=0.6

q=${q_array[$SLURM_ARRAY_TASK_ID]}

echo "Running ./scattering_bonetti $q $e with $OMP_NUM_THREADS threads"
srun --cpu-bind=cores ./scattering_bonetti $q $e
