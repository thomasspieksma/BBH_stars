#!/bin/bash
#SBATCH --job-name=three-body
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=96

# Use all allocated cores for OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running job $SLURM_ARRAY_TASK_ID with q=$q, e=$e and $OMP_NUM_THREADS threads"

# Run simulation
srun --cpu-bind=cores ./scattering 0.2 0.6