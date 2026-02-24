#!/bin/bash
#SBATCH --job-name=three-body
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=96

# Use all allocated cores for OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create arrays of parameter values
q_array=($(LC_NUMERIC=C seq 0.1 0.1 1.0))
e_array=($(LC_NUMERIC=C seq 0.0 0.1 0.9))

# Determine indices
q_index=$(( SLURM_ARRAY_TASK_ID / 10 ))
e_index=$(( SLURM_ARRAY_TASK_ID % 10 ))

# Get actual parameters
q=${q_array[$q_index]}
e=${e_array[$e_index]}

echo "Running job $SLURM_ARRAY_TASK_ID with q=$q, e=$e and $OMP_NUM_THREADS threads"

# Run simulation
srun --cpu-bind=cores ./scattering $q $e