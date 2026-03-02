#!/bin/bash
#SBATCH --job-name=3D-vel-q0.2
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=96

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

q=0.2
N_v=2000
N=10000
l_max=10

e_array=($(LC_NUMERIC=C seq 0.50 0.01 0.99))
e=${e_array[$SLURM_ARRAY_TASK_ID]}

echo "Running ./scattering $q $e $N_v $N $l_max with $OMP_NUM_THREADS threads"
srun --cpu-bind=cores ./scattering $q $e $N_v $N $l_max
