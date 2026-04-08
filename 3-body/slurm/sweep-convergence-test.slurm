#!/bin/bash
#SBATCH --job-name=conv-test
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --array=0-25
#SBATCH --nodes=1
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=96

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# q=0.001, e=0.6: tasks 0-12
# q=0.01,  e=0.6: tasks 13-25

q_array=(
    0.001 0.001 0.001 0.001 0.001
    0.001 0.001
    0.001 0.001
    0.001 0.001
    0.001 0.001
    0.01  0.01  0.01  0.01  0.01
    0.01  0.01
    0.01  0.01
    0.01  0.01
    0.01  0.01
)

e_array=(
    0.6 0.6 0.6 0.6 0.6
    0.6 0.6
    0.6 0.6
    0.6 0.6
    0.6 0.6
    0.6 0.6 0.6 0.6 0.6
    0.6 0.6
    0.6 0.6
    0.6 0.6
    0.6 0.6
)

integrator_array=(
    rk45 rk45 rk45 rk45 rk45
    verlet verlet
    pihajoki pihajoki
    yoshida yoshida
    bulirsch_stoer bulirsch_stoer
    rk45 rk45 rk45 rk45 rk45
    verlet verlet
    pihajoki pihajoki
    yoshida yoshida
    bulirsch_stoer bulirsch_stoer
)

tol_array=(
    0.05 0.02 0.01 0.005 0.002
    0.005 0.002
    0.02 0.01
    0.01 0.005
    0.01 0.005
    0.05 0.02 0.01 0.005 0.002
    0.005 0.002
    0.02 0.01
    0.01 0.005
    0.01 0.005
)

q=${q_array[$SLURM_ARRAY_TASK_ID]}
e=${e_array[$SLURM_ARRAY_TASK_ID]}
integrator=${integrator_array[$SLURM_ARRAY_TASK_ID]}
tol=${tol_array[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: q=$q, e=$e, integrator=$integrator, tol=$tol, threads=$OMP_NUM_THREADS"
srun --cpu-bind=cores ./scattering_conv $q $e $integrator $tol
