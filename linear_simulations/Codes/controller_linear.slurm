#!/bin/bash
#SBATCH --job-name=LIN_CONTROLLER
#SBATCH --account=project_465001528
#SBATCH --partition=standard
#SBATCH --time=00:10:00
#SBATCH --ntasks=1

EK_LIST=("1e-4")

for EK in "${EK_LIST[@]}"; do

    JOBNAME="Ek${EK}"

    echo "Submitting ${JOBNAME}"

    sbatch \
        --job-name="${JOBNAME}" \
        submit_ek.slurm "${EK}"

done
