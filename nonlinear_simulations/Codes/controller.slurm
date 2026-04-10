#!/bin/bash -l

#SBATCH --job-name=FAST

#SBATCH --account=project_465001528

#SBATCH --partition=standard

#SBATCH --time=12:00:00

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --output=controller_%j.out

#SBATCH --error=controller_%j.err



set -euo pipefail



module load LUMI

module load cray-python



cd /scratch/project_465001528/scalisim/THE/FAST



echo "Starting stationary controller at $(date)"

python illinois_pr_parallel.py   #find_stationary.py

echo "Controller finished at $(date)"

