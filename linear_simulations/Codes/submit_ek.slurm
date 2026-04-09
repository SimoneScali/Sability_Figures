#!/bin/bash
#SBATCH --account=project_465001528
#SBATCH --partition=standard
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=12:00:00

#SBATCH --job-name=LIN_EK
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load LUMI
module load PrgEnv-gnu
module load cray-python
module load cray-fftw
module load cray-hdf5-parallel
module load Boost
module load buildtools
module load craype-x86-milan

source /scratch/project_465001528/scalisim/env_petsc_slepc.sh

EK=$1



echo "Running EK = ${EK}"

python3 linear_controller_fixed.py "${EK}"
