#!/bin/bash -l

#SBATCH --job-name="visu_rtc"
#SBATCH --account=project_465001528
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --output=slurm_visu.%A_%a.out
#SBATCH --error=slurm_visu.%A_%a.err



# These three get set by the driver via sbatch --export
# START, END, STEP

#SBATCH --array=0-0

set -euo pipefail

cd "$(pwd)/movie"


i=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")

mkdir -p "Visu$i"

cd "Visu$i"

cp ../parameters.cfg .

ln -sf "../state${i}.hdf5" state4Visu.hdf5


#run as sbatch --dependency=afterok:<simid> --array=200-399:5 VisuLumi_array.slurm

srun /scratch/project_465001528/scalisim/Executables/ex_RTC/BoussinesqSphereRTCExplicitVisu

