
import glob

import os

import h5py

import numpy as np



folderpath = os.getcwd()  # run from .../movie/

FolderList = sorted(glob.glob(os.path.join(folderpath, "Visu*")))

# Use only first 95% of available Visu folders
N_total = len(FolderList)
N_use = int(0.95 * N_total)

FolderList = FolderList[:N_use]

print(f"Found {N_total} Visu folders")
print(f"Processing first {N_use} folders (95%)")


TargetFolder = os.path.join(folderpath, "movie_test")

os.makedirs(TargetFolder, exist_ok=True)



for i, visu_dir in enumerate(FolderList):
    print(visu_dir)
    visu_file = os.path.join(visu_dir, "visState0000.hdf5")
    with h5py.File(visu_file, "r") as hf:
        field = np.array(hf["temperature"]["temperature"])

        time = np.array(hf["run"]["time"])
        timestep = np.array(hf["run"]["timestep"])
        grid_theta = np.array(hf["mesh"]["grid_theta"]) - np.pi / 2
        grid_phi = np.array(hf["mesh"]["grid_phi"])
        grid_r = np.array(hf["mesh"]["grid_r"])

        Ek = np.array(hf["physical"]["ekman"])
        Ra = np.array(hf["physical"]["rayleigh"])
        Pr = np.array(hf["physical"]["prandtl"])


        EQindex = int(np.shape(grid_theta)[0] / 2)
        temperature = 0.5 * (field[:, EQindex - 2, :] + field[:, EQindex - 3, :])


    out = os.path.join(TargetFolder, f"test_{i:04d}.npz")
    np.savez(
        out,
        temperature=temperature,
        time=time,
        timestep=timestep,
        grid_theta=grid_theta,
        grid_phi=grid_phi,
        grid_r=grid_r,
        Ek=Ek,
        Ra=Ra,
        Pr=Pr,
    )

