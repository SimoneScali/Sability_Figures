#!/usr/bin/env python3

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
import xml.etree.ElementTree as ET


BASE = Path("/scratch/project_465001528/scalisim/THE/FAST/fs_RTC")

TEMPLATES = Path("/scratch/project_465001528/scalisim/THE/FAST/templates_rtc")

EXEC_MODEL = Path("/scratch/project_465001528/scalisim/Executables/ex_RTC/BoussinesqSphereRTCExplicitModel")


DEFAULT_VISU_START = 200
DEFAULT_VISU_END   = 399
DEFAULT_VISU_STEP  = 5

POLL_SECONDS = 60

# ---------------------------------------------------

def ra_to_folder(ra: float) -> str:
    s = f"{ra:.4f}"           
    a, b = s.split(".")
    return f"Ra{int(a)}_{b}"  


def pr_to_folder(pr: float) -> str:
    pr = float(pr)
    s = f"{pr:.4f}".rstrip("0").rstrip(".")
    return f"Pr{s}"

def ek_to_folder(ek: float) -> str:

    s = f"{ek:.3e}"                 
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    mant, exp = s.split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"Ek{mant}e{exp}"

def set_physical_params(cfg_path: Path, *, ekman: float, prandtl: float, rayleigh: float) -> None:

    tree = ET.parse(cfg_path)
    root = tree.getroot()
    sim = root.find("simulation")
    if sim is None:
        raise RuntimeError(f"No <simulation> tag in {cfg_path}")
    phys = sim.find("physical")
    if phys is None:
        raise RuntimeError(f"No <physical> tag in {cfg_path}")

    def set_tag(tag: str, value: str):
        el = phys.find(tag)
        if el is None:
            el = ET.SubElement(phys, tag)
        el.text = value

    set_tag("ekman", str(ekman))
    set_tag("prandtl", str(prandtl))
    set_tag("rayleigh", f"{rayleigh:.4f}")

    tree.write(cfg_path, encoding="utf-8", xml_declaration=False)



def sbatch(args: list[str], cwd: Path) -> str:
    out = subprocess.check_output(["sbatch"] + args, cwd=cwd, text=True).strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {out}")
    return m.group(1)

def ensure_ra_folder(ek: float, pr: float, ra: float) -> Path:
    ek_dir = BASE / ek_to_folder(ek)
    pr_dir = ek_dir / pr_to_folder(pr)
    ra_dir = pr_dir / ra_to_folder(ra)

    (ra_dir / "data").mkdir(parents=True, exist_ok=True)
    (ra_dir / "movie").mkdir(parents=True, exist_ok=True)


    def cp(src: Path, dst: Path):
        shutil.copy2(src, dst)

    cp(TEMPLATES / "sim.slurm", ra_dir / "sim.slurm")
    cp(TEMPLATES / "post.slurm", ra_dir / "post.slurm")
    cp(TEMPLATES / "VisuLumi_array.slurm", ra_dir / "VisuLumi_array.slurm")
    cp(TEMPLATES / "analyze.slurm", ra_dir / "analyze.slurm")

    cp(TEMPLATES / "parameters.cfg", ra_dir / "parameters.cfg")
    cp(TEMPLATES / "parameters.cfg", ra_dir / "movie" / "parameters.cfg")

    dst_state = ra_dir / "state_initial.hdf5"
    if not dst_state.exists():
        cp(TEMPLATES / "state_initial.hdf5", dst_state)
    cp(TEMPLATES / "state_initial.hdf5", ra_dir / "movie" / "state_initial.hdf5")

    cp(TEMPLATES / "Extract_Video_Data_auto.py", ra_dir / "movie" / "Extract_Video_Data_auto.py")
    cp(TEMPLATES / "drift_from_movie_test.py", ra_dir / "movie" / "drift_from_movie_test.py")
    cp(TEMPLATES / "clean_movies.py", ra_dir / "movie" / "clean_movies.py")

    exe_link = ra_dir / "BoussinesqSphereRTCExplicitModel"
    if not exe_link.exists():
        exe_link.symlink_to(EXEC_MODEL)

    return ra_dir



def write_cfgs(ra_dir: Path, ek: float, pr: float, ra: float) -> None:
    set_physical_params(ra_dir / "parameters.cfg", ekman=ek, prandtl=pr, rayleigh=ra)
    set_physical_params(ra_dir / "movie" / "parameters.cfg", ekman=ek, prandtl=pr, rayleigh=ra)



def drift_path(ra_dir: Path) -> Path:
    return ra_dir / "movie" / "movie_test" / "drift.json"



def submit_full_pipeline(
    ra_dir: Path,
    jobname: str,
    visu_start: int = DEFAULT_VISU_START,
    visu_end: int = DEFAULT_VISU_END,
    visu_step: int = DEFAULT_VISU_STEP,
) -> dict:

    sim_id = sbatch(["-J", f"SIM_{jobname}", "sim.slurm"], cwd=ra_dir)


    array_spec = f"--array={visu_start}-{visu_end}:{visu_step}"
    vis_id = sbatch(
        [f"--dependency=afterok:{sim_id}", array_spec, "-J", f"VIS_{jobname}", "VisuLumi_array.slurm"],
        cwd=ra_dir,
    )

    an_id = sbatch(
        [f"--dependency=afterok:{vis_id}", "-J", f"AN_{jobname}", "analyze.slurm"],
        cwd=ra_dir,
    )

    return {"sim": sim_id, "visu": vis_id, "analyze": an_id}




def wait_for_drift(ra_dir: Path) -> dict:

    p = drift_path(ra_dir)
    while not p.exists():
        time.sleep(POLL_SECONDS)
    with open(p, "r") as f:
        return json.load(f)


def run_one_case(ek: float, pr: float, ra: float,
                 visu_start: int = DEFAULT_VISU_START,
                 visu_end: int = DEFAULT_VISU_END,
                 visu_step: int = DEFAULT_VISU_STEP) -> dict:
    ra = round(ra, 4)
    ra_dir = ensure_ra_folder(ek, pr, ra)
    write_cfgs(ra_dir, ek, pr, ra)
    jobname = f"{ek_to_folder(ek)}_{pr_to_folder(pr)}_{ra_to_folder(ra)}"
    ids = submit_full_pipeline(ra_dir, jobname, visu_start, visu_end, visu_step)
    print(f"Submitted {jobname}  sim={ids['sim']}  visu={ids['visu']}  analyze={ids['analyze']}")
    drift = wait_for_drift(ra_dir)
    print(f"Done {jobname}: fd={drift.get('fd')}  m*={drift.get('m_star')}")
    return drift

def submit_one_case(
    ek: float, pr: float, ra: float,
    visu_start: int = DEFAULT_VISU_START,
    visu_end: int = DEFAULT_VISU_END,
    visu_step: int = DEFAULT_VISU_STEP,
) -> Path:

    ra = round(ra, 4)
    ra_dir = ensure_ra_folder(ek, pr, ra)
    write_cfgs(ra_dir, ek, pr, ra)


    jobname = f"{ek_to_folder(ek)}_{pr_to_folder(pr)}_{ra_to_folder(ra)}"
    ids = submit_full_pipeline(ra_dir, jobname, visu_start, visu_end, visu_step)
    print(f"Submitted {jobname}  sim={ids['sim']}  visu={ids['visu']}  analyze={ids['analyze']}")
    return ra_dir

def wait_many(ra_dirs: list[Path]) -> list[dict]:

    results = []
    for ra_dir in ra_dirs:
        d = wait_for_drift(ra_dir)
        results.append(d)
        print(f"Completed {ra_dir.name}: fd={d.get('fd')}  m*={d.get('m_star')}")

    return results




if __name__ == "__main__":

    # Example single run:
    #ek = 1.7e-3
    #pr = 1.0
    #ra = 93.2540
    #run_one_case(ek, pr, ra, visu_start=100, visu_end=250, visu_step=5)

    ek = 1.7e-3
    pr = 1.0
    ras = [93.10, 93.25, 93.40]

    # Submit all 3 
    ra_dirs = []
    for ra in ras:
        ra_dirs.append(
            submit_one_case(
                ek, pr, ra,
                visu_start=100, visu_end=250, visu_step=5
            )
        )

    # Now wait for all results (polling drift.json)

    results = wait_many(ra_dirs)
    print("\nSummary:")

    # Sort by Ra just in case
    results = sorted(results, key=lambda d: float(d["Ra"]))
    for d in results:
        print(f"Ra={float(d['Ra']):.4f}  fd={float(d['fd']):+ .6e}  m*={int(d['m_star'])}  nt_used={int(d['nt_used'])}")

