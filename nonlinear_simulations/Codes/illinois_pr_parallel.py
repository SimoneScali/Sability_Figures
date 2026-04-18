import csv
import json
import time
import os
import subprocess
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from run_stationary_pipeline import (
    ensure_ra_folder, write_cfgs, sbatch,
    ek_to_folder, pr_to_folder, ra_to_folder, drift_path,
    DEFAULT_VISU_START, DEFAULT_VISU_END, DEFAULT_VISU_STEP,
)

POLL = 60
FD_TOL = 1e-6    #5e-5   #1e-5    #drift frequency tolerance for declaring "stationary"
RA_TOL = 5e-4     #4 decimal points


def _job_exists(name: str) -> bool:
    out = subprocess.check_output(["squeue", "-u", "scalisim", "-h", "-o", "%j"], text=True)
    names = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return any(name == n for n in names)


def _states_exist(ra_dir: Path) -> bool:
    movie = ra_dir / "movie"
    return any(movie.glob("state????.hdf5"))


def _latest_sim_jobid_for_name(sim_name: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["sacct", "-u", "scalisim", "-n", "-P", "-o", "JobID,JobName,State,End", "--name", sim_name],
            text=True,
        )
    except Exception:
        return None

    rows = [r for r in out.splitlines() if r.strip()]
    if not rows:
        return None

    for r in rows:
        jobid = r.split("|")[0]
        if "." not in jobid:
            return jobid
    return None


def wait_for_file(p: Path):
    while not p.exists():
        time.sleep(POLL)


# Load Ra_c from CSV
def load_rac_from_csv(pr: float, ek: float | None = None) -> float:
    base = Path(__file__).resolve().parent

    csv_path = None

    if ek is not None:
        lin_dir = base / "linear_simulations"

        ek_dirs = [d for d in lin_dir.glob("Ek*") if d.is_dir()]

        for d in ek_dirs:
            try:
                ek_val = float(d.name.replace("Ek", ""))
            except Exception:
                continue

            if abs(ek_val - float(ek)) < 1e-12:
                ek_csv = d / f"{d.name}_all_Pr_m_critical.csv"
                if ek_csv.exists():
                    csv_path = ek_csv
                    break

    if csv_path is None:
        templates_csv = base / "templates_rtc" / "all_Pr_m_critical.csv"
        if templates_csv.exists():
            csv_path = templates_csv
        else:
            raise FileNotFoundError(
                "Could not find either:\n"
                f"  {base/'linear_simulations'/'Ek*/Ek*_all_Pr_m_critical.csv'} (for provided ek)\n"
                f"  {templates_csv} (fallback)\n"
            )

    rac_vals = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_pr = row.get("Pr", row.get("pr", row.get("PR", "")))
            try:
                row_pr_f = float(row_pr)
            except Exception:
                continue
            if abs(row_pr_f - float(pr)) > 1e-9:
                continue

            ra_str = row.get("Ra_c", row.get("Rac", row.get("RaCrit", row.get("Ra", ""))))
            if not ra_str:
                continue
            try:
                rac_vals.append(float(ra_str))
            except Exception:
                continue

    if not rac_vals:
        raise ValueError(f"No rows found in {csv_path} for Pr={pr}. Check CSV headers/content.")

    return min(rac_vals)

def submit_eval(ek, pr, ra, visu_start=100, visu_end=9999, visu_step=5):
    ra = round(float(ra), 4)
    ra_dir = ensure_ra_folder(float(ek), float(pr), ra)
    write_cfgs(ra_dir, float(ek), float(pr), ra)

    p = drift_path(ra_dir)
    if p.exists():
        d = json.loads(p.read_text())
        fd = float(d["fd"])
        mstar = int(d["m_star"])
        print(f"Reusing existing result Ra={ra:.4f}: fd={fd:+.6e}, m*={mstar}")
        return ra, fd, mstar, ra_dir

    jobname = f"{ek_to_folder(float(ek))}_{pr_to_folder(float(pr))}_{ra_to_folder(ra)}"
    sim_name = f"SIM_{jobname}"
    post_name = f"POST_{jobname}"

    if _job_exists(post_name):
        print(f"POST already queued/running for Ra={ra:.4f}. Waiting for drift...")
        wait_for_file(p)
        d = json.loads(p.read_text())
        return ra, float(d["fd"]), int(d["m_star"]), ra_dir

    if _states_exist(ra_dir):
        simid = _latest_sim_jobid_for_name(sim_name)
        dep = f"--dependency=afterany:{simid}" if simid else None

        args = []
        if dep:
            args.append(dep)
        args += [
            f"--export=ALL,START={visu_start},END={visu_end},STEP={visu_step}",
            "-J", post_name,
            "post.slurm",
        ]
        post_id = sbatch(args, cwd=ra_dir)
        print(f"Submitted POST only for Ra={ra:.4f}: post={post_id} (states already exist)")
        wait_for_file(p)
        d = json.loads(p.read_text())
        return ra, float(d["fd"]), int(d["m_star"]), ra_dir

    if _job_exists(sim_name):
        print(f"SIM already queued/running for Ra={ra:.4f}. Waiting for states then drift...")
        while not _states_exist(ra_dir):
            time.sleep(POLL)
        return submit_eval(ek, pr, ra, visu_start, visu_end, visu_step)

    sim_id = sbatch(["-J", sim_name, "sim.slurm"], cwd=ra_dir)
    post_id = sbatch(
        [
            f"--dependency=afterany:{sim_id}",
            f"--export=ALL,START={visu_start},END={visu_end},STEP={visu_step}",
            "-J", post_name,
            "post.slurm",
        ],
        cwd=ra_dir,
    )
    print(f"Submitted Ra={ra:.4f}: sim={sim_id} post={post_id}")
    wait_for_file(p)
    d = json.loads(p.read_text())
    return ra, float(d["fd"]), int(d["m_star"]), ra_dir


def coarse_scan_parallel(ek, pr, ra_min, ra_max, n=4, **kwargs):
    if n < 2:
        raise ValueError("n must be >= 2")
    ra_min = float(ra_min)
    ra_max = float(ra_max)
    ras = [round(ra_min + i * (ra_max - ra_min) / (n - 1), 4) for i in range(n)]

    results = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        fut_map = {ex.submit(submit_eval, ek, pr, ra, **kwargs): ra for ra in ras}
        for fut in as_completed(fut_map):
            results.append(fut.result())

    results_sorted = sorted(results, key=lambda t: t[0])
    return results_sorted


def find_sign_change_brackets(results_sorted, tol=FD_TOL):

    import math
    brackets = []
    near_roots = []

    def sgn(x):
        if math.isnan(x):
            return None
        if abs(x) <= tol:
            return 0
        return 1 if x > 0 else -1

    for (ra1, fd1, m1, _), (ra2, fd2, m2, _) in zip(results_sorted, results_sorted[1:]):
        s1, s2 = sgn(fd1), sgn(fd2)

        if s1 is None or s2 is None:
            continue

        if s1 == 0:
            near_roots.append((ra1, fd1, m1))
            continue
        if s2 == 0:
            near_roots.append((ra2, fd2, m2))
            continue

        if s1 != s2:
            brackets.append((ra1, ra2))

    return brackets, near_roots


def bisection_stationary(ek, pr, ra_lo, ra_hi, **kwargs):
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_lo = ex.submit(submit_eval, ek, pr, ra_lo, **kwargs)
        fut_hi = ex.submit(submit_eval, ek, pr, ra_hi, **kwargs)
        ra_lo, fd_lo, *_ = fut_lo.result()
        ra_hi, fd_hi, *_ = fut_hi.result()

    if fd_lo * fd_hi > 0:
        raise RuntimeError(f"No sign change: fd({ra_lo})={fd_lo}, fd({ra_hi})={fd_hi}")

    while True:
        ra_mid = round(0.5 * (ra_lo + ra_hi), 4)
        ra_mid, fd_mid, m_mid, _ = submit_eval(ek, pr, ra_mid, **kwargs)

        if abs(fd_mid) < FD_TOL or abs(ra_hi - ra_lo) < RA_TOL:
            print(f"\nStationary estimate: Ra0={ra_mid:.4f} (fd={fd_mid:+.3e}, m*={m_mid})")
            return ra_mid

        if fd_lo * fd_mid <= 0:
            ra_hi, fd_hi = ra_mid, fd_mid
        else:
            ra_lo, fd_lo = ra_mid, fd_mid


def illinois_stationary(
    ek, pr,
    ra_lo, ra_hi,
    **kwargs
):

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_lo = ex.submit(submit_eval, ek, pr, ra_lo, **kwargs)
        fut_hi = ex.submit(submit_eval, ek, pr, ra_hi, **kwargs)
        ra_lo, fd_lo, m_lo, _ = fut_lo.result()
        ra_hi, fd_hi, m_hi, _ = fut_hi.result()

    if fd_lo * fd_hi > 0:
        raise RuntimeError(
            f"No sign change: fd({ra_lo})={fd_lo}, fd({ra_hi})={fd_hi}"
        )
    print(
        f"[Illinois start] "
        f"Ra_lo={ra_lo:.4f} fd_lo={fd_lo:+.3e}, "
        f"Ra_hi={ra_hi:.4f} fd_hi={fd_hi:+.3e}",
        flush=True,
    )

    for it in range(10):  
        ra_new = ra_hi - fd_hi * (ra_hi - ra_lo) / (fd_hi - fd_lo)
        ra_new = round(float(ra_new), 4)
        ra_new, fd_new, m_new, _ = submit_eval(
            ek, pr, ra_new, **kwargs
        )

        print(
            f"[Illinois {it}] "
            f"Ra={ra_new:.4f} fd={fd_new:+.3e} m*={m_new}",
            flush=True,
        )

        if abs(fd_new) < FD_TOL or abs(ra_hi - ra_lo) < RA_TOL:
            print(
                f"\nStationary estimate: "
                f"Ra0={ra_new:.4f} (fd={fd_new:+.3e}, m*={m_new})"
            )
            return ra_new

        if fd_new * fd_lo < 0:
            ra_hi, fd_hi = ra_new, fd_new
            fd_lo *= 0.5   
        else:
            ra_lo, fd_lo = ra_new, fd_new
            fd_hi *= 0.5

    ra_mid = round(0.5 * (ra_lo + ra_hi), 4)
    print(f"[Illinois fallback] returning Ra={ra_mid:.4f}")
    return ra_mid



def solve_all_roots_in_range(ek, pr, ra_min, ra_max, n=4, **kwargs):
    scan = coarse_scan_parallel(ek, pr, ra_min, ra_max, n=n, **kwargs)

    print("\nCoarse scan results (sorted):")
    for ra, fd, mstar, _ in scan:
        print(f"  Ra={ra:.4f}  fd={fd:+.6e}  m*={mstar}")

    brackets, near_roots = find_sign_change_brackets(scan)

    if near_roots:
        print("\nCoarse scan hit (near) stationary points:")
        for ra, fd, m in near_roots:
            print(f"  Ra={ra:.4f}  fd={fd:+.3e}  m*={m}")

    if not brackets:
        if near_roots:
            return [ra for ra, _, _ in near_roots]
        print("\nNo fd sign changes detected in coarse scan range.")
        return []

    print("\nDetected sign-change brackets:")
    for lo, hi in brackets:
        print(f"  [{lo:.4f}, {hi:.4f}]")

    roots = []

    max_workers = min(len(brackets), 2) if len(brackets) > 0 else 1
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(illinois_stationary, ek, pr, lo, hi, **kwargs): (lo, hi) for lo, hi in brackets}
        for fut in as_completed(futs):
            lo, hi = futs[fut]
            ra0 = fut.result()
            roots.append(ra0)
            print(f"[root done] bracket [{lo:.4f}, {hi:.4f}] -> Ra0={ra0:.4f}", flush=True)

    roots.sort()
    return roots

if __name__ == "__main__":
    ek = 1.7e-4
    
    PR_LIST = [1.0, 2.0, 0.5, 3.0, 0.7, 10.0, 20.0]
    #PR_LIST = [4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 40.0]
    #PR_LIST = [0.1, 0.3, 0.7, 1.3, 1.7, 2.0, 2.3, 2.5, 2.7, 3.0, 3.3, 3.5, 3.7, 4.0, 4.3, 4.5, 4.7, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    #PR_LIST = [0.1, 0.2, 0.4, 0.6, 0.8]


    def run_one_pr(pr: float):
        try:
            # Read Ra_c(Pr) from templates_rtc/all_Pr_m_critical.csv (min over m)
            Ra_c = load_rac_from_csv(pr, ek)
            print(f"[Pr={pr}] Loaded Ra_c = {Ra_c:.4f}", flush=True)

            # Define thesis scope as moderately supercritical range.
            ra_min = round(Ra_c * 2.25, 4)  # 1% above onset
            ra_max = round(Ra_c * 4.00, 4)  # 45% above onset
            print(f"[Pr={pr}] Searching in Ra range [{ra_min:.4f}, {ra_max:.4f}] with coarse scan n=8 (parallel)", flush=True)

            roots = solve_all_roots_in_range(
                ek, pr,
                ra_min=ra_min,
                ra_max=ra_max,
                n=5,
                visu_start=100, visu_end=9999, visu_step=2,
            )

            return pr, Ra_c, roots
        except Exception as e:
            return pr, None, e

    # Run multiple Pr in parallel (each Pr internally parallelizes its coarse scan).
    # Keep this small to avoid flooding SLURM with too many simultaneous jobs.
    MAX_PR_WORKERS = int(os.environ.get("MAX_PR_WORKERS", "2"))
    print(f"Launching Pr-level concurrency with MAX_PR_WORKERS={MAX_PR_WORKERS}", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_PR_WORKERS) as ex:
        futs = {ex.submit(run_one_pr, pr): pr for pr in PR_LIST}
        for fut in as_completed(futs):
            pr = futs[fut]
            pr, rac, out = fut.result()
            if isinstance(out, Exception):
                print(f"[Pr={pr}] FAILED: {out}", flush=True)
                continue
            roots = out
            print(f"\n[Pr={pr}] DONE  Ra_c={rac:.4f}  Roots found: {len(roots)}", flush=True)
            for r in roots:
                print(f"[Pr={pr}]   Ra0 = {r:.4f}", flush=True)
