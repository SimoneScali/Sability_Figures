#!/usr/bin/env python3
import sys
import subprocess
import re
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional, List


def run(cmd, cwd=None, check: bool = True):
    subprocess.run(cmd, cwd=cwd, check=check)


def replace_xml_tag(text: str, tag: str, value: str) -> str:
    # Replace <tag>...</tag> with <tag>value</tag>
    pat = rf"<{tag}>\s*.*?\s*</{tag}>"
    repl = f"<{tag}>{value}</{tag}>"
    if re.search(pat, text) is None:
        raise RuntimeError(f"Tag <{tag}> not found in parameters.cfg")
    return re.sub(pat, repl, text, count=1)


def replace_dim3d(text: str, m: int) -> str:
    # In QuICC stability solver, dim3D is the harmonic order (m) to solve.
    pat = r"<dim3D>\s*\d+\s*</dim3D>"
    repl = f"<dim3D>{m}</dim3D>"
    if re.search(pat, text) is None:
        raise RuntimeError("Tag <dim3D> not found in parameters.cfg")
    return re.sub(pat, repl, text, count=1)


def parse_critical_rayleigh(marginal_text: str) -> Optional[float]:
    m = re.search(r"Critical rayleigh number converged to the bracket:\s*([\d.eE+-]+)\s*<\s*rayleigh\s*<\s*([\d.eE+-]+)", marginal_text, re.MULTILINE)
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2))
    return 0.5 * (a + b)


def parse_omega_at_last_rayleigh_block(marginal_text: str) -> Optional[float]:
    # Split into blocks starting with "rayleigh ="
    blocks = re.split(r"\brayleigh\s*=\s*", marginal_text)
    if len(blocks) <= 1:
        return None
    last = blocks[-1]
    # last starts with the number then newline...
    growth = re.findall(r"growth:\s*\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)", last)
    if not growth:
        return None
    eigs = [(float(re_), float(im_)) for re_, im_ in growth]
    re_max, im_at = max(eigs, key=lambda t: t[0])
    return im_at


def pr_key(pr: float) -> str:
    # Stable string key for dict/csv; matches folder naming like Pr{Pr:g}
    return f"{pr:g}"


def read_cache(path: Path) -> Dict[Tuple[str, int], Tuple[float, Optional[float]]]:
    """
    Returns dict mapping (Pr_str, m) -> (Rac, omega)
    """
    cache: Dict[Tuple[str, int], Tuple[float, Optional[float]]] = {}
    if not path.exists():
        return cache
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                pr = row["Pr"]
                m = int(row["m"])
                rac = float(row["Rac"])
                omega = float(row["omega"]) if row.get("omega") not in (None, "", "nan") else None
                cache[(pr, m)] = (rac, omega)
            except Exception:
                # ignore malformed lines
                continue
    return cache


def write_cache(path: Path, cache: Dict[Tuple[str, int], Tuple[float, Optional[float]]], ek: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (pr, m), (rac, omega) in sorted(cache.items(), key=lambda x: (float(x[0][0]), x[0][1])):
        rows.append({"Ek": ek, "Pr": pr, "m": m, "Rac": rac, "omega": "" if omega is None else omega})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Ek", "Pr", "m", "Rac", "omega"])
        w.writeheader()
        w.writerows(rows)


def guess_ra_init(
    pr: float,
    m: int,
    pr_list: List[float],
    cache: Dict[Tuple[str, int], Tuple[float, Optional[float]]],
    default: float,
) -> float:
    """
    Continuation guess:
      1) same Pr, previous m
      2) previous Pr, same m
      3) default
    """
    pk = pr_key(pr)

    # 1) same Pr, previous m
    if m > 1 and (pk, m - 1) in cache:
        return cache[(pk, m - 1)][0]

    # 2) previous Pr in list, same m
    # find immediate previous Pr in sorted list
    idx = pr_list.index(pr)
    if idx > 0:
        prev_pr = pr_list[idx - 1]
        prev_pk = pr_key(prev_pr)
        if (prev_pk, m) in cache:
            return cache[(prev_pk, m)][0]

    # 3) fallback
    return default


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 linear_controller.py <Ek>")
        sys.exit(2)

    EK = sys.argv[1]

    THE_DIR = Path("/scratch/project_465001528/scalisim/THE")
    TEMPL = THE_DIR / "linear_files"
    OUTROOT = THE_DIR / "linear_simulations"
    EKDIR = OUTROOT / f"Ek{EK}"

    EXEC = TEMPL / "BoussinesqSphereRTCImplicitStability"
    STATE = TEMPL / "state_initial.hdf5"
    CFG_TEMPLATE = TEMPL / "parameters.cfg"

    EKDIR.mkdir(parents=True, exist_ok=True)
    (EKDIR / "Data").mkdir(exist_ok=True)

    # ---- Pr list ----
    PR_LOW  = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    PR_MID  = [round(x, 10) for x in frange(1.0, 4.0, 1.0)]
    PR_HIGH = [5.0, 6.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
    PR_LIST = sorted(set(PR_LOW + PR_MID + PR_HIGH))

    M_START, M_STOP = 1, 150

    DEFAULT_RA_INIT = 400  # only used when cache has no nearby information
    RA_TRY_FACTORS = [1.0, 0.8, 1.25, 0.5, 2.0, 0.2, 3.0, 0.1, 5.0]  # retry list if solver doesn't converge

    # Basic existence checks
    for p in [EXEC, STATE, CFG_TEMPLATE]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    cfg_template_txt = CFG_TEMPLATE.read_text()

    cache_path = EKDIR / "Rac_cache.csv"
    cache = read_cache(cache_path)

    for Pr in PR_LIST:
        pk = pr_key(Pr)
        prdir = EKDIR / "Data" / f"Pr{pk}"
        prdir.mkdir(parents=True, exist_ok=True)

        for m in range(M_START, M_STOP + 1):
            mdir = prdir / f"m{m}"
            mdir.mkdir(parents=True, exist_ok=True)

            # Link exec + state into run dir
            run(["ln", "-sf", str(EXEC), str(mdir / EXEC.name)])
            run(["ln", "-sf", str(STATE), str(mdir / STATE.name)])

            # If we already have this point in cache, you can skip to save time.
            # Comment this block out if you want to re-run everything.
            if (pk, m) in cache:
                continue

            ra_guess = guess_ra_init(Pr, m, PR_LIST, cache, DEFAULT_RA_INIT)

            success = False
            last_log_path = mdir / f"run_m{m}.log"

            for fac in RA_TRY_FACTORS:
                ra_init = ra_guess * fac

                # Write parameters.cfg customized
                txt = cfg_template_txt
                txt = replace_xml_tag(txt, "ekman", EK)
                txt = replace_xml_tag(txt, "prandtl", pk)
                txt = replace_xml_tag(txt, "rayleigh", f"{ra_init:.12g}")
                txt = replace_dim3d(txt, m)

                # Sanity check dim3D == m
                check = re.search(r"<dim3D>\s*(\d+)\s*</dim3D>", txt)
                if not check or int(check.group(1)) != m:
                    raise RuntimeError(f"dim3D sanity check failed for Pr={pk}, m={m}")

                cfg_out = mdir / "parameters.cfg"
                cfg_out.write_text(txt)

                # Run solver (do NOT pass -m; dim3D defines the harmonic order)
                cmd = f"srun --mpi=cray_shasta ./{EXEC.name}"
                subprocess.run(
                    cmd,
                    cwd=mdir,
                    shell=True,
                    stdout=last_log_path.open("w"),
                    stderr=subprocess.STDOUT,
                    check=False,
                )

                marg = mdir / "marginal.log"
                if marg.exists():
                    mt = marg.read_text()
                else:
                    # best-effort extraction from run log
                    mt = ""
                    if last_log_path.exists():
                        mt = last_log_path.read_text()
                        (mdir / "marginal.log").write_text(mt)

                rac = parse_critical_rayleigh(mt)
                if rac is not None and rac > 0.0 and rac < 1e12:
                    omega = parse_omega_at_last_rayleigh_block(mt)
                    cache[(pk, m)] = (rac, omega)
                    write_cache(cache_path, cache, EK)
                    success = True
                    break

            if not success:
                # Keep a marker file to make failures visible without digging through logs
                (mdir / "FAILED").write_text(
                    f"Failed to converge critical Ra for Ek={EK}, Pr={pk}, m={m}. "
                    f"Tried RA_INIT guesses around {ra_guess} with factors {RA_TRY_FACTORS}. "
                    f"See {last_log_path.name}.\n"
                )

    # Final write
    write_cache(cache_path, cache, EK)
    print(f"Done Ek={EK}. Results under {EKDIR} and cache {cache_path}")


def frange(a, b, step):
    # inclusive end
    nmax = int(round((b - a) / step)) + 1
    for k in range(nmax):
        yield a + k * step


if __name__ == "__main__":
    main()
