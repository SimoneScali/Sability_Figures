#!/usr/bin/env python3

"""

analyze_Pr_m_critical.py



Reads all marginal.log files inside:

    Data/Pr*/m*/marginal.log



For each Pr and each m, it:

  - parses all Rayleigh blocks inside marginal.log

  - extracts the leading eigenvalue (σ, ω)

  - computes the critical Rayleigh number Ra_c(m)

      using zero-crossing of σ(Ra)

  - interpolates ω at Ra_c

  - saves all results

  - generates plots:

      • Ra_c vs m   (per Pr)

      • omega(Ra_c) vs m   (per Pr)



All outputs placed in: Data/Plots/


"""



import os

from pathlib import Path

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# -------------------------

# Directory setup

# -------------------------

BASE = Path("Data")

PLOTS = BASE / "Plots"

PLOTS.mkdir(exist_ok=True)



# -------------------------

# Parse marginal.log

# -------------------------

def parse_marginal_text(fn):

    """

    Parse a marginal.log containing blocks:



        rayleigh = 100

            growth: (sigma, omega)

            growth: ...



    Returns arrays:

        R (Rayleigh values)

        sigma_lead (leading real part)

        omega_lead (imag part of that eigenvalue)

    """

    R = []

    max_reals = []

    max_imags = []



    with open(fn, "r") as f:

        lines = f.readlines()



    ra = None

    growths = []



    for L in lines:

        Ls = L.strip()



        # Detect new "rayleigh = XYZ"

        m = re.match(r"rayleigh\s*=\s*([0-9Ee+\-\.]+)", Ls)

        if m:

            # save previous block

            if ra is not None and growths:

                reals = [g[0] for g in growths]

                imags = [g[1] for g in growths]

                idx = int(np.argmax(reals))

                max_reals.append(reals[idx])

                max_imags.append(imags[idx])

                R.append(ra)

            growths = []

            ra = float(m.group(1))

            continue



        # Detect growth line

        mg = re.search(r"growth:\s*\(\s*([\-0-9Ee.+]+)\s*,\s*([\-0-9Ee.+]+)\s*\)", Ls)

        if mg:

            real = float(mg.group(1))

            imag = float(mg.group(2))

            growths.append((real, imag))



    # Store last block

    if ra is not None and growths:

        reals = [g[0] for g in growths]

        imags = [g[1] for g in growths]

        idx = int(np.argmax(reals))

        max_reals.append(reals[idx])

        max_imags.append(imags[idx])

        R.append(ra)



    if len(R) == 0:

        return None, None, None



    R = np.array(R)

    max_reals = np.array(max_reals)

    max_imags = np.array(max_imags)

    order = np.argsort(R)



    return R[order], max_reals[order], max_imags[order]





# -------------------------

# Critical Ra finder

# -------------------------

def find_Ra_critical(R, sigma, omega):

    """

    Finds Ra_c where sigma crosses zero.

    - Linear interpolation is used.

    - Also returns omega interpolated at Ra_c.



    If no crossing: return Ra where sigma is closest to zero.

    """

    # Check exact zeros

    zero_idx = np.where(np.isclose(sigma, 0.0, atol=1e-12))[0]

    if zero_idx.size > 0:

        R0 = R[zero_idx[0]]

        w0 = omega[zero_idx[0]]

        return float(R0), float(w0)



    # Look for sign change

    for i in range(len(R) - 1):

        if sigma[i] * sigma[i + 1] < 0:  # crossing

            R1, R2 = R[i], R[i+1]

            s1, s2 = sigma[i], sigma[i+1]

            w1, w2 = omega[i], omega[i+1]



            # linear interpolation for Ra_c

            t = -s1 / (s2 - s1)

            R0 = R1 + t * (R2 - R1)

            w0 = w1 + t * (w2 - w1)



            return float(R0), float(w0)



    # No crossing → use point where |sigma| is minimum

    idx = int(np.argmin(np.abs(sigma)))

    return float(R[idx]), float(omega[idx])





# -------------------------

# MAIN LOOP: collect all data

# -------------------------

records = []



for Pr_dir in sorted(BASE.glob("Pr*")):

    mPr = re.search(r"Pr([0-9.]+)", Pr_dir.name)

    if not mPr:

        continue

    Pr = float(mPr.group(1))



    for m_dir in sorted(Pr_dir.glob("m*")):

        mm = re.search(r"m([0-9]+)", m_dir.name)

        if not mm:

            continue

        m = int(mm.group(1))



        log_file = m_dir / "marginal.log"

        if not log_file.exists():

            print(f"⚠️ Missing {log_file}")

            continue



        R, sigma, omega = parse_marginal_text(log_file)

        if R is None:

            print(f"⚠️ Could not parse {log_file}")

            continue



        # Find the critical Ra and ω

        Ra_c, omega_c = find_Ra_critical(R, sigma, omega)



        records.append({

            "Pr": Pr,

            "m": m,

            "Ra_c": Ra_c,

            "omega_at_Ra_c": omega_c

        })



# Convert to DataFrame

df = pd.DataFrame(records)

df.sort_values(["Pr", "m"], inplace=True)



# Save

df.to_csv(PLOTS / "all_Pr_m_critical.csv", index=False)

print("Saved:", PLOTS / "all_Pr_m_critical.csv")



# -------------------------

# PLOT for each Pr

# -------------------------



# ---------------------

# Produce SIDE-BY-SIDE plots for each Pr

# ---------------------

for Pr, g in df.groupby("Pr"):

    g = g.sort_values("m")



    fig, axs = plt.subplots(1, 2, figsize=(12,5))



    # Left: Ra_c vs m

    axs[0].plot(g["m"], g["Ra_c"], "o-", lw=2)

    axs[0].set_xlabel("m")

    axs[0].set_ylabel("Ra_c")

    axs[0].set_title(f"Critical Rayleigh for Pr={Pr}")

    axs[0].grid(True, ls="--", alpha=0.6)



    # Right: omega(Ra_c) vs m

    axs[1].plot(g["m"], g["omega_at_Ra_c"], "o-", lw=2)

    axs[1].set_xlabel("m")

    axs[1].set_ylabel("omega(Ra_c)")

    axs[1].set_title(f"Drift frequency at Ra_c for Pr={Pr}")

    axs[1].grid(True, ls="--", alpha=0.6)



    plt.tight_layout()

    outfile = PLOTS / f"Pr{Pr}_side_by_side.png"

    plt.savefig(outfile, dpi=200)

    plt.close()



    print("Saved", outfile)


print("\n🎉 Done! All plots saved in:", PLOTS)

