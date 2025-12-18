#!/usr/bin/env python3

"""

find_omega_zero_crossings.py



Reads:

    Data/Plots/all_Pr_m_critical.csv



This file contains, for each Pr and each m:

    Ra_c(m), omega_at_Ra_c(m)



For each Pr:

  • Sort by m

  • Detect sign changes in omega_at_Ra_c

  • Linearly interpolate between adjacent m values

      to estimate Ra where omega = 0

  • Record possibly 0, 1, or 2 zero-crossings



Outputs:

    Data/Plots/omega_zero_crossings.csv

    Data/Plots/Pr_vs_Ra_omega_zero.png

"""



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path



# ------------------------

# Load data

# ------------------------

PLOTS = Path("Data/Plots")

infile = PLOTS / "all_Pr_m_critical.csv"



if not infile.exists():

    raise FileNotFoundError("Missing all_Pr_m_critical.csv – run the previous script first.")



df = pd.read_csv(infile)



# Ensure correct types

df["Pr"] = df["Pr"].astype(float)

df["m"] = df["m"].astype(float)

df["Ra_c"] = df["Ra_c"].astype(float)

df["omega_at_Ra_c"] = df["omega_at_Ra_c"].astype(float)



# Container for results

rows = []



# ------------------------

# Utility for linear interpolation

# ------------------------

def interpolate_zero(x1, x2, y1, y2):

    """

    Find x where y=0 between (x1, y1) and (x2, y2).

    Linear interpolation.

    """

    if y2 == y1:

        return 0.5*(x1 + x2)

    t = -y1 / (y2 - y1)

    return x1 + t*(x2 - x1)



# ------------------------

# Detect zero crossings per Pr

# ------------------------

for Pr, g in df.groupby("Pr"):

    g = g.sort_values("m")



    m_vals = g["m"].values

    Ra_vals = g["Ra_c"].values

    w_vals = g["omega_at_Ra_c"].values



    # Loop over adjacent pairs in m

    for i in range(len(g) - 1):

        w1, w2 = w_vals[i], w_vals[i+1]



        # Detect sign change

        if w1 == 0:

            # Exact zero at m[i]

            rows.append({"Pr": Pr, "Ra_omega0": Ra_vals[i]})

        elif w1 * w2 < 0:  # sign change

            # Interpolate in m space

            m_zero = interpolate_zero(m_vals[i], m_vals[i+1], w1, w2)

            Ra_zero = np.interp(m_zero, m_vals, Ra_vals)



            rows.append({"Pr": Pr, "Ra_omega0": Ra_zero})



# ------------------------

# Convert to DataFrame and save

# ------------------------

df_out = pd.DataFrame(rows).sort_values("Pr")

outfile = PLOTS / "omega_zero_crossings.csv"

df_out.to_csv(outfile, index=False)

print("Saved:", outfile)



# ------------------------

# Plot Pr vs Ra

# ------------------------

plt.figure(figsize=(7,5))



plt.scatter(df_out["Pr"], df_out["Ra_omega0"], c='firebrick', s=60, label="ω = 0")



# Connect points if 2 per Pr (forming a loop)

for Pr, g in df_out.groupby("Pr"):

    if len(g) == 2:

        plt.plot([Pr, Pr], g["Ra_omega0"], color='gray', lw=1)



plt.xlabel("Prandtl number Pr")

plt.ylabel("Rayleigh number where ω = 0")

plt.title("Stationary–Drifting Transition Curve (ω = 0)")

plt.grid(True, ls="--", alpha=0.6)

plt.tight_layout()



outfig = PLOTS / "Pr_vs_Ra_omega_zero.png"

plt.savefig(outfig, dpi=200)

plt.close()



print("Saved:", outfig)

print("🎉 Finished computing omega zero crossings.")

