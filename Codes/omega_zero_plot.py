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

  • Estimate Ra where omega = 0

  • Record 0, 1, or 2 zero-crossings



Outputs:

    Data/Plots/omega_zero_crossings.csv

    Data/Plots/Pr_vs_Ra_omega_zero.png

    Data/Plots/Pr_vs_Ra_omega_zero_logPr.png

"""



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path



# ------------------------

# Load data

# ------------------------



PLOTS = Path("2nd_Data_H/Plots")

infile = PLOTS / "all_Pr_m_critical.csv"



if not infile.exists():

    raise FileNotFoundError("Missing all_Pr_m_critical.csv – run the previous script first.")



df = pd.read_csv(infile)



df["Pr"] = df["Pr"].astype(float)

df["m"] = df["m"].astype(float)

df["Ra_c"] = df["Ra_c"].astype(float)

df["omega_at_Ra_c"] = df["omega_at_Ra_c"].astype(float)



rows = []



# ------------------------

# Linear interpolation helper

# ------------------------



def interpolate_zero(x1, x2, y1, y2):

    """Return x where y=0 between (x1,y1) and (x2,y2)."""

    if y2 == y1:

        return 0.5 * (x1 + x2)

    t = -y1 / (y2 - y1)

    return x1 + t * (x2 - x1)



# ------------------------

# Find zero crossings in ω for each Pr

# ------------------------



for Pr, g in df.groupby("Pr"):

    g = g.sort_values("m")



    m_vals = g["m"].values

    Ra_vals = g["Ra_c"].values

    w_vals = g["omega_at_Ra_c"].values



    for i in range(len(g) - 1):

        w1, w2 = w_vals[i], w_vals[i+1]



        if w1 == 0:

            rows.append({"Pr": Pr, "Ra_omega0": Ra_vals[i]})



        elif w1 * w2 < 0:  # sign change

            m_zero = interpolate_zero(m_vals[i], m_vals[i+1], w1, w2)

            Ra_zero = np.interp(m_zero, m_vals, Ra_vals)

            rows.append({"Pr": Pr, "Ra_omega0": Ra_zero})



df_out = pd.DataFrame(rows).sort_values("Pr")



outfile = PLOTS / "omega_zero_crossings.csv"

df_out.to_csv(outfile, index=False)

print("Saved:", outfile)



# ==========================================================

# PLOT 1 — Linear Pr axis

# ==========================================================



plt.figure(figsize=(7,5))

plt.scatter(df_out["Pr"], df_out["Ra_omega0"], c='firebrick', s=60, label="ω = 0")



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



# ==========================================================

# PLOT 2 — LOG SCALE Pr axis

# ==========================================================



plt.figure(figsize=(7,5))

plt.scatter(df_out["Pr"], df_out["Ra_omega0"], c='royalblue', s=60, label="ω = 0")



for Pr, g in df_out.groupby("Pr"):

    if len(g) == 2:

        plt.plot([Pr, Pr], g["Ra_omega0"], color='gray', lw=1)



plt.xscale("log")

plt.xlabel("Prandtl number Pr (log scale)")

plt.ylabel("Rayleigh number where ω = 0")

plt.title("Stationary–Drifting Transition Curve (ω = 0), Log-Pr Scale")

plt.grid(True, ls="--", alpha=0.6, which="both")

plt.tight_layout()



outfig2 = PLOTS / "Pr_vs_Ra_omega_zero_logPr.png"

plt.savefig(outfig2, dpi=200)

plt.close()



print("Saved:", outfig2)

print("🎉 Finished computing and plotting omega-zero crossings (linear + log Pr).")

