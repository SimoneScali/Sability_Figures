
import glob, os, json
import numpy as np


#def linear_fit_slope_and_stderr(t, y):

    ## Fit y = a t + b, return a and stderr(a)
    ## Using least squares with covariance

    #A = np.vstack([t, np.ones_like(t)]).T

    #coef, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

    #a, b = coef

    #n = len(t)

    #if n < 3:

        #return a, np.nan

    #if residuals.size == 0:

        ## Perfect fit or underdetermined

        #return a, np.nan

    #s2 = residuals[0] / (n - 2)

    #cov = s2 * np.linalg.inv(A.T @ A)

    #stderr_a = np.sqrt(cov[0, 0])

    #return a, stderr_a


def linear_fit_slope_and_stderr(t, y, w=None):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)

    if n < 3:
        return np.nan, np.nan

    A = np.vstack([t, np.ones_like(t)]).T  

    if w is None:
        coef, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        a, b = coef

        if residuals.size == 0:
            return a, np.nan

        s2 = residuals[0] / (n - 2)
        cov = s2 * np.linalg.inv(A.T @ A)
        stderr_a = np.sqrt(cov[0, 0])
        return a, stderr_a

    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, np.inf)

    if not np.any(w > 0):
        return linear_fit_slope_and_stderr(t, y, w=None)

    W = np.sqrt(w)[:, None]          
    Aw = A * W                       
    yw = y * W[:, 0]                 

    coef, residuals, rank, s = np.linalg.lstsq(Aw, yw, rcond=None)
    a, b = coef

    yhat = A @ coef
    r = y - yhat

    rss = np.sum(w * r**2)

    dof = max(int(np.sum(w > 0)) - 2, 1)
    s2 = rss / dof

    cov = s2 * np.linalg.inv(A.T @ (A * w[:, None]))
    stderr_a = np.sqrt(cov[0, 0]) if np.isfinite(cov[0, 0]) else np.nan
    return a, stderr_a



def main(movie_test_dir="movie_test", use_last_fraction=0.6, m_max=20):
    files = sorted(glob.glob(os.path.join(movie_test_dir, "test_*.npz")))
    if not files:
        raise RuntimeError(f"No test_*.npz files found in {movie_test_dir}")

    times = []
    temps = []
    meta = None

    for f in files:
        d = np.load(f, allow_pickle=True)
        T = d["temperature"]          
        t = float(np.atleast_1d(d["time"])[0])
        times.append(t)
        temps.append(T)

        if meta is None:
            meta = {
                "Ek": float(np.atleast_1d(d["Ek"])[0]),
                "Pr": float(np.atleast_1d(d["Pr"])[0]),
                "Ra": float(np.atleast_1d(d["Ra"])[0]),
                "grid_phi": d["grid_phi"].astype(float),
                "grid_r": d["grid_r"].astype(float),
            }

    times = np.array(times)
    order = np.argsort(times)
    times = times[order]
    temps = [temps[i] for i in order]


    nr, nphi = temps[0].shape
    Tstack = np.stack(temps, axis=0)  
    nt = Tstack.shape[0]

    # Use only the last fraction 
    i0 = int((1.0 - use_last_fraction) * nt)
    tsel = times[i0:]
    Tsel = Tstack[i0:, :, :]


    F = np.fft.rfft(Tsel, axis=-1)  
    nmodes = F.shape[-1]
    m_cap = min(m_max, nmodes - 1)


    power = np.mean(np.abs(F[:, :, 1:m_cap+1])**2, axis=(0, 1))  
    m_star = int(np.argmax(power) + 1)


    amp_r = np.mean(np.abs(F[:, :, m_star]), axis=0)  
    r_star = int(np.argmax(amp_r))



    # Complex amplitude time series for  m and radius

    A = F[:, r_star, m_star]
    phase = np.unwrap(np.angle(A))

    # Fit phase = omega * t + const
    #omega, omega_stderr = linear_fit_slope_and_stderr(tsel, phase)
    #fd = omega / (2*np.pi)
    #fd_stderr = omega_stderr / (2*np.pi) if np.isfinite(omega_stderr) else None

    weights = np.abs(A)**2
    omega_phase, omega_phase_stderr = linear_fit_slope_and_stderr(tsel, phase, w=weights)
    
    # Convert Fourier-phase slope -> pattern drift frequency:
    # phase(A_m) ~ -m * Omega_d * t  (for a rigidly drifting pattern)
    sign = -1.0   # flip if you want the opposite convention

    omega_d = abs(sign) * omega_phase / m_star
    omega_d_stderr = (abs(sign) * omega_phase_stderr / m_star) if np.isfinite(omega_phase_stderr) else np.nan

    fd = omega_d / (2*np.pi)
    fd_stderr = (omega_d_stderr / (2*np.pi)) if np.isfinite(omega_d_stderr) else None


    out = {
        "Ek": meta["Ek"],
        "Pr": meta["Pr"],
        "Ra": meta["Ra"],
        "m_star": m_star,
        "r_index_star": r_star,
        "omega_phase": float(omega_phase),
        "omega_d": float(omega_d),
        "fd": float(fd),
        "fd_stderr": (float(fd_stderr) if fd_stderr is not None else None),
        "nt_total": int(nt),
        "nt_used": int(len(tsel)),
        "t_start": float(tsel[0]),
        "t_end": float(tsel[-1]),
        "use_last_fraction": float(use_last_fraction),
    }


    with open(os.path.join(movie_test_dir, "drift.json"), "w") as f:
        json.dump(out, f, indent=2)


    print("Wrote", os.path.join(movie_test_dir, "drift.json"))
    print("m* =", m_star, " r* index =", r_star, " fd =", fd, " +/-", fd_stderr)


if __name__ == "__main__":
    main()

