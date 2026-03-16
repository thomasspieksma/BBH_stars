"""
Generate H(q) and K(q) data for Figure 3, where Tmax depends on q via:
    Tmax = 8.0 * (1+q)^3 / q^1.5 * 1e10 / 49076.0

For each q value (at e=0.6), we process all available Tcut files using the
same Maxwellian-weighted integration as in plot_Tcut.py, then interpolate
in log(Tcut) to find H and K at the q-dependent Tmax.

Two a/a_h values: 0.1 and 1.0.
Outputs:
  - figure3_Tmax_discrete_a01.dat  (datapoints at each available q, a/a_h=0.1)
  - figure3_Tmax_discrete_a10.dat  (datapoints at each available q, a/a_h=1.0)
  - figure3_Tmax_interp_a01.dat    (smooth curve interpolated in q, a/a_h=0.1)
  - figure3_Tmax_interp_a10.dat    (smooth curve interpolated in q, a/a_h=1.0)
"""
import numpy as np
import os
import re
import sys
from scipy.interpolate import PchipInterpolator

# Import process_file but we need a modified version for arbitrary a/a_h
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(SCRIPT_DIR, "Data")
out_dir = os.path.join(SCRIPT_DIR, "Figure3")

rp_max = 5
rho = 1
N_a_h = 50
e_fixed = 0.6

###################################################
# Helper functions (same as in plot_Tcut.py)
###################################################
def b_max(v, rp_max=rp_max):
    return rp_max * np.sqrt(1 + 2 / (v**2 * rp_max))

def f_MB(v, sigma):
    return np.sqrt(2/np.pi) * (v**2/sigma**3) * np.exp(-v**2 / (2*sigma**2))


def process_file_two_aah(fullpath, q, e):
    """
    Like process_file in plot_Tcut.py but returns H, K (and uncertainties)
    at BOTH a/a_h = 0.1 and a/a_h = 1.0.
    """
    try:
        v, DeltaE, sDeltaE, DeltaT, sDeltaT, Deltavx, sDeltavx, Deltavy, sDeltavy, \
        Deltavz, sDeltavz, DeltaLx, sDeltaLx, DeltaLy, sDeltaLy, DeltaLz, sDeltaLz, \
        Delta_varpi, sDelta_varpi, Nresolved = np.loadtxt(
            fullpath, unpack=True, comments="#"
        )
    except Exception:
        return None

    mu = q / (1 + q)**2
    a_h = np.logspace(3, -2, N_a_h)
    sigma = np.sqrt(mu / (4*a_h))

    Pv = - np.pi * b_max(v)**2 * rho * v * DeltaE
    tauv_z = - np.pi * b_max(v)**2 * rho * v * DeltaLz
    Hv = (2 * np.pi * v**2 * b_max(v)**2) * DeltaE / mu

    sPv = np.pi * b_max(v)**2 * rho * v * sDeltaE
    stauv_z = np.pi * b_max(v)**2 * rho * v * sDeltaLz
    sHv = (2 * np.pi * v**2 * b_max(v)**2) * sDeltaE / mu

    v0 = np.hstack([0, v])
    f0 = np.vstack([np.zeros((1, N_a_h)),
                    f_MB(v[:, np.newaxis], sigma[np.newaxis, :])])

    Pv0 = np.vstack([np.zeros((1, N_a_h)),
                     np.tile(Pv[:, np.newaxis], (1, N_a_h))])
    tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
    Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])

    H_integrand0 = np.vstack([np.zeros((1, N_a_h)),
                              (sigma[np.newaxis, :] / v[:, np.newaxis]) *
                              f_MB(v[:, np.newaxis], sigma[np.newaxis, :])])

    sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])
    stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])
    sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])

    P = np.trapezoid(Pv0 * f0, x=v0, axis=0)
    tau_z = np.trapezoid(tauv_z0 * f0, x=v0, axis=0)
    H = np.trapezoid(Hv0 * H_integrand0, x=v0, axis=0)

    if e == 0:
        K = np.full_like(P, np.nan)
    else:
        K = -(1 - e**2)/(2*e) + np.sqrt(1 - e**2)/(2*e) * tau_z/P

    # Uncertainties
    weights = np.empty_like(v0)
    weights[0] = (v0[1] - v0[0]) / 2
    weights[-1] = (v0[-1] - v0[-2]) / 2
    weights[1:-1] = (v0[2:] - v0[:-2]) / 2

    sP = np.sqrt(np.sum((weights[:, None] * (sPv0*f0))**2, axis=0))
    stau_z = np.sqrt(np.sum((weights[:, None] * (stauv_z0*f0))**2, axis=0))
    sH = np.sqrt(np.sum((weights[:, None] * (sHv0 * H_integrand0))**2, axis=0))

    if e == 0:
        sK = np.full_like(K, np.nan)
    else:
        sK = np.sqrt(1-e**2)/(2*e) * np.abs(tau_z/P) * np.sqrt((stau_z/tau_z)**2 + (sP/P)**2)

    results = {}
    for a_over_a_h in [0.1, 1.0]:
        x = 1 / a_over_a_h
        H_val = np.interp(x, a_h[::-1], H[::-1])
        K_val = np.interp(x, a_h[::-1], K[::-1])
        sH_val = np.interp(x, a_h[::-1], sH[::-1])
        sK_val = np.interp(x, a_h[::-1], sK[::-1])
        key = f"a{a_over_a_h}"
        results[key] = {"H": H_val, "sH": sH_val, "K": K_val, "sK": sK_val}

    return results


def Tmax_of_q(q):
    """Tmax = 8.0 * (1+q)^3 / q^1.5 * 1e10 / 49076.0"""
    return 8.0 * (1.0 + q)**3 / q**1.5 * 1e10 / 49076.0


###################################################
# Main
###################################################
if __name__ == "__main__":
    # Collect all (q, e=0.6, Tcut) files
    pattern = re.compile(r"q=([0-9.]+)_e=([0-9.]+)_Tcut=([0-9]+)\.txt")

    by_q = {}
    for f in os.listdir(data_dir):
        m = pattern.match(f)
        if m and float(m.group(2)) == 0.6:
            q = float(m.group(1))
            Tcut = int(m.group(3))
            by_q.setdefault(q, []).append((Tcut, f))

    for q in by_q:
        by_q[q].sort()

    print(f"Found {len(by_q)} q values for e=0.6")
    for q in sorted(by_q):
        print(f"  q={q}: {len(by_q[q])} Tcut values, "
              f"Tcut range [{by_q[q][0][0]}, {by_q[q][-1][0]}], "
              f"Tmax(q) = {Tmax_of_q(q):.3e}")

    # Process all files
    all_data = {}  # q -> {Tcuts, H_a01, sH_a01, K_a01, sK_a01, H_a10, ...}
    total_files = sum(len(v) for v in by_q.values())
    n = 0
    for q in sorted(by_q):
        files = by_q[q]
        Nt = len(files)
        Tcuts = np.array([t for t, _ in files], dtype=float)
        d = {
            "Tcuts": Tcuts,
            "H_a01": np.full(Nt, np.nan), "sH_a01": np.full(Nt, np.nan),
            "K_a01": np.full(Nt, np.nan), "sK_a01": np.full(Nt, np.nan),
            "H_a10": np.full(Nt, np.nan), "sH_a10": np.full(Nt, np.nan),
            "K_a10": np.full(Nt, np.nan), "sK_a10": np.full(Nt, np.nan),
        }
        for i, (Tcut, fname) in enumerate(files):
            n += 1
            if n % 200 == 0 or n == total_files:
                print(f"  Processing file {n}/{total_files} ...")
            result = process_file_two_aah(os.path.join(data_dir, fname), q, e_fixed)
            if result is None:
                continue
            d["H_a01"][i] = result["a0.1"]["H"]
            d["sH_a01"][i] = result["a0.1"]["sH"]
            d["K_a01"][i] = result["a0.1"]["K"]
            d["sK_a01"][i] = result["a0.1"]["sK"]
            d["H_a10"][i] = result["a1.0"]["H"]
            d["sH_a10"][i] = result["a1.0"]["sH"]
            d["K_a10"][i] = result["a1.0"]["K"]
            d["sK_a10"][i] = result["a1.0"]["sK"]
        all_data[q] = d

    # For each q, interpolate in log(Tcut) to find H,K at Tmax(q)
    # Use linear interpolation in log(Tcut) space (same as make_various_Tmax.py)
    discrete_a01 = []
    discrete_a10 = []

    for q in sorted(all_data):
        d = all_data[q]
        Tcuts = d["Tcuts"]
        Tmax = Tmax_of_q(q)
        log_Tmax = np.log10(Tmax)

        for suffix, out_list in [("a01", discrete_a01), ("a10", discrete_a10)]:
            H = d[f"H_{suffix}"]
            sH = d[f"sH_{suffix}"]
            K = d[f"K_{suffix}"]
            sK = d[f"sK_{suffix}"]

            mask = np.isfinite(H) & np.isfinite(K)
            if not mask.any():
                continue

            log_T = np.log10(Tcuts[mask])

            if log_Tmax <= log_T[0]:
                vals = [H[mask][0], sH[mask][0], K[mask][0], sK[mask][0]]
                flag = "clamp_low"
            elif log_Tmax >= log_T[-1]:
                vals = [H[mask][-1], sH[mask][-1], K[mask][-1], sK[mask][-1]]
                flag = "clamp_high"
            else:
                vals = [np.interp(log_Tmax, log_T, x[mask])
                        for x in [H, sH, K, sK]]
                flag = "interp"

            out_list.append((q, Tmax, vals[0], vals[1], vals[2], vals[3], flag))

    # Write discrete (no q-interpolation) files
    header = "# H and K at Tmax(q) = 8*(1+q)^3/q^1.5 * 1e10/49076\n"
    header += "# Interpolated in log(Tcut), no interpolation in q\n"
    header += "# q            Tmax         H            sH           K            sK           flag\n"

    for suffix, data_list in [("a01", discrete_a01), ("a10", discrete_a10)]:
        fname = os.path.join(out_dir, f"figure3_Tmax_discrete_{suffix}.dat")
        with open(fname, "w") as f:
            f.write(header.replace("H and K", f"H and K (a/a_h = {'0.1' if suffix=='a01' else '1.0'})"))
            for q, Tmax, H, sH, K, sK, flag in data_list:
                f.write(f"{q:.6e} {Tmax:.6e} {H:.6e} {sH:.6e} {K:.6e} {sK:.6e} {flag}\n")
        print(f"Saved {fname}")

    # Now create smooth q-interpolated curves using PCHIP on log(q)
    for suffix, data_list in [("a01", discrete_a01), ("a10", discrete_a10)]:
        qs = np.array([x[0] for x in data_list])
        Hs = np.array([x[2] for x in data_list])
        sHs = np.array([x[3] for x in data_list])
        Ks = np.array([x[4] for x in data_list])
        sKs = np.array([x[5] for x in data_list])

        log_qs = np.log10(qs)

        # PCHIP interpolation in log(q) space
        H_interp = PchipInterpolator(log_qs, Hs)
        sH_interp = PchipInterpolator(log_qs, sHs)
        K_interp = PchipInterpolator(log_qs, Ks)
        sK_interp = PchipInterpolator(log_qs, sKs)

        # Fine q grid
        log_q_fine = np.linspace(log_qs[0], log_qs[-1], 200)
        q_fine = 10**log_q_fine

        fname = os.path.join(out_dir, f"figure3_Tmax_interp_{suffix}.dat")
        with open(fname, "w") as f:
            f.write(f"# H and K (a/a_h = {'0.1' if suffix=='a01' else '1.0'}) at Tmax(q)\n")
            f.write("# PCHIP-interpolated in log(q) from discrete datapoints\n")
            f.write("# q            H            sH           K            sK\n")
            for i in range(len(q_fine)):
                f.write(f"{q_fine[i]:.6e} {H_interp(log_q_fine[i]):.6e} "
                        f"{sH_interp(log_q_fine[i]):.6e} {K_interp(log_q_fine[i]):.6e} "
                        f"{sK_interp(log_q_fine[i]):.6e}\n")
        print(f"Saved {fname}")

    print("\nDone!")
