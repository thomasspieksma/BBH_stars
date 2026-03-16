import numpy as np
import os, re
from plot_Tcut import process_file

data_dir = "Data"
a_over_a_h = 0.1
rp_max = 5
rho = 1
N_a_h = 50
e = 0.6

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

# For each q, process all Tcut files to get H(Tcut), K(Tcut), Px, Py, Q(Tcut)
all_data = {}
for q in sorted(by_q):
    files = by_q[q]
    Tcuts = np.array([t for t, _ in files])
    H = np.full(len(files), np.nan)
    sH = np.full(len(files), np.nan)
    K = np.full(len(files), np.nan)
    sK = np.full(len(files), np.nan)
    Px = np.full(len(files), np.nan)
    sPx = np.full(len(files), np.nan)
    Py = np.full(len(files), np.nan)
    sPy = np.full(len(files), np.nan)
    Q = np.full(len(files), np.nan)
    sQ = np.full(len(files), np.nan)

    for i, (Tcut, fname) in enumerate(files):
        result = process_file(os.path.join(data_dir, fname),
                              q, e, a_over_a_h, rp_max, rho, N_a_h)
        if result is not None:
            H[i], sH[i] = result["H"], result["sH"]
            K[i], sK[i] = result["K"], result["sK"]
            Px[i], sPx[i] = result["Px"], result["sPx"]
            Py[i], sPy[i] = result["Py"], result["sPy"]
            Q[i], sQ[i] = result["Q"], result["sQ"]

    all_data[q] = (Tcuts, H, sH, K, sK, Px, sPx, Py, sPy, Q, sQ)
    print(f"q={q}: processed {len(files)} Tcut values")

# Interpolate at specific Tmax values
Tmax_values = [1e3, 1e4, 1e5, 1e6, 1e7, 1e9, 1e11]
Tmax_labels = ["1e3", "1e4", "1e5", "1e6", "1e7", "1e9", "1e11"]

for Tmax, label in zip(Tmax_values, Tmax_labels):
    fname = f"various_Tmax_{label}.dat"
    with open(fname, "w") as f:
        f.write(f"# T_max = {Tmax}\n")
        f.write("# q            H            sH           K            sK           Px            sPx           Py            sPy           Q             sQ\n")
        for q in sorted(all_data):
            Tcuts, H, sH, K, sK, Px, sPx, Py, sPy, Q, sQ = all_data[q]
            mask = np.isfinite(H)
            if not mask.any():
                continue
            log_T = np.log10(Tcuts[mask])
            log_Tm = np.log10(Tmax)
            arrays = [H, sH, K, sK, Px, sPx, Py, sPy, Q, sQ]
            if log_Tm <= log_T[0]:
                vals = [x[mask][0] for x in arrays]
            elif log_Tm >= log_T[-1]:
                vals = [x[mask][-1] for x in arrays]
            else:
                vals = [np.interp(log_Tm, log_T, x[mask]) for x in arrays]
            f.write(f"{q:.6e} " + " ".join(f"{v:.6e}" for v in vals) + "\n")
    print(f"Saved {fname}")
