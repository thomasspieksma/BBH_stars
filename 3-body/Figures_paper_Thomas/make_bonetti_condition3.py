import numpy as np
import os
from plot_Tcut import b_max, f_MB

results_dir = "results_bonetti_Tcut"
rp_max = 5
rho = 1
N_a_h = 50
e = 0.6

a_over_a_h_values = [0.1, 1.0]
a_over_a_h_labels = ["a01", "a10"]

def process_bonetti_file(fullpath, q, e, a_over_a_h_target, rp_max, rho, N_a_h):
    try:
        data = np.loadtxt(fullpath, comments="#")
        # 18 columns: v, DeltaE, sDeltaE, DeltaT, sDeltaT,
        #   Dvx, sDvx, Dvy, sDvy, Dvz, sDvz,
        #   DLx, sDLx, DLy, sDLy, DLz, sDLz, Nresolved
        v = data[:, 0]
        DeltaE = data[:, 1]; sDeltaE = data[:, 2]
        DeltaLz = data[:, 15]; sDeltaLz = data[:, 16]
    except:
        return None

    mu = q / (1 + q)**2
    a_h = np.logspace(3, -2, N_a_h)
    sigma = np.sqrt(mu / (4 * a_h))

    Pv = -np.pi * b_max(v, rp_max)**2 * rho * v * DeltaE
    Hv = (2 * np.pi * v**2 * b_max(v, rp_max)**2) * DeltaE / mu
    tauv_z = -np.pi * b_max(v, rp_max)**2 * rho * v * DeltaLz

    sPv = np.pi * b_max(v, rp_max)**2 * rho * v * sDeltaE
    sHv = (2 * np.pi * v**2 * b_max(v, rp_max)**2) * sDeltaE / mu
    stauv_z = np.pi * b_max(v, rp_max)**2 * rho * v * sDeltaLz

    v0 = np.hstack([0, v])
    f0 = np.vstack([np.zeros((1, N_a_h)),
                    f_MB(v[:, np.newaxis], sigma[np.newaxis, :])])

    Pv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Pv[:, np.newaxis], (1, N_a_h))])
    Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
    tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
    H_integrand0 = np.vstack([np.zeros((1, N_a_h)),
                              (sigma[np.newaxis, :] / v[:, np.newaxis]) *
                              f_MB(v[:, np.newaxis], sigma[np.newaxis, :])])

    sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])
    sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])
    stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])

    P = np.trapezoid(Pv0 * f0, x=v0, axis=0)
    H = np.trapezoid(Hv0 * H_integrand0, x=v0, axis=0)
    tau_z = np.trapezoid(tauv_z0 * f0, x=v0, axis=0)

    K = -(1 - e**2) / (2 * e) + np.sqrt(1 - e**2) / (2 * e) * tau_z / P

    weights = np.empty_like(v0)
    weights[0] = (v0[1] - v0[0]) / 2
    weights[-1] = (v0[-1] - v0[-2]) / 2
    weights[1:-1] = (v0[2:] - v0[:-2]) / 2

    sP = np.sqrt(np.sum((weights[:, None] * (sPv0 * f0))**2, axis=0))
    sH = np.sqrt(np.sum((weights[:, None] * (sHv0 * H_integrand0))**2, axis=0))
    stau_z = np.sqrt(np.sum((weights[:, None] * (stauv_z0 * f0))**2, axis=0))
    sK = np.sqrt(1 - e**2) / (2 * e) * np.abs(tau_z / P) * np.sqrt((stau_z / tau_z)**2 + (sP / P)**2)

    x = 1 / a_over_a_h_target
    H_val = np.interp(x, a_h[::-1], H[::-1])
    K_val = np.interp(x, a_h[::-1], K[::-1])
    sH_val = np.interp(x, a_h[::-1], sH[::-1])
    sK_val = np.interp(x, a_h[::-1], sK[::-1])

    return {"H": H_val, "K": K_val, "sH": sH_val, "sK": sK_val}


for a_over_a_h, label in zip(a_over_a_h_values, a_over_a_h_labels):
    q_values = []
    H_values = []
    sH_values = []
    K_values = []
    sK_values = []

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".txt"):
            continue
        q_str = fname.split("q=")[1].split("_e=")[0]
        q = float(q_str)

        fullpath = os.path.join(results_dir, fname)
        result = process_bonetti_file(fullpath, q, e, a_over_a_h, rp_max, rho, N_a_h)
        if result is None:
            print(f"Failed: {fname}")
            continue

        q_values.append(q)
        H_values.append(result["H"])
        sH_values.append(result["sH"])
        K_values.append(result["K"])
        sK_values.append(result["sK"])
        print(f"a/ah={a_over_a_h}, q={q}: H={result['H']:.4f} +/- {result['sH']:.4f}, K={result['K']:.4f} +/- {result['sK']:.4f}")

    outfile = f"bonetti_condition3_{label}.dat"
    with open(outfile, "w") as f:
        f.write(f"# Results using Bonetti's 3rd stopping condition (our code)\n")
        f.write(f"# a/a_h = {a_over_a_h}, e = {e}\n")
        f.write("# q            H            sH           K            sK\n")
        for i in range(len(q_values)):
            f.write(f"{q_values[i]:.6e} {H_values[i]:.6e} {sH_values[i]:.6e} {K_values[i]:.6e} {sK_values[i]:.6e}\n")

    print(f"Saved {outfile}\n")
