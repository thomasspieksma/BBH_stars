"""
Plot dimensionless evolution parameters (H, K, P_x, P_y, Q) vs T_cut.
Reads raw data from Data/ (when run from Figure/). Can export .dat for the 2x2 LaTeX figure.

Usage: python plot_Tcut.py [--quick] [--export-figure-dat]
  --quick            process only first 2 (q,e) pairs and first 10 Tcut each (for testing).
  --export-figure-dat  process only q=0.002,0.01,0.2,1 at e=0.6 and write Data/panel_q=*_e=0.6.dat for the figure.
"""
import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

###################################################
# Configuration
###################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Read raw .txt from Figure/Data/
data_dir = os.path.join(SCRIPT_DIR, "Data")

a_over_a_h_target = 0.1
rp_max = 5
rho = 1
N_a_h = 50

# Same units as notes (Section 12): time in sqrt(a^3/(GM))
XLABEL_TCUT = r"$T_{\rm cut}$ [$\sqrt{a^3/(GM)}$]"

# (q,e) pairs for the 2x2 figure (e=0.6, 4 q values)
FIGURE_PAIRS = [(0.002, 0.6), (0.01, 0.6), (0.2, 0.6), (1, 0.6)]

###################################################
# Helper functions (from analysis-Tcut.py)
###################################################
def b_max(v, rp_max=rp_max):
    return rp_max * np.sqrt(1 + 2 / (v**2 * rp_max))

def f_MB(v, sigma):
    return np.sqrt(2/np.pi) * (v**2/sigma**3) * np.exp(-v**2 / (2*sigma**2))


def process_file(fullpath, q, e, a_over_a_h_target, rp_max, rho, N_a_h):
    """
    Load one Tcut file and return Maxwellian-weighted H, K, Px, Py, Q
    and their uncertainties at the target a/a_h.
    Returns dict with keys: H, K, Px, Py, Q, sH, sK, sPx, sPy, sQ.
    Returns None if file is unreadable.
    """
    try:
        v, DeltaE, sDeltaE, DeltaT, sDeltaT, Deltavx, sDeltavx, Deltavy, sDeltavy, \
        Deltavz, sDeltavz, DeltaLx, sDeltaLx, DeltaLy, sDeltaLy, DeltaLz, sDeltaLz, \
        Delta_varpi, sDelta_varpi, Nresolved = np.loadtxt(
            fullpath, unpack=True, comments="#"
        )
    except Exception as err:
        return None

    mu = q / (1 + q)**2
    a_h = np.logspace(3, -2, N_a_h)
    sigma = np.sqrt(mu / (4*a_h))

    Pv = - np.pi * b_max(v, rp_max)**2 * rho * v * DeltaE
    Fv_x = - np.pi * b_max(v, rp_max)**2 * rho * v * Deltavx
    Fv_y = - np.pi * b_max(v, rp_max)**2 * rho * v * Deltavy
    Fv_z = - np.pi * b_max(v, rp_max)**2 * rho * v * Deltavz
    tauv_x = - np.pi * b_max(v, rp_max)**2 * rho * v * DeltaLx
    tauv_y = - np.pi * b_max(v, rp_max)**2 * rho * v * DeltaLy
    tauv_z = - np.pi * b_max(v, rp_max)**2 * rho * v * DeltaLz
    Hv = (2 * np.pi * v**2 * b_max(v, rp_max)**2) * DeltaE / mu
    varpi_dot_v = np.pi * b_max(v, rp_max)**2 * rho * v * Delta_varpi

    sPv = np.pi * b_max(v, rp_max)**2 * rho * v * sDeltaE
    sFv_x = np.pi * b_max(v, rp_max)**2 * rho * v * sDeltavx
    sFv_y = np.pi * b_max(v, rp_max)**2 * rho * v * sDeltavy
    sFv_z = np.pi * b_max(v, rp_max)**2 * rho * v * sDeltavz
    stauv_x = np.pi * b_max(v, rp_max)**2 * rho * v * sDeltaLx
    stauv_y = np.pi * b_max(v, rp_max)**2 * rho * v * sDeltaLy
    stauv_z = np.pi * b_max(v, rp_max)**2 * rho * v * sDeltaLz
    sHv = (2 * np.pi * v**2 * b_max(v, rp_max)**2) * sDeltaE / mu
    svarpi_dot_v = np.pi * b_max(v, rp_max)**2 * rho * v * sDelta_varpi

    v0 = np.hstack([0, v])
    f0 = np.vstack([np.zeros((1, N_a_h)),
                    f_MB(v[:, np.newaxis], sigma[np.newaxis, :])])

    Pv0 = np.vstack([np.zeros((1, N_a_h)),
                     np.tile(Pv[:, np.newaxis], (1, N_a_h))])
    Fv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_x[:, np.newaxis], (1, N_a_h))])
    Fv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_y[:, np.newaxis], (1, N_a_h))])
    Fv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_z[:, np.newaxis], (1, N_a_h))])
    tauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_x[:, np.newaxis], (1, N_a_h))])
    tauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_y[:, np.newaxis], (1, N_a_h))])
    tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
    Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
    varpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(varpi_dot_v[:, np.newaxis], (1, N_a_h))])

    H_integrand0 = np.vstack([np.zeros((1, N_a_h)),
                              (sigma[np.newaxis, :] / v[:, np.newaxis]) *
                              f_MB(v[:, np.newaxis], sigma[np.newaxis, :])])

    sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])
    sFv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_x[:, np.newaxis], (1, N_a_h))])
    sFv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_y[:, np.newaxis], (1, N_a_h))])
    sFv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_z[:, np.newaxis], (1, N_a_h))])
    stauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_x[:, np.newaxis], (1, N_a_h))])
    stauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_y[:, np.newaxis], (1, N_a_h))])
    stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])
    sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])
    svarpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(svarpi_dot_v[:, np.newaxis], (1, N_a_h))])

    P = np.trapezoid(Pv0 * f0, x=v0, axis=0)
    F_x = np.trapezoid(Fv_x0 * f0, x=v0, axis=0)
    F_y = np.trapezoid(Fv_y0 * f0, x=v0, axis=0)
    F_z = np.trapezoid(Fv_z0 * f0, x=v0, axis=0)
    tau_x = np.trapezoid(tauv_x0 * f0, x=v0, axis=0)
    tau_y = np.trapezoid(tauv_y0 * f0, x=v0, axis=0)
    tau_z = np.trapezoid(tauv_z0 * f0, x=v0, axis=0)
    H = np.trapezoid(Hv0 * H_integrand0, x=v0, axis=0)
    varpi_dot = np.trapezoid(varpi_dot_v0 * f0, x=v0, axis=0)

    if e == 0:
        K = np.full_like(P, np.nan)
    else:
        K = -(1 - e**2)/(2*e) + np.sqrt(1 - e**2)/(2*e) * tau_z/P

    P_x = -(mu / (2*sigma)) * F_x / P
    P_y = -(mu / (2*sigma)) * F_y / P
    Q = -(mu / 2) * varpi_dot / P

    weights = np.empty_like(v0)
    weights[0] = (v0[1] - v0[0]) / 2
    weights[-1] = (v0[-1] - v0[-2]) / 2
    weights[1:-1] = (v0[2:] - v0[:-2]) / 2

    sP = np.sqrt(np.sum((weights[:, None] * (sPv0*f0))**2, axis=0))
    sF_x = np.sqrt(np.sum((weights[:, None] * (sFv_x0*f0))**2, axis=0))
    sF_y = np.sqrt(np.sum((weights[:, None] * (sFv_y0*f0))**2, axis=0))
    sF_z = np.sqrt(np.sum((weights[:, None] * (sFv_z0*f0))**2, axis=0))
    stau_x = np.sqrt(np.sum((weights[:, None] * (stauv_x0*f0))**2, axis=0))
    stau_y = np.sqrt(np.sum((weights[:, None] * (stauv_y0*f0))**2, axis=0))
    stau_z = np.sqrt(np.sum((weights[:, None] * (stauv_z0*f0))**2, axis=0))
    sH = np.sqrt(np.sum((weights[:, None] * (sHv0 * H_integrand0))**2, axis=0))
    svarpi_dot = np.sqrt(np.sum((weights[:, None] * (svarpi_dot_v0*f0))**2, axis=0))

    if e == 0:
        sK = np.full_like(K, np.nan)
    else:
        sK = np.sqrt(1-e**2)/(2*e) * np.abs(tau_z/P) * np.sqrt((stau_z/tau_z)**2 + (sP/P)**2)

    sP_x = (mu / (2*sigma)) * np.abs(F_x / P) * np.sqrt((sF_x/F_x)**2 + (sP/P)**2)
    sP_y = (mu / (2*sigma)) * np.abs(F_y / P) * np.sqrt((sF_y/F_y)**2 + (sP/P)**2)
    sQ = (mu / 2) * np.abs(varpi_dot / P) * np.sqrt((svarpi_dot/varpi_dot)**2 + (sP/P)**2)

    x = 1 / a_over_a_h_target
    H_val = np.interp(x, a_h[::-1], H[::-1])
    K_val = np.interp(x, a_h[::-1], K[::-1])
    Px_val = np.interp(x, a_h[::-1], P_x[::-1])
    Py_val = np.interp(x, a_h[::-1], P_y[::-1])
    Q_val = np.interp(x, a_h[::-1], Q[::-1])
    sH_val = np.interp(x, a_h[::-1], sH[::-1])
    sK_val = np.interp(x, a_h[::-1], sK[::-1])
    sPx_val = np.interp(x, a_h[::-1], sP_x[::-1])
    sPy_val = np.interp(x, a_h[::-1], sP_y[::-1])
    sQ_val = np.interp(x, a_h[::-1], sQ[::-1])

    # F = force magnitude (for comparison with notes), in units GM rho a
    Fx_val = np.interp(x, a_h[::-1], F_x[::-1])
    Fy_val = np.interp(x, a_h[::-1], F_y[::-1])
    sFx_val = np.interp(x, a_h[::-1], sF_x[::-1])
    sFy_val = np.interp(x, a_h[::-1], sF_y[::-1])
    F_mag = np.sqrt(Fx_val**2 + Fy_val**2)
    if F_mag > 0:
        sF_mag = np.sqrt((Fx_val * sFx_val)**2 + (Fy_val * sFy_val)**2) / F_mag
    else:
        sF_mag = np.nan

    # P = |P| magnitude of P-vector (CoM velocity evolution) from the data
    P_mag = np.sqrt(Px_val**2 + Py_val**2)
    if P_mag > 0:
        sP_mag = np.sqrt((Px_val * sPx_val)**2 + (Py_val * sPy_val)**2) / P_mag
    else:
        sP_mag = np.nan

    # P = F/H (force / hardening rate), as in the notes for comparison
    if H_val != 0 and H_val == H_val and F_mag == F_mag:
        F_over_H_val = F_mag / H_val
        if F_mag > 0 and sF_mag == sF_mag:
            sF_over_H_val = F_over_H_val * np.sqrt((sF_mag / F_mag)**2 + (sH_val / H_val)**2)
        else:
            sF_over_H_val = np.nan
    else:
        F_over_H_val = np.nan
        sF_over_H_val = np.nan

    return {
        "H": H_val, "K": K_val, "Px": Px_val, "Py": Py_val, "Q": Q_val,
        "Fx": Fx_val, "Fy": Fy_val, "sFx": sFx_val, "sFy": sFy_val,
        "F_mag": F_mag, "sF_mag": sF_mag,
        "P_mag": P_mag, "sP_mag": sP_mag,
        "F_over_H": F_over_H_val, "sF_over_H": sF_over_H_val,
        "sH": sH_val, "sK": sK_val, "sPx": sPx_val, "sPy": sPy_val, "sQ": sQ_val,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot H, K, Px, Py, Q vs T_cut")
    parser.add_argument("--quick", action="store_true", help="Few (q,e) pairs and Tcut for testing")
    parser.add_argument("--export-figure-dat", action="store_true",
                        help="Export Data/panel_q=*_e=0.6.dat for the 2x2 LaTeX figure (q=0.002,0.01,0.2,1)")
    args = parser.parse_args()

    pattern = re.compile(r"q=([0-9.]+)_e=([0-9.]+)_Tcut=([0-9]+)\.txt")

    # Group files by (q, e), each with list of (Tcut, filename) sorted by Tcut
    by_pair = {}
    for f in os.listdir(data_dir):
        m = pattern.match(f)
        if m:
            q = float(m.group(1))
            e = float(m.group(2))
            Tcut = int(m.group(3))
            key = (q, e)
            if key not in by_pair:
                by_pair[key] = []
            by_pair[key].append((Tcut, f))

    for key in by_pair:
        by_pair[key] = sorted(by_pair[key], key=lambda x: x[0])

    if args.export_figure_dat:
        by_pair = {k: by_pair[k] for k in FIGURE_PAIRS if k in by_pair}
    elif args.quick:
        pairs = sorted(by_pair.keys())[:2]
        by_pair = {k: (by_pair[k][:10] if len(by_pair[k]) >= 10 else by_pair[k]) for k in pairs}

    q_values = sorted({k[0] for k in by_pair})
    e_values = sorted({k[1] for k in by_pair})

    # Build data arrays per (q, e): Tcuts and value arrays
    data = {}
    total = sum(len(files) for files in by_pair.values())
    n = 0
    for (q, e), files in sorted(by_pair.items()):
        Tcuts = np.array([t for t, _ in files])
        N = len(Tcuts)
        out = {
            "Tcuts": Tcuts,
            "H": np.full(N, np.nan), "sH": np.full(N, np.nan),
            "K": np.full(N, np.nan), "sK": np.full(N, np.nan),
            "Px": np.full(N, np.nan), "sPx": np.full(N, np.nan),
            "Py": np.full(N, np.nan), "sPy": np.full(N, np.nan),
            "Q": np.full(N, np.nan), "sQ": np.full(N, np.nan),
            "Fx": np.full(N, np.nan), "Fy": np.full(N, np.nan), "sFx": np.full(N, np.nan), "sFy": np.full(N, np.nan),
            "F_mag": np.full(N, np.nan), "sF_mag": np.full(N, np.nan),
            "P_mag": np.full(N, np.nan), "sP_mag": np.full(N, np.nan),
            "F_over_H": np.full(N, np.nan), "sF_over_H": np.full(N, np.nan),
        }
        for i, (Tcut, filename) in enumerate(files):
            n += 1
            if n % 500 == 0 or n == total:
                print(f"Processing file {n}/{total} ...")
            fullpath = os.path.join(data_dir, filename)
            result = process_file(
                fullpath, q, e, a_over_a_h_target, rp_max, rho, N_a_h
            )
            if result is None:
                continue
            for k in ["H", "K", "Px", "Py", "Q", "Fx", "Fy", "F_mag", "P_mag", "F_over_H",
                      "sH", "sK", "sPx", "sPy", "sQ", "sFx", "sFy", "sF_mag", "sP_mag", "sF_over_H"]:
                out[k][i] = result[k]
        data[(q, e)] = out

    # Export .dat for the 2x2 LaTeX figure
    if args.export_figure_dat:
        for (q, e) in sorted(data.keys()):
            d = data[(q, e)]
            outpath = os.path.join(data_dir, f"panel_q={q}_e={e}.dat")
            with open(outpath, "w") as f:
                f.write("Tcut H sH K sK Px sPx Py sPy Q sQ\n")
                for i in range(len(d["Tcuts"])):
                    f.write(f"{d['Tcuts'][i]} {d['H'][i]} {d['sH'][i]} {d['K'][i]} {d['sK'][i]} "
                            f"{d['Px'][i]} {d['sPx'][i]} {d['Py'][i]} {d['sPy'][i]} {d['Q'][i]} {d['sQ'][i]}\n")
            print(f"Saved {outpath}")
        print("Done.")
        return

    # Grid: nrows x ncols
    nrows = len(q_values)
    ncols = len(e_values)
    figsize_per_sub = (2.2, 1.8)
    figsize = (ncols * figsize_per_sub[0], nrows * figsize_per_sub[1])

    quantities = [
        ("H", r"$H$", "H_Tcut.pdf"),
        ("K", r"$K$", "K_Tcut.pdf"),
        ("Px", r"$P_x$", "Px_Tcut.pdf"),
        ("Py", r"$P_y$", "Py_Tcut.pdf"),
        ("Q", r"$Q$", "Q_Tcut.pdf"),
        ("P_mag", r"$P$", "P_Tcut.pdf"),
        ("Fx", "$F_x$ [$GM\\rho a$]", "Fx_Tcut.pdf"),
        ("Fy", "$F_y$ [$GM\\rho a$]", "Fy_Tcut.pdf"),
        ("F_mag", "$F$ [$GM\\rho a$]", "F_Tcut.pdf"),
        ("F_over_H", r"$F/H$", "F_over_H_Tcut.pdf"),
    ]

    for qty, ylabel, outname in quantities:
        sqty = "s" + qty
        fig, axes = plt.subplots(
            nrows, ncols, sharex="col", figsize=figsize, squeeze=False
        )
        for iq, q in enumerate(q_values):
            for ie, e in enumerate(e_values):
                ax = axes[iq, ie]
                key = (q, e)
                if key not in data:
                    ax.set_visible(False)
                    continue
                d = data[key]
                T = d["Tcuts"]
                y = d[qty]
                sy = d[sqty]
                mask = np.isfinite(y)
                if np.any(mask):
                    ax.plot(T[mask], y[mask], color="C0")
                    ax.fill_between(
                        T[mask],
                        (y - sy)[mask],
                        (y + sy)[mask],
                        alpha=0.3,
                        color="C0",
                    )
                ax.set_xscale("log")
                ax.set_title(f"q={q}, e={e}")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.4)
        for ie in range(ncols):
            axes[-1, ie].set_xlabel(XLABEL_TCUT)
        fig.suptitle(f"{ylabel} vs $T_{{\\rm cut}}$ at $a/a_h={a_over_a_h_target}$", y=1.002)
        plt.tight_layout()
        outpath = os.path.join(data_dir, outname)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {outpath}")

    # Two-panel summary per variable: (1) q=0.2, vary e  (2) e=0.6, vary q
    q_fixed = 0.2
    e_fixed = 0.6
    e_vals = [e for e in e_values if (q_fixed, e) in data]
    q_vals = [q for q in q_values if (q, e_fixed) in data]

    for qty, ylabel, outname in quantities:
        sqty = "s" + qty
        base = outname.replace("_Tcut.pdf", "")
        fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        # Left: q fixed, vary e (different lines)
        for ie, e in enumerate(sorted(e_vals)):
            key = (q_fixed, e)
            d = data[key]
            T = d["Tcuts"]
            y = d[qty]
            sy = d[sqty]
            mask = np.isfinite(y)
            if np.any(mask):
                ax_left.plot(T[mask], y[mask], color=f"C{ie}", label=f"e={e}")
                ax_left.fill_between(
                    T[mask], (y - sy)[mask], (y + sy)[mask], alpha=0.3, color=f"C{ie}"
                )
        ax_left.set_xscale("log")
        ax_left.set_xlabel(XLABEL_TCUT)
        ax_left.set_ylabel(ylabel)
        ax_left.set_title(f"$q={q_fixed}$ (vary $e$)")
        if e_vals:
            ax_left.legend()
        ax_left.grid(True, alpha=0.4)

        # Right: e=0.6, vary q (different lines)
        for iq, q in enumerate(sorted(q_vals)):
            key = (q, e_fixed)
            d = data[key]
            T = d["Tcuts"]
            y = d[qty]
            sy = d[sqty]
            mask = np.isfinite(y)
            if np.any(mask):
                ax_right.plot(T[mask], y[mask], color=f"C{iq}", label=f"q={q}")
                ax_right.fill_between(
                    T[mask], (y - sy)[mask], (y + sy)[mask], alpha=0.3, color=f"C{iq}"
                )
        ax_right.set_xscale("log")
        ax_right.set_xlabel(XLABEL_TCUT)
        ax_right.set_ylabel(ylabel)
        ax_right.set_title(f"$e={e_fixed}$ (vary $q$)")
        if q_vals:
            ax_right.legend()
        ax_right.grid(True, alpha=0.4)

        fig2.suptitle(
            f"{ylabel} vs $T_{{\\rm cut}}$ at $a/a_h={a_over_a_h_target}$", y=1.02
        )
        plt.tight_layout()
        outpath2 = os.path.join(data_dir, f"{base}_Tcut_2panel.pdf")
        fig2.savefig(outpath2, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved {outpath2}")

    print("Done.")


if __name__ == "__main__":
    main()
