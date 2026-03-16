"""
Consistency-check plot for Fig. 5 of notes-gimmy-3-body: H and K vs q at e=0.6, a/a_h=0.1.
Uses data from results-small-q, old_results/results-small-q, results-precession-soft, and results-Tmax (Tmax=1e11) with Maxwellian weighting.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import re

###################################################
# Configuration
###################################################
e_fixed = 0.6
a_over_a_h_target = 0.1
rp_max = 5
rho = 1
N_a_h = 50

# Data dirs relative to this script (run from 3-body/python/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_data_base = os.path.join(_script_dir, "..", "Data", "tests and legacy data")
_data_precession = os.path.join(_script_dir, "..", "Data", "results-precession-soft")
_data_tmax = os.path.join(_script_dir, "..", "Data", "results-Tmax")
# (label, path, file_pattern, use_20_cols): legacy = 19 cols, Tcut-based = 20 cols + Tcut in name
FILE_PATTERN_LEGACY = re.compile(r"q=([0-9.]+)_e=([0-9.]+)\.txt")
FILE_PATTERN_TCUT_1E11 = re.compile(r"q=([0-9.]+)_e=([0-9.]+)_Tcut=100000000000\.txt")
DATA_DIRS = [
    ("results-small-q", os.path.join(_data_base, "results-small-q"), FILE_PATTERN_LEGACY, False),
    ("old_results/results-small-q", os.path.join(_data_base, "old_results", "results-small-q"), FILE_PATTERN_LEGACY, False),
    ("results-precession-soft", _data_precession, FILE_PATTERN_TCUT_1E11, True),
    ("results-Tmax (Tmax=1e11)", _data_tmax, FILE_PATTERN_TCUT_1E11, True),
]

###################################################
# Functions (Maxwellian weighting as in analysis-Tcut.py / weight-Maxwellian.py)
###################################################
def b_max(v):
    return rp_max * np.sqrt(1 + 2 / (v**2 * rp_max))


def f_MB(v, sigma):
    return np.sqrt(2 / np.pi) * (v**2 / sigma**3) * np.exp(-(v**2) / (2 * sigma**2))


_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def run_pipeline_for_directory(data_dir, file_pattern=None, use_20_cols=False):
    """
    For one data directory: discover files matching file_pattern with e=e_fixed,
    run Maxwellian weighting, interpolate at a/a_h = a_over_a_h_target,
    return (q_vals, H_vals, K_vals, sH_vals, sK_vals).
    use_20_cols: True for results-precession-soft (has Delta_varpi), False for legacy 19-col.
    """
    if file_pattern is None:
        file_pattern = FILE_PATTERN_LEGACY
    parsed = []
    for f in os.listdir(data_dir):
        m = file_pattern.match(f)
        if m:
            q = float(m.group(1))
            e = float(m.group(2))
            if e == e_fixed:
                parsed.append((q, f))
    parsed = sorted(parsed, key=lambda x: x[0])
    if not parsed:
        return None, None, None, None, None

    q_vals = np.array([p[0] for p in parsed])
    N = len(q_vals)
    H_vals = np.zeros(N)
    K_vals = np.zeros(N)
    sH_vals = np.zeros(N)
    sK_vals = np.zeros(N)

    a_h = np.logspace(3, -2, N_a_h)
    x_interp = 1.0 / a_over_a_h_target  # interpolate at this 1/a_h (i.e. a/a_h = a_over_a_h_target)

    for i, (q, filename) in enumerate(parsed):
        fullpath = os.path.join(data_dir, filename)
        try:
            raw = np.loadtxt(fullpath, unpack=True)
            v = raw[0]
            DeltaE, sDeltaE = raw[1], raw[2]
            DeltaLz, sDeltaLz = raw[15], raw[16]
            if use_20_cols:
                # results-precession-soft: 20 columns including Delta_varpi, sDelta_varpi
                Delta_varpi = raw[17]
                sDelta_varpi = raw[18]
            else:
                # Legacy: 19 columns (no varpi); set to zero for H,K
                Delta_varpi = np.zeros_like(v)
                sDelta_varpi = np.zeros_like(v)
        except Exception as e:
            print(f"File unreadable {filename}: {e}")
            H_vals[i], K_vals[i] = np.nan, np.nan
            sH_vals[i], sK_vals[i] = np.nan, np.nan
            continue

        mu = q / (1 + q) ** 2
        sigma = np.sqrt(mu / (4 * a_h))

        # Per-v quantities
        Pv = -np.pi * b_max(v) ** 2 * rho * v * DeltaE
        sPv = -np.pi * b_max(v) ** 2 * rho * v * sDeltaE
        tauv_z = -np.pi * b_max(v) ** 2 * rho * v * DeltaLz
        stauv_z = -np.pi * b_max(v) ** 2 * rho * v * sDeltaLz
        Hv = (2 * np.pi * v**2 * b_max(v) ** 2) * DeltaE / mu
        sHv = (2 * np.pi * v**2 * b_max(v) ** 2) * sDeltaE / mu

        # Stack with v=0 row
        v0 = np.hstack([0, v])
        f0 = np.vstack([np.zeros((1, N_a_h)), f_MB(v[:, np.newaxis], sigma[np.newaxis, :])])
        Pv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Pv[:, np.newaxis], (1, N_a_h))])
        sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])
        tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
        stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])
        Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
        sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])
        H_integrand0 = np.vstack(
            [
                np.zeros((1, N_a_h)),
                (sigma[np.newaxis, :] / v[:, np.newaxis])
                * f_MB(v[:, np.newaxis], sigma[np.newaxis, :]),
            ]
        )

        # Integrations (trapezoidal rule, compatible with NumPy 1.x and 2.x)
        P = _trapz(Pv0 * f0, x=v0, axis=0)
        tau_z = _trapz(tauv_z0 * f0, x=v0, axis=0)
        H = _trapz(Hv0 * H_integrand0, x=v0, axis=0)

        K = -(1 - e_fixed**2) / (2 * e_fixed) + np.sqrt(1 - e_fixed**2) / (2 * e_fixed) * tau_z / P

        # Uncertainty propagation (trapezoidal weights)
        weights = np.empty_like(v0)
        weights[0] = (v0[1] - v0[0]) / 2
        weights[-1] = (v0[-1] - v0[-2]) / 2
        weights[1:-1] = (v0[2:] - v0[:-2]) / 2
        sP = np.sqrt(np.sum((weights[:, np.newaxis] * (sPv0 * f0)) ** 2, axis=0))
        stau_z = np.sqrt(np.sum((weights[:, np.newaxis] * (stauv_z0 * f0)) ** 2, axis=0))
        sH = np.sqrt(np.sum((weights[:, np.newaxis] * (sHv0 * H_integrand0)) ** 2, axis=0))
        sK = (
            np.sqrt(1 - e_fixed**2)
            / (2 * e_fixed)
            * np.abs(tau_z / P)
            * np.sqrt((stau_z / tau_z) ** 2 + (sP / P) ** 2)
        )

        # Interpolate at a/a_h = a_over_a_h_target (same pattern as grid-plot.py)
        H_vals[i] = np.interp(x_interp, a_h[::-1], H[::-1])
        K_vals[i] = np.interp(x_interp, a_h[::-1], K[::-1])
        sH_vals[i] = np.interp(x_interp, a_h[::-1], sH[::-1])
        sK_vals[i] = np.interp(x_interp, a_h[::-1], sK[::-1])

    return q_vals, H_vals, K_vals, sH_vals, sK_vals


###################################################
# Main: run for both directories and plot
###################################################
def main():
    results = []
    for item in DATA_DIRS:
        label, data_dir, file_pattern, use_20_cols = item
        if not os.path.isdir(data_dir):
            print(f"Skip (not a directory): {data_dir}")
            continue
        q_vals, H_vals, K_vals, sH_vals, sK_vals = run_pipeline_for_directory(
            data_dir, file_pattern=file_pattern, use_20_cols=use_20_cols
        )
        if q_vals is None:
            print(f"No e={e_fixed} files in {data_dir}")
            continue
        results.append((label, q_vals, H_vals, K_vals, sH_vals, sK_vals))

    if not results:
        raise RuntimeError("No data found in any of the data directories.")

    fig, (axH, axK) = plt.subplots(1, 2, figsize=(12, 5))

    for label, q_vals, H_vals, K_vals, sH_vals, sK_vals in results:
        axH.plot(q_vals, H_vals, label=label)
        axH.fill_between(q_vals, H_vals - sH_vals, H_vals + sH_vals, alpha=0.3)
        axK.plot(q_vals, K_vals, label=label)
        axK.fill_between(q_vals, K_vals - sK_vals, K_vals + sK_vals, alpha=0.3)

    axH.set_xscale("log")
    axH.set_xlabel(r"$q$")
    axH.set_ylabel(r"$H$")
    axH.set_title(rf"$H$ (e={e_fixed}, $a/a_h$={a_over_a_h_target})")
    axH.legend()
    axH.grid(True, alpha=0.3)

    axK.set_xscale("log")
    axK.set_xlabel(r"$q$")
    axK.set_ylabel(r"$K$")
    axK.set_title(rf"$K$ (e={e_fixed}, $a/a_h$={a_over_a_h_target})")
    axK.legend()
    axK.grid(True, alpha=0.3)

    fig.suptitle("Fig. 5 consistency check: results-small-q, old_results/results-small-q, results-precession-soft, results-Tmax (Tmax=1e11)")
    plt.tight_layout()
    outpath = os.path.join(_script_dir, "Thomas_testing_fig.png")
    fig.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")
    plt.show()


if __name__ == "__main__":
    main()
