"""
Compute the resolved fraction of scattering experiments for various (q, e) pairs.

For each data file the last column is N_resolved (out of N = 10000 trials per
velocity bin).  We report:
  - mean  : average of N_resolved / N over all velocity bins
  - worst : minimum of N_resolved / N over all velocity bins

Three datasets are processed:
  1. results-precession-soft  (T_max = 1e11)
  2. results_bonetti_Tcut     (T_cut set by Bonetti prescription)
  3. results-Tmax             (various T_cut values per (q, e) pair)
"""

import numpy as np
import os
import glob
import re

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')
N_TOTAL = 10_000  # number of scattering experiments per velocity bin

# (q, e) pairs for results-precession-soft
precession_soft_pairs = [
    (0.0001, 0.6),
    (0.0005, 0.6),
    (0.001,  0.6),
    (0.003,  0.6),
    (0.005,  0.6),
    (0.01,   0.6),
    (0.05,   0.6),
    (0.1,    0.6),
    (1,      0.6),
]

# (q, e) pairs for results_bonetti_Tcut
bonetti_pairs = [
    (0.0001, 0.6),
    (0.0005, 0.6),
    (0.001,  0.6),
    (0.003,  0.6),
    (0.005,  0.6),
    (0.01,   0.6),
    (0.05,   0.6),
    (0.1,    0.6),
    (1,      0.6),
]


def load_nresolved(filepath):
    """Load the N_resolved column (last column) from a data file."""
    data = np.loadtxt(filepath)
    return data[:, -1]


def resolved_stats(nresolved, n_total=N_TOTAL):
    """Return (mean_fraction, worst_fraction) as percentages."""
    fractions = nresolved / n_total
    return fractions.mean() * 100, fractions.min() * 100


def fmt_pct(value):
    """Format a percentage nicely, using >99.9% when appropriate."""
    if value > 99.95:
        return ">99.9%"
    else:
        return f"{value:.1f}%"


def print_table(label, pairs, path_func):
    """Print a formatted table for a set of (q, e) pairs."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'q':>8s}  {'e':>5s}  {'mean':>8s}  {'worst':>8s}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*8}")

    for q, e in pairs:
        fpath = path_func(q, e)
        if not os.path.isfile(fpath):
            print(f"  {q:>8g}  {e:>5.1f}  {'MISSING':>8s}  {'MISSING':>8s}")
            continue
        nresolved = load_nresolved(fpath)
        mean_pct, worst_pct = resolved_stats(nresolved)
        print(f"  {q:>8g}  {e:>5.1f}  {fmt_pct(mean_pct):>8s}  {fmt_pct(worst_pct):>8s}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. results-precession-soft
    def precession_soft_path(q, e):
        return os.path.join(
            DATA_DIR, 'results-precession-soft',
            f'q={q}_e={e}_Tcut=100000000000.txt'
        )

    print_table(
        "results-precession-soft  (T_max = 1e11)",
        precession_soft_pairs,
        precession_soft_path,
    )

    # 2. Combined: results-Tmax + results_bonetti_Tcut
    #    For each Bonetti (q, e) pair, show rows for various Tmax values
    #    plus a final "Bonetti" row with the Bonetti-prescribed T_cut.
    tmax_dir = os.path.join(DATA_DIR, 'results-Tmax')
    bonetti_dir = os.path.join(DATA_DIR, 'results_bonetti_Tcut')

    tcut_display = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]

    print(f"\n{'='*60}")
    print(f"  Resolved fractions vs T_cut  (results-Tmax + Bonetti)")
    print(f"{'='*60}")

    for q, e in bonetti_pairs:
        # --- Gather results-Tmax files for this (q, e) ---
        pattern = os.path.join(tmax_dir, f'q={q}_e={e}_Tcut=*.txt')
        files = glob.glob(pattern)

        tcut_data = []
        for f in files:
            m = re.search(r'Tcut=([0-9.e+]+)\.txt', os.path.basename(f))
            if m:
                tcut_data.append((float(m.group(1)), f))
        tcut_data.sort()

        # --- Header ---
        print(f"\n  q={q:<8g}  e={e}")
        print(f"  {'T_cut':>14s}  {'mean':>8s}  {'worst':>8s}")
        print(f"  {'-'*14}  {'-'*8}  {'-'*8}")

        # --- Tmax rows ---
        if tcut_data:
            available_tcuts = np.array([t for t, _ in tcut_data])
            for tcut_target in tcut_display:
                idx = np.argmin(np.abs(np.log10(available_tcuts)
                                       - np.log10(tcut_target)))
                tcut_actual, fpath = tcut_data[idx]
                if abs(np.log10(tcut_actual) - np.log10(tcut_target)) > np.log10(2):
                    continue
                nresolved = load_nresolved(fpath)
                mean_pct, worst_pct = resolved_stats(nresolved)
                print(f"  {tcut_actual:>14.3g}  {fmt_pct(mean_pct):>8s}"
                      f"  {fmt_pct(worst_pct):>8s}")
        else:
            print(f"  {'(no Tmax data)':>14s}")

        # --- Bonetti row ---
        bonetti_fpath = os.path.join(bonetti_dir, f'q={q}_e={e}.txt')
        if os.path.isfile(bonetti_fpath):
            # Read the Bonetti T_cut from the header
            with open(bonetti_fpath) as fh:
                header_line = fh.readline()
            m = re.search(r'T_cut\s*=\s*([0-9.e+]+)', header_line)
            bonetti_tcut = float(m.group(1)) if m else None

            nresolved = load_nresolved(bonetti_fpath)
            mean_pct, worst_pct = resolved_stats(nresolved)

            label = f"Bonetti ({bonetti_tcut:.2g})" if bonetti_tcut else "Bonetti"
            print(f"  {label:>14s}  {fmt_pct(mean_pct):>8s}"
                  f"  {fmt_pct(worst_pct):>8s}")

    print()
