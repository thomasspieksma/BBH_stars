"""
Plot resolved scattering fractions for a fixed eccentricity (e=0.6 by default).

For each data file the last column is N_resolved (out of N = 10000 trials per
velocity bin). We report:
  - mean  : average of N_resolved / N over all velocity bins
  - worst : minimum of N_resolved / N over all velocity bins

Compared datasets (all plotted vs mass ratio q):
  1. `results_bonetti_Tcut`       : Bonetti Tmax prescription ("Hubble-time")
  2. `results_Bonetti_Tcut1e11` : Bonetti Tmax fixed to 1e11
  3. `results-Tmax`              : various Tmax values (dashed curves)
"""

import numpy as np
import os
import glob
import re
import argparse

import matplotlib
matplotlib.use("Agg")  # allow saving in headless environments
import matplotlib.pyplot as plt

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


def q_filename_str(q: float) -> str:
    """Filename-safe formatting matching how q appears in stored results."""
    return f"{q:g}"


def e_filename_str(e: float) -> str:
    """Filename-safe formatting matching how e appears in stored results."""
    return f"{e:g}"


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
    parser = argparse.ArgumentParser(
        description="Resolved fraction plots vs q for fixed e"
    )
    parser.add_argument("--e", type=float, default=0.6, help="Eccentricity (default: 0.6)")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to write plots (default: 3-body/python/plots)",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save PNGs")
    args = parser.parse_args()

    e_fixed = args.e

    # We only need q values for the chosen e; current data are all at e=0.6
    q_values = sorted(set([q for q, e in bonetti_pairs if e == e_fixed]))
    if not q_values:
        raise RuntimeError(f"No (q, e={e_fixed}) pairs found in this script's config.")

    tmax_dir = os.path.join(DATA_DIR, "results-Tmax")
    bonetti_dir_hubble = os.path.join(DATA_DIR, "results_bonetti_Tcut")
    bonetti_dir_1e11 = os.path.join(DATA_DIR, "results_Bonetti_Tcut1e11")

    # Dashed Tmax values from results-Tmax.
    # Include 1e11 so we can directly compare against the solid Bonetti fixed line.
    tcut_targets = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]

    # Storage arrays (percentages)
    mean_bonetti_hubble = np.full(len(q_values), np.nan)
    worst_bonetti_hubble = np.full(len(q_values), np.nan)
    mean_bonetti_1e11 = np.full(len(q_values), np.nan)
    worst_bonetti_1e11 = np.full(len(q_values), np.nan)

    mean_dashed = {t: np.full(len(q_values), np.nan) for t in tcut_targets}
    worst_dashed = {t: np.full(len(q_values), np.nan) for t in tcut_targets}

    # Create output directory
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Load Bonetti solid curves
    # -------------------------
    for i, q in enumerate(q_values):
        q_str = q_filename_str(q)
        e_str = e_filename_str(e_fixed)

        # results_bonetti_Tcut: pick the only matching Tcut file
        hubble_pattern = os.path.join(
            bonetti_dir_hubble, f"q={q_str}_e={e_str}_Tcut=*.txt"
        )
        hubble_files = sorted(glob.glob(hubble_pattern))
        if not hubble_files:
            print(f"[WARN] Missing Bonetti hubble file for q={q_str}, e={e_str}")
        else:
            if len(hubble_files) > 1:
                print(
                    f"[WARN] Multiple Bonetti hubble matches for q={q_str}, e={e_str}; "
                    f"using first: {os.path.basename(hubble_files[0])}"
                )
            mean_bonetti_hubble[i], worst_bonetti_hubble[i] = resolved_stats(
                load_nresolved(hubble_files[0])
            )

        # results_Bonetti_Tcut1e11: fixed Tcut filename
        bonetti_1e11_fpath = os.path.join(
            bonetti_dir_1e11,
            f"q={q_str}_e={e_str}_Tcut=100000000000.txt",
        )
        if not os.path.isfile(bonetti_1e11_fpath):
            print(
                f"[WARN] Missing Bonetti 1e11 file for q={q_str}, e={e_str}: "
                f"{bonetti_1e11_fpath}"
            )
        else:
            mean_bonetti_1e11[i], worst_bonetti_1e11[i] = resolved_stats(
                load_nresolved(bonetti_1e11_fpath)
            )

    # -------------------------
    # Load dashed results-Tmax curves
    # -------------------------
    tcut_regex = re.compile(r"Tcut=([0-9.eE\+\-]+)\.txt$")
    for i, q in enumerate(q_values):
        q_str = q_filename_str(q)
        e_str = e_filename_str(e_fixed)

        pattern = os.path.join(tmax_dir, f"q={q_str}_e={e_str}_Tcut=*.txt")
        files = glob.glob(pattern)
        if not files:
            print(f"[WARN] Missing results-Tmax files for q={q_str}, e={e_str}")
            continue

        tcut_data = []
        for fpath in files:
            m = tcut_regex.search(os.path.basename(fpath))
            if m:
                tcut_data.append((float(m.group(1)), fpath))
        tcut_data.sort()

        if not tcut_data:
            print(f"[WARN] No parseable results-Tmax Tcut values for q={q_str}, e={e_str}")
            continue

        available_tcuts = np.array([t for t, _ in tcut_data], dtype=float)
        tcut_to_file = {t: fpath for t, fpath in tcut_data}

        for t_target in tcut_targets:
            idx = np.argmin(np.abs(np.log10(available_tcuts) - np.log10(t_target)))
            t_actual = float(available_tcuts[idx])
            if abs(np.log10(t_actual) - np.log10(t_target)) > np.log10(2.0):
                continue

            fpath = tcut_to_file.get(t_actual)
            if fpath is None:
                continue

            mean_dashed[t_target][i], worst_dashed[t_target][i] = resolved_stats(
                load_nresolved(fpath)
            )

    # -------------------------
    # Plotting
    # -------------------------
    def plot_panel(
        y_bonetti_hubble,
        y_bonetti_1e11,
        y_dashed_map,
        ylabel,
        title,
        yscale=None,
        ylim=None,
    ):
        fig, ax = plt.subplots(figsize=(9, 5.8))

        ax.plot(q_values, y_bonetti_hubble, "-", lw=2.2, label="Bonetti: Hubble-time Tmax")
        ax.plot(q_values, y_bonetti_1e11, "-", lw=2.2, label="Bonetti: Tmax = 1e11")

        for t_target in tcut_targets:
            ax.plot(
                q_values,
                y_dashed_map[t_target],
                "--",
                lw=1.5,
                label=f"results-Tmax: Tmax ~ {t_target:g}",
            )

        ax.set_xscale("log")
        ax.set_xlabel(r"mass ratio $q$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        if yscale is not None:
            ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            # Default for resolved-fractions-like quantities.
            if yscale is None:
                ax.set_ylim(0, 100)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        return fig

    fig_mean = plot_panel(
        mean_bonetti_hubble,
        mean_bonetti_1e11,
        mean_dashed,
        ylabel="resolved fraction [%]",
        title=f"Resolved fraction (mean) at e={e_fixed:g}",
    )
    fig_worst = plot_panel(
        worst_bonetti_hubble,
        worst_bonetti_1e11,
        worst_dashed,
        ylabel="resolved fraction [%]",
        title=f"Resolved fraction (worst) at e={e_fixed:g}",
    )

    # Better visualization when resolved is close to 100%:
    # plot the complement (unresolved fraction = 100 - resolved) on a log scale.
    mean_unres_hubble = 100.0 - mean_bonetti_hubble
    mean_unres_1e11 = 100.0 - mean_bonetti_1e11
    mean_unres_dashed = {t: 100.0 - arr for t, arr in mean_dashed.items()}

    worst_unres_hubble = 100.0 - worst_bonetti_hubble
    worst_unres_1e11 = 100.0 - worst_bonetti_1e11
    worst_unres_dashed = {t: 100.0 - arr for t, arr in worst_dashed.items()}

    # Choose sensible y-limits for log scale (avoid non-positive values).
    def auto_log_ylim(*arrays):
        vals = np.concatenate([a[np.isfinite(a)] for a in arrays])
        vals = vals[vals > 0]
        if vals.size == 0:
            return (0.1, 100.0)
        ymin = float(vals.min())
        ymax = float(vals.max())
        return (ymin / 2.0, ymax * 1.2)

    mean_unres_ylim = auto_log_ylim(mean_unres_hubble, mean_unres_1e11, *mean_unres_dashed.values())
    worst_unres_ylim = auto_log_ylim(worst_unres_hubble, worst_unres_1e11, *worst_unres_dashed.values())

    fig_mean_unres = plot_panel(
        mean_unres_hubble,
        mean_unres_1e11,
        mean_unres_dashed,
        ylabel="unresolved fraction [%] = 100 - resolved",
        title=f"Unresolved fraction (mean) at e={e_fixed:g}",
        yscale="log",
        ylim=mean_unres_ylim,
    )
    fig_worst_unres = plot_panel(
        worst_unres_hubble,
        worst_unres_1e11,
        worst_unres_dashed,
        ylabel="unresolved fraction [%] = 100 - resolved",
        title=f"Unresolved fraction (worst) at e={e_fixed:g}",
        yscale="log",
        ylim=worst_unres_ylim,
    )

    if not args.no_save:
        e_tag = f"e{e_fixed:g}".replace(".", "p")
        mean_path = os.path.join(save_dir, f"resolved_fractions_{e_tag}_mean.png")
        worst_path = os.path.join(save_dir, f"resolved_fractions_{e_tag}_worst.png")
        fig_mean.savefig(mean_path, bbox_inches="tight", dpi=200)
        fig_worst.savefig(worst_path, bbox_inches="tight", dpi=200)
        print(f"Saved: {mean_path}")
        print(f"Saved: {worst_path}")

        mean_unres_path = os.path.join(save_dir, f"unresolved_fractions_{e_tag}_mean.png")
        worst_unres_path = os.path.join(save_dir, f"unresolved_fractions_{e_tag}_worst.png")
        fig_mean_unres.savefig(mean_unres_path, bbox_inches="tight", dpi=200)
        fig_worst_unres.savefig(worst_unres_path, bbox_inches="tight", dpi=200)
        print(f"Saved: {mean_unres_path}")
        print(f"Saved: {worst_unres_path}")

    plt.close("all")
