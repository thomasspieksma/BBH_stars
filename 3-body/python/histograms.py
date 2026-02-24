import numpy as np
import matplotlib.pyplot as plt

def load_histograms(filename):
    data = {}
    current_v = None
    current_hist = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # New velocity block
            if line.startswith("# v ="):
                try:
                    current_v = float(line.split("=")[1])
                except ValueError:
                    continue
                data[current_v] = {}

            # Histogram header
            elif line.startswith("# Histogram"):
                name = line.split("Histogram")[1].strip()
                current_hist = name
                data[current_v][current_hist] = []

            # Histogram data lines
            elif line and not line.startswith("#"):
                try:
                    x, y = line.split()
                    data[current_v][current_hist].append((float(x), int(y)))
                except Exception:
                    continue

    return data


# Example usage
if __name__ == "__main__":
    filename = "histograms_q=0.1_e=0.6_Tmax=100000.txt"  # change as needed
    hist = load_histograms(filename)

    # Pick the first velocity if available
    all_vs = sorted(hist.keys())
    if not all_vs:
        print("No histogram data found.")
        exit()

    v = all_vs[0] ### <<------ change this line to change the value of v
    print(f"Using v = {v}")

    # Print info for each histogram
    for name, bins in hist[v].items():
        xs, ys = zip(*bins)
        print(name, "→", len(xs), "bins")

    # Plot all histograms for this velocity in a single multi‑panel figure
    import math

    hist_names = list(hist[v].keys())
    k = len(hist_names)
    ncols = 3
    nrows = math.ceil(k / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for ax, name in zip(axes, hist_names):
        bins = hist[v][name]
        xs, ys = zip(*bins)

        if name == "DeltaT":
            # Compute log‑space widths from midpoints
            widths = []
            for i in range(len(xs)):
                if i == 0:
                    # use geometric spacing between first two midpoints
                    w = xs[1] / xs[0]
                    left = xs[i] / np.sqrt(w)
                    right = xs[i] * np.sqrt(w)
                elif i == len(xs) - 1:
                    w = xs[i] / xs[i-1]
                    left = xs[i] / np.sqrt(w)
                    right = xs[i] * np.sqrt(w)
                else:
                    # approximate using neighboring midpoints
                    w1 = xs[i] / xs[i-1]
                    w2 = xs[i+1] / xs[i]
                    left = xs[i] / np.sqrt(w1)
                    right = xs[i] * np.sqrt(w2)
                widths.append(right - left)
            ax.bar(xs, ys, width=widths)
            ax.set_xscale("log")
        else:
            # Linear‑space width
            width = (xs[1] - xs[0]) if len(xs) > 1 else 1
            ax.bar(xs, ys, width=width)

        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_ylabel("Counts")

    # Hide any unused subplots
    for ax in axes[k:]:
        ax.set_visible(False)

    fig.suptitle(f"All histograms for v={v}", fontsize=16)
    plt.tight_layout()
    plt.show()