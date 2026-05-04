"""Quick plot of H and K vs Tcut for the smallest q (test data)."""
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(SCRIPT_DIR, "Figure3")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for suffix, a_label, color in [("a01", "a/a_h=0.1", "C0"), ("a10", "a/a_h=1.0", "C1")]:
    fname = os.path.join(fig_dir, f"test_various_Tcut_q0.0001_{suffix}.dat")
    Tcut, H, sH, K, sK = np.loadtxt(fname, unpack=True)

    ax1.plot(Tcut, H, '-o', ms=2, color=color, label=a_label)
    ax1.fill_between(Tcut, H - sH, H + sH, alpha=0.2, color=color)

    ax2.plot(Tcut, K, '-o', ms=2, color=color, label=a_label)
    ax2.fill_between(Tcut, K - sK, K + sK, alpha=0.2, color=color)

for ax, ylabel in [(ax1, "H"), (ax2, "K")]:
    ax.set_xscale("log")
    ax.set_xlabel("Tcut")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("q = 0.0001,  e = 0.6", fontsize=13)
fig.tight_layout()
out = os.path.join(fig_dir, "test_Tcut_q0.0001.png")
fig.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
