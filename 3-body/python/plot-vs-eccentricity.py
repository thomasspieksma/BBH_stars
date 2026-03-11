#!/usr/bin/env python3
"""
plot-vs-eccentricity.py

Plots dimensionless scattering parameters (H, K, P_x, P_y, Q, R_x, R_y)
as a function of binary eccentricity for fixed mass ratio q and hardness
ratio a/a_h, using isotropic (V=0) Maxwellian reweighting of 3-body
scattering harmonics data.

Usage:
    python plot-vs-eccentricity.py
    python plot-vs-eccentricity.py --q 0.5 --a-over-ah 0.1
    python plot-vs-eccentricity.py --save quantities_vs_e.pdf --no-show
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import importlib
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))

_wm3d = importlib.import_module('weight-Maxwellian-3D-velocity')
reweight_from_harmonics = _wm3d.reweight_from_harmonics
read_harmonics = _wm3d.read_harmonics

sys.path.insert(0, os.path.join(_HERE, '..', 'binary-evolution'))
from evolve import load_harmonics_data


def main():
    parser = argparse.ArgumentParser(
        description='Plot scattering parameters vs eccentricity')
    parser.add_argument('--q', type=float, default=0.2,
                        help='Mass ratio (default: 0.2)')
    parser.add_argument('--a-over-ah', type=float, default=0.33,
                        help='Hardness ratio a/a_h (default: 0.33)')
    parser.add_argument('--data-dir', default=None,
                        help='Directory containing harmonics_q=*_e=*.bin files')
    parser.add_argument('--save', default=None,
                        help='Save figure to this file (e.g. plot.pdf)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not open interactive plot window')
    args = parser.parse_args()

    q = args.q
    a_over_ah = args.a_over_ah
    mu = q / (1 + q)**2
    a_h = 1.0 / a_over_ah
    sigma = np.sqrt(mu / (4 * a_h))

    print(f"q = {q},  a/a_h = {a_over_ah},  a_h = {a_h:.4f},  "
          f"sigma = {sigma:.6f},  mu = {mu:.6f}")

    data, e_grid = load_harmonics_data(q, data_dir=args.data_dir)

    # K is singular at e=0 and e=1; skip those endpoints
    e_grid = e_grid[(e_grid > 0) & (e_grid < 1)]

    V_zero = np.zeros(3)

    keys = ['H', 'K', 'P_x', 'P_y', 'Q', 'R_x', 'R_y']
    err_keys = ['s' + k for k in keys]
    arrays = {k: np.empty(len(e_grid)) for k in keys + err_keys}

    for i, e_val in enumerate(e_grid):
        meta, harm_bins = data[e_val]
        r = reweight_from_harmonics(meta, harm_bins, V_zero, sigma)
        for k in keys:
            arrays[k][i] = r[k]
            arrays['s' + k][i] = r['s' + k]
        print(f"  e = {e_val:.2f}:  H = {r['H']:.4f},  K = {r['K']:+.4f},  "
              f"P_x = {r['P_x']:+.4f},  P_y = {r['P_y']:+.4f},  "
              f"Q = {r['Q']:+.4f}")

    # ── Plotting ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    panels = [
        ('H',   r'$H$',                        axes[0, 0]),
        ('K',   r'$K$',                         axes[0, 1]),
        ('P_x', r'$P_{\hat e}$',                axes[0, 2]),
        ('P_y', r'$P_{\hat n}$',                axes[1, 0]),
        ('Q',   r'$Q$',                         axes[1, 1]),
    ]
    ax_R = axes[1, 2]

    for key, ylabel, ax in panels:
        y = arrays[key]
        sy = arrays['s' + key]
        mask = np.isfinite(y)
        ax.plot(e_grid[mask], y[mask], 'o-', ms=3, lw=1.2)
        ax.fill_between(e_grid[mask], (y - sy)[mask], (y + sy)[mask],
                        alpha=0.2)
        ax.set_xlabel(r'$e$')
        ax.set_ylabel(ylabel)
        ax.axhline(0, color='grey', lw=0.5, ls='--')

    Rx = arrays['R_x']
    sRx = arrays['s' + 'R_x']
    Ry = arrays['R_y']
    sRy = arrays['s' + 'R_y']
    mask_Rx = np.isfinite(Rx)
    mask_Ry = np.isfinite(Ry)
    ax_R.plot(e_grid[mask_Rx], Rx[mask_Rx], 'o-', ms=3, lw=1.2,
              label=r'$R_{\hat e}$')
    ax_R.fill_between(e_grid[mask_Rx], (Rx - sRx)[mask_Rx],
                      (Rx + sRx)[mask_Rx], alpha=0.2)
    ax_R.plot(e_grid[mask_Ry], Ry[mask_Ry], 's-', ms=3, lw=1.2,
              label=r'$R_{\hat n}$')
    ax_R.fill_between(e_grid[mask_Ry], (Ry - sRy)[mask_Ry],
                      (Ry + sRy)[mask_Ry], alpha=0.2)
    ax_R.set_xlabel(r'$e$')
    ax_R.set_ylabel(r'$R$')
    ax_R.axhline(0, color='grey', lw=0.5, ls='--')
    ax_R.legend()

    fig.suptitle(f'$q = {q}$,  $a/a_h = {a_over_ah}$', fontsize=13)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, bbox_inches='tight')
        print(f"Saved to {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
