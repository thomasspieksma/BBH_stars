#!/usr/bin/env python3
"""Wrapper that runs the C `evolve` binary and plots the results."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVOLVE = os.path.join(_HERE, 'evolve')
_DATA_DIR = os.path.join(_HERE, '..', 'Data', 'results-precession-3D-velocity-soft')


def load_dat(path):
    """Load a space-delimited .dat file (comment line + header line + data)."""
    with open(path) as f:
        f.readline()                     # skip comment
        header = f.readline().split()    # column names
    data = np.loadtxt(path, skiprows=2)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {name: data[:, i] for i, name in enumerate(header)}


def main():
    ap = argparse.ArgumentParser(
        description='Run the C evolve binary and plot results')
    ap.add_argument('--q', type=float, default=1.0)
    ap.add_argument('--e0', type=float, default=0.5)
    ap.add_argument('--Vx0', type=float, default=0.0)
    ap.add_argument('--Vy0', type=float, default=0.0)
    ap.add_argument('--varpi0', type=float, default=0.0)
    ap.add_argument('--xi-start', type=float, default=None)
    ap.add_argument('--a0', type=float, default=None)
    ap.add_argument('--xi-end', type=float, default=5.0)
    ap.add_argument('--chandrasekhar',
                    choices=['integral', 'constant', 'none'],
                    default='integral')
    ap.add_argument('--data-dir', type=str, default=None)
    ap.add_argument('--freeze-e', action='store_true')
    ap.add_argument('--freeze-Vx', action='store_true')
    ap.add_argument('--freeze-Vy', action='store_true')
    ap.add_argument('--freeze-varpi', action='store_true')
    ap.add_argument('--n-ellipses', type=int, default=10,
                    help='Number of uncertainty ellipses on CoM trajectory '
                         '(default: 10)')
    ap.add_argument('--no-show', action='store_true',
                    help='Save PDF but do not call plt.show()')
    ap.add_argument('--pdf', type=str, default=None,
                    help='Output PDF path (default: evolution.pdf in this dir)')
    ap.add_argument('--save-data', type=str, default='.', metavar='DIR',
                    help='Save evolution_full.dat and evolution_V0.dat to DIR '
                         '(default: current directory)')
    ap.add_argument('--no-save-data', action='store_true',
                    help='Do not save .dat files to disk')
    args = ap.parse_args()

    if not os.path.isfile(_EVOLVE):
        print(f"Error: C binary not found at {_EVOLVE}\n"
              f"Build it first with 'make' in {_HERE}", file=sys.stderr)
        sys.exit(1)

    data_dir = args.data_dir or _DATA_DIR
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found at {data_dir}", file=sys.stderr)
        sys.exit(1)

    # build the command
    with tempfile.TemporaryDirectory(dir=_HERE) as tmpdir:
        base = os.path.join(tmpdir, 'ev')
        cmd = [
            _EVOLVE,
            '--q', str(args.q),
            '--e0', str(args.e0),
            '--Vx0', str(args.Vx0),
            '--Vy0', str(args.Vy0),
            '--varpi0', str(args.varpi0),
            '--xi-end', str(args.xi_end),
            '--chandrasekhar', args.chandrasekhar,
            '--data-dir', data_dir,
            '--output', base,
        ]
        if args.xi_start is not None:
            cmd += ['--xi-start', str(args.xi_start)]
        if args.a0 is not None:
            cmd += ['--a0', str(args.a0)]
        for flag in ('freeze_e', 'freeze_Vx', 'freeze_Vy', 'freeze_varpi'):
            if getattr(args, flag):
                cmd.append('--' + flag.replace('_', '-'))

        print(f"Running: {' '.join(cmd)}\n", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"evolve exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

        fn_full = base + '_full.dat'
        fn_v0 = base + '_V0.dat'
        if not os.path.isfile(fn_full):
            print(f"Error: expected output {fn_full} not found", file=sys.stderr)
            sys.exit(1)

        full = load_dat(fn_full)
        v0 = load_dat(fn_v0)

        if not args.no_save_data:
            os.makedirs(args.save_data, exist_ok=True)
            shutil.copy2(fn_full, os.path.join(args.save_data, 'evolution_full.dat'))
            shutil.copy2(fn_v0,   os.path.join(args.save_data, 'evolution_V0.dat'))
            print(f"Data saved to {os.path.abspath(args.save_data)}/")

    # ── Plotting ───────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    xi      = full['xi']
    a_ah    = full['a_over_ah']
    e       = full['e']
    sig_e   = full['sig_e']
    Vx      = full['Vx']
    sig_Vx  = full['sig_Vx']
    Vy      = full['Vy']
    sig_Vy  = full['sig_Vy']
    varpi   = full['varpi']
    sig_var = full['sig_varpi']
    t       = full['t']
    sig_t   = full['sig_t']
    x_pos   = full['x']
    sig_x   = full['sig_x']
    y_pos   = full['y']
    sig_y   = full['sig_y']
    C_xx    = full.get('C_xx')
    C_xy    = full.get('C_xy')
    C_yy    = full.get('C_yy')

    xi0     = v0['xi']
    a_ah0   = v0['a_over_ah']
    e0_ref  = v0['e']
    sig_e0  = v0['sig_e']
    t0_ref  = v0['t']
    sig_t0  = v0['sig_t']

    band_alpha = 0.2

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

    # Row 0, col 0: e(xi)
    ax = fig.add_subplot(gs[0, 0])
    ln, = ax.plot(xi, e)
    ax.fill_between(xi, e - sig_e, e + sig_e,
                    color=ln.get_color(), alpha=band_alpha)
    ax.set(xlabel=r'$\xi = \ln(a_h/a)$', ylabel='$e$')

    # Row 0, col 1: a/a_h(xi)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xi, a_ah)
    ax.set(xlabel=r'$\xi$', ylabel='$a/a_h$', yscale='log')

    # Row 0, col 2: CoM trajectory
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(x_pos, y_pos)
    ax.plot(x_pos[0], y_pos[0], 'o', ms=5, label='start')
    ax.plot(x_pos[-1], y_pos[-1], 's', ms=5, label='end')
    n_ell = min(args.n_ellipses, len(xi))
    ell_idx = np.linspace(0, len(xi) - 1, n_ell, dtype=int)
    has_cov = C_xx is not None and C_xy is not None and C_yy is not None
    for ii in ell_idx:
        if has_cov:
            cov2 = np.array([[C_xx[ii], C_xy[ii]],
                             [C_xy[ii], C_yy[ii]]], dtype=float)
        else:
            cov2 = np.array([[sig_x[ii] * sig_x[ii], 0.0],
                             [0.0, sig_y[ii] * sig_y[ii]]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(cov2)
        eigvals = np.maximum(eigvals, 0.0)
        width = 2 * np.sqrt(eigvals[1])
        height = 2 * np.sqrt(eigvals[0])
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        ell = Ellipse((x_pos[ii], y_pos[ii]),
                      width=width, height=height, angle=angle,
                      facecolor='C0', alpha=0.15, edgecolor='none')
        ax.add_patch(ell)
    ax.set(xlabel=r'$x\;/\;(\sigma\, T_{\rm hard})$',
           ylabel=r'$y\;/\;(\sigma\, T_{\rm hard})$')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize='small')

    # Row 1, col 0: Vx, Vy(xi)
    ax = fig.add_subplot(gs[1, 0])
    ln1, = ax.plot(xi, Vx, label=r'$V_x/\sigma$')
    ax.fill_between(xi, Vx - sig_Vx, Vx + sig_Vx,
                    color=ln1.get_color(), alpha=band_alpha)
    ln2, = ax.plot(xi, Vy, label=r'$V_y/\sigma$')
    ax.fill_between(xi, Vy - sig_Vy, Vy + sig_Vy,
                    color=ln2.get_color(), alpha=band_alpha)
    ax.set(xlabel=r'$\xi$', ylabel=r'$V/\sigma$')
    ax.legend(fontsize='small')

    # Row 1, col 1: varpi(xi)
    ax = fig.add_subplot(gs[1, 1])
    ln, = ax.plot(xi, varpi)
    ax.fill_between(xi, varpi - sig_var, varpi + sig_var,
                    color=ln.get_color(), alpha=band_alpha)
    ax.set(xlabel=r'$\xi$', ylabel=r'$\varpi$ [rad]')

    # Row 2, col 0: e(t) with V=0 ref
    ax = fig.add_subplot(gs[2, 0])
    ln, = ax.plot(t, e, label='full')
    ax.fill_between(t, e - sig_e, e + sig_e,
                    color=ln.get_color(), alpha=band_alpha)
    ln0, = ax.plot(t0_ref, e0_ref, '--', label=r'$V{=}0$ ref')
    ax.fill_between(t0_ref, e0_ref - sig_e0, e0_ref + sig_e0,
                    color=ln0.get_color(), alpha=band_alpha)
    ax.set(xlabel=r'$t\;/\;T_{\rm hard}$', ylabel='$e$')
    ax.legend(fontsize='small')

    # Row 2, col 1: a/a_h(t) with V=0 ref
    ax = fig.add_subplot(gs[2, 1])
    ln, = ax.plot(t, a_ah, label='full')
    ax.fill_betweenx(a_ah, t - sig_t, t + sig_t,
                     alpha=band_alpha, color=ln.get_color())
    ln0, = ax.plot(t0_ref, a_ah0, '--', label=r'$V{=}0$ ref')
    ax.fill_betweenx(a_ah0, t0_ref - sig_t0, t0_ref + sig_t0,
                     alpha=band_alpha, color=ln0.get_color())
    ax.set(xlabel=r'$t\;/\;T_{\rm hard}$', ylabel='$a/a_h$',
           yscale='log')
    ax.legend(fontsize='small')

    fig.suptitle(f'Binary evolution: $q={args.q}$, $e_0={args.e0}$, '
                 f'$V_0/\\sigma=({args.Vx0}, {args.Vy0})$, '
                 f'Chandrasekhar = {args.chandrasekhar}')

    pdf_path = args.pdf or os.path.join(_HERE, 'evolution.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"\nPlot saved to {pdf_path}")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
