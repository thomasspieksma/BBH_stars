#!/usr/bin/env python3
"""
evolve.py — Binary evolution solver.

Evolves the binary state (e, V/sigma, varpi) as a function of the
hardening variable xi = ln(a_h/a), using three-body scattering data
and Chandrasekhar dynamical friction.

All quantities are dimensionless:
  - Velocities in units of sigma
  - Time in units of T_hard = sigma / (G rho a_h)
  - Lengths in units of a_h

Usage:
    python evolve.py                                 # default example
    python evolve.py --q 0.5 --e0 0.3               # custom parameters
    python evolve.py --Vx0 0.5 --chandrasekhar integral
"""

import numpy as np
import os
import sys
import glob
import re
import importlib
import time as _time
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf

# ---------------------------------------------------------------------------
# Import the reweighting module (hyphenated filename requires importlib)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'python'))
_wm3d = importlib.import_module('weight-Maxwellian-3D-velocity')
read_harmonics = _wm3d.read_harmonics
reweight_from_harmonics = _wm3d.reweight_from_harmonics

_DEFAULT_DATA_DIR = os.path.join(
    _HERE, '..', 'Data', 'results-precession-3D-velocity-soft')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_harmonics_data(q, data_dir=None):
    """Load harmonics files for all available *e* values at a given *q*.

    Parameters
    ----------
    q : float
        Mass ratio.
    data_dir : str or None
        Directory containing ``harmonics_q=..._e=...bin`` files.

    Returns
    -------
    data : dict
        ``{e_value: (meta, harm_bins)}``
    e_grid : ndarray
        Sorted array of available eccentricity values.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    # q may be stored as "1" or "1.0" in filenames — try both
    candidates = [str(q)]
    if isinstance(q, float) and q == int(q):
        candidates.append(str(int(q)))
    elif isinstance(q, int):
        candidates.append(str(float(q)))

    files = []
    for q_str in candidates:
        files = glob.glob(os.path.join(data_dir, f'harmonics_q={q_str}_e=*.bin'))
        if files:
            break

    if not files:
        raise FileNotFoundError(
            f"No harmonics files found for q={q} in {data_dir}")

    data = {}
    for fpath in files:
        m = re.search(r'_e=([0-9.]+)\.bin$', fpath)
        if m:
            e_val = float(m.group(1))
            meta, harm_bins = read_harmonics(fpath)
            data[e_val] = (meta, harm_bins)

    e_grid = np.sort(np.array(list(data.keys())))
    print(f"  Loaded q={q}: {len(data)} eccentricity values "
          f"(e = {e_grid[0]:.1f} … {e_grid[-1]:.1f})")
    return data, e_grid


# ── Rate evaluation (with eccentricity interpolation) ─────────────────────────

def compute_rates(xi, e, Vx_s, Vy_s, varpi, q, data, e_grid):
    """Compute dimensionless evolution parameters at the current state.

    Parameters
    ----------
    xi : float
        Hardening variable ln(a_h / a).
    e : float
        Eccentricity.
    Vx_s, Vy_s : float
        Centre-of-mass velocity / sigma in the lab frame.
    varpi : float
        Longitude of periapsis.
    q : float
        Mass ratio.
    data, e_grid :
        Output of :func:`load_harmonics_data`.

    Returns
    -------
    H, K, Px, Py, Q : float
        Hardening rate, eccentricity growth rate, lab-frame acceleration
        components, and precession parameter.
    """
    cos_w, sin_w = np.cos(varpi), np.sin(varpi)

    # Project velocity to binary frame
    Ve_s = Vx_s * cos_w + Vy_s * sin_w
    Vn_s = -Vx_s * sin_w + Vy_s * cos_w

    # Code-unit velocity dispersion: sigma / sqrt(GM/a)
    #   a_h = G mu / (4 sigma^2)  =>  GM/sigma^2 = 4(1+q)^2 a_h / q
    #   sigma_code = sigma / sqrt(GM/a) = sqrt(q / (4(1+q)^2)) * sqrt(a/a_h)
    sigma_code = np.sqrt(q) / (2.0 * (1.0 + q)) * np.exp(-xi / 2.0)

    V0_code = np.array([Ve_s * sigma_code,
                        Vn_s * sigma_code,
                        0.0])

    def _eval_at_e(e_val):
        meta, harm_bins = data[e_val]
        res = reweight_from_harmonics(meta, harm_bins, V0_code, sigma_code)
        H = res['H']
        K = res['K'] if np.isfinite(res['K']) else 0.0
        Pe = res['Q_x'] if np.isfinite(res['Q_x']) else 0.0
        Pn = res['Q_y'] if np.isfinite(res['Q_y']) else 0.0
        Q = res['tildeQ'] if np.isfinite(res['tildeQ']) else 0.0
        return H, K, Pe, Pn, Q

    # Use only e > 0 grid points (K formula is singular at e=0)
    eg = e_grid[e_grid > 0] if e_grid[0] == 0.0 else e_grid

    # Snap to nearest grid point if within tolerance
    snap_idx = np.argmin(np.abs(eg - e))
    if abs(eg[snap_idx] - e) < 1e-10:
        H, K, Pe, Pn, Q = _eval_at_e(eg[snap_idx])
    elif e <= eg[0]:
        H, K, Pe, Pn, Q = _eval_at_e(eg[0])
    elif e >= eg[-1]:
        H, K, Pe, Pn, Q = _eval_at_e(eg[-1])
    else:
        idx = int(np.searchsorted(eg, e)) - 1
        idx = max(0, min(idx, len(eg) - 2))
        e_lo, e_hi = eg[idx], eg[idx + 1]

        H_lo, K_lo, Pe_lo, Pn_lo, Q_lo = _eval_at_e(e_lo)
        H_hi, K_hi, Pe_hi, Pn_hi, Q_hi = _eval_at_e(e_hi)

        frac = (e - e_lo) / (e_hi - e_lo)
        H  = H_lo  + frac * (H_hi  - H_lo)
        K  = K_lo  + frac * (K_hi  - K_lo)
        Pe = Pe_lo + frac * (Pe_hi - Pe_lo)
        Pn = Pn_lo + frac * (Pn_hi - Pn_lo)
        Q  = Q_lo  + frac * (Q_hi  - Q_lo)

    # Rotate P from binary frame to lab frame
    Px = Pe * cos_w - Pn * sin_w
    Py = Pe * sin_w + Pn * cos_w

    return H, K, Px, Py, Q


# ── Precomputed rate tables ───────────────────────────────────────────────────

def _sigma_code(xi, q):
    return np.sqrt(q) / (2.0 * (1.0 + q)) * np.exp(-xi / 2.0)


def precompute_rate_tables(q, data, e_grid, xi_span, n_xi=50, eps_V=0.01):
    r"""Precompute H, K, Q and P linear-response coefficients on a (xi, e) grid.

    At each grid point, three calls to ``reweight_from_harmonics``:
    V=0 (for H, K, Q), V along e-hat (for A_ee, A_ne), V along n-hat
    (for A_en, A_nn).  The acceleration parameter is then
    P_ehat ≈ A_ee * Ve/σ + A_en * Vn/σ  (linear in velocity).
    """
    eg = e_grid[e_grid > 0] if e_grid[0] == 0.0 else e_grid.copy()
    margin = 0.5
    xi_lo = max(xi_span[0] - margin, 0.0)
    xi_hi = xi_span[1] + margin
    xi_arr = np.linspace(xi_lo, xi_hi, n_xi)
    n_e = len(eg)

    H_tab  = np.zeros((n_xi, n_e))
    K_tab  = np.zeros((n_xi, n_e))
    Q_tab  = np.zeros((n_xi, n_e))
    A_ee_tab = np.zeros((n_xi, n_e))
    A_en_tab = np.zeros((n_xi, n_e))
    A_ne_tab = np.zeros((n_xi, n_e))
    A_nn_tab = np.zeros((n_xi, n_e))

    total = n_xi * n_e
    t0 = _time.time()

    for i, xi in enumerate(xi_arr):
        sc = _sigma_code(xi, q)
        for j, e_val in enumerate(eg):
            meta, hb = data[e_val]

            # V = 0
            res0 = reweight_from_harmonics(meta, hb, np.zeros(3), sc)
            H_tab[i, j] = res0['H']
            K_tab[i, j] = res0['K'] if np.isfinite(res0['K']) else 0.0
            Q_tab[i, j] = res0['tildeQ'] if np.isfinite(res0['tildeQ']) else 0.0

            # V along e-hat  (→ A_ee, A_ne)
            res_e = reweight_from_harmonics(
                meta, hb, np.array([eps_V * sc, 0.0, 0.0]), sc)
            Pe_e = res_e['Q_x'] if np.isfinite(res_e['Q_x']) else 0.0
            Pn_e = res_e['Q_y'] if np.isfinite(res_e['Q_y']) else 0.0
            A_ee_tab[i, j] = Pe_e / eps_V
            A_ne_tab[i, j] = Pn_e / eps_V

            # V along n-hat  (→ A_en, A_nn)
            res_n = reweight_from_harmonics(
                meta, hb, np.array([0.0, eps_V * sc, 0.0]), sc)
            Pe_n = res_n['Q_x'] if np.isfinite(res_n['Q_x']) else 0.0
            Pn_n = res_n['Q_y'] if np.isfinite(res_n['Q_y']) else 0.0
            A_en_tab[i, j] = Pe_n / eps_V
            A_nn_tab[i, j] = Pn_n / eps_V

        done = (i + 1) * n_e
        elapsed = _time.time() - t0
        eta = elapsed / (i + 1) * (n_xi - i - 1)
        print(f"\r  Precomputing rates: {done}/{total} "
              f"({100*done/total:.0f}%), ETA {eta:.0f}s   ",
              end='', flush=True)

    print(f"\r  Precomputed rates in {_time.time()-t0:.1f}s"
          + " " * 30)

    def _interp(tab):
        return RegularGridInterpolator(
            (xi_arr, eg), tab, method='linear',
            bounds_error=False, fill_value=None)

    return {
        'xi_grid': xi_arr, 'e_grid': eg, 'q': q,
        'H': _interp(H_tab), 'K': _interp(K_tab), 'Q': _interp(Q_tab),
        'A_ee': _interp(A_ee_tab), 'A_en': _interp(A_en_tab),
        'A_ne': _interp(A_ne_tab), 'A_nn': _interp(A_nn_tab),
    }


def compute_rates_fast(xi, e, Vx_s, Vy_s, varpi, tables):
    """Fast rate evaluation via precomputed interpolation tables."""
    cos_w, sin_w = np.cos(varpi), np.sin(varpi)

    Ve_s =  Vx_s * cos_w + Vy_s * sin_w
    Vn_s = -Vx_s * sin_w + Vy_s * cos_w

    xi_c = np.clip(xi, tables['xi_grid'][0], tables['xi_grid'][-1])
    e_c  = np.clip(e,  tables['e_grid'][0],  tables['e_grid'][-1])
    pt = np.array([[xi_c, e_c]])

    H = tables['H'](pt).item()
    K = tables['K'](pt).item()
    Q = tables['Q'](pt).item()

    A_ee = tables['A_ee'](pt).item()
    A_en = tables['A_en'](pt).item()
    A_ne = tables['A_ne'](pt).item()
    A_nn = tables['A_nn'](pt).item()

    Pe = A_ee * Ve_s + A_en * Vn_s
    Pn = A_ne * Ve_s + A_nn * Vn_s

    Px = Pe * cos_w - Pn * sin_w
    Py = Pe * sin_w + Pn * cos_w

    return H, K, Px, Py, Q


# ── Chandrasekhar dynamical friction ──────────────────────────────────────────

def _ln_lambda(u_tilde, xi, q, r_outer_ah):
    """Coulomb logarithm ln(r_outer / b_max) as a function of u/sigma.

    In dimensionless variables (lengths in a_h, velocities in sigma):
        b_max / a_h = 5 exp(-xi) sqrt(1 + 8(1+q)^2 exp(xi) / (5 q u_tilde^2))
    """
    a_over_ah = np.exp(-xi)
    ratio = 8.0 * (1.0 + q)**2 * np.exp(xi) / (5.0 * q * u_tilde**2)
    bmax_ah = 5.0 * a_over_ah * np.sqrt(1.0 + ratio)
    lnL = np.log(r_outer_ah / bmax_ah)
    if lnL < 0.0:
        return 0.0
    return lnL


def chandrasekhar_decel_integral(V_tilde, xi, q, r_outer_ah):
    r"""Full Chandrasekhar integral with velocity-dependent ln Lambda.

    Returns the dimensionless scalar *J* such that the Chandrasekhar
    contribution to :math:`d\tilde V_x / d\xi` is

    .. math::
        \frac{16\pi (1+q)^2 e^\xi}{q\,H}\; J\; \hat V_x

    *J* is negative (deceleration).
    """
    if V_tilde < 1e-15:
        return 0.0

    def integrand(u):
        if u < 1e-30:
            return 0.0
        lnL = _ln_lambda(u, xi, q, r_outer_ah)
        if lnL <= 0.0:
            return 0.0
        alpha = u * V_tilde
        if alpha < 1e-4:
            # Taylor: sinh(a)-a*cosh(a) ≈ -a^3/3 - a^5/30
            kernel_exp = np.exp(-(u**2 + V_tilde**2) / 2.0) * (
                -alpha**3 / 3.0 - alpha**5 / 30.0)
        else:
            # Stable form: exp(-s/2)*(sinh(a)-a*cosh(a))
            #   = [(1-a)*exp(-(u-V)^2/2) - (1+a)*exp(-(u+V)^2/2)] / 2
            e_minus = np.exp(-0.5 * (u - V_tilde)**2)
            e_plus = np.exp(-0.5 * (u + V_tilde)**2)
            kernel_exp = 0.5 * ((1.0 - alpha) * e_minus
                                - (1.0 + alpha) * e_plus)
        return lnL * 2.0 * kernel_exp / alpha**2

    J, _ = quad(integrand, 0.0, np.inf, limit=200)
    J /= np.sqrt(2.0 * np.pi)
    return J


def chandrasekhar_decel_constant(V_tilde, xi, q, r_outer_ah):
    r"""Constant-ln-Lambda Chandrasekhar approximation.

    Evaluates ln Lambda at :math:`u = \sigma` (``u_tilde = 1``) and
    uses the standard erf formula.  Returns *J* with the same meaning
    as :func:`chandrasekhar_decel_integral`.
    """
    if V_tilde < 1e-15:
        return 0.0

    lnL = _ln_lambda(1.0, xi, q, r_outer_ah)
    if lnL <= 0.0:
        return 0.0

    X = V_tilde / np.sqrt(2.0)
    bracket = erf(X) - 2.0 * X / np.sqrt(np.pi) * np.exp(-X**2)
    return -lnL / V_tilde**2 * bracket


# ── Solver ────────────────────────────────────────────────────────────────────

def solve(q, e0, Vx0_s=0.0, Vy0_s=0.0, varpi0=0.0,
          xi_span=(0.0, 5.0), r_outer_ah=None,
          chandrasekhar='integral', data_dir=None, **solve_ivp_kwargs):
    r"""Integrate the binary-evolution ODEs.

    Parameters
    ----------
    q : float
        Mass ratio :math:`M_2/M_1 \le 1`.
    e0 : float
        Initial eccentricity.
    Vx0_s, Vy0_s : float
        Initial centre-of-mass velocity / sigma (lab frame).
    varpi0 : float
        Initial longitude of periapsis [rad].
    xi_span : (float, float)
        Integration interval in :math:`\xi = \ln(a_h/a)`.
    r_outer_ah : float or None
        Chandrasekhar outer cutoff in units of :math:`a_h`.
        Default ``4(1+q)^2 / q`` (≈ influence radius).
    chandrasekhar : str
        ``'integral'`` (full v-dependent ln Lambda, default),
        ``'constant'`` (standard erf formula), or ``'none'``.
    data_dir : str or None
        Path to harmonics data directory.
    **solve_ivp_kwargs
        Forwarded to :func:`scipy.integrate.solve_ivp`
        (e.g. ``method``, ``max_step``, ``rtol``, ``atol``).

    Returns
    -------
    sol : OdeResult
        ``.t`` = xi, ``.y`` = ``(e, Vx/σ, Vy/σ, ϖ, t/T_hard, x̃, ỹ)``.
        Positions x̃, ỹ are in units of σ · T_hard.
    """
    if r_outer_ah is None:
        r_outer_ah = 4.0 * (1.0 + q)**2 / q

    data, e_grid = load_harmonics_data(q, data_dir)
    e_max = e_grid[-1] - 0.01

    ch_funcs = {
        'integral': chandrasekhar_decel_integral,
        'constant': chandrasekhar_decel_constant,
        'none':     None,
    }
    if chandrasekhar not in ch_funcs:
        raise ValueError(f"Unknown chandrasekhar mode: {chandrasekhar!r}")
    ch_func = ch_funcs[chandrasekhar]

    _rhs_count = [0]
    _rhs_t0 = [None]

    def rhs(xi, y):
        if _rhs_t0[0] is None:
            _rhs_t0[0] = _time.time()
        _rhs_count[0] += 1
        if _rhs_count[0] % 20 == 0:
            elapsed = _time.time() - _rhs_t0[0]
            print(f"\r  xi={xi:.3f}  e={y[0]:.4f}  |V|/σ={np.hypot(y[1],y[2]):.5f}  "
                  f"[{_rhs_count[0]} evals, {elapsed:.1f}s]   ",
                  end='', flush=True)

        e_cur, Vx_s, Vy_s, varpi, _t, _x, _y = y
        e_cur = np.clip(e_cur, 1e-6, 1.0 - 1e-6)

        H, K, Px, Py, Q = compute_rates(
            xi, e_cur, Vx_s, Vy_s, varpi, q, data, e_grid)

        if H < 1e-6:
            H = 1e-6

        # Chandrasekhar deceleration
        Ch_x = Ch_y = 0.0
        V_mag = np.hypot(Vx_s, Vy_s)
        if ch_func is not None and V_mag > 1e-10:
            J = ch_func(V_mag, xi, q, r_outer_ah)
            prefactor = 16.0 * np.pi * (1.0 + q)**2 * np.exp(xi) / (q * H)
            Ch_x = prefactor * J * Vx_s / V_mag
            Ch_y = prefactor * J * Vy_s / V_mag

        dt_dxi = np.exp(xi) / H

        return [K,
                Px + Ch_x,
                Py + Ch_y,
                Q,
                dt_dxi,
                Vx_s * dt_dxi,
                Vy_s * dt_dxi]

    # Stop when eccentricity leaves the data grid
    def _e_boundary(xi, y):
        return e_max - y[0]
    _e_boundary.terminal = True
    _e_boundary.direction = -1

    y0 = [e0, Vx0_s, Vy0_s, varpi0, 0.0, 0.0, 0.0]

    defaults = dict(method='RK45', rtol=1e-8, atol=1e-10, dense_output=True,
                    events=_e_boundary)
    defaults.update(solve_ivp_kwargs)

    sol = solve_ivp(rhs, xi_span, y0, **defaults)
    elapsed = _time.time() - _rhs_t0[0] if _rhs_t0[0] else 0
    print(f"\r  Done: {_rhs_count[0]} evals in {elapsed:.1f}s" + " " * 30)
    return sol


def solve_simple(q, e0, xi_span=(0.0, 5.0), data_dir=None, **solve_ivp_kwargs):
    r"""Integrate the reduced (V=0) binary-evolution ODEs.

    Only eccentricity and time evolve; velocity and precession are ignored.
    Useful as a reference solution.

    Returns
    -------
    sol : OdeResult
        ``.t`` = xi, ``.y`` = ``(e, t/T_hard)``.
    """
    data, e_grid = load_harmonics_data(q, data_dir)
    e_max = e_grid[-1] - 0.01

    _rhs_count = [0]
    _rhs_t0 = [None]

    def rhs(xi, y):
        if _rhs_t0[0] is None:
            _rhs_t0[0] = _time.time()
        _rhs_count[0] += 1
        if _rhs_count[0] % 20 == 0:
            elapsed = _time.time() - _rhs_t0[0]
            print(f"\r  (V=0 ref) xi={xi:.3f}  e={y[0]:.4f}  "
                  f"[{_rhs_count[0]} evals, {elapsed:.1f}s]   ",
                  end='', flush=True)

        e_cur, _t = y
        e_cur = np.clip(e_cur, 1e-6, 1.0 - 1e-6)
        H, K, _, _, _ = compute_rates(xi, e_cur, 0.0, 0.0, 0.0,
                                      q, data, e_grid)
        if H < 1e-6:
            H = 1e-6
        return [K, np.exp(xi) / H]

    def _e_boundary(xi, y):
        return e_max - y[0]
    _e_boundary.terminal = True
    _e_boundary.direction = -1

    y0 = [e0, 0.0]

    defaults = dict(method='RK45', rtol=1e-8, atol=1e-10, dense_output=True,
                    events=_e_boundary)
    defaults.update(solve_ivp_kwargs)

    sol = solve_ivp(rhs, xi_span, y0, **defaults)
    elapsed = _time.time() - _rhs_t0[0] if _rhs_t0[0] else 0
    print(f"\r  (V=0 ref) Done: {_rhs_count[0]} evals in {elapsed:.1f}s" + " " * 20)
    return sol


# ── CLI entry point ───────────────────────────────────────────────────────────

def _status(sol):
    if sol.t_events is not None and any(len(te) > 0 for te in sol.t_events):
        return "terminated (e → grid boundary)"
    return "completed" if sol.success else f"FAILED: {sol.message}"


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Binary evolution solver')
    parser.add_argument('--q', type=float, default=1.0,
                        help='Mass ratio (default: 1.0)')
    parser.add_argument('--e0', type=float, default=0.5,
                        help='Initial eccentricity (default: 0.5)')
    parser.add_argument('--Vx0', type=float, default=0.0,
                        help='Initial Vx/sigma (default: 0)')
    parser.add_argument('--Vy0', type=float, default=0.0,
                        help='Initial Vy/sigma (default: 0)')
    parser.add_argument('--varpi0', type=float, default=0.0,
                        help='Initial varpi in rad (default: 0)')
    parser.add_argument('--xi-start', type=float, default=None,
                        help='Initial xi = ln(a_h/a).  Alternatively use --a0.')
    parser.add_argument('--a0', type=float, default=None,
                        help='Initial a/a_h (converted to xi-start)')
    parser.add_argument('--xi-end', type=float, default=5.0,
                        help='Final xi (default: 5)')
    parser.add_argument('--chandrasekhar',
                        choices=['integral', 'constant', 'none'],
                        default='integral',
                        help='Chandrasekhar mode (default: integral)')
    args = parser.parse_args()

    if args.a0 is not None and args.xi_start is not None:
        parser.error("Use --a0 or --xi-start, not both")
    if args.a0 is not None:
        xi_start = -np.log(args.a0)
    elif args.xi_start is not None:
        xi_start = args.xi_start
    else:
        xi_start = 0.0

    print(f"Binary evolution: q={args.q}, e0={args.e0}, "
          f"V0/sigma=({args.Vx0}, {args.Vy0}), varpi0={args.varpi0}")
    print(f"xi in [{xi_start:.4f}, {args.xi_end}]  "
          f"(a/a_h: {np.exp(-xi_start):.4f} → {np.exp(-args.xi_end):.6f}), "
          f"Chandrasekhar: {args.chandrasekhar}")

    xi_span = (xi_start, args.xi_end)

    # ── Full solution ─────────────────────────────────────────────────────
    print("Integrating (full)...")
    sol = solve(args.q, args.e0, args.Vx0, args.Vy0, args.varpi0,
                xi_span=xi_span, chandrasekhar=args.chandrasekhar)

    if not sol.success and _status(sol).startswith("FAILED"):
        print(f"\nIntegration {_status(sol)}")
        sys.exit(1)

    xi = sol.t
    e, Vx, Vy, varpi, t, x_pos, y_pos = sol.y
    a_ah = np.exp(-xi)
    V = np.hypot(Vx, Vy)

    print(f"\nFull solution — {len(xi)} steps, {_status(sol)}")
    print(f"  xi:        {xi[0]:.2f} → {xi[-1]:.2f}  "
          f"(a/a_h: {a_ah[0]:.3f} → {a_ah[-1]:.5f})")
    print(f"  e:         {e[0]:.4f} → {e[-1]:.4f}")
    print(f"  |V|/sigma: {V[0]:.4f} → {V[-1]:.4f}")
    print(f"  varpi:     {varpi[0]:.4f} → {varpi[-1]:.4f} rad")
    print(f"  t/T_hard:  {t[0]:.4f} → {t[-1]:.4f}")

    # ── V=0 reference solution ────────────────────────────────────────────
    print("Integrating (V=0 reference)...")
    sol0 = solve_simple(args.q, args.e0, xi_span=xi_span)

    xi0 = sol0.t
    e0_ref, t0_ref = sol0.y
    a_ah0 = np.exp(-xi0)

    print(f"V=0 reference — {len(xi0)} steps, {_status(sol0)}")

    # ── Plots ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 11))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

        # Row 0: e(xi), a/a_h(xi), CoM trajectory (x, y)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(xi, e)
        ax.set(xlabel=r'$\xi = \ln(a_h/a)$', ylabel='$e$')

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(xi, a_ah)
        ax.set(xlabel=r'$\xi$', ylabel='$a/a_h$', yscale='log')

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(x_pos, y_pos)
        ax.plot(x_pos[0], y_pos[0], 'o', ms=5, label='start')
        ax.plot(x_pos[-1], y_pos[-1], 's', ms=5, label='end')
        ax.set(xlabel=r'$x\;/\;(\sigma\, T_{\rm hard})$',
               ylabel=r'$y\;/\;(\sigma\, T_{\rm hard})$')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize='small')

        # Row 1: Vx,Vy(xi), varpi(xi)
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(xi, Vx, label=r'$V_x/\sigma$')
        ax.plot(xi, Vy, label=r'$V_y/\sigma$')
        ax.set(xlabel=r'$\xi$', ylabel=r'$V/\sigma$')
        ax.legend(fontsize='small')

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(xi, varpi)
        ax.set(xlabel=r'$\xi$', ylabel=r'$\varpi$ [rad]')

        # Row 2: e(t) and a/a_h(t) with V=0 dashed reference
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(t, e, label='full')
        ax.plot(t0_ref, e0_ref, '--', label=r'$V{=}0$ ref')
        ax.set(xlabel=r'$t\;/\;T_{\rm hard}$', ylabel='$e$')
        ax.legend(fontsize='small')

        ax = fig.add_subplot(gs[2, 1])
        ax.plot(t, a_ah, label='full')
        ax.plot(t0_ref, a_ah0, '--', label=r'$V{=}0$ ref')
        ax.set(xlabel=r'$t\;/\;T_{\rm hard}$', ylabel='$a/a_h$',
               yscale='log')
        ax.legend(fontsize='small')

        fig.suptitle(f'Binary evolution: $q={args.q}$, $e_0={args.e0}$, '
                     f'Chandrasekhar = {args.chandrasekhar}')
        out = os.path.join(_HERE, 'evolution.pdf')
        fig.savefig(out, bbox_inches='tight')
        print(f"\nPlot saved to {out}")
        plt.show()
    except ImportError:
        pass
