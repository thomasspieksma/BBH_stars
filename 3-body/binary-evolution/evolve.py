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


_N_STATE = 7
_N_RATES = 5   # H, K, Pe, Pn, Q


def _pack_upper_tri(C):
    """Symmetric 7x7 matrix -> upper-triangle flat array (28 elements)."""
    n = C.shape[0]
    flat = np.empty(n * (n + 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i, n):
            flat[k] = C[i, j]
            k += 1
    return flat


# ── Rate evaluation (with eccentricity interpolation) ─────────────────────────

def _lagrange_weights(e, stencil_e):
    """Lagrange interpolation weights and their derivatives w.r.t. *e*.

    Returns ``(w, dw)`` where ``w[k]`` is the Lagrange basis value at *e*
    and ``dw[k]`` is its derivative, so that for stencil values ``f_k``,
    ``sum(w * f) ≈ f(e)`` and ``sum(dw * f) ≈ df/de(e)``.
    """
    n = len(stencil_e)
    w = np.ones(n)
    dw = np.zeros(n)
    for k in range(n):
        for j in range(n):
            if j != k:
                w[k] *= (e - stencil_e[j]) / (stencil_e[k] - stencil_e[j])
    for k in range(n):
        for m in range(n):
            if m != k:
                term = 1.0 / (stencil_e[k] - stencil_e[m])
                for j in range(n):
                    if j != k and j != m:
                        term *= (e - stencil_e[j]) / (stencil_e[k] - stencil_e[j])
                dw[k] += term
    return w, dw


def compute_rates(xi, e, Vx_s, Vy_s, varpi, q, data, e_grid, n_stencil=4,
                  return_e_derivs=False, return_file_info=False):
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
    n_stencil : int
        Number of eccentricity grid points in the Lagrange interpolation
        stencil (2 = linear, 4 = cubic).
    return_e_derivs : bool
        If True, also return analytic derivatives of all five rates
        w.r.t. eccentricity from the Lagrange interpolant.
    return_file_info : bool
        If True, also return a dict with per-file stencil information
        (indices into the e>0 grid, Lagrange weights, and per-file
        binary-frame uncertainties).

    Returns
    -------
    H, K, Px, Py, Q, sH, sK, sPx, sPy, sQ : float
        Hardening rate, eccentricity growth rate, lab-frame acceleration
        components, precession parameter, and their 1-sigma uncertainties.
    dK_de, dH_de, dPx_de, dPy_de, dQ_de : float  (only when *return_e_derivs* is True)
        Derivatives of all five rates w.r.t. eccentricity (P in lab frame).
    file_info : dict  (only when *return_file_info* is True)
        ``stencil_indices`` (indices into eg), ``weights`` (Lagrange
        weights), ``file_sigmas`` (n_stencil x 5 array of per-file
        binary-frame uncertainties: sH, sK, sPe, sPn, sQ).
    """
    cos_w, sin_w = np.cos(varpi), np.sin(varpi)

    # Project velocity to binary frame
    Ve_s = Vx_s * cos_w + Vy_s * sin_w
    Vn_s = -Vx_s * sin_w + Vy_s * cos_w

    # Code-unit velocity dispersion: sigma / sqrt(GM/a)
    #   a_h = G mu / (4 sigma^2)  =>  GM/sigma^2 = 4(1+q)^2 a_h / q
    #   sigma_code = sigma / sqrt(GM/a) = sqrt(q / (4(1+q)^2)) * sqrt(a/a_h)
    sigma_code = np.sqrt(q) / (2.0 * (1.0 + q)) * np.exp(-xi / 2.0)

    V_code = np.array([Ve_s * sigma_code,
                       Vn_s * sigma_code,
                       0.0])

    def _eval_at_e(e_val):
        meta, harm_bins = data[e_val]
        res = reweight_from_harmonics(meta, harm_bins, V_code, sigma_code)
        H   = res['H']
        K   = res['K']   if np.isfinite(res['K'])   else 0.0
        Pe  = res['P_x'] if np.isfinite(res['P_x']) else 0.0
        Pn  = res['P_y'] if np.isfinite(res['P_y']) else 0.0
        Q   = res['Q']   if np.isfinite(res['Q'])   else 0.0
        sH  = res['sH']   if np.isfinite(res['sH'])   else 0.0
        sK  = res['sK']   if np.isfinite(res['sK'])   else 0.0
        sPe = res['sP_x'] if np.isfinite(res['sP_x']) else 0.0
        sPn = res['sP_y'] if np.isfinite(res['sP_y']) else 0.0
        sQ  = res['sQ']   if np.isfinite(res['sQ'])   else 0.0
        return H, K, Pe, Pn, Q, sH, sK, sPe, sPn, sQ

    # Use only e > 0 grid points (K formula is singular at e=0)
    eg = e_grid[e_grid > 0] if e_grid[0] == 0.0 else e_grid

    dK_de = dH_de = dPe_de = dPn_de = dQ_de = 0.0
    _finfo = None

    # Fast path: snap / boundary when derivatives are not requested
    snap_idx = np.argmin(np.abs(eg - e))
    use_fast = not return_e_derivs and not return_file_info
    if use_fast and abs(eg[snap_idx] - e) < 1e-10:
        H, K, Pe, Pn, Q, sH, sK, sPe, sPn, sQ = _eval_at_e(eg[snap_idx])
        _finfo_idx = snap_idx
    elif use_fast and e <= eg[0]:
        H, K, Pe, Pn, Q, sH, sK, sPe, sPn, sQ = _eval_at_e(eg[0])
        _finfo_idx = 0
    elif use_fast and e >= eg[-1]:
        H, K, Pe, Pn, Q, sH, sK, sPe, sPn, sQ = _eval_at_e(eg[-1])
        _finfo_idx = len(eg) - 1
    elif len(eg) < 2:
        H, K, Pe, Pn, Q, sH, sK, sPe, sPn, sQ = _eval_at_e(eg[0])
        _finfo_idx = 0
    else:
        _finfo_idx = None
        e_eval = np.clip(e, eg[0], eg[-1])
        n_stencil = min(n_stencil, len(eg))
        idx = int(np.searchsorted(eg, e_eval))
        i0 = max(0, min(idx - n_stencil // 2, len(eg) - n_stencil))
        stencil_e = eg[i0:i0 + n_stencil]
        vals = np.array([_eval_at_e(ei) for ei in stencil_e])

        w, dw = _lagrange_weights(e_eval, stencil_e)

        H, K, Pe, Pn, Q = w @ vals[:, :5]
        sH, sK, sPe, sPn, sQ = np.sqrt((w**2) @ (vals[:, 5:]**2))

        if return_e_derivs:
            dH_de  = dw @ vals[:, 0]
            dK_de  = dw @ vals[:, 1]
            dPe_de = dw @ vals[:, 2]
            dPn_de = dw @ vals[:, 3]
            dQ_de  = dw @ vals[:, 4]

        if return_file_info:
            _finfo = {
                'stencil_indices': np.arange(i0, i0 + n_stencil),
                'weights': w,
                'file_sigmas': vals[:, 5:],   # (n_stencil, 5): sH,sK,sPe,sPn,sQ
            }

    # Single-file path: build trivial file_info
    if return_file_info and _finfo is None:
        _finfo = {
            'stencil_indices': np.array([_finfo_idx]),
            'weights': np.array([1.0]),
            'file_sigmas': np.array([[sH, sK, sPe, sPn, sQ]]),
        }

    # Rotate P (and its uncertainty) from binary frame to lab frame
    Px = Pe * cos_w - Pn * sin_w
    Py = Pe * sin_w + Pn * cos_w
    sPx = np.sqrt(cos_w**2 * sPe**2 + sin_w**2 * sPn**2)
    sPy = np.sqrt(sin_w**2 * sPe**2 + cos_w**2 * sPn**2)

    result = (H, K, Px, Py, Q, sH, sK, sPx, sPy, sQ)
    if return_e_derivs:
        dPx_de = dPe_de * cos_w - dPn_de * sin_w
        dPy_de = dPe_de * sin_w + dPn_de * cos_w
        result = result + (dK_de, dH_de, dPx_de, dPy_de, dQ_de)
    if return_file_info:
        result = result + (_finfo,)
    return result


# ── Precomputed rate tables ───────────────────────────────────────────────────

def _sigma_code(xi, q):
    return np.sqrt(q) / (2.0 * (1.0 + q)) * np.exp(-xi / 2.0)


def precompute_rate_tables(q, data, e_grid, xi_span,
                           n_xi=50, V_max=1.5, n_Ve=13, n_Vn=13):
    r"""Precompute H, K, Pe, Pn, Q on a 4D ``(xi, e, Ve/σ, Vn/σ)`` grid.

    Rates are evaluated exactly at each grid point via
    ``reweight_from_harmonics`` — no velocity linearization.
    The resulting tables use cubic interpolation in all four dimensions.
    """
    eg = e_grid[e_grid > 0] if e_grid[0] == 0.0 else e_grid.copy()
    margin = 0.5
    xi_lo = max(xi_span[0] - margin, 0.0)
    xi_hi = xi_span[1] + margin
    xi_arr = np.linspace(xi_lo, xi_hi, n_xi)
    Ve_arr = np.linspace(-V_max, V_max, n_Ve)
    Vn_arr = np.linspace(-V_max, V_max, n_Vn)
    n_e = len(eg)

    shape = (n_xi, n_e, n_Ve, n_Vn)
    H_tab  = np.zeros(shape)
    K_tab  = np.zeros(shape)
    Pe_tab = np.zeros(shape)
    Pn_tab = np.zeros(shape)
    Q_tab  = np.zeros(shape)

    total = n_xi * n_e * n_Ve * n_Vn
    t0 = _time.time()
    done = 0

    for i, xi in enumerate(xi_arr):
        sc = _sigma_code(xi, q)
        for j, e_val in enumerate(eg):
            meta, hb = data[e_val]
            for k, Ve_s in enumerate(Ve_arr):
                for l, Vn_s in enumerate(Vn_arr):
                    V_code = np.array([Ve_s * sc, Vn_s * sc, 0.0])
                    res = reweight_from_harmonics(meta, hb, V_code, sc)
                    H_tab[i, j, k, l]  = res['H']
                    K_tab[i, j, k, l]  = res['K'] if np.isfinite(res['K']) else 0.0
                    Pe_tab[i, j, k, l] = res['P_x'] if np.isfinite(res['P_x']) else 0.0
                    Pn_tab[i, j, k, l] = res['P_y'] if np.isfinite(res['P_y']) else 0.0
                    Q_tab[i, j, k, l]  = res['Q'] if np.isfinite(res['Q']) else 0.0
                    done += 1

        elapsed = _time.time() - t0
        eta = elapsed / (i + 1) * (n_xi - i - 1)
        print(f"\r  Precomputing rates: {done}/{total} "
              f"({100*done/total:.0f}%), ETA {eta:.0f}s   ",
              end='', flush=True)

    print(f"\r  Precomputed rates in {_time.time()-t0:.1f}s"
          + " " * 30)

    axes = (xi_arr, eg, Ve_arr, Vn_arr)

    def _interp(tab):
        return RegularGridInterpolator(
            axes, tab, method='cubic',
            bounds_error=False, fill_value=None)

    return {
        'xi_grid': xi_arr, 'e_grid': eg, 'q': q,
        'Ve_grid': Ve_arr, 'Vn_grid': Vn_arr,
        'H': _interp(H_tab), 'K': _interp(K_tab), 'Q': _interp(Q_tab),
        'Pe': _interp(Pe_tab), 'Pn': _interp(Pn_tab),
    }


def compute_rates_fast(xi, e, Vx_s, Vy_s, varpi, tables):
    """Fast rate evaluation via precomputed 4D interpolation tables.

    Uncertainties are not available from precomputed tables and are
    returned as zero.
    """
    cos_w, sin_w = np.cos(varpi), np.sin(varpi)

    Ve_s =  Vx_s * cos_w + Vy_s * sin_w
    Vn_s = -Vx_s * sin_w + Vy_s * cos_w

    xi_c = np.clip(xi,   tables['xi_grid'][0],  tables['xi_grid'][-1])
    e_c  = np.clip(e,    tables['e_grid'][0],   tables['e_grid'][-1])
    Ve_c = np.clip(Ve_s, tables['Ve_grid'][0],  tables['Ve_grid'][-1])
    Vn_c = np.clip(Vn_s, tables['Vn_grid'][0],  tables['Vn_grid'][-1])
    pt = np.array([[xi_c, e_c, Ve_c, Vn_c]])

    H  = tables['H'](pt).item()
    K  = tables['K'](pt).item()
    Pe = tables['Pe'](pt).item()
    Pn = tables['Pn'](pt).item()
    Q  = tables['Q'](pt).item()

    Px = Pe * cos_w - Pn * sin_w
    Py = Pe * sin_w + Pn * cos_w

    return H, K, Px, Py, Q, 0.0, 0.0, 0.0, 0.0, 0.0


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
          chandrasekhar='integral', data_dir=None,
          precompute=False, V_max=1.5, n_Ve=13, n_Vn=13, n_xi=50,
          e_interp='cubic', freeze=(), jacobian=True,
          full_covariance=True,
          **solve_ivp_kwargs):
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
    precompute : bool
        If True, precompute rates on a 4D ``(xi, e, Ve/σ, Vn/σ)``
        grid with cubic interpolation (fast path).  If False (default),
        evaluate rates on the fly with 4-point Lagrange interpolation
        in eccentricity (slow path).
    V_max : float
        Half-width of the velocity grid in units of σ (precompute only).
    n_Ve, n_Vn : int
        Number of velocity grid points along each axis (precompute only).
    n_xi : int
        Number of xi grid points (precompute only).
    jacobian : bool
        If True (default), include Jacobian feedback in the uncertainty
        ODEs.  If False, revert to the forcing-only model (no feedback).
        Ignored when *precompute* is True.
    full_covariance : bool
        If True (default), propagate uncertainties via the per-file
        response-matrix model dF/dξ = J F + G, where each data file
        contributes independent systematic noise sources.  The state
        vector has 7 + 7 × 5 × N_files elements.  If False, use the
        diagonal approximation (14-component state vector).
        Requires *jacobian* = True and *precompute* = False.
    **solve_ivp_kwargs
        Forwarded to :func:`scipy.integrate.solve_ivp`
        (e.g. ``method``, ``max_step``, ``rtol``, ``atol``).

    Returns
    -------
    sol : OdeResult
        ``.t`` = xi, ``.y`` always has 14 rows:
        ``(e, Vx/σ, Vy/σ, ϖ, t/T_ref, x̃, ỹ,
          σ_e, σ_Vx, σ_Vy, σ_ϖ, σ_t, σ_x, σ_y)``.
        When *full_covariance* is True, σ values are extracted as
        ``sqrt(diag(C))`` where ``C = F @ F.T``.  The upper-triangle
        covariance history is in ``sol.cov`` (shape ``(28, n_steps)``)
        and the raw response matrix in ``sol.F``.
    """
    if r_outer_ah is None:
        r_outer_ah = 4.0 * (1.0 + q)**2 / q

    _freeze = frozenset(freeze)
    _valid_freeze = {'e', 'Vx', 'Vy', 'varpi'}
    if _freeze - _valid_freeze:
        raise ValueError(f"Unknown freeze parameters: {_freeze - _valid_freeze}")
    if _freeze:
        print(f"  Frozen: {', '.join(sorted(_freeze))}")

    _use_jacobian = bool(jacobian)
    _full_cov = bool(full_covariance)
    if _full_cov:
        if precompute:
            raise ValueError("full_covariance requires on-the-fly evaluation "
                             "(precompute must be False)")
        _use_jacobian = True

    data, e_grid = load_harmonics_data(q, data_dir)

    _n_stencil = 4
    if e_interp == 'linear':
        coarse = e_grid[np.abs(np.round(e_grid, 1) - e_grid) < 1e-9]
        e_grid = coarse
        _n_stencil = 2
        print(f"  Coarse grid: {len(e_grid)} points "
              f"(e = {e_grid[0]:.1f} … {e_grid[-1]:.1f}), linear interp")

    e_max = e_grid[-1] - 0.01

    # Number of per-file noise sources for the response-matrix model
    _eg_pos = e_grid[e_grid > 0] if e_grid[0] == 0.0 else e_grid
    _n_noise = _N_RATES * len(_eg_pos) if _full_cov else 0
    if _full_cov:
        print(f"  Response matrix: {len(_eg_pos)} files × "
              f"{_N_RATES} rates = {_n_noise} noise sources "
              f"({_N_STATE + _N_STATE * _n_noise} total state)")

    tables = None
    if precompute:
        tables = precompute_rate_tables(
            q, data, e_grid, xi_span,
            n_xi=n_xi, V_max=V_max, n_Ve=n_Ve, n_Vn=n_Vn)

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

    _DELTA_V = 0.01  # finite-difference step for velocity Jacobian (in sigma)
    _DELTA_W = 0.01  # finite-difference step for varpi Jacobian (in rad)

    def _compute_ch(Vx_p, Vy_p, H_p):
        """Chandrasekhar deceleration at an arbitrary (Vx, Vy, H) state."""
        Vm = np.hypot(Vx_p, Vy_p)
        if ch_func is None or Vm < 1e-10:
            return 0.0, 0.0
        Jval = ch_func(Vm, xi_cur[0], q, r_outer_ah)
        pf = _ch_base[0] / H_p
        return pf * Jval * Vx_p / Vm, pf * Jval * Vy_p / Vm

    # Mutable containers shared between rhs and _compute_ch
    xi_cur = [0.0]
    _ch_base = [0.0]

    def rhs(xi, y):
        if _rhs_t0[0] is None:
            _rhs_t0[0] = _time.time()
        _rhs_count[0] += 1
        if _rhs_count[0] % 20 == 0:
            elapsed = _time.time() - _rhs_t0[0]
            print(f"\r  xi={xi:.3f}  e={y[0]:.4f}  |V|/σ={np.hypot(y[1],y[2]):.5f}  "
                  f"[{_rhs_count[0]} evals, {elapsed:.1f}s]   ",
                  end='', flush=True)

        e_cur, Vx_s, Vy_s, varpi = y[0], y[1], y[2], y[3]
        e_cur = np.clip(e_cur, 1e-6, 1.0 - 1e-6)

        # Shared state for _compute_ch helper
        xi_cur[0] = xi
        _ch_base[0] = 16.0 * np.pi * (1.0 + q)**2 * np.exp(xi) / q

        need_jac = _use_jacobian and tables is None

        # ── Nominal rates ─────────────────────────────────────────────
        _file_info = None
        if tables is not None:
            H, K, Px, Py, Q, sH, sK, sPx, sPy, sQ = compute_rates_fast(
                xi, e_cur, Vx_s, Vy_s, varpi, tables)
            dK_de = dH_de_val = dPx_de_val = dPy_de_val = dQ_de_val = 0.0
        elif _full_cov:
            (H, K, Px, Py, Q, sH, sK, sPx, sPy, sQ,
             dK_de, dH_de_val, dPx_de_val, dPy_de_val,
             dQ_de_val, _file_info) = compute_rates(
                xi, e_cur, Vx_s, Vy_s, varpi, q, data, e_grid,
                n_stencil=_n_stencil, return_e_derivs=True,
                return_file_info=True)
        elif need_jac:
            (H, K, Px, Py, Q, sH, sK, sPx, sPy, sQ,
             dK_de, dH_de_val, dPx_de_val, dPy_de_val,
             dQ_de_val) = compute_rates(
                xi, e_cur, Vx_s, Vy_s, varpi, q, data, e_grid,
                n_stencil=_n_stencil, return_e_derivs=True)
        else:
            H, K, Px, Py, Q, sH, sK, sPx, sPy, sQ = compute_rates(
                xi, e_cur, Vx_s, Vy_s, varpi, q, data, e_grid,
                n_stencil=_n_stencil)
            dK_de = dH_de_val = dPx_de_val = dPy_de_val = dQ_de_val = 0.0

        if H < 1e-6:
            H = 1e-6

        Ch_x, Ch_y = _compute_ch(Vx_s, Vy_s, H)

        exi = np.exp(xi)
        exi_H = exi / H
        exi_H2 = exi / H**2

        # ── State derivatives ─────────────────────────────────────────
        de  = 0.0 if 'e'     in _freeze else K
        dVx = 0.0 if 'Vx'    in _freeze else Px + Ch_x
        dVy = 0.0 if 'Vy'    in _freeze else Py + Ch_y
        dw  = 0.0 if 'varpi' in _freeze else Q
        dt_dxi = exi_H
        dy_state = [de, dVx, dVy, dw, dt_dxi,
                    Vx_s * dt_dxi, Vy_s * dt_dxi]

        # ── Full covariance mode ──────────────────────────────────────
        if _full_cov:
            # Build 7x7 Jacobian
            J = np.zeros((_N_STATE, _N_STATE))

            # Column 0: eccentricity derivatives (analytic)
            ch_over_H = np.array([Ch_x, Ch_y]) / H if H > 1e-10 else np.zeros(2)
            J[0, 0] = dK_de
            J[1, 0] = dPx_de_val - ch_over_H[0] * dH_de_val
            J[2, 0] = dPy_de_val - ch_over_H[1] * dH_de_val
            J[3, 0] = dQ_de_val
            J[4, 0] = -exi_H2 * dH_de_val
            J[5, 0] = Vx_s * J[4, 0]
            J[6, 0] = Vy_s * J[4, 0]

            # Columns 1-2: velocity derivatives (finite differences)
            dv = _DELTA_V
            for col, (dVx_off, dVy_off) in enumerate([(dv, 0.0), (0.0, dv)],
                                                       start=1):
                if ('Vx' in _freeze and col == 1) or \
                   ('Vy' in _freeze and col == 2):
                    continue
                Hp, Kp, Pxp, Pyp, Qp, *_ = compute_rates(
                    xi, e_cur, Vx_s + dVx_off, Vy_s + dVy_off, varpi,
                    q, data, e_grid, n_stencil=_n_stencil)
                if Hp < 1e-6: Hp = 1e-6
                Chxp, Chyp = _compute_ch(Vx_s + dVx_off, Vy_s + dVy_off, Hp)

                Hm, Km, Pxm, Pym, Qm, *_ = compute_rates(
                    xi, e_cur, Vx_s - dVx_off, Vy_s - dVy_off, varpi,
                    q, data, e_grid, n_stencil=_n_stencil)
                if Hm < 1e-6: Hm = 1e-6
                Chxm, Chym = _compute_ch(Vx_s - dVx_off, Vy_s - dVy_off, Hm)

                inv2d = 1.0 / (2.0 * dv)
                J[0, col] = (Kp - Km) * inv2d
                J[1, col] = ((Pxp + Chxp) - (Pxm + Chxm)) * inv2d
                J[2, col] = ((Pyp + Chyp) - (Pym + Chym)) * inv2d
                J[3, col] = (Qp - Qm) * inv2d
                dt_p = exi / max(Hp, 1e-6)
                dt_m = exi / max(Hm, 1e-6)
                J[4, col] = (dt_p - dt_m) * inv2d

            # dx/dxi = Vx * dt/dxi  =>  d(dx/dxi)/dVx = dt/dxi + Vx * d(dt/dxi)/dVx
            J[5, 1] = dt_dxi + Vx_s * J[4, 1]
            J[5, 2] = Vx_s * J[4, 2]
            J[6, 1] = Vy_s * J[4, 1]
            J[6, 2] = dt_dxi + Vy_s * J[4, 2]

            # Column 3: varpi derivatives (finite differences)
            if 'varpi' not in _freeze:
                dw_fd = _DELTA_W
                Hp, Kp, Pxp, Pyp, Qp, *_ = compute_rates(
                    xi, e_cur, Vx_s, Vy_s, varpi + dw_fd,
                    q, data, e_grid, n_stencil=_n_stencil)
                if Hp < 1e-6: Hp = 1e-6
                Chxp, Chyp = _compute_ch(Vx_s, Vy_s, Hp)

                Hm, Km, Pxm, Pym, Qm, *_ = compute_rates(
                    xi, e_cur, Vx_s, Vy_s, varpi - dw_fd,
                    q, data, e_grid, n_stencil=_n_stencil)
                if Hm < 1e-6: Hm = 1e-6
                Chxm, Chym = _compute_ch(Vx_s, Vy_s, Hm)

                inv2d = 1.0 / (2.0 * dw_fd)
                J[0, 3] = (Kp - Km) * inv2d
                J[1, 3] = ((Pxp + Chxp) - (Pxm + Chxm)) * inv2d
                J[2, 3] = ((Pyp + Chyp) - (Pym + Chym)) * inv2d
                J[3, 3] = (Qp - Qm) * inv2d
                dt_p = exi / max(Hp, 1e-6)
                dt_m = exi / max(Hm, 1e-6)
                J[4, 3] = (dt_p - dt_m) * inv2d
                J[5, 3] = Vx_s * J[4, 3]
                J[6, 3] = Vy_s * J[4, 3]

            # Binary-frame loading matrix B (7x5): maps unit perturbation
            # in (H, K, Pe, Pn, Q) to state derivatives
            cos_w = np.cos(varpi)
            sin_w = np.sin(varpi)
            B = np.zeros((_N_STATE, _N_RATES))
            ch_over_H_arr = (np.array([Ch_x, Ch_y]) / H
                             if H > 1e-10 else np.zeros(2))
            B[0, 1] = 1.0                                            # e  <- K
            B[1, 0] = -ch_over_H_arr[0]                              # Vx <- H
            B[1, 2] = cos_w;  B[1, 3] = -sin_w                      # Vx <- Pe,Pn
            B[2, 0] = -ch_over_H_arr[1]                              # Vy <- H
            B[2, 2] = sin_w;  B[2, 3] = cos_w                       # Vy <- Pe,Pn
            B[3, 4] = 1.0                                            # w  <- Q
            B[4, 0] = -exi_H2                                        # t  <- H
            B[5, 0] = -Vx_s * exi_H2                                 # x  <- H
            B[6, 0] = -Vy_s * exi_H2                                 # y  <- H

            # Per-file loading G (7 x n_noise): only stencil files contribute
            G = np.zeros((_N_STATE, _n_noise))
            st_idx = _file_info['stencil_indices']
            st_w   = _file_info['weights']
            st_sig = _file_info['file_sigmas']    # (n_stencil, 5)
            for s in range(len(st_idx)):
                base = _N_RATES * st_idx[s]
                for r in range(_N_RATES):
                    G[:, base + r] = st_w[s] * st_sig[s, r] * B[:, r]

            # Response-matrix equation: dF/dxi = J @ F + G
            F = y[_N_STATE:].reshape(_N_STATE, _n_noise)
            dF = J @ F + G
            return list(dy_state) + dF.ravel().tolist()

        # ── Diagonal mode ─────────────────────────────────────────────
        sig_e, sig_Vx, sig_Vy = y[7], y[8], y[9]

        J_e = dK_de if need_jac else 0.0
        J_Vx = J_Vy = 0.0

        if need_jac:
            dv = _DELTA_V
            if 'Vx' not in _freeze:
                Hp, _, Pxp, _, _, *_ = compute_rates(
                    xi, e_cur, Vx_s + dv, Vy_s, varpi, q, data, e_grid,
                    n_stencil=_n_stencil)
                if Hp < 1e-6: Hp = 1e-6
                Chxp, _ = _compute_ch(Vx_s + dv, Vy_s, Hp)

                Hm, _, Pxm, _, _, *_ = compute_rates(
                    xi, e_cur, Vx_s - dv, Vy_s, varpi, q, data, e_grid,
                    n_stencil=_n_stencil)
                if Hm < 1e-6: Hm = 1e-6
                Chxm, _ = _compute_ch(Vx_s - dv, Vy_s, Hm)

                J_Vx = ((Pxp + Chxp) - (Pxm + Chxm)) / (2.0 * dv)

            if 'Vy' not in _freeze:
                Hp, _, _, Pyp, _, *_ = compute_rates(
                    xi, e_cur, Vx_s, Vy_s + dv, varpi, q, data, e_grid,
                    n_stencil=_n_stencil)
                if Hp < 1e-6: Hp = 1e-6
                _, Chyp = _compute_ch(Vx_s, Vy_s + dv, Hp)

                Hm, _, _, Pym, _, *_ = compute_rates(
                    xi, e_cur, Vx_s, Vy_s - dv, varpi, q, data, e_grid,
                    n_stencil=_n_stencil)
                if Hm < 1e-6: Hm = 1e-6
                _, Chym = _compute_ch(Vx_s, Vy_s - dv, Hm)

                J_Vy = ((Pyp + Chyp) - (Pym + Chym)) / (2.0 * dv)

        sig_t_rate = exi * sH / H**2
        sigma_fVx = np.sqrt(sPx**2 + (Ch_x * sH / H)**2)
        sigma_fVy = np.sqrt(sPy**2 + (Ch_y * sH / H)**2)

        d_sig_e  = 0.0 if 'e'     in _freeze else J_e * sig_e + sK
        d_sig_Vx = 0.0 if 'Vx'    in _freeze else J_Vx * sig_Vx + sigma_fVx
        d_sig_Vy = 0.0 if 'Vy'    in _freeze else J_Vy * sig_Vy + sigma_fVy
        d_sig_w  = 0.0 if 'varpi' in _freeze else sQ

        return list(dy_state) + [d_sig_e, d_sig_Vx, d_sig_Vy, d_sig_w,
                sig_t_rate,
                sig_Vx * dt_dxi + abs(Vx_s) * sig_t_rate,
                sig_Vy * dt_dxi + abs(Vy_s) * sig_t_rate]

    # Stop when eccentricity leaves the data grid
    def _e_boundary(xi, y):
        return e_max - y[0]
    _e_boundary.terminal = True
    _e_boundary.direction = -1

    n_F = _N_STATE * _n_noise
    if _full_cov:
        y0 = [e0, Vx0_s, Vy0_s, varpi0, 0.0, 0.0, 0.0] + [0.0] * n_F
    else:
        y0 = [e0, Vx0_s, Vy0_s, varpi0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    defaults = dict(method='RK45', rtol=1e-8, atol=1e-10, dense_output=True,
                    events=_e_boundary)
    defaults.update(solve_ivp_kwargs)

    sol = solve_ivp(rhs, xi_span, y0, **defaults)
    elapsed = _time.time() - _rhs_t0[0] if _rhs_t0[0] else 0
    print(f"\r  Done: {_rhs_count[0]} evals in {elapsed:.1f}s" + " " * 30)

    if _full_cov:
        n_steps = sol.y.shape[1]
        F_raw = sol.y[_N_STATE:, :]        # (n_F, n_steps)
        sig = np.zeros((_N_STATE, n_steps))
        n_cov_tri = _N_STATE * (_N_STATE + 1) // 2
        cov_tri = np.zeros((n_cov_tri, n_steps))
        for k in range(n_steps):
            F = F_raw[:, k].reshape(_N_STATE, _n_noise)
            C = F @ F.T
            sig[:, k] = np.sqrt(np.maximum(np.diag(C), 0.0))
            cov_tri[:, k] = _pack_upper_tri(C)
        sol.y = np.vstack([sol.y[:_N_STATE, :], sig])
        sol.cov = cov_tri
        sol.F = F_raw

    return sol


def solve_simple(q, e0, xi_span=(0.0, 5.0), data_dir=None, e_interp='cubic',
                 jacobian=True, **solve_ivp_kwargs):
    r"""Integrate the reduced (V=0) binary-evolution ODEs.

    Only eccentricity and time evolve; velocity and precession are ignored.
    Useful as a reference solution.

    Uncertainty is propagated with the same per-file response-matrix model
    used by :func:`solve`: a ``2 × N_noise`` matrix *F* tracks how each
    independent file-rate noise source perturbs ``(e, t)``, and the
    covariance is ``C = F @ F^T``.

    Returns
    -------
    sol : OdeResult
        ``.t`` = xi, ``.y`` = ``(e, t/T_hard, σ_e, σ_t)``.
    """
    data, e_grid = load_harmonics_data(q, data_dir)

    _n_stencil = 4
    if e_interp == 'linear':
        coarse = e_grid[np.abs(np.round(e_grid, 1) - e_grid) < 1e-9]
        e_grid = coarse
        _n_stencil = 2

    _use_jacobian = bool(jacobian)

    e_max = e_grid[-1] - 0.01

    _N_S = 2  # reduced state: (e, t)
    eg = e_grid[e_grid > 0] if e_grid[0] == 0.0 else e_grid
    _n_noise = _N_RATES * len(eg)
    print(f"  (V=0 ref) Response matrix: {len(eg)} files × "
          f"{_N_RATES} rates = {_n_noise} noise sources "
          f"({_N_S + _N_S * _n_noise} total state)")

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

        e_cur = np.clip(y[0], 1e-6, 1.0 - 1e-6)

        (H, K, _, _, _, sH, sK, _, _, _,
         dK_de, dH_de, *_, _file_info) = compute_rates(
            xi, e_cur, 0.0, 0.0, 0.0, q, data, e_grid,
            n_stencil=_n_stencil, return_e_derivs=True,
            return_file_info=True)

        if H < 1e-6:
            H = 1e-6

        exi = np.exp(xi)
        exi_H2 = exi / H**2

        # Reduced 2x2 Jacobian for (e, t)
        J = np.zeros((_N_S, _N_S))
        if _use_jacobian:
            J[0, 0] = dK_de                  # de/dxi = K  => dK/de
            J[1, 0] = -exi_H2 * dH_de        # dt/dxi = e^xi/H  => d/de

        # Per-file loading G (2 x n_noise)
        G = np.zeros((_N_S, _n_noise))
        st_idx = _file_info['stencil_indices']
        st_w   = _file_info['weights']
        st_sig = _file_info['file_sigmas']    # (n_stencil, 5)
        for s in range(len(st_idx)):
            base = _N_RATES * st_idx[s]
            G[0, base + 1] = st_w[s] * st_sig[s, 1]              # e <- sK
            G[1, base + 0] = st_w[s] * st_sig[s, 0] * (-exi_H2)  # t <- sH

        # Response-matrix equation: dF/dxi = J @ F + G
        F = y[_N_S:].reshape(_N_S, _n_noise)
        dF = J @ F + G

        return [K, exi / H] + dF.ravel().tolist()

    def _e_boundary(xi, y):
        return e_max - y[0]
    _e_boundary.terminal = True
    _e_boundary.direction = -1

    n_F = _N_S * _n_noise
    y0 = [e0, 0.0] + [0.0] * n_F

    defaults = dict(method='RK45', rtol=1e-8, atol=1e-10, dense_output=True,
                    events=_e_boundary)
    defaults.update(solve_ivp_kwargs)

    sol = solve_ivp(rhs, xi_span, y0, **defaults)
    elapsed = _time.time() - _rhs_t0[0] if _rhs_t0[0] else 0
    print(f"\r  (V=0 ref) Done: {_rhs_count[0]} evals in {elapsed:.1f}s" + " " * 20)

    # Post-process: C = F @ F^T => sigma_e, sigma_t
    n_steps = sol.y.shape[1]
    sig = np.zeros((_N_S, n_steps))
    F_raw = sol.y[_N_S:, :]
    for k in range(n_steps):
        F = F_raw[:, k].reshape(_N_S, _n_noise)
        C = F @ F.T
        sig[:, k] = np.sqrt(np.maximum(np.diag(C), 0.0))
    sol.y = np.vstack([sol.y[:_N_S, :], sig])

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
    parser.add_argument('--precompute', action='store_true',
                        help='Precompute 4D rate tables for fast integration')
    parser.add_argument('--V-max', type=float, default=1.5,
                        help='Velocity grid half-width in sigma (default: 1.5)')
    parser.add_argument('--n-Ve', type=int, default=13,
                        help='Ve grid points (default: 13)')
    parser.add_argument('--n-Vn', type=int, default=13,
                        help='Vn grid points (default: 13)')
    parser.add_argument('--n-xi', type=int, default=50,
                        help='xi grid points for precomputation (default: 50)')
    parser.add_argument('--coarse', action='store_true',
                        help='Use coarse e-grid (0.1 spacing) with linear interp')
    parser.add_argument('--freeze-e', action='store_true',
                        help='Freeze eccentricity at its initial value')
    parser.add_argument('--freeze-Vx', action='store_true',
                        help='Freeze Vx at its initial value')
    parser.add_argument('--freeze-Vy', action='store_true',
                        help='Freeze Vy at its initial value')
    parser.add_argument('--freeze-varpi', action='store_true',
                        help='Freeze varpi at its initial value')
    parser.add_argument('--no-jacobian', action='store_true',
                        help='Disable Jacobian feedback in uncertainty ODEs')
    parser.add_argument('--diagonal', action='store_true',
                        help='Use diagonal uncertainty propagation instead '
                             'of the full covariance matrix (default)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file basename (writes <base>_full.dat '
                             'and <base>_V0.dat).  Omit for no file output.')
    args = parser.parse_args()

    if args.a0 is not None and args.xi_start is not None:
        parser.error("Use --a0 or --xi-start, not both")
    if args.a0 is not None:
        xi_start = -np.log(args.a0)
    elif args.xi_start is not None:
        xi_start = args.xi_start
    else:
        xi_start = 0.0

    e_interp = 'linear' if args.coarse else 'cubic'

    freeze = set()
    if args.freeze_e:     freeze.add('e')
    if args.freeze_Vx:    freeze.add('Vx')
    if args.freeze_Vy:    freeze.add('Vy')
    if args.freeze_varpi: freeze.add('varpi')

    use_jacobian = not args.no_jacobian
    use_full_cov = not args.diagonal

    print(f"Binary evolution: q={args.q}, e0={args.e0}, "
          f"V0/sigma=({args.Vx0}, {args.Vy0}), varpi0={args.varpi0}")
    mode = "precomputed 4D tables" if args.precompute else "on-the-fly (4-pt Lagrange)"
    if args.coarse:
        mode = "on-the-fly (coarse, linear)"
    if use_full_cov:
        unc_str = "full covariance (7×7)"
    elif use_jacobian:
        unc_str = "diagonal + Jacobian"
    else:
        unc_str = "diagonal (no Jacobian)"
    print(f"xi in [{xi_start:.4f}, {args.xi_end}]  "
          f"(a/a_h: {np.exp(-xi_start):.4f} → {np.exp(-args.xi_end):.6f}), "
          f"Chandrasekhar: {args.chandrasekhar}, rates: {mode}, "
          f"uncertainties: {unc_str}")

    xi_span = (xi_start, args.xi_end)

    # ── Full solution ─────────────────────────────────────────────────────
    print("Integrating (full)...")
    sol = solve(args.q, args.e0, args.Vx0, args.Vy0, args.varpi0,
                xi_span=xi_span, chandrasekhar=args.chandrasekhar,
                precompute=args.precompute, V_max=args.V_max,
                n_Ve=args.n_Ve, n_Vn=args.n_Vn, n_xi=args.n_xi,
                e_interp=e_interp, freeze=freeze, jacobian=use_jacobian,
                full_covariance=use_full_cov)

    if not sol.success and _status(sol).startswith("FAILED"):
        print(f"\nIntegration {_status(sol)}")
        sys.exit(1)

    xi = sol.t
    (e, Vx, Vy, varpi, t, x_pos, y_pos,
     sig_e, sig_Vx, sig_Vy, sig_varpi, sig_t, sig_x, sig_y) = sol.y
    a_ah = np.exp(-xi)
    V = np.hypot(Vx, Vy)

    print(f"\nFull solution — {len(xi)} steps, {_status(sol)}")
    print(f"  xi:        {xi[0]:.2f} → {xi[-1]:.2f}  "
          f"(a/a_h: {a_ah[0]:.3f} → {a_ah[-1]:.5f})")
    print(f"  e:         {e[0]:.4f} → {e[-1]:.4f}  "
          f"(±{sig_e[-1]:.4f})")
    print(f"  |V|/sigma: {V[0]:.4f} → {V[-1]:.4f}  "
          f"(±{np.hypot(sig_Vx[-1], sig_Vy[-1]):.4f})")
    print(f"  varpi:     {varpi[0]:.4f} → {varpi[-1]:.4f} rad  "
          f"(±{sig_varpi[-1]:.4f})")
    print(f"  t/T_hard:  {t[0]:.4f} → {t[-1]:.4f}  "
          f"(±{sig_t[-1]:.4f})")

    # ── V=0 reference solution ────────────────────────────────────────────
    print("Integrating (V=0 reference)...")
    sol0 = solve_simple(args.q, args.e0, xi_span=xi_span,
                        e_interp=e_interp, jacobian=use_jacobian)

    xi0 = sol0.t
    e0_ref, t0_ref, sig_e0, sig_t0 = sol0.y
    a_ah0 = np.exp(-xi0)

    print(f"V=0 reference — {len(xi0)} steps, {_status(sol0)}")

    # ── Data output ────────────────────────────────────────────────────────
    if args.output is not None:
        fn_full = f"{args.output}_full.dat"
        fn_v0   = f"{args.output}_V0.dat"
        has_cov = hasattr(sol, 'cov') and sol.cov is not None

        with open(fn_full, 'w') as f:
            f.write(f"# evolve: q={args.q} e0={args.e0} Vx0={args.Vx0} "
                    f"Vy0={args.Vy0} varpi0={args.varpi0} "
                    f"xi=[{xi_span[0]},{xi_span[1]}] "
                    f"chandrasekhar={args.chandrasekhar}\n")
            header = ("xi a_over_ah e sig_e Vx sig_Vx Vy sig_Vy "
                      "varpi sig_varpi t sig_t x sig_x y sig_y")
            if has_cov:
                header += " C_xx C_xy C_yy"
            f.write(header + "\n")
            for k in range(len(xi)):
                row = (f"{xi[k]:.15e} {a_ah[k]:.15e} "
                       f"{e[k]:.15e} {sig_e[k]:.15e} "
                       f"{Vx[k]:.15e} {sig_Vx[k]:.15e} "
                       f"{Vy[k]:.15e} {sig_Vy[k]:.15e} "
                       f"{varpi[k]:.15e} {sig_varpi[k]:.15e} "
                       f"{t[k]:.15e} {sig_t[k]:.15e} "
                       f"{x_pos[k]:.15e} {sig_x[k]:.15e} "
                       f"{y_pos[k]:.15e} {sig_y[k]:.15e}")
                if has_cov:
                    Cxx = sol.cov[25, k]
                    Cxy = sol.cov[26, k]
                    Cyy = sol.cov[27, k]
                    row += f" {Cxx:.15e} {Cxy:.15e} {Cyy:.15e}"
                f.write(row + "\n")
        print(f"  Written {fn_full}  ({len(xi)} rows)")

        with open(fn_v0, 'w') as f:
            f.write(f"# evolve V=0 ref: q={args.q} e0={args.e0} "
                    f"xi=[{xi_span[0]},{xi_span[1]}]\n")
            f.write("xi a_over_ah e sig_e t sig_t\n")
            for k in range(len(xi0)):
                f.write(f"{xi0[k]:.15e} {a_ah0[k]:.15e} "
                        f"{e0_ref[k]:.15e} {sig_e0[k]:.15e} "
                        f"{t0_ref[k]:.15e} {sig_t0[k]:.15e}\n")
        print(f"  Written {fn_v0}  ({len(xi0)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        fig = plt.figure(figsize=(14, 11))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
        band_alpha = 0.2

        # Row 0: e(xi), a/a_h(xi), CoM trajectory (x, y)
        ax = fig.add_subplot(gs[0, 0])
        ln, = ax.plot(xi, e)
        ax.fill_between(xi, e - sig_e, e + sig_e,
                        color=ln.get_color(), alpha=band_alpha)
        ax.set(xlabel=r'$\xi = \ln(a_h/a)$', ylabel='$e$')

        ax = fig.add_subplot(gs[0, 1])
        ln, = ax.plot(xi, a_ah)
        ax.set(xlabel=r'$\xi$', ylabel='$a/a_h$', yscale='log')

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(x_pos, y_pos)
        ax.plot(x_pos[0], y_pos[0], 'o', ms=5, label='start')
        ax.plot(x_pos[-1], y_pos[-1], 's', ms=5, label='end')
        n_ell = min(10, len(xi))
        ell_idx = np.linspace(0, len(xi) - 1, n_ell, dtype=int)
        _has_cov = hasattr(sol, 'cov') and sol.cov is not None
        for ii in ell_idx:
            if _has_cov:
                Cxx = sol.cov[25, ii]
                Cxy = sol.cov[26, ii]
                Cyy = sol.cov[27, ii]
                cov2 = np.array([[Cxx, Cxy], [Cxy, Cyy]])
                eigvals, eigvecs = np.linalg.eigh(cov2)
                eigvals = np.maximum(eigvals, 0.0)
                w_ell = 2 * np.sqrt(eigvals[1])
                h_ell = 2 * np.sqrt(eigvals[0])
                angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
            else:
                w_ell = 2 * sig_x[ii]
                h_ell = 2 * sig_y[ii]
                angle = 0.0
            ell = Ellipse((x_pos[ii], y_pos[ii]),
                          width=w_ell, height=h_ell, angle=angle,
                          facecolor='C0', alpha=0.15, edgecolor='none')
            ax.add_patch(ell)
        ax.set(xlabel=r'$x\;/\;(\sigma\, T_{\rm hard})$',
               ylabel=r'$y\;/\;(\sigma\, T_{\rm hard})$')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize='small')

        # Row 1: Vx,Vy(xi), varpi(xi)
        ax = fig.add_subplot(gs[1, 0])
        ln1, = ax.plot(xi, Vx, label=r'$V_x/\sigma$')
        ax.fill_between(xi, Vx - sig_Vx, Vx + sig_Vx,
                        color=ln1.get_color(), alpha=band_alpha)
        ln2, = ax.plot(xi, Vy, label=r'$V_y/\sigma$')
        ax.fill_between(xi, Vy - sig_Vy, Vy + sig_Vy,
                        color=ln2.get_color(), alpha=band_alpha)
        ax.set(xlabel=r'$\xi$', ylabel=r'$V/\sigma$')
        ax.legend(fontsize='small')

        ax = fig.add_subplot(gs[1, 1])
        ln, = ax.plot(xi, varpi)
        ax.fill_between(xi, varpi - sig_varpi, varpi + sig_varpi,
                        color=ln.get_color(), alpha=band_alpha)
        ax.set(xlabel=r'$\xi$', ylabel=r'$\varpi$ [rad]')

        # Row 2: e(t) and a/a_h(t) with V=0 dashed reference
        ax = fig.add_subplot(gs[2, 0])
        ln, = ax.plot(t, e, label='full')
        ax.fill_between(t, e - sig_e, e + sig_e,
                        color=ln.get_color(), alpha=band_alpha)
        ln0, = ax.plot(t0_ref, e0_ref, '--', label=r'$V{=}0$ ref')
        ax.fill_between(t0_ref, e0_ref - sig_e0, e0_ref + sig_e0,
                        color=ln0.get_color(), alpha=band_alpha)
        ax.set(xlabel=r'$t\;/\;T_{\rm hard}$', ylabel='$e$')
        ax.legend(fontsize='small')

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
                     f'Chandrasekhar = {args.chandrasekhar}')
        out = os.path.join(_HERE, 'evolution.pdf')
        fig.savefig(out, bbox_inches='tight')
        print(f"\nPlot saved to {out}")
        plt.show()
    except ImportError:
        pass
