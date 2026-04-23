#!/usr/bin/env python3
"""
weight-Maxwellian-3D-velocity.py

Reads per-particle OR spherical-harmonic-moment binary output from
main-3D-velocity.cpp and computes Maxwellian-weighted quantities
(H, K, P, P_x, P_y, Q, tau, F) for arbitrary CoM velocity V and dispersion sigma.
File format is auto-detected.

Includes:
  - Isotropic consistency check (V=0 vs existing text-file results)
  - Harmonics vs per-particle consistency check
  - Chandrasekhar dynamical friction check (large V)
  - Plotting of H, K, Q, P_x, P_y, tau as functions of a/a_h for multiple V

Usage:
    python weight-Maxwellian-3D-velocity.py particles_q=0.2_e=0.6.bin
    python weight-Maxwellian-3D-velocity.py harmonics_q=0.2_e=0.6.bin
    python weight-Maxwellian-3D-velocity.py harmonics_q=0.2_e=0.6.bin --check-harmonics particles_q=0.2_e=0.6.bin
    python weight-Maxwellian-3D-velocity.py particles_q=0.2_e=0.6.bin --check-iso q=0.2_e=0.6_Tcut=100000000000.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, lpmv
from scipy.integrate import quad
import struct
import argparse

_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# ── I/O ──────────────────────────────────────────────────────────────────────

def read_binary(filename):
    """Read per-particle binary file produced by main-3D-velocity.cpp.

    Returns (meta, bins) where meta is a dict of scalar parameters and
    bins is a list of per-v-bin dicts with arrays v_in, dE, dv, dL, dT.
    """
    bins = []
    with open(filename, 'rb') as f:
        q        = struct.unpack('d', f.read(8))[0]
        e        = struct.unpack('d', f.read(8))[0]
        rp_max   = struct.unpack('d', f.read(8))[0]
        r_sphere = struct.unpack('d', f.read(8))[0]
        n_v      = struct.unpack('i', f.read(4))[0]
        n_per_v  = struct.unpack('i', f.read(4))[0]

        for _ in range(n_v):
            v_inf = struct.unpack('d', f.read(8))[0]
            n_esc = struct.unpack('i', f.read(4))[0]
            if n_esc > 0:
                raw = np.frombuffer(f.read(n_esc * 96),
                                    dtype=np.float64).reshape(n_esc, 12)
                bins.append(dict(
                    v_inf=v_inf, n_esc=n_esc,
                    v_in=raw[:, :3].copy(),
                    dE=raw[:, 3].copy(),
                    dv=raw[:, 4:7].copy(),
                    dL=raw[:, 7:10].copy(),
                    dT=raw[:, 10].copy(),
                    delta_varpi=raw[:, 11].copy(),
                ))
            else:
                bins.append(dict(
                    v_inf=v_inf, n_esc=0,
                    v_in=np.empty((0, 3)), dE=np.empty(0),
                    dv=np.empty((0, 3)), dL=np.empty((0, 3)),
                    dT=np.empty(0), delta_varpi=np.empty(0),
                ))

    meta = dict(q=q, e=e, rp_max=rp_max, r_sphere=r_sphere, n_per_v=n_per_v)
    return meta, bins


def read_harmonics(filename):
    """Read SH-moment binary file produced by main-3D-velocity.cpp (l_max > 0).

    Returns (meta, harm_bins) where meta includes l_max, and each bin has
    v_inf, n_esc, M (7 x n_sh first moments), S (7 x n_sh second moments).
    """
    N_Q = 8
    harm_bins = []
    with open(filename, 'rb') as f:
        q        = struct.unpack('d', f.read(8))[0]
        e        = struct.unpack('d', f.read(8))[0]
        rp_max   = struct.unpack('d', f.read(8))[0]
        r_sphere = struct.unpack('d', f.read(8))[0]
        n_v      = struct.unpack('i', f.read(4))[0]
        n_per_v  = struct.unpack('i', f.read(4))[0]
        l_max    = struct.unpack('i', f.read(4))[0]

        n_sh = (l_max + 1)**2

        for _ in range(n_v):
            v_inf = struct.unpack('d', f.read(8))[0]
            n_esc = struct.unpack('i', f.read(4))[0]

            M_raw = np.frombuffer(f.read(N_Q * n_sh * 8),
                                  dtype=np.float64).reshape(N_Q, n_sh).copy()
            S_raw = np.frombuffer(f.read(N_Q * n_sh * 8),
                                  dtype=np.float64).reshape(N_Q, n_sh).copy()

            harm_bins.append(dict(v_inf=v_inf, n_esc=n_esc, M=M_raw, S=S_raw))

    meta = dict(q=q, e=e, rp_max=rp_max, r_sphere=r_sphere,
                n_per_v=n_per_v, l_max=l_max)
    return meta, harm_bins


def detect_file_type(filename):
    """Peek at the header to distinguish per-particle vs harmonics binary."""
    with open(filename, 'rb') as f:
        f.read(4 * 8)  # skip q, e, rp_max, r_sphere
        f.read(2 * 4)  # skip n_v, n_per_v
        extra = f.read(4)
        if len(extra) == 0:
            return 'per-particle'
        # In per-particle format, the next bytes are v_inf (double, 8 bytes).
        # In harmonics format, next is l_max (int, 4 bytes) then v_inf (double).
        # Peek: read 4 more bytes. If per-particle, the 8 bytes (extra + next4)
        # form a valid double v_inf (small positive). If harmonics, extra is
        # l_max (small int like 10) and next 4 bytes start v_inf.
        next4 = f.read(4)
        if len(next4) < 4:
            return 'per-particle'
        candidate_int = struct.unpack('i', extra)[0]
        # l_max is a small positive integer (1..50); v_inf as int would be huge
        if 1 <= candidate_int <= 100:
            return 'harmonics'
        return 'per-particle'


# ── Helper functions ─────────────────────────────────────────────────────────

def b_max_val(v, rp_max):
    return rp_max * np.sqrt(1 + 2 / (v**2 * rp_max))


def trapezoidal_weights(v_values):
    """Trapezoidal weights for the v-grid with a v=0 left boundary."""
    v_ext = np.concatenate([[0.0], v_values])
    w = np.empty(len(v_ext))
    w[0] = (v_ext[1] - v_ext[0]) / 2
    w[-1] = (v_ext[-1] - v_ext[-2]) / 2
    if len(v_ext) > 2:
        w[1:-1] = (v_ext[2:] - v_ext[:-2]) / 2
    return w[1:]

# ── Reweighting engine ───────────────────────────────────────────────────────

def reweight(meta, bins, V, sigma, rho=1.0):
    """
    Compute Maxwellian-weighted integrals for given CoM velocity V and
    velocity dispersion sigma.

    The integral being estimated for a generic per-particle quantity X is:

        I_X = -rho * int d^3v  f_3D(v + V, sigma) * v * pi*b_max^2(v) * X(v)

    where v is the velocity of the incoming star in the binary rest frame
    and V is the binary centre-of-mass velocity in the lab frame.

    Split as: trapezoidal in |v| over the v-grid, MC over directions using the
    N random directions per v-bin.  See plan for derivation.

    Returns dict with P, F, tau (and H, K, P_x, P_y, Q derived from them),
    plus uncertainties sP, sF, stau, sH, sK, sP_x, sP_y, sQ.
    """
    q      = meta['q']
    e_ecc  = meta['e']
    rp_max = meta['rp_max']
    mu     = q / (1 + q)**2

    V = np.asarray(V, dtype=np.float64)
    v_values = np.array([b['v_inf'] for b in bins])
    dv = trapezoidal_weights(v_values)

    norm_f = (2 * np.pi * sigma**2)**(-1.5)
    inv_2s2 = 0.5 / sigma**2

    acc_E = 0.0;    var_E = 0.0
    acc_F = np.zeros(3);  var_F = np.zeros(3)
    acc_L = np.zeros(3);  var_L = np.zeros(3)
    acc_varpi = 0.0;  var_varpi = 0.0

    for j, b in enumerate(bins):
        Nj = b['n_esc']
        if Nj == 0:
            continue

        vj    = b['v_inf']
        bmax  = rp_max * np.sqrt(1 + 2 / (vj**2 * rp_max))
        A_j   = 4 * np.pi**2 * vj**3 * bmax**2 * dv[j]

        delta = b['v_in'] + V[np.newaxis, :]
        f3d   = norm_f * np.exp(-np.sum(delta**2, axis=1) * inv_2s2)

        wE = f3d * b['dE']
        acc_E += A_j * np.mean(wE)
        var_E += A_j**2 * np.var(wE) / Nj

        wF = f3d[:, np.newaxis] * b['dv']
        acc_F += A_j * np.mean(wF, axis=0)
        var_F += A_j**2 * np.var(wF, axis=0) / Nj

        wL = f3d[:, np.newaxis] * b['dL']
        acc_L += A_j * np.mean(wL, axis=0)
        var_L += A_j**2 * np.var(wL, axis=0) / Nj

        wV = f3d * b['delta_varpi']
        acc_varpi += A_j * np.mean(wV)
        var_varpi += A_j**2 * np.var(wV) / Nj

    P    = -rho * acc_E
    sP   =  rho * np.sqrt(max(var_E, 0))
    F    = -rho * acc_F
    sF   =  rho * np.sqrt(np.maximum(var_F, 0))
    tau  = -rho * acc_L
    stau =  rho * np.sqrt(np.maximum(var_L, 0))
    varpi_dot  = rho * acc_varpi
    svarpi_dot = rho * np.sqrt(max(var_varpi, 0))

    H  = -2 * sigma * P / (mu * rho)
    sH =  2 * sigma * sP / (mu * rho)

    def _safe_ratio_err(a, sa, b, sb):
        if abs(a) == 0 or abs(b) == 0:
            return np.nan
        return abs(a / b) * np.sqrt((sa / a)**2 + (sb / b)**2)

    if abs(P) > 0 and abs(tau[2]) > 1e-300:
        c = np.sqrt(1 - e_ecc**2) / (2 * e_ecc)
        K  = -(1 - e_ecc**2) / (2 * e_ecc) + c * tau[2] / P
        sK = c * _safe_ratio_err(tau[2], stau[2], P, sP)
    else:
        K = np.nan; sK = np.nan

    def _P_comp(Fk, sFk):
        if abs(P) > 0 and abs(Fk) > 1e-300:
            Pval = -(mu / (2 * sigma)) * Fk / P
            sPval = (mu / (2 * sigma)) * _safe_ratio_err(Fk, sFk, P, sP)
            return Pval, sPval
        return np.nan, np.nan

    P_x, sP_x = _P_comp(F[0], sF[0])
    P_y, sP_y = _P_comp(F[1], sF[1])
    P_z, sP_z = _P_comp(F[2], sF[2])

    if abs(P) > 0 and abs(varpi_dot) > 1e-300:
        Q = -(mu / 2) * varpi_dot / P
        sQ = (mu / 2) * _safe_ratio_err(varpi_dot, svarpi_dot, P, sP)
    else:
        Q = np.nan; sQ = np.nan

    def _R_comp(tau_perp, stau_perp):
        denom = 2 * P * np.sqrt(1 - e_ecc**2)
        if abs(P) > 0 and abs(tau_perp) > 1e-300:
            Rval = tau_perp / denom
            sRval = _safe_ratio_err(tau_perp, stau_perp, P, sP) / (2 * np.sqrt(1 - e_ecc**2))
            return Rval, sRval
        return np.nan, np.nan

    R_x, sR_x = _R_comp(tau[1], stau[1])
    R_y, sR_y = _R_comp(-tau[0], stau[0])

    return dict(
        P=P, sP=sP, F=F, sF=sF, tau=tau, stau=stau,
        H=H, sH=sH, K=K, sK=sK,
        P_x=P_x, sP_x=sP_x, P_y=P_y, sP_y=sP_y, P_z=P_z, sP_z=sP_z,
        varpi_dot=varpi_dot, svarpi_dot=svarpi_dot,
        Q=Q, sQ=sQ, R_x=R_x, sR_x=sR_x, R_y=R_y, sR_y=sR_y,
    )

# ── Real spherical harmonics (matches C++ compute_real_Ylm exactly) ──────────

def real_Ylm_all(cos_theta, phi, l_max):
    """All real SH Y^R_lm for l=0..l_max, m=-l..l.
    Returns array of size (l_max+1)^2 with index l*l + l + m."""
    sz = l_max + 1
    n_sh = sz * sz
    out = np.empty(n_sh)
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))

    cm = np.empty(sz); sm = np.empty(sz)
    cm[0] = 1.0; sm[0] = 0.0
    if l_max >= 1:
        cm[1] = np.cos(phi); sm[1] = np.sin(phi)
    for m in range(2, sz):
        cm[m] = 2*cm[1]*cm[m-1] - cm[m-2]
        sm[m] = 2*cm[1]*sm[m-1] - sm[m-2]

    plm = np.zeros((sz, sz))
    plm[0, 0] = 1.0
    for m in range(1, sz):
        plm[m, m] = -(2*m - 1) * sin_theta * plm[m-1, m-1]
    for m in range(sz - 1):
        plm[m+1, m] = (2*m + 1) * cos_theta * plm[m, m]
    for m in range(sz):
        for l in range(m + 2, sz):
            plm[l, m] = ((2*l-1)*cos_theta*plm[l-1, m] - (l+m-1)*plm[l-2, m]) / (l - m)

    fact = np.ones(2*l_max + 2)
    for i in range(1, len(fact)):
        fact[i] = fact[i-1] * i

    for l in range(sz):
        K0 = np.sqrt((2*l+1) / (4*np.pi))
        out[l*l + l] = K0 * plm[l, 0]
        for m in range(1, l+1):
            Km = np.sqrt((2*l+1)/(4*np.pi) * fact[l-m]/fact[l+m])
            val = Km * plm[l, m]
            out[l*l + l + m] = np.sqrt(2) * val * cm[m]
            out[l*l + l - m] = np.sqrt(2) * val * sm[m]
    return out


# ── SH-based reweighting engine ──────────────────────────────────────────────

def reweight_from_harmonics(meta, harm_bins, V, sigma, rho=1.0):
    """Reconstruct Maxwellian-weighted integrals from SH moment data.

    V is the binary centre-of-mass velocity (in code units).  The shifted
    Maxwellian f(v + V) introduces the exponential factor exp(-v.V/sigma^2),
    which is expanded using the addition theorem:
        exp(-alpha * n.Vhat) = 4pi sum_{lm} i_l(alpha) Y_lm(-Vhat) Y_lm(n)
    where i_l are modified spherical Bessel functions of the first kind
    and alpha = v|V|/sigma^2.

    Returns the same dict as reweight() (P, F, tau, H, K, P_x, P_y, Q, uncertainties).
    """
    from scipy.special import ive

    q      = meta['q']
    e_ecc  = meta['e']
    rp_max = meta['rp_max']
    l_max  = meta['l_max']
    mu     = q / (1 + q)**2
    n_sh   = (l_max + 1)**2

    V = np.asarray(V, dtype=np.float64)
    V_mag = np.linalg.norm(V)

    v_values = np.array([b['v_inf'] for b in harm_bins])
    dv = trapezoidal_weights(v_values)

    if V_mag > 1e-30:
        neg_Vhat = -V / V_mag
        Ylm_negV = real_Ylm_all(neg_Vhat[2], np.arctan2(neg_Vhat[1], neg_Vhat[0]), l_max)
    else:
        Ylm_negV = np.zeros(n_sh)
        Ylm_negV[0] = 1.0 / np.sqrt(4 * np.pi)

    # l-index for each flat SH index (precompute once)
    l_idx = np.empty(n_sh, dtype=int)
    for l in range(l_max + 1):
        l_idx[l*l : (l+1)*(l+1)] = l

    inv_s2 = 1.0 / sigma**2
    norm1  = (2 * np.pi * sigma**2)**(-1.5)

    acc_E = 0.0;    var_E = 0.0
    acc_F = np.zeros(3);  var_F = np.zeros(3)
    acc_L = np.zeros(3);  var_L = np.zeros(3)
    acc_varpi = 0.0;  var_varpi = 0.0

    for j, b in enumerate(harm_bins):
        Nj = b['n_esc']
        if Nj == 0:
            continue

        vj   = b['v_inf']
        bmax = rp_max * np.sqrt(1 + 2 / (vj**2 * rp_max))
        A_j  = 4 * np.pi**2 * vj**3 * bmax**2 * dv[j]

        alpha = vj * V_mag * inv_s2

        # Scaled modified spherical Bessel: exp(-x)*i_l(x) = sqrt(pi/(2x)) * ive(l+0.5, x)
        if alpha > 1e-30:
            orders = np.arange(l_max + 1) + 0.5
            s1 = np.sqrt(np.pi / (2 * alpha))  * ive(orders, alpha)
            s2 = np.sqrt(np.pi / (4 * alpha))  * ive(orders, 2 * alpha)
        else:
            s1 = np.zeros(l_max + 1); s1[0] = 1.0
            s2 = np.zeros(l_max + 1); s2[0] = 1.0

        G1 = norm1 * np.exp(-(vj - V_mag)**2 / (2 * sigma**2))
        G2 = norm1**2 * np.exp(-(vj - V_mag)**2 / sigma**2)

        w1 = s1[l_idx] * Ylm_negV          # shape (n_sh,)
        w2 = s2[l_idx] * Ylm_negV

        fourpi = 4 * np.pi
        M  = b['M']    # (8, n_sh)
        Sq = b['S']    # (8, n_sh)

        mean_X  = G1 * fourpi * (M  @ w1)    # (8,)
        mean_X2 = G2 * fourpi * (Sq @ w2)    # (8,)
        var_X   = mean_X2 - mean_X**2

        acc_E += A_j * mean_X[0]
        var_E += A_j**2 * max(var_X[0], 0) / Nj

        acc_F += A_j * mean_X[1:4]
        var_F += A_j**2 * np.maximum(var_X[1:4], 0) / Nj

        acc_L += A_j * mean_X[4:7]
        var_L += A_j**2 * np.maximum(var_X[4:7], 0) / Nj

        acc_varpi += A_j * mean_X[7]
        var_varpi += A_j**2 * max(var_X[7], 0) / Nj

    P    = -rho * acc_E
    sP   =  rho * np.sqrt(max(var_E, 0))
    F    = -rho * acc_F
    sF   =  rho * np.sqrt(np.maximum(var_F, 0))
    tau  = -rho * acc_L
    stau =  rho * np.sqrt(np.maximum(var_L, 0))
    varpi_dot  = rho * acc_varpi
    svarpi_dot = rho * np.sqrt(max(var_varpi, 0))

    H  = -2 * sigma * P / (mu * rho)
    sH =  2 * sigma * sP / (mu * rho)

    def _safe_ratio_err(a, sa, b, sb):
        if abs(a) == 0 or abs(b) == 0:
            return np.nan
        return abs(a / b) * np.sqrt((sa / a)**2 + (sb / b)**2)

    if abs(P) > 0 and abs(tau[2]) > 1e-300:
        c = np.sqrt(1 - e_ecc**2) / (2 * e_ecc)
        K  = -(1 - e_ecc**2) / (2 * e_ecc) + c * tau[2] / P
        sK = c * _safe_ratio_err(tau[2], stau[2], P, sP)
    else:
        K = np.nan; sK = np.nan

    def _P_comp(Fk, sFk):
        if abs(P) > 0 and abs(Fk) > 1e-300:
            Pval = -(mu / (2 * sigma)) * Fk / P
            sPval = (mu / (2 * sigma)) * _safe_ratio_err(Fk, sFk, P, sP)
            return Pval, sPval
        return np.nan, np.nan

    P_x, sP_x = _P_comp(F[0], sF[0])
    P_y, sP_y = _P_comp(F[1], sF[1])
    P_z, sP_z = _P_comp(F[2], sF[2])

    if abs(P) > 0 and abs(varpi_dot) > 1e-300:
        Q = -(mu / 2) * varpi_dot / P
        sQ = (mu / 2) * _safe_ratio_err(varpi_dot, svarpi_dot, P, sP)
    else:
        Q = np.nan; sQ = np.nan

    def _R_comp(tau_perp, stau_perp):
        denom = 2 * P * np.sqrt(1 - e_ecc**2)
        if abs(P) > 0 and abs(tau_perp) > 1e-300:
            Rval = tau_perp / denom
            sRval = _safe_ratio_err(tau_perp, stau_perp, P, sP) / (2 * np.sqrt(1 - e_ecc**2))
            return Rval, sRval
        return np.nan, np.nan

    R_x, sR_x = _R_comp(tau[1], stau[1])
    R_y, sR_y = _R_comp(-tau[0], stau[0])

    return dict(
        P=P, sP=sP, F=F, sF=sF, tau=tau, stau=stau,
        H=H, sH=sH, K=K, sK=sK,
        P_x=P_x, sP_x=sP_x, P_y=P_y, sP_y=sP_y, P_z=P_z, sP_z=sP_z,
        varpi_dot=varpi_dot, svarpi_dot=svarpi_dot,
        Q=Q, sQ=sQ, R_x=R_x, sR_x=sR_x, R_y=R_y, sR_y=sR_y,
    )

# ── Isotropic check via original text-file method ────────────────────────────

def isotropic_from_text(text_file, q, e, rp_max, rho=1.0):
    """Reproduce H, K, P, tau, F vs a/a_h from the original text-file pipeline."""
    mu = q / (1 + q)**2
    data = np.loadtxt(text_file, unpack=True)
    v       = data[0]
    DeltaE  = data[1];  sDeltaE  = data[2]
    Deltavx = data[5];  sDeltavx = data[6]
    Deltavy = data[7];  sDeltavy = data[8]
    Deltavz = data[9];  sDeltavz = data[10]
    DeltaLx = data[11]; sDeltaLx = data[12]
    DeltaLy = data[13]; sDeltaLy = data[14]
    DeltaLz = data[15]; sDeltaLz = data[16]
    Delta_varpi = data[17]; sDelta_varpi = data[18]

    def bm(vv):
        return rp_max * np.sqrt(1 + 2 / (vv**2 * rp_max))

    def f_speed(vv, sig):
        return np.sqrt(2 / np.pi) * (vv**2 / sig**3) * np.exp(-vv**2 / (2 * sig**2))

    N_ah = 50
    a_h   = np.logspace(3, -2, N_ah)
    sigma = np.sqrt(mu / (4 * a_h))

    Pv   = -np.pi * bm(v)**2 * rho * v * DeltaE
    sPv  = np.pi * bm(v)**2 * rho * v * sDeltaE
    Fvx  = -np.pi * bm(v)**2 * rho * v * Deltavx
    sFvx = np.pi * bm(v)**2 * rho * v * sDeltavx
    Fvy  = -np.pi * bm(v)**2 * rho * v * Deltavy
    sFvy = np.pi * bm(v)**2 * rho * v * sDeltavy
    Fvz  = -np.pi * bm(v)**2 * rho * v * Deltavz
    sFvz = np.pi * bm(v)**2 * rho * v * sDeltavz
    Tvx  = -np.pi * bm(v)**2 * rho * v * DeltaLx
    sTvx = np.pi * bm(v)**2 * rho * v * sDeltaLx
    Tvy  = -np.pi * bm(v)**2 * rho * v * DeltaLy
    sTvy = np.pi * bm(v)**2 * rho * v * sDeltaLy
    Tvz  = -np.pi * bm(v)**2 * rho * v * DeltaLz
    sTvz = np.pi * bm(v)**2 * rho * v * sDeltaLz
    Hv   = (2 * np.pi * v**2 * bm(v)**2) * DeltaE / mu
    sHv  = (2 * np.pi * v**2 * bm(v)**2) * sDeltaE / mu
    varpi_dot_v  = np.pi * bm(v)**2 * rho * v * Delta_varpi
    svarpi_dot_v = np.pi * bm(v)**2 * rho * v * sDelta_varpi

    v0 = np.hstack([0, v])
    f0 = np.vstack([np.zeros((1, N_ah)),
                    f_speed(v[:, None], sigma[None, :])])
    H_int0 = np.vstack([np.zeros((1, N_ah)),
                         (sigma[None, :] / v[:, None]) * f_speed(v[:, None], sigma[None, :])])

    def _tile(arr):
        return np.vstack([np.zeros((1, N_ah)), np.tile(arr[:, None], (1, N_ah))])

    P   = _trapz(_tile(Pv)  * f0, x=v0, axis=0)
    Fx  = _trapz(_tile(Fvx) * f0, x=v0, axis=0)
    Fy  = _trapz(_tile(Fvy) * f0, x=v0, axis=0)
    Fz  = _trapz(_tile(Fvz) * f0, x=v0, axis=0)
    tx  = _trapz(_tile(Tvx) * f0, x=v0, axis=0)
    ty  = _trapz(_tile(Tvy) * f0, x=v0, axis=0)
    tz  = _trapz(_tile(Tvz) * f0, x=v0, axis=0)
    H   = _trapz(_tile(Hv)  * H_int0, x=v0, axis=0)
    varpi_dot_int = _trapz(_tile(varpi_dot_v) * f0, x=v0, axis=0)

    weights = np.empty_like(v0)
    weights[0]    = (v0[1] - v0[0]) / 2
    weights[-1]   = (v0[-1] - v0[-2]) / 2
    weights[1:-1] = (v0[2:] - v0[:-2]) / 2

    def _unc(s_arr, f_arr):
        integ = _tile(s_arr) * f_arr
        return np.sqrt(np.sum((weights[:, None] * integ)**2, axis=0))

    sP  = _unc(sPv,  f0)
    sFx = _unc(sFvx, f0)
    sFy = _unc(sFvy, f0)
    sFz = _unc(sFvz, f0)
    stx = _unc(sTvx, f0)
    sty = _unc(sTvy, f0)
    stz = _unc(sTvz, f0)
    sH  = _unc(sHv,  H_int0)
    svarpi_dot_int = _unc(svarpi_dot_v, f0)

    K  = -(1 - e**2) / (2 * e) + np.sqrt(1 - e**2) / (2 * e) * tz / P
    sK = np.sqrt(1 - e**2) / (2 * e) * np.abs(tz / P) * np.sqrt((stz / tz)**2 + (sP / P)**2)
    Q = -(mu / 2) * varpi_dot_int / P
    sQ = -(mu / 2) * (varpi_dot_int / P) * np.sqrt((svarpi_dot_int / varpi_dot_int)**2 + (sP / P)**2)

    return dict(a_h=a_h, sigma=sigma,
                P=P, sP=sP, H=H, sH=sH, K=K, sK=sK,
                F=np.array([Fx, Fy, Fz]), sF=np.array([sFx, sFy, sFz]),
                tau=np.array([tx, ty, tz]), stau=np.array([stx, sty, stz]),
                varpi_dot=varpi_dot_int, svarpi_dot=svarpi_dot_int,
                Q=Q, sQ=sQ)

# ── Chandrasekhar dynamical friction ─────────────────────────────────────────

def chandrasekhar_F(V_mag, sigma, rho, ln_Lambda):
    """Chandrasekhar dynamical friction force on the binary in the
    constant-Coulomb-log approximation (used as a sanity-check curve).

    Returns the force component along V_hat (negative = deceleration),
    where V is the binary centre-of-mass velocity.  Units: G=M=1.

    *ln_Lambda* should be the effective Coulomb logarithm derived from the
    non-perturbative formula, e.g.
        ln_Lambda = (1/2) * ln((b_max^2 + b_90^2) / (b_min^2 + b_90^2))
    evaluated at a representative velocity (e.g. v ~ sigma).  See
    :func:`effective_ln_lambda`.
    """
    X = V_mag / (np.sqrt(2) * sigma)
    bracket = erf(X) - 2 * X / np.sqrt(np.pi) * np.exp(-X**2)
    return -4 * np.pi * rho * ln_Lambda / V_mag**2 * bracket


def _ln_lambda(u_tilde, xi, q, r_outer_ah):
    r"""Effective Coulomb logarithm for dynamical friction at relative
    speed :math:`u = u\_tilde \cdot \sigma`.

    Uses the *non-perturbative* expression obtained from the exact Keplerian
    hyperbola integral over impact parameters,

    .. math::
        \ln\Lambda(u) = \tfrac12 \ln\!\Bigl(
            \frac{b_{\max}^2 + b_{90}^2(u)}{b_{\min}^2(u) + b_{90}^2(u)}
        \Bigr)\,,

    with the 90-degree deflection scale :math:`b_{90}(u) = G M / u^2`.  This
    reduces to the standard :math:`\ln(b_{\max}/b_{\min})` when
    :math:`b_{90} \ll b_{\min}`, and to :math:`\ln(b_{\max}/b_{90})` when
    :math:`b_{\min} \ll b_{90} \ll b_{\max}`; it is finite as
    :math:`b_{\min} \to 0`.

    In dimensionless variables (lengths in :math:`a_h`, velocities in
    :math:`\sigma`):

    * :math:`b_{\min}/a_h = 5\, e^{-\xi}\,\sqrt{1 + 8(1+q)^2 e^{\xi} / (5 q\, \tilde u^2)}`
      (cutoff above which the analytic Chandrasekhar piece takes over from
      the 3-body simulations, see :eq:`bmin-v` in the paper);
    * :math:`b_{90}/a_h = 4 (1+q)^2 / (q\, \tilde u^2)`;
    * :math:`b_{\max}/a_h =` ``r_outer_ah``.
    """
    a_over_ah = np.exp(-xi)
    ratio = 8.0 * (1.0 + q)**2 * np.exp(xi) / (5.0 * q * u_tilde**2)
    bmin_ah = 5.0 * a_over_ah * np.sqrt(1.0 + ratio)
    b90_ah = 4.0 * (1.0 + q)**2 / (q * u_tilde**2)
    num = r_outer_ah * r_outer_ah + b90_ah * b90_ah
    den = bmin_ah * bmin_ah + b90_ah * b90_ah
    lnL = 0.5 * np.log(num / den)
    if lnL < 0.0:
        return 0.0
    return lnL


def effective_ln_lambda(V_tilde, xi, q, r_outer_ah, u_tilde=1.0):
    """Effective constant Coulomb log evaluated at a representative speed.

    Convenience wrapper around :func:`_ln_lambda` that picks a single value
    of the relative speed (default :math:`u = \\sigma`, i.e.
    ``u_tilde = 1``) for use in the constant-:math:`\\ln\\Lambda` Chandrasekhar
    erf formula.
    """
    return _ln_lambda(u_tilde, xi, q, r_outer_ah)


def chandrasekhar_decel_integral(V_tilde, xi, q, r_outer_ah):
    r"""Full Chandrasekhar integral with velocity-dependent, *non-perturbative*
    Coulomb logarithm (cured at small impact parameter by :math:`b_{90}`).

    Returns the dimensionless scalar *J* such that the Chandrasekhar
    force vector in code units (G=M=a=1) is

        F_Ch = (4 pi rho / sigma^2) * J * V_hat

    where V_hat is the CoM velocity direction.
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
            kernel_exp = np.exp(-(u**2 + V_tilde**2) / 2.0) * (
                -alpha**3 / 3.0 - alpha**5 / 30.0)
        else:
            e_minus = np.exp(-0.5 * (u - V_tilde)**2)
            e_plus = np.exp(-0.5 * (u + V_tilde)**2)
            kernel_exp = 0.5 * ((1.0 - alpha) * e_minus
                                - (1.0 + alpha) * e_plus)
        return lnL * 2.0 * kernel_exp / alpha**2

    J, _ = quad(integrand, 0.0, np.inf, limit=200)
    J /= np.sqrt(2.0 * np.pi)
    return J

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Anisotropic Maxwellian reweighting',
        epilog='The input file is auto-detected as per-particle or SH harmonics format.')
    parser.add_argument('binary_file',
                        help='Binary file (per-particle or harmonics) from main-3D-velocity.cpp')
    parser.add_argument('--check-iso', default=None,
                        help='Text file from original code for isotropic consistency check')
    parser.add_argument('--check-harmonics', default=None,
                        help='Per-particle .bin file to compare against SH harmonics results')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not open interactive plot windows')
    args = parser.parse_args()

    # ── Load data (auto-detect format) ──
    ftype = detect_file_type(args.binary_file)
    if ftype == 'harmonics':
        meta, data_bins = read_harmonics(args.binary_file)
        def _reweight(V, sig):
            return reweight_from_harmonics(meta, data_bins, V, sig, rho)
        total_particles = sum(b['n_esc'] for b in data_bins)
        print(f"Loaded harmonics file: {args.binary_file} "
              f"(l_max={meta['l_max']}, N_v={len(data_bins)}, "
              f"N_per_v={meta['n_per_v']}, total escaped={total_particles})")
    else:
        meta, data_bins = read_binary(args.binary_file)
        def _reweight(V, sig):
            return reweight(meta, data_bins, V, sig, rho)
        total_particles = sum(b['n_esc'] for b in data_bins)
        print(f"Loaded per-particle file: {args.binary_file} "
              f"(N_v={len(data_bins)}, N_per_v={meta['n_per_v']}, "
              f"total escaped={total_particles})")

    q      = meta['q']
    e_ecc  = meta['e']
    rp_max = meta['rp_max']
    mu     = q / (1 + q)**2
    rho    = 1.0
    print(f"  q={q}, e={e_ecc}, rp_max={rp_max}")

    # ── a/a_h and sigma grid ──
    N_ah  = 50
    a_h   = np.logspace(3, -2, N_ah)
    sigma = np.sqrt(mu / (4 * a_h))

    # ── Isotropic consistency check ──
    if args.check_iso:
        print("\n=== Isotropic consistency check (V = 0) ===")
        txt = isotropic_from_text(args.check_iso, q, e_ecc, rp_max, rho)

        H_rw  = np.empty(N_ah)
        sH_rw = np.empty(N_ah)
        for i, sig in enumerate(sigma):
            r = _reweight(np.zeros(3), sig)
            H_rw[i]  = r['H']
            sH_rw[i] = r['sH']

        H_txt  = txt['H']
        sH_txt = txt['sH']

        rel_H = np.abs(H_rw - H_txt) / (np.abs(H_txt) + 1e-300)
        within_1s = np.abs(H_rw - H_txt) < (sH_rw + sH_txt)

        print(f"  H: max |relative deviation| = {np.nanmax(rel_H):.4e}")
        print(f"  H: fraction within combined 1-sigma = {np.nanmean(within_1s):.2%}")

        fig_iso, ax_iso = plt.subplots(1, 2, figsize=(12, 5))
        ax_iso[0].plot(1/a_h, H_txt, 'k-', label='text-file (original)')
        ax_iso[0].fill_between(1/a_h, H_txt - sH_txt, H_txt + sH_txt, color='k', alpha=0.15)
        ax_iso[0].plot(1/a_h, H_rw, 'r--', label=f'{ftype} $V$=0')
        ax_iso[0].fill_between(1/a_h, H_rw - sH_rw, H_rw + sH_rw, color='r', alpha=0.15)
        ax_iso[0].set_xscale('log')
        ax_iso[0].set_xlabel(r'$a/a_h$')
        ax_iso[0].set_ylabel(r'$H$')
        ax_iso[0].legend()
        ax_iso[0].set_title('Isotropic consistency')

        ax_iso[1].plot(1/a_h, rel_H, 'b-')
        ax_iso[1].set_xscale('log')
        ax_iso[1].set_yscale('log')
        ax_iso[1].set_xlabel(r'$a/a_h$')
        ax_iso[1].set_ylabel(r'$|H_{\mathrm{rw}} - H_{\mathrm{txt}}| / |H_{\mathrm{txt}}|$')
        ax_iso[1].set_title('Relative deviation')
        fig_iso.tight_layout()

    # ── Harmonics vs per-particle consistency check ──
    if args.check_harmonics:
        print("\n=== Harmonics vs per-particle consistency check ===")
        meta_pp, bins_pp = read_binary(args.check_harmonics)
        if ftype != 'harmonics':
            print("  WARNING: primary file is not harmonics; swapping roles.")
            meta_h, bins_h = meta_pp, bins_pp
            meta_pp, bins_pp = meta, data_bins
        else:
            meta_h, bins_h = meta, data_bins

        check_V_configs = []
        test_dirs = {'z': np.array([0., 0., 1.]), 'x': np.array([1., 0., 0.]),
                     'y': np.array([0., 1., 0.])}
        for dname, dhat in test_dirs.items():
            for Vr in [-2, -1, 0, 1, 2]:
                check_V_configs.append((f'{dname},{Vr:+d}', Vr, dhat))

        test_sigma_indices = [N_ah // 4, N_ah // 2, 3 * N_ah // 4]

        max_rel = {k: 0.0 for k in ['H', 'K', 'F', 'tau']}
        for label, Vr, dhat in check_V_configs:
            for si in test_sigma_indices:
                sig = sigma[si]
                V = Vr * sig * dhat
                r_pp = reweight(meta_pp, bins_pp, V, sig, rho)
                r_sh = reweight_from_harmonics(meta_h, bins_h, V, sig, rho)

                for key in ['H', 'K']:
                    if abs(r_pp[key]) > 1e-300:
                        rel = abs(r_sh[key] - r_pp[key]) / (abs(r_pp[key]) + 1e-300)
                        max_rel[key] = max(max_rel[key], rel)
                for k in range(3):
                    if abs(r_pp['F'][k]) > 1e-300:
                        rel = abs(r_sh['F'][k] - r_pp['F'][k]) / (abs(r_pp['F'][k]) + 1e-300)
                        max_rel['F'] = max(max_rel['F'], rel)
                    if abs(r_pp['tau'][k]) > 1e-300:
                        rel = abs(r_sh['tau'][k] - r_pp['tau'][k]) / (abs(r_pp['tau'][k]) + 1e-300)
                        max_rel['tau'] = max(max_rel['tau'], rel)

        print("  Max relative deviation (SH vs per-particle) over all V configs:")
        for k, v in max_rel.items():
            print(f"    {k:4s}: {v:.4e}")

    # ── Chandrasekhar outer cutoff (influence radius in a_h units) ──
    r_outer_ah = 4.0 * (1.0 + q)**2 / q

    # ── Compute H, K, tau, F for several signed V directions ──
    V_ratios = [-2, -1, 0, 1, 2]
    directions = {
        r'$\hat z$ (perp. to binary)':          np.array([0., 0., 1.]),
        r'$\hat x$ (eccentricity)':              np.array([1., 0., 0.]),
        r'$\hat y$ (in-plane, $\perp$ ecc.)':    np.array([0., 1., 0.]),
    }

    results = {}
    first_dir = list(directions.keys())[0]
    for dir_label, dir_hat in directions.items():
        for Vr in V_ratios:
            key = (dir_label, Vr)

            if Vr == 0 and dir_label != first_dir:
                results[key] = results[(first_dir, 0)]
                continue

            H_arr      = np.empty(N_ah)
            sH_arr     = np.empty(N_ah)
            K_arr      = np.empty(N_ah)
            sK_arr     = np.empty(N_ah)
            F_arr      = np.empty((N_ah, 3))
            sF_arr     = np.empty((N_ah, 3))
            tau_arr    = np.empty((N_ah, 3))
            stau_arr   = np.empty((N_ah, 3))
            varpi_dot_arr  = np.empty(N_ah)
            svarpi_dot_arr = np.empty(N_ah)
            Q_arr          = np.empty(N_ah)
            sQ_arr         = np.empty(N_ah)
            P_x_arr        = np.empty(N_ah)
            sP_x_arr       = np.empty(N_ah)
            P_y_arr        = np.empty(N_ah)
            sP_y_arr       = np.empty(N_ah)
            P_z_arr        = np.empty(N_ah)
            sP_z_arr       = np.empty(N_ah)
            R_x_arr        = np.empty(N_ah)
            sR_x_arr       = np.empty(N_ah)
            R_y_arr        = np.empty(N_ah)
            sR_y_arr       = np.empty(N_ah)
            F_total_arr    = np.empty((N_ah, 3))
            P_comp_total_arr = np.empty((N_ah, 3))

            for i, sig in enumerate(sigma):
                V = Vr * sig * dir_hat
                r = _reweight(V, sig)
                H_arr[i]      = r['H']
                sH_arr[i]     = r['sH']
                K_arr[i]      = r['K']
                sK_arr[i]     = r['sK']
                F_arr[i]      = r['F']
                sF_arr[i]     = r['sF']
                tau_arr[i]    = r['tau']
                stau_arr[i]   = r['stau']
                varpi_dot_arr[i]  = r['varpi_dot']
                svarpi_dot_arr[i] = r['svarpi_dot']
                Q_arr[i]          = r['Q']
                sQ_arr[i]         = r['sQ']
                P_x_arr[i]        = r['P_x']
                sP_x_arr[i]       = r['sP_x']
                P_y_arr[i]        = r['P_y']
                sP_y_arr[i]       = r['sP_y']
                P_z_arr[i]        = r['P_z']
                sP_z_arr[i]       = r['sP_z']
                R_x_arr[i]        = r['R_x']
                sR_x_arr[i]       = r['sR_x']
                R_y_arr[i]        = r['R_y']
                sR_y_arr[i]       = r['sR_y']

                V_tilde = abs(Vr)
                if V_tilde > 1e-15:
                    xi_i = np.log(a_h[i])
                    J = chandrasekhar_decel_integral(V_tilde, xi_i, q,
                                                    r_outer_ah)
                    V_hat = V / (V_tilde * sig)
                    F_Ch = (4.0 * np.pi * rho / sig**2) * J * V_hat
                else:
                    F_Ch = np.zeros(3)
                F_total_arr[i] = r['F'] + F_Ch
                if abs(r['P']) > 0:
                    P_comp_total_arr[i] = -(mu / (2 * sig)) * F_total_arr[i] / r['P']
                else:
                    P_comp_total_arr[i] = np.nan

            results[key] = dict(
                H=H_arr, sH=sH_arr, K=K_arr, sK=sK_arr,
                F=F_arr, sF=sF_arr, tau=tau_arr, stau=stau_arr,
                varpi_dot=varpi_dot_arr, svarpi_dot=svarpi_dot_arr,
                Q=Q_arr, sQ=sQ_arr,
                P_x=P_x_arr, sP_x=sP_x_arr, P_y=P_y_arr, sP_y=sP_y_arr,
                P_z=P_z_arr, sP_z=sP_z_arr,
                R_x=R_x_arr, sR_x=sR_x_arr, R_y=R_y_arr, sR_y=sR_y_arr,
                F_total=F_total_arr, P_comp_total=P_comp_total_arr,
            )
            print(f"  Computed: V/sigma={Vr:+d}, dir={dir_label}")

    # ── Compute for general-direction V/sigma vectors ──
    cart_configs = [
        (np.array([1., 2., 3.]),  r'$V/\sigma=(1,2,3)$'),
        (np.array([1., 2., -3.]), r'$V/\sigma=(1,2,\!-\!3)$'),
    ]
    cart_results = []
    for V_cart, cart_label in cart_configs:
        V_cart_mag = np.linalg.norm(V_cart)
        H_c      = np.empty(N_ah);  sH_c     = np.empty(N_ah)
        K_c      = np.empty(N_ah);  sK_c     = np.empty(N_ah)
        F_c      = np.empty((N_ah, 3));  sF_c   = np.empty((N_ah, 3))
        tau_c    = np.empty((N_ah, 3));  stau_c = np.empty((N_ah, 3))
        vd_c     = np.empty(N_ah);  svd_c    = np.empty(N_ah)
        Q_c      = np.empty(N_ah);  sQ_c     = np.empty(N_ah)
        Px_c     = np.empty(N_ah);  sPx_c    = np.empty(N_ah)
        Py_c     = np.empty(N_ah);  sPy_c    = np.empty(N_ah)
        Pz_c     = np.empty(N_ah);  sPz_c    = np.empty(N_ah)
        Rx_c     = np.empty(N_ah);  sRx_c    = np.empty(N_ah)
        Ry_c     = np.empty(N_ah);  sRy_c    = np.empty(N_ah)
        Ftot_c   = np.empty((N_ah, 3))
        Ptot_c   = np.empty((N_ah, 3))

        for i, sig in enumerate(sigma):
            V = sig * V_cart
            r = _reweight(V, sig)
            H_c[i]    = r['H'];      sH_c[i]   = r['sH']
            K_c[i]    = r['K'];      sK_c[i]   = r['sK']
            F_c[i]    = r['F'];      sF_c[i]   = r['sF']
            tau_c[i]  = r['tau'];    stau_c[i] = r['stau']
            vd_c[i]   = r['varpi_dot'];  svd_c[i] = r['svarpi_dot']
            Q_c[i]    = r['Q'];      sQ_c[i]   = r['sQ']
            Px_c[i]   = r['P_x'];   sPx_c[i]  = r['sP_x']
            Py_c[i]   = r['P_y'];   sPy_c[i]  = r['sP_y']
            Pz_c[i]   = r['P_z'];   sPz_c[i]  = r['sP_z']
            Rx_c[i]   = r['R_x'];   sRx_c[i]  = r['sR_x']
            Ry_c[i]   = r['R_y'];   sRy_c[i]  = r['sR_y']

            xi_i = np.log(a_h[i])
            J = chandrasekhar_decel_integral(V_cart_mag, xi_i, q, r_outer_ah)
            V_hat = V_cart / V_cart_mag
            F_Ch = (4.0 * np.pi * rho / sig**2) * J * V_hat
            Ftot_c[i] = r['F'] + F_Ch
            if abs(r['P']) > 0:
                Ptot_c[i] = -(mu / (2 * sig)) * Ftot_c[i] / r['P']
            else:
                Ptot_c[i] = np.nan

        cart_results.append(dict(
            label=cart_label,
            H=H_c, sH=sH_c, K=K_c, sK=sK_c,
            F=F_c, sF=sF_c, tau=tau_c, stau=stau_c,
            varpi_dot=vd_c, svarpi_dot=svd_c,
            Q=Q_c, sQ=sQ_c,
            P_x=Px_c, sP_x=sPx_c, P_y=Py_c, sP_y=sPy_c,
            P_z=Pz_c, sP_z=sPz_c,
            R_x=Rx_c, sR_x=sRx_c, R_y=Ry_c, sR_y=sRy_c,
            F_total=Ftot_c, P_comp_total=Ptot_c,
        ))
        print(f"  Computed: {cart_label}")

    def _label(Vr, dir_label):
        if Vr == 0:
            return '$V=0$'
        return f'$V/\\sigma={Vr:+d}$, {dir_label}'

    # ── Plot H(a/a_h) ──
    fig_H, ax_H = plt.subplots(figsize=(7, 5))
    seen_zero = False
    for (dir_label, Vr), res in results.items():
        if Vr == 0:
            if seen_zero: continue
            seen_zero = True
        ax_H.plot(1/a_h, res['H'], label=_label(Vr, dir_label))
        ax_H.fill_between(1/a_h, res['H'] - res['sH'],
                          res['H'] + res['sH'], alpha=0.08)
    for rc in cart_results:
        ax_H.plot(1/a_h, rc['H'], '--', label=rc['label'])
        ax_H.fill_between(1/a_h, rc['H'] - rc['sH'],
                          rc['H'] + rc['sH'], alpha=0.08)
    ax_H.set_xscale('log')
    ax_H.set_xlabel(r'$a/a_h$')
    ax_H.set_ylabel(r'$H$')
    ax_H.legend(fontsize=7)
    ax_H.set_title(f'Hardening rate (q={q}, e={e_ecc})')
    fig_H.tight_layout()

    # ── Plot K(a/a_h) ──
    fig_K, ax_K = plt.subplots(figsize=(7, 5))
    seen_zero = False
    for (dir_label, Vr), res in results.items():
        if Vr == 0:
            if seen_zero: continue
            seen_zero = True
        ax_K.plot(1/a_h, res['K'], label=_label(Vr, dir_label))
        ax_K.fill_between(1/a_h, res['K'] - res['sK'],
                          res['K'] + res['sK'], alpha=0.08)
    for rc in cart_results:
        ax_K.plot(1/a_h, rc['K'], '--', label=rc['label'])
        ax_K.fill_between(1/a_h, rc['K'] - rc['sK'],
                          rc['K'] + rc['sK'], alpha=0.08)
    ax_K.set_xscale('log')
    ax_K.set_xlabel(r'$a/a_h$')
    ax_K.set_ylabel(r'$K$')
    ax_K.legend(fontsize=7)
    ax_K.set_title(f'Eccentricity growth (q={q}, e={e_ecc})')
    fig_K.tight_layout()

    # ── Plot varpi_dot(a/a_h) ──
    fig_varpi, ax_varpi = plt.subplots(figsize=(7, 5))
    seen_zero = False
    for (dir_label, Vr), res in results.items():
        if Vr == 0:
            if seen_zero: continue
            seen_zero = True
        ax_varpi.plot(1/a_h, res['varpi_dot'], label=_label(Vr, dir_label))
        ax_varpi.fill_between(1/a_h, res['varpi_dot'] - res['svarpi_dot'],
                              res['varpi_dot'] + res['svarpi_dot'], alpha=0.08)
    for rc in cart_results:
        ax_varpi.plot(1/a_h, rc['varpi_dot'], '--', label=rc['label'])
        ax_varpi.fill_between(1/a_h, rc['varpi_dot'] - rc['svarpi_dot'],
                              rc['varpi_dot'] + rc['svarpi_dot'], alpha=0.08)
    ax_varpi.set_xscale('log')
    ax_varpi.set_xlabel(r'$a/a_h$')
    ax_varpi.set_ylabel(r'$\dot\varpi$ [$\rho\sqrt{Ga^3/M}$]')
    ax_varpi.legend(fontsize=7)
    ax_varpi.set_title(f'Precession rate (q={q}, e={e_ecc})')
    fig_varpi.tight_layout()

    # ── Plot Q(a/a_h) ──
    fig_tQ, ax_tQ = plt.subplots(figsize=(7, 5))
    seen_zero = False
    for (dir_label, Vr), res in results.items():
        if Vr == 0:
            if seen_zero: continue
            seen_zero = True
        ax_tQ.plot(1/a_h, res['Q'], label=_label(Vr, dir_label))
        ax_tQ.fill_between(1/a_h, res['Q'] - res['sQ'],
                           res['Q'] + res['sQ'], alpha=0.08)
    for rc in cart_results:
        ax_tQ.plot(1/a_h, rc['Q'], '--', label=rc['label'])
        ax_tQ.fill_between(1/a_h, rc['Q'] - rc['sQ'],
                           rc['Q'] + rc['sQ'], alpha=0.08)
    ax_tQ.set_xscale('log')
    ax_tQ.set_xlabel(r'$a/a_h$')
    ax_tQ.set_ylabel(r'$Q$')
    ax_tQ.legend(fontsize=7)
    ax_tQ.set_title(f'$Q$ (q={q}, e={e_ecc})')
    fig_tQ.tight_layout()

    # ── Plot R_x and R_y (a/a_h) ──
    fig_R, axes_R = plt.subplots(1, 2, figsize=(14, 5))
    R_keys = ['R_x', 'R_y']
    R_err_keys = ['sR_x', 'sR_y']
    R_labels = [r'$R_{\hat e}$', r'$R_{\hat n}$']
    for k, ax in enumerate(axes_R):
        seen_zero = False
        for (dir_label, Vr), res in results.items():
            if Vr == 0:
                if seen_zero:
                    continue
                seen_zero = True
            Rk  = res[R_keys[k]]
            sRk = res[R_err_keys[k]]
            ax.plot(1/a_h, Rk, label=_label(Vr, dir_label))
            ax.fill_between(1/a_h, Rk - sRk, Rk + sRk, alpha=0.08)
        for rc in cart_results:
            Rk_c  = rc[R_keys[k]]
            sRk_c = rc[R_err_keys[k]]
            ax.plot(1/a_h, Rk_c, '--', label=rc['label'])
            ax.fill_between(1/a_h, Rk_c - sRk_c, Rk_c + sRk_c, alpha=0.08)
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'$a/a_h$')
        ax.set_ylabel(R_labels[k])
        ax.legend(fontsize=7)
    fig_R.suptitle(f'Rotation parameter (q={q}, e={e_ecc})', fontsize=11)
    fig_R.tight_layout()

    # ── Plot P_x, P_y, P_z (a/a_h) ──
    fig_P, axes_P = plt.subplots(1, 3, figsize=(18, 5))
    Pcomp_keys = ['P_x', 'P_y', 'P_z']
    Pcomp_err_keys = ['sP_x', 'sP_y', 'sP_z']
    Pcomp_labels = [r'$P_{\hat e}$', r'$P_{\hat n}$', r'$P_{\hat L}$']
    line_colors_P = {}
    for k, ax in enumerate(axes_P):
        seen_zero = False
        for (dir_label, Vr), res in results.items():
            if Vr == 0:
                if seen_zero:
                    continue
                seen_zero = True
            Pk  = res[Pcomp_keys[k]]
            sPk = res[Pcomp_err_keys[k]]
            line, = ax.plot(1/a_h, Pk, label=_label(Vr, dir_label))
            ax.fill_between(1/a_h, Pk - sPk, Pk + sPk, alpha=0.08)
            if k == 0:
                line_colors_P[(dir_label, Vr)] = line.get_color()
        for (dir_label, Vr), res in results.items():
            if Vr == 0:
                continue
            lbl_tot = _label(Vr, dir_label) + ' +Ch'
            ax.plot(1/a_h, res['P_comp_total'][:, k], '--',
                    color=line_colors_P[(dir_label, Vr)],
                    label=lbl_tot if k == 0 else None)
        for rc in cart_results:
            Pk_c  = rc[Pcomp_keys[k]]
            sPk_c = rc[Pcomp_err_keys[k]]
            ax.plot(1/a_h, Pk_c, '--', label=rc['label'])
            ax.fill_between(1/a_h, Pk_c - sPk_c, Pk_c + sPk_c, alpha=0.08)
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'$a/a_h$')
        ax.set_ylabel(Pcomp_labels[k])
        ax.legend(fontsize=6)
    fig_P.suptitle(f'Acceleration parameter (q={q}, e={e_ecc})', fontsize=11)
    fig_P.tight_layout()

    # ── Plot signed F components: one figure per V direction ──
    comp_names = ['x', 'y', 'z']
    for dir_label, dir_hat in directions.items():
        fig_F, axes_F = plt.subplots(1, 3, figsize=(16, 5))
        seen_zero = False
        line_colors = {}
        for Vr in V_ratios:
            res = results[(dir_label, Vr)]
            if Vr == 0:
                if seen_zero: continue
                seen_zero = True
            lbl = _label(Vr, dir_label)
            for k, ax in enumerate(axes_F):
                Fk  = res['F'][:, k]
                sFk = res['sF'][:, k]
                line, = ax.plot(1/a_h, Fk, label=lbl)
                ax.fill_between(1/a_h, Fk - sFk, Fk + sFk, alpha=0.08)
                if k == 0:
                    line_colors[Vr] = line.get_color()
        for Vr in V_ratios:
            if Vr == 0:
                continue
            res = results[(dir_label, Vr)]
            lbl_tot = _label(Vr, dir_label) + ' +Ch'
            for k, ax in enumerate(axes_F):
                ax.plot(1/a_h, res['F_total'][:, k], '--',
                        color=line_colors[Vr], label=lbl_tot if k == 0 else None)
        for rc in cart_results:
            for k, ax in enumerate(axes_F):
                Fk_c  = rc['F'][:, k]
                sFk_c = rc['sF'][:, k]
                ax.plot(1/a_h, Fk_c, '--',
                        label=rc['label'] if k == 0 else None)
                ax.fill_between(1/a_h, Fk_c - sFk_c, Fk_c + sFk_c,
                                alpha=0.08)
        for k, ax in enumerate(axes_F):
            ax.axhline(0, color='grey', lw=0.5)
            ax.set_xscale('log')
            ax.set_xlabel(r'$a/a_h$')
            ax.set_ylabel(f'$F_{{{comp_names[k]}}}$ [$GM\\rho a$]')
        axes_F[0].legend(fontsize=6)
        fig_F.suptitle(
            f'Signed force components, $V$ along {dir_label} (q={q}, e={e_ecc})',
            fontsize=11)
        fig_F.tight_layout()

    # ── Plot signed tau components: one figure per V direction ──
    for dir_label, dir_hat in directions.items():
        fig_tau, axes_tau = plt.subplots(1, 3, figsize=(16, 5))
        seen_zero = False
        for Vr in V_ratios:
            res = results[(dir_label, Vr)]
            if Vr == 0:
                if seen_zero: continue
                seen_zero = True
            lbl = _label(Vr, dir_label)
            for k, ax in enumerate(axes_tau):
                tk  = res['tau'][:, k]
                stk = res['stau'][:, k]
                ax.plot(1/a_h, tk, label=lbl)
                ax.fill_between(1/a_h, tk - stk, tk + stk, alpha=0.08)
        for rc in cart_results:
            for k, ax in enumerate(axes_tau):
                tk_c  = rc['tau'][:, k]
                stk_c = rc['stau'][:, k]
                ax.plot(1/a_h, tk_c, '--',
                        label=rc['label'] if k == 0 else None)
                ax.fill_between(1/a_h, tk_c - stk_c, tk_c + stk_c,
                                alpha=0.08)
        for k, ax in enumerate(axes_tau):
            ax.axhline(0, color='grey', lw=0.5)
            ax.set_xscale('log')
            ax.set_xlabel(r'$a/a_h$')
            ax.set_ylabel(f'$\\tau_{{{comp_names[k]}}}$ [$GM\\rho a^2$]')
        axes_tau[0].legend(fontsize=6)
        fig_tau.suptitle(
            f'Signed torque components, $V$ along {dir_label} (q={q}, e={e_ecc})',
            fontsize=11)
        fig_tau.tight_layout()

    # ── Chandrasekhar check: |F_parallel| vs |V|/sigma ─────
    ch_directions = {
        r'$\hat z$ (perp.)':    np.array([0., 0., 1.]),
        r'$\hat x$ (ecc.)':     np.array([1., 0., 0.]),
        r'$\hat y$ (in-plane)': np.array([0., 1., 0.]),
    }
    test_ah_indices = [N_ah // 5, 2 * N_ah // 5, 3 * N_ah // 5, 4 * N_ah // 5]
    V_test_ratios = np.logspace(np.log10(0.5), np.log10(15), 25)

    fig_ch, axes_ch = plt.subplots(2, len(ch_directions), figsize=(20, 10),
                                    sharex='col')

    for col, (dir_label, ch_dir) in enumerate(ch_directions.items()):
        ax_F = axes_ch[0, col]
        ax_R = axes_ch[1, col]

        for idx in test_ah_indices:
            sig = sigma[idx]
            ah_val = a_h[idx]
            F_par_sim = []
            sF_par_sim = []
            for Vr in V_test_ratios:
                V = Vr * sig * ch_dir
                r = _reweight(V, sig)
                F_par_sim.append(r['F'] @ ch_dir)
                sF_par_sim.append(np.sqrt(np.sum(r['sF']**2 * ch_dir**2)))
            F_par_sim = np.array(F_par_sim)
            sF_par_sim = np.array(sF_par_sim)

            xi_i = np.log(ah_val)
            J_arr = np.array([
                chandrasekhar_decel_integral(Vr, xi_i, q, r_outer_ah)
                for Vr in V_test_ratios])
            F_ch = (4.0 * np.pi * rho / sig**2) * J_arr
            ln_L_arr = np.array([
                _ln_lambda(max(Vr, 1e-3), xi_i, q, r_outer_ah)
                for Vr in V_test_ratios])
            mean_lnL = np.mean(ln_L_arr)

            lbl_sim = f'$a/a_h={1/ah_val:.2g}$'
            lbl_ch  = f'Ch. (exact), $\\langle\\ln\\Lambda\\rangle\\approx{mean_lnL:.1f}$'

            line = ax_F.errorbar(V_test_ratios, np.abs(F_par_sim),
                                  yerr=sF_par_sim,
                                  fmt='o-', ms=3, capsize=2, label=lbl_sim)
            c = line[0].get_color()
            ax_F.plot(V_test_ratios, np.abs(F_ch), '--', color=c, alpha=0.6,
                      label=lbl_ch)

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio  = np.where(np.abs(F_ch) > 0, F_par_sim / F_ch, np.nan)
                sratio = np.where(np.abs(F_ch) > 0, sF_par_sim / np.abs(F_ch), np.nan)
            ax_R.errorbar(V_test_ratios, ratio, yerr=sratio,
                          fmt='o-', ms=3, capsize=2, color=c, label=lbl_sim)

        ax_F.set_xscale('log')
        ax_F.set_yscale('log')
        ax_F.set_ylabel(r'$|F_\parallel|$ [$GM\rho a$]')
        ax_F.set_title(f'$V$ along {dir_label}')
        ax_F.legend(fontsize=6, ncol=2)

        ax_R.axhline(1, color='grey', lw=0.8, ls='--')
        ax_R.set_xscale('log')
        ax_R.set_xlabel(r'$|V| / \sigma$')
        ax_R.set_ylabel(r'$F_{\mathrm{sim}} / F_{\mathrm{Chandrasekhar}}$')
        ax_R.set_ylim(0, 3)
        ax_R.legend(fontsize=6)

    fig_ch.suptitle(
        f'Chandrasekhar comparison (q={q}, e={e_ecc})\n'
        r'$\ln\Lambda = \ln(b_{\max}(V) / a)$  with $a=1$',
        fontsize=11)
    fig_ch.tight_layout()

    if not args.no_show:
        plt.show()
