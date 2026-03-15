import numpy as np
import matplotlib.pyplot as plt
import os
import re

###################################################
# Configuration
###################################################

data_dir = "../Data/results-precession-soft"

# Detect data files and extract q,e values
files = [f for f in os.listdir(data_dir) if f.startswith("q=") and f.endswith(".txt")]
pattern = re.compile(r"q=([0-9.]+)_e=([0-9.]+)\_Tcut=100000000000.txt")
q_list = []
e_list = []
parsed_files = []

for f in files:
    match = pattern.match(f)
    if match:
        q_val = float(match.group(1))
        e_val = float(match.group(2))
        q_list.append(q_val)
        e_list.append(e_val)
        parsed_files.append((q_val, e_val, f))

# Build sorted unique q and e arrays
q_values = np.array(sorted(set(q_list)))
e_values = np.array(sorted(set(e_list)))

a_over_a_h_target = 1

rp_max = 5
rho = 1

###################################################
# Functions (copied from your original script)
###################################################

def b_max(v):
    return rp_max * np.sqrt(1 + 2 / (v**2 * rp_max))

def f(v, sigma):
    return np.sqrt(2/np.pi) * (v**2/sigma**3) * np.exp(-v**2 / (2*sigma**2))

###################################################
# Storage arrays
###################################################
P_grid = np.zeros((len(e_values), len(q_values)))
Fx_grid = np.zeros((len(e_values), len(q_values)))
Fy_grid = np.zeros((len(e_values), len(q_values)))
Fz_grid = np.zeros((len(e_values), len(q_values)))
taux_grid = np.zeros((len(e_values), len(q_values)))
tauy_grid = np.zeros((len(e_values), len(q_values)))
tauz_grid = np.zeros((len(e_values), len(q_values)))
H_grid = np.zeros((len(e_values), len(q_values)))
K_grid = np.zeros((len(e_values), len(q_values)))
Px_grid = np.zeros((len(e_values), len(q_values)))
Py_grid = np.zeros((len(e_values), len(q_values)))
varpi_dot_grid = np.zeros((len(e_values), len(q_values)))
Q_grid = np.zeros((len(e_values), len(q_values)))

sP_grid = np.zeros((len(e_values), len(q_values)))
sFx_grid = np.zeros((len(e_values), len(q_values)))
sFy_grid = np.zeros((len(e_values), len(q_values)))
sFz_grid = np.zeros((len(e_values), len(q_values)))
staux_grid = np.zeros((len(e_values), len(q_values)))
stauy_grid = np.zeros((len(e_values), len(q_values)))
stauz_grid = np.zeros((len(e_values), len(q_values)))
sH_grid = np.zeros((len(e_values), len(q_values)))
sK_grid = np.zeros((len(e_values), len(q_values)))
sPx_grid = np.zeros((len(e_values), len(q_values)))
sPy_grid = np.zeros((len(e_values), len(q_values)))
svarpi_dot_grid = np.zeros((len(e_values), len(q_values)))
sQ_grid = np.zeros((len(e_values), len(q_values)))

###################################################
# Main loop over (q, e)
###################################################
for q, e, filename in parsed_files:
    iq = np.where(q_values == q)[0][0]
    ie = np.where(e_values == e)[0][0]

    mu = q / (1 + q)**2

    try:
        v, DeltaE, sDeltaE, DeltaT, sDeltaT, Deltavx, sDeltavx, Deltavy, sDeltavy, \
        Deltavz, sDeltavz, DeltaLx, sDeltaLx, DeltaLy, sDeltaLy, DeltaLz, sDeltaLz, \
        Delta_varpi, sDelta_varpi, Nresolved = np.loadtxt(os.path.join(data_dir, filename), unpack=True)
    except:
        print(f"File not found or unreadable: {filename}")
        Fx_grid[ie, iq] = np.nan
        Fy_grid[ie, iq] = np.nan
        Px_grid[ie, iq] = np.nan
        Py_grid[ie, iq] = np.nan
        continue

    # Hardening radii sampling
    N_a_h = 50
    a_h = np.logspace(3, -2, N_a_h)
    sigma = np.sqrt(mu / (4*a_h))

    Pv = - np.pi * b_max(v)**2 * rho * v * DeltaE
    Fv_x = - np.pi * b_max(v)**2 * rho * v * Deltavx
    Fv_y = - np.pi * b_max(v)**2 * rho * v * Deltavy
    Fv_z = - np.pi * b_max(v)**2 * rho * v * Deltavz
    tauv_x = - np.pi * b_max(v)**2 * rho * v * DeltaLx
    tauv_y = - np.pi * b_max(v)**2 * rho * v * DeltaLy
    tauv_z = - np.pi * b_max(v)**2 * rho * v * DeltaLz
    Hv = (2 * np.pi * v**2 * b_max(v)**2) * DeltaE / mu
    varpi_dot_v = np.pi * b_max(v)**2 * rho * v * Delta_varpi

    sPv = - np.pi * b_max(v)**2 * rho * v * sDeltaE
    sFv_x = - np.pi * b_max(v)**2 * rho * v * sDeltavx
    sFv_y = - np.pi * b_max(v)**2 * rho * v * sDeltavy
    sFv_z = - np.pi * b_max(v)**2 * rho * v * sDeltavz
    stauv_x = - np.pi * b_max(v)**2 * rho * v * sDeltaLx
    stauv_y = - np.pi * b_max(v)**2 * rho * v * sDeltaLy
    stauv_z = - np.pi * b_max(v)**2 * rho * v * sDeltaLz
    sHv = (2 * np.pi * v**2 * b_max(v)**2) * sDeltaE / mu
    svarpi_dot_v = np.pi * b_max(v)**2 * rho * v * sDelta_varpi

    # Prepare integration arrays
    v0 = np.hstack([0, v])
    f0 = np.vstack([np.zeros((1, N_a_h)), f(v[:, np.newaxis], sigma[np.newaxis, :])])

    Pv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Pv[:, np.newaxis], (1, N_a_h))])
    Fv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_x[:, np.newaxis], (1, N_a_h))])
    Fv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_y[:, np.newaxis], (1, N_a_h))])
    Fv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_z[:, np.newaxis], (1, N_a_h))])
    tauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_x[:, np.newaxis], (1, N_a_h))])
    tauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_y[:, np.newaxis], (1, N_a_h))])
    tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
    Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
    varpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(varpi_dot_v[:, np.newaxis], (1, N_a_h))])
    H_integrand0 = np.vstack([np.zeros((1, N_a_h)), (sigma[np.newaxis, :] / v[:, np.newaxis]) * f(v[:, np.newaxis], sigma[np.newaxis, :])])

    sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])
    sFv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_x[:, np.newaxis], (1, N_a_h))])
    sFv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_y[:, np.newaxis], (1, N_a_h))])
    sFv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_z[:, np.newaxis], (1, N_a_h))])
    stauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_x[:, np.newaxis], (1, N_a_h))])
    stauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_y[:, np.newaxis], (1, N_a_h))])
    stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])
    sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])
    svarpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(svarpi_dot_v[:, np.newaxis], (1, N_a_h))])

    # Integrations
    P = np.trapezoid(Pv0 * f0, x=v0, axis=0)
    F_x = np.trapezoid(Fv_x0 * f0, x=v0, axis=0)
    F_y = np.trapezoid(Fv_y0 * f0, x=v0, axis=0)
    F_z = np.trapezoid(Fv_z0 * f0, x=v0, axis=0)
    tau_x = np.trapezoid(tauv_x0 * f0, x=v0, axis=0)
    tau_y = np.trapezoid(tauv_y0 * f0, x=v0, axis=0)
    tau_z = np.trapezoid(tauv_z0 * f0, x=v0, axis=0)
    H = np.trapezoid(Hv0 * H_integrand0, x=v0, axis=0)
    varpi_dot = np.trapezoid(varpi_dot_v0 * f0, x=v0, axis=0)
    if (e == 0):
        K = np.full_like(P, np.nan)
    else:
        K = - (1-e**2)/(2*e) + np.sqrt(1-e**2)/(2*e) * tau_z/P
    P_x = - (mu / (2*sigma)) * F_x / P
    P_y = - (mu / (2*sigma)) * F_y / P
    Q = - (mu / 2) * varpi_dot / P

    # Calculatre uncertainties
    sP_integrand = sPv0 * f0
    sF_x_integrand = sFv_x0 * f0
    sF_y_integrand = sFv_y0 * f0
    sF_z_integrand = sFv_z0 * f0
    stau_x_integrand = stauv_x0 * f0
    stau_y_integrand = stauv_y0 * f0
    stau_z_integrand = stauv_z0 * f0
    sH_integrand = sHv0 * H_integrand0
    svarpi_dot_integrand = svarpi_dot_v0 * f0

    weights = np.empty_like(v0)
    weights[0] = (v0[1] - v0[0]) / 2
    weights[-1] = (v0[-1] - v0[-2]) / 2
    weights[1:-1] = (v0[2:] - v0[:-2]) / 2

    sP = np.sqrt(np.sum((weights[:, np.newaxis] * sP_integrand)**2, axis=0))
    sF_x = np.sqrt(np.sum((weights[:, np.newaxis] * sF_x_integrand)**2, axis=0))
    sF_y = np.sqrt(np.sum((weights[:, np.newaxis] * sF_y_integrand)**2, axis=0))
    sF_z = np.sqrt(np.sum((weights[:, np.newaxis] * sF_z_integrand)**2, axis=0))
    stau_x = np.sqrt(np.sum((weights[:, np.newaxis] * stau_x_integrand)**2, axis=0))
    stau_y = np.sqrt(np.sum((weights[:, np.newaxis] * stau_y_integrand)**2, axis=0))
    stau_z = np.sqrt(np.sum((weights[:, np.newaxis] * stau_z_integrand)**2, axis=0))
    sH = np.sqrt(np.sum((weights[:, np.newaxis] * sH_integrand)**2, axis=0))
    svarpi_dot = np.sqrt(np.sum((weights[:, np.newaxis] * svarpi_dot_integrand)**2, axis=0))
    if (e == 0):
        sK = np.full_like(K, np.nan)
    else:
        sK = np.sqrt(1-e**2)/(2*e) * np.abs(tau_z/P) * np.sqrt((stau_z/tau_z)**2 + (sP/P)**2)
    sP_x = (mu / (2*sigma)) * np.abs(F_x / P) * np.sqrt((sF_x/F_x)**2 + (sP/P)**2)
    sP_y = (mu / (2*sigma)) * np.abs(F_y / P) * np.sqrt((sF_y/F_y)**2 + (sP/P)**2)
    sQ = - (mu / 2) * (varpi_dot / P) * np.sqrt((svarpi_dot/varpi_dot)**2 + (sP/P)**2)

    # Pick requested a_h value by interpolation
    P_val = np.interp(1/a_over_a_h_target, a_h[::-1], P[::-1])
    Fx_val = np.interp(1/a_over_a_h_target, a_h[::-1], F_x[::-1])
    Fy_val = np.interp(1/a_over_a_h_target, a_h[::-1], F_y[::-1])
    Fz_val = np.interp(1/a_over_a_h_target, a_h[::-1], F_z[::-1])
    taux_val = np.interp(1/a_over_a_h_target, a_h[::-1], tau_x[::-1])
    tauy_val = np.interp(1/a_over_a_h_target, a_h[::-1], tau_y[::-1])
    tauz_val = np.interp(1/a_over_a_h_target, a_h[::-1], tau_z[::-1])
    H_val = np.interp(1/a_over_a_h_target, a_h[::-1], H[::-1])
    K_val = np.interp(1/a_over_a_h_target, a_h[::-1], K[::-1])
    Px_val = np.interp(1/a_over_a_h_target, a_h[::-1], P_x[::-1])
    Py_val = np.interp(1/a_over_a_h_target, a_h[::-1], P_y[::-1])
    varpi_dot_val = np.interp(1/a_over_a_h_target, a_h[::-1], varpi_dot[::-1])
    Q_val = np.interp(1/a_over_a_h_target, a_h[::-1], Q[::-1])

    sP_val = np.interp(1/a_over_a_h_target, a_h[::-1], sP[::-1])
    sFx_val = np.interp(1/a_over_a_h_target, a_h[::-1], sF_x[::-1])
    sFy_val = np.interp(1/a_over_a_h_target, a_h[::-1], sF_y[::-1])
    sFz_val = np.interp(1/a_over_a_h_target, a_h[::-1], sF_z[::-1])
    staux_val = np.interp(1/a_over_a_h_target, a_h[::-1], stau_x[::-1])
    stauy_val = np.interp(1/a_over_a_h_target, a_h[::-1], stau_y[::-1])
    stauz_val = np.interp(1/a_over_a_h_target, a_h[::-1], stau_z[::-1])
    sH_val = np.interp(1/a_over_a_h_target, a_h[::-1], sH[::-1])
    sK_val = np.interp(1/a_over_a_h_target, a_h[::-1], sK[::-1])
    sPx_val = np.interp(1/a_over_a_h_target, a_h[::-1], sP_x[::-1])
    sPy_val = np.interp(1/a_over_a_h_target, a_h[::-1], sP_y[::-1])
    svarpi_dot_val = np.interp(1/a_over_a_h_target, a_h[::-1], svarpi_dot[::-1])
    sQ_val = np.interp(1/a_over_a_h_target, a_h[::-1], sQ[::-1])

    P_grid[ie, iq] = P_val
    Fx_grid[ie, iq] = Fx_val
    Fy_grid[ie, iq] = Fy_val
    Fz_grid[ie, iq] = Fz_val
    taux_grid[ie, iq] = taux_val
    tauy_grid[ie, iq] = tauy_val
    tauz_grid[ie, iq] = tauz_val
    H_grid[ie, iq] = H_val
    K_grid[ie, iq] = K_val
    Px_grid[ie, iq] = Px_val
    Py_grid[ie, iq] = Py_val
    varpi_dot_grid[ie, iq] = varpi_dot_val
    Q_grid[ie, iq] = Q_val

    sP_grid[ie, iq] = sP_val
    sFx_grid[ie, iq] = sFx_val
    sFy_grid[ie, iq] = sFy_val
    sFz_grid[ie, iq] = sFz_val
    staux_grid[ie, iq] = staux_val
    stauy_grid[ie, iq] = stauy_val
    stauz_grid[ie, iq] = stauz_val
    sH_grid[ie, iq] = sH_val
    sK_grid[ie, iq] = sK_val
    sPx_grid[ie, iq] = sPx_val
    sPy_grid[ie, iq] = sPy_val
    svarpi_dot_grid[ie, iq] = svarpi_dot_val
    sQ_grid[ie, iq] = sQ_val

###################################################
# Plotting
###################################################
Q, E = np.meshgrid(q_values, e_values)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# First plot: Force
axes[0].quiver(Q, E, Fx_grid, Fy_grid, angles='xy', scale_units='xy', scale=10)
axes[0].set_xlabel(r"$q$")
axes[0].set_ylabel(r"$e$")
axes[0].set_title(r"Force $(F_x, F_y)$ at $a/a_h = 10^{-2}$")
axes[0].grid(True)
axes[0].set_xlim(-0.5, 1.1)
axes[0].set_ylim(-0.1, 1)

# Second plot: P-vector
axes[1].quiver(Q, E, Px_grid, Py_grid, angles='xy', scale_units='xy', scale=1)
axes[1].set_xlabel(r"$q$")
axes[1].set_ylabel(r"$e$")
axes[1].set_title(r"P-vector $(P_x, P_y)$ at $a/a_h = 10^{-2}$")
axes[1].grid(True)
axes[1].set_xlim(-0.5, 1.1)
axes[1].set_ylim(-0.1, 1)

fig2, ax2 = plt.subplots(1, 4, figsize=(24, 6))

for ie in range(len(e_values)):
    ax2[0].plot(Q[ie], H_grid[ie], label=f'e={E[ie,0]}')
    ax2[0].fill_between(Q[ie], H_grid[ie]-sH_grid[ie], H_grid[ie]+sH_grid[ie], alpha=0.3)
ax2[0].set_xscale('log')
ax2[0].set_xlabel(r'$q$')
ax2[0].set_ylabel(r'$H$')
ax2[0].legend()

for ie in range(len(e_values)):
    ax2[1].plot(Q[ie], K_grid[ie], label=f'e={E[ie,0]}')
    ax2[1].fill_between(Q[ie], K_grid[ie]-sK_grid[ie], K_grid[ie]+sK_grid[ie], alpha=0.3)
ax2[1].set_xscale('log')
ax2[1].set_xlabel(r'$q$')
ax2[1].set_ylabel(r'$K$')
ax2[1].legend()

for ie in range(len(e_values)):
    ax2[2].plot(Q[ie], varpi_dot_grid[ie], label=f'e={E[ie,0]}')
    ax2[2].fill_between(Q[ie], varpi_dot_grid[ie]-svarpi_dot_grid[ie], varpi_dot_grid[ie]+svarpi_dot_grid[ie], alpha=0.3)
ax2[2].set_xscale('log')
ax2[2].set_xlabel(r'$q$')
ax2[2].set_ylabel(r'$\dot\varpi$ [$\rho\sqrt{Ga^3/M}$]')
ax2[2].legend()

for ie in range(len(e_values)):
    ax2[3].plot(Q[ie], Q_grid[ie], label=f'e={E[ie,0]}')
    ax2[3].fill_between(Q[ie], Q_grid[ie]-sQ_grid[ie], Q_grid[ie]+sQ_grid[ie], alpha=0.3)
ax2[3].set_xscale('log')
ax2[3].set_xlabel(r'$q$')
ax2[3].set_ylabel(r'$Q$')
ax2[3].legend()

plt.tight_layout()
plt.show()