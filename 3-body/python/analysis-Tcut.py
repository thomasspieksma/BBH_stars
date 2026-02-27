import numpy as np
import matplotlib.pyplot as plt
import os
import re

###################################################
# User configuration
###################################################
data_dir = "results-Tcut"

q_fixed = 0.1
e_fixed = 0.6
a_over_a_h_target = 0.1

rp_max = 5
rho = 1

###################################################
# Detect data files with this q,e and extract Tcut
###################################################
pattern = re.compile(r"q=([0-9.]+)_e=([0-9.]+)_Tcut=([0-9]+)\.txt")

Tcuts = []
parsed_files = []

for f in os.listdir(data_dir):
    m = pattern.match(f)
    if m:
        q = float(m.group(1))
        e = float(m.group(2))
        Tcut = int(m.group(3))
        if q == q_fixed and e == e_fixed:
            Tcuts.append(Tcut)
            parsed_files.append((Tcut, f))

# Sort by Tcut
parsed_files = sorted(parsed_files, key=lambda x: x[0])
Tcuts = np.array(sorted(Tcuts))

N = len(Tcuts)
if N == 0:
    raise RuntimeError("No matching q,e,Tcut files found.")

###################################################
# Storage arrays (1D instead of 2D)
###################################################
P_vals  = np.zeros(N)
Fx_vals = np.zeros(N)
Fy_vals = np.zeros(N)
Fz_vals = np.zeros(N)
taux_vals = np.zeros(N)
tauy_vals = np.zeros(N)
tauz_vals = np.zeros(N)
H_vals  = np.zeros(N)
K_vals  = np.zeros(N)
Qx_vals = np.zeros(N)
Qy_vals = np.zeros(N)
tildeQ_vals = np.zeros(N)

sP_vals  = np.zeros(N)
sFx_vals = np.zeros(N)
sFy_vals = np.zeros(N)
sFz_vals = np.zeros(N)
staux_vals = np.zeros(N)
stauy_vals = np.zeros(N)
stauz_vals = np.zeros(N)
sH_vals  = np.zeros(N)
sK_vals  = np.zeros(N)
sQx_vals = np.zeros(N)
sQy_vals = np.zeros(N)
stildeQ_vals = np.zeros(N)

###################################################
# Functions (as in your script)
###################################################
def b_max(v):
    return rp_max * np.sqrt(1 + 2 / (v**2 * rp_max))

def f_MB(v, sigma):
    return np.sqrt(2/np.pi) * (v**2/sigma**3) * np.exp(-v**2 / (2*sigma**2))


###################################################
# Main loop over Tcut
###################################################
for i, (Tcut, filename) in enumerate(parsed_files):

    fullpath = os.path.join(data_dir, filename)

    try:
        v, DeltaE, sDeltaE, DeltaT, sDeltaT, Deltavx, sDeltavx, Deltavy, sDeltavy, \
        Deltavz, sDeltavz, DeltaLx, sDeltaLx, DeltaLy, sDeltaLy, DeltaLz, sDeltaLz, \
        Delta_varpi, sDelta_varpi, Nresolved = np.loadtxt(fullpath, unpack=True)
    except:
        print("File unreadable:", filename)
        for arr in [P_vals, Fx_vals, Fy_vals, Fz_vals, taux_vals, tauy_vals, tauz_vals,
                    H_vals, K_vals, Qx_vals, Qy_vals, tildeQ_vals,
                    sP_vals, sFx_vals, sFy_vals, sFz_vals, staux_vals, stauy_vals, stauz_vals,
                    sH_vals, sK_vals, sQx_vals, sQy_vals, stildeQ_vals]:
            arr[i] = np.nan
        continue

    mu = q_fixed / (1 + q_fixed)**2

    # Hardening radii sampling
    N_a_h = 50
    a_h = np.logspace(3, -2, N_a_h)
    sigma = np.sqrt(mu / (4*a_h))

    # Velocity-dependent quantities
    Pv = - np.pi * b_max(v)**2 * rho * v * DeltaE
    Fv_x = - np.pi * b_max(v)**2 * rho * v * Deltavx
    Fv_y = - np.pi * b_max(v)**2 * rho * v * Deltavy
    Fv_z = - np.pi * b_max(v)**2 * rho * v * Deltavz
    tauv_x = - np.pi * b_max(v)**2 * rho * v * DeltaLx
    tauv_y = - np.pi * b_max(v)**2 * rho * v * DeltaLy
    tauv_z = - np.pi * b_max(v)**2 * rho * v * DeltaLz
    Hv = (2 * np.pi * v**2 * b_max(v)**2) * DeltaE / mu
    varpi_dot_v = np.pi * b_max(v)**2 * rho * v * Delta_varpi

    # Uncertainties
    sPv = - np.pi * b_max(v)**2 * rho * v * sDeltaE
    sFv_x = - np.pi * b_max(v)**2 * rho * v * sDeltavx
    sFv_y = - np.pi * b_max(v)**2 * rho * v * sDeltavy
    sFv_z = - np.pi * b_max(v)**2 * rho * v * sDeltavz
    stauv_x = - np.pi * b_max(v)**2 * rho * v * sDeltaLx
    stauv_y = - np.pi * b_max(v)**2 * rho * v * sDeltaLy
    stauv_z = - np.pi * b_max(v)**2 * rho * v * sDeltaLz
    sHv = (2 * np.pi * v**2 * b_max(v)**2) * sDeltaE / mu
    svarpi_dot_v = np.pi * b_max(v)**2 * rho * v * sDelta_varpi

    # Build v0 arrays (same as your script)
    v0 = np.hstack([0, v])
    f0 = np.vstack([np.zeros((1, N_a_h)),
                    f_MB(v[:, np.newaxis], sigma[np.newaxis, :])])

    Pv0 = np.vstack([np.zeros((1, N_a_h)),
                     np.tile(Pv[:, np.newaxis], (1, N_a_h))])
    # same for all
    Fv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_x[:, np.newaxis], (1, N_a_h))])
    Fv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_y[:, np.newaxis], (1, N_a_h))])
    Fv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_z[:, np.newaxis], (1, N_a_h))])
    tauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_x[:, np.newaxis], (1, N_a_h))])
    tauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_y[:, np.newaxis], (1, N_a_h))])
    tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
    Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
    varpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(varpi_dot_v[:, np.newaxis], (1, N_a_h))])

    H_integrand0 = np.vstack([np.zeros((1, N_a_h)),
                              (sigma[np.newaxis, :] / v[:, np.newaxis]) *
                              f_MB(v[:, np.newaxis], sigma[np.newaxis, :])])

    # Uncertainty integrands
    sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])
    sFv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_x[:, np.newaxis], (1, N_a_h))])
    sFv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_y[:, np.newaxis], (1, N_a_h))])
    sFv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_z[:, np.newaxis], (1, N_a_h))])
    stauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_x[:, np.newaxis], (1, N_a_h))])
    stauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_y[:, np.newaxis], (1, N_a_h))])
    stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])
    sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])
    svarpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(svarpi_dot_v[:, np.newaxis], (1, N_a_h))])

    ###################################################
    # Integrations
    ###################################################
    P = np.trapz(Pv0 * f0, x=v0, axis=0)
    F_x = np.trapz(Fv_x0 * f0, x=v0, axis=0)
    F_y = np.trapz(Fv_y0 * f0, x=v0, axis=0)
    F_z = np.trapz(Fv_z0 * f0, x=v0, axis=0)
    tau_x = np.trapz(tauv_x0 * f0, x=v0, axis=0)
    tau_y = np.trapz(tauv_y0 * f0, x=v0, axis=0)
    tau_z = np.trapz(tauv_z0 * f0, x=v0, axis=0)
    H = np.trapz(Hv0 * H_integrand0, x=v0, axis=0)
    varpi_dot = np.trapz(varpi_dot_v0 * f0, x=v0, axis=0)

    # K(a_h)
    if e_fixed == 0:
        K = np.full_like(P, np.nan)
    else:
        K = -(1 - e_fixed**2)/(2*e_fixed) + np.sqrt(1 - e_fixed**2)/(2*e_fixed) * tau_z/P

    # Q(a_h)
    Q_x = -(mu / (2*sigma)) * F_x / P
    Q_y = -(mu / (2*sigma)) * F_y / P
    tildeQ = -(mu / 2) * varpi_dot / P

    ###################################################
    # Uncertainty propagation
    ###################################################
    weights = np.empty_like(v0)
    weights[0] = (v0[1] - v0[0]) / 2
    weights[-1] = (v0[-1] - v0[-2]) / 2
    weights[1:-1] = (v0[2:] - v0[:-2]) / 2

    sP = np.sqrt(np.sum((weights[:,None] * (sPv0*f0))**2, axis=0))
    sF_x = np.sqrt(np.sum((weights[:,None] * (sFv_x0*f0))**2, axis=0))
    sF_y = np.sqrt(np.sum((weights[:,None] * (sFv_y0*f0))**2, axis=0))
    sF_z = np.sqrt(np.sum((weights[:,None] * (sFv_z0*f0))**2, axis=0))
    stau_x = np.sqrt(np.sum((weights[:,None] * (stauv_x0*f0))**2, axis=0))
    stau_y = np.sqrt(np.sum((weights[:,None] * (stauv_y0*f0))**2, axis=0))
    stau_z = np.sqrt(np.sum((weights[:,None] * (stauv_z0*f0))**2, axis=0))
    sH = np.sqrt(np.sum((weights[:,None] * (sHv0 * H_integrand0))**2, axis=0))
    svarpi_dot = np.sqrt(np.sum((weights[:,None] * (svarpi_dot_v0*f0))**2, axis=0))

    if e_fixed == 0:
        sK = np.full_like(K, np.nan)
    else:
        sK = np.sqrt(1-e_fixed**2)/(2*e_fixed) * np.abs(tau_z/P) * np.sqrt((stau_z/tau_z)**2 + (sP/P)**2)

    sQ_x = (mu / (2*sigma)) * np.abs(F_x / P) * np.sqrt((sF_x/F_x)**2 + (sP/P)**2)
    sQ_y = (mu / (2*sigma)) * np.abs(F_y / P) * np.sqrt((sF_y/F_y)**2 + (sP/P)**2)
    stildeQ = -(mu / 2) * (varpi_dot / P) * np.sqrt((svarpi_dot/varpi_dot)**2 + (sP/P)**2)

    ###################################################
    # Interpolate all quantities at chosen a_h
    ###################################################
    x = 1/a_over_a_h_target

    P_vals[i]  = np.interp(x, a_h[::-1], P[::-1])
    Fx_vals[i] = np.interp(x, a_h[::-1], F_x[::-1])
    Fy_vals[i] = np.interp(x, a_h[::-1], F_y[::-1])
    Fz_vals[i] = np.interp(x, a_h[::-1], F_z[::-1])
    taux_vals[i] = np.interp(x, a_h[::-1], tau_x[::-1])
    tauy_vals[i] = np.interp(x, a_h[::-1], tau_y[::-1])
    tauz_vals[i] = np.interp(x, a_h[::-1], tau_z[::-1])
    H_vals[i]  = np.interp(x, a_h[::-1], H[::-1])
    K_vals[i]  = np.interp(x, a_h[::-1], K[::-1])
    Qx_vals[i] = np.interp(x, a_h[::-1], Q_x[::-1])
    Qy_vals[i] = np.interp(x, a_h[::-1], Q_y[::-1])
    tildeQ_vals[i] = np.interp(x, a_h[::-1], tildeQ[::-1])

    sP_vals[i]  = np.interp(x, a_h[::-1], sP[::-1])
    sFx_vals[i] = np.interp(x, a_h[::-1], sF_x[::-1])
    sFy_vals[i] = np.interp(x, a_h[::-1], sF_y[::-1])
    sFz_vals[i] = np.interp(x, a_h[::-1], sF_z[::-1])
    staux_vals[i] = np.interp(x, a_h[::-1], stau_x[::-1])
    stauy_vals[i] = np.interp(x, a_h[::-1], stau_y[::-1])
    stauz_vals[i] = np.interp(x, a_h[::-1], stau_z[::-1])
    sH_vals[i]  = np.interp(x, a_h[::-1], sH[::-1])
    sK_vals[i]  = np.interp(x, a_h[::-1], sK[::-1])
    sQx_vals[i] = np.interp(x, a_h[::-1], sQ_x[::-1])
    sQy_vals[i] = np.interp(x, a_h[::-1], sQ_y[::-1])
    stildeQ_vals[i] = np.interp(x, a_h[::-1], stildeQ[::-1])


###################################################
# Plots
###################################################
fig, ax = plt.subplots()

ax.plot(Tcuts, H_vals)
ax.fill_between(Tcuts, H_vals - sH_vals, H_vals + sH_vals, alpha=0.3)

ax.set_xscale("log")
ax.set_xlabel(r"$T_{\rm cut}$")
ax.set_ylabel(r"$H$")
ax.set_title(fr"$H$ for q={q_fixed}, e={e_fixed}, and $a/a_h=${a_over_a_h_target}")

fig2, ax2 = plt.subplots()

ax2.plot(Tcuts, K_vals)
ax2.fill_between(Tcuts, K_vals - sK_vals, K_vals + sK_vals, alpha=0.3)

ax2.set_xscale("log")
ax2.set_xlabel(r"$T_{\rm cut}$")
ax2.set_ylabel(r"$K$")
ax2.set_title(fr"$K$ for q={q_fixed}, e={e_fixed}, and $a/a_h=${a_over_a_h_target}")

fig3, ax3 = plt.subplots()

ax3.plot(Tcuts, Fx_vals)
ax3.fill_between(Tcuts, Fx_vals - sFx_vals, Fx_vals + sFx_vals, alpha=0.3)
ax3.plot(Tcuts, Fy_vals)
ax3.fill_between(Tcuts, Fy_vals - sFy_vals, Fy_vals + sFy_vals, alpha=0.3)

ax3.set_xscale("log")
ax3.set_xlabel(r"$T_{\rm cut}$")
ax3.set_ylabel(r"$F$ [$G\rho a$]")
ax3.set_title(fr"$F$ for q={q_fixed}, e={e_fixed}, and $a/a_h=${a_over_a_h_target}")

fig4, ax4 = plt.subplots()

ax4.plot(Tcuts, tildeQ_vals)
ax4.fill_between(Tcuts, tildeQ_vals - stildeQ_vals, tildeQ_vals + stildeQ_vals, alpha=0.3)

ax4.set_xscale("log")
ax4.set_xlabel(r"$T_{\rm cut}$")
ax4.set_ylabel(r"$\tilde Q$")
ax4.set_title(fr"$\tilde Q$ for q={q_fixed}, e={e_fixed}, and $a/a_h=${a_over_a_h_target}")

plt.show()