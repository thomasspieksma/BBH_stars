import numpy as np
import matplotlib.pyplot as plt

rp_max = 5
r_sphere = 50
t_max = 1e+10

rho = 1
q = 0.001
e = 0.9

mu = q / (1+q)**2

def b_max(v):
    return rp_max * np.sqrt( 1 + 2 / (v**2 * rp_max) )

def f(v, sigma):
    return np.sqrt(2/np.pi) * (v**2/sigma**3) * np.exp(-v**2 / (2*sigma**2))

v, DeltaE, sDeltaE, DeltaT, sDeltaT, Deltavx, sDeltavx, Deltavy, sDeltavy, Deltavz, sDeltavz, DeltaLx, sDeltaLx, DeltaLy, sDeltaLy, DeltaLz, sDeltaLz, Delta_varpi, sDelta_varpi, Nresolved = np.loadtxt('results-small-q/q='+str(q)+'_e='+str(e)+'.txt', unpack=True)

N_v = len(v)

N_a_h = 50

a_h = np.logspace(3,-2,N_a_h) # hardening radius, in units of a

sigma = np.sqrt( mu / (4 * a_h) )

Pv = - np.pi * b_max(v)**2 * rho * v * DeltaE
sPv = - np.pi * b_max(v)**2 * rho * v * sDeltaE

Fv_x = - np.pi * b_max(v)**2 * rho * v * Deltavx
sFv_x = - np.pi * b_max(v)**2 * rho * v * sDeltavx

Fv_y = - np.pi * b_max(v)**2 * rho * v * Deltavy
sFv_y = - np.pi * b_max(v)**2 * rho * v * sDeltavy

Fv_z = - np.pi * b_max(v)**2 * rho * v * Deltavz
sFv_z = - np.pi * b_max(v)**2 * rho * v * sDeltavz

tauv_x = - np.pi * b_max(v)**2 * rho * v * DeltaLx
stauv_x = - np.pi * b_max(v)**2 * rho * v * sDeltaLx

tauv_y = - np.pi * b_max(v)**2 * rho * v * DeltaLy
stauv_y = - np.pi * b_max(v)**2 * rho * v * sDeltaLy

tauv_z = - np.pi * b_max(v)**2 * rho * v * DeltaLz
stauv_z = - np.pi * b_max(v)**2 * rho * v * sDeltaLz

Hv = (2 * np.pi * v**2 * b_max(v)**2) * DeltaE / mu
sHv = (2 * np.pi * v**2 * b_max(v)**2) * sDeltaE / mu

varpi_dot_v = np.pi * b_max(v)**2 * rho * v * Delta_varpi
svarpi_dot_v = np.pi * b_max(v)**2 * rho * v * sDelta_varpi

# Reshape for broadcasting and prepend zero row
v0 = np.hstack([0, v])
f0 = np.vstack([np.zeros((1, N_a_h)), f(v[:, np.newaxis], sigma[np.newaxis, :])])

Pv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Pv[:, np.newaxis], (1, N_a_h))])
sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])

Fv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_x[:, np.newaxis], (1, N_a_h))])
sFv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_x[:, np.newaxis], (1, N_a_h))])

Fv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_y[:, np.newaxis], (1, N_a_h))])
sFv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_y[:, np.newaxis], (1, N_a_h))])

Fv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_z[:, np.newaxis], (1, N_a_h))])
sFv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_z[:, np.newaxis], (1, N_a_h))])

tauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_x[:, np.newaxis], (1, N_a_h))])
stauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_x[:, np.newaxis], (1, N_a_h))])

tauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_y[:, np.newaxis], (1, N_a_h))])
stauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_y[:, np.newaxis], (1, N_a_h))])

tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])

Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])

varpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(varpi_dot_v[:, np.newaxis], (1, N_a_h))])
svarpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(svarpi_dot_v[:, np.newaxis], (1, N_a_h))])

H_integrand0 = np.vstack([np.zeros((1, N_a_h)), (sigma[np.newaxis, :] / v[:, np.newaxis]) * f(v[:, np.newaxis], sigma[np.newaxis, :])])

# Integrate over v
P = np.trapz(Pv0 * f0, x=v0, axis=0)
F_x = np.trapz(Fv_x0 * f0, x=v0, axis=0)
F_y = np.trapz(Fv_y0 * f0, x=v0, axis=0)
F_z = np.trapz(Fv_z0 * f0, x=v0, axis=0)
tau_x = np.trapz(tauv_x0 * f0, x=v0, axis=0)
tau_y = np.trapz(tauv_y0 * f0, x=v0, axis=0)
tau_z = np.trapz(tauv_z0 * f0, x=v0, axis=0)
H = np.trapz(Hv0 * H_integrand0, x=v0, axis=0)
varpi_dot = np.trapz(varpi_dot_v0 * f0, x=v0, axis=0)
K = - (1-e**2)/(2*e) + np.sqrt(1-e**2)/(2*e) * tau_z/P
Q_x = - (mu / (2*sigma)) * F_x / P
Q_y = - (mu / (2*sigma)) * F_y / P
tildeQ = - (mu / 2) * varpi_dot / P

# Calculate uncertainties using propagation of uncertainty with trapezoidal rule
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
sK = np.sqrt(1-e**2)/(2*e) * np.abs(tau_z/P) * np.sqrt((stau_z/tau_z)**2 + (sP/P)**2)
sQ_x = (mu / (2*sigma)) * np.abs(F_x / P) * np.sqrt((sF_x/F_x)**2 + (sP/P)**2)
sQ_y = (mu / (2*sigma)) * np.abs(F_y / P) * np.sqrt((sF_y/F_y)**2 + (sP/P)**2)
stildeQ = - (mu / 2) * (varpi_dot / P) * np.sqrt((svarpi_dot/varpi_dot)**2 + (sP/P)**2)

fig, ax = plt.subplots()

ax.plot(v, Hv, label='tol=0.02')
ax.fill_between(v, Hv-sHv, Hv+sHv, alpha=0.3)

ax.set_xscale('log')
ax.set_xlabel(r'$v$')
ax.set_ylabel(r'$H_v$')

fig0, ax0 = plt.subplots()

ax0.plot(v, tauv_z)
ax0.fill_between(v, tauv_z-stauv_z, tauv_z+stauv_z, alpha=0.3)

ax0.set_xscale('log')
ax0.set_xlabel(r'$v$')
ax0.set_ylabel(r'$tau_v$')

fig1, ax1 = plt.subplots()

ax1.plot(v, DeltaT)
ax1.fill_between(v, DeltaT-sDeltaT, DeltaT+sDeltaT, alpha=0.3)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$v$')
ax1.set_ylabel(r'$\Delta T$')

fig2, ax2 = plt.subplots()

ax2.plot(1/a_h, H, label='tol=0.02')
ax2.fill_between(1/a_h, H-sH, H+sH, alpha=0.3)

ax2.set_xscale('log')
ax2.set_xlabel(r'$a/a_h$')
ax2.set_ylabel(r'$H$')

fig3, ax3 = plt.subplots()

ax3.plot(1/a_h, K, label='tol=0.02')
ax3.fill_between(1/a_h, K-sK, K+sK, alpha=0.3)
ax3.set_xlabel(r'$a/a_h$')
ax3.set_ylabel(r'$K$')

ax3.set_xscale('log')

fig4, ax4 = plt.subplots()

ax4.plot(1/a_h, Q_x, label='tol=0.02')
ax4.fill_between(1/a_h, Q_x-sQ_x, Q_x+sQ_x, alpha=0.3)

ax4.plot(1/a_h, Q_y, label='tol=0.02')
ax4.fill_between(1/a_h, Q_y-sQ_y, Q_y+sQ_y, alpha=0.3)

ax4.set_xlabel(r'$a/a_h$')
ax4.set_ylabel(r'$Q_x$, $Q_y$')

ax4.set_xscale('log')

fig5, ax5 = plt.subplots()

ax5.plot(1/a_h, tau_x)
ax5.fill_between(1/a_h, tau_x-stau_x, tau_x+stau_x, alpha=0.3)

ax5.plot(1/a_h, tau_y)
ax5.fill_between(1/a_h, tau_y-stau_y, tau_y+stau_y, alpha=0.3)

ax5.plot(1/a_h, tau_z)
ax5.fill_between(1/a_h, tau_z-stau_z, tau_z+stau_z, alpha=0.3)

ax5.set_xlabel(r'$a/a_h$')
ax5.set_ylabel(r'$\tau_x$, $\tau_y$, $\tau_z$ [$GM\rho a^2$]')

ax5.set_xscale('log')

fig6, ax6 = plt.subplots()

ax6.plot(1/a_h, F_x)
ax6.fill_between(1/a_h, F_x-sF_x, F_x+sF_x, alpha=0.3)

ax6.plot(1/a_h, F_y)
ax6.fill_between(1/a_h, F_y-sF_y, F_y+sF_y, alpha=0.3)

ax6.plot(1/a_h, F_z)
ax6.fill_between(1/a_h, F_z-sF_z, F_z+sF_z, alpha=0.3)

ax6.set_xlabel(r'$a/a_h$')
ax6.set_ylabel(r'$F_x$, $F_y$, $F_z$ [$GM\rho a$]')

ax6.set_xscale('log')

fig7, ax7 = plt.subplots()

ax7.plot(v, Nresolved)

ax7.set_xscale('log')
ax7.set_xlabel(r'$v$')
ax7.set_ylabel(r'$N_{\mathrm{resolved}}$')

##################################################

v, DeltaE, sDeltaE, DeltaT, sDeltaT, Deltavx, sDeltavx, Deltavy, sDeltavy, Deltavz, sDeltavz, DeltaLx, sDeltaLx, DeltaLy, sDeltaLy, DeltaLz, sDeltaLz, Delta_varpi, sDelta_varpi, Nresolved = np.loadtxt('results-small-q-prec-0.01/q='+str(q)+'_e='+str(e)+'.txt', unpack=True)

N_v = len(v)

N_a_h = 50

a_h = np.logspace(3,-2,N_a_h) # hardening radius, in units of a

sigma = np.sqrt( mu / (4 * a_h) )

Pv = - np.pi * b_max(v)**2 * rho * v * DeltaE
sPv = - np.pi * b_max(v)**2 * rho * v * sDeltaE

Fv_x = - np.pi * b_max(v)**2 * rho * v * Deltavx
sFv_x = - np.pi * b_max(v)**2 * rho * v * sDeltavx

Fv_y = - np.pi * b_max(v)**2 * rho * v * Deltavy
sFv_y = - np.pi * b_max(v)**2 * rho * v * sDeltavy

Fv_z = - np.pi * b_max(v)**2 * rho * v * Deltavz
sFv_z = - np.pi * b_max(v)**2 * rho * v * sDeltavz

tauv_x = - np.pi * b_max(v)**2 * rho * v * DeltaLx
stauv_x = - np.pi * b_max(v)**2 * rho * v * sDeltaLx

tauv_y = - np.pi * b_max(v)**2 * rho * v * DeltaLy
stauv_y = - np.pi * b_max(v)**2 * rho * v * sDeltaLy

tauv_z = - np.pi * b_max(v)**2 * rho * v * DeltaLz
stauv_z = - np.pi * b_max(v)**2 * rho * v * sDeltaLz

Hv = (2 * np.pi * v**2 * b_max(v)**2) * DeltaE / mu
sHv = (2 * np.pi * v**2 * b_max(v)**2) * sDeltaE / mu

varpi_dot_v = np.pi * b_max(v)**2 * rho * v * Delta_varpi
svarpi_dot_v = np.pi * b_max(v)**2 * rho * v * sDelta_varpi

# Reshape for broadcasting and prepend zero row
v0 = np.hstack([0, v])
f0 = np.vstack([np.zeros((1, N_a_h)), f(v[:, np.newaxis], sigma[np.newaxis, :])])

Pv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Pv[:, np.newaxis], (1, N_a_h))])
sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])

Fv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_x[:, np.newaxis], (1, N_a_h))])
sFv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_x[:, np.newaxis], (1, N_a_h))])

Fv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_y[:, np.newaxis], (1, N_a_h))])
sFv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_y[:, np.newaxis], (1, N_a_h))])

Fv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_z[:, np.newaxis], (1, N_a_h))])
sFv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_z[:, np.newaxis], (1, N_a_h))])

tauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_x[:, np.newaxis], (1, N_a_h))])
stauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_x[:, np.newaxis], (1, N_a_h))])

tauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_y[:, np.newaxis], (1, N_a_h))])
stauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_y[:, np.newaxis], (1, N_a_h))])

tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])

Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])

varpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(varpi_dot_v[:, np.newaxis], (1, N_a_h))])
svarpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(svarpi_dot_v[:, np.newaxis], (1, N_a_h))])

H_integrand0 = np.vstack([np.zeros((1, N_a_h)), (sigma[np.newaxis, :] / v[:, np.newaxis]) * f(v[:, np.newaxis], sigma[np.newaxis, :])])

# Integrate over v
P = np.trapz(Pv0 * f0, x=v0, axis=0)
F_x = np.trapz(Fv_x0 * f0, x=v0, axis=0)
F_y = np.trapz(Fv_y0 * f0, x=v0, axis=0)
F_z = np.trapz(Fv_z0 * f0, x=v0, axis=0)
tau_x = np.trapz(tauv_x0 * f0, x=v0, axis=0)
tau_y = np.trapz(tauv_y0 * f0, x=v0, axis=0)
tau_z = np.trapz(tauv_z0 * f0, x=v0, axis=0)
H = np.trapz(Hv0 * H_integrand0, x=v0, axis=0)
varpi_dot = np.trapz(varpi_dot_v0 * f0, x=v0, axis=0)
K = - (1-e**2)/(2*e) + np.sqrt(1-e**2)/(2*e) * tau_z/P
Q_x = - (mu / (2*sigma)) * F_x / P
Q_y = - (mu / (2*sigma)) * F_y / P
tildeQ = - (mu / 2) * varpi_dot / P

# Calculate uncertainties using propagation of uncertainty with trapezoidal rule
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
sK = np.sqrt(1-e**2)/(2*e) * np.abs(tau_z/P) * np.sqrt((stau_z/tau_z)**2 + (sP/P)**2)
sQ_x = (mu / (2*sigma)) * np.abs(F_x / P) * np.sqrt((sF_x/F_x)**2 + (sP/P)**2)
sQ_y = (mu / (2*sigma)) * np.abs(F_y / P) * np.sqrt((sF_y/F_y)**2 + (sP/P)**2)
stildeQ = - (mu / 2) * (varpi_dot / P) * np.sqrt((svarpi_dot/varpi_dot)**2 + (sP/P)**2)

ax.plot(v, Hv)
ax.fill_between(v, Hv-sHv, Hv+sHv, alpha=0.3)

ax2.plot(1/a_h, H, label='RK, tol=0.01')
ax2.fill_between(1/a_h, H-sH, H+sH, alpha=0.3)

ax3.plot(1/a_h, K, label='RK, tol=0.01')
ax3.fill_between(1/a_h, K-sK, K+sK, alpha=0.3)
ax3.set_xlabel(r'$a/a_h$')
ax3.set_ylabel(r'$K$')

ax4.plot(1/a_h, Q_x, label='RK, tol=0.01')
ax4.fill_between(1/a_h, Q_x-sQ_x, Q_x+sQ_x, alpha=0.3)
ax4.plot(1/a_h, Q_y, label='RK, tol=0.01')
ax4.fill_between(1/a_h, Q_y-sQ_y, Q_y+sQ_y, alpha=0.3)

##################################################

v, DeltaE, sDeltaE, DeltaT, sDeltaT, Deltavx, sDeltavx, Deltavy, sDeltavy, Deltavz, sDeltavz, DeltaLx, sDeltaLx, DeltaLy, sDeltaLy, DeltaLz, sDeltaLz, Delta_varpi, sDelta_varpi, Nresolved = np.loadtxt('results-small-q-prec-0.005/q='+str(q)+'_e='+str(e)+'.txt', unpack=True)

N_v = len(v)

N_a_h = 50

a_h = np.logspace(3,-2,N_a_h) # hardening radius, in units of a

sigma = np.sqrt( mu / (4 * a_h) )

Pv = - np.pi * b_max(v)**2 * rho * v * DeltaE
sPv = - np.pi * b_max(v)**2 * rho * v * sDeltaE

Fv_x = - np.pi * b_max(v)**2 * rho * v * Deltavx
sFv_x = - np.pi * b_max(v)**2 * rho * v * sDeltavx

Fv_y = - np.pi * b_max(v)**2 * rho * v * Deltavy
sFv_y = - np.pi * b_max(v)**2 * rho * v * sDeltavy

Fv_z = - np.pi * b_max(v)**2 * rho * v * Deltavz
sFv_z = - np.pi * b_max(v)**2 * rho * v * sDeltavz

tauv_x = - np.pi * b_max(v)**2 * rho * v * DeltaLx
stauv_x = - np.pi * b_max(v)**2 * rho * v * sDeltaLx

tauv_y = - np.pi * b_max(v)**2 * rho * v * DeltaLy
stauv_y = - np.pi * b_max(v)**2 * rho * v * sDeltaLy

tauv_z = - np.pi * b_max(v)**2 * rho * v * DeltaLz
stauv_z = - np.pi * b_max(v)**2 * rho * v * sDeltaLz

Hv = (2 * np.pi * v**2 * b_max(v)**2) * DeltaE / mu
sHv = (2 * np.pi * v**2 * b_max(v)**2) * sDeltaE / mu
varpi_dot_v = np.pi * b_max(v)**2 * rho * v * Delta_varpi
svarpi_dot_v = np.pi * b_max(v)**2 * rho * v * sDelta_varpi

# Reshape for broadcasting and prepend zero row
v0 = np.hstack([0, v])
f0 = np.vstack([np.zeros((1, N_a_h)), f(v[:, np.newaxis], sigma[np.newaxis, :])])

Pv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Pv[:, np.newaxis], (1, N_a_h))])
sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])

Fv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_x[:, np.newaxis], (1, N_a_h))])
sFv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_x[:, np.newaxis], (1, N_a_h))])

Fv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_y[:, np.newaxis], (1, N_a_h))])
sFv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_y[:, np.newaxis], (1, N_a_h))])

Fv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_z[:, np.newaxis], (1, N_a_h))])
sFv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_z[:, np.newaxis], (1, N_a_h))])

tauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_x[:, np.newaxis], (1, N_a_h))])
stauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_x[:, np.newaxis], (1, N_a_h))])

tauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_y[:, np.newaxis], (1, N_a_h))])
stauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_y[:, np.newaxis], (1, N_a_h))])

tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])

Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])
varpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(varpi_dot_v[:, np.newaxis], (1, N_a_h))])
svarpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(svarpi_dot_v[:, np.newaxis], (1, N_a_h))])

H_integrand0 = np.vstack([np.zeros((1, N_a_h)), (sigma[np.newaxis, :] / v[:, np.newaxis]) * f(v[:, np.newaxis], sigma[np.newaxis, :])])

# Integrate over v
P = np.trapz(Pv0 * f0, x=v0, axis=0)
F_x = np.trapz(Fv_x0 * f0, x=v0, axis=0)
F_y = np.trapz(Fv_y0 * f0, x=v0, axis=0)
F_z = np.trapz(Fv_z0 * f0, x=v0, axis=0)
tau_x = np.trapz(tauv_x0 * f0, x=v0, axis=0)
tau_y = np.trapz(tauv_y0 * f0, x=v0, axis=0)
tau_z = np.trapz(tauv_z0 * f0, x=v0, axis=0)
H = np.trapz(Hv0 * H_integrand0, x=v0, axis=0)
varpi_dot = np.trapz(varpi_dot_v0 * f0, x=v0, axis=0)
K = - (1-e**2)/(2*e) + np.sqrt(1-e**2)/(2*e) * tau_z/P
Q_x = - (mu / (2*sigma)) * F_x / P
Q_y = - (mu / (2*sigma)) * F_y / P
tildeQ = - (mu / 2) * varpi_dot / P

# Calculate uncertainties using propagation of uncertainty with trapezoidal rule
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
sK = np.sqrt(1-e**2)/(2*e) * np.abs(tau_z/P) * np.sqrt((stau_z/tau_z)**2 + (sP/P)**2)
sQ_x = (mu / (2*sigma)) * np.abs(F_x / P) * np.sqrt((sF_x/F_x)**2 + (sP/P)**2)
sQ_y = (mu / (2*sigma)) * np.abs(F_y / P) * np.sqrt((sF_y/F_y)**2 + (sP/P)**2)
stildeQ = - (mu / 2) * (varpi_dot / P) * np.sqrt((svarpi_dot/varpi_dot)**2 + (sP/P)**2)

ax.plot(v, Hv)
ax.fill_between(v, Hv-sHv, Hv+sHv, alpha=0.3)

ax2.plot(1/a_h, H, label='RK, tol=0.005')
ax2.fill_between(1/a_h, H-sH, H+sH, alpha=0.3)

ax3.plot(1/a_h, K, label='RK, tol=0.005')
ax3.fill_between(1/a_h, K-sK, K+sK, alpha=0.3)

ax4.plot(1/a_h, Q_x, label='RK, tol=0.005')
ax4.fill_between(1/a_h, Q_x-sQ_x, Q_x+sQ_x, alpha=0.3)
ax4.plot(1/a_h, Q_y, label='RK, tol=0.005')
ax4.fill_between(1/a_h, Q_y-sQ_y, Q_y+sQ_y, alpha=0.3)

##################################################

v, DeltaE, sDeltaE, DeltaT, sDeltaT, Deltavx, sDeltavx, Deltavy, sDeltavy, Deltavz, sDeltavz, DeltaLx, sDeltaLx, DeltaLy, sDeltaLy, DeltaLz, sDeltaLz, Delta_varpi, sDelta_varpi, Nresolved = np.loadtxt('results-small-q-prec-0.002/q='+str(q)+'_e='+str(e)+'.txt', unpack=True)

N_v = len(v)

N_a_h = 50

a_h = np.logspace(3,-2,N_a_h) # hardening radius, in units of a

sigma = np.sqrt( mu / (4 * a_h) )

Pv = - np.pi * b_max(v)**2 * rho * v * DeltaE
sPv = - np.pi * b_max(v)**2 * rho * v * sDeltaE

Fv_x = - np.pi * b_max(v)**2 * rho * v * Deltavx
sFv_x = - np.pi * b_max(v)**2 * rho * v * sDeltavx

Fv_y = - np.pi * b_max(v)**2 * rho * v * Deltavy
sFv_y = - np.pi * b_max(v)**2 * rho * v * sDeltavy

Fv_z = - np.pi * b_max(v)**2 * rho * v * Deltavz
sFv_z = - np.pi * b_max(v)**2 * rho * v * sDeltavz

tauv_x = - np.pi * b_max(v)**2 * rho * v * DeltaLx
stauv_x = - np.pi * b_max(v)**2 * rho * v * sDeltaLx

tauv_y = - np.pi * b_max(v)**2 * rho * v * DeltaLy
stauv_y = - np.pi * b_max(v)**2 * rho * v * sDeltaLy

tauv_z = - np.pi * b_max(v)**2 * rho * v * DeltaLz
stauv_z = - np.pi * b_max(v)**2 * rho * v * sDeltaLz

Hv = (2 * np.pi * v**2 * b_max(v)**2) * DeltaE / mu
sHv = (2 * np.pi * v**2 * b_max(v)**2) * sDeltaE / mu

varpi_dot_v = np.pi * b_max(v)**2 * rho * v * Delta_varpi
svarpi_dot_v = np.pi * b_max(v)**2 * rho * v * sDelta_varpi

# Reshape for broadcasting and prepend zero row
v0 = np.hstack([0, v])
f0 = np.vstack([np.zeros((1, N_a_h)), f(v[:, np.newaxis], sigma[np.newaxis, :])])

Pv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Pv[:, np.newaxis], (1, N_a_h))])
sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])

Fv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_x[:, np.newaxis], (1, N_a_h))])
sFv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_x[:, np.newaxis], (1, N_a_h))])

Fv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_y[:, np.newaxis], (1, N_a_h))])
sFv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_y[:, np.newaxis], (1, N_a_h))])

Fv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_z[:, np.newaxis], (1, N_a_h))])
sFv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_z[:, np.newaxis], (1, N_a_h))])

tauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_x[:, np.newaxis], (1, N_a_h))])
stauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_x[:, np.newaxis], (1, N_a_h))])

tauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_y[:, np.newaxis], (1, N_a_h))])
stauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_y[:, np.newaxis], (1, N_a_h))])

tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])

Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])

varpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(varpi_dot_v[:, np.newaxis], (1, N_a_h))])
svarpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(svarpi_dot_v[:, np.newaxis], (1, N_a_h))])

H_integrand0 = np.vstack([np.zeros((1, N_a_h)), (sigma[np.newaxis, :] / v[:, np.newaxis]) * f(v[:, np.newaxis], sigma[np.newaxis, :])])

# Integrate over v
P = np.trapz(Pv0 * f0, x=v0, axis=0)
F_x = np.trapz(Fv_x0 * f0, x=v0, axis=0)
F_y = np.trapz(Fv_y0 * f0, x=v0, axis=0)
F_z = np.trapz(Fv_z0 * f0, x=v0, axis=0)
tau_x = np.trapz(tauv_x0 * f0, x=v0, axis=0)
tau_y = np.trapz(tauv_y0 * f0, x=v0, axis=0)
tau_z = np.trapz(tauv_z0 * f0, x=v0, axis=0)
H = np.trapz(Hv0 * H_integrand0, x=v0, axis=0)
varpi_dot = np.trapz(varpi_dot_v0 * f0, x=v0, axis=0)
K = - (1-e**2)/(2*e) + np.sqrt(1-e**2)/(2*e) * tau_z/P
Q_x = - (mu / (2*sigma)) * F_x / P
Q_y = - (mu / (2*sigma)) * F_y / P
tildeQ = - (mu / 2) * varpi_dot / P

# Calculate uncertainties using propagation of uncertainty with trapezoidal rule
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
sK = np.sqrt(1-e**2)/(2*e) * np.abs(tau_z/P) * np.sqrt((stau_z/tau_z)**2 + (sP/P)**2)
sQ_x = (mu / (2*sigma)) * np.abs(F_x / P) * np.sqrt((sF_x/F_x)**2 + (sP/P)**2)
sQ_y = (mu / (2*sigma)) * np.abs(F_y / P) * np.sqrt((sF_y/F_y)**2 + (sP/P)**2)
stildeQ = - (mu / 2) * (varpi_dot / P) * np.sqrt((svarpi_dot/varpi_dot)**2 + (sP/P)**2)

ax.plot(v, Hv)
ax.fill_between(v, Hv-sHv, Hv+sHv, alpha=0.3)

ax2.plot(1/a_h, H, label='RK, tol=0.002')
ax2.fill_between(1/a_h, H-sH, H+sH, alpha=0.3)

ax3.plot(1/a_h, K, label='RK, tol=0.002')
ax3.fill_between(1/a_h, K-sK, K+sK, alpha=0.3)

ax4.plot(1/a_h, Q_x, label='RK, tol=0.002')
ax4.fill_between(1/a_h, Q_x-sQ_x, Q_x+sQ_x, alpha=0.3)
ax4.plot(1/a_h, Q_y, label='RK, tol=0.002')
ax4.fill_between(1/a_h, Q_y-sQ_y, Q_y+sQ_y, alpha=0.3)

##################################################

v, DeltaE, sDeltaE, DeltaT, sDeltaT, Deltavx, sDeltavx, Deltavy, sDeltavy, Deltavz, sDeltavz, DeltaLx, sDeltaLx, DeltaLy, sDeltaLy, DeltaLz, sDeltaLz, Delta_varpi, sDelta_varpi, Nresolved = np.loadtxt('results-small-q-pihajoki-prec-0.02/q='+str(q)+'_e='+str(e)+'.txt', unpack=True)

N_v = len(v)

N_a_h = 50

a_h = np.logspace(3,-2,N_a_h) # hardening radius, in units of a

sigma = np.sqrt( mu / (4 * a_h) )

Pv = - np.pi * b_max(v)**2 * rho * v * DeltaE
sPv = - np.pi * b_max(v)**2 * rho * v * sDeltaE

Fv_x = - np.pi * b_max(v)**2 * rho * v * Deltavx
sFv_x = - np.pi * b_max(v)**2 * rho * v * sDeltavx

Fv_y = - np.pi * b_max(v)**2 * rho * v * Deltavy
sFv_y = - np.pi * b_max(v)**2 * rho * v * sDeltavy

Fv_z = - np.pi * b_max(v)**2 * rho * v * Deltavz
sFv_z = - np.pi * b_max(v)**2 * rho * v * sDeltavz

tauv_x = - np.pi * b_max(v)**2 * rho * v * DeltaLx
stauv_x = - np.pi * b_max(v)**2 * rho * v * sDeltaLx

tauv_y = - np.pi * b_max(v)**2 * rho * v * DeltaLy
stauv_y = - np.pi * b_max(v)**2 * rho * v * sDeltaLy

tauv_z = - np.pi * b_max(v)**2 * rho * v * DeltaLz
stauv_z = - np.pi * b_max(v)**2 * rho * v * sDeltaLz

Hv = (2 * np.pi * v**2 * b_max(v)**2) * DeltaE / mu
sHv = (2 * np.pi * v**2 * b_max(v)**2) * sDeltaE / mu
varpi_dot_v = np.pi * b_max(v)**2 * rho * v * Delta_varpi
svarpi_dot_v = np.pi * b_max(v)**2 * rho * v * sDelta_varpi

# Reshape for broadcasting and prepend zero row
v0 = np.hstack([0, v])
f0 = np.vstack([np.zeros((1, N_a_h)), f(v[:, np.newaxis], sigma[np.newaxis, :])])

Pv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Pv[:, np.newaxis], (1, N_a_h))])
sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])

Fv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_x[:, np.newaxis], (1, N_a_h))])
sFv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_x[:, np.newaxis], (1, N_a_h))])

Fv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_y[:, np.newaxis], (1, N_a_h))])
sFv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_y[:, np.newaxis], (1, N_a_h))])

Fv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_z[:, np.newaxis], (1, N_a_h))])
sFv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_z[:, np.newaxis], (1, N_a_h))])

tauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_x[:, np.newaxis], (1, N_a_h))])
stauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_x[:, np.newaxis], (1, N_a_h))])

tauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_y[:, np.newaxis], (1, N_a_h))])
stauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_y[:, np.newaxis], (1, N_a_h))])

tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])

Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])
varpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(varpi_dot_v[:, np.newaxis], (1, N_a_h))])
svarpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(svarpi_dot_v[:, np.newaxis], (1, N_a_h))])

H_integrand0 = np.vstack([np.zeros((1, N_a_h)), (sigma[np.newaxis, :] / v[:, np.newaxis]) * f(v[:, np.newaxis], sigma[np.newaxis, :])])

# Integrate over v
P = np.trapz(Pv0 * f0, x=v0, axis=0)
F_x = np.trapz(Fv_x0 * f0, x=v0, axis=0)
F_y = np.trapz(Fv_y0 * f0, x=v0, axis=0)
F_z = np.trapz(Fv_z0 * f0, x=v0, axis=0)
tau_x = np.trapz(tauv_x0 * f0, x=v0, axis=0)
tau_y = np.trapz(tauv_y0 * f0, x=v0, axis=0)
tau_z = np.trapz(tauv_z0 * f0, x=v0, axis=0)
H = np.trapz(Hv0 * H_integrand0, x=v0, axis=0)
varpi_dot = np.trapz(varpi_dot_v0 * f0, x=v0, axis=0)
K = - (1-e**2)/(2*e) + np.sqrt(1-e**2)/(2*e) * tau_z/P
Q_x = - (mu / (2*sigma)) * F_x / P
Q_y = - (mu / (2*sigma)) * F_y / P
tildeQ = - (mu / 2) * varpi_dot / P

# Calculate uncertainties using propagation of uncertainty with trapezoidal rule
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
sK = np.sqrt(1-e**2)/(2*e) * np.abs(tau_z/P) * np.sqrt((stau_z/tau_z)**2 + (sP/P)**2)
sQ_x = (mu / (2*sigma)) * np.abs(F_x / P) * np.sqrt((sF_x/F_x)**2 + (sP/P)**2)
sQ_y = (mu / (2*sigma)) * np.abs(F_y / P) * np.sqrt((sF_y/F_y)**2 + (sP/P)**2)
stildeQ = - (mu / 2) * (varpi_dot / P) * np.sqrt((svarpi_dot/varpi_dot)**2 + (sP/P)**2)

ax.plot(v, Hv, linestyle='dashed')
ax.fill_between(v, Hv-sHv, Hv+sHv, alpha=0.3)

ax2.plot(1/a_h, H, label='Pihajoki, tol=0.02', linestyle='dashed')
ax2.fill_between(1/a_h, H-sH, H+sH, alpha=0.3)

ax3.plot(1/a_h, K, label='Pihajoki, tol=0.02', linestyle='dashed')
ax3.fill_between(1/a_h, K-sK, K+sK, alpha=0.3)

ax4.plot(1/a_h, Q_x, label='Pihajoki, tol=0.02', linestyle='dashed')
ax4.fill_between(1/a_h, Q_x-sQ_x, Q_x+sQ_x, alpha=0.3)
ax4.plot(1/a_h, Q_y, label='Pihajoki, tol=0.02', linestyle='dashed')
ax4.fill_between(1/a_h, Q_y-sQ_y, Q_y+sQ_y, alpha=0.3)

##################################################

v, DeltaE, sDeltaE, DeltaT, sDeltaT, Deltavx, sDeltavx, Deltavy, sDeltavy, Deltavz, sDeltavz, DeltaLx, sDeltaLx, DeltaLy, sDeltaLy, DeltaLz, sDeltaLz, Delta_varpi, sDelta_varpi, Nresolved = np.loadtxt('results-small-q-pihajoki-prec-0.01/q='+str(q)+'_e='+str(e)+'.txt', unpack=True)

N_v = len(v)

N_a_h = 50

a_h = np.logspace(3,-2,N_a_h) # hardening radius, in units of a

sigma = np.sqrt( mu / (4 * a_h) )

Pv = - np.pi * b_max(v)**2 * rho * v * DeltaE
sPv = - np.pi * b_max(v)**2 * rho * v * sDeltaE

Fv_x = - np.pi * b_max(v)**2 * rho * v * Deltavx
sFv_x = - np.pi * b_max(v)**2 * rho * v * sDeltavx

Fv_y = - np.pi * b_max(v)**2 * rho * v * Deltavy
sFv_y = - np.pi * b_max(v)**2 * rho * v * sDeltavy

Fv_z = - np.pi * b_max(v)**2 * rho * v * Deltavz
sFv_z = - np.pi * b_max(v)**2 * rho * v * sDeltavz

tauv_x = - np.pi * b_max(v)**2 * rho * v * DeltaLx
stauv_x = - np.pi * b_max(v)**2 * rho * v * sDeltaLx

tauv_y = - np.pi * b_max(v)**2 * rho * v * DeltaLy
stauv_y = - np.pi * b_max(v)**2 * rho * v * sDeltaLy

tauv_z = - np.pi * b_max(v)**2 * rho * v * DeltaLz
stauv_z = - np.pi * b_max(v)**2 * rho * v * sDeltaLz

Hv = (2 * np.pi * v**2 * b_max(v)**2) * DeltaE / mu
sHv = (2 * np.pi * v**2 * b_max(v)**2) * sDeltaE / mu
varpi_dot_v = np.pi * b_max(v)**2 * rho * v * Delta_varpi
svarpi_dot_v = np.pi * b_max(v)**2 * rho * v * sDelta_varpi

# Reshape for broadcasting and prepend zero row
v0 = np.hstack([0, v])
f0 = np.vstack([np.zeros((1, N_a_h)), f(v[:, np.newaxis], sigma[np.newaxis, :])])

Pv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Pv[:, np.newaxis], (1, N_a_h))])
sPv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sPv[:, np.newaxis], (1, N_a_h))])

Fv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_x[:, np.newaxis], (1, N_a_h))])
sFv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_x[:, np.newaxis], (1, N_a_h))])

Fv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_y[:, np.newaxis], (1, N_a_h))])
sFv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_y[:, np.newaxis], (1, N_a_h))])

Fv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Fv_z[:, np.newaxis], (1, N_a_h))])
sFv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sFv_z[:, np.newaxis], (1, N_a_h))])

tauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_x[:, np.newaxis], (1, N_a_h))])
stauv_x0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_x[:, np.newaxis], (1, N_a_h))])

tauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_y[:, np.newaxis], (1, N_a_h))])
stauv_y0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_y[:, np.newaxis], (1, N_a_h))])

tauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(tauv_z[:, np.newaxis], (1, N_a_h))])
stauv_z0 = np.vstack([np.zeros((1, N_a_h)), np.tile(stauv_z[:, np.newaxis], (1, N_a_h))])

Hv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(Hv[:, np.newaxis], (1, N_a_h))])
sHv0 = np.vstack([np.zeros((1, N_a_h)), np.tile(sHv[:, np.newaxis], (1, N_a_h))])
varpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(varpi_dot_v[:, np.newaxis], (1, N_a_h))])
svarpi_dot_v0 = np.vstack([np.zeros((1, N_a_h)), np.tile(svarpi_dot_v[:, np.newaxis], (1, N_a_h))])

H_integrand0 = np.vstack([np.zeros((1, N_a_h)), (sigma[np.newaxis, :] / v[:, np.newaxis]) * f(v[:, np.newaxis], sigma[np.newaxis, :])])

# Integrate over v
P = np.trapz(Pv0 * f0, x=v0, axis=0)
F_x = np.trapz(Fv_x0 * f0, x=v0, axis=0)
F_y = np.trapz(Fv_y0 * f0, x=v0, axis=0)
F_z = np.trapz(Fv_z0 * f0, x=v0, axis=0)
tau_x = np.trapz(tauv_x0 * f0, x=v0, axis=0)
tau_y = np.trapz(tauv_y0 * f0, x=v0, axis=0)
tau_z = np.trapz(tauv_z0 * f0, x=v0, axis=0)
H = np.trapz(Hv0 * H_integrand0, x=v0, axis=0)
varpi_dot = np.trapz(varpi_dot_v0 * f0, x=v0, axis=0)
K = - (1-e**2)/(2*e) + np.sqrt(1-e**2)/(2*e) * tau_z/P
Q_x = - (mu / (2*sigma)) * F_x / P
Q_y = - (mu / (2*sigma)) * F_y / P
tildeQ = - (mu / 2) * varpi_dot / P

# Calculate uncertainties using propagation of uncertainty with trapezoidal rule
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
sK = np.sqrt(1-e**2)/(2*e) * np.abs(tau_z/P) * np.sqrt((stau_z/tau_z)**2 + (sP/P)**2)
sQ_x = (mu / (2*sigma)) * np.abs(F_x / P) * np.sqrt((sF_x/F_x)**2 + (sP/P)**2)
sQ_y = (mu / (2*sigma)) * np.abs(F_y / P) * np.sqrt((sF_y/F_y)**2 + (sP/P)**2)
stildeQ = - (mu / 2) * (varpi_dot / P) * np.sqrt((svarpi_dot/varpi_dot)**2 + (sP/P)**2)

ax.plot(v, Hv, linestyle='dashed')
ax.fill_between(v, Hv-sHv, Hv+sHv, alpha=0.3)

ax2.plot(1/a_h, H, label='Pihajoki, tol=0.01', linestyle='dashed')
ax2.fill_between(1/a_h, H-sH, H+sH, alpha=0.3)

ax3.plot(1/a_h, K, label='Pihajoki, tol=0.01', linestyle='dashed')
ax3.fill_between(1/a_h, K-sK, K+sK, alpha=0.3)

ax4.plot(1/a_h, Q_x, label='Pihajoki, tol=0.01', linestyle='dashed')
ax4.fill_between(1/a_h, Q_x-sQ_x, Q_x+sQ_x, alpha=0.3)
ax4.plot(1/a_h, Q_y, label='Pihajoki, tol=0.01', linestyle='dashed')
ax4.fill_between(1/a_h, Q_y-sQ_y, Q_y+sQ_y, alpha=0.3)

ax2.legend()
ax3.legend()
ax4.legend()
ax3.set_title('q='+str(q)+', e='+str(e))
plt.show()