# Solving for the binary evolution

This document summarizes the physics and equations needed to solve for the
coupled evolution of a Keplerian binary immersed in a uniform stellar
background. The binary shrinks due to three-body slingshots, and we track
both its internal orbital parameters and the motion of its center of mass.

---

## 1. Setup

A binary of total mass $M$, mass ratio $q = M_2/M_1 \le 1$, and reduced mass
$\mu = qM/(1+q)^2$ sits in a uniform, isotropic medium of density $\rho$ and
one-dimensional velocity dispersion $\sigma$. The hard-binary radius is

$$a_h = \frac{G\mu}{4\sigma^2}\,.$$

We restrict to **in-plane motion**: the binary orbital plane is fixed and the
center-of-mass velocity $\vec V$ lies within it. (Out-of-plane motion is
damped; see Section 2.3.3 of the draft.)

### Coordinate frames

- **Lab frame** $(x, y)$: fixed inertial frame in the orbital plane.
- **Binary frame** $(\hat e, \hat n)$: $\hat e$ points along the eccentricity
  vector and $\hat n = \hat L \times \hat e$. The binary frame is rotated by
  the longitude of periapsis $\varpi$ relative to the lab frame:

$$\hat e = (\cos\varpi,\; \sin\varpi), \qquad \hat n = (-\sin\varpi,\; \cos\varpi).$$

Velocity projections between frames:

$$V_{\hat e} = V_x \cos\varpi + V_y \sin\varpi, \qquad V_{\hat n} = -V_x \sin\varpi + V_y \cos\varpi.$$

---

## 2. State variables

The binary state at any instant is described by five quantities:

| Symbol | Meaning |
|--------|---------|
| $a$ | semi-major axis |
| $e$ | eccentricity |
| $V_x,\, V_y$ | center-of-mass velocity in the lab frame |
| $\varpi$ | longitude of periapsis (angle from lab $x$-axis to $\hat e$) |

The mass ratio $q$ is constant. The environment parameters $\rho$ and $\sigma$
are also constant (uniform medium, no feedback).

---

## 3. Evolution parameters from three-body data

Five dimensionless parameters, computed from the three-body scattering
experiments and Maxwellian reweighting, govern the binary evolution:

**Hardening rate** — controls shrinkage of the semi-major axis:

$$H = \frac{\sigma}{G\rho} \frac{d}{dt}\!\left(\frac{1}{a}\right).$$

**Eccentricity growth rate** — change of $e$ per $e$-fold of hardening:

$$K = -a\,\frac{de}{da}\,.$$

**Acceleration parameter** — dimensionless CoM force:

$$\vec P = -\frac{a}{\sigma}\,\frac{d\vec V}{da} = \frac{\vec F}{GM\rho a H}\,.$$

This is a vector in the orbital plane. Its lab-frame components are
$P_x = \hat x \cdot \vec P$ and $P_y = \hat y \cdot \vec P$. The existing
Python code, however, returns the binary-frame components
$P_{\hat e} = \hat e \cdot \vec P$ and $P_{\hat n} = \hat n \cdot \vec P$
(called `P_x` and `P_y` in the code). The two sets are related by:

$$P_x = P_{\hat e}\cos\varpi - P_{\hat n}\sin\varpi\,,\qquad P_y = P_{\hat e}\sin\varpi + P_{\hat n}\cos\varpi\,.$$

**Precession parameter** — apsidal advance per $e$-fold of hardening:

$$Q = -a\,\frac{d\varpi}{da} = \frac{\sigma\,\dot\varpi}{G\rho a H}\,.$$

The quantities $H$, $K$, $P_{\hat e}$, $P_{\hat n}$, $Q$ returned by the
reweighting code are functions of $(a/a_h,\; e,\; V_{\hat e}/\sigma,\;
V_{\hat n}/\sigma)$:

- The $a/a_h$ dependence enters through the Maxwellian velocity dispersion
  $\sigma$ used for the reweighting.
- The $\vec V$ dependence enters through the shifted Maxwellian (see Section 6).
  The velocity must be projected into the binary frame, $(V_{\hat e},
  V_{\hat n})$, before being passed to the reweighting code; this projection
  requires knowing $\varpi$.
- At $\vec V = 0$, the parameters reduce to the isotropic-reweighting results
  and depend only on $(a/a_h, e)$.

---

## 4. The ODE system

We use the hardening variable

$$\xi \equiv \ln\!\left(\frac{a_h}{a}\right), \qquad a = a_h\, e^{-\xi},$$

as the independent variable. It increases monotonically as the binary hardens;
one $e$-fold of $\xi$ corresponds to one $e$-fold of shrinkage.

The relation to physical time is:

$$\frac{dt}{d\xi} = \frac{\sigma}{G\rho\, a\, H}\,.$$

The system is:

$$\boxed{\frac{de}{d\xi} = K}$$

$$\boxed{\frac{dV_x}{d\xi} = \sigma\,P_x \;+\; \frac{\sigma\, \dot V_{{\rm Ch},x}}{G\rho\, a\, H}}$$

$$\boxed{\frac{dV_y}{d\xi} = \sigma\,P_y \;+\; \frac{\sigma\, \dot V_{{\rm Ch},y}}{G\rho\, a\, H}}$$

$$\boxed{\frac{d\varpi}{d\xi} = Q}$$

Here $P_x$ and $P_y$ are the lab-frame components of $\vec P$ (obtained from
the binary-frame output of the reweighting code by rotating by $\varpi$; see
Section 3), and $\dot{\vec V}_{\rm Ch}$ is the Chandrasekhar deceleration from
distant encounters (Section 5).

$K$ and $Q$ directly give the per-$e$-fold changes in $e$ and $\varpi$, and
$\sigma \vec P$ gives the velocity change per $e$-fold (before the
Chandrasekhar correction).

---

## 5. Chandrasekhar dynamical friction (distant encounters)

The three-body scattering experiments capture all encounters with pericenter
$r_p < r_{p,\max} = 5a$, i.e., impact parameter $b < b_{\max}(u)$ where

$$b_{\max}(u) = r_{p,\max}\sqrt{1 + \frac{2GM}{u^2\, r_{p,\max}}} = 5a\sqrt{1 + \frac{2GM}{5a\,u^2}}$$

and $u$ is the speed of the star relative to the binary. For more distant
encounters ($b > b_{\max}$), the binary acts as a point mass $M$ and the
back-reaction is well described by two-body scattering.

### Integral formula

For a star approaching the binary at relative speed $u$ from direction
$\hat u$ (in the binary rest frame), the two-body drag from impact parameters
$b \in [b_{\max}(u),\, r_{\rm outer}]$ produces an acceleration of the binary
along $\hat u$ equal to

$$\frac{4\pi G^2 M \rho}{u^2}\,\ln\!\left(\frac{r_{\rm outer}}{b_{\max}(u)}\right)$$

per unit phase-space density. Integrating over all stars, weighted by the
shifted Maxwellian $f_{\rm 3D}(\vec u + \vec V)$ (where $\vec u + \vec V$ is
the lab-frame velocity of the star), gives the total Chandrasekhar
deceleration:

$$\boxed{\dot{\vec V}_{\rm Ch} = 4\pi G^2 M \rho \int \frac{\hat u}{u^2}\,\ln\!\left(\frac{r_{\rm outer}}{b_{\max}(u)}\right) f_{\rm 3D}(\vec u + \vec V)\, d^3u\,,}$$

where $r_{\rm outer}$ is an outer cutoff (e.g., the influence radius
$r_i \sim GM/\sigma^2$). This integral has the same structure as the
three-body force integral and can be evaluated numerically in the same way.

### Constant-$\ln\Lambda$ limit

When $\ln\Lambda \equiv \ln(r_{\rm outer}/b_{\max})$ is pulled out of the
integral (i.e., treated as independent of $u$), the angular integration can be
done analytically and the standard Chandrasekhar formula is recovered:

$$\dot{\vec V}_{\rm Ch} = -\frac{4\pi G^2 M \rho\,\ln\Lambda}{V^2}\left[\operatorname{erf}(X) - \frac{2X}{\sqrt\pi}\,e^{-X^2}\right] \hat V\,,$$

where $X = V/(\sqrt 2\,\sigma)$ and $V = |\vec V|$. This approximation is
adequate when $\ln\Lambda$ varies slowly over the range of $u$ that
contributes to the integral. For a quick estimate, $b_{\max}$ can be evaluated
at a characteristic velocity, e.g. $u \sim \sigma$.

### Note

The Chandrasekhar deceleration only affects the CoM velocity, not the internal
orbital parameters $(a, e, \varpi)$, because distant encounters see the binary
as a point mass. Thus $H$, $K$, and $Q$ are computed purely from the
three-body data, while only the velocity equations receive the Chandrasekhar
correction.

---

## 6. Computing evolution parameters from data

### 6.1 Data pipeline

The evolution parameters $H$, $K$, $P_x$, $P_y$, $Q$ are obtained from
numerical three-body scattering experiments:

1. **Scattering experiments** (C++ code): for each $(q, e)$ pair and many
   values of the incoming velocity $v$, run $10^4$ scatterings and record the
   per-particle changes $(\Delta E,\, \Delta\vec v,\, \Delta\vec L,\,
   \Delta\varpi)$. These are stored either as per-particle binary files or as
   spherical-harmonic moments (harmonics `.bin` files).

2. **Maxwellian reweighting** (Python): integrate over the velocity
   distribution to obtain the physical rates $(P,\, \vec\tau,\, \vec F,\,
   \dot\varpi)$ at a given $\sigma$ (equivalently, $a/a_h$) and CoM velocity
   $\vec V$. For $\vec V = 0$, the isotropic Maxwellian suffices. For $\vec V
   \ne 0$, the shifted Maxwellian $f_{\rm 3D}(\vec v + \vec V)$ is expanded in
   spherical harmonics using the plane-wave formula, and the harmonics data is
   used.

3. **Dimensionless parameters**: fold the physical rates into $H$, $K$, $\vec
   P$, $Q$ using the definitions in Section 3.

### 6.2 Handling $\vec V \ne 0$

The harmonics files store the spherical-harmonic moments
$Z_v^{\ell m}$ of each per-particle quantity (energy, velocity, angular
momentum, $\Delta\varpi$) as a function of the incoming direction, for each
velocity bin $v$. The shifted Maxwellian $f(\vec v + \vec V)$ introduces the
exponential factor $e^{-\vec v \cdot \vec V / \sigma^2}$, which is expanded
using the addition theorem:

$$e^{-\vec v \cdot \vec V / \sigma^2} = \sum_{\ell m} 4\pi\, i_\ell(vV/\sigma^2)\, Y_{\ell m}(-\hat V)\, Y_{\ell m}^*(\hat v)\,,$$

where $i_\ell$ is the modified spherical Bessel function of the first kind
and $\vec V$ is the binary centre-of-mass velocity.
This is implemented in `reweight_from_harmonics()` in
`3-body/python/weight-Maxwellian-3D-velocity.py`.

The velocity $\vec V$ passed to the reweighting engine is expressed in code
units ($G = M = a = 1$). The conversion from physical units is $\vec
V_{\rm code} = \vec V / \sqrt{GM/a}$, and correspondingly $\sigma_{\rm code} =
\sigma / \sqrt{GM/a} = \frac{1}{2}\frac{\sqrt q}{1+q}\sqrt{a/a_h}$.

### 6.3 Interpolation in eccentricity

The scattering data exists on a discrete grid of $e$ values (typically $e = 0,
0.1, 0.2, \ldots, 0.9$). As $e$ evolves, we interpolate rates to the current
$e$.

- For a given $q$, load harmonics data for all available $e$ values.
- **Production C integrator (`evolve.c`):** use a **four-point Lagrange**
  stencil on the sorted list of **$e>0$** grid points (stencil size capped by
  how many positive-$e$ samples exist). At each stencil point, evaluate
  $(H, K, P_{\hat e}, P_{\hat n}, Q)$ and their 1-$\sigma$ Monte Carlo
  uncertainties by harmonics reweighting at the current $(a/a_h,\,\vec V)$;
  combine the central values with Lagrange weights $w_k(e)$. Combine
  **uncertainties in quadrature** across the stencil,
  $\sigma_R^2(e) = \sum_k w_k^2(e)\,\sigma_{R,k}^2$, separately for each rate
  $R$, then propagate the $P_{\hat e}$, $P_{\hat n}$ errors to lab-frame
  $P_x$, $P_y$ by the same rotation as the means. (If an $e=0$ dataset exists,
  it is not part of this $e>0$ stencil; see `load_dataset` in `evolve.c`.)
- **Python solver (`evolve.py`):** uses the same Lagrange–stencil logic when
  evaluating rates on the fly; optional precomputed tables are interpolated
  differently depending on build options.

### 6.4 Available data

Harmonics files exist in `3-body/Data/results-precession-3D-velocity-soft/` for:

| $q$ | Available $e$ values |
|-----|---------------------|
| 0.0005 | 0.6 |
| 0.001 | 0.3, 0.6, 0.9 |
| 0.002 | 0.3, 0.6, 0.9 |
| 0.01 | 0.3, 0.6, 0.9 |
| 0.02 | 0.3, 0.6, 0.9 |
| 0.1 | 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| 0.2 | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| 0.3 | 0.9 |
| 0.4 | 0.2, 0.4, 0.5 |
| 0.5 | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| 0.6 | 0, 0.4, 0.6 |
| 0.8 | 0.7 |
| 0.9 | 0.5, 0.6 |
| 1.0 | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |

Full eccentricity coverage ($e = 0$ to $0.9$ in steps of $0.1$) exists for
$q \in \{0.2, 0.5, 1.0\}$.

---

## 7. Notation mapping

The draft paper and the Python analysis code use different symbols for the
same quantities:

| Draft paper | Python code (`weight-Maxwellian-3D-velocity.py`) | Meaning |
|-------------|--------------------------------------------------|---------|
| $P_{\hat e}$ | `P_x` | acceleration parameter, $\hat e$ component |
| $P_{\hat n}$ | `P_y` | acceleration parameter, $\hat n$ component |
| $Q$ | `Q` | dimensionless precession parameter |
| $H$ | `H` | hardening rate |
| $K$ | `K` | eccentricity growth rate |
| $\vec F$ | `F` | physical force on the CoM (binary frame) |
| $\dot\varpi$ | `varpi_dot` | physical precession rate |

The Python functions `reweight()` and `reweight_from_harmonics()` return a
dictionary with keys `H`, `K`, `P_x`, `P_y`, `Q`, `F`, `tau`,
`varpi_dot`, and their uncertainties (`sH`, `sK`, etc.). All vector quantities
(`F`, `tau`, and correspondingly `P_x`, `P_y`) are in the binary frame.

---

## 8. Implementation notes

### 8.1 Code paths (C vs Python)

There are **two** integrators:

- **C (`evolve.c`):** compiled binary `evolve`. Used by the plotting wrapper
  `run_evolve.py`. Always evaluates rates **on the fly** from harmonics data
  and always propagates uncertainties with the **full response-matrix** method
  (Section 9.3). There is **no** `--diagonal` mode in C.
- **Python (`evolve.py`):** uses `scipy.integrate.solve_ivp`. Supports full
  covariance (default) and a **diagonal** shortcut (`--diagonal` /
  `full_covariance=False`), and can use **precomputed** rate tables for speed.

Default figures produced via `run_evolve.py` follow the **C** behaviour.

### 8.2 ODE integration

The independent variable is $\xi$ and the seven primary unknowns are
$(e,\, \tilde V_x,\, \tilde V_y,\, \varpi,\, \tilde t,\, \tilde x,\,
\tilde y)$.

**C (`evolve.c`):** embedded **Dormand–Prince RK45** with adaptive steps
(`rk45_solve`). The local error estimate uses only the **first seven**
components; the appended block that stores the response matrix $F$ is advanced
along with the state but does not enter the error norm. The extended state has
$7 + 7 \times 5 N_e$ components, where $N_e$ is the number of **eccentricity
samples** at the chosen $q$ that participate in the $e>0$ stencil (Section 6.3).
If neither `--xi-start` nor `--a0` is given, $\xi_0 = 0$ ($a/a_h = 1$).
Integration may terminate early when $e$ reaches $e_{\max,\mathrm{grid}} - 0.01$
(event in `full_event` / `v0_event`).

**Python (`evolve.py`):** `scipy.integrate.solve_ivp` (e.g. RK45 or DOP853).
In **full covariance** mode, the same $7 \times 5 N_e$ response matrix is
appended. In **diagonal** mode, seven scalar uncertainties are appended instead
(14-component ODE).

**$V=0$ reference:** **C** integrates $(e,\tilde t)$ plus a $2 \times 5N_e$
response block (`v0_rhs`). **Python** uses `solve_simple` with the analogous
reduced Jacobian (Section 9.3).

**Saved uncertainties:** both paths can report **marginal** 1-$\sigma$
standard deviations $\sigma_i = \sqrt{(FF^\top)_{ii}}$ for each primary
component. The C writer `write_full` outputs these as `sig_*` columns; the full
$7\times 7$ covariance is not serialized.

### 8.3 Evaluating the RHS

At each call to the RHS function:

1. From the current $\xi$, compute $a = a_h\, e^{-\xi}$ and $a/a_h$.
2. Project the lab-frame velocity into the binary frame using $\varpi$:
   $V_{\hat e} = V_x\cos\varpi + V_y\sin\varpi$,
   $V_{\hat n} = -V_x\sin\varpi + V_y\cos\varpi$.
3. Express both velocity and $\sigma$ in code units.
4. Obtain $H$, $K$, $P_{\hat e}$, $P_{\hat n}$, $Q$ and their uncertainties:
   **C:** `reweight` / `compute_rates` in `evolve.c` (harmonics pipeline
   duplicated from the Python reference). **Python:** `reweight_from_harmonics()`
   or interpolation from a precomputed table.
5. Interpolate in $e$ as in Section 6.3 (four-point Lagrange on the $e>0$
   grid in C; quadrature combination of rate uncertainties at stencil points).
6. Rotate $\vec P$ to the lab frame:
   $P_x = P_{\hat e}\cos\varpi - P_{\hat n}\sin\varpi$,
   $P_y = P_{\hat e}\sin\varpi + P_{\hat n}\cos\varpi$.
7. Compute the Chandrasekhar deceleration $\dot{\vec V}_{\rm Ch}$ (Section 5)
   and add its contribution to the velocity derivatives.

### 8.4 Precomputation vs. on-the-fly evaluation

- **C (`evolve.c`):** always **on-the-fly** harmonics reweighting at each RHS
  evaluation (numerical integration over velocity bins $N_v \sim O(10^3)$ per
  call).

- **Python (`evolve.py`):** either the same on-the-fly
  `reweight_from_harmonics()` or a **precomputed** lookup table on
  $(a/a_h,\, e,\, V_{\hat e}/\sigma,\, V_{\hat n}/\sigma)$ for faster RHS
  evaluation.

### 8.5 Physical scales

All three-body data are in code units ($G = M = a = 1$). To convert to
physical units, one specifies:

- $M$ (total binary mass)
- $\sigma$ (velocity dispersion of the environment)
- $\rho$ (density of the environment)

These fix $a_h = G\mu/(4\sigma^2)$ and the hardening timescale $T_{\rm hard} =
\sigma/(G\rho\, a_h\, H)$. Alternatively, one can work entirely in
dimensionless variables: $a/a_h$, $e$, $V/\sigma$, and $t/T_{\rm hard}$.

### 8.6 Outputs and plotting (`run_evolve.py`)

`run_evolve.py` runs the C binary, reads `*_full.dat` / `*_V0.dat`, and plots
mean trajectories with shaded **marginal** bands using `sig_*`. Centre-of-mass
ellipses use **axis-aligned** semi-axes $\sigma_x$ and $\sigma_y$. If the
off-diagonal $x$–$y$ entry of $C = FF^\top$ is nonzero, the true
$1$-$\sigma$ contour in the CoM plane is a **tilted** ellipse; the plot is
then a convenient approximation unless extended to use the full $2\times 2$
block of $C$.

---

## 9. Uncertainty propagation: variational equations

A compact, paper-oriented summary (covariance definition, noise model, and
reported marginals) is in
[UNCERTAINTY_METHODS.md](UNCERTAINTY_METHODS.md).

The evolution parameters $H$, $K$, $\vec P$, $Q$ carry 1-$\sigma$ Monte Carlo
uncertainties $(\sigma_H,\, \sigma_K,\, \sigma_{P_x},\, \sigma_{P_y},\,
\sigma_Q)$ from the three-body scattering experiments. These uncertainties
propagate into the integrated state variables $(e,\, V_x,\, V_y,\, \varpi,\,
t,\, x,\, y)$.

### 9.1 Why naïve sigma accumulation fails

A tempting first approach is to integrate the absolute uncertainty on each
state variable as

$$\frac{d\sigma_Y}{d\xi} = \sigma_{f_Y}\,,$$

where $\sigma_{f_Y}$ is the 1-$\sigma$ uncertainty on the RHS $f_Y$ of
the ODE for $Y$.  This treats the uncertainty at each step as an
independent additive forcing, giving $\sigma_Y \propto \xi$ (linear growth).

The problem is that this ignores the **state-dependence** of the RHS.  When
$f_Y$ depends on $Y$ itself—as it does for the velocity equations through
Chandrasekhar drag—perturbations in $Y$ feed back into $f_Y$, and this
feedback is missing from the naïve accumulation.

**Toy model.** Consider $dy/dx = -A\,y$ where $A > 0$ has a 1-$\sigma$
uncertainty $\sigma_A$.  The exact solution is $y(x) = y_0\, e^{-Ax}$, and
the exact uncertainty from propagating $\sigma_A$ is

$$\sigma_y(x) = \sigma_A\, x\, |y(x)| = \sigma_A\, x\, y_0\, e^{-Ax}\,.$$

This *decays* with $y$: as the system damps, the uncertainty damps too.

The naïve accumulation instead integrates
$d\sigma_y/dx = \sigma_A\, |y| = \sigma_A\, y_0\, e^{-Ax}$, giving

$$\sigma_y^{\rm naïve}(x) = \frac{\sigma_A\, y_0}{A}\,\bigl(1 - e^{-Ax}\bigr)
\;\xrightarrow{x\to\infty}\; \frac{\sigma_A\, y_0}{A}\,.$$

This saturates to a constant instead of decaying—wrong by a factor that
grows as $e^{Ax}$ relative to the true uncertainty.  The missing ingredient
is the **feedback** $\partial f / \partial y = -A$, which damps perturbations
at the same rate as it damps the solution.

### 9.2 The variational equation

For a general ODE $dy/d\xi = f(y,\xi)$, the linearised perturbation
$\delta y$ due to parameter uncertainty satisfies

$$\frac{d(\delta y)}{d\xi} = \frac{\partial f}{\partial y}\,\delta y + \delta f_{\rm noise}\,,$$

where the first term is the **Jacobian feedback** and $\delta f_{\rm noise}$
is the forcing from rate uncertainties.  For a system of equations, $y$ is a
vector and $\partial f/\partial y$ is the Jacobian matrix $J$.

In the toy model, $J = -A$ and the variational equation is
$d(\delta y)/dx = -A\,\delta y + \sigma_A\, y$, which has the solution

$$\delta y(x) = e^{-Ax} \int_0^x \sigma_A\, y(x')\, e^{Ax'}\, dx'
= \sigma_A\, x\, y_0\, e^{-Ax}\,,$$

recovering the correct result.

### 9.3 Response-matrix model (C default; Python full-covariance default)

**C (`evolve.c`):** uncertainty propagation is **always** the response-matrix
method below. **Python (`evolve.py`):** the same model is the default; an
optional **diagonal** mode is described at the end of this subsection.

#### Noise correlation structure

Rates and their errors come from Monte Carlo scattering data at **discrete
eccentricity samples** (one harmonics dataset per grid $e$). Index these samples
by $k = 1,\ldots,N_e$ in the order used for the Lagrange stencil (in C, only
**$e>0$** samples; $N_e$ matches `g_nep` in `evolve.c`). Two properties set the
correlation structure:

1. **Independent between samples.**  Each eccentricity sample comes from a
   separate Monte Carlo campaign; sampling errors at $k$ are independent of
   those at $j \ne k$.

2. **Systematic within a sample.**  When the same sample is evaluated at
   different velocities or $a/a_h$ (reweighting the same scattering outcomes),
   the Monte Carlo error is a fixed but unknown property of that sample—not a
   new random draw. The *magnitude* of the quoted uncertainty may change with
   reweighting, but the underlying error *realization* is the same.

#### Response-matrix equation

For each eccentricity sample $k$ and each rate
$r \in \{H, K, P_{\hat e}, P_{\hat n}, Q\}$, define a **response vector**
$\vec f_{k,r}(\xi) \in \mathbb R^7$ satisfying

$$\boxed{\frac{d\vec f_{k,r}}{d\xi} = J\,\vec f_{k,r} + w_k\bigl(e(\xi)\bigr)\;\sigma_{R_{k,r}}\bigl(V(\xi)\bigr)\;\vec b_r\,,
\qquad \vec f_{k,r}(0) = 0\,.}$$

Here $J$ is the $7\times 7$ Jacobian (Section 9.4), $w_k(e)$ is the Lagrange
interpolation weight for sample $k$ at the current $e$, $\sigma_{R_{k,r}}$ is
the 1-$\sigma$ uncertainty on rate $r$ for sample $k$ at the current velocity
(binary frame), and $\vec b_r$ is column $r$ of the loading matrix $B$
(Section 9.5). Samples outside the current stencil have $w_k = 0$, so they
receive no new forcing, but their accumulated response vectors still evolve via
$J\,\vec f$.

Collecting all response vectors into a $7 \times N_{\rm noise}$ matrix
$F$ whose columns are $\vec f_{k,r}$ for global sample index
$k = 0,\ldots,N_e-1$ and $r \in \{0,\ldots,4\}$, with
$N_{\rm noise} = 5\, N_e$, the equation is

$$\frac{dF}{d\xi} = J\, F + G\,,$$

where $G$ is the $7 \times N_{\rm noise}$ loading matrix (Section 9.5).

The full $7\times 7$ **covariance matrix** is

$$C = F\, F^\top = \sum_{k,r} \vec f_{k,r}\,\vec f_{k,r}^\top\,,$$

assuming independent unit-variance normals per column (variance absorbed into
$G$). Marginal uncertainties are $\sigma_Y = \sqrt{C_{YY}}$; off-diagonal entries
of $C$ give correlations between state components.

**State vector size:** $7 + 7 \times 5 N_e$. For example, $q=1$ with ten
eccentricity datasets including $e=0$ yields $N_e = 9$ (only $e>0$ samples
count toward `g_nep` in `evolve.c`), i.e. $7 + 315 = 322$ components; if no
$e=0$ file exists, $N_e$ equals the number of datasets. The C integrator uses
an explicit adaptive RK45; Python uses `solve_ivp`.

**Early-time behaviour.**  For small $\xi$, $F \approx G\,\xi$, so
$\sigma_Y \propto \xi$ (linear growth) and the signal-to-noise ratio
$f_Y / \sigma_{f_Y}$ is constant from the first step—matching the physical
expectation that a force with a definite sign produces a definite
displacement.

**Stencil transitions.**  When $e$ evolves enough that a new sample $j$ enters
the Lagrange stencil, its response vectors start from zero and build up as
$w_j$ grows. There is no artificial correlation between the new sample’s
uncertainty and uncertainty accumulated from other samples.

#### $V{=}0$ reference

**C:** `v0_rhs` uses the same response-matrix bookkeeping on the reduced state
$(e, \tilde t)$. **Python:** `solve_simple` is the analogue. The Jacobian is

$$J = \begin{pmatrix}
\partial K / \partial e & 0 \\
-e^\xi H^{-2}\,\partial H/\partial e & 0
\end{pmatrix},$$

and the loading is $G_{0,\,5k+1} = w_k\,\sigma_{K_k}$ (eccentricity from $K$
noise) and $G_{1,\,5k+0} = -w_k\,\sigma_{H_k}\,e^\xi / H^2$ (time from $H$
noise). The $2\times 2$ covariance is $C = F\,F^\top$ as usual.

This ensures an apples-to-apples comparison of the $V{=}0$ and full-solver
uncertainty bands.

#### Why the diagonal model overcounts at stencil transitions (Python)

The scalar variational equation
$d\sigma_Y/d\xi = J_{YY}\,\sigma_Y + \sigma_{f_Y}$
uses the *aggregate* Lagrange-interpolated uncertainty
$\sigma_{f_Y} = \sqrt{\sum_k w_k^2\,\sigma_{R_k}^2}$ as the forcing at
every step.  This correctly treats each step's noise as systematic
(linear growth), but it implicitly assumes the aggregate noise at
consecutive steps is *perfectly correlated*—the same systematic bias
accumulating coherently.

In reality, when the Lagrange stencil shifts (a new sample enters and an old
one exits), the aggregate at step $n$ and at step $n{+}1$ are composed of
*partially different* samples.  The new sample contributes a genuinely
independent bias that should be added in *quadrature*, not coherently.
Consecutive 4-point stencils share 3 of 4 samples, so the overcounting is
moderate per transition but compounds over many shifts.  For a fine
eccentricity grid (e.g., $q{=}0.2$ with many samples and ${\sim}40{-}50$
stencil transitions), the overcounting factor can be of order
$\sqrt{N_{\rm transitions}}$.

The response-matrix model avoids this: each eccentricity sample has its own
response vector, accumulated continuously as $w_k(e)$ changes. Samples shared
across consecutive stencils are a single noise source with no double-counting.

#### Diagonal fallback (Python `evolve.py` only: `--diagonal`)

Only the seven marginal uncertainties $\sigma_Y$ are tracked (14-component
state: 7 dynamical + 7 uncertainties). Each obeys

$$\frac{d\sigma_Y}{d\xi} = J_{YY}\,\sigma_Y + \sigma_{f_Y}\,.$$

This uses aggregate lab-frame uncertainties and neglects off-diagonal coupling
in $C$, but retains diagonal Jacobian feedback (notably Chandrasekhar damping).
It requires fewer finite-difference Jacobian columns than full covariance, but
overcounts noise at stencil transitions as above. **Not available in C.**

### 9.4 Jacobian structure

The Jacobian $J_{ij} = \partial f_i / \partial y_j$ is a $7\times 7$
matrix. **C and Python (full covariance):** all seven columns are computed.
**Python diagonal mode only:** only $J_{00}$, $J_{11}$, $J_{22}$ are used for
the scalar $\sigma_Y$ equations (with $J_{33} \approx 0$).

**Column 0 (eccentricity derivatives)** — analytic from the Lagrange
interpolant.  The interpolation weights
$w_k(e) = \prod_{j \ne k} (e - e_j)/(e_k - e_j)$ have derivatives

$$\frac{dw_k}{de} = \sum_{m \ne k}
\frac{1}{e_k - e_m}\,\prod_{\substack{j \ne k \\ j \ne m}}
\frac{e - e_j}{e_k - e_j}\,,$$

giving $\partial R / \partial e = \sum_k (dw_k/de)\, R_k$ for any rate
$R \in \{H, K, P_x, P_y, Q\}$.  The Jacobian entries are:

| Row | Entry |
|-----|-------|
| 0 ($e$) | $\partial K / \partial e$ |
| 1 ($\tilde V_x$) | $\partial P_x/\partial e - (\mathrm{Ch}_x / H)\,\partial H/\partial e$ |
| 2 ($\tilde V_y$) | $\partial P_y/\partial e - (\mathrm{Ch}_y / H)\,\partial H/\partial e$ |
| 3 ($\varpi$) | $\partial Q / \partial e$ |
| 4 ($\tilde t$) | $-e^\xi H^{-2}\,\partial H/\partial e$ |
| 5 ($\tilde x$) | $\tilde V_x \cdot J_{40}$ |
| 6 ($\tilde y$) | $\tilde V_y \cdot J_{40}$ |

These use the same stencil values already evaluated for interpolation, so the
cost is essentially zero.

**Columns 1–2 (velocity derivatives)** — central finite differences at
$V_x \pm \delta$ (or $V_y \pm \delta$), capturing all five rates and the
Chandrasekhar terms. In **C**, $\delta = 0.01$ (in units of $\tilde V$). Each
column needs two extra rate evaluations plus Chandrasekhar at perturbed
velocities. Rows 5–6 use the chain rule:
$\partial(V_x \, dt/d\xi)/\partial V_x = dt/d\xi + V_x\,\partial(dt/d\xi)/\partial V_x$.

**Column 3 (precession derivatives)** — central finite differences at
$\varpi \pm \delta_\varpi$ (C: $\delta_\varpi = 0.01$). This captures the
dependence of rates on $\varpi$ through projection into the binary frame.
**Python diagonal mode** omits this column ($J_{33} \approx 0$).

**Columns 4–6** are zero: $t$, $x$, $y$ do not appear in any RHS.

**Cost:** Full covariance: nominal `compute_rates` plus four velocity
perturbations plus two $\varpi$ perturbations ($7$ rate evaluations per RHS
in C). Python diagonal mode: $1 + 4 = 5$ evaluations.

### 9.5 Loading matrix and noise columns

The $7\times 5$ **loading matrix** $B$ maps a unit perturbation in the five
binary-frame rates $(H,\, K,\, P_{\hat e},\, P_{\hat n},\, Q)$ to the
seven state derivatives, incorporating the $\varpi$-rotation from binary
frame to lab frame:

$$B = \begin{pmatrix}
0 & 1 & 0 & 0 & 0 \\
-\mathrm{Ch}_x/H & 0 & \cos\varpi & -\sin\varpi & 0 \\
-\mathrm{Ch}_y/H & 0 & \sin\varpi & \phantom{-}\cos\varpi & 0 \\
0 & 0 & 0 & 0 & 1 \\
-e^\xi/H^2 & 0 & 0 & 0 & 0 \\
-\tilde V_x\,e^\xi/H^2 & 0 & 0 & 0 & 0 \\
-\tilde V_y\,e^\xi/H^2 & 0 & 0 & 0 & 0
\end{pmatrix}$$

The loading matrix $G$ (size $7 \times N_{\rm noise}$) has mostly zero columns;
only eccentricity samples in the current Lagrange stencil contribute:

$$G_{:,\,5k+r} = w_k(e)\;\sigma_{R_{k,r}}(V)\;\vec b_r$$

for sample index $k$ (globally ordered as in the integrator) and
$r \in \{0,\ldots,4\}$, where $\sigma_{R_{k,r}}$ is the 1-$\sigma$ uncertainty
on rate $r$ for that sample at the current velocity (binary frame).

**Approximation:** The five rate uncertainties within each eccentricity sample
are treated as **independent** noise sources. In reality $H$, $K$,
$P_{\hat e}$, $P_{\hat n}$, $Q$ from the same data are correlated; a full
$5\times 5$ within-sample covariance from `reweight_from_harmonics` is not
used. This is secondary to the between-sample independence structure.

**Python diagonal mode:** aggregate forcing uses
$\sigma_{f_e} = \sigma_K$,
$\sigma_{f_{V_x}} = \sqrt{\sigma_{P_x}^2 + (\mathrm{Ch}_x\,\sigma_H/H)^2}$,
etc., with Lagrange-combined (interpolated) uncertainties.

### 9.6 Quantities without self-feedback

Rows 4–6 ($t$, $x$, $y$) have zero diagonal Jacobian elements—these
variables do not appear in their own RHS. In **full covariance** (C default),
their uncertainties receive contributions through the off-diagonal elements
of $C = F\,F^\top$, which capture correlations (e.g., velocity vs. position).
**Python diagonal mode** uses the approximate scalar forcings:

- $d\sigma_t/d\xi = e^\xi\, \sigma_H / H^2$.
- $d\sigma_x/d\xi \approx \sigma_{V_x}\, dt/d\xi + |V_x|\,\sigma_t'$.
- $d\sigma_y/d\xi$ analogous.
- $d\sigma_\varpi/d\xi = \sigma_Q$.
