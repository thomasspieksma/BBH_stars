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
0.1, 0.2, \ldots, 0.9$). As $e$ evolves, we need to interpolate. Strategy:

- For a given $q$, load harmonics data for all available $e$ values.
- At each ODE evaluation, compute $H$, $K$, $P_x$, $P_y$, $Q$ at the two (or
  more) nearest grid values of $e$, and interpolate to the current $e$.
- Linear interpolation in $e$ should be adequate given the typical grid spacing
  of $\Delta e = 0.1$.

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

### 8.1 ODE integration

Use `scipy.integrate.solve_ivp` with an adaptive-step method (e.g., RK45 or
DOP853). The independent variable is $\xi$ and the state vector is $\vec y =
(e,\, V_x,\, V_y,\, \varpi)$ (four components). Physical time can be
recovered by integrating $dt/d\xi = \sigma/(G\rho\, a\, H)$ alongside the
main system.

### 8.2 Evaluating the RHS

At each call to the RHS function:

1. From the current $\xi$, compute $a = a_h\, e^{-\xi}$ and $a/a_h$.
2. Project the lab-frame velocity into the binary frame using $\varpi$:
   $V_{\hat e} = V_x\cos\varpi + V_y\sin\varpi$,
   $V_{\hat n} = -V_x\sin\varpi + V_y\cos\varpi$.
3. Express both velocity and $\sigma$ in code units.
4. Call `reweight_from_harmonics()` (or interpolate from a precomputed table)
   to obtain $H$, $K$, $P_{\hat e}$ (`P_x`), $P_{\hat n}$ (`P_y`), $Q$
   (`Q`).
5. If interpolating in $e$: evaluate at the two nearest grid $e$ values and
   interpolate.
6. Rotate $\vec P$ to the lab frame:
   $P_x = P_{\hat e}\cos\varpi - P_{\hat n}\sin\varpi$,
   $P_y = P_{\hat e}\sin\varpi + P_{\hat n}\cos\varpi$.
7. Compute the Chandrasekhar deceleration $\dot{\vec V}_{\rm Ch}$ (Section 5)
   and add its contribution to the velocity derivatives.

### 8.3 Precomputation vs. on-the-fly evaluation

Two strategies are possible:

- **On-the-fly**: call `reweight_from_harmonics()` at every RHS evaluation.
  The harmonics reweighting is fast (matrix operations, no Monte Carlo), so
  this is feasible. However, each call involves a numerical integral over $v$,
  which takes $O(N_v)$ time with $N_v \sim 2000$ velocity bins.

- **Precomputed table**: before integration, build a lookup table of $H$, $K$,
  $P_x$, $P_y$, $Q$ on a grid of $(a/a_h,\, e,\, V_{\hat e}/\sigma,\,
  V_{\hat n}/\sigma)$, and use multi-dimensional interpolation during the ODE
  integration. Faster per RHS evaluation but requires upfront investment and
  care with grid resolution.

The on-the-fly approach is simpler and recommended for an initial
implementation. A precomputed table can be added later for performance if
needed.

### 8.4 Physical scales

All three-body data are in code units ($G = M = a = 1$). To convert to
physical units, one specifies:

- $M$ (total binary mass)
- $\sigma$ (velocity dispersion of the environment)
- $\rho$ (density of the environment)

These fix $a_h = G\mu/(4\sigma^2)$ and the hardening timescale $T_{\rm hard} =
\sigma/(G\rho\, a_h\, H)$. Alternatively, one can work entirely in
dimensionless variables: $a/a_h$, $e$, $V/\sigma$, and $t/T_{\rm hard}$.
