# Uncertainty propagation for binary evolution (methods summary)

This note is written for reuse in a paper methods section. It describes the
covariance of the evolved state induced by Monte Carlo uncertainties on the
three-body rates, in the same dimensionless formulation as the ODE in
[PHYSICS.md](PHYSICS.md) (independent variable $\xi = \ln(a_h/a)$, state in
code/scaled units).

---

## State and rates

Let the dynamical state be
$\mathbf{y} = (e,\,\tilde V_x,\,\tilde V_y,\,\varpi,\,\tilde t,\,\tilde x,\,\tilde y)^\top$:
eccentricity, centre-of-mass velocity in units of $\sigma$, longitude of
periapsis, and time / CoM position in units of the hardening time
$T_{\mathrm{hard}} = \sigma/(G\rho\, a_h\, H)$ and $\sigma T_{\mathrm{hard}}$
as in the code.

The three-body pipeline supplies five dimensionless rates in the **binary**
frame,
$\mathbf{R} = (H,\, K,\, P_{\hat e},\, P_{\hat n},\, Q)^\top$, each evaluated
by Maxwellian reweighting of the scattering ensemble at discrete **eccentricity
samples** $k = 0,\ldots,N_e-1$. For every $(k,r)$ the reweighting also returns a
1-$\sigma$ Monte Carlo standard error $\sigma_{R_{k,r}}(\mathbf{V})$ (in
binary-frame coordinates for the $P$ components), generally depending on
$a/a_h$ and the CoM velocity because reweighting changes effective weights.

---

## Covariance of the integrated state

We seek the $7\times 7$ covariance matrix $\mathbf{C}(\xi)$ of the state
$\mathbf{y}(\xi)$ induced by uncertainty in the $\sigma_{R_{k,r}}$. The model
assumes **independent** standard-normal shocks $\eta_{k,r}$:
$\mathbb{E}[\eta_{k,r}\,\eta_{k',r'}] = \delta_{kk'}\delta_{rr'}$. Each
$(k,r)$ is one scalar noise source tied to one eccentricity sample and one rate
component.

**Between-sample independence** reflects separate Monte Carlo campaigns per
eccentricity. **Within a sample**, the five rates are in reality correlated;
the implementation approximates them by five **independent** scalar noise
sources, ignoring the $5\times 5$ within-sample covariance. This is a minor
approximation compared to treating each eccentricity sample as a separate bias
(Section 9.5 of [PHYSICS.md](PHYSICS.md)).

---

## Interpolation in eccentricity

At runtime $e(\xi)$ lies between grid values. A Lagrange stencil gives weights
$w_k(e)$ for samples $k$ in the stencil ($\sum_k w_k = 1$). The **effective**
amplitude of source $(k,r)$ entering the ODE at $\xi$ is
$w_k(e)\,\sigma_{R_{k,r}}(\mathbf{V}(\xi))$. Samples outside the stencil have
$w_k=0$ and receive no instantaneous forcing, but their **accumulated**
response columns still evolve (see below), which avoids double-counting when
the stencil slides along the $e$ grid.

Interpolated central values of the rates use the same $w_k$. Interpolated
**variances** for diagonal / aggregate uses combine stencil-point errors in
quadrature, $\sigma_R^2(e) = \sum_k w_k^2\,\sigma_{R,k}^2$, before mapping
$P_{\hat e},P_{\hat n}$ uncertainties to the lab frame.

---

## Loading: from rate noise to $\mathrm{d}\mathbf{y}/\mathrm{d}\xi$

Let $\mathbf{f}(\mathbf{y},\xi)$ be the deterministic RHS of the seven ODEs
(including Chandrasekhar terms). A small perturbation $\delta\mathbf{R}$ in the
binary-frame rates maps linearly to a perturbation of $\mathrm{d}\mathbf{y}/\mathrm{d}\xi$
via a $7\times 5$ matrix $\mathbf{B}(\mathbf{y},\xi)$ (rotation of
$P_{\hat e},P_{\hat n}$ to lab $P_x,P_y$, and the chain through $H$ in the
velocity and time equations). With columns $\mathbf{b}_r$,
$r=0,\ldots,4$ for $(H,K,P_{\hat e},P_{\hat n},Q)$,

$$\delta\!\left(\frac{\mathrm{d}\mathbf{y}}{\mathrm{d}\xi}\right)
  = \mathbf{B}\,\delta\mathbf{R}\,.$$

Column $r$ is $\mathbf{b}_r$ (explicit form in [PHYSICS.md](PHYSICS.md) §9.5).

---

## Coupled propagation: response matrix and covariance

Linearising $\mathrm{d}\mathbf{y}/\mathrm{d}\xi = \mathbf{f}(\mathbf{y},\xi)$
gives the Jacobian
$\mathbf{J}(\mathbf{y},\xi) = \partial \mathbf{f}/\partial \mathbf{y}$.
For each noise source $(k,r)$ define a **response vector**
$\mathbf{f}_{k,r}(\xi) \in \mathbb{R}^7$ with

$$\frac{\mathrm{d}\mathbf{f}_{k,r}}{\mathrm{d}\xi}
  = \mathbf{J}\,\mathbf{f}_{k,r} + w_k(e)\,\sigma_{R_{k,r}}(\mathbf{V})\,\mathbf{b}_r\,, \qquad \mathbf{f}_{k,r}(0) = \mathbf{0}\,.$$

Stack columns into $\mathbf{F} \in \mathbb{R}^{7\times N_{\mathrm{noise}}}$,
$N_{\mathrm{noise}} = 5 N_e$:

$$\frac{\mathrm{d}\mathbf{F}}{\mathrm{d}\xi}
  = \mathbf{J}\,\mathbf{F} + \mathbf{G}\,,$$

where $\mathbf{G}$ has columns
$w_k(e)\,\sigma_{R_{k,r}}(\mathbf{V})\,\mathbf{b}_r$
in the global ordering of $(k,r)$.

With independent unit-variance $\eta_{k,r}$, the state perturbation is
$\delta\mathbf{y} = \mathbf{F}\,\boldsymbol{\eta}$ and the **covariance** is

$$\mathbf{C}(\xi) = \mathbf{F}(\xi)\,\mathbf{F}(\xi)^\top
  = \sum_{k,r} \mathbf{f}_{k,r}\,\mathbf{f}_{k,r}^\top\,.$$

**Marginal** 1-$\sigma$ uncertainties are
$\sigma_{y_i} = \sqrt{(\mathbf{C})_{ii}}$. Off-diagonal entries encode
correlations (e.g. between CoM velocity and position).

---

## Jacobian $\mathbf{J}$

- **$\partial/\partial e$:** analytic derivatives of the Lagrange weights and
  hence of the interpolated rates; zero extra rate evaluations.
- **$\partial/\partial \tilde V_x$, $\partial/\partial \tilde V_y$:** symmetric
  finite differences at step $\delta_V = 0.01$ (in $\tilde V$ units) in the
  production C integrator, including Chandrasekhar terms.
- **$\partial/\partial\varpi$:** symmetric differences at
  $\delta_\varpi = 0.01\,\mathrm{rad}$.

---

## Reported quantities and figures

Time series and tables typically show means $\mathbf{y}(\xi)$ with **marginal**
bands $\pm\sigma_{y_i}$. The full matrix $\mathbf{C}$ is not required for those
bands. Two-dimensional CoM plots sometimes draw **axis-aligned** ellipses from
$\sigma_x,\sigma_y$; if $(\mathbf{C})_{xy}\neq 0$, the $1$-$\sigma$ contour is
a tilted ellipse obtained from the upper-left $2\times 2$ block of
$\mathbf{C}$.

---

## $V=0$ reference track

For comparison at vanishing CoM speed, the same noise bookkeeping applies to
the reduced state $(e,\tilde t)^\top$ with a $2\times 2$ Jacobian and only the
rows of $\mathbf{B}$ that feed $K$ and $H$ into $\mathrm{d}e/\mathrm{d}\xi$ and
$\mathrm{d}\tilde t/\mathrm{d}\xi$. The covariance is again
$\mathbf{C}^{(2)} = \mathbf{F}^{(2)}(\mathbf{F}^{(2)})^\top$.

---

## Relation to code

The production C binary (`evolve.c`) implements exactly this response-matrix
construction with fixed full covariance. A separate Python solver
(`evolve.py`) can optionally replace $\mathbf{C}$ by a diagonal evolution of
the $\sigma_{y_i}$ (faster, but can overcount uncertainty when the Lagrange
stencil changes); details are in [PHYSICS.md](PHYSICS.md) §9.
