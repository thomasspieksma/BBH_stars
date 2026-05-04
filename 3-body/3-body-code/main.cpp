// =============================================================================
//  3-body-scattering : single-file Monte-Carlo scattering integrator
//  ----------------------------------------------------------------------------
//
//  PHYSICS
//  -------
//  Simulates the scattering of a massless test particle by a Keplerian binary.
//  The two binary members of mass m1 = 1/(1+q) and m2 = q/(1+q) follow a fixed
//  Keplerian orbit with eccentricity `e` and semi-major axis a = 1; the test
//  particle is released from r = r_sphere along a random direction with random
//  impact parameter b in [0, b_max(v_inf, rp_max)] and integrated until either
//  it escapes (r > r_sphere with E > 0 and r·v > 0) or a stopping criterion is
//  hit. For each escape the code records the asymptotic energy change ΔE,
//  velocity kick Δv = Δv_inf, angular-momentum change ΔL, total time Δt and
//  the integrated apsidal-precession angle Δϖ ("delta_varpi", from Gauss'
//  perturbation equations).
//
//  Per velocity bin v_inf the simulation runs N independent samples. The set
//  of velocities is log-spaced over a v-range that depends on q.
//
//  UNITS  : G = 1, M_total = 1, a = 1  ⇒  binary period = 2π.
//
//  COMPILE
//  -------
//      g++ -O3 -fopenmp -std=c++17 main_merged.cpp -o scattering
//      # On macOS with Homebrew:  CXX=$(brew --prefix llvm)/bin/clang++
//
//  RUN  (default build)
//  --------------------
//      ./scattering <q> <e>
//          q : mass ratio m2/m1   (0 < q ≤ 1)
//          e : binary eccentricity (0 ≤ e < 1)
//
//      Example:  ./scattering 0.5 0.6
//
//  RUN  (with ENABLE_HARMONIC_OUTPUT = true)
//  -----------------------------------------
//      ./scattering <q> <e> [N_v] [N] [l_max]
//          N_v   : number of velocity bins (default 2000)
//          N     : particles per bin (default 10000)
//          l_max : >0 → write harmonics file only (default 10)
//                  =0 → write per-particle binary file only
//                  <0 → write both files (uses |l_max|)
//
//  OUTPUT (always written)
//  -----------------------
//      q=<q>_e=<e>_Tcut=<Tmax>.txt     One row per velocity. Columns:
//          v  ⟨ΔE⟩  SEM_ΔE  ⟨ΔT⟩  SEM_ΔT  ⟨Δvx⟩  SEM_Δvx  …  ⟨ΔLz⟩  SEM_ΔLz
//          ⟨Δϖ⟩  SEM_Δϖ  N_resolved
//
//  OUTPUT (when ENABLE_HARMONIC_OUTPUT = true)
//  -------------------------------------------
//      particles_q=<q>_e=<e>.bin   Per-particle binary records.
//          Header (little-endian native): q, e, rp_max, r_sphere (4 doubles),
//                                         N_v, N (2 ints).
//          Per velocity:  v (double), n_esc (int),
//                         then n_esc records of 12 doubles each:
//                         (v_in_x, v_in_y, v_in_z,
//                          dE, dvx, dvy, dvz,
//                          dLx, dLy, dLz,
//                          dT, delta_varpi).
//
//      harmonics_q=<q>_e=<e>.bin   Real-spherical-harmonic moments.
//          Header: q, e, rp_max, r_sphere (doubles), N_v, N, l_max (ints).
//          Per velocity:  v (double), n_esc (int),
//                         M[8 × n_sh] doubles  (first moments  ⟨X · Y_lm⟩),
//                         S[8 × n_sh] doubles  (second moments ⟨X² · Y_lm⟩),
//                         where n_sh = (l_max+1)² and the 8 quantities X are
//                         (dE, dvx, dvy, dvz, dLx, dLy, dLz, delta_varpi).
//                         Y_lm is evaluated at the incoming velocity direction.
//
//  PARALLELISM : the loop over velocity bins is OpenMP-parallel
//                (#pragma omp parallel for schedule(dynamic)). Set the number
//                of threads with OMP_NUM_THREADS.
//
//  =============================================================================
//  OPTIONAL FEATURES — flip to `true` to enable, recompile.
//  -----------------------------------------------------------------------------
//   • ENABLE_BONETTI_CONDITION_3
//       Adds a third stopping criterion (Rasskazov et al. 2019; Bonetti et al., 2020): integration is
//       aborted once the particle has spent more than `max_time_inside` units
//       of time inside r_sphere. Also rescales the global Tmax to a dynamical
//       timescale  Tmax = 8·(1+q)³ / q^{3/2} · 1e10/49076.
//
//   • ENABLE_HARMONIC_OUTPUT
//       Adds a real-spherical-harmonic decomposition of the scattering response
//       and writes the two binary files described above. Also enables the
//       extended command-line interface  q e [N_v] [N] [l_max].
//  =============================================================================

#include <iostream>
#include <vector>
#include <array>
#include <functional>
#include <fstream>
#include <cmath>
#include <chrono>
#include <random>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <omp.h>
#include <mutex>


// ============================================================================
//  Optional features — set to true to enable, then recompile.
// ============================================================================
constexpr bool ENABLE_BONETTI_CONDITION_3 = false;
constexpr bool ENABLE_HARMONIC_OUTPUT     = false;
// ============================================================================


// ============================================================================
//  Global physics parameters
// ============================================================================
// r_sphere : "infinity" cutoff radius. Particles are launched and considered
//            escaped at this radius.
// eps_soft : Plummer softening length on the binary potential (regularises the
//            point-mass singularity for very close encounters).
double r_sphere = 50.0;
double eps_soft = 1e-5;


// ============================================================================
//  Kepler solver
// ============================================================================
// Solve Kepler's equation  M = E − e·sin E  for the eccentric anomaly E given
// the mean anomaly M_anom. Newton–Raphson with a sensible initial guess.
double EccentricAnomaly(double M_anom, double e, double tol = 1e-15, int max_iter = 100)
{
    M_anom = fmod(M_anom, 2.0 * M_PI);
    if (M_anom < 0) M_anom += 2.0 * M_PI;

    double E = (e < 0.8) ? M_anom : M_PI;

    for (int i = 0; i < max_iter; ++i)
    {
        double f = E - e * sin(E) - M_anom;
        double f_prime = 1 - e * cos(E);
        double delta = f / f_prime;
        E -= delta;
        if (fabs(delta) < tol) break;
    }
    return E;
}


// ============================================================================
//  3D vector type and operators
// ============================================================================
using Vec3 = std::array<double, 3>;
Vec3 operator+(const Vec3& a, const Vec3& b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
Vec3 operator*(double c, const Vec3& a)      { return {c*a[0], c*a[1], c*a[2]}; }
Vec3 operator*(const Vec3& a, double c)      { return c * a; }
Vec3 operator/(const Vec3& a, double c)      { return {a[0]/c, a[1]/c, a[2]/c}; }
double norm(const Vec3& a)                   { return std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]); }
double operator*(const Vec3& a, const Vec3& b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
inline Vec3 cross(const Vec3 &a, const Vec3 &b)
{
    return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
}
std::ostream& operator<<(std::ostream& os, const Vec3& a)
{
    os << "(" << a[0] << ", " << a[1] << ", " << a[2] << ")";
    return os;
}


// ============================================================================
//  Single-step integrators
//  ----------------------------------------------------------------------------
//  The hot path uses rk45Step() (Dormand–Prince) inside an externally-driven
//  adaptive loop — see evolveParticle(). The other routines are kept for
//  reference and easy A/B testing of the integrator (uncomment one inside
//  evolveParticle() to switch).
// ============================================================================

// Standard velocity-Verlet (symplectic, 2nd-order). Test-particle mass = 1.
void velocityVerletStep(Vec3 & r, Vec3 & v, double & t, double dt,
                        std::function<Vec3(const Vec3&, double)> force)
{
    Vec3 a = force(r, t);
    Vec3 r_new = r + v*dt + (0.5*dt*dt)*a;
    Vec3 a_new = force(r_new, t+dt);
    Vec3 v_new = v + 0.5*dt*(a + a_new);

    t += dt;
    r = r_new;
    v = v_new;
}

// Yoshida 4th-order composition of Verlet (symplectic, 4th-order).
void yoshida4Step(Vec3 &r, Vec3 &v, double &t, double dt,
                  std::function<Vec3(const Vec3&, double)> force)
{
    const double w1 =  1.3512071919596578;
    const double w0 = -1.7024143839193153;

    velocityVerletStep(r, v, t, w1 * dt, force);
    velocityVerletStep(r, v, t, w0 * dt, force);
    velocityVerletStep(r, v, t, w1 * dt, force);
}

// Dormand–Prince RK45 single step. Computes a 5(4) embedded pair; the embedded
// error is computed but not used here (the outer loop in evolveParticle does
// its own velocity-based adaptive control).
void rk45Step(Vec3 &r, Vec3 &v, double &t, double dt,
              const std::function<Vec3(const Vec3&, double)> &force)
{
    auto a = [&](const Vec3 &r_, double t_) { return force(r_, t_); };

    static const double c2 = 1.0/5.0,        c3 = 3.0/10.0,     c4 = 4.0/5.0,
                        c5 = 8.0/9.0,        c6 = 1.0;
    static const double a21 = 1.0/5.0;
    static const double a31 = 3.0/40.0,      a32 = 9.0/40.0;
    static const double a41 = 44.0/45.0,     a42 = -56.0/15.0,  a43 = 32.0/9.0;
    static const double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0,
                        a53 = 64448.0/6561.0, a54 = -212.0/729.0;
    static const double a61 = 9017.0/3168.0,  a62 = -355.0/33.0,
                        a63 = 46732.0/5247.0, a64 = 49.0/176.0,  a65 = -5103.0/18656.0;
    static const double a71 = 35.0/384.0,    a72 = 0.0,          a73 = 500.0/1113.0,
                        a74 = 125.0/192.0,   a75 = -2187.0/6784.0, a76 = 11.0/84.0;

    Vec3 k1_v = a(r, t);
    Vec3 k1_r = v;

    Vec3 k2_v = a(r + (a21*dt)*k1_r, t + c2*dt);
    Vec3 k2_r = v + (a21*dt)*k1_v;

    Vec3 k3_v = a(r + (a31*dt)*k1_r + (a32*dt)*k2_r, t + c3*dt);
    Vec3 k3_r = v + (a31*dt)*k1_v + (a32*dt)*k2_v;

    Vec3 k4_v = a(r + (a41*dt)*k1_r + (a42*dt)*k2_r + (a43*dt)*k3_r, t + c4*dt);
    Vec3 k4_r = v + (a41*dt)*k1_v + (a42*dt)*k2_v + (a43*dt)*k3_v;

    Vec3 k5_v = a(r + (a51*dt)*k1_r + (a52*dt)*k2_r + (a53*dt)*k3_r + (a54*dt)*k4_r, t + c5*dt);
    Vec3 k5_r = v + (a51*dt)*k1_v + (a52*dt)*k2_v + (a53*dt)*k3_v + (a54*dt)*k4_v;

    Vec3 k6_v = a(r + (a61*dt)*k1_r + (a62*dt)*k2_r + (a63*dt)*k3_r + (a64*dt)*k4_r + (a65*dt)*k5_r, t + c6*dt);
    Vec3 k6_r = v + (a61*dt)*k1_v + (a62*dt)*k2_v + (a63*dt)*k3_v + (a64*dt)*k4_v + (a65*dt)*k5_v;

    Vec3 r_new = r + dt*(a71*k1_r + a72*k2_r + a73*k3_r + a74*k4_r + a75*k5_r + a76*k6_r);
    Vec3 v_new = v + dt*(a71*k1_v + a72*k2_v + a73*k3_v + a74*k4_v + a75*k5_v + a76*k6_v);

    // Dormand–Prince embedded 5th-order error coefficients (computed but unused).
    static const double e1 = 71.0/57600.0,    e3 = -71.0/16695.0,  e4 = 71.0/1920.0,
                        e5 = -17253.0/339200.0, e6 = 22.0/525.0,    e7 = -1.0/40.0;

    Vec3 k7_v = a(r_new, t + dt);
    Vec3 k7_r = v_new;

    Vec3 err_r = dt * (e1*k1_r + e3*k3_r + e4*k4_r + e5*k5_r + e6*k6_r + e7*k7_r);
    Vec3 err_v = dt * (e1*k1_v + e3*k3_v + e4*k4_v + e5*k5_v + e6*k6_v + e7*k7_v);
    (void)err_r; (void)err_v;

    r = r_new;
    v = v_new;
    t += dt;
}

// Pihajoki-style symmetrised symplectic step (auxiliary copy + mid-step
// averaging) wrapped in a Yoshida-4 composition. Symplectic, 4th-order.
void pihajokiStep(Vec3 &r, Vec3 &v, double &t, double dt,
                  const std::function<Vec3(const Vec3&, double)> &force)
{
    static const double w1 =  1.0 / (2.0 - std::cbrt(2.0));
    static const double w0 = -std::cbrt(2.0) / (2.0 - std::cbrt(2.0));

    auto drift = [&](Vec3 &rr, Vec3 &vv, double hh) { rr = rr + hh * vv; };
    auto kick  = [&](Vec3 &rr, Vec3 &vv, double tt, double hh) {
        Vec3 a = force(rr, tt);
        vv = vv + hh * a;
    };

    Vec3 rc = r;
    Vec3 vc = v;

    auto substep = [&](double h)
    {
        kick(r,  v,  t, 0.5*h);
        kick(rc, vc, t, 0.5*h);

        drift(r,  v,  h);
        drift(rc, vc, h);

        Vec3 r_mix = 0.5 * (r  + rc);
        Vec3 v_mix = 0.5 * (v  + vc);
        r = r_mix; v = v_mix; rc = r_mix; vc = v_mix;

        kick(r,  v,  t + h, 0.5*h);
        kick(rc, vc, t + h, 0.5*h);

        t += h;
    };

    substep(w1 * dt);
    substep(w0 * dt);
    substep(w1 * dt);
}

// Bulirsch–Stoer with modified-midpoint base + Richardson extrapolation.
// Uses an internal hard-coded tolerance of 1e-12 for column acceptance.
void bulirschStoerStep(Vec3 &r, Vec3 &v, double &t, double dt,
                       const std::function<Vec3(const Vec3&, double)> &force)
{
    const int    MAX_COL = 8;
    const double SAFE    = 0.9;
    const double EPS     = 1e-14;

    std::vector<Vec3> Rr(MAX_COL),      Rv(MAX_COL);
    std::vector<Vec3> Rr_prev(MAX_COL), Rv_prev(MAX_COL);

    auto accel = [&](const Vec3 &rr, double tt) { return force(rr, tt); };

    auto midpointIntegrate = [&](const Vec3 &r0, const Vec3 &v0,
                                 double t0, double h, int nSteps,
                                 Vec3 &r_out, Vec3 &v_out)
    {
        Vec3 r_mid = r0 + 0.5 * h * v0;
        Vec3 v_mid = v0 + h * accel(r_mid, t0 + 0.5*h);
        double t_mid = t0 + h;

        for (int i = 1; i < nSteps; ++i)
        {
            r_mid = r_mid + h * v_mid;
            v_mid = v_mid + h * accel(r_mid, t_mid);
            t_mid += h;
        }

        r_out = r_mid + 0.5 * h * v_mid;
        v_out = v_mid + 0.5 * h * accel(r_out, t_mid);
    };

    Vec3 r0 = r, v0 = v;
    double t0 = t;

    for (int k = 1; k <= MAX_COL; ++k)
    {
        int nSteps = 2 * k;
        double h = dt / nSteps;

        Vec3 rk, vk;
        midpointIntegrate(r0, v0, t0, h, nSteps, rk, vk);

        Rr[0] = rk;
        Rv[0] = vk;

        for (int j = 1; j < k; ++j)
        {
            double factor = std::pow(double(nSteps) / double(2*(k-j)), 2*j) - 1.0;
            if (factor < EPS) factor = EPS;
            Rr[j] = Rr_prev[j-1] + (Rr[j-1] - Rr_prev[j-1]) / factor;
            Rv[j] = Rv_prev[j-1] + (Rv[j-1] - Rv_prev[j-1]) / factor;
        }

        if (k > 1)
        {
            Vec3 err_r = Rr[k-1] - Rr_prev[k-2];
            Vec3 err_v = Rv[k-1] - Rv_prev[k-2];
            double err = norm(err_r) + norm(err_v);
            if (err < SAFE * 1e-12)
            {
                r = Rr[k-1];
                v = Rv[k-1];
                t = t0 + dt;
                return;
            }
        }

        for (int j = 0; j < k; ++j)
        {
            Rr_prev[j] = Rr[j];
            Rv_prev[j] = Rv[j];
        }
    }

    r = Rr_prev[MAX_COL-1];
    v = Rv_prev[MAX_COL-1];
    t = t0 + dt;
}


// ============================================================================
//  Binary orbit (precomputed lookup table)
// ============================================================================

struct BinaryOrbit
{
    std::vector<double> times;
    std::vector<Vec3>   pos1;
    std::vector<Vec3>   pos2;
};

// Precompute the two binary positions on a uniform mean-anomaly grid.
// Linear interpolation in interpolateBinaryPositions() then gives the position
// of the two stars at arbitrary t, shared (read-only) across all OpenMP threads.
BinaryOrbit precomputeBinaryOrbit(double q, double e, int nSteps)
{
    BinaryOrbit orbit;
    orbit.times.resize(nSteps);
    orbit.pos1.resize(nSteps);
    orbit.pos2.resize(nSteps);

    double m1 = 1.0 / (1.0 + q);
    double m2 = 1.0 - m1;

    for (int i = 0; i < nSteps; i++)
    {
        double t = i * 2.0 * M_PI / nSteps;
        double E = EccentricAnomaly(t, e);
        Vec3   r12 = { std::cos(E) - e, std::sqrt(1 - e*e) * std::sin(E), 0.0 };

        orbit.times[i] = t;
        orbit.pos1[i]  = -m2 * r12;
        orbit.pos2[i]  =  m1 * r12;
    }
    return orbit;
}

// Linear interpolation of (pos1, pos2) at time t (modulo the binary period).
void interpolateBinaryPositions(const BinaryOrbit & orbit, double t, Vec3 & r1, Vec3 & r2)
{
    double T = 2.0 * M_PI;
    t = fmod(t, T);
    if (t < 0) t += T;

    size_t nSteps = orbit.times.size();
    double dt = T / nSteps;
    int i = static_cast<int>(t / dt);
    int j = (i + 1) % nSteps;

    double alpha = (t - orbit.times[i]) / dt;
    r1 = (1 - alpha) * orbit.pos1[i] + alpha * orbit.pos1[j];
    r2 = (1 - alpha) * orbit.pos2[i] + alpha * orbit.pos2[j];
}

// Returns a closure that evaluates the gravitational force on the test particle
// at (r, t), given the precomputed binary trajectory. Plummer-softened with
// eps_soft.
std::function<Vec3(const Vec3 &, double)> makeBinaryForce(const BinaryOrbit & orbit, double q)
{
    double m1 = 1.0 / (1.0 + q);
    double m2 = 1.0 - m1;

    return [=, &orbit](const Vec3& r, double t)
    {
        Vec3 r1, r2;
        interpolateBinaryPositions(orbit, t, r1, r2);

        Vec3 dr1 = r - r1;
        Vec3 dr2 = r - r2;
        double eps2  = eps_soft * eps_soft;
        double d1_sq = dr1 * dr1 + eps2;
        double d2_sq = dr2 * dr2 + eps2;
        double d1_32 = d1_sq * std::sqrt(d1_sq);
        double d2_32 = d2_sq * std::sqrt(d2_sq);

        return Vec3{} + (-m1 / d1_32) * dr1 + (-m2 / d2_32) * dr2;
    };
}


// ============================================================================
//  Conserved-quantity helpers (point-mass approximation)
// ============================================================================

// Specific energy in a unit-mass point potential at the origin.
inline double energy_approx(const Vec3 & r, const Vec3 & v)
{
    return 0.5 * pow(norm(v), 2) - 1.0 / norm(r);
}

// Specific angular momentum.
inline Vec3 angular_momentum(const Vec3 & r, const Vec3 & v) { return cross(r, v); }

// Map (r, v) at finite radius to the asymptotic velocity vector that a hyperbolic
// orbit in a unit-mass point-mass potential (mu = 1) would have at infinity.
// Uses specific energy and angular momentum to recover both |v_inf| and direction.
// Returns NaN if the orbit isn't hyperbolic.
Vec3 asymptotic_velocity_approx(const Vec3 &r, const Vec3 &v)
{
    const double mu  = 1.0;
    const double eps = energy_approx(r, v);
    if (eps <= 0.0) return {NAN, NAN, NAN};

    const double v_inf  = std::sqrt(2.0 * eps);
    const Vec3   h      = angular_momentum(r, v);
    const double h_norm = norm(h);
    const double r_norm = norm(r);
    const double rv     = r * v;

    if (h_norm < 1e-14 || r_norm < 1e-14)
    {
        // Nearly radial hyperbolic orbit: asymptotic direction (anti)parallel to r.
        Vec3 r_hat = r / std::max(r_norm, 1e-14);
        return (rv < 0.0 ? 1.0 : -1.0) * v_inf * r_hat;
    }

    const Vec3   e_vec = (cross(v, h) / mu) - (r / r_norm);
    const double e     = norm(e_vec);
    if (e <= 1.0) return {NAN, NAN, NAN};

    const Vec3 e_hat = e_vec / e;
    const Vec3 h_hat = h / h_norm;
    const Vec3 q_hat = cross(h_hat, e_hat);

    const double p      = h_norm * h_norm / mu;
    const double f_inf  = std::acos(-1.0 / e);
    const double f_asym = (rv >= 0.0) ? f_inf : -f_inf; // outgoing vs incoming branch

    const double pref = std::sqrt(mu / p);
    return pref * (-std::sin(f_asym) * e_hat + (e + std::cos(f_asym)) * q_hat);
}


// ============================================================================
//  Real spherical harmonics
//  ----------------------------------------------------------------------------
//  Compiled in unconditionally — only invoked when ENABLE_HARMONIC_OUTPUT is on,
//  and zero overhead otherwise.
// ============================================================================

// Computes all real spherical harmonics Y^R_lm(cos_theta, phi) for l = 0..l_max,
// m = -l..l, stored in out[l*l + l + m].
//
// Convention: Condon–Shortley phase absorbed into P_l^m. Real SH defined as
//     m  > 0 :  sqrt(2) · K_lm · P_l^m · cos(m·phi)
//     m == 0 :              K_l0 · P_l^0
//     m  < 0 :  sqrt(2) · K_lm · P_l^m · sin(|m|·phi)
// These satisfy the addition theorem
//     P_l(cos γ) = (4π / (2l+1)) · Σ_m  Y_lm(n1) · Y_lm(n2).
void compute_real_Ylm(double cos_theta, double phi, int l_max, double* out)
{
    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    const int sz = l_max + 1;

    // cos(m·phi), sin(m·phi) via Chebyshev recurrence
    std::vector<double> cm(sz), sm(sz);
    cm[0] = 1.0;  sm[0] = 0.0;
    if (l_max >= 1) { cm[1] = std::cos(phi);  sm[1] = std::sin(phi); }
    for (int m = 2; m <= l_max; ++m) {
        cm[m] = 2.0 * cm[1] * cm[m-1] - cm[m-2];
        sm[m] = 2.0 * cm[1] * sm[m-1] - sm[m-2];
    }

    // Associated Legendre P_l^m(cos_theta) with Condon–Shortley phase
    std::vector<double> plm(sz * sz, 0.0);
    auto P = [&](int l, int m) -> double& { return plm[l * sz + m]; };

    P(0, 0) = 1.0;
    for (int m = 1; m <= l_max; ++m)
        P(m, m) = -(2*m - 1) * sin_theta * P(m-1, m-1);
    for (int m = 0; m < l_max; ++m)
        P(m+1, m) = (2*m + 1) * cos_theta * P(m, m);
    for (int m = 0; m <= l_max; ++m)
        for (int l = m + 2; l <= l_max; ++l)
            P(l, m) = ((2*l - 1) * cos_theta * P(l-1, m) - (l + m - 1) * P(l-2, m))
                      / static_cast<double>(l - m);

    // Factorials (safe up to l_max ≈ 20 ⇒ 40! ≈ 1e48, well within double range)
    std::vector<double> fact(2 * l_max + 2, 1.0);
    for (int i = 1; i < (int)fact.size(); ++i)
        fact[i] = fact[i-1] * i;

    // Assemble real spherical harmonics
    for (int l = 0; l <= l_max; ++l) {
        double K0 = std::sqrt((2.0*l + 1.0) / (4.0 * M_PI));
        out[l*l + l] = K0 * P(l, 0);
        for (int m = 1; m <= l; ++m) {
            double Km = std::sqrt((2.0*l + 1.0) / (4.0 * M_PI) * fact[l-m] / fact[l+m]);
            double val = Km * P(l, m);
            out[l*l + l + m] = std::sqrt(2.0) * val * cm[m];
            out[l*l + l - m] = std::sqrt(2.0) * val * sm[m];
        }
    }
}


// ============================================================================
//  Stopping condition
//  ----------------------------------------------------------------------------
//  Defaulting `max_time_inside` to +infinity makes the Bonetti criterion a no-op
//  unless the caller explicitly passes a finite value, so this single signature
//  serves both the base and the Bonetti build.
// ============================================================================
bool stoppingCondition(const Vec3 & r, const Vec3 & v,
                       double t, double Tmax,
                       int steps, int max_steps,
                       double time_inside     = 0.0,
                       double max_time_inside = std::numeric_limits<double>::infinity(),
                       bool *escaped          = nullptr)
{
    // Particle has escaped: outside r_sphere with positive energy and moving outward.
    if (norm(r) > r_sphere && energy_approx(r, v) > 0 && r * v > 0)
    {
        if (escaped) *escaped = true;
        return true;
    }
    // Wall-clock / step / Bonetti exhaustion.
    if (t > Tmax || steps > max_steps || time_inside > max_time_inside)
    {
        if (escaped) *escaped = false;
        return true;
    }
    return false;
}


// ============================================================================
//  Initial conditions
// ============================================================================

Vec3 randomUnitVector(std::mt19937 & gen)
{
    std::uniform_real_distribution<> dist(0.0, 1.0);
    double u     = dist(gen);
    double w     = dist(gen);
    double theta = std::acos(2.0 * u - 1.0);
    double phi   = 2.0 * M_PI * w;
    return { std::sin(theta) * std::cos(phi),
             std::sin(theta) * std::sin(phi),
             std::cos(theta) };
}

struct ParticleResult
{
    Vec3   r;
    Vec3   v;
    double t;
    double delta_varpi = 0.0;
};

// Sample a fresh scattering: random incoming direction, random impact parameter
// b ∈ [0, b_max] (area-weighted so the sample is uniform in disk area), random
// initial binary phase. The particle is launched from r_init = r_sphere along
// −n_in with the speed required by energy conservation.
ParticleResult generateInitialConditions(double v_inf, double r_p = 5.0)
{
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> uniform01(0.0, 1.0);

    // b < b_max  ⇔  pericenter < r_p
    double b_max = r_p * std::sqrt(1.0 + 2.0 / (v_inf * v_inf * r_p));
    double b     = std::sqrt(uniform01(gen)) * b_max;

    Vec3 n_in  = randomUnitVector(gen);                 // incoming direction
    Vec3 n_tmp = randomUnitVector(gen);
    if (std::fabs(n_in * n_tmp) > 0.99) n_tmp = randomUnitVector(gen);
    Vec3 ex = n_tmp - (n_in * n_tmp) * n_in;            // orthogonal to n_in
    ex = ex / norm(ex);

    double r_init = r_sphere;
    Vec3   r0     = -r_init * n_in;                      // launch on the −n_in side

    double v_local = std::sqrt(v_inf * v_inf + 2.0 / r_init);   // energy conservation
    double v_tan   = (b * v_inf) / r_init;                       // angular-momentum conservation

    if (v_tan > v_local) {
        v_tan = v_local;
        std::cout << "ERROR: r_init is not large enough for this value of v" << std::endl;
    }
    double v_rad = std::sqrt(std::max(0.0, v_local * v_local - v_tan * v_tan));

    // Radial component points inward (toward decreasing r), which is +n_in here.
    Vec3 v0 = v_rad * n_in + v_tan * ex;
    double t0 = 2.0 * M_PI * uniform01(gen);            // random binary phase

    return {r0, v0, t0};
}


// ============================================================================
//  Far-field analytic Kepler shortcut (currently unused)
//  ----------------------------------------------------------------------------
//  When the particle is bound and far outside r_sphere, in principle one can
//  collapse the binary to a point mass and propagate analytically until
//  re-entry. This routine implements that. In practice it gives only marginal
//  speed-up for typical (q, e), so the call site in evolveParticle() is left
//  commented out. Kept here for reference and easy re-activation.
// ============================================================================
bool handleFarBoundKeplerMotion(Vec3 & r_old, Vec3 & v_old, double & t_old)
{
    if (!(norm(r_old) > r_sphere && energy_approx(r_old, v_old) < 0)) return false;

    const double mu = 1.0;
    double r_norm = norm(r_old);

    Vec3 h = angular_momentum(r_old, v_old);
    double h_norm = norm(h);
    if (h_norm < 1e-15) return false; // nearly radial → fall back to numerical integration

    double E_spec = energy_approx(r_old, v_old);
    double a = -mu / (2.0 * E_spec);
    if (a <= 0) return false;

    double e2 = 1.0 + 2.0 * E_spec * h_norm * h_norm / (mu * mu);
    if (e2 < 0) e2 = 0.0;
    double e = std::sqrt(e2);
    if (e == 0.0) return false; // circular: no radius motion to speed up

    double cosE0 = std::clamp((1.0 - r_norm / a) / e, -1.0, 1.0);
    double E0    = std::acos(cosE0);
    double vr    = (r_old * v_old) / r_norm;
    if (vr < 0) E0 = 2.0 * M_PI - E0;
    double M0 = E0 - e * std::sin(E0);

    double n_mean = std::sqrt(mu / (a * a * a));

    double r_exit     = r_sphere - 0.01;
    double cosE_bound = (1.0 - r_exit / a) / e;
    if (cosE_bound < -1.0 || cosE_bound > 1.0) return false;

    double E_bound_1 = std::acos(std::clamp(cosE_bound, -1.0, 1.0));
    double E_bound_2 = 2.0 * M_PI - E_bound_1;
    double M_bound_1 = E_bound_1 - e * std::sin(E_bound_1);
    double M_bound_2 = E_bound_2 - e * std::sin(E_bound_2);

    auto deltaMpos = [&](double M_target) {
        double dM = M_target - M0;
        while (dM <= 1e-12) dM += 2.0 * M_PI;
        return dM;
    };

    double dM1     = deltaMpos(M_bound_1);
    double dM2     = deltaMpos(M_bound_2);
    double dM      = (dM1 < dM2) ? dM1 : dM2;
    double delta_t = dM / n_mean;

    double M_target = M0 + dM;
    double E_target = EccentricAnomaly(M_target, e);
    double cosE_t   = std::cos(E_target);
    double sinE_t   = std::sin(E_target);

    Vec3   e_vec = (cross(v_old, h) / mu) - (r_old / r_norm);
    double e_mag = norm(e_vec);
    Vec3   e_hat = (e_mag < 1e-15) ? r_old / r_norm : e_vec / e_mag;
    Vec3   k_hat = h / h_norm;
    Vec3   q_hat = cross(k_hat, e_hat);

    Vec3 r_new = (a * (cosE_t - e)) * e_hat
               + (a * std::sqrt(std::max(0.0, 1 - e*e)) * sinE_t) * q_hat;
    double r_new_norm = norm(r_new);

    double coef  = std::sqrt(mu * a) / r_new_norm;
    Vec3   v_new = coef * (-sinE_t * e_hat
                           + std::sqrt(std::max(0.0, 1 - e*e)) * cosE_t * q_hat);

    t_old += delta_t;
    r_old  = r_new;
    v_old  = v_new;
    return true;
}


// ============================================================================
//  Particle evolution
//  ----------------------------------------------------------------------------
//  RK45 inside an externally-driven adaptive-step loop. The error proxy is the
//  relative change in |v| over the step, which is cheap and works well here
//  because the binary force varies smoothly except in close encounters.
//
//  The default tol = 0.01 was tuned with Pihajoki; use tol ≈ 0.02 for RK45
//  alone, ≈ 0.005 for plain velocity-Verlet.
//
//  Apsidal precession Δϖ is integrated alongside the trajectory using Gauss'
//  perturbation equation in the (ϖ, e) Lagrange-planetary form (zero inclination
//  assumed for the binary, which is correct here by construction).
// ============================================================================
ParticleResult evolveParticle(const BinaryOrbit& orbit,
                              const ParticleResult init,
                              [[maybe_unused]] double q, double e_bin,
                              double Tmax, int max_steps,
                              const std::function<Vec3(const Vec3&, double)> & forceFunc,
                              double tol = 0.01)
{
    Vec3   r_old = init.r, v_old = init.v;
    Vec3   r     = init.r, v     = init.v;
    double t     = init.t, t_old = init.t;
    double dt    = std::min(0.01, 0.01 / norm(v));

    const double safety = 0.9;
    const double p      = 1.0;

    // Tighten tolerance at high v_inf: small absolute errors on a brief
    // trajectory produce large fractional energy errors; this empirical
    // 1/v^4 scaling was found to keep ⟨ΔE⟩ stable across the velocity grid.
    tol = std::min(tol, tol / pow(std::sqrt(norm(v)), 4));

    int    steps        = 0;
    double time_inside  = 0.0;     // accumulated time spent inside r_sphere

    // Trapezoidal accumulator for Δϖ.
    double varpi_dot_prev  = 0.0;
    double varpi_integral  = 0.0;
    double t_first_step    = 0.0;
    double t_last_step     = 0.0;
    bool   first_accepted  = true;
    double gauss_prefactor = -std::sqrt((1.0 - e_bin * e_bin) / (e_bin * e_bin));

    // Bonetti option: cap interior dwell time. Disabled by default (+inf).
    constexpr double max_time_inside =
        ENABLE_BONETTI_CONDITION_3 ? 1e5
                                   : std::numeric_limits<double>::infinity();

    bool escaped = false;
    while (!stoppingCondition(r, v, t, Tmax, steps, max_steps,
                              time_inside, max_time_inside, &escaped))
    {
        // Optional: analytic Kepler shortcut when far and bound. Only meaningful
        // for very small q; left commented to keep behaviour identical to the
        // reference build. Uncomment to enable.
        // if (handleFarBoundKeplerMotion(r_old, v_old, t_old)) continue;

        r = r_old;
        v = v_old;
        t = t_old;

        // RK45 is the production choice: with adaptive dt, Verlet/Yoshida lose
        // their symplectic property and accumulate energy drift.
        rk45Step(r, v, t, dt, forceFunc);
        // velocityVerletStep(r, v, t, dt, forceFunc);
        // yoshida4Step       (r, v, t, dt, forceFunc);
        // pihajokiStep       (r, v, t, dt, forceFunc);
        // bulirschStoerStep  (r, v, t, dt, forceFunc);

        Vec3   dv        = v - v_old;
        double relChange = norm(dv) / (norm(v_old) + 1e-12);

        double factor = safety * std::pow(tol / (relChange + 1e-12), 1.0 / (p + 1.0));
        factor = std::clamp(factor, 0.5, 2.0);
        double new_dt = dt * factor;

        if (relChange < tol)            // step accepted
        {
            steps++;

            if constexpr (ENABLE_BONETTI_CONDITION_3) {
                if (norm(r) <= r_sphere) time_inside += t - t_old;
            }

            r_old = r;
            v_old = v;
            t_old = t;

            // Gauss' equation for apsidal precession rate (zero inclination).
            Vec3 r1_bin, r2_bin;
            interpolateBinaryPositions(orbit, t, r1_bin, r2_bin);
            Vec3   r12      = r2_bin - r1_bin;
            double r12_norm = norm(r12);

            Vec3   dr_to_2 = r - r2_bin;
            Vec3   dr_to_1 = r - r1_bin;
            double eps2    = eps_soft * eps_soft;
            double d2s     = dr_to_2 * dr_to_2 + eps2;
            double d1s     = dr_to_1 * dr_to_1 + eps2;
            double d2_32   = d2s * std::sqrt(d2s);
            double d1_32   = d1s * std::sqrt(d1s);
            Vec3   a_rel   = dr_to_2 / d2_32 - dr_to_1 / d1_32;

            double cos_phi = r12[0] / r12_norm;
            double sin_phi = r12[1] / r12_norm;
            Vec3   r_hat   = r12 / r12_norm;
            Vec3   phi_hat = {-r_hat[1], r_hat[0], 0.0};

            double a_r_val   = a_rel * r_hat;
            double a_phi_val = a_rel * phi_hat;
            double e_cos_phi = e_bin * cos_phi;

            double varpi_dot_curr = gauss_prefactor *
                (cos_phi * a_r_val
                 - (2.0 + e_cos_phi) / (1.0 + e_cos_phi) * sin_phi * a_phi_val);

            if (first_accepted) {
                t_first_step    = t;
                first_accepted  = false;
            } else {
                varpi_integral += 0.5 * (varpi_dot_prev + varpi_dot_curr)
                                       * (t - t_last_step);
            }
            varpi_dot_prev = varpi_dot_curr;
            t_last_step    = t;

            dt = std::min(new_dt, dt * 1.5);   // limit aggressive growth
        }
        else                            // step rejected: shrink dt and retry
            dt = new_dt;
    }
    (void)t_first_step;

    if (escaped)
        return {r, v, t, varpi_integral};
    else
        return { {NAN, NAN, NAN}, {NAN, NAN, NAN}, t, NAN };
}


// ============================================================================
//  Statistics helpers
// ============================================================================

// Sample mean and population standard deviation.
std::pair<double, double> mean_std(const std::vector<double>& data)
{
    size_t n = data.size();
    if (n == 0) return {NAN, NAN};
    if (n == 1) return {data[0], 0.0};

    double mean  = std::accumulate(data.begin(), data.end(), 0.0) / n;
    double accum = 0.0;
    for (double x : data) accum += (x - mean) * (x - mean);
    double var = accum / n;
    if (var < 0.0) var = 0.0;
    return {mean, std::sqrt(var)};
}

// Histograms (linear bin counts only used by printHist below).
std::vector<int> makeHistogram(const std::vector<double>& data,
                               double minVal, double maxVal, int nBins)
{
    std::vector<int> hist(nBins, 0);
    double binWidth = (maxVal - minVal) / nBins;
    for (double x : data) {
        if (x < minVal || x >= maxVal) continue;
        int bin = static_cast<int>((x - minVal) / binWidth);
        if (bin >= 0 && bin < nBins) hist[bin]++;
    }
    return hist;
}

// Diagnostic: pretty-print a labelled histogram to std::cout. Symmetric option
// chooses bins centred on zero; logBins switches to log-spaced bins.
void printHist(const std::string& name, const std::vector<double>& data,
               int nBins, bool symmetric = true, bool logBins = false)
{
    double minVal, maxVal;
    std::vector<int>    hist(nBins, 0);
    std::vector<double> binEdges(nBins + 1, 0.0);

    if (logBins) {
        minVal = std::max(1e-12, *std::min_element(data.begin(), data.end()));
        maxVal = *std::max_element(data.begin(), data.end());
        double logMin = std::log10(minVal);
        double logMax = std::log10(maxVal);
        double binWidth = (logMax - logMin) / nBins;
        for (int i = 0; i <= nBins; ++i) binEdges[i] = std::pow(10.0, logMin + i * binWidth);
        for (double x : data) {
            if (x < minVal || x > maxVal) continue;
            int bin = static_cast<int>((std::log10(x) - logMin) / binWidth);
            if (bin < 0)       bin = 0;
            if (bin >= nBins)  bin = nBins - 1;
            hist[bin]++;
        }
        std::cout << "# Histogram " << name << "\n";
        for (int i = 0; i < nBins; ++i) {
            double center = std::sqrt(binEdges[i] * binEdges[i+1]);
            std::cout << center << " " << hist[i] << "\n";
        }
    } else {
        if (symmetric) {
            double maxAbs = 0;
            for (double x : data) maxAbs = std::max(maxAbs, std::abs(x));
            minVal = -maxAbs; maxVal = maxAbs;
        } else {
            minVal = 0.0;
            maxVal = *std::max_element(data.begin(), data.end());
        }
        hist = makeHistogram(data, minVal, maxVal, nBins);
        std::cout << "# Histogram " << name << "\n";
        double binWidth = (maxVal - minVal) / nBins;
        for (int i = 0; i < nBins; ++i) {
            double center = minVal + (i + 0.5) * binWidth;
            std::cout << center << " " << hist[i] << "\n";
        }
    }
}


// ============================================================================
//  Velocity grid
// ============================================================================
// N_v log-spaced velocities between v_min(q) and v_max(q). The bounds are
// chosen so the grid extends well beyond both the diffusive regime (small v)
// and the high-speed ballistic regime (large v).
std::vector<double> v_list(double q, int N_v = 2000)
{
    double v_min = (1.0 / 10.0) * (std::sqrt(q) / (2.0 * (1.0 + q))) * std::sqrt(0.001);
    double v_max = 4.0 * (2.0 * std::sqrt(q) / (1.0 + q)) * std::sqrt(100.0);

    std::vector<double> out(N_v);
    if (N_v == 1)
        out[0] = v_min;
    else
        for (int i = 0; i < N_v; ++i)
            out[i] = std::exp(std::log(v_min)
                              + i * (std::log(v_max) - std::log(v_min)) / (N_v - 1));
    return out;
}


// ============================================================================
//  main()
// ============================================================================
int main(int argc, char* argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;

    // ----- Argument parsing -------------------------------------------------
    int  N_v_arg   = 2000;
    int  N         = 10000;
    int  l_max     = 10;
    int  n_sh      = (l_max + 1) * (l_max + 1);
    bool write_per_particle = false;
    bool write_harmonics    = true;

    if constexpr (ENABLE_HARMONIC_OUTPUT)
    {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " q e [N_v] [N] [l_max]\n"
                      << "  l_max > 0 : write SH harmonics file only (default: 10)\n"
                      << "  l_max = 0 : write per-particle file only\n"
                      << "  l_max < 0 : write both files (use |l_max|); for consistency checks\n";
            return 1;
        }
    }
    else
    {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " q e\n";
            return 1;
        }
    }

    double q = std::stod(argv[1]);
    double e = std::stod(argv[2]);

    if constexpr (ENABLE_HARMONIC_OUTPUT)
    {
        N_v_arg            = (argc >= 4) ? std::stoi(argv[3]) : 2000;
        N                  = (argc >= 5) ? std::stoi(argv[4]) : 10000;
        int  l_max_arg     = (argc >= 6) ? std::stoi(argv[5]) : 10;
        write_per_particle = (l_max_arg <= 0);
        write_harmonics    = (l_max_arg != 0);
        l_max              = std::abs(l_max_arg);
        n_sh               = (l_max + 1) * (l_max + 1);
    }

    // ----- Tmax -------------------------------------------------------------
    double Tmax;
    if constexpr (ENABLE_BONETTI_CONDITION_3)
        Tmax = 8.0 * pow(1.0 + q, 3) / pow(q, 1.5) * 1e10 / 49076.0;
    else
        Tmax = 1e11;

    int    max_steps    = static_cast<int>(1e9);
    double rp_max       = 5.0;
    bool   split_by_Tcut = false;     // toggle to enable multi-Tcut output

    // Optional log-spaced cutoff times for splitting statistics.
    std::vector<double> TmaxCuts;
    if (split_by_Tcut) {
        const int    n_cuts  = 100;
        const double Tmin_cut = 1e3;
        double log_min = std::log10(Tmin_cut);
        double log_max = std::log10(Tmax);
        for (int i = 0; i < n_cuts; ++i) {
            double logv = log_min + (log_max - log_min) * i / (n_cuts - 1);
            TmaxCuts.push_back(std::pow(10.0, logv));
        }
        TmaxCuts.push_back(Tmax);
    } else {
        TmaxCuts = {Tmax};
    }

    std::vector<double> v_values = v_list(q, N_v_arg);

    if constexpr (ENABLE_HARMONIC_OUTPUT)
    {
        std::cout << "Parameters: q=" << q << ", e=" << e
                  << ", N_v=" << v_values.size()
                  << ", N="   << N
                  << ", l_max=" << l_max
                  << (write_harmonics    ? " [harmonics]"    : "")
                  << (write_per_particle ? " [per-particle]" : "")
                  << std::endl;
    }

    // Precompute the binary orbit (shared, read-only across threads).
    BinaryOrbit orbit     = precomputeBinaryOrbit(q, e, 100000);
    auto        forceFunc = makeBinaryForce(orbit, q);

    // Per-velocity result buckets: (init, final) pair for every escaped particle.
    std::vector<std::vector<std::pair<ParticleResult, ParticleResult>>>
        all_results(v_values.size());

    // ----- Parallel Monte-Carlo loop ---------------------------------------
    #pragma omp parallel for schedule(dynamic)
    for (int iv = 0; iv < (int)v_values.size(); ++iv)
    {
        double v          = v_values[iv];
        auto   local_start = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<ParticleResult, ParticleResult>> results;
        results.reserve(N);

        for (int i = 0; i < N; ++i)
        {
            ParticleResult init = generateInitialConditions(v, rp_max);
            ParticleResult fin  = evolveParticle(orbit, init, q, e, Tmax, max_steps, forceFunc);
            if (!std::isnan(fin.r[0]))
                results.push_back({init, fin});
        }

        all_results[iv] = std::move(results);

        auto   local_end     = std::chrono::high_resolution_clock::now();
        double local_elapsed = std::chrono::duration<double>(local_end - local_start).count();
        size_t N_resolved    = all_results[iv].size();

        #pragma omp critical
        std::cout << "Elapsed time for v=" << v << ": " << local_elapsed
                  << " seconds,\t resolved " << N_resolved << " particles\n";
    }

    // ----- Optional: per-particle binary output ----------------------------
    if constexpr (ENABLE_HARMONIC_OUTPUT)
    {
        if (write_per_particle)
        {
            std::ostringstream fname;
            fname << "particles_q=" << q << "_e=" << e << ".bin";
            std::ofstream bin_out(fname.str(), std::ios::binary);
            if (!bin_out.is_open()) {
                std::cerr << "Error: could not open " << fname.str() << " for writing\n";
            } else {
                bin_out.write(reinterpret_cast<const char*>(&q),     sizeof(double));
                bin_out.write(reinterpret_cast<const char*>(&e),     sizeof(double));
                bin_out.write(reinterpret_cast<const char*>(&rp_max), sizeof(double));
                double rs = r_sphere;
                bin_out.write(reinterpret_cast<const char*>(&rs),    sizeof(double));
                int n_v = static_cast<int>(v_values.size());
                bin_out.write(reinterpret_cast<const char*>(&n_v),   sizeof(int));
                bin_out.write(reinterpret_cast<const char*>(&N),     sizeof(int));

                for (int iv = 0; iv < (int)v_values.size(); ++iv) {
                    double v = v_values[iv];
                    bin_out.write(reinterpret_cast<const char*>(&v), sizeof(double));

                    int n_esc = static_cast<int>(all_results[iv].size());
                    bin_out.write(reinterpret_cast<const char*>(&n_esc), sizeof(int));

                    for (const auto &pair : all_results[iv]) {
                        const auto &init = pair.first;
                        const auto &fin  = pair.second;

                        Vec3   v_in = asymptotic_velocity_approx(init.r, init.v);
                        double dE   = energy_approx(fin.r, fin.v) - energy_approx(init.r, init.v);
                        Vec3   dv   = asymptotic_velocity_approx(fin.r, fin.v) - v_in;
                        Vec3   dL   = angular_momentum(fin.r, fin.v) - angular_momentum(init.r, init.v);
                        double dT   = fin.t - init.t;

                        double record[12] = {
                            v_in[0], v_in[1], v_in[2],
                            dE, dv[0], dv[1], dv[2],
                            dL[0], dL[1], dL[2], dT,
                            fin.delta_varpi
                        };
                        bin_out.write(reinterpret_cast<const char*>(record), sizeof(record));
                    }
                }
                std::cout << "Per-particle file written: " << fname.str() << std::endl;
            }
        }

        // ----- Optional: spherical-harmonic moments output -----------------
        if (write_harmonics)
        {
            std::ostringstream fname;
            fname << "harmonics_q=" << q << "_e=" << e << ".bin";
            std::ofstream hout(fname.str(), std::ios::binary);
            if (!hout.is_open()) {
                std::cerr << "Error: could not open " << fname.str() << " for writing\n";
            } else {
                // Header: q, e, rp_max, r_sphere (doubles), N_v, N_per_v, l_max (ints)
                hout.write(reinterpret_cast<const char*>(&q),     sizeof(double));
                hout.write(reinterpret_cast<const char*>(&e),     sizeof(double));
                hout.write(reinterpret_cast<const char*>(&rp_max), sizeof(double));
                double rs = r_sphere;
                hout.write(reinterpret_cast<const char*>(&rs),    sizeof(double));
                int n_v = static_cast<int>(v_values.size());
                hout.write(reinterpret_cast<const char*>(&n_v),   sizeof(int));
                hout.write(reinterpret_cast<const char*>(&N),     sizeof(int));
                hout.write(reinterpret_cast<const char*>(&l_max), sizeof(int));

                std::vector<double> ylm_buf(n_sh);
                constexpr int N_Q = 8;          // dE, dv{x,y,z}, dL{x,y,z}, delta_varpi

                for (int iv = 0; iv < (int)v_values.size(); ++iv) {
                    double v = v_values[iv];
                    hout.write(reinterpret_cast<const char*>(&v), sizeof(double));
                    int n_esc = static_cast<int>(all_results[iv].size());
                    hout.write(reinterpret_cast<const char*>(&n_esc), sizeof(int));

                    std::vector<double> M(N_Q * n_sh, 0.0);   // first  moments
                    std::vector<double> S(N_Q * n_sh, 0.0);   // second moments

                    for (const auto &pair : all_results[iv]) {
                        const auto &init = pair.first;
                        const auto &fin  = pair.second;

                        Vec3   v_in = asymptotic_velocity_approx(init.r, init.v);
                        double dE   = energy_approx(fin.r, fin.v) - energy_approx(init.r, init.v);
                        Vec3   dv   = asymptotic_velocity_approx(fin.r, fin.v) - v_in;
                        Vec3   dL   = angular_momentum(fin.r, fin.v) - angular_momentum(init.r, init.v);

                        double X[N_Q] = {dE, dv[0], dv[1], dv[2],
                                         dL[0], dL[1], dL[2], fin.delta_varpi};

                        // Direction of incoming asymptotic velocity.
                        double v_mag  = norm(v_in);
                        double cos_th = (v_mag > 1e-30) ? v_in[2] / v_mag : 1.0;
                        double phi_az = std::atan2(v_in[1], v_in[0]);

                        compute_real_Ylm(cos_th, phi_az, l_max, ylm_buf.data());

                        for (int iq = 0; iq < N_Q; ++iq)
                            for (int lm = 0; lm < n_sh; ++lm) {
                                M[iq * n_sh + lm] += ylm_buf[lm] * X[iq];
                                S[iq * n_sh + lm] += ylm_buf[lm] * X[iq] * X[iq];
                            }
                    }

                    if (n_esc > 0) {
                        double inv_n = 1.0 / n_esc;
                        for (auto &val : M) val *= inv_n;
                        for (auto &val : S) val *= inv_n;
                    }

                    hout.write(reinterpret_cast<const char*>(M.data()),
                               M.size() * sizeof(double));
                    hout.write(reinterpret_cast<const char*>(S.data()),
                               S.size() * sizeof(double));
                }

                double file_mb = (44.0 + v_values.size()
                                  * (12.0 + 2.0 * N_Q * n_sh * 8.0)) / (1024.0 * 1024.0);
                std::cout << "Harmonics file written: " << fname.str()
                          << " (l_max=" << l_max << ", ~"
                          << std::fixed << std::setprecision(1) << file_mb << " MB)"
                          << std::endl;
            }
        }
    }

    // ----- Always: text summary file(s) ------------------------------------
    for (double TmaxCut : TmaxCuts)
    {
        std::ostringstream fcut;
        fcut << "q=" << q << "_e=" << e << "_Tcut=";
        long long Tcut_int = static_cast<long long>(TmaxCut);
        fcut << Tcut_int << ".txt";
        std::ofstream out(fcut.str());

        out << "# N = " << N << ", rp_max = " << rp_max
            << ", r_sphere = " << r_sphere << ", T_cut = " << TmaxCut << "\n";
        out << "# v\tmean∆E\tSEM_∆E\t∆T\tSEM_∆T"
            << "\t∆vx\tSEM_∆vx\t∆vy\tSEM_∆vy\t∆vz\tSEM_∆vz"
            << "\t∆Lx\tSEM_∆Lx\t∆Ly\tSEM_∆Ly\t∆Lz\tSEM_∆Lz"
            << "\tΔvarpi\tSEM_Δvarpi\tNresolved\n";

        for (int iv = 0; iv < (int)v_values.size(); ++iv) {
            double v = v_values[iv];

            std::vector<double> deltaE, deltaT, dvx, dvy, dvz, dLx, dLy, dLz, delta_varpi_vals;
            for (const auto &pair : all_results[iv]) {
                const auto &init = pair.first;
                const auto &fin  = pair.second;
                if (fin.t < TmaxCut) {
                    deltaE.push_back(energy_approx(fin.r, fin.v) - energy_approx(init.r, init.v));
                    deltaT.push_back(fin.t - init.t);
                    Vec3 dv = asymptotic_velocity_approx(fin.r, fin.v)
                            - asymptotic_velocity_approx(init.r, init.v);
                    Vec3 dL = angular_momentum(fin.r, fin.v) - angular_momentum(init.r, init.v);
                    dvx.push_back(dv[0]); dvy.push_back(dv[1]); dvz.push_back(dv[2]);
                    dLx.push_back(dL[0]); dLy.push_back(dL[1]); dLz.push_back(dL[2]);
                    delta_varpi_vals.push_back(fin.delta_varpi);
                }
            }

            size_t N_res = deltaE.size();
            if (N_res < 2) {
                out << v << "\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN"
                    << "\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\t" << N_res << "\n";
                continue;
            }

            auto msE  = mean_std(deltaE);
            auto msT  = mean_std(deltaT);
            auto msvx = mean_std(dvx);
            auto msvy = mean_std(dvy);
            auto msvz = mean_std(dvz);
            auto msLx = mean_std(dLx);
            auto msLy = mean_std(dLy);
            auto msLz = mean_std(dLz);
            auto msVd = mean_std(delta_varpi_vals);

            double sN = std::sqrt(static_cast<double>(N_res));
            out << v << "\t"
                << msE.first  << "\t" << msE.second  / sN << "\t"
                << msT.first  << "\t" << msT.second  / sN << "\t"
                << msvx.first << "\t" << msvx.second / sN << "\t"
                << msvy.first << "\t" << msvy.second / sN << "\t"
                << msvz.first << "\t" << msvz.second / sN << "\t"
                << msLx.first << "\t" << msLx.second / sN << "\t"
                << msLy.first << "\t" << msLy.second / sN << "\t"
                << msLz.first << "\t" << msLz.second / sN << "\t"
                << msVd.first << "\t" << msVd.second / sN << "\t"
                << N_res << "\n";
        }
    }

    auto   end_time       = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total elapsed time: " << elapsed_seconds << " seconds" << std::endl;

    return 0;
}
