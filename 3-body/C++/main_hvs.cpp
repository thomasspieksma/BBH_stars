//
//  main.cpp
//  3-body-scattering
//
//  Created by Giovanni Maria Tomaselli on 31/07/25.
//

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
#include <omp.h>
#include <mutex>

// Units: G = 1, M = 1 (total binary mass), a = 1 (semi-major axis). The orbital period is then 2π.

double r_sphere = 50.0;

// Calculate the eccentric anomaly E as a function of the mean anomaly M_anom
double EccentricAnomaly(double M_anom, double e, double tol = 1e-15, int max_iter = 100)
{
    M_anom = fmod(M_anom, 2.0 * M_PI); // Normalize M to [0, 2π]
    if (M_anom < 0) M_anom += 2.0 * M_PI;

    double E = (e < 0.8) ? M_anom : M_PI; // Initial guess (fast convergence)

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

// A 3D vector type for convenience, and its utility functions
using Vec3 = std::array<double, 3>;
Vec3 operator+(const Vec3& a, const Vec3& b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
Vec3 operator*(double c, const Vec3& a) { return {c*a[0], c*a[1], c*a[2]}; }
Vec3 operator*(const Vec3& a, double c) { return c * a; }
Vec3 operator/(const Vec3& a, double c) { return {a[0]/c, a[1]/c, a[2]/c}; }
double norm(const Vec3& a) { return std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]); }
double operator*(const Vec3& a, const Vec3& b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
inline Vec3 cross(const Vec3 &a, const Vec3 &b) { return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]}; }
std::ostream& operator<<(std::ostream& os, const Vec3& a) { os << "(" << a[0] << ", " << a[1] << ", " << a[2] << ")"; return os; }

struct EscRecord {
    double vf_inf;
    double cos_theta;
    double phi;
    double delta_t;
};

void write_bin(const std::string& fname,
               double q, double e, double rp_max, double r_sphere,
               const std::vector<double>& v_values,
               int N_per_v,
               const std::vector<std::vector<EscRecord>>& all_records)
{
    std::ofstream out(fname, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open output file: " + fname);
    }

    int n_v = (int)v_values.size();

    // Header (must match hvs_analysis_1.py read order)
    out.write(reinterpret_cast<const char*>(&q),        sizeof(double));
    out.write(reinterpret_cast<const char*>(&e),        sizeof(double));
    out.write(reinterpret_cast<const char*>(&rp_max),   sizeof(double));
    out.write(reinterpret_cast<const char*>(&r_sphere), sizeof(double));
    out.write(reinterpret_cast<const char*>(&n_v),      sizeof(int));
    out.write(reinterpret_cast<const char*>(&N_per_v),  sizeof(int));

    // Per v-bin block
    for (int iv = 0; iv < n_v; ++iv) {
        double v0 = v_values[iv];
        int n_esc = (int)all_records[iv].size();

        out.write(reinterpret_cast<const char*>(&v0),    sizeof(double));
        out.write(reinterpret_cast<const char*>(&n_esc), sizeof(int));

        for (const auto& r : all_records[iv]) {
            out.write(reinterpret_cast<const char*>(&r.vf_inf),    sizeof(double));
            out.write(reinterpret_cast<const char*>(&r.cos_theta), sizeof(double));
            out.write(reinterpret_cast<const char*>(&r.phi),       sizeof(double));
            out.write(reinterpret_cast<const char*>(&r.delta_t),   sizeof(double));
        }
    }
}


// Velocity Verlet integrator step, the mass of the test particle is set to 1
void velocityVerletStep(Vec3 & r, Vec3 & v, double & t, double dt, std::function<Vec3(const Vec3&, double)> force)
{
    Vec3 a = force(r, t);
    Vec3 r_new = r + v*dt + (0.5*dt*dt)*a;
    Vec3 a_new = force(r_new, t+dt);
    Vec3 v_new = v + 0.5*dt*(a + a_new);
    
    t += dt;
    r = r_new;
    v = v_new;
}

void yoshida4Step(Vec3 &r, Vec3 &v, double &t, double dt, std::function<Vec3(const Vec3&, double)> force)
{
    const double w1 = 1.3512071919596578;
    const double w0 = -1.7024143839193153;

    velocityVerletStep(r, v, t, w1 * dt, force);
    velocityVerletStep(r, v, t, w0 * dt, force);
    velocityVerletStep(r, v, t, w1 * dt, force);
}

void rk45Step(Vec3 &r, Vec3 &v, double &t, double dt, const std::function<Vec3(const Vec3&, double)> &force)
{
    // Define acceleration a(r, t)
    auto a = [&](const Vec3 &r_, double t_) {
        return force(r_, t_);
    };

    // Stage coefficients for Dormand–Prince RK45
    static const double c2 = 1.0/5.0,       c3 = 3.0/10.0,     c4 = 4.0/5.0,
                        c5 = 8.0/9.0,       c6 = 1.0,          c7 = 1.0;
    static const double a21 = 1.0/5.0;
    static const double a31 = 3.0/40.0,     a32 = 9.0/40.0;
    static const double a41 = 44.0/45.0,    a42 = -56.0/15.0,  a43 = 32.0/9.0;
    static const double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0,
                        a53 = 64448.0/6561.0, a54 = -212.0/729.0;
    static const double a61 = 9017.0/3168.0,  a62 = -355.0/33.0,
                        a63 = 46732.0/5247.0, a64 = 49.0/176.0,  a65 = -5103.0/18656.0;
    static const double a71 = 35.0/384.0,   a72 = 0.0,          a73 = 500.0/1113.0,
                        a74 = 125.0/192.0,  a75 = -2187.0/6784.0, a76 = 11.0/84.0;

    // Initialize accelerations (we integrate both position and velocity)
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

    // 5th-order "error estimate" (optional, could use to tune dt externally)
    // Commonly used embedded coefficients (Dormand–Prince pair)
    static const double e1 = 71.0/57600.0, e3 = -71.0/16695.0, e4 = 71.0/1920.0,
                        e5 = -17253.0/339200.0, e6 = 22.0/525.0, e7 = -1.0/40.0;

    Vec3 k7_v = a(r_new, t + dt);
    Vec3 k7_r = v_new;

    Vec3 err_r = dt * (e1*k1_r + e3*k3_r + e4*k4_r + e5*k5_r + e6*k6_r + e7*k7_r);
    Vec3 err_v = dt * (e1*k1_v + e3*k3_v + e4*k4_v + e5*k5_v + e6*k6_v + e7*k7_v);

    // Overwrite state
    r = r_new;
    v = v_new;
    t += dt;

    // (Optional) You can return or compute an error norm if you want
    // to adapt dt automatically based on |err|
}

void pihajokiStep(Vec3 &r, Vec3 &v, double &t, double dt, const std::function<Vec3(const Vec3&, double)> &force)
{
    // Yoshida 4th-order coefficients
    static const double w1 = 1.0 / (2.0 - std::cbrt(2.0));
    static const double w0 = - std::cbrt(2.0) / (2.0 - std::cbrt(2.0));

    auto drift = [&](Vec3 &rr, Vec3 &vv, double hh) {
        rr = rr + hh * vv;
    };

    auto kick = [&](Vec3 &rr, Vec3 &vv, double tt, double hh) {
        Vec3 a = force(rr, tt);
        vv = vv + hh * a;
    };

    // Duplicate variables (Pihajoki trick)
    Vec3 rc = r;
    Vec3 vc = v;

    auto substep = [&](double h)
    {
        // Kick both copies
        kick(r,  v,  t, 0.5*h);
        kick(rc, vc, t, 0.5*h);

        // Drift both copies
        drift(r,  v,  h);
        drift(rc, vc, h);

        // Mix the two copies (keeps symplectic structure)
        Vec3 r_mix  = 0.5 * (r  + rc);
        Vec3 v_mix  = 0.5 * (v  + vc);

        r  = r_mix;
        v  = v_mix;
        rc = r_mix;
        vc = v_mix;

        // Kick again
        kick(r,  v,  t + h, 0.5*h);
        kick(rc, vc, t + h, 0.5*h);

        // Advance physical time
        t += h;
    };

    // Yoshida 4th-order composition
    substep(w1 * dt);
    substep(w0 * dt);
    substep(w1 * dt);
}

void bulirschStoerStep(Vec3 &r, Vec3 &v, double &t, double dt, const std::function<Vec3(const Vec3&, double)> &force)
{
    const int MAX_COL = 8;      // max Richardson columns
    const int MAX_M = 2000;     // max midpoint substeps
    const double SAFE = 0.9;    // safety for Richardson
    const double EPS = 1e-14;   // avoid division by zero

    // Storage for Richardson extrapolation
    std::vector<Vec3> Rr(MAX_COL), Rv(MAX_COL);
    std::vector<Vec3> Rr_prev(MAX_COL), Rv_prev(MAX_COL);

    auto accel = [&](const Vec3 &rr, double tt) {
        return force(rr, tt);
    };

    auto midpointIntegrate = [&](const Vec3 &r0, const Vec3 &v0,
                                 double t0, double h, int nSteps,
                                 Vec3 &r_out, Vec3 &v_out)
    {
        // Modified midpoint initialization
        Vec3 r_mid = r0 + 0.5 * h * v0;
        Vec3 v_mid = v0 + h * accel(r_mid, t0 + 0.5*h);

        double t_mid = t0 + h;

        // Perform nSteps midpoint iterations
        for (int i = 1; i < nSteps; ++i)
        {
            r_mid = r_mid + h * v_mid;
            v_mid = v_mid + h * accel(r_mid, t_mid);
            t_mid += h;
        }

        // Final correction to get r_out, v_out
        r_out = r_mid + 0.5 * h * v_mid;
        v_out = v_mid + 0.5 * h * accel(r_out, t_mid);

        // The above corresponds to one full dt using midpoint
    };

    // Starting point for extrapolation
    Vec3 r0 = r, v0 = v;
    double t0 = t;

    // Extrapolation loop
    for (int k = 1; k <= MAX_COL; ++k)
    {
        int nSteps = 2*k;                // midpoint subdivisions
        double h = dt / nSteps;

        Vec3 rk, vk;
        midpointIntegrate(r0, v0, t0, h, nSteps, rk, vk);

        // Store first column
        Rr[0] = rk;
        Rv[0] = vk;

        // Richardson extrapolation
        for (int j = 1; j < k; ++j)
        {
            double factor = std::pow(double(nSteps) / double(2*(k-j)), 2*j) - 1.0;

            if (factor < EPS) factor = EPS;

            Rr[j] = Rr_prev[j-1] + (Rr[ j-1 ] - Rr_prev[j-1]) / factor;
            Rv[j] = Rv_prev[j-1] + (Rv[ j-1 ] - Rv_prev[j-1]) / factor;
        }

        // Error estimate for this column
        if (k > 1)
        {
            Vec3 err_r = Rr[k-1] - Rr_prev[k-2];
            Vec3 err_v = Rv[k-1] - Rv_prev[k-2];

            double err =
                norm(err_r) + norm(err_v);

            // If error small enough: accept step
            if (err < SAFE * 1e-12)  // replace 1e-12 by tolerance if you want internal control
            {
                r = Rr[k-1];
                v = Rv[k-1];
                t = t0 + dt;
                return;
            }
        }

        // Copy this column to previous for next iteration
        for (int j = 0; j < k; ++j)
        {
            Rr_prev[j] = Rr[j];
            Rv_prev[j] = Rv[j];
        }
    }

    // If maxed out columns, accept last extrapolation
    r = Rr_prev[MAX_COL-1];
    v = Rv_prev[MAX_COL-1];
    t = t0 + dt;
}

struct BinaryOrbit { std::vector<double> times; std::vector<Vec3> pos1; std::vector<Vec3> pos2;};

BinaryOrbit precomputeBinaryOrbit(double q, double e, int nSteps)
{
    BinaryOrbit orbit;
    orbit.times.resize(nSteps);
    orbit.pos1.resize(nSteps);
    orbit.pos2.resize(nSteps);

    double m1 = 1 / (1.0 + q);
    double m2 = 1 - m1;
    
    for (int i = 0; i < nSteps; i++)
    {
        double t = i * 2.0 * M_PI / nSteps;
        double E = EccentricAnomaly(t, e);
        Vec3 r12 = { std::cos(E) - e, std::sqrt(1 - e*e) * std::sin(E), 0.0 };

        orbit.times[i] = t;
        orbit.pos1[i] = -m2 * r12;
        orbit.pos2[i] = m1 * r12;
    }

    return orbit;
}

void interpolateBinaryPositions(const BinaryOrbit & orbit, double t, Vec3 & r1, Vec3 & r2)
{
    double T = 2.0 * M_PI;
    t = fmod(t, T);
    if (t < 0) t += T;

    // Find nearest indices
    size_t nSteps = orbit.times.size();
    double dt = T / nSteps; // can incorporate this into the parent function
    int i = static_cast<int>(t / dt);
    int j = (i+1) % nSteps;

    double alpha = (t - orbit.times[i]) / dt;
    r1 = (1-alpha)*orbit.pos1[i] + alpha*orbit.pos1[j];
    r2 = (1-alpha)*orbit.pos2[i] + alpha*orbit.pos2[j];
}

std::function<Vec3(const Vec3 &, double)> makeBinaryForce(const BinaryOrbit & orbit, double q)
{
    double m1 = 1 / (1.0 + q);
    double m2 = 1 - m1;

    return [=,&orbit](const Vec3& r, double t)
    {
        Vec3 r1, r2;
        interpolateBinaryPositions(orbit, t, r1, r2);

        Vec3 F = {0,0,0};

        Vec3 dr1 = r - r1;
        Vec3 dr2 = r - r2;
        double d1 = norm(dr1);
        double d2 = norm(dr2);

        if (d1 > 1e-15) F = F + (-m1/(d1*d1*d1)) * dr1;
        if (d2 > 1e-15) F = F + (-m2/(d2*d2*d2)) * dr2;

        return F;
    };
}

inline double energy_approx(const Vec3 & r, const Vec3 & v) { return 0.5 * pow(norm(v), 2) - 1 / norm(r); }

inline Vec3 angular_momentum(const Vec3 & r, const Vec3 & v) { return cross(r, v); }

// Map a local state (r, v) to the asymptotic velocity vector in a point-mass potential (mu=1).
// Uses specific energy and angular momentum to recover both |v_inf| and direction.
Vec3 asymptotic_velocity_approx(const Vec3 &r, const Vec3 &v)
{
    const double mu = 1.0;
    const double eps = energy_approx(r, v);
    if (eps <= 0.0) return {NAN, NAN, NAN}; // no real asymptotic state for bound/parabolic motion

    const double v_inf = std::sqrt(2.0 * eps);
    const Vec3 h = angular_momentum(r, v);
    const double h_norm = norm(h);
    const double r_norm = norm(r);
    const double rv = r * v;

    // Nearly radial hyperbolic orbit: asymptotic direction is (anti)parallel to radius.
    if (h_norm < 1e-14 || r_norm < 1e-14)
    {
        Vec3 r_hat = r / std::max(r_norm, 1e-14);
        return (rv < 0.0 ? 1.0 : -1.0) * v_inf * r_hat;
    }

    const Vec3 e_vec = (cross(v, h) / mu) - (r / r_norm);
    const double e = norm(e_vec);
    if (e <= 1.0) return {NAN, NAN, NAN}; // not hyperbolic

    const Vec3 e_hat = e_vec / e;
    const Vec3 h_hat = h / h_norm;
    const Vec3 q_hat = cross(h_hat, e_hat);

    const double p = h_norm * h_norm / mu;
    const double f_inf = std::acos(-1.0 / e);
    const double f_asym = (rv >= 0.0) ? f_inf : -f_inf; // outgoing vs incoming branch

    const double pref = std::sqrt(mu / p);
    return pref * (-std::sin(f_asym) * e_hat + (e + std::cos(f_asym)) * q_hat);
}

bool stoppingCondition(const Vec3 & r, const Vec3 & v, double t, double Tmax, int steps, int max_steps, bool *escaped = nullptr)
{
    if (norm(r) > r_sphere && energy_approx(r, v) > 0 && r * v > 0)
    {
        if (escaped) *escaped = true;
        return true;
    }
    if (t > Tmax || steps > max_steps)
    {
        if (escaped) *escaped = false;
        return true;
    }
    return false;
}

Vec3 randomUnitVector(std::mt19937 & gen)
{
    std::uniform_real_distribution<> dist(0.0, 1.0);
    double u = dist(gen);
    double v = dist(gen);
    double theta = std::acos(2.0*u - 1.0);
    double phi = 2.0 * M_PI * v;
    return { std::sin(theta)*std::cos(phi), std::sin(theta)*std::sin(phi), std::cos(theta) };
}

struct ParticleResult { Vec3 r; Vec3 v; double t; };

// Generates randomized initial conditions for one scattering (v_inf is speed at infinity)
ParticleResult generateInitialConditions(double v_inf, double r_p = 5.0)
{
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> uniform01(0.0, 1.0);
    
    double b_max = r_p * std::sqrt(1.0 + 2.0 / (v_inf * v_inf * r_p)); // b < b-max <==> pericenter < r_p
    double b = std::sqrt(uniform01(gen)) * b_max; // Sample impact parameter b in [0, b_max] using area-weighted distribution
    
    Vec3 n_in = randomUnitVector(gen); // incoming direction (from far away toward origin)
    Vec3 n_tmp = randomUnitVector(gen);
    if (std::fabs(n_in * n_tmp) > 0.99) n_tmp = randomUnitVector(gen);
    Vec3 ex = n_tmp - (n_in * n_tmp) * n_in; // component orthogonal to n_in
    ex = ex / norm(ex);
    
    double r_init = r_sphere; // Sufficiently far distance
    Vec3 r0 = -r_init * n_in; // Place particle at finite distance along -n_in (approaching the origin)
    
    double v_local = std::sqrt(v_inf * v_inf + 2.0 / r_init); // Calculate v from energy conservation
    
    double v_tan = (b * v_inf) / r_init; // Calculate v_tangential from angular momentum conservation
    
    if (v_tan > v_local) // Ensure feasibility: v_tan cannot exceed v_local. Clamp to feasible value and adjust b accordingly (rare for very large b or small r_init)
    {
        v_tan = v_local;
        std::cout << "ERORR: r_init is not large enough for this value of v" << std::endl;
    }
    
    double v_rad = std::sqrt(std::max(0.0, v_local * v_local - v_tan * v_tan));
    
    Vec3 v0 = v_rad * n_in + v_tan * ex; // Radial component should point inward (toward decreasing r), which is +n_in because r0 = -r_init*n_in
    
    double t0 = 2.0 * M_PI * uniform01(gen); // Random binary phase
    
    return {r0, v0, t0};
}

bool handleFarBoundKeplerMotion(Vec3 & r_old, Vec3 & v_old, double & t_old) // This function was entirely written by chatGPT. I checked (by comparing tranjectories with same initial conditions) that it seems to do its job approximately correctly, i.e.: when the particle exits r_sphere, collapse the binary to a point, solve the Keplerian motion of the particle until it reenters, and update the binary phase accordingly. However, it does not seem to lead to a significant performance improvement. Therefore, the function is not currently used.
{
    // If far outside but still bound, propagate analytically under a point-mass potential
    if (!(norm(r_old) > r_sphere && energy_approx(r_old, v_old) < 0)) return false;

    const double mu = 1.0;
    double r_norm = norm(r_old);

    // Specific angular momentum
    Vec3 h = angular_momentum(r_old, v_old);
    double h_norm = norm(h);
    if (h_norm < 1e-15) return false; // degenerate: nearly radial orbit, fallback to numerical integration

    // Specific energy
    double E_spec = energy_approx(r_old, v_old); // = v^2/2 - mu/r
    // semi-major axis
    double a = -mu / (2.0 * E_spec);
    if (a <= 0) return false; // not an ellipse (shouldn't happen given E_spec < 0 guard)

    // eccentricity from energy & ang.mom: e^2 = 1 + 2 E h^2 / mu^2
    double e2 = 1.0 + 2.0 * E_spec * h_norm * h_norm / (mu * mu);
    if (e2 < 0) e2 = 0.0; // numerical safety
    double e = std::sqrt(e2);

    // Compute eccentric anomaly E0 from current radius:
    // r = a (1 - e cos E)  =>  cos E = (1 - r/a) / e
    if (e == 0.0) {
        // circular orbit: nothing to do analytically for re-entry decision (radius fixed), bail out
        return false;
    }

    double cosE0 = (1.0 - r_norm / a) / e;
    cosE0 = std::clamp(cosE0, -1.0, 1.0);
    double E0 = std::acos(cosE0);
    // choose sign for E0 from radial velocity sign
    double vr = (r_old * v_old) / r_norm; // radial velocity
    if (vr < 0) E0 = 2.0 * M_PI - E0;

    double M0 = E0 - e * std::sin(E0);

    // mean motion
    double n_mean = std::sqrt(mu / (a * a * a));

    // Solve for eccentric anomaly(s) at r = r_exit (using the same radius as when the particle exited the sphere)
    double r_exit = r_sphere - 0.01;
    double cosE_bound = (1.0 - r_exit / a) / e;
    if (cosE_bound < -1.0 || cosE_bound > 1.0) {
        // the orbit never reaches r_exit (e.g. pericenter > r_exit) -> nothing to do
        return false;
    }

    double E_bound_1 = std::acos(std::clamp(cosE_bound, -1.0, 1.0));
    double E_bound_2 = 2.0 * M_PI - E_bound_1;

    double M_bound_1 = E_bound_1 - e * std::sin(E_bound_1);
    double M_bound_2 = E_bound_2 - e * std::sin(E_bound_2);

    // compute forward positive mean-anomaly differences (mod 2π)
    auto deltaMpos = [&](double M_target){
        double dM = M_target - M0;
        while (dM <= 1e-12) dM += 2.0 * M_PI; // ensure strictly positive
        return dM;
    };

    double dM1 = deltaMpos(M_bound_1);
    double dM2 = deltaMpos(M_bound_2);

    // pick the earliest future crossing
    double dM = (dM1 < dM2) ? dM1 : dM2;

    // time to crossing
    double delta_t = dM / n_mean;

    // advance mean anomaly and solve for eccentric anomaly at arrival (robust solver)
    double M_target = M0 + dM;
    double E_target = EccentricAnomaly(M_target, e);

    // compute true anomaly and radius at target (sanity)
    double cosE_t = std::cos(E_target);
    double sinE_t = std::sin(E_target);

    // Build orbital-plane unit vectors: e_hat points toward periapsis (Laplace-Runge-Lenz direction)
    Vec3 e_vec = (cross(v_old, h) / mu) - (r_old / r_norm);
    double e_mag = norm(e_vec);
    Vec3 e_hat;
    if (e_mag < 1e-15) {
        // nearly circular: choose periapsis direction along current position
        e_hat = r_old / r_norm;
    } else {
        e_hat = e_vec / e_mag;
    }

    Vec3 k_hat = h / h_norm; // orbital angular momentum direction
    Vec3 q_hat = cross(k_hat, e_hat); // second basis vector in orbital plane

    // Position and velocity in orbital plane using eccentric anomaly
    Vec3 r_new = (a * (cosE_t - e)) * e_hat + (a * std::sqrt(std::max(0.0, 1 - e*e)) * sinE_t) * q_hat;
    double r_new_norm = norm(r_new);

    double coef = std::sqrt(mu * a) / r_new_norm;
    Vec3 v_new = coef * (-sinE_t * e_hat + std::sqrt(std::max(0.0, 1 - e*e)) * cosE_t * q_hat);

    // Advance the binary phase/time by the elapsed delta_t
    t_old += delta_t;

    // update particle state for resume of numerical integration
    r_old = r_new;
    v_old = v_new;

    return true;
}

ParticleResult evolveParticle(const BinaryOrbit& orbit, const ParticleResult init, double Tmax, int max_steps, const std::function<Vec3(const Vec3&, double)> & forceFunc, double tol = 0.01) // Example values: tol = 0.02 for RK45, tol = 0.01 for Pihajoki, tol = 0.005 for velocity Verlet
{
    Vec3 r_old = init.r, v_old = init.v;
    Vec3 r = init.r, v = init.v;
    double t = init.t, t_old = init.t;
    double dt = std::min(0.01, 0.01 / norm(v));
    
    const double safety = 0.9; // Safety factor for dt updates
    const double p = 1.0; // Effective order for the error estimate
    
    tol = std::min(tol, tol / pow(std::sqrt(norm(v)), 4)); // Need a very small tolerance for large values of v, otherwise small numerical errors lead to huge changes in the particle's energy as it quickly shoots through the system. I played with the tolerance and settled on the current formula for the tolerance after a lot of trial and error
    
    int steps = 0;

    bool escaped = false;
    while (!stoppingCondition(r, v, t, Tmax, steps, max_steps, &escaped))
    {
        // This only seems to lead to a significant performance improvement for small q, otherwise it can be commented out:
        //if (handleFarBoundKeplerMotion(r_old, v_old, t_old)) continue;

        r = r_old;
        v = v_old;
        t = t_old;
        
        // with adaptive time step, Verlet and Yoshida are no longer symplectic, so it's better to use RK45
        //velocityVerletStep(r, v, t, dt, forceFunc);
        //yoshida4Step(r, v, t, dt, forceFunc);
        rk45Step(r, v, t, dt, forceFunc);
        //pihajokiStep(r, v, t, dt, forceFunc);
        //bulirschStoerStep(r, v, t, dt, forceFunc);
        
        Vec3 dv = v - v_old;
        double relChange = norm(dv) / (norm(v_old) + 1e-12); // Estimate relative change in velocity, avoid divide by 0
                
        double factor = safety * std::pow(tol / (relChange + 1e-12), 1.0 / (p + 1.0));
        factor = std::clamp(factor, 0.5, 2.0);
        double new_dt = dt * factor; // Adaptively adjust dt based on the measured change vs tolerance
        
        if (relChange < tol) // Step accepted: update positions, velocities, and time
        {
            //Vec3 r1, r2; interpolateBinaryPositions(orbit, t, r1, r2); std::cout << t << " " << r1 << " " << r2 << " " << r << " " << norm(r) << " " << energy_approx(r, v) << std::endl;
            steps++;

            r_old = r;
            v_old = v;
            t_old = t;
            
            dt = std::min(new_dt, dt * 1.5); // Limit too aggressive dt growth
        }
        else // Step rejected: shrink dt and retry
            dt = new_dt;
    }

    if (escaped)
        return {r, v, t};
    else
        return { {NAN, NAN, NAN}, {NAN, NAN, NAN}, t };
}

// Helper to compute mean and std deviation
std::pair<double, double> mean_std(const std::vector<double>& data)
{
    size_t n = data.size();
    if (n == 0)
        return {NAN, NAN};
    if (n == 1)
        return {data[0], 0.0};

    double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;

    double accum = 0.0;
    for (double x : data)
        accum += (x - mean) * (x - mean);

    double var = accum / n;
    if (var < 0.0) var = 0.0;  // guard against negative due to rounding

    double stdev = std::sqrt(var);
    return {mean, stdev};
}

// Helper for histograms
std::vector<int> makeHistogram(const std::vector<double>& data, double minVal, double maxVal, int nBins)
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

void printHist(const std::string& name, const std::vector<double>& data, int nBins, bool symmetric=true, bool logBins=false)
{
    double minVal, maxVal;
    std::vector<int> hist(nBins, 0);
    std::vector<double> binEdges(nBins+1, 0.0);
    bool useLogBins = logBins;
    if (useLogBins) {
        // Logarithmically spaced bins: minVal > 0
        minVal = std::max(1e-12, *std::min_element(data.begin(), data.end()));
        maxVal = *std::max_element(data.begin(), data.end());
        double logMin = std::log10(minVal);
        double logMax = std::log10(maxVal);
        double binWidth = (logMax - logMin) / nBins;
        // Compute bin edges
        for (int i = 0; i <= nBins; ++i) {
            binEdges[i] = std::pow(10.0, logMin + i * binWidth);
        }
        // Count in bins
        for (double x : data) {
            if (x < minVal || x > maxVal) continue;
            int bin = static_cast<int>((std::log10(x) - logMin) / binWidth);
            if (bin < 0) bin = 0;
            if (bin >= nBins) bin = nBins-1;
            hist[bin]++;
        }
        std::cout << "# Histogram " << name << "\n";
        for (int i = 0; i < nBins; ++i) {
            // Print the geometric mean of bin edges
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

std::vector<double> v_list(double q) // Generate N_v log-spaced values between v_min(q) and v_max(q)
{
    int N_v = 2000;
    double v_min = (1.0/10) * (std::sqrt(q)/(2*(1+q))) * std::sqrt(0.001);
    double v_max = 4 * (2*std::sqrt(q)/(1+q)) * std::sqrt(100);
    
    std::vector<double> v_list(N_v);
    if (N_v == 1)
        v_list[0] = v_min;
    else
        for (int i = 0; i < N_v; ++i)
            v_list[i] = std::exp( std::log(v_min) + i * (std::log(v_max) - std::log(v_min)) / (N_v - 1) );
    return v_list;
}

int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
    
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " q e\n";
        return 1;
    }
    
    //double q = 0.1, e = 0.6;
    double q = std::stod(argv[1]);
    double e = std::stod(argv[2]);
    
    double Tmax = 1e11;
    bool split_by_Tcut = false;  // toggle to enable/disable multi‑Tcut output

    int max_steps = 1e9;
    int N = 1e4;
    double rp_max = 5.0;
    
    // Define cutoff times for statistics (logarithmically spaced)
    std::vector<double> TmaxCuts;
    double Tmin_cut = 1e3;
    if (split_by_Tcut) {
        int n_cuts = 40;   // number of log‑spaced cut values
        double log_min = std::log10(Tmin_cut);
        double log_max = std::log10(Tmax);
        for (int i = 0; i < n_cuts; ++i) {
            double logv = log_min + (log_max - log_min) * i / (n_cuts - 1);
            TmaxCuts.push_back(std::pow(10.0, logv));
        }
        TmaxCuts.push_back(Tmax);  // ensure the final one includes all particles
    } else {
        TmaxCuts = {Tmax};
    }

    std::vector<double> v_values = v_list(q); // compute list of velocities

    // Precompute binary orbit (shared by all threads)
    BinaryOrbit orbit = precomputeBinaryOrbit(q, e, 100000);
    auto forceFunc = makeBinaryForce(orbit, q);

    // Prepare to collect all results for each v
    std::vector<std::vector<std::pair<ParticleResult,ParticleResult>>> all_results(v_values.size());
    std::vector<std::vector<EscRecord>> all_records(v_values.size());

    // Parallel loop over v-values
    #pragma omp parallel for schedule(dynamic)
    for (int iv = 0; iv < (int)v_values.size(); ++iv)
    {
        double v = v_values[iv];
        auto local_start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<ParticleResult, ParticleResult>> results;
        results.reserve(N);

        // NEW: store kernel observables for this v-bin
        std::vector<EscRecord> records;
        records.reserve(N); // upper bound

        for (int i = 0; i < N; ++i)
        {
            ParticleResult init = generateInitialConditions(v, rp_max);
            ParticleResult fin  = evolveParticle(orbit, init, Tmax, max_steps, forceFunc);

            if (!std::isnan(fin.r[0])) {
                results.push_back({init, fin});

                // --- NEW: compute asymptotic velocity vector and angles ---
                Vec3 v_inf_vec = asymptotic_velocity_approx(fin.r, fin.v);
                if (!std::isnan(v_inf_vec[0])) {
                    double vf = norm(v_inf_vec);
                    if (vf > 0.0) {
                        double cos_th = v_inf_vec[2] / vf;
                        // numerical safety
                        cos_th = std::clamp(cos_th, -1.0, 1.0);
                        double phi = std::atan2(v_inf_vec[1], v_inf_vec[0]);  // [-pi, pi]
                        double dt  = fin.t - init.t;

                        records.push_back({vf, cos_th, phi, dt});
                    }
                }
            }
        }

        all_results[iv]  = std::move(results);
        all_records[iv]  = std::move(records);

        // --- Write histograms to a file named with q and e ---
        /*
        static std::mutex hist_mutex;
        {
            std::lock_guard<std::mutex> lock(hist_mutex);
            static std::ofstream hist_out(
                ("histograms_q=" + std::to_string(q) + "_e=" + std::to_string(e) + ".txt")
            );
            if (!hist_out.is_open()) {
                std::cerr << "Error: could not open histograms_q=... for writing\n";
            } else {
                hist_out << "# v = " << v << "\n";
                std::streambuf* oldbuf = std::cout.rdbuf(hist_out.rdbuf());
                int nBins = 100;
                std::vector<double> deltaE, deltaT, dvx, dvy, dvz, dLx, dLy, dLz;
                for (const auto &pair : results) {
                    const auto &init = pair.first;
                    const auto &fin = pair.second;
                    deltaE.push_back(energy_approx(fin.r, fin.v) - energy_approx(init.r, init.v));
                    deltaT.push_back(fin.t - init.t);
                    Vec3 dv = asymptotic_velocity_approx(fin.r, fin.v) - asymptotic_velocity_approx(init.r, init.v);
                    Vec3 dL = angular_momentum(fin.r, fin.v) - angular_momentum(init.r, init.v);
                    dvx.push_back(dv[0]); dvy.push_back(dv[1]); dvz.push_back(dv[2]);
                    dLx.push_back(dL[0]); dLy.push_back(dL[1]); dLz.push_back(dL[2]);
                }
                printHist("DeltaE",  deltaE, nBins, true);
                printHist("DeltaVx", dvx,    nBins, true);
                printHist("DeltaVy", dvy,    nBins, true);
                printHist("DeltaVz", dvz,    nBins, true);
                printHist("DeltaLx", dLx,    nBins, true);
                printHist("DeltaLy", dLy,    nBins, true);
                printHist("DeltaLz", dLz,    nBins, true);
                printHist("DeltaT",  deltaT, nBins, false, true);
                hist_out << "\n";
                std::cout.rdbuf(oldbuf);
            }
        }
        */

        // Optionally print timing info
        auto local_end = std::chrono::high_resolution_clock::now();
        double local_elapsed = std::chrono::duration<double>(local_end - local_start).count();
        size_t N_resolved = results.size();
        #pragma omp critical
        std::cout << "Elapsed time for v=" << v << ": " << local_elapsed
                  << " seconds,\t resolved " << N_resolved << " particles\n";
    }
    // After parallel loop, write summary files for each TmaxCut
    // --- NEW: write binary file for hvs_analysis_1.py ---
    {
        std::ostringstream fbin;
        fbin << "hvs_q=" << q << "_e=" << e << ".bin";
        write_bin(fbin.str(), q, e, rp_max, r_sphere, v_values, N, all_records);
        std::cout << "Wrote " << fbin.str() << " (binary kernel file)\n";
    }

    for (double TmaxCut : TmaxCuts) {
        std::ostringstream fcut;
        fcut << "q=" << q << "_e=" << e << "_Tcut=";
        long long Tcut_int = static_cast<long long>(TmaxCut);
        fcut << Tcut_int << ".txt";
        std::ofstream out(fcut.str());
        out << "# N = " << N << ", rp_max = " << rp_max << ", r_sphere = " << r_sphere << ", T_cut = " << TmaxCut << "\n";
        out << "# v\tmean∆E\tSEM_∆E\t∆T\tSEM_∆T\t∆vx\tSEM_∆vx\t∆vy\tSEM_∆vy\t∆vz\tSEM_∆vz\t∆Lx\tSEM_∆Lx\t∆Ly\tSEM_∆Ly\t∆Lz\tSEM_∆Lz\tNresolved\n";
        for (int iv = 0; iv < (int)v_values.size(); ++iv) {
            double v = v_values[iv];
            // Filter particles with fin.t < TmaxCut
            // Recompute deltas only for these
            std::vector<double> deltaE, deltaT, dvx, dvy, dvz, dLx, dLy, dLz;
            for (const auto &pair : all_results[iv]) {
                const auto &init = pair.first;
                const auto &fin  = pair.second;
                if (fin.t < TmaxCut) {
                    deltaE.push_back(energy_approx(fin.r, fin.v) - energy_approx(init.r, init.v));
                    deltaT.push_back(fin.t - init.t);
                    Vec3 dv = asymptotic_velocity_approx(fin.r, fin.v) - asymptotic_velocity_approx(init.r, init.v);
                    //std::cout << init.v << " " << fin.v << std::endl;
                    //std::cout << asymptotic_velocity_approx(init.r, init.v) << " " << asymptotic_velocity_approx(fin.r, fin.v) << std::endl;
                    Vec3 dL = angular_momentum(fin.r, fin.v) - angular_momentum(init.r, init.v);
                    dvx.push_back(dv[0]); dvy.push_back(dv[1]); dvz.push_back(dv[2]);
                    dLx.push_back(dL[0]); dLy.push_back(dL[1]); dLz.push_back(dL[2]);
                }
            }
            size_t N_res = deltaE.size();
            if (N_res < 2) {
                out << v << "\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\t" << N_res << "\n";
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
            out << v << "\t"
                << msE.first  << "\t" << msE.second  / std::sqrt(N_res) << "\t"
                << msT.first  << "\t" << msT.second  / std::sqrt(N_res) << "\t"
                << msvx.first << "\t" << msvx.second / std::sqrt(N_res) << "\t"
                << msvy.first << "\t" << msvy.second / std::sqrt(N_res) << "\t"
                << msvz.first << "\t" << msvz.second / std::sqrt(N_res) << "\t"
                << msLx.first << "\t" << msLx.second / std::sqrt(N_res) << "\t"
                << msLy.first << "\t" << msLy.second / std::sqrt(N_res) << "\t"
                << msLz.first << "\t" << msLz.second / std::sqrt(N_res) << "\t"
                << N_res << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total elapsed time: " << elapsed_seconds << " seconds" << std::endl;

    return 0;
}
