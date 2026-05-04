/*
 * nbody_core.c
 * ============
 * C implementation of the binary-BH-in-uniform-medium N-body simulation.
 * Mirrors the arithmetic order of sim_uniform_medium.py for floating-point
 * reproducibility.  Compiled as a shared library and called via ctypes.
 */

#include "nbody_core.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Force computation                                                  */
/* ------------------------------------------------------------------ */

/*
 * Acceleration on every star from the two BHs (softened).
 * Matches the NumPy evaluation order:
 *   acc[i] += ((G * bh_mass[j]) * dr[i]) * inv_r3[i]
 */

static inline double min_image(double dx, double L)
{
    return dx - L * nearbyint(dx / L);
}

static void compute_star_accel(int N,
                               const double *star_pos,
                               const double *bh_pos,
                               const double *bh_mass,
                               double G, double eps,
                               double L,
                               double *acc)
{
    memset(acc, 0, (size_t)N * 3 * sizeof(double));
    double eps2 = eps * eps;

    for (int j = 0; j < 2; j++) {
        double Gm  = G * bh_mass[j];
        double bpx = bh_pos[j * 3 + 0];
        double bpy = bh_pos[j * 3 + 1];
        double bpz = bh_pos[j * 3 + 2];

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < N; i++) {
            double dx = min_image(bpx - star_pos[i * 3 + 0], L);
            double dy = min_image(bpy - star_pos[i * 3 + 1], L);
            double dz = min_image(bpz - star_pos[i * 3 + 2], L);
            double r2s = dx * dx + dy * dy + dz * dz + eps2;
            double inv_r3 = 1.0 / (r2s * sqrt(r2s));
            acc[i * 3 + 0] += (Gm * dx) * inv_r3;
            acc[i * 3 + 1] += (Gm * dy) * inv_r3;
            acc[i * 3 + 2] += (Gm * dz) * inv_r3;
        }
    }
}

/*
 * Acceleration on the two BHs from each other (unsoftened) and from
 * all stars (softened).
 *
 * BH-BH:  acc[0] += ((G * bh_mass[1]) * inv_r3) * dr
 *          acc[1] -= ((G * bh_mass[0]) * inv_r3) * dr
 *
 * Stars -> BH j:
 *   sum_k = sum_i( dr_ik * inv_r3_i )   (sequential accumulation)
 *   acc[j][k] += (G * m_star) * sum_k
 */
static void compute_bh_accel(const double *bh_pos,
                              const double *bh_mass,
                              int N,
                              const double *star_pos,
                              double m_star,
                              double G, double eps,
                              double L,
                              double *acc)
{
    memset(acc, 0, 6 * sizeof(double));
    double eps2 = eps * eps;

    /* BH-BH (no softening) */
    double drx = min_image(bh_pos[3] - bh_pos[0], L);
    double dry = min_image(bh_pos[4] - bh_pos[1], L);
    double drz = min_image(bh_pos[5] - bh_pos[2], L);
    double r2  = drx * drx + dry * dry + drz * drz;
    double r   = sqrt(r2);
    double inv_r3 = (r > 0.0) ? 1.0 / (r2 * r) : 0.0;

    double f0 = (G * bh_mass[1]) * inv_r3;
    double f1 = (G * bh_mass[0]) * inv_r3;
    acc[0] += f0 * drx;  acc[1] += f0 * dry;  acc[2] += f0 * drz;
    acc[3] -= f1 * drx;  acc[4] -= f1 * dry;  acc[5] -= f1 * drz;

    /* Stars -> each BH */
    for (int j = 0; j < 2; j++) {
        double bpx = bh_pos[j * 3 + 0];
        double bpy = bh_pos[j * 3 + 1];
        double bpz = bh_pos[j * 3 + 2];
        double sx = 0.0, sy = 0.0, sz = 0.0;

        for (int i = 0; i < N; i++) {
            double dx = min_image(star_pos[i * 3 + 0] - bpx, L);
            double dy = min_image(star_pos[i * 3 + 1] - bpy, L);
            double dz = min_image(star_pos[i * 3 + 2] - bpz, L);
            double r2s = dx * dx + dy * dy + dz * dz + eps2;
            double ir3 = 1.0 / (r2s * sqrt(r2s));
            sx += dx * ir3;
            sy += dy * ir3;
            sz += dz * ir3;
        }

        double Gms = G * m_star;
        acc[j * 3 + 0] += Gms * sx;
        acc[j * 3 + 1] += Gms * sy;
        acc[j * 3 + 2] += Gms * sz;
    }
}

/* ------------------------------------------------------------------ */
/*  Periodic wrapping: equivalent to ((x + L/2) % L) - L/2            */
/*  Python/NumPy % always returns non-negative for positive divisor.   */
/* ------------------------------------------------------------------ */

static inline double wrap_one(double x, double L)
{
    double r = fmod(x + 0.5 * L, L);
    if (r < 0.0) r += L;
    return r - 0.5 * L;
}

/* ------------------------------------------------------------------ */
/*  Orbital elements from BH state                                     */
/* ------------------------------------------------------------------ */

static void orbital_elements(const double *bh_pos,
                              const double *bh_vel,
                              const double *bh_mass,
                              double G,
                              double *a_out, double *e_out,
                              double *evec_out)
{
    double Mtot = bh_mass[0] + bh_mass[1];
    double drx = bh_pos[3] - bh_pos[0];
    double dry = bh_pos[4] - bh_pos[1];
    double drz = bh_pos[5] - bh_pos[2];
    double dvx = bh_vel[3] - bh_vel[0];
    double dvy = bh_vel[4] - bh_vel[1];
    double dvz = bh_vel[5] - bh_vel[2];

    double r  = sqrt(drx * drx + dry * dry + drz * drz);
    double v2 = dvx * dvx + dvy * dvy + dvz * dvz;
    double E_orb = 0.5 * v2 - G * Mtot / r;

    /* Angular momentum h = dr x dv */
    double hx = dry * dvz - drz * dvy;
    double hy = drz * dvx - drx * dvz;
    double hz = drx * dvy - dry * dvx;

    /* Eccentricity vector: A = (v x h) / (G*M) - r_hat
     * where v x h: */
    double vxhx = dvy * hz - dvz * hy;
    double vxhy = dvz * hx - dvx * hz;
    double vxhz = dvx * hy - dvy * hx;

    double inv_GM = 1.0 / (G * Mtot);
    double inv_r  = (r > 0.0) ? 1.0 / r : 0.0;

    double Ax = vxhx * inv_GM - drx * inv_r;
    double Ay = vxhy * inv_GM - dry * inv_r;
    double Az = vxhz * inv_GM - drz * inv_r;

    double e = sqrt(Ax * Ax + Ay * Ay + Az * Az);

    if (evec_out) {
        evec_out[0] = Ax;
        evec_out[1] = Ay;
        evec_out[2] = Az;
    }

    if (E_orb >= 0.0) {
        *a_out = -1.0;
        *e_out = e;
        return;
    }

    double a = -G * Mtot / (2.0 * E_orb);

    *a_out = a;
    *e_out = e;
}

/* ------------------------------------------------------------------ */
/*  Sink handling: remove stars inside r_sink of either BH, compact,   */
/*  then append replacements from the random buffer.                   */
/* ------------------------------------------------------------------ */

static int handle_sinks(double *star_pos, double *star_vel, int N,
                         const double *bh_pos, double r_sink,
                         int replenish,
                         const double *rand_pos_buf,
                         const double *rand_vel_buf,
                         int rand_buf_len, int *rand_idx,
                         const double *cm,
                         const double *cm_v)
{
    double r_sink2 = r_sink * r_sink;
    int n_removed = 0;

    /* Pass 1: compact surviving stars to the front */
    int write = 0;
    for (int i = 0; i < N; i++) {
        double dx0 = star_pos[i * 3 + 0] - bh_pos[0];
        double dy0 = star_pos[i * 3 + 1] - bh_pos[1];
        double dz0 = star_pos[i * 3 + 2] - bh_pos[2];
        double dx1 = star_pos[i * 3 + 0] - bh_pos[3];
        double dy1 = star_pos[i * 3 + 1] - bh_pos[4];
        double dz1 = star_pos[i * 3 + 2] - bh_pos[5];
        double d0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
        double d1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1;

        if (d0 >= r_sink2 && d1 >= r_sink2) {
            /* keep this star */
            if (write != i) {
                star_pos[write * 3 + 0] = star_pos[i * 3 + 0];
                star_pos[write * 3 + 1] = star_pos[i * 3 + 1];
                star_pos[write * 3 + 2] = star_pos[i * 3 + 2];
                star_vel[write * 3 + 0] = star_vel[i * 3 + 0];
                star_vel[write * 3 + 1] = star_vel[i * 3 + 1];
                star_vel[write * 3 + 2] = star_vel[i * 3 + 2];
            }
            write++;
        }
    }
    n_removed = N - write;

    /* Pass 2: append new particles from random buffer */
    if (replenish && n_removed > 0) {
        for (int k = 0; k < n_removed; k++) {
            int ri = *rand_idx;
            if (ri >= rand_buf_len) return -1;
            int slot = write + k;
            star_pos[slot * 3 + 0] = rand_pos_buf[ri * 3 + 0] + cm[0];
            star_pos[slot * 3 + 1] = rand_pos_buf[ri * 3 + 1] + cm[1];
            star_pos[slot * 3 + 2] = rand_pos_buf[ri * 3 + 2] + cm[2];
            star_vel[slot * 3 + 0] = rand_vel_buf[ri * 3 + 0] + cm_v[0];
            star_vel[slot * 3 + 1] = rand_vel_buf[ri * 3 + 1] + cm_v[1];
            star_vel[slot * 3 + 2] = rand_vel_buf[ri * 3 + 2] + cm_v[2];
            (*rand_idx)++;
        }
    }

    return n_removed;
}

/* ------------------------------------------------------------------ */
/*  Velocity-based ejection: remove stars with |v - v_cm| > v_cut,     */
/*  compact, then append replacements from the random buffer.          */
/*  This mimics the infinite medium where slingshot-ejected stars      */
/*  leave to infinity rather than wrapping around periodically.        */
/* ------------------------------------------------------------------ */

static int handle_velocity_ejection(double *star_pos, double *star_vel, int N,
                                    const double *cm_vel, double v_cut2,
                                    double r_protect2,
                                    int replenish,
                                    const double *rand_pos_buf,
                                    const double *rand_vel_buf,
                                    int rand_buf_len, int *rand_idx,
                                    const double *cm)
{
    int n_removed = 0;

    /* Pass 1: compact surviving stars to the front */
    int write = 0;
    for (int i = 0; i < N; i++) {
        double dvx = star_vel[i * 3 + 0] - cm_vel[0];
        double dvy = star_vel[i * 3 + 1] - cm_vel[1];
        double dvz = star_vel[i * 3 + 2] - cm_vel[2];
        double v2  = dvx * dvx + dvy * dvy + dvz * dvz;

        /* Also compute distance from binary CoM — only eject fast stars
         * that are far from the binary (completed their encounter).
         * Stars close to the binary may be mid-scattering and should
         * not be removed, even if their velocity exceeds the cut. */
        double drx = star_pos[i * 3 + 0] - cm[0];
        double dry = star_pos[i * 3 + 1] - cm[1];
        double drz = star_pos[i * 3 + 2] - cm[2];
        double r2  = drx * drx + dry * dry + drz * drz;

        if (v2 <= v_cut2 || r2 < r_protect2) {
            /* keep this star */
            if (write != i) {
                star_pos[write * 3 + 0] = star_pos[i * 3 + 0];
                star_pos[write * 3 + 1] = star_pos[i * 3 + 1];
                star_pos[write * 3 + 2] = star_pos[i * 3 + 2];
                star_vel[write * 3 + 0] = star_vel[i * 3 + 0];
                star_vel[write * 3 + 1] = star_vel[i * 3 + 1];
                star_vel[write * 3 + 2] = star_vel[i * 3 + 2];
            }
            write++;
        }
    }
    n_removed = N - write;

    /* Pass 2: append new particles from random buffer */
    if (replenish && n_removed > 0) {
        for (int k = 0; k < n_removed; k++) {
            int ri = *rand_idx;
            if (ri >= rand_buf_len) return -1;
            int slot = write + k;
            star_pos[slot * 3 + 0] = rand_pos_buf[ri * 3 + 0] + cm[0];
            star_pos[slot * 3 + 1] = rand_pos_buf[ri * 3 + 1] + cm[1];
            star_pos[slot * 3 + 2] = rand_pos_buf[ri * 3 + 2] + cm[2];
            star_vel[slot * 3 + 0] = rand_vel_buf[ri * 3 + 0] + cm_vel[0];
            star_vel[slot * 3 + 1] = rand_vel_buf[ri * 3 + 1] + cm_vel[1];
            star_vel[slot * 3 + 2] = rand_vel_buf[ri * 3 + 2] + cm_vel[2];
            (*rand_idx)++;
        }
    }

    return n_removed;
}

/* ------------------------------------------------------------------ */
/*  CoM force from stars on the binary (for output recording)          */
/* ------------------------------------------------------------------ */

static void compute_cm_force(int N,
                             const double *star_pos,
                             const double *bh_pos,
                             const double *bh_mass,
                             double m_star, double G, double eps,
                             double L,
                             double *f_cm)
{
    double eps2 = eps * eps;
    f_cm[0] = f_cm[1] = f_cm[2] = 0.0;

    for (int j = 0; j < 2; j++) {
        double bpx = bh_pos[j * 3 + 0];
        double bpy = bh_pos[j * 3 + 1];
        double bpz = bh_pos[j * 3 + 2];
        double sx = 0.0, sy = 0.0, sz = 0.0;

        for (int i = 0; i < N; i++) {
            double dx = min_image(star_pos[i * 3 + 0] - bpx, L);
            double dy = min_image(star_pos[i * 3 + 1] - bpy, L);
            double dz = min_image(star_pos[i * 3 + 2] - bpz, L);
            double r2s = dx * dx + dy * dy + dz * dz + eps2;
            double inv_r3 = 1.0 / (r2s * sqrt(r2s));
            sx += dx * inv_r3;
            sy += dy * inv_r3;
            sz += dz * inv_r3;
        }

        double Gmms = (G * bh_mass[j]) * m_star;
        f_cm[0] += Gmms * sx;
        f_cm[1] += Gmms * sy;
        f_cm[2] += Gmms * sz;
    }
}

/* ------------------------------------------------------------------ */
/*  Main simulation loop                                               */
/* ------------------------------------------------------------------ */

int run_simulation(
    double G, double M, double m1, double m2, double m_star,
    double L, double dt, double softening, double r_sink,
    double v_eject_cut, double r_eject_factor,
    int N, int n_steps, int output_every, int replenish,
    double *bh_pos, double *bh_vel, const double *bh_mass,
    double *star_pos, double *star_vel,
    const double *rand_pos_buf, const double *rand_vel_buf,
    int rand_buf_len,
    double *output, int *n_rows, int *rand_consumed)
{
    double *bh_acc   = (double *)malloc(6 * sizeof(double));
    double *star_acc = (double *)malloc((size_t)N * 3 * sizeof(double));
    if (!bh_acc || !star_acc) { free(bh_acc); free(star_acc); return -2; }

    int    row = 0;
    int    n_removed_total = 0;
    int    n_vejected_total = 0;
    int    rand_idx = 0;
    double t = 0.0;
    double half_dt = 0.5 * dt;
    double v_cut2 = (v_eject_cut > 0.0) ? v_eject_cut * v_eject_cut : -1.0;
    int    progress_interval = n_steps / 20;
    if (progress_interval < 1) progress_interval = 1;

    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (int step = 0; step < n_steps; step++) {

        /* ---- First half-kick ---- */
        compute_bh_accel(bh_pos, bh_mass, N, star_pos, m_star,
                         G, softening, L, bh_acc);
        compute_star_accel(N, star_pos, bh_pos, bh_mass,
                           G, softening, L, star_acc);

        for (int k = 0; k < 6; k++)
            bh_vel[k] += half_dt * bh_acc[k];

        for (int i = 0; i < N * 3; i++)
            star_vel[i] += half_dt * star_acc[i];

        /* ---- Drift ---- */
        for (int k = 0; k < 6; k++)
            bh_pos[k] += dt * bh_vel[k];

        for (int i = 0; i < N * 3; i++)
            star_pos[i] += dt * star_vel[i];

        /* ---- Periodic wrapping relative to binary CoM ---- */
        double cm[3];
        for (int k = 0; k < 3; k++)
            cm[k] = (bh_mass[0] * bh_pos[k] + bh_mass[1] * bh_pos[3 + k]) / M;

        for (int i = 0; i < N; i++) {
            for (int k = 0; k < 3; k++) {
                double rel = star_pos[i * 3 + k] - cm[k];
                star_pos[i * 3 + k] = wrap_one(rel, L) + cm[k];
            }
        }

        /* ---- Second half-kick ---- */
        compute_bh_accel(bh_pos, bh_mass, N, star_pos, m_star,
                         G, softening, L, bh_acc);
        compute_star_accel(N, star_pos, bh_pos, bh_mass,
                           G, softening, L, star_acc);

        for (int k = 0; k < 6; k++)
            bh_vel[k] += half_dt * bh_acc[k];

        for (int i = 0; i < N * 3; i++)
            star_vel[i] += half_dt * star_acc[i];

        t += dt;


        double cm_v[3];
        for (int k = 0; k < 3; k++) {
            cm[k] = (bh_mass[0] * bh_pos[k] + bh_mass[1] * bh_pos[3 + k]) / M;
            cm_v[k] = (bh_mass[0] * bh_vel[k] + bh_mass[1] * bh_vel[3 + k]) / M;
        }

        int n_rm = handle_sinks(star_pos, star_vel, N,
                                bh_pos, r_sink, replenish,
                                rand_pos_buf, rand_vel_buf,
                                rand_buf_len, &rand_idx, cm, cm_v);
        if (n_rm < 0) {
            free(bh_acc); free(star_acc);
            return -1;
        }
        n_removed_total += n_rm;

        /* ---- Velocity-based ejection ---- */
        if (v_cut2 > 0.0) {
            /* Protection radius: don't eject fast stars within r_eject_factor*a
             * of the binary CoM — they may be mid-encounter, not yet ejected.
             * Compute current semi-major axis for this. */
            double a_cur, e_dum;
            orbital_elements(bh_pos, bh_vel, bh_mass, G, &a_cur, &e_dum, NULL);
            double r_prot = (a_cur > 0.0) ? r_eject_factor * a_cur : r_eject_factor;
            double r_protect2 = r_prot * r_prot;

            int n_vej = handle_velocity_ejection(
                            star_pos, star_vel, N,
                            cm_v, v_cut2, r_protect2,
                            replenish,
                            rand_pos_buf, rand_vel_buf,
                            rand_buf_len, &rand_idx, cm);
            if (n_vej < 0) {
                free(bh_acc); free(star_acc);
                return -1;
            }
            n_vejected_total += n_vej;
        }

        /* ---- Output recording ---- */
        if (step % output_every == 0 || step == n_steps - 1) {
            double cm_p[3], cm_v[3];
            for (int k = 0; k < 3; k++) {
                cm_p[k] = (bh_mass[0] * bh_pos[k] +
                           bh_mass[1] * bh_pos[3 + k]) / M;
                cm_v[k] = (bh_mass[0] * bh_vel[k] +
                           bh_mass[1] * bh_vel[3 + k]) / M;
            }

            double a_bin, e_bin;
            double evec[3];
            orbital_elements(bh_pos, bh_vel, bh_mass, G, &a_bin, &e_bin, evec);

            double f_cm[3];
            compute_cm_force(N, star_pos, bh_pos, bh_mass, m_star,
                             G, softening, L, f_cm);

            double *r = output + row * 18;
            r[0]  = t;
            r[1]  = cm_p[0];  r[2]  = cm_p[1];  r[3]  = cm_p[2];
            r[4]  = cm_v[0];  r[5]  = cm_v[1];  r[6]  = cm_v[2];
            r[7]  = a_bin;    r[8]  = e_bin;
            r[9]  = (double)N;
            r[10] = f_cm[0];  r[11] = f_cm[1];  r[12] = f_cm[2];
            r[13] = (double)n_removed_total;
            r[14] = (double)n_vejected_total;
            r[15] = evec[0];  r[16] = evec[1];  r[17] = evec[2];
            row++;
        }

        /* ---- Progress ---- */
        if ((step + 1) % progress_interval == 0 || step == n_steps - 1) {
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            double wall = (double)(ts_now.tv_sec  - ts_start.tv_sec) +
                          (double)(ts_now.tv_nsec - ts_start.tv_nsec) * 1e-9;
            double cm_p[3], cm_v[3];
            for (int k = 0; k < 3; k++) {
                cm_p[k] = (bh_mass[0] * bh_pos[k] +
                           bh_mass[1] * bh_pos[3 + k]) / M;
                cm_v[k] = (bh_mass[0] * bh_vel[k] +
                           bh_mass[1] * bh_vel[3 + k]) / M;
            }
            double R = sqrt(cm_p[0]*cm_p[0] + cm_p[1]*cm_p[1] + cm_p[2]*cm_p[2]);
            double V = sqrt(cm_v[0]*cm_v[0] + cm_v[1]*cm_v[1] + cm_v[2]*cm_v[2]);
            double a_bin, e_dum;
            orbital_elements(bh_pos, bh_vel, bh_mass, G, &a_bin, &e_dum, NULL);
            printf("  step %8d/%d  t=%.2f  |R|=%.4f  |V|=%.5f  "
                   "a=%.4f  N*=%d  wall=%.1fs\n",
                   step + 1, n_steps, t, R, V, a_bin, N, wall);
            fflush(stdout);
        }
    }

    *n_rows = row;
    *rand_consumed = rand_idx;

    free(bh_acc);
    free(star_acc);
    return 0;
}
