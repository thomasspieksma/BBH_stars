#ifndef NBODY_CORE_H
#define NBODY_CORE_H

int run_simulation(
    double G, double M, double m1, double m2, double m_star,
    double L, double dt, double softening, double r_sink,
    double v_eject_cut, double r_eject_factor,
    int N, int n_steps, int output_every, int replenish,
    double *bh_pos, double *bh_vel, const double *bh_mass,
    double *star_pos, double *star_vel,
    const double *rand_pos_buf, const double *rand_vel_buf,
    int rand_buf_len,
    double *output, int *n_rows, int *rand_consumed);

#endif
