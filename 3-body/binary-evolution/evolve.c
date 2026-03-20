/*
 * evolve.c — Binary evolution solver (C implementation).
 *
 * Evolves the binary state (e, V/sigma, varpi) as a function of the
 * hardening variable xi = ln(a_h/a), using three-body scattering data
 * and Chandrasekhar dynamical friction.
 *
 * Compile:  cc -O2 -march=native -Wall -Wextra -std=c99 -o evolve evolve.c -lm
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <glob.h>
#include <time.h>
#include <getopt.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NS       7          /* state dimension: e, Vx, Vy, varpi, t, x, y */
#define NR       5          /* rate components: H, K, Pe, Pn, Q            */
#define NQ       8          /* quantities per SH bin                       */
#define MAX_LM   50
#define MAX_NSH  ((MAX_LM+1)*(MAX_LM+1))
#define MAX_ST   4          /* max Lagrange stencil size                   */
#define MAX_EF   200        /* max eccentricity files                      */

/* ══════════════════════════════════════════════════════════════════════════
 *  Data structures
 * ══════════════════════════════════════════════════════════════════════════*/
typedef struct { double v_inf; int n_esc; double *M, *S; } HBin;

typedef struct {
    double q, e, rp_max, r_sphere;
    int n_per_v, l_max, n_sh, n_v;
    HBin *bins;
    double *tw;     /* trapezoidal weights, length n_v */
} HFile;

typedef struct {
    int n_used;
    int idx[MAX_ST];
    double w[MAX_ST];
    double sig[MAX_ST][NR];   /* per-file binary-frame sigmas */
} FInfo;

/* Global dataset */
static int    g_nf;                     /* number of loaded files       */
static double g_ev[MAX_EF];            /* sorted e values              */
static HFile  g_hf[MAX_EF];           /* harmonics data per e         */
static int    g_nep;                   /* count of e > 0 values        */
static double *g_eg;                   /* pointer into g_ev past e=0   */

/* Solver globals */
static double g_q, g_roah;
static int    g_chmode;                /* 0=none 1=integral 2=constant */
static int    g_nst;                   /* Lagrange stencil size        */
static int    g_fr_e, g_fr_Vx, g_fr_Vy, g_fr_w;
static int    g_nnoise, g_ntot;
static int    g_rhs_count;
static double g_rhs_t0;

/* ══════════════════════════════════════════════════════════════════════════
 *  Utilities
 * ══════════════════════════════════════════════════════════════════════════*/
static inline double clamp(double x, double lo, double hi)
{ return x < lo ? lo : (x > hi ? hi : x); }



/* ══════════════════════════════════════════════════════════════════════════
 *  Real spherical harmonics  Y^R_{lm}(cos_theta, phi)
 * ══════════════════════════════════════════════════════════════════════════*/
static void real_Ylm_all(double ct, double phi, int lm, double *out)
{
    int sz = lm + 1, nsh = sz * sz;
    double st = sqrt(fmax(0.0, 1.0 - ct*ct));
    double cm[MAX_LM+1], sm[MAX_LM+1];
    cm[0] = 1.0; sm[0] = 0.0;
    if (lm >= 1) { cm[1] = cos(phi); sm[1] = sin(phi); }
    for (int m = 2; m < sz; m++) {
        cm[m] = 2*cm[1]*cm[m-1] - cm[m-2];
        sm[m] = 2*cm[1]*sm[m-1] - sm[m-2];
    }
    double plm[MAX_LM+1][MAX_LM+1];
    memset(plm, 0, sizeof(plm));
    plm[0][0] = 1.0;
    for (int m = 1; m < sz; m++)
        plm[m][m] = -(2*m - 1) * st * plm[m-1][m-1];
    for (int m = 0; m < sz - 1; m++)
        plm[m+1][m] = (2*m + 1) * ct * plm[m][m];
    for (int m = 0; m < sz; m++)
        for (int l = m + 2; l < sz; l++)
            plm[l][m] = ((2*l-1)*ct*plm[l-1][m] - (l+m-1)*plm[l-2][m]) / (l - m);

    double fact[2*MAX_LM+2];
    fact[0] = 1.0;
    for (int i = 1; i < 2*lm+2; i++) fact[i] = fact[i-1] * i;

    memset(out, 0, nsh * sizeof(double));
    for (int l = 0; l < sz; l++) {
        double K0 = sqrt((2*l+1) / (4.0*M_PI));
        out[l*l + l] = K0 * plm[l][0];
        for (int m = 1; m <= l; m++) {
            double Km = sqrt((2*l+1)/(4.0*M_PI) * fact[l-m]/fact[l+m]);
            double val = Km * plm[l][m];
            out[l*l + l + m] = sqrt(2.0) * val * cm[m];
            out[l*l + l - m] = sqrt(2.0) * val * sm[m];
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Scaled modified spherical Bessel functions  exp(-x) i_l(x)
 *  via Miller backward recurrence.
 * ══════════════════════════════════════════════════════════════════════════*/
static void smsb(double x, int lm, double *out)
{
    /* For very small x, i_l(x)*exp(-x) ≈ x^l/(2l+1)!! ≈ 0 for l>0.
       The backward recurrence overflows when (2*lm+51)/x > ~10^8. */
    if (x < 1e-4) {
        out[0] = (x > 1e-30) ? -expm1(-2.0*x) / (2.0*x) : 1.0;
        double c = out[0];
        for (int l = 1; l <= lm; l++) {
            c *= x / (2*l + 1);
            out[l] = c;
        }
        return;
    }
    int Ls = lm + 25;
    double r[MAX_LM + 27];   /* stack-allocated, avoids malloc in hot loop */
    r[Ls + 1] = 0.0;
    r[Ls]     = 1.0;
    for (int l = Ls; l >= 1; l--)
        r[l-1] = r[l+1] + (2*l + 1) / x * r[l];
    double exact_i0 = -expm1(-2.0 * x) / (2.0 * x);
    double scale = exact_i0 / r[0];
    for (int l = 0; l <= lm; l++) out[l] = r[l] * scale;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Trapezoidal weights (prepend v=0 left boundary)
 * ══════════════════════════════════════════════════════════════════════════*/
static void trap_weights(const double *v, int n, double *w)
{
    /* extended array: v_ext = [0, v[0], v[1], ..., v[n-1]]  (n+1 elems) */
    int ne = n + 1;
    double *we = (double*)malloc(ne * sizeof(double));
    we[0] = v[0] / 2.0;                          /* (v_ext[1]-v_ext[0])/2 */
    we[ne-1] = (v[n-1] - v[n-2]) / 2.0;          /* last endpoint        */
    for (int i = 1; i < ne - 1; i++) {
        double vp = (i+1 < ne) ? v[i] : v[n-1];  /* v_ext[i+1] = v[i]    */
        double vm = (i-1 > 0)  ? v[i-2] : 0.0;   /* v_ext[i-1]           */
        we[i] = (vp - vm) / 2.0;
    }
    for (int j = 0; j < n; j++) w[j] = we[j+1];
    free(we);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Binary file I/O — read harmonics_*.bin
 * ══════════════════════════════════════════════════════════════════════════*/
static int read_harmonics(const char *fn, HFile *hf)
{
    FILE *f = fopen(fn, "rb");
    if (!f) return -1;
    int32_t nv32, npv32, lm32;
    if (fread(&hf->q,       8, 1, f) != 1) goto fail;
    if (fread(&hf->e,       8, 1, f) != 1) goto fail;
    if (fread(&hf->rp_max,  8, 1, f) != 1) goto fail;
    if (fread(&hf->r_sphere,8, 1, f) != 1) goto fail;
    if (fread(&nv32,        4, 1, f) != 1) goto fail;
    if (fread(&npv32,       4, 1, f) != 1) goto fail;
    if (fread(&lm32,        4, 1, f) != 1) goto fail;
    hf->n_v     = nv32;
    hf->n_per_v = npv32;
    hf->l_max   = lm32;
    hf->n_sh    = (lm32 + 1) * (lm32 + 1);

    hf->bins = (HBin*)malloc(hf->n_v * sizeof(HBin));
    for (int j = 0; j < hf->n_v; j++) {
        int32_t ne32;
        fread(&hf->bins[j].v_inf, 8, 1, f);
        fread(&ne32,              4, 1, f);
        hf->bins[j].n_esc = ne32;
        int sz = NQ * hf->n_sh;
        hf->bins[j].M = (double*)malloc(sz * sizeof(double));
        hf->bins[j].S = (double*)malloc(sz * sizeof(double));
        fread(hf->bins[j].M, 8, sz, f);
        fread(hf->bins[j].S, 8, sz, f);
    }

    /* trapezoidal weights */
    double *vv = (double*)malloc(hf->n_v * sizeof(double));
    for (int j = 0; j < hf->n_v; j++) vv[j] = hf->bins[j].v_inf;
    hf->tw = (double*)malloc(hf->n_v * sizeof(double));
    trap_weights(vv, hf->n_v, hf->tw);
    free(vv);

    fclose(f);
    return 0;
fail:
    fclose(f);
    return -1;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Dataset loading — glob for all eccentricity files at given q
 * ══════════════════════════════════════════════════════════════════════════*/
static int load_dataset(double q, const char *data_dir)
{
    char pattern[2048], q_str[64];
    glob_t gl;
    int found = 0;

    /* try %g first (gives "1" for q=1.0, "0.2" for q=0.2, etc.) */
    snprintf(q_str, sizeof(q_str), "%g", q);
    snprintf(pattern, sizeof(pattern), "%s/harmonics_q=%s_e=*.bin", data_dir, q_str);
    if (glob(pattern, 0, NULL, &gl) == 0 && gl.gl_pathc > 0) found = 1;

    /* fallback: try integer form */
    if (!found && q == (int)q) {
        snprintf(q_str, sizeof(q_str), "%d", (int)q);
        snprintf(pattern, sizeof(pattern), "%s/harmonics_q=%s_e=*.bin", data_dir, q_str);
        if (glob(pattern, 0, NULL, &gl) == 0 && gl.gl_pathc > 0) found = 1;
    }
    /* fallback: try float form */
    if (!found) {
        snprintf(q_str, sizeof(q_str), "%.1f", q);
        snprintf(pattern, sizeof(pattern), "%s/harmonics_q=%s_e=*.bin", data_dir, q_str);
        if (glob(pattern, 0, NULL, &gl) == 0 && gl.gl_pathc > 0) found = 1;
    }
    if (!found) {
        fprintf(stderr, "No harmonics files for q=%g in %s\n", q, data_dir);
        return -1;
    }

    g_nf = 0;
    for (size_t i = 0; i < gl.gl_pathc && g_nf < MAX_EF; i++) {
        const char *p = strstr(gl.gl_pathv[i], "_e=");
        if (!p) continue;
        double ev;
        if (sscanf(p + 3, "%lf", &ev) != 1) continue;
        if (read_harmonics(gl.gl_pathv[i], &g_hf[g_nf]) != 0) continue;
        g_ev[g_nf] = ev;
        g_nf++;
    }
    globfree(&gl);

    if (g_nf == 0) {
        fprintf(stderr, "Failed to read any harmonics files\n");
        return -1;
    }

    /* sort by eccentricity */
    /* simple insertion sort (small n) */
    for (int i = 1; i < g_nf; i++) {
        double key_e = g_ev[i];
        HFile key_h = g_hf[i];
        int j = i - 1;
        while (j >= 0 && g_ev[j] > key_e) {
            g_ev[j+1] = g_ev[j];
            g_hf[j+1] = g_hf[j];
            j--;
        }
        g_ev[j+1] = key_e;
        g_hf[j+1] = key_h;
    }

    /* set up e > 0 grid */
    g_nep = g_nf;
    g_eg  = g_ev;
    if (g_nf > 0 && g_ev[0] == 0.0) {
        g_nep = g_nf - 1;
        g_eg  = g_ev + 1;
    }

    printf("  Loaded q=%g: %d eccentricity values (e = %.1f ... %.1f)\n",
           q, g_nf, g_ev[0], g_ev[g_nf-1]);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Reweight from harmonics — core computation engine
 * ══════════════════════════════════════════════════════════════════════════*/

/* out[0..4] = H,K,Pe,Pn,Q   out[5..9] = sH,sK,sPe,sPn,sQ */
static void reweight(const HFile *hf, const double V[3], double sigma,
                     double *out)
{
    double q = hf->q, ecc = hf->e, rp = hf->rp_max;
    int lm = hf->l_max, nsh = hf->n_sh;
    double mu = q / ((1+q)*(1+q));
    double Vm = sqrt(V[0]*V[0] + V[1]*V[1] + V[2]*V[2]);

    double Ylm[MAX_NSH];
    if (Vm > 1e-30) {
        double nv[3] = {-V[0]/Vm, -V[1]/Vm, -V[2]/Vm};
        real_Ylm_all(nv[2], atan2(nv[1], nv[0]), lm, Ylm);
    } else {
        memset(Ylm, 0, nsh * sizeof(double));
        Ylm[0] = 1.0 / sqrt(4.0 * M_PI);
    }

    int lidx[MAX_NSH];
    for (int l = 0; l <= lm; l++)
        for (int m = l*l; m < (l+1)*(l+1); m++) lidx[m] = l;

    double is2 = 1.0 / (sigma * sigma);
    double n1 = pow(2.0 * M_PI * sigma * sigma, -1.5);

    double aE = 0, vE = 0, aF[3]={0}, vF[3]={0};
    double aL[3]={0}, vL[3]={0}, aW = 0, vW = 0;
    double s1[MAX_LM+1], s2a[MAX_LM+1];

    for (int j = 0; j < hf->n_v; j++) {
        int Nj = hf->bins[j].n_esc;
        if (Nj == 0) continue;
        double vj = hf->bins[j].v_inf;
        double bm = rp * sqrt(1.0 + 2.0 / (vj*vj * rp));
        double Aj = 4.0 * M_PI * M_PI * vj*vj*vj * bm*bm * hf->tw[j];
        double al = vj * Vm * is2;

        if (al > 1e-30) { smsb(al, lm, s1); smsb(2.0*al, lm, s2a); }
        else {
            memset(s1, 0, (lm+1)*sizeof(double)); s1[0] = 1.0;
            memset(s2a,0, (lm+1)*sizeof(double)); s2a[0]= 1.0;
        }

        double G1 = n1 * exp(-(vj-Vm)*(vj-Vm) / (2.0*sigma*sigma));
        double G2 = n1*n1 * exp(-(vj-Vm)*(vj-Vm) / (sigma*sigma));

        double w1[MAX_NSH], w2[MAX_NSH];
        for (int k = 0; k < nsh; k++) {
            w1[k] = s1[lidx[k]] * Ylm[k];
            w2[k] = s2a[lidx[k]] * Ylm[k];
        }

        double mX[NQ], mX2[NQ];
        for (int r = 0; r < NQ; r++) {
            double d1 = 0, d2 = 0;
            const double *Mr = hf->bins[j].M + r * nsh;
            const double *Sr = hf->bins[j].S + r * nsh;
            for (int k = 0; k < nsh; k++) { d1 += Mr[k]*w1[k]; d2 += Sr[k]*w2[k]; }
            mX[r]  = G1 * 4.0*M_PI * d1;
            mX2[r] = G2 * 4.0*M_PI * d2;
        }

        double A2 = Aj*Aj;
        aE += Aj*mX[0]; vE += A2*fmax(mX2[0]-mX[0]*mX[0],0.0)/Nj;
        for (int k = 0; k < 3; k++) {
            double vx = mX2[1+k] - mX[1+k]*mX[1+k];
            aF[k] += Aj*mX[1+k]; vF[k] += A2*fmax(vx,0.0)/Nj;
        }
        for (int k = 0; k < 3; k++) {
            double vx = mX2[4+k] - mX[4+k]*mX[4+k];
            aL[k] += Aj*mX[4+k]; vL[k] += A2*fmax(vx,0.0)/Nj;
        }
        double vw = mX2[7] - mX[7]*mX[7];
        aW += Aj*mX[7]; vW += A2*fmax(vw,0.0)/Nj;
    }

    double P  = -aE,   sP  = sqrt(fmax(vE,0.0));
    double Fx = -aF[0], sFx = sqrt(fmax(vF[0],0.0));
    double Fy = -aF[1], sFy = sqrt(fmax(vF[1],0.0));
    double tz = -aL[2], stz = sqrt(fmax(vL[2],0.0));
    double wd = aW,     swd = sqrt(fmax(vW,0.0));

    double H  = -2.0*sigma*P/mu, sH = 2.0*sigma*sP/mu;

    double K = 0.0, sK = 0.0;
    if (fabs(P) > 0 && fabs(tz) > 1e-300) {
        double c = sqrt(1.0 - ecc*ecc) / (2.0*ecc);
        K  = -(1.0 - ecc*ecc)/(2.0*ecc) + c * tz / P;
        sK = c * fabs(tz/P) * sqrt((stz/tz)*(stz/tz) + (sP/P)*(sP/P));
    } else { K = NAN; sK = NAN; }

    double Pe = NAN, sPe = NAN, Pn = NAN, sPn = NAN;
    double Qv = NAN, sQv = NAN;
    double c2s = mu / (2.0*sigma);
    if (fabs(P) > 0 && fabs(Fx) > 1e-300) {
        Pe  = -c2s * Fx / P;
        sPe = c2s * fabs(Fx/P) * sqrt((sFx/Fx)*(sFx/Fx)+(sP/P)*(sP/P));
    }
    if (fabs(P) > 0 && fabs(Fy) > 1e-300) {
        Pn  = -c2s * Fy / P;
        sPn = c2s * fabs(Fy/P) * sqrt((sFy/Fy)*(sFy/Fy)+(sP/P)*(sP/P));
    }
    if (fabs(P) > 0 && fabs(wd) > 1e-300) {
        Qv  = -(mu/2.0) * wd / P;
        sQv = (mu/2.0)*fabs(wd/P)*sqrt((swd/wd)*(swd/wd)+(sP/P)*(sP/P));
    }

    out[0] = H;
    out[1] = isfinite(K)  ? K  : 0.0;
    out[2] = isfinite(Pe) ? Pe : 0.0;
    out[3] = isfinite(Pn) ? Pn : 0.0;
    out[4] = isfinite(Qv) ? Qv : 0.0;
    out[5] = isfinite(sH)  ? sH  : 0.0;
    out[6] = isfinite(sK)  ? sK  : 0.0;
    out[7] = isfinite(sPe) ? sPe : 0.0;
    out[8] = isfinite(sPn) ? sPn : 0.0;
    out[9] = isfinite(sQv) ? sQv : 0.0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Lagrange interpolation weights and derivatives
 * ══════════════════════════════════════════════════════════════════════════*/
static void lagrange_wd(double e, const double *se, int n, double *w, double *dw)
{
    for (int k = 0; k < n; k++) {
        w[k] = 1.0;
        for (int j = 0; j < n; j++)
            if (j != k) w[k] *= (e - se[j]) / (se[k] - se[j]);
    }
    for (int k = 0; k < n; k++) {
        dw[k] = 0.0;
        for (int m = 0; m < n; m++) {
            if (m == k) continue;
            double t = 1.0 / (se[k] - se[m]);
            for (int j = 0; j < n; j++)
                if (j != k && j != m) t *= (e - se[j]) / (se[k] - se[j]);
            dw[k] += t;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  compute_rates — eccentricity interpolation + frame rotation
 *  vals[10] = {H,K,Px,Py,Q, sH,sK,sPx,sPy,sQ}  (lab frame)
 *  derivs[5] = {dK_de, dH_de, dPx_de, dPy_de, dQ_de}  (if non-NULL)
 *  fi filled if non-NULL
 * ══════════════════════════════════════════════════════════════════════════*/
static void compute_rates(double xi, double e, double Vx, double Vy,
                          double varpi, double *vals, double *derivs,
                          FInfo *fi)
{
    double cw = cos(varpi), sw = sin(varpi);
    double Ve = Vx*cw + Vy*sw, Vn = -Vx*sw + Vy*cw;
    double sc = sqrt(g_q) / (2.0*(1.0+g_q)) * exp(-xi/2.0);
    double Vc[3] = {Ve*sc, Vn*sc, 0.0};

    int need_deriv = (derivs != NULL), need_fi = (fi != NULL);
    int fast = !need_deriv && !need_fi;

    /* find the index in eg closest to e */
    int snap = 0;
    { double best = fabs(g_eg[0] - e);
      for (int i = 1; i < g_nep; i++) {
          double d = fabs(g_eg[i] - e);
          if (d < best) { best = d; snap = i; }
      }
    }

    /* Index into g_ev: offset if g_eg != g_ev */
    int eoff = (int)(g_eg - g_ev);  /* 0 or 1 */

    /* Fast path: snap or boundary */
    if (fast && fabs(g_eg[snap] - e) < 1e-10) {
        reweight(&g_hf[eoff + snap], Vc, sc, vals);
        /* rotate binary→lab for P */
        double Pe = vals[2], Pn = vals[3], sPe = vals[7], sPn = vals[8];
        vals[2] = Pe*cw - Pn*sw;
        vals[3] = Pe*sw + Pn*cw;
        vals[7] = sqrt(cw*cw*sPe*sPe + sw*sw*sPn*sPn);
        vals[8] = sqrt(sw*sw*sPe*sPe + cw*cw*sPn*sPn);
        if (fi) { fi->n_used=1; fi->idx[0]=snap; fi->w[0]=1.0;
                   for(int r=0;r<NR;r++) fi->sig[0][r]=vals[5+r]; }
        if (derivs) memset(derivs, 0, 5*sizeof(double));
        return;
    }
    if (fast && e <= g_eg[0]) {
        reweight(&g_hf[eoff], Vc, sc, vals);
        double Pe=vals[2],Pn=vals[3],sPe=vals[7],sPn=vals[8];
        vals[2]=Pe*cw-Pn*sw; vals[3]=Pe*sw+Pn*cw;
        vals[7]=sqrt(cw*cw*sPe*sPe+sw*sw*sPn*sPn);
        vals[8]=sqrt(sw*sw*sPe*sPe+cw*cw*sPn*sPn);
        return;
    }
    if (fast && e >= g_eg[g_nep-1]) {
        reweight(&g_hf[eoff + g_nep - 1], Vc, sc, vals);
        double Pe=vals[2],Pn=vals[3],sPe=vals[7],sPn=vals[8];
        vals[2]=Pe*cw-Pn*sw; vals[3]=Pe*sw+Pn*cw;
        vals[7]=sqrt(cw*cw*sPe*sPe+sw*sw*sPn*sPn);
        vals[8]=sqrt(sw*sw*sPe*sPe+cw*cw*sPn*sPn);
        return;
    }
    if (g_nep < 2) {
        reweight(&g_hf[eoff], Vc, sc, vals);
        double Pe=vals[2],Pn=vals[3],sPe=vals[7],sPn=vals[8];
        vals[2]=Pe*cw-Pn*sw; vals[3]=Pe*sw+Pn*cw;
        vals[7]=sqrt(cw*cw*sPe*sPe+sw*sw*sPn*sPn);
        vals[8]=sqrt(sw*sw*sPe*sPe+cw*cw*sPn*sPn);
        if (fi) { fi->n_used=1; fi->idx[0]=0; fi->w[0]=1.0;
                   for(int r=0;r<NR;r++) fi->sig[0][r]=vals[5+r]; }
        if (derivs) memset(derivs, 0, 5*sizeof(double));
        return;
    }

    /* Lagrange interpolation */
    double ec = clamp(e, g_eg[0], g_eg[g_nep-1]);
    int ns = g_nst < g_nep ? g_nst : g_nep;
    /* find insertion point */
    int idx = 0;
    while (idx < g_nep && g_eg[idx] < ec) idx++;
    int i0 = idx - ns/2;
    if (i0 < 0) i0 = 0;
    if (i0 + ns > g_nep) i0 = g_nep - ns;

    double se[MAX_ST];
    for (int k = 0; k < ns; k++) se[k] = g_eg[i0 + k];

    double wl[MAX_ST], dwl[MAX_ST];
    lagrange_wd(ec, se, ns, wl, dwl);

    /* evaluate at each stencil point */
    double sv[MAX_ST][10];   /* [stencil][H,K,Pe,Pn,Q,sH,sK,sPe,sPn,sQ] */
    for (int k = 0; k < ns; k++)
        reweight(&g_hf[eoff + i0 + k], Vc, sc, sv[k]);

    /* interpolate central values */
    double H=0,K=0,Pe=0,Pn=0,Qv=0;
    for (int k = 0; k < ns; k++) {
        H  += wl[k]*sv[k][0]; K  += wl[k]*sv[k][1];
        Pe += wl[k]*sv[k][2]; Pn += wl[k]*sv[k][3]; Qv += wl[k]*sv[k][4];
    }
    /* interpolate uncertainties (quadrature) */
    double sH=0,sK=0,sPe2=0,sPn2=0,sQ2=0;
    for (int k = 0; k < ns; k++) {
        double w2 = wl[k]*wl[k];
        sH  += w2*sv[k][5]*sv[k][5]; sK  += w2*sv[k][6]*sv[k][6];
        sPe2 += w2*sv[k][7]*sv[k][7]; sPn2 += w2*sv[k][8]*sv[k][8];
        sQ2  += w2*sv[k][9]*sv[k][9];
    }
    sH = sqrt(sH); sK = sqrt(sK);
    double sPe_v = sqrt(sPe2), sPn_v = sqrt(sPn2), sQv = sqrt(sQ2);

    /* derivatives */
    if (derivs) {
        double dH=0,dK=0,dPe=0,dPn=0,dQ=0;
        for (int k = 0; k < ns; k++) {
            dH += dwl[k]*sv[k][0]; dK += dwl[k]*sv[k][1];
            dPe += dwl[k]*sv[k][2]; dPn += dwl[k]*sv[k][3];
            dQ += dwl[k]*sv[k][4];
        }
        /* rotate Pe,Pn derivatives to lab frame */
        derivs[0] = dK;
        derivs[1] = dH;
        derivs[2] = dPe*cw - dPn*sw;
        derivs[3] = dPe*sw + dPn*cw;
        derivs[4] = dQ;
    }

    /* file info */
    if (fi) {
        fi->n_used = ns;
        for (int k = 0; k < ns; k++) {
            fi->idx[k] = i0 + k;
            fi->w[k]   = wl[k];
            for (int r = 0; r < NR; r++) fi->sig[k][r] = sv[k][5+r];
        }
    }

    /* rotate to lab frame */
    vals[0] = H;   vals[1] = K;
    vals[2] = Pe*cw - Pn*sw;
    vals[3] = Pe*sw + Pn*cw;
    vals[4] = Qv;
    vals[5] = sH;  vals[6] = sK;
    vals[7] = sqrt(cw*cw*sPe_v*sPe_v + sw*sw*sPn_v*sPn_v);
    vals[8] = sqrt(sw*sw*sPe_v*sPe_v + cw*cw*sPn_v*sPn_v);
    vals[9] = sQv;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Chandrasekhar dynamical friction
 * ══════════════════════════════════════════════════════════════════════════*/
static double ln_lambda(double u, double xi, double q, double ro)
{
    double ao = exp(-xi);
    double ratio = 8.0*(1+q)*(1+q)*exp(xi) / (5.0*q*u*u);
    double bm = 5.0*ao*sqrt(1.0 + ratio);
    double lnL = log(ro / bm);
    return lnL < 0.0 ? 0.0 : lnL;
}

/* Adaptive Simpson's rule */
typedef struct { double Vt, xi, q, ro; } ChCtx;

static double ch_integrand(double u, void *ctx_)
{
    ChCtx *c = (ChCtx*)ctx_;
    if (u < 1e-30) return 0.0;
    double lnL = ln_lambda(u, c->xi, c->q, c->ro);
    if (lnL <= 0.0) return 0.0;
    double al = u * c->Vt;
    double ke;
    if (al < 1e-4) {
        ke = exp(-(u*u + c->Vt*c->Vt)/2.0) * (-al*al*al/3.0 - al*al*al*al*al/30.0);
    } else {
        double em = exp(-0.5*(u - c->Vt)*(u - c->Vt));
        double ep = exp(-0.5*(u + c->Vt)*(u + c->Vt));
        ke = 0.5*((1.0 - al)*em - (1.0 + al)*ep);
    }
    return lnL * 2.0 * ke / (al*al);
}

static double adapt_simpson_r(double (*f)(double,void*), void *ctx,
    double a, double b, double fa, double fb, double fm,
    double whole, double tol, int d)
{
    double c = (a+b)/2.0;
    double fl = f((a+c)/2.0, ctx), fr = f((c+b)/2.0, ctx);
    double left  = (c-a)/6.0*(fa + 4.0*fl + fm);
    double right = (b-c)/6.0*(fm + 4.0*fr + fb);
    double s = left + right;
    double err = fabs(s - whole);
    if (d >= 20 || err <= 15.0*tol || err < 1e-15*(fabs(s)+1e-300))
        return s + (s - whole)/15.0;
    return adapt_simpson_r(f,ctx,a,c,fa,fm,fl,left,tol/2,d+1)
         + adapt_simpson_r(f,ctx,c,b,fm,fb,fr,right,tol/2,d+1);
}

static double adapt_simpson(double (*f)(double,void*), void *ctx,
                            double a, double b, double tol)
{
    double fa = f(a,ctx), fb = f(b,ctx), fm = f((a+b)/2.0, ctx);
    double w = (b-a)/6.0*(fa + 4.0*fm + fb);
    return adapt_simpson_r(f, ctx, a, b, fa, fb, fm, w, tol, 0);
}

static double ch_decel_integral(double Vt, double xi, double q, double ro)
{
    if (Vt < 1e-15) return 0.0;
    ChCtx ctx = {Vt, xi, q, ro};
    double umax = fmax(Vt, 5.0) + 8.0;
    double J = adapt_simpson(ch_integrand, &ctx, 0.0, umax, 1e-10);
    return J / sqrt(2.0 * M_PI);
}

static double ch_decel_constant(double Vt, double xi, double q, double ro)
{
    if (Vt < 1e-15) return 0.0;
    double lnL = ln_lambda(1.0, xi, q, ro);
    if (lnL <= 0.0) return 0.0;
    double X = Vt / sqrt(2.0);
    double br = erf(X) - 2.0*X/sqrt(M_PI)*exp(-X*X);
    return -lnL / (Vt*Vt) * br;
}

static void compute_ch(double Vx, double Vy, double H, double xi,
                       double ch_base, double *Cx, double *Cy)
{
    double Vm = hypot(Vx, Vy);
    if (g_chmode == 0 || Vm < 1e-10 || H < 1e-6) { *Cx = *Cy = 0; return; }
    double Jv = (g_chmode == 1)
        ? ch_decel_integral(Vm, xi, g_q, g_roah)
        : ch_decel_constant(Vm, xi, g_q, g_roah);
    double pf = ch_base / H;
    *Cx = pf * Jv * Vx / Vm;
    *Cy = pf * Jv * Vy / Vm;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ODE solution storage
 * ══════════════════════════════════════════════════════════════════════════*/
typedef struct {
    int n, cap, ns;    /* n=steps stored, cap=capacity, ns=state dim */
    double *xi, *y;    /* xi[cap], y[cap*ns] row-major               */
    int ok, term;
} Sol;

static void sol_init(Sol *s, int ns, int cap0) {
    s->n = 0; s->cap = cap0; s->ns = ns;
    s->xi = (double*)malloc(cap0 * sizeof(double));
    s->y  = (double*)malloc(cap0 * ns * sizeof(double));
    s->ok = 1; s->term = 0;
}
static void sol_push(Sol *s, double xi, const double *y) {
    if (s->n >= s->cap) {
        s->cap *= 2;
        s->xi = (double*)realloc(s->xi, s->cap * sizeof(double));
        s->y  = (double*)realloc(s->y,  s->cap * s->ns * sizeof(double));
    }
    s->xi[s->n] = xi;
    memcpy(s->y + s->n * s->ns, y, s->ns * sizeof(double));
    s->n++;
}
static void sol_free(Sol *s) { free(s->xi); free(s->y); }

/* ══════════════════════════════════════════════════════════════════════════
 *  Dormand-Prince RK45 with adaptive step control
 * ══════════════════════════════════════════════════════════════════════════*/
typedef void (*rhs_fn)(double xi, const double *y, double *dy, void *ctx);

static void rk45_solve(rhs_fn rhs, double xi0, double xi1, const double *y0,
                       int n, double rtol, double atol, double max_step,
                       void *ctx,
                       double (*event)(double, const double *),
                       Sol *sol)
{
    /* Dormand-Prince coefficients */
    static const double
        c2=1./5, c3=3./10, c4=4./5, c5=8./9,
        a21=1./5,
        a31=3./40,       a32=9./40,
        a41=44./45,      a42=-56./15,     a43=32./9,
        a51=19372./6561, a52=-25360./2187,a53=64448./6561,a54=-212./729,
        a61=9017./3168,  a62=-355./33,    a63=46732./5247,a64=49./176,  a65=-5103./18656,
        a71=35./384,                      a73=500./1113,  a74=125./192, a75=-2187./6784, a76=11./84,
        e1=71./57600,    e3=-71./16695,   e4=71./1920,    e5=-17253./339200,
        e6=22./525,      e7=-1./40;

    sol_init(sol, n, 512);

    double *y   = (double*)malloc(n * sizeof(double));
    double *yn  = (double*)malloc(n * sizeof(double));
    double *yt  = (double*)malloc(n * sizeof(double));
    double *k1  = (double*)malloc(n * sizeof(double));
    double *k2  = (double*)malloc(n * sizeof(double));
    double *k3  = (double*)malloc(n * sizeof(double));
    double *k4  = (double*)malloc(n * sizeof(double));
    double *k5  = (double*)malloc(n * sizeof(double));
    double *k6  = (double*)malloc(n * sizeof(double));
    double *k7  = (double*)malloc(n * sizeof(double));

    memcpy(y, y0, n * sizeof(double));
    sol_push(sol, xi0, y);

    double xi = xi0;
    double h;

    rhs(xi0, y, k1, ctx);   /* first evaluation (also used for h0 estimate) */

    /* initial step selection (Hairer-Norsett-Wanner algorithm) */
    {
        double dy_max = 0, y_max = 0;
        int nc = n < NS ? n : NS;
        for (int i = 0; i < nc; i++) {
            double a = fabs(k1[i]);
            if (a > dy_max) dy_max = a;
            a = fabs(y[i]);
            if (a > y_max) y_max = a;
        }
        double h0 = (dy_max > 1e-30) ? 0.01 * fmax(y_max, 1.0) / dy_max : 1e-4;
        h0 = fmin(h0, (xi1 - xi0) * 0.01);
        h0 = fmin(h0, max_step);
        h0 = fmax(h0, 1e-6);
        for (int i = 0; i < n; i++) yt[i] = y[i] + h0 * k1[i];
        rhs(xi0 + h0, yt, k2, ctx);
        double d2_max = 0;
        for (int i = 0; i < nc; i++) {
            double a = fabs(k2[i] - k1[i]) / h0;
            if (a > d2_max) d2_max = a;
        }
        double h1 = (d2_max > 1e-30)
            ? sqrt(0.01 / d2_max)
            : fmax(10.0 * h0, (xi1 - xi0) * 1e-3);
        h1 = fmin(h1, 100.0 * h0);
        h1 = fmin(h1, max_step);
        h1 = fmax(h1, 1e-6);
        h = fmin(h0, h1);
    }
    rhs(xi, y, k1, ctx);   /* recompute k1 at xi0 (FSAL start) */

    int n_eval = 0, max_eval = 10000, n_reject = 0;

    while (xi < xi1 - 1e-14 * fabs(xi1)) {
        if (xi + h > xi1) h = xi1 - xi;
        if (h < 1e-14 * fmax(1.0, fabs(xi))) break;
        if (n_eval >= max_eval) { sol->term = 2; break; }

        /* stages */
        for (int i=0;i<n;i++) yt[i] = y[i] + h*a21*k1[i];
        rhs(xi+c2*h, yt, k2, ctx);

        for (int i=0;i<n;i++) yt[i] = y[i] + h*(a31*k1[i]+a32*k2[i]);
        rhs(xi+c3*h, yt, k3, ctx);

        for (int i=0;i<n;i++) yt[i] = y[i] + h*(a41*k1[i]+a42*k2[i]+a43*k3[i]);
        rhs(xi+c4*h, yt, k4, ctx);

        for (int i=0;i<n;i++) yt[i] = y[i] + h*(a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i]);
        rhs(xi+c5*h, yt, k5, ctx);

        for (int i=0;i<n;i++) yt[i] = y[i] + h*(a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i]);
        rhs(xi+h, yt, k6, ctx);

        /* 5th order solution */
        for (int i=0;i<n;i++) yn[i] = y[i] + h*(a71*k1[i]+a73*k3[i]+a74*k4[i]+a75*k5[i]+a76*k6[i]);

        rhs(xi+h, yn, k7, ctx);  /* FSAL */

        /* error estimate (over primary state variables only) */
        int nerr = n < NS ? n : NS;
        double err_norm = 0;
        for (int i = 0; i < nerr; i++) {
            double sc = atol + rtol * fmax(fabs(y[i]), fabs(yn[i]));
            double ei = h * (e1*k1[i]+e3*k3[i]+e4*k4[i]+e5*k5[i]+e6*k6[i]+e7*k7[i]);
            err_norm += (ei/sc)*(ei/sc);
        }
        err_norm = sqrt(err_norm / nerr);

        n_eval += 7;

        /* NaN protection: reject step and shrink */
        if (!isfinite(err_norm)) {
            h *= 0.1;
            if (h < 1e-14 * fmax(1.0, fabs(xi))) break;
            rhs(xi, y, k1, ctx);
            n_reject++;
            if (n_reject > 200) { sol->term = 2; break; }
            continue;
        }

        if (err_norm <= 1.0) {
            /* accept */
            xi += h;
            memcpy(y, yn, n * sizeof(double));
            memcpy(k1, k7, n * sizeof(double));  /* FSAL */
            sol_push(sol, xi, y);
            n_reject = 0;

            /* event check */
            if (event && event(xi, y) < 0.0) { sol->term = 1; break; }
        } else {
            n_reject++;
            if (n_reject > 200) { sol->term = 2; break; }
        }

        /* step size update */
        double factor = (err_norm > 0.0) ? 0.9 * pow(err_norm, -0.2) : 5.0;
        if (factor > 5.0) factor = 5.0;
        if (factor < 0.2) factor = 0.2;
        h *= factor;
        if (h > max_step) h = max_step;
    }

    free(y); free(yn); free(yt);
    free(k1); free(k2); free(k3); free(k4); free(k5); free(k6); free(k7);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Full ODE right-hand side  (full_covariance mode)
 * ══════════════════════════════════════════════════════════════════════════*/
static double g_emax;

static void full_rhs(double xi, const double *y, double *dy, void *ctx)
{
    (void)ctx;
    g_rhs_count++;
    if (g_rhs_count == 1) g_rhs_t0 = (double)clock() / CLOCKS_PER_SEC;
    if (g_rhs_count % 20 == 0) {
        double el = (double)clock()/CLOCKS_PER_SEC - g_rhs_t0;
        printf("\r  xi=%.3f  e=%.4f  |V|/s=%.5f  [%d evals, %.1fs]   ",
               xi, y[0], hypot(y[1],y[2]), g_rhs_count, el);
        fflush(stdout);
    }

    double ec = clamp(y[0], 1e-6, g_eg[g_nep-1]);
    double Vx = isfinite(y[1]) ? y[1] : 0.0;
    double Vy = isfinite(y[2]) ? y[2] : 0.0;
    double ww = isfinite(y[3]) ? y[3] : 0.0;

    double ch_base = 16.0*M_PI*(1+g_q)*(1+g_q)*exp(xi)/g_q;
    double DV = 0.01, DW = 0.01;

    /* nominal rates */
    double v0[10], dv0[5];
    FInfo finfo;
    compute_rates(xi, ec, Vx, Vy, ww, v0, dv0, &finfo);
    double H = v0[0]; if (H < 1e-6) H = 1e-6;
    double K=v0[1], Px=v0[2], Py=v0[3], Qv=v0[4];
    double sH=v0[5], sK=v0[6], sPe=v0[7], sPn=v0[8], sQv=v0[9];
    (void)sH; (void)sK; (void)sPe; (void)sPn; (void)sQv;

    double Cx, Cy;
    compute_ch(Vx, Vy, H, xi, ch_base, &Cx, &Cy);

    if (!isfinite(H) || H < 1e-6) H = 1e-6;
    double exi = exp(xi), exiH = exi/H, exiH2 = exi/(H*H);

    /* state derivatives */
    double de  = g_fr_e  ? 0.0 : K;
    double dVx = g_fr_Vx ? 0.0 : Px + Cx;
    double dVy = g_fr_Vy ? 0.0 : Py + Cy;
    double dw  = g_fr_w  ? 0.0 : Qv;
    double dt  = exiH;
    dy[0] = de;  dy[1] = dVx;  dy[2] = dVy;  dy[3] = dw;
    dy[4] = dt;  dy[5] = Vx*dt;  dy[6] = Vy*dt;

    /* Jacobian J (7x7) */
    double J[NS][NS];
    memset(J, 0, sizeof(J));

    /* column 0: eccentricity (analytic) */
    double coH[2] = {0,0};
    if (H > 1e-10) { coH[0] = Cx/H; coH[1] = Cy/H; }
    J[0][0] = dv0[0];                              /* dK/de */
    J[1][0] = dv0[2] - coH[0]*dv0[1];              /* dPx/de - Ch_x/H * dH/de */
    J[2][0] = dv0[3] - coH[1]*dv0[1];
    J[3][0] = dv0[4];
    J[4][0] = -exiH2 * dv0[1];
    J[5][0] = Vx * J[4][0];
    J[6][0] = Vy * J[4][0];

    /* columns 1,2: velocity (finite differences) */
    for (int col = 1; col <= 2; col++) {
        if ((col == 1 && g_fr_Vx) || (col == 2 && g_fr_Vy)) continue;
        double dVxo = (col==1) ? DV : 0.0;
        double dVyo = (col==2) ? DV : 0.0;
        double vp[10], vm[10];
        compute_rates(xi, ec, Vx+dVxo, Vy+dVyo, ww, vp, NULL, NULL);
        compute_rates(xi, ec, Vx-dVxo, Vy-dVyo, ww, vm, NULL, NULL);
        double Hp = fmax(vp[0],1e-6), Hm = fmax(vm[0],1e-6);
        double Cxp,Cyp,Cxm,Cym;
        compute_ch(Vx+dVxo, Vy+dVyo, Hp, xi, ch_base, &Cxp, &Cyp);
        compute_ch(Vx-dVxo, Vy-dVyo, Hm, xi, ch_base, &Cxm, &Cym);
        double i2d = 1.0/(2.0*DV);
        J[0][col] = (vp[1]-vm[1])*i2d;
        J[1][col] = ((vp[2]+Cxp)-(vm[2]+Cxm))*i2d;
        J[2][col] = ((vp[3]+Cyp)-(vm[3]+Cym))*i2d;
        J[3][col] = (vp[4]-vm[4])*i2d;
        J[4][col] = (exi/Hp - exi/Hm)*i2d;
    }
    J[5][1] = dt + Vx*J[4][1];  J[5][2] = Vx*J[4][2];
    J[6][1] = Vy*J[4][1];       J[6][2] = dt + Vy*J[4][2];

    /* column 3: varpi */
    if (!g_fr_w) {
        double vp[10], vm[10];
        compute_rates(xi, ec, Vx, Vy, ww+DW, vp, NULL, NULL);
        compute_rates(xi, ec, Vx, Vy, ww-DW, vm, NULL, NULL);
        double Hp = fmax(vp[0],1e-6), Hm = fmax(vm[0],1e-6);
        double Cxp,Cyp,Cxm,Cym;
        compute_ch(Vx, Vy, Hp, xi, ch_base, &Cxp, &Cyp);
        compute_ch(Vx, Vy, Hm, xi, ch_base, &Cxm, &Cym);
        double i2d = 1.0/(2.0*DW);
        J[0][3] = (vp[1]-vm[1])*i2d;
        J[1][3] = ((vp[2]+Cxp)-(vm[2]+Cxm))*i2d;
        J[2][3] = ((vp[3]+Cyp)-(vm[3]+Cym))*i2d;
        J[3][3] = (vp[4]-vm[4])*i2d;
        J[4][3] = (exi/Hp - exi/Hm)*i2d;
        J[5][3] = Vx*J[4][3];
        J[6][3] = Vy*J[4][3];
    }

    /* Loading matrix B (7x5) and per-file loading G (7 x n_noise) */
    double cw = cos(ww), sw = sin(ww);
    double B[NS][NR];
    memset(B, 0, sizeof(B));
    B[0][1] = 1.0;
    if (H > 1e-10) { B[1][0] = -Cx/H; B[2][0] = -Cy/H; }
    B[1][2] = cw;  B[1][3] = -sw;
    B[2][2] = sw;  B[2][3] = cw;
    B[3][4] = 1.0;
    B[4][0] = -exiH2;
    B[5][0] = -Vx*exiH2;
    B[6][0] = -Vy*exiH2;

    /* G: 7 x n_noise  (stored in dy after the state part) */
    int nn = g_nnoise;
    /* dF = J*F + G */
    const double *F = y + NS;       /* F[i*nn+j] */
    double *dF = dy + NS;

    /* build G and multiply simultaneously */
    for (int i = 0; i < NS; i++) {
        for (int j = 0; j < nn; j++) {
            double sum = 0.0;
            /* J*F row i, col j */
            for (int k = 0; k < NS; k++)
                sum += J[i][k] * F[k*nn + j];
            /* add G[i][j] */
            /* G[:, base+r] = w_s * sig_s_r * B[:, r] */
            /* check if j belongs to a stencil file */
            int file_idx = j / NR;
            int rate_idx = j % NR;
            double g_val = 0.0;
            for (int s = 0; s < finfo.n_used; s++) {
                if (finfo.idx[s] == file_idx) {
                    g_val = finfo.w[s] * finfo.sig[s][rate_idx] * B[i][rate_idx];
                    break;
                }
            }
            dF[i*nn + j] = sum + g_val;
        }
    }
}

static double full_event(double xi, const double *y)
{
    (void)xi;
    return g_emax - y[0];
}

/* ══════════════════════════════════════════════════════════════════════════
 *  V=0 reference ODE right-hand side
 * ══════════════════════════════════════════════════════════════════════════*/
static int g_v0_rhs_count;
static double g_v0_rhs_t0;

static void v0_rhs(double xi, const double *y, double *dy, void *ctx)
{
    (void)ctx;
    g_v0_rhs_count++;
    if (g_v0_rhs_count == 1)
        g_v0_rhs_t0 = (double)clock()/CLOCKS_PER_SEC;
    if (g_v0_rhs_count % 20 == 0) {
        double el = (double)clock()/CLOCKS_PER_SEC - g_v0_rhs_t0;
        printf("\r  (V=0 ref) xi=%.3f  e=%.4f  [%d evals, %.1fs]   ",
               xi, y[0], g_v0_rhs_count, el);
        fflush(stdout);
    }

    double ec = clamp(y[0], 1e-6, 1.0-1e-6);
    double v0[10], dv[5];
    FInfo finfo;
    compute_rates(xi, ec, 0.0, 0.0, 0.0, v0, dv, &finfo);

    double H = fmax(v0[0], 1e-6);
    double K = v0[1];
    double exi = exp(xi), exiH2 = exi/(H*H);

    dy[0] = K;
    dy[1] = exi/H;

    /* 2x2 Jacobian */
    double Jm[2][2] = {{dv[0], 0.0}, {-exiH2 * dv[1], 0.0}};

    int nn = g_nnoise;
    const double *F = y + 2;
    double *dF = dy + 2;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < nn; j++) {
            double sum = Jm[i][0]*F[0*nn+j] + Jm[i][1]*F[1*nn+j];

            double g_val = 0.0;
            int fi_idx = j / NR, ri = j % NR;
            for (int s = 0; s < finfo.n_used; s++) {
                if (finfo.idx[s] == fi_idx) {
                    if (i == 0 && ri == 1)      /* e <- sK */
                        g_val = finfo.w[s] * finfo.sig[s][1];
                    else if (i == 1 && ri == 0) /* t <- sH */
                        g_val = finfo.w[s] * finfo.sig[s][0] * (-exiH2);
                    break;
                }
            }
            dF[i*nn + j] = sum + g_val;
        }
    }
}

static double v0_event(double xi, const double *y)
{
    (void)xi;
    return g_emax - y[0];
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Post-processing: extract sigma from response matrix F
 * ══════════════════════════════════════════════════════════════════════════*/
static void extract_sigma(const double *F, int ns, int nn, double *sig)
{
    for (int i = 0; i < ns; i++) {
        double sum = 0.0;
        for (int j = 0; j < nn; j++) {
            double v = F[i*nn + j];
            sum += v*v;
        }
        sig[i] = sqrt(fmax(sum, 0.0));
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Data output
 * ══════════════════════════════════════════════════════════════════════════*/
static void write_full(const Sol *sol, const char *fn,
                       double q, double e0, double Vx0, double Vy0,
                       double w0, double xi0, double xi1, const char *chm)
{
    FILE *f = fopen(fn, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", fn); return; }
    fprintf(f, "# evolve: q=%g e0=%g Vx0=%g Vy0=%g varpi0=%g xi=[%g,%g] chandrasekhar=%s\n",
            q, e0, Vx0, Vy0, w0, xi0, xi1, chm);
    fprintf(f, "xi a_over_ah e sig_e Vx sig_Vx Vy sig_Vy varpi sig_varpi t sig_t x sig_x y sig_y\n");
    int nn = g_nnoise, ns_full = NS;
    double sig[NS];
    for (int k = 0; k < sol->n; k++) {
        double xi = sol->xi[k];
        const double *yy = sol->y + k * sol->ns;
        extract_sigma(yy + NS, ns_full, nn, sig);
        fprintf(f, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e "
                   "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                xi, exp(-xi),
                yy[0], sig[0],
                yy[1], sig[1],
                yy[2], sig[2],
                yy[3], sig[3],
                yy[4], sig[4],
                yy[5], sig[5],
                yy[6], sig[6]);
    }
    fclose(f);
}

static void write_v0(const Sol *sol, const char *fn,
                     double q, double e0, double xi0, double xi1)
{
    FILE *f = fopen(fn, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", fn); return; }
    fprintf(f, "# evolve V=0 ref: q=%g e0=%g xi=[%g,%g]\n", q, e0, xi0, xi1);
    fprintf(f, "xi a_over_ah e sig_e t sig_t\n");
    int nn = g_nnoise;
    double sig[2];
    for (int k = 0; k < sol->n; k++) {
        double xi = sol->xi[k];
        const double *yy = sol->y + k * sol->ns;
        extract_sigma(yy + 2, 2, nn, sig);
        fprintf(f, "%.15e %.15e %.15e %.15e %.15e %.15e\n",
                xi, exp(-xi), yy[0], sig[0], yy[1], sig[1]);
    }
    fclose(f);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  CLI and main
 * ══════════════════════════════════════════════════════════════════════════*/
static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --q VALUE          Mass ratio (default 1.0)\n"
        "  --e0 VALUE         Initial eccentricity (default 0.5)\n"
        "  --Vx0 VALUE        Initial Vx/sigma (default 0)\n"
        "  --Vy0 VALUE        Initial Vy/sigma (default 0)\n"
        "  --varpi0 VALUE     Initial varpi [rad] (default 0)\n"
        "  --xi-start VALUE   Initial xi = ln(a_h/a)\n"
        "  --a0 VALUE         Initial a/a_h (alternative to --xi-start)\n"
        "  --xi-end VALUE     Final xi (default 5)\n"
        "  --chandrasekhar MODE  integral|constant|none (default integral)\n"
        "  --data-dir PATH    Path to harmonics data\n"
        "  --output BASE      Output file basename (default 'evolution')\n"
        "  --freeze-e         Freeze eccentricity\n"
        "  --freeze-Vx        Freeze Vx\n"
        "  --freeze-Vy        Freeze Vy\n"
        "  --freeze-varpi     Freeze varpi\n", prog);
}

int main(int argc, char **argv)
{
    double q = 1.0, e0 = 0.5, Vx0 = 0.0, Vy0 = 0.0, w0 = 0.0;
    double xi_start = -1.0, a0_val = -1.0, xi_end = 5.0;
    const char *ch_mode_str = "integral";
    const char *data_dir = NULL;
    const char *output_base = "evolution";
    g_fr_e = g_fr_Vx = g_fr_Vy = g_fr_w = 0;

    static struct option lopts[] = {
        {"q",             required_argument, 0, 1},
        {"e0",            required_argument, 0, 2},
        {"Vx0",           required_argument, 0, 3},
        {"Vy0",           required_argument, 0, 4},
        {"varpi0",        required_argument, 0, 5},
        {"xi-start",      required_argument, 0, 6},
        {"a0",            required_argument, 0, 7},
        {"xi-end",        required_argument, 0, 8},
        {"chandrasekhar", required_argument, 0, 9},
        {"data-dir",      required_argument, 0, 10},
        {"output",        required_argument, 0, 11},
        {"freeze-e",      no_argument,       0, 12},
        {"freeze-Vx",     no_argument,       0, 13},
        {"freeze-Vy",     no_argument,       0, 14},
        {"freeze-varpi",  no_argument,       0, 15},
        {"help",          no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "h", lopts, NULL)) != -1) {
        switch (opt) {
        case 1:  q = atof(optarg); break;
        case 2:  e0 = atof(optarg); break;
        case 3:  Vx0 = atof(optarg); break;
        case 4:  Vy0 = atof(optarg); break;
        case 5:  w0  = atof(optarg); break;
        case 6:  xi_start = atof(optarg); break;
        case 7:  a0_val   = atof(optarg); break;
        case 8:  xi_end   = atof(optarg); break;
        case 9:  ch_mode_str = optarg; break;
        case 10: data_dir = optarg; break;
        case 11: output_base = optarg; break;
        case 12: g_fr_e  = 1; break;
        case 13: g_fr_Vx = 1; break;
        case 14: g_fr_Vy = 1; break;
        case 15: g_fr_w  = 1; break;
        case 'h': default: usage(argv[0]); return opt == 'h' ? 0 : 1;
        }
    }

    if (a0_val > 0 && xi_start >= 0) {
        fprintf(stderr, "Use --a0 or --xi-start, not both\n"); return 1;
    }
    if (a0_val > 0) xi_start = -log(a0_val);
    if (xi_start < 0) xi_start = 0.0;

    if (strcmp(ch_mode_str, "integral") == 0) g_chmode = 1;
    else if (strcmp(ch_mode_str, "constant") == 0) g_chmode = 2;
    else if (strcmp(ch_mode_str, "none") == 0) g_chmode = 0;
    else { fprintf(stderr, "Unknown chandrasekhar mode: %s\n", ch_mode_str); return 1; }

    /* Default data directory: relative to this source file's location */
    char default_dd[2048];
    if (!data_dir) {
        /* try to find from argv[0] */
        const char *slash = strrchr(argv[0], '/');
        if (slash) {
            int len = (int)(slash - argv[0]);
            snprintf(default_dd, sizeof(default_dd),
                     "%.*s/../Data/results-precession-3D-velocity-soft", len, argv[0]);
        } else {
            snprintf(default_dd, sizeof(default_dd),
                     "../Data/results-precession-3D-velocity-soft");
        }
        data_dir = default_dd;
    }

    g_q = q;
    g_roah = 4.0*(1+q)*(1+q)/q;
    g_nst = 4;

    printf("Binary evolution: q=%g, e0=%g, V0/sigma=(%g, %g), varpi0=%g\n",
           q, e0, Vx0, Vy0, w0);
    printf("xi in [%.4f, %g]  (a/a_h: %.4f -> %.6f), Chandrasekhar: %s\n",
           xi_start, xi_end, exp(-xi_start), exp(-xi_end), ch_mode_str);

    /* Load data */
    if (load_dataset(q, data_dir) != 0) return 1;

    g_nnoise = NR * g_nep;
    printf("  Response matrix: %d files x %d rates = %d noise sources\n",
           g_nep, NR, g_nnoise);

    g_emax = g_ev[g_nf-1] - 0.01;

    /* ── Full solution ── */
    g_ntot = NS + NS * g_nnoise;
    printf("  Full state: %d + %d*%d = %d elements\n", NS, NS, g_nnoise, g_ntot);
    printf("Integrating (full)...\n");

    double *y0_full = (double*)calloc(g_ntot, sizeof(double));
    y0_full[0] = e0; y0_full[1] = Vx0; y0_full[2] = Vy0; y0_full[3] = w0;

    g_rhs_count = 0;
    Sol sol_full;
    rk45_solve(full_rhs, xi_start, xi_end, y0_full, g_ntot,
               1e-8, 1e-10, 0.5, NULL, full_event, &sol_full);

    {
        double el = (double)clock()/CLOCKS_PER_SEC - g_rhs_t0;
        printf("\r  Done: %d evals in %.1fs                              \n",
               g_rhs_count, el);
    }

    /* Print summary */
    {
        const double *y_end = sol_full.y + (sol_full.n-1)*sol_full.ns;
        double sig[NS]; extract_sigma(y_end+NS, NS, g_nnoise, sig);
        printf("Full solution -- %d steps, %s\n", sol_full.n,
               sol_full.term == 1 ? "terminated (e -> grid boundary)" :
               sol_full.term == 2 ? "terminated (stiffness/max evals)" : "completed");
        printf("  xi:        %.2f -> %.2f  (a/a_h: %.3f -> %.5f)\n",
               sol_full.xi[0], sol_full.xi[sol_full.n-1],
               exp(-sol_full.xi[0]), exp(-sol_full.xi[sol_full.n-1]));
        printf("  e:         %.4f -> %.4f  (+/-%.4f)\n", e0, y_end[0], sig[0]);
        printf("  |V|/sigma: %.4f -> %.4f  (+/-%.4f)\n",
               hypot(Vx0,Vy0), hypot(y_end[1],y_end[2]),
               hypot(sig[1],sig[2]));
        printf("  varpi:     %.4f -> %.4f rad  (+/-%.4f)\n", w0, y_end[3], sig[3]);
        printf("  t/T_hard:  %.4f -> %.4f  (+/-%.4f)\n", 0.0, y_end[4], sig[4]);
    }

    /* ── V=0 reference ── */
    int ntot_v0 = 2 + 2 * g_nnoise;
    printf("Integrating (V=0 reference)...\n");
    printf("  (V=0 ref) Response matrix: %d files x %d rates = %d noise sources "
           "(%d total state)\n", g_nep, NR, g_nnoise, ntot_v0);

    double *y0_v0 = (double*)calloc(ntot_v0, sizeof(double));
    y0_v0[0] = e0;

    g_v0_rhs_count = 0;
    Sol sol_v0;
    rk45_solve(v0_rhs, xi_start, xi_end, y0_v0, ntot_v0,
               1e-8, 1e-10, 0.5, NULL, v0_event, &sol_v0);

    {
        double el = (double)clock()/CLOCKS_PER_SEC - g_v0_rhs_t0;
        printf("\r  (V=0 ref) Done: %d evals in %.1fs                   \n",
               g_v0_rhs_count, el);
        printf("V=0 reference -- %d steps, %s\n", sol_v0.n,
               sol_v0.term ? "terminated" : "completed");
    }

    /* ── Write output ── */
    char fn_full[2048], fn_v0[2048];
    snprintf(fn_full, sizeof(fn_full), "%s_full.dat", output_base);
    snprintf(fn_v0,   sizeof(fn_v0),   "%s_V0.dat",   output_base);

    write_full(&sol_full, fn_full, q, e0, Vx0, Vy0, w0, xi_start, xi_end, ch_mode_str);
    write_v0(&sol_v0, fn_v0, q, e0, xi_start, xi_end);
    printf("\nOutput written to %s and %s\n", fn_full, fn_v0);

    /* cleanup */
    free(y0_full); free(y0_v0);
    sol_free(&sol_full); sol_free(&sol_v0);

    return 0;
}
