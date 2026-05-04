#!/usr/bin/env python3
"""
sim_uniform_medium_c.py
=======================
Drop-in replacement for sim_uniform_medium.py that offloads the simulation
loop to a compiled C shared library (nbody_core) via ctypes.

Same Params class, same CLI, same CSV/JSON output format.
All random numbers are generated in Python so the RNG sequence is identical.

Prerequisites
-------------
Compile the C shared library (requires a C compiler, e.g. gcc or clang):

    cd new_trial_Nbody
    make            # builds nbody_core.dylib (macOS) or nbody_core.so (Linux)

To enable OpenMP parallelism (faster, but non-deterministic force reductions):

    make omp

Usage
-----
Run with default parameters (same defaults as sim_uniform_medium.py):

    python sim_uniform_medium_c.py

Override parameters via CLI flags:

    python sim_uniform_medium_c.py --N 30000 --n_steps 40000 --seed 40
    python sim_uniform_medium_c.py --q 0.3 --e0 0.5 --N 10000 --n_steps 20000

All available flags (same as the original script):

    --q             mass ratio m2/m1        (default 0.2)
    --e0            initial eccentricity    (default 0.6)
    --a0_over_ah    initial a/a_h           (default 0.6)
    --N             number of background stars (default 30000)
    --L             box side length         (default 30.0)
    --n_steps       integration steps       (default 40000)
    --dt            timestep                (default 0.005)
    --seed          RNG seed                (default 40)
    --output_dir    output directory        (default "Data")
    --n_runs        number of parallel runs with consecutive seeds (default 1)

Output is written to <output_dir>/<label>.csv and <label>.json, identical
in format to sim_uniform_medium.py.

Ensemble runs
-------------
Run multiple seeds in parallel and produce an ensemble-averaged output:

    python sim_uniform_medium_c.py --n_runs 10 --seed 40 --N 30000

This launches 10 processes (seeds 40..49), writes each individual CSV/JSON,
then writes ensemble_mean.csv, ensemble_std.csv, and ensemble.json.
The individual CSVs are compatible with plot_analysis.py --ensemble.

Validation
----------
To check that the C backend reproduces the Python results:

    python validate.py --N 1000 --n_steps 2000
"""

from __future__ import annotations
import ctypes, time, math, os, json, sys, platform
import numpy as np
from numpy.ctypeslib import ndpointer
from dataclasses import dataclass

# ========================== Parameters ==========================
# (duplicated from sim_uniform_medium.py to avoid modifying the original)

@dataclass
class Params:
    q: float = 1.0
    e0: float = 0.6
    a0_over_ah: float = 0.6
    N: int = 30000
    L: float = 30.0
    dt: float = 0.005
    n_steps: int = 40000
    softening: float = 0.005
    r_sink: float = 0.0005
    v_eject_cut: float = -1.0   # velocity ejection cutoff (in units of sigma); <=0 disables
    r_eject_cut: float = 5.0   # protection radius for ejection (in units of current semi-major axis)
    replenish: bool = True
    output_every: int = 40
    output_dir: str = "Data"
    run_label: str = ""
    seed: int = 20
    m_star_override: float = -1.0  # if > 0, use this value; otherwise auto
    G: float = 1.0
    M: float = 1.0
    a0: float = 1.0

    @property
    def m1(self):
        return self.M / (1.0 + self.q)

    @property
    def m2(self):
        return self.M * self.q / (1.0 + self.q)

    @property
    def mu(self):
        return self.m1 * self.m2 / self.M

    @property
    def a_h(self):
        return self.a0 / self.a0_over_ah

    @property
    def sigma(self):
        return math.sqrt(self.G * self.mu / (4.0 * self.a_h))

    @property
    def rho(self):
        return self.N * self.m_star / self.L**3

    @property
    def m_star(self):
        if self.m_star_override > 0:
            return self.m_star_override
        H_target = 15.0
        T_hard_target = 1000.0 * self.T_orb
        rho_target = self.sigma / (self.G * self.a0 * H_target * T_hard_target)
        return 5.16871e-05 #rho_target * self.L**3 / self.N

    @property
    def T_orb(self):
        return 2.0 * math.pi * math.sqrt(self.a0**3 / (self.G * self.M))

    @property
    def T_hard_est(self):
        H = 15.0
        return self.sigma / (self.G * self.rho * self.a0 * H)

    @property
    def label(self):
        if self.run_label:
            return self.run_label
        return f"q{self.q}_e{self.e0}_aah{self.a0_over_ah}_N{self.N}_s{self.seed}"

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Binary BH in uniform medium — N-body (C backend)",
            "=" * 60,
            f"  q = {self.q},  e0 = {self.e0},  a0/a_h = {self.a0_over_ah}",
            f"  a_h = {self.a_h:.4f},  sigma = {self.sigma:.5f}",
            f"  N = {self.N},  L = {self.L},  m_star = {self.m_star:.5e}",
            f"  rho = {self.rho:.5e},  m_star/mu = {self.m_star/self.mu:.5e},  m_star/m2 = {self.m_star/self.m2:.5e}",
            f"  T_orb = {self.T_orb:.4f},  T_hard ~ {self.T_hard_est:.1f}",
            f"  dt = {self.dt},  n_steps = {self.n_steps},  t_end = {self.n_steps*self.dt:.1f}",
            f"  t_end / T_hard ~ {self.n_steps*self.dt/self.T_hard_est:.2f}",
            f"  softening = {self.softening},  r_sink = {self.r_sink}",
            f"  v_eject_cut = {self.v_eject_cut} sigma  ({'disabled' if self.v_eject_cut <= 0 else f'{self.v_eject_cut * self.sigma:.5f} code units'})",
            f"  r_eject_cut = {self.r_eject_cut} a(t)  (protection radius for velocity ejection)",
            f"  seed = {self.seed},  label = {self.label}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ========================== Initial Conditions ==========================

def init_binary(p: Params):
    m1, m2, M = p.m1, p.m2, p.M
    a, e, G = p.a0, p.e0, p.G
    r_peri = a * (1.0 - e)
    v_peri = math.sqrt(G * M / a * (1.0 + e) / (1.0 - e))
    r1 = np.array([-m2 / M * r_peri, 0.0, 0.0])
    r2 = np.array([+m1 / M * r_peri, 0.0, 0.0])
    v1 = np.array([0.0, -m2 / M * v_peri, 0.0])
    v2 = np.array([0.0, +m1 / M * v_peri, 0.0])
    return np.array([r1, r2]), np.array([v1, v2])


def init_stars(p: Params, rng: np.random.Generator):
    pos = rng.uniform(-p.L / 2, p.L / 2, size=(p.N, 3))
    vel = rng.normal(0.0, p.sigma, size=(p.N, 3))
    return pos, vel


# ========================== C library loader ==========================

def _load_lib():
    """Load the compiled shared library from the same directory as this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    if platform.system() == "Darwin":
        name = "nbody_core.dylib"
    else:
        name = "nbody_core.so"
    path = os.path.join(here, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"C library not found at {path}.  Run 'make' first.")
    lib = ctypes.CDLL(path)

    dbl_p = ctypes.POINTER(ctypes.c_double)
    int_p = ctypes.POINTER(ctypes.c_int)

    lib.run_simulation.restype = ctypes.c_int
    lib.run_simulation.argtypes = [
        ctypes.c_double, ctypes.c_double,  # G, M
        ctypes.c_double, ctypes.c_double,  # m1, m2
        ctypes.c_double,                   # m_star
        ctypes.c_double, ctypes.c_double,  # L, dt
        ctypes.c_double, ctypes.c_double,  # softening, r_sink
        ctypes.c_double, ctypes.c_double,  # v_eject_cut, r_eject_factor
        ctypes.c_int, ctypes.c_int,        # N, n_steps
        ctypes.c_int, ctypes.c_int,        # output_every, replenish
        dbl_p, dbl_p, dbl_p,               # bh_pos, bh_vel, bh_mass
        dbl_p, dbl_p,                      # star_pos, star_vel
        dbl_p, dbl_p, ctypes.c_int,        # rand_pos_buf, rand_vel_buf, rand_buf_len
        dbl_p, int_p, int_p,               # output, n_rows, rand_consumed
    ]
    return lib


# ========================== Run ==========================

def run(p: Params) -> str:
    print(p.summary())
    lib = _load_lib()

    rng = np.random.default_rng(p.seed)
    bh_pos, bh_vel = init_binary(p)
    bh_mass = np.array([p.m1, p.m2])
    star_pos, star_vel = init_stars(p, rng)

    # Pre-generate replenishment random numbers.
    # With velocity ejection enabled, many more replacements are needed.
    # Scale with both n_steps and N (more particles → more ejections per step).
    if p.v_eject_cut > 0:
        max_replenish = max(500000, p.n_steps * 200)
    else:
        max_replenish = max(10000, p.n_steps * 2)
    rand_pos_buf = rng.uniform(-p.L / 2, p.L / 2, size=(max_replenish, 3))
    rand_vel_buf = rng.normal(0.0, p.sigma, size=(max_replenish, 3))

    # Ensure contiguous C-order double arrays
    bh_pos    = np.ascontiguousarray(bh_pos, dtype=np.float64)
    bh_vel    = np.ascontiguousarray(bh_vel, dtype=np.float64)
    bh_mass   = np.ascontiguousarray(bh_mass, dtype=np.float64)
    star_pos  = np.ascontiguousarray(star_pos, dtype=np.float64)
    star_vel  = np.ascontiguousarray(star_vel, dtype=np.float64)
    rand_pos_buf = np.ascontiguousarray(rand_pos_buf, dtype=np.float64)
    rand_vel_buf = np.ascontiguousarray(rand_vel_buf, dtype=np.float64)

    max_rows = p.n_steps // p.output_every + 2
    output   = np.zeros(max_rows * 18, dtype=np.float64)
    n_rows   = ctypes.c_int(0)
    rand_consumed = ctypes.c_int(0)

    # Convert v_eject_cut from units of sigma to code units; <=0 disables
    v_eject_cut_code = p.v_eject_cut * p.sigma if p.v_eject_cut > 0 else -1.0

    dbl_p = ctypes.POINTER(ctypes.c_double)

    t_wall = time.time()
    ret = lib.run_simulation(
        p.G, p.M, p.m1, p.m2, p.m_star,
        p.L, p.dt, p.softening, p.r_sink,
        v_eject_cut_code, p.r_eject_cut,
        p.N, p.n_steps, p.output_every, int(p.replenish),
        bh_pos.ctypes.data_as(dbl_p),
        bh_vel.ctypes.data_as(dbl_p),
        bh_mass.ctypes.data_as(dbl_p),
        star_pos.ctypes.data_as(dbl_p),
        star_vel.ctypes.data_as(dbl_p),
        rand_pos_buf.ctypes.data_as(dbl_p),
        rand_vel_buf.ctypes.data_as(dbl_p),
        max_replenish,
        output.ctypes.data_as(dbl_p),
        ctypes.byref(n_rows),
        ctypes.byref(rand_consumed),
    )
    wall = time.time() - t_wall

    if ret != 0:
        print(f"ERROR: run_simulation returned {ret}", file=sys.stderr)
        if ret == -1:
            print("Random buffer exhausted — increase max_replenish.",
                  file=sys.stderr)
        return ""

    nr = n_rows.value
    data = output[:nr * 18].reshape(nr, 18)

    # Save CSV (same format as original + velocity ejection count + eccentricity vector)
    os.makedirs(p.output_dir, exist_ok=True)
    csv_path = os.path.join(p.output_dir, f"{p.label}.csv")
    header = "t,cm_x,cm_y,cm_z,cm_vx,cm_vy,cm_vz,a,e,N_stars,Fx,Fy,Fz,n_removed_total,n_vejected_total,evec_x,evec_y,evec_z"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    # Save JSON metadata
    meta = {
        "q": p.q, "e0": p.e0, "a0_over_ah": p.a0_over_ah,
        "a_h": p.a_h, "sigma": p.sigma, "rho": p.rho,
        "m_star": p.m_star, "m_star_over_mu": p.m_star / p.mu,
        "N": p.N, "L": p.L, "dt": p.dt, "n_steps": p.n_steps,
        "softening": p.softening, "r_sink": p.r_sink,
        "v_eject_cut": p.v_eject_cut, "r_eject_cut": p.r_eject_cut,
        "seed": p.seed, "T_orb": p.T_orb, "T_hard_est": p.T_hard_est,
    }
    json_path = os.path.join(p.output_dir, f"{p.label}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {csv_path}  ({nr} rows)")
    print(f"Wrote {json_path}")
    print(f"Rand buffer used: {rand_consumed.value}/{max_replenish}")
    print(f"Total wall time: {wall:.1f} s")
    return csv_path


# ========================== Ensemble ==========================

def _run_single(args):
    """Worker callable for ProcessPoolExecutor (must be module-level)."""
    seed, params_dict = args
    p = Params(**params_dict)
    p.seed = seed
    return run(p)


def run_ensemble(p: Params, n_runs: int) -> str:
    """Run *n_runs* simulations in parallel with seeds p.seed .. p.seed+n_runs-1,
    then write ensemble mean/std CSVs and metadata JSON."""
    from concurrent.futures import ProcessPoolExecutor

    seeds = [p.seed + i for i in range(n_runs)]
    params_dict = {f.name: getattr(p, f.name)
                   for f in p.__dataclass_fields__.values()}

    print(f"Launching {n_runs} parallel runs  (seeds {seeds[0]}..{seeds[-1]})")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_runs) as pool:
        csv_paths = list(pool.map(_run_single,
                                  [(s, params_dict) for s in seeds]))

    wall = time.time() - t0
    failed = [p for p in csv_paths if not p]
    if failed:
        print(f"WARNING: {len(failed)} run(s) failed.", file=sys.stderr)
    csv_paths = [p for p in csv_paths if p]
    if not csv_paths:
        print("ERROR: all runs failed.", file=sys.stderr)
        return ""

    # Load all CSVs and compute ensemble statistics
    datasets = []
    for cp in csv_paths:
        raw = np.genfromtxt(cp, delimiter=",", names=True)
        datasets.append(raw)

    col_names = datasets[0].dtype.names
    n_t = min(len(d) for d in datasets)
    stack = np.column_stack  # alias

    label_base = f"q{p.q}_e{p.e0}_aah{p.a0_over_ah}_N{p.N}"

    mean_cols = []
    std_cols = []
    for col in col_names:
        vals = np.array([d[col][:n_t] for d in datasets])
        mean_cols.append(np.mean(vals, axis=0))
        std_cols.append(np.std(vals, axis=0))

    mean_data = np.column_stack(mean_cols)
    std_data = np.column_stack(std_cols)

    header = ",".join(col_names)
    os.makedirs(p.output_dir, exist_ok=True)

    mean_path = os.path.join(p.output_dir, f"{label_base}_ensemble_mean.csv")
    np.savetxt(mean_path, mean_data, delimiter=",", header=header, comments="")

    std_path = os.path.join(p.output_dir, f"{label_base}_ensemble_std.csv")
    np.savetxt(std_path, std_data, delimiter=",", header=header, comments="")

    meta = {
        "q": p.q, "e0": p.e0, "a0_over_ah": p.a0_over_ah,
        "a_h": p.a_h, "sigma": p.sigma, "rho": p.rho,
        "m_star": p.m_star, "m_star_over_mu": p.m_star / p.mu,
        "N": p.N, "L": p.L, "dt": p.dt, "n_steps": p.n_steps,
        "softening": p.softening, "r_sink": p.r_sink,
        "T_orb": p.T_orb, "T_hard_est": p.T_hard_est,
        "n_runs": n_runs,
        "seeds": seeds,
        "ensemble_label": label_base,
    }
    json_path = os.path.join(p.output_dir, f"{label_base}_ensemble_mean.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Ensemble complete: {len(csv_paths)} runs, {n_t} time rows")
    print(f"  {mean_path}")
    print(f"  {std_path}")
    print(f"  {json_path}")
    print(f"Total wall time (all runs): {wall:.1f} s")
    print(f"{'='*60}")
    return mean_path


# ========================== CLI ==========================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Binary BH in uniform medium (C backend)")
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--e0", type=float, default=0.6)
    parser.add_argument("--a0_over_ah", type=float, default=0.6)
    parser.add_argument("--N", type=int, default=30000)
    parser.add_argument("--L", type=float, default=50.0)
    parser.add_argument("--n_steps", type=int, default=40000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--v_eject_cut", type=float, default=3.0,
                        help="Eject stars with |v-v_cm| > v_eject_cut*sigma (<=0 disables)")
    parser.add_argument("--r_eject_cut", type=float, default=5.0,
                        help="Protection radius for velocity ejection, in units of a(t) (default: 5.0)")
    parser.add_argument("--m_star", type=float, default=-1.0,
                        help="Star particle mass (>0 to override auto; default: auto)")
    parser.add_argument("--output_dir", type=str, default="Data")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of parallel runs with consecutive seeds")
    args = parser.parse_args()

    p = Params(
        q=args.q, e0=args.e0, a0_over_ah=args.a0_over_ah,
        N=args.N, L=args.L, n_steps=args.n_steps, dt=args.dt,
        seed=args.seed, output_dir=args.output_dir,
        v_eject_cut=args.v_eject_cut, r_eject_cut=args.r_eject_cut,
        m_star_override=args.m_star,
    )
    if args.n_runs > 1:
        run_ensemble(p, args.n_runs)
    else:
        run(p)
