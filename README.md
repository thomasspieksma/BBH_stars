# BBH_stars

Code and data for the paper

> **Self-acceleration of Hardening Binaries**
> Giovanni Maria Tomaselli & Thomas F. M. Spieksma
> arXiv:[2605.00976](https://arxiv.org/abs/2605.00976) · INSPIRE:[3151176](https://inspirehep.net/literature/3151176)

A Keplerian binary embedded in a bath of lighter particles is traditionally
described by two evolution parameters — the hardening rate $H$ and the
eccentricity-growth rate $K$. Combining symmetry arguments with extensive
three-body scattering experiments, the paper shows that the secular dynamics
also includes an acceleration parameter $\vec P$, an apsidal-precession
parameter $Q$, and a rotation parameter $\vec R$. These drive the binary's
centre of mass along an outward spiral, with implications for the final-parsec
problem and gravitational-wave source populations.

This repository contains the three-body scattering integrator, the
binary-evolution solver, the N-body cluster simulation, and the corresponding
output data.

---

## Repository layout

```
3-body/
  3-body-code/         main.cpp        — Monte-Carlo three-body scattering integrator (C++/OpenMP)
  binary-evolution/    evolve.c        — Binary evolution ODE solver (C)
                       run_evolve.py   — Python wrapper + plotting
  analysis-scripts/    weight-Maxwellian.py             — V=0 reweighting + plots
                       weight-Maxwellian-3D-velocity.py — V≠0 reweighting + plots
  3-body-data/         convergence-tests/        — convergence study (Fig. 12)
                       data-V=0/                 — main isotropic dataset (Figs. 2-4)
                       data-V=0-varying-Tmax/    — Tmax-resolved scans (Figs. 7-8)   ⚠ ~2.4 GB
                       data-Vneq0/               — anisotropic SH-moment files (LFS)  ⚠ ~3.5 GB
                       stopping-condition-3/     — Criterion to compare against Bonetti et al. (2020)
N-body/
  nbody_core.{c,h}     — direct N-body force/integrator (C shared library)
  Makefile
  sim_uniform_medium_c.py  — Python driver (ctypes) for the cluster simulation
download_data.sh       — selectively fetch datasets after cloning (see below)
```

---

## ⚠ Data size and partial download

A full clone with all data is **~6 GB**. The two large datasets are both stored
in Git LFS:

| Path                                            | Size    |
|-------------------------------------------------|---------|
| `3-body/3-body-data/data-V=0-varying-Tmax/`     | ~2.4 GB |
| `3-body/3-body-data/data-Vneq0/`                | ~3.5 GB |

Everything else is **<100 MB**. You almost certainly do not need both large
datasets to reproduce a given figure. Pick one of the workflows below.

### Recommended: minimal clone

Skip LFS downloads at clone time, then opt in to what you need:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
    https://github.com/thomasspieksma/BBH_stars.git
cd BBH_stars
./download_data.sh           # interactive menu
```

`download_data.sh` has a few preset modes (also accepted as the first argument):

| Mode        | Includes                                                     | Approx. size |
|-------------|--------------------------------------------------------------|--------------|
| `code-only` | scripts only                                                 | <1 MB        |
| `small`     | code + small data (`convergence-tests`, `data-V=0`, …)       | ~95 MB       |
| `V0`        | `small` + `data-V=0-varying-Tmax` (LFS)                      | ~2.5 GB      |
| `Vneq0`     | `small` + all LFS files in `data-Vneq0`                      | ~3.6 GB      |
| `all`       | everything                                                   | ~6 GB        |

You can also pull a subset of the LFS files by glob:

```bash
./download_data.sh Vneq0 'q=0.001_*.bin'
git lfs pull --include='3-body/3-body-data/data-Vneq0/q=0.2_e=0.6.bin'
```

A normal `git clone` (without `GIT_LFS_SKIP_SMUDGE=1`) auto-fetches all
~5.9 GB of LFS data if Git LFS is installed — ~7 GB total on disk.

---

## Building

```bash
# 3-body scattering integrator (OpenMP)
cd 3-body/3-body-code
g++ -O3 -fopenmp -std=c++17 main.cpp -o scattering
# macOS Homebrew clang: CXX=$(brew --prefix llvm)/bin/clang++

# Binary evolution solver
cd ../binary-evolution
make

# N-body cluster code
cd ../../N-body
make            # add `omp` for OpenMP build
```

---

## Running the scripts

### 1. Three-body scattering experiments → produces the raw `.txt`/`.bin` data

`3-body/3-body-code/main.cpp` runs the Monte-Carlo scattering of test particles
off a Keplerian binary and writes one row per incoming-velocity bin.

```bash
./scattering <q> <e>                        # default build
./scattering <q> <e> [N_v] [N] [l_max]      # with ENABLE_HARMONIC_OUTPUT=true
```

- `q`: mass ratio $m_2/m_1 \in (0,1]$
- `e`: binary eccentricity $\in [0,1)$

There are two compile-time toggles at the top of `main.cpp`:
  - `ENABLE_BONETTI_CONDITION_3` — adds the additional stopping criterion from Rasskazov et al. (2019)/Bonetti et al. (2020)
  - `ENABLE_HARMONIC_OUTPUT` — emits per-particle and/or spherical-harmonic-moment binary files (used for $V \neq 0$ analysis)

Output files are named `q=<q>_e=<e>_Tcut=<Tmax>.txt`,
`particles_q=<q>_e=<e>.bin`, `harmonics_q=<q>_e=<e>.bin`. Format details are
documented in the header comment of `main.cpp`.

### 2. Evolution-parameter plots ($H$, $K$, $\vec P$, $Q$, $\vec R$)

These plots come from reweighting the scattering output against a Maxwellian
velocity distribution.

**Isotropic ($\vec V = 0$):** Figures 2–3 of the paper.
```bash
cd 3-body/analysis-scripts
python weight-Maxwellian.py         # reads ../3-body-data/data-V=0/
```
Edit the `q`, `e`, `Tcut` constants at the top of the script to choose the
dataset.

**Anisotropic ($\vec V \neq 0$):** Figures 5–6 of the paper. Requires the
LFS-tracked harmonics binaries.
```bash
python weight-Maxwellian-3D-velocity.py \
    ../3-body-data/data-Vneq0/harmonics_q=0.2_e=0.6.bin
# optional consistency checks:
python weight-Maxwellian-3D-velocity.py harmonics.bin --check-iso  text-file.txt
python weight-Maxwellian-3D-velocity.py harmonics.bin --check-harmonics particles.bin
```

**Tmax dependence (convergence of $K$, Figs. 7–8):** uses
`data-V=0-varying-Tmax/`, processed by the same `weight-Maxwellian.py` after
pointing it at the desired Tcut.

### 3. Binary-evolution trajectories (Figs. 9–10)

```bash
cd 3-body/binary-evolution
./run_evolve.py --q 0.2 --e0 0.5 --Vx0 0.0 --Vy0 0.0
```

`run_evolve.py --help` lists every flag. The script invokes the compiled C
solver, reads back its `_full.dat` / `_V0.dat` outputs, and produces the
combined evolution panel as `evolution.pdf`. By default it expects
harmonics data at `3-body/Data/results-precession-3D-velocity-soft/`; pass
`--data-dir` to point at `3-body/3-body-data/data-Vneq0/` (or wherever you
fetched the LFS files).

### 4. N-body cluster simulation

```bash
cd N-body
python3 sim_uniform_medium_c.py --q 0.2 --e0 0.5 --a0_over_ah 0.6 --N 5000000 --L 38 --m_star 5.6e-7 --n_steps 240000 --dt 0.005 --seed 10 --n_runs 20 --v_eject_cut -1
```

Outputs `Data/<label>.csv` and `<label>.json`; see the script's docstring for
the full flag list.

### 5. Convergence tests (Fig. 12)

The data in `3-body-data/convergence-tests/` is produced by switching the
integrator inside `evolveParticle()` of `main.cpp` (`rk45Step`, `pihajokiStep`,
`bulirschStoerStep`, …) and rerunning at fixed $(q,e)$.

---

## Citation

```bibtex
@article{Tomaselli:2026uqg,
    author = "Tomaselli, Giovanni Maria and Spieksma, Thomas F. M.",
    title = "{Self-acceleration of Hardening Binaries}",
    eprint = "2605.00976",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.GA",
    month = "5",
    year = "2026"
}
```

## License

See [LICENSE](LICENSE).
