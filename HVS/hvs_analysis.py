#!/usr/bin/env python3
"""hvs_analysis_2.py — correlated ejection kernel + diagnostics for HVS scattering runs.

This is a drop-in, extended version of hvs_analysis_1.py.

What it does
------------
Given the binary output from your 3-body scattering code (per-ejected particle):
    (v_f, cos(theta), phi, delta_t)
this script
  (i) builds the *joint* (3D) ejection kernel
        d^3\Gamma / (dv_f dcos\theta d\phi)
      with Maxwellian + cross-section weighting over v_\infty bins;
  (ii) produces the paper-style diagnostics:
      - marginals: dP/d\tilde v_f, dP/dcos\theta, dP/d\phi
      - conditional angular maps in velocity slices
      - planarity-vs-velocity-cut curve
  (iii) adds visualization helpers:
      - corner/correlation plot for (\tilde v_f, cos\theta, \phi)
      - 3D unit-sphere scatter of ejection directions colored by \tilde v_f

Notes
-----
* Your current C++ code writes *only ejected* stars into the .bin file.
  Therefore any "ejection fraction" f_HVS relative to *all* encounters is not
  meaningful yet (you'd need total trial counts including non-ejections).
  This script focuses on the kernel shape/correlations.

Usage
-----
  python3 hvs_analysis_2.py hvs_q=0.1_e=0.6.bin --aah 0.01 0.1 1.0

Common options
--------------
  --aah ...           Use binary hardness a/a_h values (preferred)
  --sigma ...         Or directly specify sigma (code units)
  --tcut T            Filter out events with delta_t >= T
  --outdir DIR        Where to write outputs (default: alongside .bin)

Kernel output
-------------
  --save-kernel3d     Save the 3D kernel + edges to NPZ for reuse
  --v-bins 120        Number of v_f bins (log-spaced)
  --ct-bins 50        Number of cos(theta) bins (linear)
  --phi-bins 60       Number of phi bins (linear, [-pi,pi])

Extra plots
-----------
  --corner            Make a weighted corner/correlation plot
  --corner-vmin X     Only include events with v_f > X*sigma in corner plot

  --sphere            3D unit-sphere scatter of directions
  --sphere-vmin X     Only include events with v_f > X*sigma in sphere plot

  --no-show           Do not open interactive windows (still saves files)

Outputs
-------
Writes PDFs (and optionally NPZ) named after the input .bin file.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


# ---- numeric integration helper (NumPy renamed trapz→trapezoid) ----


def _trapz(y, x):
    """Version-safe trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# ═══════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════

def read_particles(filename: str):
    """Read binary per-particle HVS file."""
    with open(filename, "rb") as f:
        q = struct.unpack("d", f.read(8))[0]
        e = struct.unpack("d", f.read(8))[0]
        rp_max = struct.unpack("d", f.read(8))[0]
        r_sphere = struct.unpack("d", f.read(8))[0]
        n_v = struct.unpack("i", f.read(4))[0]
        N_per_v = struct.unpack("i", f.read(4))[0]

        bins = []
        for _ in range(n_v):
            v_inf = struct.unpack("d", f.read(8))[0]
            n_esc = struct.unpack("i", f.read(4))[0]
            if n_esc > 0:
                raw = np.frombuffer(
                    f.read(n_esc * 4 * 8), dtype=np.float64
                ).reshape(n_esc, 4)
                bins.append(
                    dict(
                        v_inf=v_inf,
                        n_esc=n_esc,
                        vf_inf=raw[:, 0].copy(),
                        cos_theta=raw[:, 1].copy(),
                        phi=raw[:, 2].copy(),
                        delta_t=raw[:, 3].copy(),
                    )
                )
            else:
                bins.append(
                    dict(
                        v_inf=v_inf,
                        n_esc=0,
                        vf_inf=np.empty(0),
                        cos_theta=np.empty(0),
                        phi=np.empty(0),
                        delta_t=np.empty(0),
                    )
                )

    meta = dict(q=q, e=e, rp_max=rp_max, r_sphere=r_sphere, N_per_v=N_per_v)
    return meta, bins


# ═══════════════════════════════════════════════════════════════
# Physics helpers  (code units: G = M = a = 1)
# ═══════════════════════════════════════════════════════════════

def maxwellian(v: np.ndarray, sigma: float) -> np.ndarray:
    """Maxwell-Boltzmann speed pdf, normalised to 1."""
    return np.sqrt(2 / np.pi) * v**2 / sigma**3 * np.exp(-v**2 / (2 * sigma**2))


def b_max(v: float, rp: float) -> float:
    return rp * np.sqrt(1 + 2 / (v**2 * rp))


def sigma_from_aah(aah: float, q: float) -> float:
    """sigma in code units for given a/a_h."""
    return np.sqrt(q * aah / (4 * (1 + q)))


# ═══════════════════════════════════════════════════════════════
# Maxwellian weighting
# ═══════════════════════════════════════════════════════════════

def _particle_weights(bins, sigma: float, meta, tcut: float | None = None):
    """Return (mask, per-particle weight) for each v_inf bin.

    weight per (retained) particle in a given v_inf bin:
        w = v_inf * pi * b_max(v_inf)^2 * f_MB(v_inf; sigma) * dv / N_per_v

    Particles with delta_t >= tcut are masked out.
    """

    N = meta["N_per_v"]
    rp = meta["rp_max"]
    v_arr = np.array([b["v_inf"] for b in bins], dtype=float)
    nv = len(v_arr)

    # dv for trapezoidal-ish rule on nonuniform grid
    dv = np.empty(nv)
    for i in range(nv):
        if nv == 1:
            dv[i] = v_arr[i]
        elif i == 0:
            dv[i] = v_arr[1] - v_arr[0]
        elif i == nv - 1:
            dv[i] = v_arr[-1] - v_arr[-2]
        else:
            dv[i] = 0.5 * (v_arr[i + 1] - v_arr[i - 1])

    result = []
    for i, bn in enumerate(bins):
        v = float(v_arr[i])
        bm = b_max(v, rp)
        w = v * np.pi * bm**2 * maxwellian(v, sigma) * dv[i] / N

        if bn["n_esc"] == 0:
            result.append((np.empty(0, dtype=bool), w))
        else:
            mask = np.ones(bn["n_esc"], dtype=bool)
            if tcut is not None:
                mask &= bn["delta_t"] < tcut
            result.append((mask, w))

    return result


def _gather(bins, fields: list[str], pw):
    """Concatenate multiple fields across bins, applying masks and weights."""
    cols = [[] for _ in fields]
    wts = []
    for i, bn in enumerate(bins):
        mask, w = pw[i]
        n = int(mask.sum())
        if n == 0:
            continue
        for j, fld in enumerate(fields):
            cols[j].append(bn[fld][mask])
        wts.append(np.full(n, w, dtype=float))

    if not wts:
        return [np.empty(0) for _ in fields], np.empty(0)

    cols = [np.concatenate(c) for c in cols]
    wts = np.concatenate(wts)
    return cols, wts


# ═══════════════════════════════════════════════════════════════
# Kernel construction
# ═══════════════════════════════════════════════════════════════

def _vf_edges_from_data(vf: np.ndarray, nbins: int):
    """Stable log edges for v_f, avoiding 0 and extreme tails."""
    if vf.size == 0:
        return np.logspace(-3, 1, nbins + 1)
    vf_pos = vf[vf > 0]
    lo = max(1e-8, np.percentile(vf_pos, 0.5))
    hi = np.percentile(vf, 99.8) * 1.8
    if not np.isfinite(hi) or hi <= lo:
        hi = lo * 10
    return np.logspace(np.log10(lo), np.log10(hi), nbins + 1)


def build_kernel3d(
    meta,
    bins,
    sigma: float,
    tcut: float | None = None,
    v_bins: int = 120,
    ct_bins: int = 50,
    phi_bins: int = 60,
):
    """Build the weighted 3D kernel d^3Γ/(dv dcosθ dφ).

    Returns
    -------
    v_edges, ct_edges, phi_edges, kernel
        kernel shape = (v_bins, ct_bins, phi_bins)
        Units: rate density (same arbitrary normalization as weights)
    """
    pw = _particle_weights(bins, sigma, meta, tcut)
    (vf, ct, ph), w = _gather(bins, ["vf_inf", "cos_theta", "phi"], pw)

    # wrap phi into [-pi, pi]
    if ph.size:
        ph = (ph + np.pi) % (2 * np.pi) - np.pi

    v_edges = _vf_edges_from_data(vf, v_bins)
    ct_edges = np.linspace(-1.0, 1.0, ct_bins + 1)
    phi_edges = np.linspace(-np.pi, np.pi, phi_bins + 1)

    if vf.size == 0:
        kernel = np.zeros((v_bins, ct_bins, phi_bins), dtype=float)
        return v_edges, ct_edges, phi_edges, kernel

    H, edges = np.histogramdd(
        sample=np.column_stack([vf, ct, ph]),
        bins=[v_edges, ct_edges, phi_edges],
        weights=w,
    )

    # Convert counts to density
    dv = np.diff(v_edges)[:, None, None]
    dct = np.diff(ct_edges)[None, :, None]
    dph = np.diff(phi_edges)[None, None, :]
    kernel = H / (dv * dct * dph)

    return v_edges, ct_edges, phi_edges, kernel


def marginal_from_kernel(kernel, v_edges, ct_edges, phi_edges):
    """Return marginals as *probability* densities by normalizing the kernel."""
    dv = np.diff(v_edges)
    dct = np.diff(ct_edges)
    dph = np.diff(phi_edges)

    # total rate (integral over all bins)
    total = np.sum(kernel * dv[:, None, None] * dct[None, :, None] * dph[None, None, :])
    if total <= 0:
        return None

    # dP/dv
    dP_dv = np.sum(kernel * dct[None, :, None] * dph[None, None, :], axis=(1, 2)) / total

    # dP/dcosθ
    dP_dct = np.sum(kernel * dv[:, None, None] * dph[None, None, :], axis=(0, 2)) / total

    # dP/dphi
    dP_dph = np.sum(kernel * dv[:, None, None] * dct[None, :, None], axis=(0, 1)) / total

    v_cent = np.sqrt(v_edges[:-1] * v_edges[1:])
    ct_cent = 0.5 * (ct_edges[:-1] + ct_edges[1:])
    ph_cent = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    return (v_cent, dP_dv), (ct_cent, dP_dct), (ph_cent, dP_dph)


# ═══════════════════════════════════════════════════════════════
# Diagnostics (direct-from-particles; kept for convenience)
# ═══════════════════════════════════════════════════════════════

def conditional_2d(meta, bins, sigma, vf_lo, vf_hi, tcut=None, nct=40, nphi=60):
    """Conditional angular density d^2P/(dcosθ dφ) in a vf slice."""
    pw = _particle_weights(bins, sigma, meta, tcut)
    ct_all, phi_all, w_all = [], [], []
    for i, bn in enumerate(bins):
        mask, w = pw[i]
        if mask.sum() == 0:
            continue
        sel = mask & (bn["vf_inf"] >= vf_lo) & (bn["vf_inf"] < vf_hi)
        n = int(sel.sum())
        if n > 0:
            ct_all.append(bn["cos_theta"][sel])
            phi_all.append(bn["phi"][sel])
            w_all.append(np.full(n, w))

    ct_e = np.linspace(-1, 1, nct + 1)
    phi_e = np.linspace(-np.pi, np.pi, nphi + 1)

    if not ct_all:
        return ct_e, phi_e, np.zeros((nct, nphi))

    ct_all = np.concatenate(ct_all)
    phi_all = np.concatenate(phi_all)
    phi_all = (phi_all + np.pi) % (2 * np.pi) - np.pi
    w_all = np.concatenate(w_all)

    h, _, _ = np.histogram2d(ct_all, phi_all, bins=[ct_e, phi_e], weights=w_all)

    area = np.diff(ct_e)[:, None] * np.diff(phi_e)[None, :]
    total = h.sum()
    if total > 0:
        h /= (total * area)  # integrates to 1
    return ct_e, phi_e, h


def planarity(meta, bins, sigma, vtilde_cuts, tcut=None):
    """Planarity P(vcut) = <|cosθ|>_{v>vcut}/0.5. P=1 isotropic; P<1 planar."""
    pw = _particle_weights(bins, sigma, meta, tcut)
    vtilde_cuts = np.atleast_1d(vtilde_cuts)
    result = np.full(len(vtilde_cuts), np.nan)
    for j, vc in enumerate(vtilde_cuts):
        vmin = vc * sigma
        num, denom = 0.0, 0.0
        for i, bn in enumerate(bins):
            mask, w = pw[i]
            if mask.sum() == 0:
                continue
            sel = mask & (bn["vf_inf"] > vmin)
            n = int(sel.sum())
            if n > 0:
                num += w * np.sum(np.abs(bn["cos_theta"][sel]))
                denom += w * n
        if denom > 0:
            result[j] = (num / denom) / 0.5
    return result


# ═══════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════

def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")


def make_corner(vtilde, ct, phi, w, outpath: Path, title: str = ""):
    """Simple weighted corner plot for (vtilde, cosθ, φ)."""
    # wrap phi
    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    # Layout: 3x3; hist on diagonal, scatter below diag
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    # turn off upper triangle
    for i in range(3):
        for j in range(3):
            if j > i:
                axes[i, j].axis("off")

    # 1D hists
    bins_v = np.logspace(np.log10(max(1e-4, np.percentile(vtilde[vtilde > 0], 1))),
                         np.log10(np.percentile(vtilde, 99.5) * 1.2), 60)
    axes[0, 0].hist(vtilde, bins=bins_v, weights=w, histtype="step")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel(r"$\tilde v_f$")

    bins_ct = np.linspace(-1, 1, 60)
    axes[1, 1].hist(ct, bins=bins_ct, weights=w, histtype="step")
    axes[1, 1].set_xlabel(r"$\cos\theta$")

    bins_ph = np.linspace(-np.pi, np.pi, 60)
    axes[2, 2].hist(phi, bins=bins_ph, weights=w, histtype="step")
    axes[2, 2].set_xlabel(r"$\phi$")

    # 2D scatters (subsample for speed)
    nmax = 50000
    if vtilde.size > nmax:
        rng = np.random.default_rng(0)
        idx = rng.choice(vtilde.size, size=nmax, replace=False, p=w / w.sum())
        v_s, ct_s, ph_s, w_s = vtilde[idx], ct[idx], phi[idx], w[idx]
    else:
        v_s, ct_s, ph_s, w_s = vtilde, ct, phi, w

    axes[1, 0].scatter(v_s, ct_s, s=2, c=w_s, alpha=0.4)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylabel(r"$\cos\theta$")

    axes[2, 0].scatter(v_s, ph_s, s=2, c=w_s, alpha=0.4)
    axes[2, 0].set_xscale("log")
    axes[2, 0].set_ylabel(r"$\phi$")
    axes[2, 0].set_xlabel(r"$\tilde v_f$")

    axes[2, 1].scatter(ct_s, ph_s, s=2, c=w_s, alpha=0.4)
    axes[2, 1].set_xlabel(r"$\cos\theta$")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    _savefig(fig, outpath)


def make_sphere_plot(vtilde, ct, phi, w, outpath: Path, title: str = ""):
    """3D unit-sphere scatter colored by vtilde."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    theta = np.arccos(np.clip(ct, -1, 1))

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = ct

    # subsample by weight to keep it readable
    nmax = 30000
    if vtilde.size > nmax:
        rng = np.random.default_rng(1)
        idx = rng.choice(vtilde.size, size=nmax, replace=False, p=w / w.sum())
        x, y, z, vtilde = x[idx], y[idx], z[idx], vtilde[idx]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x, y, z, c=vtilde, s=3, alpha=0.7)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\tilde v_f$")

    # unit sphere wireframe for context
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title)

    plt.tight_layout()
    _savefig(fig, outpath)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="HVS correlated ejection kernel analysis (v2)")
    p.add_argument("file", help="Binary particle file (.bin)")
    p.add_argument("--sigma", nargs="+", type=float, default=None,
                   help="Velocity dispersions (code units)")
    p.add_argument("--aah", nargs="+", type=float, default=None,
                   help="Binary hardness values a/a_h")
    p.add_argument("--tcut", type=float, default=None,
                   help="Max interaction time filter")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (default: alongside .bin)")

    p.add_argument("--save-kernel3d", action="store_true",
                   help="Save 3D kernel to NPZ")
    p.add_argument("--v-bins", type=int, default=120)
    p.add_argument("--ct-bins", type=int, default=50)
    p.add_argument("--phi-bins", type=int, default=60)

    p.add_argument("--corner", action="store_true", help="Make corner plot")
    p.add_argument("--corner-vmin", type=float, default=0.0,
                   help="Corner plot cut: include only v_f > corner_vmin*sigma")

    p.add_argument("--sphere", action="store_true", help="Make 3D sphere plot")
    p.add_argument("--sphere-vmin", type=float, default=0.0,
                   help="Sphere plot cut: include only v_f > sphere_vmin*sigma")

    p.add_argument("--no-show", action="store_true", help="Do not call plt.show()")

    args = p.parse_args()

    inpath = Path(args.file)
    outdir = Path(args.outdir) if args.outdir else inpath.parent
    stem = inpath.stem

    meta, bins = read_particles(str(inpath))
    q, ecc = meta["q"], meta["e"]
    print(f"Loaded: q={q:.4g}, e={ecc:.2f}, {len(bins)} v-bins, N={meta['N_per_v']}/bin")
    if args.tcut is not None:
        print(f"  T_cut filter: delta_t < {args.tcut:.6g}")

    # Determine sigma / a/a_h values
    if args.sigma is not None:
        sigmas = list(args.sigma)
        aahs = [4 * (1 + q) * s**2 / q for s in sigmas]
    elif args.aah is not None:
        aahs = list(args.aah)
        sigmas = [sigma_from_aah(x, q) for x in aahs]
    else:
        aahs = [0.01, 0.03, 0.1, 0.3, 1.0]
        sigmas = [sigma_from_aah(x, q) for x in aahs]

    print(f"  a/a_h = {[f'{a:.3g}' for a in aahs]}")

    # Use the middle hardness for the angle-correlation diagnostics
    sig_mid = sigmas[len(sigmas) // 2]
    aah_mid = aahs[len(aahs) // 2]

    # ── Build 3D kernel (at sig_mid) ──
    v_edges, ct_edges, phi_edges, kernel3d = build_kernel3d(
        meta, bins, sig_mid, tcut=args.tcut, v_bins=args.v_bins, ct_bins=args.ct_bins, phi_bins=args.phi_bins
    )

    if args.save_kernel3d:
        npz_path = outdir / f"{stem}_kernel3d_aah={aah_mid:.3g}.npz"
        np.savez_compressed(
            npz_path,
            q=q,
            e=ecc,
            sigma=sig_mid,
            aah=aah_mid,
            v_edges=v_edges,
            ct_edges=ct_edges,
            phi_edges=phi_edges,
            kernel3d=kernel3d,
        )
        print(f"Saved {npz_path}")

    # ── Figure 1: Marginals (from kernel, normalized to probability) ──
    marg = marginal_from_kernel(kernel3d, v_edges, ct_edges, phi_edges)
    fig1, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel (a): dP/dvtilde for several aah values (computed directly via kernels per sigma)
    ax = axes[0]
    cmap = colormaps["viridis"]
    colors = [cmap(i / (len(sigmas) - 1)) for i in range(len(sigmas))] if len(sigmas) > 1 else ["C0"]

    for sigma, aah, col in zip(sigmas, aahs, colors):
        ve, cte, phe, ker = build_kernel3d(
            meta, bins, sigma, tcut=args.tcut, v_bins=args.v_bins, ct_bins=args.ct_bins, phi_bins=args.phi_bins
        )
        m = marginal_from_kernel(ker, ve, cte, phe)
        if m is None:
            continue
        (vcent, dPdv), _, _ = m
        # convert to tilde-v density: dP/dvtilde = sigma * dP/dv
        ax.loglog(vcent / sigma, dPdv * sigma, color=col, label=f"$a/a_h={aah:.2g}$")

    ax.set_xlabel(r"$\tilde{v}_f = v_{f,\infty}/\sigma$")
    ax.set_ylabel(r"$dP/d\tilde{v}_f$")
    ax.set_title("(a) Velocity distribution")
    ax.set_ylim(1e-10,1e2)
    ax.legend(fontsize=7)

    # Panel (b,c): angular marginals at mid sigma, with velocity cuts
    # We'll compute these from particles for flexibility w/ cuts.

    ax = axes[1]
    for vf_fac, ls in [(0, "-"), (2, "--"), (4, ":"), (6, "-.")]:
        pw = _particle_weights(bins, sig_mid, meta, args.tcut)
        (vf, ct, ph), w = _gather(bins, ["vf_inf", "cos_theta", "phi"], pw)
        sel = vf > (vf_fac * sig_mid)
        if sel.sum() < 5:
            continue
        ct_sel, w_sel = ct[sel], w[sel]
        edges = np.linspace(-1, 1, 60 + 1)
        h, _ = np.histogram(ct_sel, bins=edges, weights=w_sel)
        dens = h / np.diff(edges)
        # normalize to probability
        norm = _trapz(dens, 0.5 * (edges[:-1] + edges[1:]))
        if norm > 0:
            dens /= norm
        lbl = "all" if vf_fac == 0 else rf"$v_f>{vf_fac}\sigma$"
        ax.plot(0.5 * (edges[:-1] + edges[1:]), dens, ls, label=lbl)

    ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="isotropic")
    ax.set_xlabel(r"$\cos\theta$")
    ax.set_ylabel(r"$dP/d\cos\theta$")
    ax.set_title(f"(b) Polar angle ($a/a_h={aah_mid:.2g}$)")
    ax.legend(fontsize=7)

    ax = axes[2]
    for vf_fac, ls in [(0, "-"), (2, "--"), (4, ":"), (6, "-.")]:
        pw = _particle_weights(bins, sig_mid, meta, args.tcut)
        (vf, ct, ph), w = _gather(bins, ["vf_inf", "cos_theta", "phi"], pw)
        sel = vf > (vf_fac * sig_mid)
        if sel.sum() < 5:
            continue
        ph_sel, w_sel = ph[sel], w[sel]
        ph_sel = (ph_sel + np.pi) % (2 * np.pi) - np.pi
        edges = np.linspace(-np.pi, np.pi, 60 + 1)
        h, _ = np.histogram(ph_sel, bins=edges, weights=w_sel)
        dens = h / np.diff(edges)
        # normalize
        norm = _trapz(dens, 0.5 * (edges[:-1] + edges[1:]))
        if norm > 0:
            dens /= norm
        lbl = "all" if vf_fac == 0 else rf"$v_f>{vf_fac}\sigma$"
        ax.plot(0.5 * (edges[:-1] + edges[1:]), dens, ls, label=lbl)

    ax.axhline(1 / (2 * np.pi), color="gray", ls=":", alpha=0.5, label="isotropic")
    ax.set_xlabel(r"$\phi$ (rad)")
    ax.set_ylabel(r"$dP/d\phi$")
    ax.set_title(f"(c) Azimuthal angle ($a/a_h={aah_mid:.2g}$)")
    ax.legend(fontsize=7)

    fig1.suptitle(f"$q={q:.4g}$, $e={ecc:.2f}$", fontsize=14)
    plt.tight_layout()
    _savefig(fig1, outdir / f"{stem}_marginals_v2.pdf")

    # ── Figure 2: Conditional 2D angular distributions in velocity slices ──
    slices = [(0, 1), (1, 2), (2, 4), (4, 10)]
    fig2, axes2 = plt.subplots(1, len(slices), figsize=(5 * len(slices), 4.5))
    for k, (vlo_s, vhi_s) in enumerate(slices):
        ax = axes2[k]
        ct_e, phi_e, kern = conditional_2d(meta, bins, sig_mid, vlo_s * sig_mid, vhi_s * sig_mid, tcut=args.tcut)
        if kern.max() > 0:
            im = ax.pcolormesh(np.degrees(phi_e), ct_e, kern, shading="flat", cmap="inferno")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(r"$\phi$ (deg)")
        if k == 0:
            ax.set_ylabel(r"$\cos\theta$")
        ax.set_title(rf"${vlo_s}\sigma < v_f < {vhi_s}\sigma$")

    fig2.suptitle(
        f"Conditional angular distribution | q={q:.4g}, e={ecc:.2f}, a/a_h={aah_mid:.2g}",
        fontsize=13,
    )
    plt.tight_layout()
    _savefig(fig2, outdir / f"{stem}_kernel2d_v2.pdf")

    # ── Figure 3: Planarity vs vtilde_cut ──
    fig3, ax3 = plt.subplots(1, 1, figsize=(7.5, 5))
    vtcuts = np.linspace(0.5, 10, 40)
    P_vals = planarity(meta, bins, sig_mid, vtcuts, tcut=args.tcut)
    good = np.isfinite(P_vals)
    if good.any():
        ax3.plot(vtcuts[good], P_vals[good], "k-", lw=2)
    ax3.axhline(1.0, color="gray", ls=":", alpha=0.5, label="isotropic")
    ax3.set_xlabel(r"$\tilde{v}_{\mathrm{cut}}$")
    ax3.set_ylabel(r"$\mathcal{P}(\tilde{v}_{\mathrm{cut}})$")
    ax3.set_title(f"Planarity parameter (a/a_h={aah_mid:.2g})")
    ax3.legend()
    plt.tight_layout()
    _savefig(fig3, outdir / f"{stem}_planarity_v2.pdf")

    # ── Optional: corner + sphere plots (weighted samples) ──
    if args.corner or args.sphere:
        pw = _particle_weights(bins, sig_mid, meta, args.tcut)
        (vf, ct, ph), w = _gather(bins, ["vf_inf", "cos_theta", "phi"], pw)
        if vf.size:
            vtilde = vf / sig_mid
            if args.corner:
                sel = vtilde > args.corner_vmin
                if sel.sum() > 20:
                    make_corner(
                        vtilde[sel], ct[sel], ph[sel], w[sel],
                        outdir / f"{stem}_corner_vmin={args.corner_vmin:g}.pdf",
                        title=f"Corner: q={q:.4g}, e={ecc:.2f}, a/a_h={aah_mid:.2g}, vtilde>{args.corner_vmin:g}",
                    )
                else:
                    print("Corner plot skipped: too few samples after vmin cut")

            if args.sphere:
                sel = vtilde > args.sphere_vmin
                if sel.sum() > 20:
                    make_sphere_plot(
                        vtilde[sel], ct[sel], ph[sel], w[sel],
                        outdir / f"{stem}_sphere_vmin={args.sphere_vmin:g}.pdf",
                        title=f"Ejection directions: a/a_h={aah_mid:.2g}, vtilde>{args.sphere_vmin:g}",
                    )
                else:
                    print("Sphere plot skipped: too few samples after vmin cut")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
