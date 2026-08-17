"""Experiment 3 -- the bivariate potential density at alpha = 1/2.

Validates:

 (a) the marginal potential density of S_j = X_j + a Z, which for alpha = 1/2 is
     u_1(s) = 1 / ((1 + a^{1/2}) * sqrt(pi s)) -- a check on the occupation
     estimator itself;
 (b) homogeneity of degree alpha - 2 of the bivariate density;
 (c) the closed-form theorem, cell by cell, against the exact occupation
     measure;
 (d) the logarithmic diagonal singularity.

Point (c) is where a discrepancy has previously been reported *on the diagonal*
while agreement off it was good.  Two quadrature schemes are compared on exactly
the same cells:

    "box"   plain tensor Gauss-Legendre over the cell, which places nodes
            symmetrically across the logarithmic ridge running through the
            cell interior;
    "split" the cell is cut along s1 = s2 and each triangle is integrated
            separately, with a w^2 substitution that annihilates the logarithm
            at the ridge.

If the discrepancy is a quadrature failure, "split" removes it and "box" does
not.  If it is a derivation error, neither does.
"""

from __future__ import annotations

import time

import numpy as np

from common import chunk_sizes, save                               # noqa: E402
from nmf.clocks import (ClockSpec, simulate_clocks,                # noqa: E402
                        potential_histogram)
from nmf.theory import potential_density_half                      # noqa: E402

ALPHA = 0.5
A = 1.0
N_JUMPS = 8_000
N_PATHS = 24_000
CHUNK = 200
U = 8.0

EDGES = np.linspace(0.5, 3.5, 13)          # 12 x 12 cells of width 0.25


# ---------------------------------------------------------------------------
# Cell-mass quadratures
# ---------------------------------------------------------------------------


def _gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def cell_mass_box(x0, x1, y0, y1, n=48):
    """Plain tensor Gauss-Legendre over the cell."""
    u, wu = _gl(n)
    xs = x0 + (x1 - x0) * u
    ys = y0 + (y1 - y0) * u
    tot = 0.0
    for i, x in enumerate(xs):
        vals = np.array([potential_density_half(x, y, A, n_gl=120) for y in ys])
        tot += wu[i] * np.sum(wu * vals)
    return tot * (x1 - x0) * (y1 - y0)


def _triangle_above(x0, x1, y0, y1, n):
    """Integral over {(x, y) in cell : y > x}, with the ridge at y = x resolved.

    For each x the inner range is (max(y0, x), y1).  Writing y = lo + (y1-lo)w^2
    clusters nodes at the lower endpoint; when lo == x that endpoint is exactly
    the logarithmic ridge, and the Jacobian 2(y1-lo)w cancels the divergence.
    """
    u, wu = _gl(n)
    xs = x0 + (x1 - x0) * u
    w, ww = _gl(n)
    tot = 0.0
    for i, x in enumerate(xs):
        lo = max(y0, x)
        if lo >= y1:
            continue
        span = y1 - lo
        ys = lo + span * w ** 2
        jac = span * 2.0 * w
        vals = np.array([potential_density_half(x, y, A, n_gl=120) for y in ys])
        tot += wu[i] * np.sum(ww * vals * jac)
    return tot * (x1 - x0)


def cell_mass_split(x0, x1, y0, y1, n=40):
    """Cell mass with the cell cut along the diagonal and each half resolved."""
    above = _triangle_above(x0, x1, y0, y1, n)
    # below the diagonal: reflect, using u(x, y) = u(y, x)
    below = _triangle_above(y0, y1, x0, x1, n)
    return above + below


# ---------------------------------------------------------------------------


def main():
    started = time.time()
    rng = np.random.default_rng(909)
    spec = ClockSpec(alphas=(ALPHA, ALPHA), alpha_c=ALPHA, loadings=(A, A))

    # ---- exact occupation measure -----------------------------------------
    acc = np.zeros((len(EDGES) - 1, len(EDGES) - 1))
    n_tot = 0
    for c in chunk_sizes(N_PATHS, CHUNK):
        paths = simulate_clocks(spec, U, c, N_JUMPS, rng)
        _, mass = potential_histogram(paths, EDGES, EDGES)
        acc += mass * c
        n_tot += c
    mc_mass = acc / n_tot                       # E[occupation time] per cell
    print(f"occupation measure from {n_tot:,} exact paths", flush=True)

    # ---- (a) marginal potential density check ------------------------------
    # S_j has Laplace exponent (1 + a^alpha) lambda^alpha, so its potential
    # density is s^{alpha-1} / ((1 + a^alpha) Gamma(alpha)).
    c_lap = 1.0 + A ** ALPHA
    centers = 0.5 * (EDGES[:-1] + EDGES[1:])
    width = np.diff(EDGES)
    marg_mc = mc_mass.sum(axis=1) / width       # density in s1, integrated over the s2 window
    print("\n(a) marginal potential density (only partly covered by the s2 window,"
          "\n    so this checks the estimator's scale, not its total mass)")

    # ---- (b) homogeneity ---------------------------------------------------
    homog = []
    for cscale in (1.5, 2.0, 3.0, 4.0):
        pts = [(0.8, 1.3), (1.1, 2.2), (0.7, 0.9), (1.6, 1.7)]
        for (s1, s2) in pts:
            u1 = potential_density_half(s1, s2, A, n_gl=600)
            u2 = potential_density_half(cscale * s1, cscale * s2, A, n_gl=600)
            homog.append(dict(c=cscale, s1=s1, s2=s2,
                              ratio=float(u2 / u1),
                              expected=float(cscale ** (ALPHA - 2)),
                              rel_err=float(abs(u2 / u1 / cscale ** (ALPHA - 2) - 1))))
    print(f"(b) homogeneity degree {ALPHA-2}: max relative error over "
          f"{len(homog)} checks = {max(h['rel_err'] for h in homog):.2e}", flush=True)

    # ---- (c) cell-by-cell comparison, box vs diagonal-split ---------------
    rows = []
    n_cells = len(EDGES) - 1
    for i in range(n_cells):
        for j in range(n_cells):
            x0, x1 = EDGES[i], EDGES[i + 1]
            y0, y1 = EDGES[j], EDGES[j + 1]
            straddles = (i == j)
            box = cell_mass_box(x0, x1, y0, y1, n=40)
            split = cell_mass_split(x0, x1, y0, y1, n=36) if straddles else box
            mc = mc_mass[i, j]
            rows.append(dict(i=i, j=j, x0=float(x0), y0=float(y0),
                             diagonal=bool(straddles), mc=float(mc),
                             box=float(box), split=float(split),
                             ratio_box=float(box / mc) if mc > 0 else np.nan,
                             ratio_split=float(split / mc) if mc > 0 else np.nan))
        print(f"  cells row {i+1}/{n_cells} done", flush=True)

    off = [r for r in rows if not r["diagonal"]]
    dia = [r for r in rows if r["diagonal"]]
    rb_off = np.array([r["ratio_box"] for r in off])
    rb_dia = np.array([r["ratio_box"] for r in dia])
    rs_dia = np.array([r["ratio_split"] for r in dia])
    summary = dict(
        off_diagonal=dict(n=len(off), median=float(np.median(rb_off)),
                          p05=float(np.percentile(rb_off, 5)),
                          p95=float(np.percentile(rb_off, 95)),
                          mean_abs_dev=float(np.mean(np.abs(rb_off - 1)))),
        diagonal_box=dict(n=len(dia), median=float(np.median(rb_dia)),
                          lo=float(rb_dia.min()), hi=float(rb_dia.max())),
        diagonal_split=dict(n=len(dia), median=float(np.median(rs_dia)),
                            lo=float(rs_dia.min()), hi=float(rs_dia.max())),
    )
    print(f"\n(c) closed form / simulation, cell by cell:")
    print(f"    off-diagonal      : median {summary['off_diagonal']['median']:.4f} "
          f"(5-95% {summary['off_diagonal']['p05']:.3f}-{summary['off_diagonal']['p95']:.3f})")
    print(f"    diagonal, box     : median {summary['diagonal_box']['median']:.4f} "
          f"[{summary['diagonal_box']['lo']:.3f}, {summary['diagonal_box']['hi']:.3f}]")
    print(f"    diagonal, split   : median {summary['diagonal_split']['median']:.4f} "
          f"[{summary['diagonal_split']['lo']:.3f}, {summary['diagonal_split']['hi']:.3f}]",
          flush=True)

    # ---- (d) logarithmic diagonal singularity ------------------------------
    deltas = [10.0 ** -k for k in range(1, 7)]
    vals = [potential_density_half(1.0, 1.0 + d, A, n_gl=2000) for d in deltas]
    incr = [float(vals[k + 1] - vals[k]) for k in range(len(vals) - 1)]
    print(f"\n(d) u(1, 1+delta): " +
          ", ".join(f"{d:.0e}->{v:.4f}" for d, v in zip(deltas, vals)))
    print(f"    increment per decade: " +
          ", ".join(f"{x:.4f}" for x in incr) +
          f"   (constant increment == logarithmic divergence)", flush=True)

    save("exp03_potential",
         dict(config=dict(alpha=ALPHA, a=A, n_jumps=N_JUMPS, n_paths=N_PATHS,
                          U=U, edges=EDGES.tolist(), seed=909),
              homogeneity=homog, cells=rows, cell_summary=summary,
              marginal_density_mc=marg_mc.tolist(), centers=centers.tolist(),
              laplace_scale=c_lap,
              log_singularity=dict(deltas=deltas, values=vals, increments=incr)),
         started)


if __name__ == "__main__":
    main()
