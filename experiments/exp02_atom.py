"""Experiment 2 -- the simultaneous-staleness atom.

Three claims are tested at once.

 (a) Proposition (existence and scale-freeness): P[L_1(t) = L_2(t)] > 0 whenever
     both loadings are positive, and does not depend on t.  It is exactly zero
     when either loading vanishes.

 (b) Theorem (compensation-formula representation): the atom equals

         pi = int_{[0,t]^2} Pibar_c( (t-s1)/a1 v (t-s2)/a2 ) U(ds1, ds2).

     Evaluating the right-hand side needs the closed-form bivariate potential
     density, so agreement with simulation validates that theorem too.

 (c) The atom is not a discretization artifact.  This simulator never
     discretizes time -- crossings are compared by *event index*, so
     {L_1 = L_2} is decided exactly -- which settles the question by
     construction rather than by grid refinement.
"""

from __future__ import annotations

import time

import numpy as np

from common import chunk_sizes, save, se_prop                      # noqa: E402
from nmf.clocks import ClockSpec, simulate_clocks, undershoot      # noqa: E402
from nmf.theory import atom_compensation_half                      # noqa: E402

ALPHA = 0.5
N_JUMPS = 8_000
N_PATHS = 60_000
CHUNK = 300
U0 = 8.0

LOADINGS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
T_GRID = (0.25, 1.0, 4.0, 16.0)


def mc_atom(a, t, rng, n_paths=N_PATHS):
    spec = ClockSpec(alphas=(ALPHA, ALPHA), alpha_c=ALPHA, loadings=(a, a))
    U = U0 * t ** ALPHA
    hits, n = 0, 0
    for c in chunk_sizes(n_paths, CHUNK):
        paths = simulate_clocks(spec, U, c, N_JUMPS, rng)
        _, idx, ok = undershoot(paths, t)
        hits += int((idx[ok, 0] == idx[ok, 1]).sum())
        n += int(ok.sum())
    p = hits / n
    return p, se_prop(p, n), n


def main():
    started = time.time()
    rng = np.random.default_rng(4242)
    out = {"config": dict(alpha=ALPHA, n_jumps=N_JUMPS, n_paths=N_PATHS,
                          loadings=LOADINGS, t_grid=T_GRID, U0=U0, seed=4242)}

    # ---- quadrature convergence of the closed-form representation ----------
    conv = []
    for n_q in (100, 200, 300, 400, 600):
        v = atom_compensation_half(t=1.0, a=1.0, n_outer=n_q, n_inner=n_q)
        conv.append(dict(n_nodes=n_q, pi=float(v)))
        print(f"quadrature n={n_q:>4d}: pi={v:.8f}", flush=True)
    out["quadrature_convergence"] = conv

    # ---- loading sweep: theory vs exact simulation -------------------------
    sweep = []
    for a in LOADINGS:
        t0 = time.time()
        p, se, n = mc_atom(a, 1.0, rng)
        th = 0.0 if a == 0 else atom_compensation_half(t=1.0, a=a,
                                                       n_outer=500, n_inner=500)
        z = (p - th) / se if se > 0 else np.nan
        sweep.append(dict(a=a, mc=p, mc_se=se, n=n, theory=float(th),
                          diff=float(p - th), z=float(z),
                          seconds=round(time.time() - t0, 1)))
        print(f"a={a:<5g} MC={p:.5f} +/- {se:.5f} (n={n:,})  theory={th:.5f}  "
              f"diff={p-th:+.5f}  z={z:+.2f}", flush=True)
    out["loading_sweep"] = sweep

    # ---- horizon sweep: the atom must be t-free ---------------------------
    horiz = []
    for t in T_GRID:
        p, se, n = mc_atom(1.0, t, rng, n_paths=40_000)
        horiz.append(dict(t=t, mc=p, mc_se=se, n=n))
        print(f"t={t:<6g} MC={p:.5f} +/- {se:.5f} (n={n:,})", flush=True)
    ps = np.array([h["mc"] for h in horiz])
    ses = np.array([h["mc_se"] for h in horiz])
    w = 1.0 / ses ** 2
    pooled = float((w * ps).sum() / w.sum())
    chi2 = float((((ps - pooled) / ses) ** 2).sum())
    out["horizon_sweep"] = dict(rows=horiz, ratio=T_GRID[-1] / T_GRID[0],
                                pooled=pooled, chi2=chi2, df=len(ps) - 1,
                                spread=float(ps.max() - ps.min()))
    print(f"\nt-invariance over a {T_GRID[-1]/T_GRID[0]:g}x range: "
          f"pooled={pooled:.5f} spread={ps.max()-ps.min():.5f} "
          f"chi2={chi2:.2f} on {len(ps)-1} df", flush=True)

    save("exp02_atom", out, started)


if __name__ == "__main__":
    main()
