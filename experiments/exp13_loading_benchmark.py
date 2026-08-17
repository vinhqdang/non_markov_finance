"""Experiment 13 -- how large a common-shock loading would the data imply?

Experiment 12 measures the co-trading odds ratio between assets after
stratifying out the diurnal cycle.  This experiment produces the same statistic
*inside the model*, so the observed value can be read as a magnitude.

For a common-shock clock pair the calendar-time trade instants of asset j are
the points of the range of S_j, i.e. the partial sums of its jumps.  Binning
calendar time at width b and recording which bins contain a point of each range
gives exactly the 2x2 table of Experiment 12.  Because the clock is
self-similar, only the ratio of b to the clock scale matters, so sweeping b
traces a curve of (activity, odds ratio) for each loading a; reading it at the
observed activity level gives the model-implied odds ratio.

Caveat, stated in the manuscript as well: the empirical section rejects the
alpha < 1 marginal, so this is a magnitude benchmark for the *dependence*
structure -- which Section 7 argues is the part robust to the tail index -- and
not a fitted model.
"""

from __future__ import annotations

import time

import numpy as np

from common import chunk_sizes, save                               # noqa: E402
from nmf.clocks import ClockSpec, simulate_clocks                  # noqa: E402

ALPHA = 0.5
N_JUMPS = 4_000
N_PATHS = 4_000
CHUNK = 200
U = 8.0
LOADINGS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
BIN_GRID = np.geomspace(0.004, 1.2, 26)
T_WINDOW = 4.0                    # calendar window analysed per path
TARGETS = (0.362, 0.451, 0.368)   # observed activity levels (Experiment 12)


N_STRATA = 8          # sub-windows per path, mirroring the empirical strata


def tables_for_bins(cum, t_window, bins):
    """Per-stratum 2x2 co-occupancy counts, for each bin width.

    cum : (n_paths, 2, n_events) cumulative clock values = calendar trade times.

    Strata are (path, sub-window) cells.  This matters: activity varies a great
    deal both across paths and along a path, and pooling heterogeneous tables
    manufactures association even when the two clocks are independent.  That is
    the same confound the empirical analysis removes by stratifying on
    time-of-day, so the model statistic must be built the same way -- the
    a = 0 case below is the check that it is.
    """
    out = {b: [] for b in bins}
    n_paths = cum.shape[0]
    for p in range(n_paths):
        r1 = cum[p, 0]
        r2 = cum[p, 1]
        r1 = r1[(r1 > 0) & (r1 < t_window)]
        r2 = r2[(r2 > 0) & (r2 < t_window)]
        for b in bins:
            nb = int(np.ceil(t_window / b))
            o1 = np.zeros(nb, dtype=bool)
            o2 = np.zeros(nb, dtype=bool)
            if r1.size:
                o1[np.minimum((r1 / b).astype(int), nb - 1)] = True
            if r2.size:
                o2[np.minimum((r2 / b).astype(int), nb - 1)] = True
            edges = np.linspace(0, nb, N_STRATA + 1).astype(int)
            for s in range(N_STRATA):
                lo, hi = edges[s], edges[s + 1]
                if hi <= lo:
                    continue
                x, y = o1[lo:hi], o2[lo:hi]
                a = int(np.count_nonzero(x & y))
                bb = int(np.count_nonzero(x & ~y))
                c = int(np.count_nonzero(~x & y))
                d = (hi - lo) - a - bb - c
                out[b].append((a, bb, c, d))
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}, n_paths


def main():
    started = time.time()
    rng = np.random.default_rng(8080)
    curves, rows = {}, []

    for a in LOADINGS:
        spec = ClockSpec(alphas=(ALPHA, ALPHA), alpha_c=ALPHA, loadings=(a, a))
        acc = {b: [] for b in BIN_GRID}
        for c in chunk_sizes(N_PATHS, CHUNK):
            paths = simulate_clocks(spec, U, c, N_JUMPS, rng)
            t, _ = tables_for_bins(paths.cum, T_WINDOW, BIN_GRID)
            for b in BIN_GRID:
                acc[b].append(t[b])
        act, ors = [], []
        for b in BIN_GRID:
            T = np.concatenate(acc[b])
            A, B, C, D = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
            n = A + B + C + D
            ok = n > 0
            # Mantel-Haenszel across (path, sub-window) strata
            num = float((A[ok] * D[ok] / n[ok]).sum())
            den = float((B[ok] * C[ok] / n[ok]).sum())
            orv = num / den if den > 0 else np.nan
            tot = n.sum()
            p1 = float((A.sum() + B.sum()) / tot)
            p2 = float((A.sum() + C.sum()) / tot)
            act.append(0.5 * (p1 + p2))
            ors.append(float(orv))
        act = np.array(act)
        ors = np.array(ors)
        order = np.argsort(act)
        curves[a] = dict(bin_width=BIN_GRID.tolist(), activity=act.tolist(),
                         odds_ratio=ors.tolist())
        implied = {}
        for tg in TARGETS:
            good = np.isfinite(ors[order])
            implied[f"{tg:.3f}"] = float(
                np.interp(tg, act[order][good], ors[order][good]))
        rows.append(dict(a=a, **{f"or_at_{k}": v for k, v in implied.items()}))
        print(f"a={a:<5g}  model OR at activity "
              + "  ".join(f"{k}: {v:.3f}" for k, v in implied.items()),
              flush=True)

    # invert: which loading reproduces the observed odds ratios?
    obs = {"binance_1s": (0.362, 1.03), "binance_5s": (0.451, 1.04),
           "hose_5s": (0.368, 1.13)}
    inv = {}
    for name, (act_t, or_t) in obs.items():
        xs = [r["a"] for r in rows]
        ys = [r[f"or_at_{act_t:.3f}"] for r in rows]
        ys = np.array(ys)
        if or_t <= ys.min():
            inv[name] = 0.0
        elif or_t >= ys.max():
            inv[name] = float(max(xs))
        else:
            inv[name] = float(np.interp(or_t, ys, xs))
        print(f"observed {name}: OR={or_t} at activity {act_t} "
              f"-> implied loading a ~ {inv[name]:.2f}", flush=True)

    save("exp13_loading_benchmark",
         dict(config=dict(alpha=ALPHA, n_paths=N_PATHS, n_jumps=N_JUMPS,
                          U=U, t_window=T_WINDOW, loadings=LOADINGS,
                          targets=TARGETS, seed=8080),
              curves={str(k): v for k, v in curves.items()},
              implied_or=rows, implied_loading=inv),
         started)


if __name__ == "__main__":
    main()
