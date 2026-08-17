"""Experiment 5 -- short-horizon correlation and the correlation term structure.

Model.  Asset i carries an idiosyncratic factor on its own staleness clock plus
a common factor on the common clock,

    P_i(t) = sigma_i Y_i(H_i(t)) + a_i Y_0(H_0(t)),
    dY = -kappa Y du + sigma dW      (sigma_0 = 1),

with H_i the undershoot of a standard alpha_i-stable subordinator and H_0 that
of a standard alpha_c-stable subordinator, independent of the factors.

For a stationary OU factor, Var(Y(u) - Y(v)) = (sigma^2/kappa)(1 - e^{-kappa|u-v|}),
so every second moment of the calendar-time return over [t, t+D] reduces to

    g_alpha(kappa, D) := E[ 1 - exp(-kappa (H(t+D) - H(t))) ].

That single function is tabulated once by exact simulation and then reused, so
the parameter sweep costs no further simulation.

Two claims are tested:

  (a) rho(0+) = a_i a_h alpha_c / sqrt((s_i^2 al_i + a_i^2 al_c)(s_h^2 al_h + a_h^2 al_c)),
      which follows from E[H(t)] = alpha t and is free of every kappa;

  (b) the *shape* of D -> rho(D) is not pinned down by the model.  All four
      monotonicity classes are counted across a large grid.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
from scipy.stats import kstest

from common import chunk_sizes, save                               # noqa: E402
from nmf.clocks import ClockSpec, simulate_clocks, undershoot      # noqa: E402
from nmf.theory import rho_zero, undershoot_law                    # noqa: E402

T0 = 1.0
N_JUMPS = 8_000
N_PATHS = 30_000
CHUNK = 250
U0 = 8.0

ALPHA_GRID = (0.2, 0.35, 0.5, 0.65)
DELTAS = tuple(np.round(np.geomspace(0.01, 2.0, 12), 5))
KAPPA_TAB = np.geomspace(1e-3, 1e3, 241)

SIGMAS = (0.5, 1.0, 2.0)
LOADS = (0.5, 1.0, 2.0)
KAPPAS = (0.1, 1.0, 10.0)


# ---------------------------------------------------------------------------


def tabulate(rng):
    """g_alpha(kappa, D) = E[1 - exp(-kappa (H(t+D) - H(t)))], by exact simulation."""
    table, checks = {}, []
    for alpha in ALPHA_GRID:
        spec = ClockSpec(alphas=(alpha,), alpha_c=alpha, loadings=(0.0,))
        U = U0 * (T0 + DELTAS[-1]) ** alpha
        base, incs = [], {d: [] for d in DELTAS}
        for c in chunk_sizes(N_PATHS, CHUNK):
            paths = simulate_clocks(spec, U, c, N_JUMPS, rng)
            H0, _, ok0 = undershoot(paths, T0)
            for d in DELTAS:
                Hd, _, okd = undershoot(paths, T0 + d)
                m = ok0 & okd
                incs[d].append((Hd[m, 0] - H0[m, 0]))
            base.append(H0[ok0, 0])
        base = np.concatenate(base)
        ks = kstest(base / T0, undershoot_law(alpha).cdf)
        checks.append(dict(alpha=alpha, n=int(base.size),
                           mean=float(base.mean()), exact=alpha,
                           ks_D=float(ks.statistic), ks_p=float(ks.pvalue)))
        print(f"  alpha={alpha}: n={base.size:,} mean(H/t)={base.mean():.5f} "
              f"(exact {alpha}) KS p={ks.pvalue:.3f}", flush=True)
        for d in DELTAS:
            dh = np.concatenate(incs[d])
            # E[1 - e^{-k dh}] on the kappa grid
            table[(alpha, d)] = 1.0 - np.exp(-np.outer(KAPPA_TAB, dh)).mean(axis=1)
            table[(alpha, d, "mean")] = float(dh.mean())
    return table, checks


def g_interp(table, alpha, d, kappa):
    """Log-linear interpolation of g_alpha(., d) at the requested kappas."""
    y = table[(alpha, d)]
    return np.interp(np.log(kappa), np.log(KAPPA_TAB), y)


def classify(rho, tol=1e-4):
    """Monotonicity class of a correlation term structure."""
    r = np.asarray(rho, dtype=float)
    if not np.all(np.isfinite(r)):
        return "invalid"
    if r.max() - r.min() < tol * max(abs(r).max(), 1e-12):
        return "flat"
    hi, lo = int(np.argmax(r)), int(np.argmin(r))
    n = r.size - 1
    if hi == n and lo == 0:
        return "rising"
    if hi == 0 and lo == n:
        return "falling"
    if 0 < hi < n:
        return "hump"
    if 0 < lo < n:
        return "dip"
    return "other"


def main():
    started = time.time()
    rng = np.random.default_rng(2718)

    print("tabulating clock-increment transforms ...", flush=True)
    table, checks = tabulate(rng)

    # ---- (a) rho(0+) : simulation-based limit vs closed form ---------------
    small = DELTAS[0]
    rho0_rows = []
    for (ai, ah, si, sh, al_i, al_h, al_c) in [
        (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5),
        (2.0, 0.5, 1.0, 2.0, 0.2, 0.65, 0.35),
        (0.5, 0.5, 2.0, 0.5, 0.65, 0.2, 0.5),
        (1.0, 2.0, 0.5, 1.0, 0.35, 0.35, 0.2),
    ]:
        closed = rho_zero(ai, ah, si, sh, al_i, al_h, al_c)
        # numeric limit using the tabulated increments at the smallest Delta,
        # swept over a 4096-fold range of every mean-reversion speed
        vals = []
        for (ki, kh, k0) in itertools.product((0.05, 0.5, 5.0, 51.2),
                                              (0.05, 0.5, 5.0, 51.2),
                                              (0.05, 0.5, 5.0, 51.2)):
            gi = g_interp(table, al_i, small, np.array([ki]))[0]
            gh = g_interp(table, al_h, small, np.array([kh]))[0]
            g0 = g_interp(table, al_c, small, np.array([k0]))[0]
            cov = ai * ah * g0 / k0
            vi = si ** 2 * gi / ki + ai ** 2 * g0 / k0
            vh = sh ** 2 * gh / kh + ah ** 2 * g0 / k0
            vals.append(cov / np.sqrt(vi * vh))
        vals = np.array(vals)
        rho0_rows.append(dict(
            a_i=ai, a_h=ah, sigma_i=si, sigma_h=sh,
            alpha_i=al_i, alpha_h=al_h, alpha_c=al_c,
            closed_form=float(closed),
            numeric_mean=float(vals.mean()),
            numeric_spread=float(vals.max() - vals.min()),
            kappa_range=51.2 / 0.05,
            rel_err=float(abs(vals.mean() - closed) / closed),
        ))
        r = rho0_rows[-1]
        print(f"rho(0+) closed={closed:.6f}  numeric={vals.mean():.6f}  "
              f"rel err={r['rel_err']:.2e}  spread over a "
              f"{r['kappa_range']:.0f}x kappa range = {r['numeric_spread']:.2e}",
              flush=True)

    # ---- (b) term-structure shape sweep ------------------------------------
    print("\nsweeping the correlation term structure ...", flush=True)
    combos = list(itertools.product(ALPHA_GRID, ALPHA_GRID, ALPHA_GRID,
                                    SIGMAS, SIGMAS, LOADS, LOADS,
                                    KAPPAS, KAPPAS, KAPPAS))
    counts, slopes = {}, {"rise": 0, "fall": 0, "tie": 0}
    rho_first_last = []
    for (al_i, al_h, al_c, si, sh, ai, ah, ki, kh, k0) in combos:
        gi = g_interp(table, al_i, DELTAS[0], np.array([ki]))
        rho = np.empty(len(DELTAS))
        for m, d in enumerate(DELTAS):
            gi = g_interp(table, al_i, d, np.array([ki]))[0]
            gh = g_interp(table, al_h, d, np.array([kh]))[0]
            g0 = g_interp(table, al_c, d, np.array([k0]))[0]
            cov = ai * ah * g0 / k0
            vi = si ** 2 * gi / ki + ai ** 2 * g0 / k0
            vh = sh ** 2 * gh / kh + ah ** 2 * g0 / k0
            rho[m] = cov / np.sqrt(vi * vh)
        cls = classify(rho)
        counts[cls] = counts.get(cls, 0) + 1
        rho_first_last.append((rho[0], rho[-1]))
        # does the sign of the initial slope follow alpha_i kappa_i vs alpha_c kappa_0?
        pred_up = (al_i * ki + al_h * kh) / 2.0 > al_c * k0
        actual_up = rho[1] > rho[0]
        slopes["tie"] += int(abs(rho[1] - rho[0]) < 1e-12)
        if pred_up == actual_up:
            slopes["rise"] += 1
        else:
            slopes["fall"] += 1

    n = len(combos)
    shares = {k: 100.0 * v / n for k, v in sorted(counts.items(),
                                                  key=lambda kv: -kv[1])}
    print(f"\n{n:,} parameter combinations")
    for k, v in shares.items():
        print(f"  {k:<9s} {v:6.2f}%   ({counts[k]:,})")
    rule_acc = 100.0 * slopes["rise"] / n
    print(f"\ninitial-slope sign rule (alpha_j kappa_j vs alpha_c kappa_0): "
          f"{rule_acc:.1f}% of cells", flush=True)

    save("exp05_correlation",
         dict(config=dict(t0=T0, deltas=list(DELTAS), alpha_grid=ALPHA_GRID,
                          sigmas=SIGMAS, loads=LOADS, kappas=KAPPAS,
                          n_jumps=N_JUMPS, n_paths=N_PATHS, seed=2718),
              simulator_checks=checks,
              rho_zero=rho0_rows,
              sweep=dict(n_combinations=n, counts=counts, shares=shares,
                         slope_rule_accuracy_pct=rule_acc)),
         started)


if __name__ == "__main__":
    main()
