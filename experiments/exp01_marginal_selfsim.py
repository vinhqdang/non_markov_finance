"""Experiment 1 -- simulator validation, marginal law, and exact self-similarity.

Checks, for the common-shock clock vector S_j = X_j + a_j Z:

 (a) H_j(t)/t ~ Beta(alpha, 1-alpha) exactly, for every t and every loading a_j
     (Proposition: common shocks are invisible in the marginal).  Because the
     marginal law is known in closed form, this doubles as the validation of
     the exact-jump simulator.

 (b) the joint law of (H_1(t)/t, H_2(t)/t) does not depend on t
     (Proposition: exact self-similarity), tested over a 64-fold range of t.

The operational horizon is scaled as U = U0 * t^alpha so that the *relative*
series-truncation error is identical at every t; U itself does not affect the
law of H(t) provided the path crosses level t, which is verified per run.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import kstest, ks_2samp

from common import RESULTS, chunk_sizes, save, se_prop           # noqa: E402
from nmf.clocks import (ClockSpec, simulate_clocks, undershoot,  # noqa: E402
                        truncation_mass)
from nmf.theory import undershoot_law                            # noqa: E402

ALPHAS = (0.3, 0.4, 0.5)
T_GRID = (0.25, 1.0, 4.0, 16.0)          # 64-fold range
U0 = 8.0
N_JUMPS = 8_000
N_PATHS = 40_000
CHUNK = 200
LOADINGS = (1.0, 1.0)

# Series length for the convergence study.  The analytic truncation bound is the
# expected omitted jump mass over the whole operational horizon [0, U]; what
# actually matters is the omitted mass over [0, L(t)], which is far smaller.
# This sweep measures the realized bias so that N_JUMPS is chosen from evidence
# rather than from the (very conservative) bound.
CONV_N = (1_000, 2_000, 4_000, 8_000, 16_000)
CONV_PATHS = 16_000


def run_one(alpha, t, rng):
    """Simulate the clock pair and return the normalized undershoots."""
    U = U0 * t ** alpha
    spec = ClockSpec(alphas=(alpha, alpha), alpha_c=alpha, loadings=LOADINGS)
    H, atom, n_ok, n_tot = [], [], 0, 0
    for c in chunk_sizes(N_PATHS, CHUNK):
        paths = simulate_clocks(spec, U, c, N_JUMPS, rng)
        h, idx, ok = undershoot(paths, t)
        n_tot += c
        n_ok += int(ok.sum())
        H.append(h[ok])
        atom.append(idx[ok, 0] == idx[ok, 1])
    H = np.concatenate(H) / t
    atom = np.concatenate(atom)
    return H, atom, n_ok / n_tot, truncation_mass(alpha, U, N_JUMPS) / t


def convergence(alpha, t, rng):
    """Realized bias of the simulator as a function of series truncation."""
    law = undershoot_law(alpha)
    rows = []
    for N in CONV_N:
        U = U0 * t ** alpha
        spec = ClockSpec(alphas=(alpha, alpha), alpha_c=alpha, loadings=LOADINGS)
        H = []
        for c in chunk_sizes(CONV_PATHS, 200):
            paths = simulate_clocks(spec, U, c, N, rng)
            h, _, ok = undershoot(paths, t)
            H.append(h[ok])
        H = np.concatenate(H)[:, 0] / t
        ks = kstest(H, law.cdf)
        mc_se = float(np.sqrt(alpha * (1 - alpha) / 2.0 / len(H)))
        rows.append(dict(
            n_jumps=N, n=int(len(H)),
            analytic_bound=truncation_mass(alpha, U, N) / t,
            mean=float(H.mean()), bias=float(H.mean() - alpha), mc_se=mc_se,
            bias_in_se=float((H.mean() - alpha) / mc_se),
            ks_D=float(ks.statistic), ks_p=float(ks.pvalue),
        ))
        print(f"  conv alpha={alpha} N={N:>6d}: bound={rows[-1]['analytic_bound']:.1e} "
              f"bias={rows[-1]['bias']:+.5f} ({rows[-1]['bias_in_se']:+.1f} s.e.) "
              f"KSp={ks.pvalue:.3f}", flush=True)
    return rows


def main():
    started = time.time()
    rng = np.random.default_rng(20260817)
    out = {"config": dict(alphas=ALPHAS, t_grid=T_GRID, U0=U0, n_jumps=N_JUMPS,
                          n_paths=N_PATHS, loadings=LOADINGS, seed=20260817,
                          conv_n=CONV_N, conv_paths=CONV_PATHS),
           "runs": []}

    print("--- truncation convergence (alpha=0.5, t=1) ---", flush=True)
    out["convergence"] = convergence(0.5, 1.0, rng)

    samples = {}
    for alpha in ALPHAS:
        law = undershoot_law(alpha)
        for t in T_GRID:
            t0 = time.time()
            H, atom, cover, rel_trunc = run_one(alpha, t, rng)
            samples[(alpha, t)] = H
            ks1 = kstest(H[:, 0], law.cdf)
            ks2 = kstest(H[:, 1], law.cdf)
            rec = dict(
                alpha=alpha, t=t, n=int(len(H)), coverage=cover,
                rel_truncation=rel_trunc,
                mean_1=float(H[:, 0].mean()), mean_2=float(H[:, 1].mean()),
                mean_exact=alpha,
                sd_1=float(H[:, 0].std()),
                sd_exact=float(np.sqrt(alpha * (1 - alpha) / 2.0)),
                ks_D_1=float(ks1.statistic), ks_p_1=float(ks1.pvalue),
                ks_D_2=float(ks2.statistic), ks_p_2=float(ks2.pvalue),
                corr=float(np.corrcoef(H[:, 0], H[:, 1])[0, 1]),
                atom_prob=float(atom.mean()), atom_se=se_prop(atom.mean(), len(atom)),
                seconds=round(time.time() - t0, 1),
            )
            out["runs"].append(rec)
            print(f"alpha={alpha} t={t:<6g} n={rec['n']:>6d} cover={cover:.4f} "
                  f"relTrunc={rel_trunc:.1e} mean={rec['mean_1']:.4f}/{alpha} "
                  f"KSp={ks1.pvalue:.3f},{ks2.pvalue:.3f} "
                  f"corr={rec['corr']:.4f} atom={rec['atom_prob']:.4f}", flush=True)

    # ---- self-similarity: two-sample tests between the extreme horizons -----
    ss = []
    for alpha in ALPHAS:
        lo, hi = samples[(alpha, T_GRID[0])], samples[(alpha, T_GRID[-1])]
        rows = dict(alpha=alpha, t_lo=T_GRID[0], t_hi=T_GRID[-1], ratio=T_GRID[-1] / T_GRID[0])
        for j in (0, 1):
            r = ks_2samp(lo[:, j], hi[:, j])
            rows[f"ks2_D_{j+1}"] = float(r.statistic)
            rows[f"ks2_p_{j+1}"] = float(r.pvalue)
        # projections onto random directions probe the joint law, not just marginals
        g = np.random.default_rng(7)
        ps = []
        for _ in range(8):
            v = g.normal(size=2)
            v /= np.linalg.norm(v)
            ps.append(float(ks_2samp(lo @ v, hi @ v).pvalue))
        rows["proj_p_min"] = min(ps)
        rows["proj_p_median"] = float(np.median(ps))
        ss.append(rows)
        print(f"self-similarity alpha={alpha}: marginal p="
              f"{rows['ks2_p_1']:.3f},{rows['ks2_p_2']:.3f}  "
              f"projection p min={rows['proj_p_min']:.3f} "
              f"median={rows['proj_p_median']:.3f}", flush=True)
    out["self_similarity"] = ss

    save("exp01_marginal_selfsim", out, started)


if __name__ == "__main__":
    main()
