"""Experiment 6 -- the short-horizon correlation, done exactly.

Part (a).  With driftless Brownian factors,

    P_i(t) = sigma_i B_i(H_i(t)) + a_i B_0(H_0(t)),

the conditional covariance given the clocks is a_i a_h * (H_0(t+D) - H_0(t)) and
the conditional variance is sigma_i^2 dH_i + a_i^2 dH_0.  Since E[H(t)] = alpha t
*exactly* (the mean of Beta(alpha, 1-alpha) is alpha), the horizon D cancels and

    rho_{ih}(D) = a_i a_h alpha_c
                  / sqrt((s_i^2 al_i + a_i^2 al_c)(s_h^2 al_h + a_h^2 al_c))

for EVERY D and every t: staleness alone produces a perfectly flat correlation
term structure.  This is verified end to end by simulating the prices.

Part (b).  With Ornstein-Uhlenbeck factors the same expression requires
replacing alpha_j D by (1/kappa_j) E[1 - exp(-kappa_j dH_j)].  The linearization
E[1 - e^{-kappa dH}] ~ kappa E[dH] does NOT become exact as D -> 0, because dH is
zero on most paths and of order t on the rest; the correction depends on
kappa * t, not on D.  Part (b) measures that ratio, which shows exactly when the
closed form of part (a) may be used and when it may not.
"""

from __future__ import annotations

import time

import numpy as np

from common import chunk_sizes, save                               # noqa: E402
from nmf.clocks import ClockSpec, simulate_clocks, undershoot      # noqa: E402
from nmf.theory import rho_zero                                    # noqa: E402

T0 = 1.0
N_JUMPS = 16_000
N_PATHS = 40_000
CHUNK = 200
U0 = 8.0
DELTAS = (0.02, 0.1, 0.5, 2.0)
N_BOOT = 300
KAPPA_T = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

CASES = [
    dict(a_i=1.0, a_h=1.0, s_i=1.0, s_h=1.0, al_i=0.5, al_h=0.5, al_c=0.5),
    dict(a_i=2.0, a_h=0.5, s_i=1.0, s_h=2.0, al_i=0.5, al_h=0.5, al_c=0.5),
    dict(a_i=1.0, a_h=1.0, s_i=2.0, s_h=0.5, al_i=0.5, al_h=0.5, al_c=0.5),
]


def clock_increments(alpha, deltas, rng, n_paths=N_PATHS, n_clocks=3):
    """dH_j(t, t+D) for n_clocks independent clocks, on common paths."""
    spec = ClockSpec(alphas=(alpha,) * n_clocks, alpha_c=alpha,
                     loadings=(0.0,) * n_clocks)
    U = U0 * (T0 + max(deltas)) ** alpha
    out = {d: [] for d in deltas}
    for c in chunk_sizes(n_paths, CHUNK):
        paths = simulate_clocks(spec, U, c, N_JUMPS, rng)
        H0, _, ok0 = undershoot(paths, T0)
        for d in deltas:
            Hd, _, okd = undershoot(paths, T0 + d)
            m = ok0 & okd
            out[d].append(Hd[m] - H0[m])
    return {d: np.concatenate(v) for d, v in out.items()}


def main():
    started = time.time()
    rng = np.random.default_rng(1123)
    alpha = 0.5
    inc = clock_increments(alpha, DELTAS, rng)
    out = {"config": dict(alpha=alpha, t0=T0, deltas=DELTAS, n_jumps=N_JUMPS,
                          n_paths=N_PATHS, kappa_t=KAPPA_T, seed=1123)}

    # E[dH] = alpha * D exactly
    mean_rows = []
    for d in DELTAS:
        m = inc[d][:, 0]
        se = m.std() / np.sqrt(m.size)
        mean_rows.append(dict(delta=d, mean=float(m.mean()), exact=alpha * d,
                              se=float(se), z=float((m.mean() - alpha * d) / se)))
        print(f"E[dH] at D={d:<5g}: {m.mean():.6f}  exact {alpha*d:.6f}  "
              f"z={mean_rows[-1]['z']:+.2f}", flush=True)
    out["clock_mean"] = mean_rows

    # ---- (a) Brownian factors: end-to-end price simulation -----------------
    rows = []
    for ci, c in enumerate(CASES):
        for d in DELTAS:
            dH = inc[d]
            n = dH.shape[0]
            g = rng.standard_normal((n, 3))
            # conditional sd of each independent Brownian piece
            dPi = c["s_i"] * np.sqrt(dH[:, 0]) * g[:, 0] + \
                c["a_i"] * np.sqrt(dH[:, 2]) * g[:, 2]
            dPh = c["s_h"] * np.sqrt(dH[:, 1]) * g[:, 1] + \
                c["a_h"] * np.sqrt(dH[:, 2]) * g[:, 2]
            emp = float(np.corrcoef(dPi, dPh)[0, 1])
            # The conditional variances are heavy-tailed, so the usual
            # (1-r^2)/sqrt(n-3) standard error is not valid here; bootstrap it.
            bs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, n)
                bs[b] = np.corrcoef(dPi[idx], dPh[idx])[0, 1]
            se = float(bs.std(ddof=1))
            closed = float(rho_zero(c["a_i"], c["a_h"], c["s_i"], c["s_h"],
                                    c["al_i"], c["al_h"], c["al_c"]))
            rows.append(dict(case=ci, delta=d, empirical=emp, se=se,
                             closed_form=closed, z=float((emp - closed) / se)))
            print(f"case {ci} D={d:<5g}: corr={emp:.5f}+/-{se:.5f}  "
                  f"closed={closed:.5f}  z={(emp-closed)/se:+.2f}", flush=True)
    out["brownian"] = rows
    z = np.array([r["z"] for r in rows])
    flat = {}
    for ci in range(len(CASES)):
        v = [r["empirical"] for r in rows if r["case"] == ci]
        flat[ci] = float(max(v) - min(v))
    out["brownian_summary"] = dict(max_abs_z=float(np.abs(z).max()),
                                   max_spread_over_delta=float(max(flat.values())))
    print(f"\nBrownian factors: max |z| = {np.abs(z).max():.2f}; "
          f"largest spread of rho across a {DELTAS[-1]/DELTAS[0]:.0f}-fold "
          f"range of D = {max(flat.values()):.5f}", flush=True)

    # ---- (b) OU factors: when may the closed form be used? -----------------
    lin = []
    for d in DELTAS:
        dh = inc[d][:, 0]
        for kt in KAPPA_T:
            k = kt / T0
            ratio = float(np.mean(1 - np.exp(-k * dh)) / (k * dh.mean()))
            lin.append(dict(delta=d, kappa_t=kt, ratio=ratio))
    out["ou_linearization"] = lin
    print("\nE[1-e^{-k dH}] / (k E[dH])  -- 1 means the closed form is exact")
    print(f"{'kappa*t':>9} " + "".join(f"{'D='+str(d):>10}" for d in DELTAS))
    for kt in KAPPA_T:
        vals = [next(r["ratio"] for r in lin if r["delta"] == d and r["kappa_t"] == kt)
                for d in DELTAS]
        print(f"{kt:9g} " + "".join(f"{v:10.4f}" for v in vals), flush=True)
    small = [r["ratio"] for r in lin if r["kappa_t"] <= 0.1]
    big = [r["ratio"] for r in lin if r["kappa_t"] >= 10.0]
    out["ou_summary"] = dict(min_ratio_small_kt=float(min(small)),
                             max_ratio_large_kt=float(max(big)),
                             kt_small=0.1, kt_large=10.0)
    print(f"\nfor kappa t <= 0.1 the ratio stays above {min(small):.3f}; "
          f"for kappa t >= 10 it never exceeds {max(big):.3f}", flush=True)

    save("exp06_rho_exact", out, started)


if __name__ == "__main__":
    main()
