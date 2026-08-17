"""Experiment 4 -- the local evolution equation and the failure of tensorization.

Test functional.  With independent Ornstein-Uhlenbeck factors and the test
function phi(y) = y_1 y_2, every object in the evolution equation reduces to the
joint Laplace transform of the normalized clock vector,

    M(t) := E[ exp(-kappa_1 H_1(t) - kappa_2 H_2(t)) ].

Writing s_j = H_j(t)/t, the claims become

  (local equation)     M'(t) = -E[ (kappa_1 s_1 + kappa_2 s_2) e^{-t(k.s)} ]
  (size-bias constant) E[s_j] = alpha_j
  (Kummer marginal)    E[e^{-c H_j(t)}] = 1F1(alpha; 1; -ct)

and the tensorized guess -- the one obtained by replacing the joint mixing
measure with the product of its marginals -- is fully classical,

    M_tensor'(t) = - sum_j kappa_j alpha_j 1F1(alpha_j+1; 2; -kappa_j t)
                     * prod_{i != j} 1F1(alpha_i; 1; -kappa_i t),

because each marginal is exactly Beta(alpha, 1-alpha).  The ratio between the
tensorized right-hand side and the true derivative measures precisely what the
common shock contributes, and is the quantitative form of the non-tensorization
statement.

M'(t) is estimated by a central difference evaluated on *the same simulated
paths* at t-h and t+h (common random numbers), which removes almost all of the
Monte Carlo noise from the difference quotient.
"""

from __future__ import annotations

import time

import numpy as np

from common import chunk_sizes, save                               # noqa: E402
from nmf.clocks import ClockSpec, simulate_clocks, undershoot      # noqa: E402
from nmf.theory import undershoot_laplace, undershoot_laplace_quad  # noqa: E402

ALPHA = 0.5
LOADINGS = (1.0, 1.0)
N_JUMPS = 8_000
N_PATHS = 60_000
CHUNK = 250
U0 = 8.0
T = 1.0
H_STEP = 0.02
KAPPAS = ((1.0, 1.0), (0.5, 2.0), (2.0, 2.0), (0.3, 3.0), (1.0, 0.25))


def collect(alpha, loadings, t, h, rng):
    """Undershoots at t-h, t, t+h on common paths."""
    spec = ClockSpec(alphas=(alpha, alpha), alpha_c=alpha, loadings=loadings)
    U = U0 * (t + h) ** alpha
    out = {lvl: [] for lvl in ("lo", "mid", "hi")}
    n = 0
    for c in chunk_sizes(N_PATHS, CHUNK):
        paths = simulate_clocks(spec, U, c, N_JUMPS, rng)
        for lvl, tt in (("lo", t - h), ("mid", t), ("hi", t + h)):
            H, _, ok = undershoot(paths, tt)
            out[lvl].append(np.where(ok[:, None], H, np.nan))
        n += c
    return {k: np.concatenate(v) for k, v in out.items()}, n


def main():
    started = time.time()
    rng = np.random.default_rng(31415)
    out = {"config": dict(alpha=ALPHA, loadings=LOADINGS, n_jumps=N_JUMPS,
                          n_paths=N_PATHS, t=T, h=H_STEP, kappas=KAPPAS,
                          U0=U0, seed=31415)}

    H, n = collect(ALPHA, LOADINGS, T, H_STEP, rng)
    good = np.isfinite(H["lo"]).all(1) & np.isfinite(H["mid"]).all(1) & \
        np.isfinite(H["hi"]).all(1)
    H = {k: v[good] for k, v in H.items()}
    n_eff = int(good.sum())
    print(f"paths usable at all three horizons: {n_eff:,} / {n:,}", flush=True)

    # ---- size-bias constant E[s_j] = alpha ---------------------------------
    s = H["mid"] / T
    sb = dict(mean_s1=float(s[:, 0].mean()), mean_s2=float(s[:, 1].mean()),
              exact=ALPHA,
              se=float(np.sqrt(ALPHA * (1 - ALPHA) / 2.0 / n_eff)))
    sb["z1"] = (sb["mean_s1"] - ALPHA) / sb["se"]
    sb["z2"] = (sb["mean_s2"] - ALPHA) / sb["se"]
    out["size_bias_constant"] = sb
    print(f"E[s_j] = {sb['mean_s1']:.5f}, {sb['mean_s2']:.5f}  (exact {ALPHA}, "
          f"z = {sb['z1']:+.2f}, {sb['z2']:+.2f})", flush=True)

    # ---- Kummer marginal transform -----------------------------------------
    kum = []
    for c in (0.25, 1.0, 4.0):
        mc = float(np.exp(-c * H["mid"][:, 0]).mean())
        se = float(np.exp(-c * H["mid"][:, 0]).std() / np.sqrt(n_eff))
        th = float(undershoot_laplace(c, T, ALPHA))
        q, qerr = undershoot_laplace_quad(c, T, ALPHA)
        kum.append(dict(c=c, mc=mc, mc_se=se, hyp1f1=th, quad=float(q),
                        quad_err=float(qerr),
                        rel_hyp_vs_quad=float(abs(th - q) / q),
                        z=float((mc - th) / se)))
        print(f"E[e^-{c}H] : MC={mc:.6f}+/-{se:.6f}  1F1={th:.6f}  "
              f"quad={q:.6f}  (1F1 vs quad rel {abs(th-q)/q:.1e}, z={((mc-th)/se):+.2f})",
              flush=True)
    out["kummer"] = kum

    # ---- local equation and tensorization ----------------------------------
    from scipy.special import hyp1f1
    rows = []
    for (k1, k2) in KAPPAS:
        kv = np.array([k1, k2])
        e_lo = np.exp(-(H["lo"] * kv).sum(1))
        e_hi = np.exp(-(H["hi"] * kv).sum(1))
        e_mid = np.exp(-(H["mid"] * kv).sum(1))

        lhs = float((e_hi.mean() - e_lo.mean()) / (2 * H_STEP))
        # paired standard error thanks to common random numbers
        lhs_se = float((e_hi - e_lo).std() / np.sqrt(n_eff) / (2 * H_STEP))

        w = (H["mid"] * kv).sum(1) / T
        rhs = float(-(w * e_mid).mean())
        rhs_se = float((w * e_mid).std() / np.sqrt(n_eff))

        # tensorized (independent-clock) prediction, closed form
        f1, f2 = hyp1f1(ALPHA, 1.0, -k1 * T), hyp1f1(ALPHA, 1.0, -k2 * T)
        g1 = ALPHA * hyp1f1(ALPHA + 1, 2.0, -k1 * T)
        g2 = ALPHA * hyp1f1(ALPHA + 1, 2.0, -k2 * T)
        rhs_tensor = float(-(k1 * g1 * f2 + k2 * g2 * f1))
        m_tensor = float(f1 * f2)

        rows.append(dict(
            kappa1=k1, kappa2=k2,
            M_mc=float(e_mid.mean()), M_tensor=m_tensor,
            M_ratio=float(m_tensor / e_mid.mean()),
            lhs=lhs, lhs_se=lhs_se, rhs=rhs, rhs_se=rhs_se,
            local_ratio=float(rhs / lhs),
            local_z=float((rhs - lhs) / np.hypot(rhs_se, lhs_se)),
            rhs_tensor=rhs_tensor, tensor_ratio=float(rhs_tensor / lhs),
        ))
        r = rows[-1]
        print(f"k=({k1},{k2}): M'(t) LHS={lhs:+.5f}+/-{lhs_se:.5f}  "
              f"RHS={rhs:+.5f}+/-{rhs_se:.5f}  ratio={r['local_ratio']:.4f} "
              f"(z={r['local_z']:+.2f}) | tensorized RHS={rhs_tensor:+.5f} "
              f"ratio={r['tensor_ratio']:.4f}", flush=True)
    out["evolution"] = rows

    tr = np.array([r["tensor_ratio"] for r in rows])
    lr = np.array([r["local_ratio"] for r in rows])
    out["summary"] = dict(
        local_ratio_min=float(lr.min()), local_ratio_max=float(lr.max()),
        tensor_ratio_min=float(tr.min()), tensor_ratio_max=float(tr.max()),
        max_local_z=float(max(abs(r["local_z"]) for r in rows)),
    )
    print(f"\nlocal equation ratio in [{lr.min():.4f}, {lr.max():.4f}]  "
          f"(max |z| = {out['summary']['max_local_z']:.2f})")
    print(f"tensorized ratio in [{tr.min():.4f}, {tr.max():.4f}]  "
          f"-- departure from 1 is the common-shock contribution", flush=True)

    save("exp04_evolution", out, started)


if __name__ == "__main__":
    main()
