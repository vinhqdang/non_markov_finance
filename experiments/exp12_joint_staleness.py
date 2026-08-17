"""Experiment 12 -- is trade arrival dependent across assets, beyond the clock?

The theory says the common shock is invisible in any single asset's staleness
distribution (Proposition: marginals are exactly Beta whatever the loading) and
is not recoverable from the correlation term structure (Proposition: flat under
staleness alone).  What it does predict is an atom of *simultaneous* staleness:
clocks that share a jump component advance together, so two assets freeze and
un-freeze together more often than independent clocks allow.

That is the one prediction of the framework that does not depend on the tail
index, and it is directly observable.  Bin calendar time at width b and let

    A_i(k) = 1 { asset i traded in bin k }.

For each asset pair this gives a 2x2 table of co-trading counts, whose odds
ratio is the natural symmetric dependence measure: independent clocks give 1, a
shared clock component gives more than 1.

Two things would otherwise fake a positive result and are handled here.

  * Both assets are quiet at the same times of day, which by itself makes trade
    arrival positively dependent.  We therefore stratify by time-of-day bucket
    and pool with the Mantel-Haenszel estimator, which is exactly the tool for a
    common stratifying confound.  The unstratified ratio is reported alongside
    so the size of the confound is visible.

  * Bins are strongly autocorrelated, so naive counts overstate the effective
    sample size.  Confidence intervals come from a block bootstrap that
    resamples whole time-of-day strata.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd

from common import ROOT, save                                      # noqa: E402

CRYPTO = ROOT / "data" / "binance"
HOSE = ROOT / "data" / "hose"

BINS_CRYPTO = (1.0, 5.0, 30.0)
BINS_HOSE = (5.0, 15.0, 60.0)
STRATUM_S_CRYPTO = 1800.0          # 30-minute time-of-day strata
STRATUM_S_HOSE = 600.0             # 10-minute strata within the session
MIN_TRADES = 400
N_BOOT = 400
# a pair is informative only if neither asset trades in almost every bin, nor in
# almost none; outside that range the 2x2 table is degenerate
P_LO, P_HI = 0.02, 0.98


def activity_matrix(times_list, t0, t1, bin_s):
    """Boolean (n_assets, n_bins): did asset i trade in bin k?"""
    edges = np.arange(t0, t1 + bin_s, bin_s)
    n_bins = edges.size - 1
    out = np.zeros((len(times_list), n_bins), dtype=bool)
    for i, ts in enumerate(times_list):
        idx = np.searchsorted(edges, ts, side="right") - 1
        idx = idx[(idx >= 0) & (idx < n_bins)]
        out[i, np.unique(idx)] = True
    return out, edges[:-1]


def mh_tables(ai, ah, stratum, n_strata):
    """Per-stratum 2x2 counts (a=both, b=i only, c=h only, d=neither)."""
    both = (ai & ah).astype(np.int64)
    ionly = (ai & ~ah).astype(np.int64)
    honly = (~ai & ah).astype(np.int64)
    neither = (~ai & ~ah).astype(np.int64)
    a = np.bincount(stratum, weights=both, minlength=n_strata)
    b = np.bincount(stratum, weights=ionly, minlength=n_strata)
    c = np.bincount(stratum, weights=honly, minlength=n_strata)
    d = np.bincount(stratum, weights=neither, minlength=n_strata)
    return a, b, c, d


def mh_or(a, b, c, d):
    """Mantel-Haenszel common odds ratio across strata."""
    n = a + b + c + d
    ok = n > 0
    num = float((a[ok] * d[ok] / n[ok]).sum())
    den = float((b[ok] * c[ok] / n[ok]).sum())
    if den <= 0:
        return np.nan
    return num / den


def crude_or(a, b, c, d):
    A, B, C, D = a.sum(), b.sum(), c.sum(), d.sum()
    if B * C <= 0:
        return np.nan
    return float(A * D / (B * C))


def analyse(name, series, t0, t1, bin_list, stratum_s, rng):
    syms = sorted(series)
    rows, per_pair = [], []
    for bin_s in bin_list:
        A, starts = activity_matrix([series[s] for s in syms], t0, t1, bin_s)
        stratum = ((starts - t0) // stratum_s).astype(int)
        n_str = int(stratum.max()) + 1
        p = A.mean(axis=1)
        keep = [i for i in range(len(syms)) if P_LO <= p[i] <= P_HI]
        mh_list, cr_list, sig = [], [], 0
        for i, h in itertools.combinations(keep, 2):
            a, b, c, d = mh_tables(A[i], A[h], stratum, n_str)
            m, cr = mh_or(a, b, c, d), crude_or(a, b, c, d)
            if not np.isfinite(m) or not np.isfinite(cr):
                continue
            # block bootstrap over strata
            bs = np.empty(N_BOOT)
            for t in range(N_BOOT):
                s = rng.integers(0, n_str, n_str)
                bs[t] = mh_or(a[s], b[s], c[s], d[s])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            mh_list.append(m)
            cr_list.append(cr)
            sig += int(lo > 1.0)
            per_pair.append(dict(market=name, bin_s=bin_s,
                                 sym_i=syms[i], sym_h=syms[h],
                                 p_i=float(p[i]), p_h=float(p[h]),
                                 or_crude=cr, or_mh=m,
                                 ci_lo=float(lo), ci_hi=float(hi)))
        mh_arr, cr_arr = np.array(mh_list), np.array(cr_list)
        rec = dict(market=name, bin_s=bin_s, n_assets=len(keep),
                   n_pairs=int(mh_arr.size),
                   mean_activity=float(p[keep].mean()),
                   or_crude_median=float(np.median(cr_arr)),
                   or_mh_median=float(np.median(mh_arr)),
                   or_mh_q25=float(np.percentile(mh_arr, 25)),
                   or_mh_q75=float(np.percentile(mh_arr, 75)),
                   frac_above_one=float((mh_arr > 1).mean()),
                   frac_sig_above_one=float(sig / max(mh_arr.size, 1)))
        rows.append(rec)
        print(f"  {name} bin={bin_s:>5g}s  {rec['n_pairs']:>4d} pairs  "
              f"activity={rec['mean_activity']:.3f}  "
              f"OR crude={rec['or_crude_median']:.2f} -> MH={rec['or_mh_median']:.2f} "
              f"[{rec['or_mh_q25']:.2f},{rec['or_mh_q75']:.2f}]  "
              f">1: {100*rec['frac_above_one']:.0f}%  "
              f"CI excludes 1: {100*rec['frac_sig_above_one']:.0f}%", flush=True)
    return rows, per_pair, syms


def main():
    started = time.time()
    rng = np.random.default_rng(606)
    out, pairs = {}, []

    man = json.loads((CRYPTO / "manifest.json").read_text())
    t0, t1 = man["window_start_ms"] / 1000.0, man["window_end_ms"] / 1000.0
    series = {}
    for m in man["symbols"]:
        p = CRYPTO / f"{m['symbol']}.parquet"
        if p.exists() and m["n_agg"] >= MIN_TRADES:
            ts = pd.read_parquet(p, columns=["ts_ms"])["ts_ms"].to_numpy(float) / 1e3
            series[m["symbol"]] = np.sort(ts)
    print(f"binance: {len(series)} symbols, {(t1-t0)/3600:.0f}h window", flush=True)
    r, pp, syms = analyse("binance", series, t0, t1, BINS_CRYPTO,
                          STRATUM_S_CRYPTO, rng)
    out["binance"] = dict(rows=r, symbols=syms, stratum_s=STRATUM_S_CRYPTO)
    pairs += pp

    hman = json.loads((HOSE / "manifest.json").read_text())
    hs, lo, hi = {}, None, None
    for m in hman["symbols"]:
        p = HOSE / f"{m['symbol']}.parquet"
        if p.exists() and m["n_ticks"] >= 150:
            ts = np.sort(pd.read_parquet(p, columns=["time"])["time"]
                         .astype("int64").to_numpy() / 1e9)
            hs[m["symbol"]] = ts
            lo = ts[0] if lo is None else min(lo, ts[0])
            hi = ts[-1] if hi is None else max(hi, ts[-1])
    print(f"\nhose: {len(hs)} symbols, {(hi-lo)/60:.0f} min session", flush=True)
    r2, pp2, hsyms = analyse("hose", hs, lo, hi, BINS_HOSE, STRATUM_S_HOSE, rng)
    out["hose"] = dict(rows=r2, symbols=hsyms, stratum_s=STRATUM_S_HOSE,
                       session_s=float(hi - lo))
    pairs += pp2

    pd.DataFrame(pairs).to_csv(ROOT / "results" / "joint_staleness_pairs.csv",
                               index=False)
    pd.DataFrame(out["binance"]["rows"] + out["hose"]["rows"]).to_csv(
        ROOT / "results" / "joint_staleness.csv", index=False)

    save("exp12_joint_staleness",
         dict(config=dict(bins_crypto=BINS_CRYPTO, bins_hose=BINS_HOSE,
                          stratum_s_crypto=STRATUM_S_CRYPTO,
                          stratum_s_hose=STRATUM_S_HOSE, n_boot=N_BOOT,
                          p_lo=P_LO, p_hi=P_HI, seed=606), **out),
         started)


if __name__ == "__main__":
    main()
