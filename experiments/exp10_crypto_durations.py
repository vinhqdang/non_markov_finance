"""Experiment 10 -- inter-trade duration tails on Binance spot.

The model requires the duration tail index alpha to lie in (0,1).  We estimate
it under four different *event definitions* on the same download, because the
choice of event definition is exactly what the raw-data literature leaves
implicit:

  fill    every raw fill is an event (all fills of one marketable order share a
          timestamp, so this series carries n_fills - 1 zero durations per order)
  order   every aggregated trade is an event -- the economically correct unit
  W       order-level events further merged within a window W seconds
  deseas  order-level durations divided by a smoothed diurnal activity profile

For each series we report the Hill index (fixed fraction and Hill-plot plateau),
the assumption-free running-mean ratio, and the Clauset-Shalizi-Newman
goodness-of-fit p-value for the power-law hypothesis itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, save                                     # noqa: E402
from nmf.durations import (burst_aggregate, csn_gof, deseasonalize,  # noqa: E402
                           durations_from_times, hill, hill_stable,
                           running_mean_ratio)

DATA = ROOT / "data" / "binance"
WINDOWS = (0.1, 1.0, 5.0, 30.0)
CSN_CAP = 20_000
CSN_BOOT = 200


def expand_fills(ts_s, n_fills):
    """Fill-level timestamps: each order contributes n_fills events at its timestamp."""
    return np.repeat(ts_s, n_fills)


def summarize(d, label, times=None, rng=None, do_csn=False):
    a_f, se_f, k_f = hill(d, frac=0.10)
    a_s, se_s, k_s = hill_stable(d)
    pos = d[d > 0]
    rec = dict(
        series=label, n=int(d.size), n_pos=int(pos.size),
        zero_share=float((d == 0).mean()) if d.size else np.nan,
        median_gap_s=float(np.median(pos)) if pos.size else np.nan,
        mean_gap_s=float(d.mean()) if d.size else np.nan,
        hill_frac10=a_f, hill_frac10_se=se_f, hill_frac10_k=k_f,
        hill_plateau=a_s, hill_plateau_se=se_s, hill_plateau_k=k_s,
        rm_ratio=running_mean_ratio(d),
    )
    if do_csn and pos.size >= 200:
        x = pos
        if x.size > CSN_CAP:
            x = rng.choice(x, CSN_CAP, replace=False)
        g = csn_gof(x, n_boot=CSN_BOOT, rng=rng, n_xmin=30)
        rec.update(csn_alpha=g["alpha"], csn_xmin=g["xmin"],
                   csn_n_tail=g["n_tail"], csn_p=g["p"], csn_n_used=int(x.size))
    return rec


def main():
    started = time.time()
    rng = np.random.default_rng(20260817)
    manifest = json.loads((DATA / "manifest.json").read_text())
    rows = []

    for i, meta in enumerate(manifest["symbols"], 1):
        sym = meta["symbol"]
        path = DATA / f"{sym}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).sort_values("ts_ms").reset_index(drop=True)
        ts = df["ts_ms"].to_numpy(dtype=float) / 1000.0
        nf = df["n_fills"].to_numpy(dtype=np.int64)
        t0 = time.time()

        recs = []
        # fill level
        d_fill = durations_from_times(expand_fills(ts, nf))
        recs.append(summarize(d_fill, "fill", rng=rng, do_csn=True))
        # order level
        d_ord = durations_from_times(ts)
        recs.append(summarize(d_ord, "order", rng=rng, do_csn=True))
        # burst aggregation on top of order level
        for W in WINDOWS:
            recs.append(summarize(durations_from_times(burst_aggregate(ts, W)),
                                  f"W{W:g}"))
        # diurnally adjusted order level
        if d_ord.size > 500:
            d_adj, prof, _ = deseasonalize(ts[:-1], d_ord, n_bins=48)
            r = summarize(d_adj, "deseas", rng=rng, do_csn=True)
            r["diurnal_range"] = float(np.nanmax(prof) / np.nanmin(prof))
            recs.append(r)

        for r in recs:
            r.update(symbol=sym, qvol_24h=meta["qvol_24h"],
                     n_agg=meta["n_agg"], n_fills_total=meta["n_fills"],
                     fills_per_order=meta["n_fills"] / max(meta["n_agg"], 1))
        rows.extend(recs)

        o = [r for r in recs if r["series"] == "order"][0]
        f = [r for r in recs if r["series"] == "fill"][0]
        print(f"[{i:2d}] {sym:11s} qvol={meta['qvol_24h']:>13,.0f} n={o['n']:>7,d} "
              f"med={o['median_gap_s']:>7.3f}s  a_fill={f['hill_plateau']:.2f} "
              f"a_ord={o['hill_plateau']:.2f}  rm={o['rm_ratio']:.2f} "
              f"csn_p={o.get('csn_p', float('nan')):.2f}  ({time.time()-t0:.0f}s)",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "crypto_durations.csv", index=False)

    # ---- cross-section: does the tail index fall with illiquidity? -----------
    cross = {}
    for series in ("fill", "order", "deseas"):
        sub = df[(df["series"] == series) & df["hill_plateau"].notna()]
        if len(sub) < 4:
            continue
        x = np.log10(sub["median_gap_s"].values)
        y = sub["hill_plateau"].values
        from scipy.stats import pearsonr, spearmanr
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        xv = np.log10(sub["qvol_24h"].values)
        pv = pearsonr(xv, y)
        cross[series] = dict(
            n_symbols=int(len(sub)),
            corr_loggap_alpha=float(pr[0]), p_loggap=float(pr[1]),
            spearman_loggap=float(sr[0]), p_spearman=float(sr[1]),
            corr_logvol_alpha=float(pv[0]), p_logvol=float(pv[1]),
            alpha_min=float(y.min()), alpha_max=float(y.max()),
            alpha_median=float(np.median(y)),
            frac_below_one=float((y < 1).mean()),
        )
        print(f"\ncross-section [{series}]: n={len(sub)} "
              f"corr(log med gap, alpha)={pr[0]:+.3f} (p={pr[1]:.3f})  "
              f"alpha in [{y.min():.2f},{y.max():.2f}] median {np.median(y):.2f}  "
              f"frac<1 = {(y<1).mean():.2f}", flush=True)

    save("exp10_crypto_durations",
         dict(config=dict(windows=WINDOWS, csn_cap=CSN_CAP, csn_boot=CSN_BOOT,
                          window_start_ms=manifest["window_start_ms"],
                          collected_utc=manifest["collected_utc"]),
              cross_section=cross,
              n_rows=int(len(df))),
         started)


if __name__ == "__main__":
    main()
