"""Experiment 11 -- inter-trade duration tails on HOSE (Ho Chi Minh Stock Exchange).

HOSE timestamps have one-second resolution, so consecutive prints frequently tie.
Those ties are the equity-market analogue of Binance's fill bursts: they inject
zero durations that push the Hill threshold deep into the body of the
distribution and bias the tail index downward.  The event definitions are

  print   every print is an event (ties give zero durations)
  second  all prints sharing a timestamp are merged into one event
  W       further merged within a window of W seconds
  deseas  `second` durations divided by a smoothed intraday activity profile

The exchange's own resolution limit means `print` is not observable at higher
frequency, so `second` is the finest event definition the data can support.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import ROOT, save                                      # noqa: E402
from nmf.durations import (burst_aggregate, csn_gof, deseasonalize,  # noqa: E402
                           durations_from_times, hill, hill_stable,
                           running_mean_ratio)

DATA = ROOT / "data" / "hose"
WINDOWS = (2.0, 5.0, 30.0)
CSN_BOOT = 300


def summarize(d, label, rng=None, do_csn=False):
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
    if do_csn and pos.size >= 150:
        g = csn_gof(pos, n_boot=CSN_BOOT, rng=rng, n_xmin=30)
        rec.update(csn_alpha=g["alpha"], csn_xmin=g["xmin"],
                   csn_n_tail=g["n_tail"], csn_p=g["p"])
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
        df = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
        ts = df["time"].astype("int64").to_numpy() / 1e9          # seconds
        if ts.size < 150:
            continue

        recs = [summarize(durations_from_times(ts), "print", rng=rng, do_csn=True)]
        sec = np.unique(ts)
        recs.append(summarize(durations_from_times(sec), "second", rng=rng, do_csn=True))
        for W in WINDOWS:
            recs.append(summarize(durations_from_times(burst_aggregate(sec, W)),
                                  f"W{W:g}"))
        d_sec = durations_from_times(sec)
        if d_sec.size > 300:
            d_adj, prof, _ = deseasonalize(sec[:-1], d_sec, n_bins=24)
            r = summarize(d_adj, "deseas", rng=rng, do_csn=True)
            r["diurnal_range"] = float(np.nanmax(prof) / np.nanmin(prof))
            recs.append(r)

        for r in recs:
            r.update(symbol=sym, n_ticks=meta["n_ticks"],
                     tie_share=meta["zero_gap_share"])
        rows.extend(recs)

        p = [r for r in recs if r["series"] == "print"][0]
        s = [r for r in recs if r["series"] == "second"][0]
        print(f"[{i:2d}] {sym:5s} ticks={meta['n_ticks']:>6,d} ties={meta['zero_gap_share']:.2f} "
              f"med={s['median_gap_s']:>6.1f}s  a_print={p['hill_plateau']:.2f} "
              f"a_sec={s['hill_plateau']:.2f}  rm={s['rm_ratio']:.2f} "
              f"csn_p={s.get('csn_p', float('nan')):.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "hose_durations.csv", index=False)

    cross = {}
    for series in ("print", "second", "deseas"):
        sub = df[(df["series"] == series) & df["hill_plateau"].notna()]
        if len(sub) < 4:
            continue
        x = np.log10(sub["median_gap_s"].values)
        y = sub["hill_plateau"].values
        pr, sr = pearsonr(x, y), spearmanr(x, y)
        cross[series] = dict(
            n_symbols=int(len(sub)),
            corr_loggap_alpha=float(pr[0]), p_loggap=float(pr[1]),
            spearman_loggap=float(sr[0]), p_spearman=float(sr[1]),
            alpha_min=float(y.min()), alpha_max=float(y.max()),
            alpha_median=float(np.median(y)), frac_below_one=float((y < 1).mean()),
            rm_median=float(sub["rm_ratio"].median()),
            rm_min=float(sub["rm_ratio"].min()), rm_max=float(sub["rm_ratio"].max()),
        )
        print(f"\ncross-section [{series}]: n={len(sub)} "
              f"corr(log med gap, alpha)={pr[0]:+.3f} (p={pr[1]:.3f})  "
              f"alpha in [{y.min():.2f},{y.max():.2f}] median {np.median(y):.2f}  "
              f"frac<1={(y<1).mean():.2f}", flush=True)

    save("exp11_hose_durations",
         dict(config=dict(windows=WINDOWS, csn_boot=CSN_BOOT,
                          source=manifest["source"],
                          collected_local=manifest["collected_local"]),
              cross_section=cross, n_rows=int(len(df))),
         started)


if __name__ == "__main__":
    main()
