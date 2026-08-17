"""Collect Binance spot aggregated-trade records across the liquidity spectrum.

Why aggTrades rather than raw trades
------------------------------------
Binance's aggTrades endpoint merges consecutive fills that belong to a single
marketable order executed at one price.  Each record carries `f` and `l`, the
first and last raw trade ids, so `l - f + 1` is the number of raw fills behind
the record.  That gives us both event definitions from one download:

  * the *order-level* duration series  (gaps between aggTrade timestamps), and
  * the *fill-level* duration series   (each aggTrade contributes l-f extra
    zero-length gaps, because all fills of one order share a timestamp).

The difference between the two is precisely the "burst" artifact that inflates
apparent tail heaviness, and here it is measured rather than assumed.

Output: one parquet per symbol in data/binance/, plus a manifest.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://api.binance.com"
OUT = Path(__file__).resolve().parent / "binance"
OUT.mkdir(parents=True, exist_ok=True)

N_SYMBOLS = 24
MAX_TRADES = 300_000          # cap per symbol
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "academic-research/1.0"})


def get(path, params, tries=6):
    for attempt in range(tries):
        try:
            r = SESSION.get(BASE + path, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                time.sleep(5 * (attempt + 1))
                continue
            print(f"    HTTP {r.status_code}: {r.text[:120]}", flush=True)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    {type(exc).__name__}: {exc}", flush=True)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed: {path} {params}")


def choose_symbols():
    """Log-uniform sample of USDT pairs across the 24h quote-volume spectrum."""
    tick = get("/api/v3/ticker/24hr", {})
    rows = [
        dict(symbol=t["symbol"], qvol=float(t["quoteVolume"]), count=int(t["count"]))
        for t in tick
        if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 0
        and int(t["count"]) > 150
        and not any(t["symbol"].startswith(p) for p in ("USDC", "FDUSD", "TUSD", "EUR", "AEUR"))
    ]
    df = pd.DataFrame(rows).sort_values("qvol", ascending=False).reset_index(drop=True)
    lv = np.log10(df["qvol"].values)
    targets = np.linspace(lv.max(), lv.min(), N_SYMBOLS)
    picks, used = [], set()
    for tg in targets:
        i = int(np.argmin(np.abs(lv - tg)))
        while i in used:
            i += 1
            if i >= len(df):
                i = 0
        used.add(i)
        picks.append(df.iloc[i])
    return pd.DataFrame(picks).reset_index(drop=True)


def fetch_symbol(symbol, start_ms, end_ms):
    rows, from_id, cursor = [], None, start_ms
    while True:
        params = {"symbol": symbol, "limit": 1000}
        if from_id is None:
            params.update(startTime=cursor, endTime=min(cursor + 3_600_000, end_ms))
        else:
            params.update(fromId=from_id)
        batch = get("/api/v3/aggTrades", params)
        if not batch:
            if from_id is not None:
                break
            cursor += 3_600_000
            if cursor >= end_ms:
                break
            continue
        batch = [b for b in batch if b["T"] < end_ms]
        if not batch:
            break
        rows.extend(batch)
        from_id = batch[-1]["a"] + 1
        if batch[-1]["T"] >= end_ms - 1 or len(rows) >= MAX_TRADES:
            break
        time.sleep(0.06)
    if not rows:
        return None
    df = pd.DataFrame(rows)[["a", "p", "q", "f", "l", "T", "m"]]
    df.columns = ["agg_id", "price", "qty", "first_id", "last_id", "ts_ms", "buyer_maker"]
    for c in ("price", "qty"):
        df[c] = df[c].astype(float)
    df["n_fills"] = df["last_id"] - df["first_id"] + 1
    return df.drop_duplicates("agg_id").sort_values("ts_ms").reset_index(drop=True)


def main():
    day = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=2)
    start_ms = int(day.timestamp() * 1000)
    end_ms = start_ms + 24 * 3600 * 1000
    print(f"window: {day.date()} 00:00 UTC + 24h", flush=True)

    universe = choose_symbols()
    manifest = []
    for i, row in universe.iterrows():
        sym = row["symbol"]
        path = OUT / f"{sym}.parquet"
        if path.exists():
            print(f"[{i+1}/{len(universe)}] {sym} cached", flush=True)
            df = pd.read_parquet(path)
        else:
            print(f"[{i+1}/{len(universe)}] {sym} qvol={row['qvol']:,.0f} ...",
                  end=" ", flush=True)
            t0 = time.time()
            df = fetch_symbol(sym, start_ms, end_ms)
            if df is None or len(df) < 150:
                print("too few trades, skipped", flush=True)
                continue
            df.to_parquet(path, index=False)
            print(f"{len(df):,} aggTrades, {df['n_fills'].sum():,} fills, "
                  f"{time.time()-t0:.0f}s", flush=True)
        manifest.append(dict(
            symbol=sym, qvol_24h=float(row["qvol"]), n_agg=int(len(df)),
            n_fills=int(df["n_fills"].sum()),
            span_s=float((df["ts_ms"].iloc[-1] - df["ts_ms"].iloc[0]) / 1000),
        ))
    (OUT / "manifest.json").write_text(json.dumps(
        dict(window_start_ms=start_ms, window_end_ms=end_ms,
             collected_utc=pd.Timestamp.utcnow().isoformat(), symbols=manifest),
        indent=1))
    print(f"\ndone: {len(manifest)} symbols", flush=True)


if __name__ == "__main__":
    sys.exit(main())
