"""Collect HOSE trade-by-trade records via vnstock (KBS source).

HOSE timestamps have one-second resolution, so a large share of consecutive
prints share a timestamp.  We keep every print and record the tie structure
explicitly: it is the equity-market analogue of the fill-level bursts measured
on Binance, and it bounds what any duration statistic can resolve.

Output: one parquet per symbol in data/hose/, plus a manifest.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("ACCEPT_TC", "tôi đồng ý")

import pandas as pd                                              # noqa: E402
from vnstock import Quote                                        # noqa: E402

OUT = Path(__file__).resolve().parent / "hose"
OUT.mkdir(parents=True, exist_ok=True)

# HOSE tickers spanning roughly three orders of magnitude in typical turnover.
SYMBOLS = [
    # index heavyweights / most traded
    "SSI", "VPB", "HPG", "STB", "MBB", "SHB", "VIX", "TCB", "CTG", "VND",
    # mid caps
    "VCB", "VNM", "FPT", "MWG", "GEX", "DIG", "DXG", "POW", "HSG", "NLG",
    # thinner names
    "PVD", "KDH", "REE", "TCH", "HHV", "FCN", "BCG", "LCG", "ITA", "HAG",
    "SBT", "DCM", "GAS", "MSN", "BID",
]
PAGE = 30_000


def fetch(symbol, tries=3):
    for attempt in range(tries):
        try:
            q = Quote(symbol=symbol, source="kbs")
            df = q.intraday(page_size=PAGE, show_log=False)
            if df is None or len(df) == 0:
                return None
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"])
            df = df.drop_duplicates("id").sort_values("time").reset_index(drop=True)
            return df
        except Exception as exc:                                  # noqa: BLE001
            print(f"    attempt {attempt+1}: {type(exc).__name__}: {str(exc)[:90]}",
                  flush=True)
            time.sleep(20 * (attempt + 1))
    return None


def main():
    manifest = []
    for i, sym in enumerate(SYMBOLS, 1):
        path = OUT / f"{sym}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            print(f"[{i}/{len(SYMBOLS)}] {sym} cached ({len(df):,})", flush=True)
        else:
            print(f"[{i}/{len(SYMBOLS)}] {sym} ...", end=" ", flush=True)
            df = fetch(sym)
            if df is None or len(df) < 100:
                print("too few ticks, skipped", flush=True)
                continue
            df.to_parquet(path, index=False)
            print(f"{len(df):,} ticks", flush=True)
            time.sleep(5.0)          # guest tier allows 20 requests/minute
        secs = df["time"].astype("int64") // 10**9
        gaps = secs.diff().dropna()
        manifest.append(dict(
            symbol=sym, n_ticks=int(len(df)),
            session_start=str(df["time"].iloc[0]), session_end=str(df["time"].iloc[-1]),
            zero_gap_share=float((gaps == 0).mean()),
            median_gap_s=float(gaps[gaps > 0].median()) if (gaps > 0).any() else None,
        ))
    (OUT / "manifest.json").write_text(json.dumps(
        dict(collected_local=pd.Timestamp.now().isoformat(),
             source="vnstock/kbs", symbols=manifest), indent=1))
    print(f"\ndone: {len(manifest)} symbols", flush=True)


if __name__ == "__main__":
    sys.exit(main())
