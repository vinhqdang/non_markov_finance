"""Shared helpers for the simulation experiments."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))


def chunk_sizes(total: int, chunk: int):
    """Split `total` into pieces of at most `chunk`."""
    out = []
    while total > 0:
        out.append(min(chunk, total))
        total -= chunk
    return out


def save(name: str, payload: dict, started: float):
    payload = dict(payload)
    payload["_meta"] = dict(
        experiment=name,
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        runtime_s=round(time.time() - started, 1),
        python=sys.version.split()[0],
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
    )
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=_default))
    print(f"\n[saved] {path}  ({payload['_meta']['runtime_s']}s)", flush=True)
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def se_prop(p, n):
    """Standard error of a proportion."""
    return float(np.sqrt(max(p * (1 - p), 0.0) / max(n, 1)))
