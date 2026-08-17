"""Estimators for inter-trade duration tails.

The model of the paper needs the duration tail index alpha to lie in (0, 1),
i.e. an infinite mean waiting time.  This module implements the tools used to
test that, ordered from most to least assumption-heavy:

  * `hill`                  -- tail index at a chosen threshold, with s.e.
  * `hill_stable`           -- Hill estimate on a plateau of the Hill plot
  * `running_mean_ratio`    -- assumption-free finite/infinite mean diagnostic
  * `burst_aggregate`       -- merge fills of one order into a single event
  * `deseasonalize`         -- remove the diurnal activity cycle
  * `csn_fit` / `csn_gof`   -- Clauset-Shalizi-Newman fit and goodness of fit

`deseasonalize` matters more than it looks.  A Poisson process whose rate varies
over the day produces a duration distribution that is a *mixture* of
exponentials, and mixtures of exponentials have heavier-than-exponential tails.
Any tail index estimated on a full trading day therefore confounds genuine tail
behaviour with the diurnal cycle, and must be checked against the
diurnally-adjusted series.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Tail index
# ---------------------------------------------------------------------------


def _clean(x):
    """Keep every finite non-negative observation, zeros included.

    Zero durations are real data: they are what a burst of fills at one
    timestamp produces.  Discarding them would hide the very mechanism this
    module is meant to measure, because the Hill threshold k is a fraction of
    the sample size n -- more zeros means a larger n, a deeper k, and hence a
    downward-biased alpha.
    """
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x) & (x >= 0)]


def hill(x, k=None, frac=0.10):
    """Hill estimator of the tail index alpha from the top `k` order statistics.

    Returns (alpha_hat, se, k).  The asymptotic standard error is alpha/sqrt(k).
    Zeros contribute to n (and therefore to the choice of k) but can never enter
    the top k; if the k-th order statistic is zero, k is reduced accordingly.
    """
    x = _clean(x)
    n = x.size
    if n < 20:
        return np.nan, np.nan, 0
    if k is None:
        k = max(10, int(round(frac * n)))
    xs = np.sort(x)[::-1]
    n_pos = int((xs > 0).sum())
    k = min(k, n - 1, max(n_pos - 1, 0))
    if k < 10 or xs[k] <= 0:
        return np.nan, np.nan, int(k)
    logs = np.log(xs[:k]) - np.log(xs[k])
    inv = logs.mean()
    if inv <= 0:
        return np.nan, np.nan, int(k)
    alpha = 1.0 / inv
    return float(alpha), float(alpha / np.sqrt(k)), int(k)


def hill_curve(x, k_min=15, k_max_frac=0.35, n_pts=60):
    """Hill estimate as a function of k, for plotting and plateau selection."""
    x = _clean(x)
    n = x.size
    if n < 50:
        return np.array([]), np.array([])
    xs = np.sort(x)[::-1]
    n_pos = int((xs > 0).sum())
    k_max = min(int(k_max_frac * n), n - 1, max(n_pos - 1, 0))
    if k_max <= k_min:
        return np.array([]), np.array([])
    ks = np.unique(np.linspace(k_min, k_max, n_pts).astype(int))
    logx = np.log(np.maximum(xs, 1e-300))
    csum = np.cumsum(logx)
    out = []
    for k in ks:
        inv = (csum[k - 1] - k * logx[k]) / k
        out.append(1.0 / inv if inv > 0 else np.nan)
    return ks, np.array(out)


def hill_stable(x, k_min=15, k_max_frac=0.35, window=9):
    """Hill estimate at the flattest window of the Hill plot.

    Chooses the window of `window` consecutive k values minimizing the standard
    deviation of alpha_hat, and reports the median there.  This removes the
    arbitrary single-threshold choice.
    """
    ks, al = hill_curve(x, k_min, k_max_frac)
    if ks.size < window:
        return hill(x)
    best, best_sd, best_k = np.nan, np.inf, 0
    for i in range(len(ks) - window + 1):
        seg = al[i:i + window]
        if not np.all(np.isfinite(seg)):
            continue
        sd = float(np.std(seg))
        if sd < best_sd:
            best_sd, best, best_k = sd, float(np.median(seg)), int(np.median(ks[i:i + window]))
    if not np.isfinite(best):
        return hill(x)
    return best, float(best / np.sqrt(max(best_k, 1))), best_k


# ---------------------------------------------------------------------------
# Assumption-free mean diagnostic
# ---------------------------------------------------------------------------


def running_mean_ratio(x, n_blocks=2):
    """Ratio of the sample mean over the last block to that over the first.

    For a finite-mean distribution this tends to 1.  For an infinite-mean
    (alpha < 1) distribution the running mean grows without bound, so the ratio
    drifts upward with sample size.  No threshold and no parametric assumption.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size < 4 * n_blocks:
        return np.nan
    parts = np.array_split(x, n_blocks)
    m0, m1 = parts[0].mean(), parts[-1].mean()
    return float(m1 / m0) if m0 > 0 else np.nan


def running_mean_path(x, n_pts=200):
    """Running mean as a function of sample size, normalized by the final mean."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size < 50:
        return np.array([]), np.array([])
    idx = np.unique(np.linspace(20, x.size, n_pts).astype(int))
    cs = np.cumsum(x)
    means = cs[idx - 1] / idx
    return idx, means / means[-1]


# ---------------------------------------------------------------------------
# Event definition: bursts and seasonality
# ---------------------------------------------------------------------------


def burst_aggregate(times, window):
    """Merge events whose timestamps differ by less than `window` into one event.

    `times` must be sorted and in the same units as `window`.  Returns the
    timestamps of the merged events (the first timestamp of each burst).
    """
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return t
    if window <= 0:
        return t
    keep = np.ones(t.size, dtype=bool)
    last = t[0]
    for i in range(1, t.size):
        if t[i] - last < window:
            keep[i] = False
        else:
            last = t[i]
    return t[keep]


def durations_from_times(times, drop_zero=False):
    """Inter-event durations from sorted timestamps."""
    t = np.asarray(times, dtype=float)
    d = np.diff(t)
    d = d[np.isfinite(d) & (d >= 0)]
    return d[d > 0] if drop_zero else d


def deseasonalize(times, durations, n_bins=48, min_per_bin=20):
    """Divide durations by a smoothed diurnal profile of the local mean duration.

    `times` are the event timestamps (same units as `durations`), aligned so that
    `times[i]` is the start of `durations[i]`.  The profile is the bin-wise mean
    duration over the observation window, smoothed by a 3-bin moving average and
    normalized to mean one.  Returns (adjusted_durations, profile, bin_edges).
    """
    t = np.asarray(times, dtype=float)
    d = np.asarray(durations, dtype=float)
    n = min(t.size, d.size)
    t, d = t[:n], d[:n]
    if n < n_bins * min_per_bin:
        n_bins = max(4, n // max(min_per_bin, 1))
    edges = np.linspace(t.min(), t.max() + 1e-9, n_bins + 1)
    which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_bins - 1)
    prof = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = d[which == b]
        if sel.size >= max(5, min_per_bin // 4):
            prof[b] = sel.mean()
    # fill gaps, then smooth with a 3-bin moving average
    ok = np.isfinite(prof)
    if ok.sum() < 3:
        return d.copy(), prof, edges
    prof = np.interp(np.arange(n_bins), np.flatnonzero(ok), prof[ok])
    ker = np.ones(3) / 3.0
    prof = np.convolve(np.pad(prof, 1, mode="edge"), ker, mode="valid")
    prof = prof / prof.mean()
    return d / prof[which], prof, edges


# ---------------------------------------------------------------------------
# Clauset-Shalizi-Newman power-law fit and goodness of fit
# ---------------------------------------------------------------------------


def _pl_mle(x, xmin):
    """Continuous power-law MLE: P(X>x) ~ (x/xmin)^{-alpha}."""
    tail = x[x >= xmin]
    n = tail.size
    if n < 10:
        return np.nan, 0
    s = np.log(tail / xmin).sum()
    if s <= 0:
        return np.nan, n
    return float(n / s), int(n)


def _ks_distance(x, xmin, alpha):
    tail = np.sort(x[x >= xmin])
    n = tail.size
    if n < 2 or not np.isfinite(alpha):
        return np.inf
    emp = np.arange(1, n + 1) / n
    theo = 1.0 - (tail / xmin) ** (-alpha)
    return float(np.max(np.abs(emp - theo)))


def csn_fit(x, n_xmin=60):
    """Fit xmin by KS minimization and alpha by MLE above it."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 50:
        return dict(xmin=np.nan, alpha=np.nan, n_tail=0, ks=np.nan)
    cands = np.unique(np.quantile(x, np.linspace(0.05, 0.95, n_xmin)))
    best = dict(xmin=np.nan, alpha=np.nan, n_tail=0, ks=np.inf)
    for xm in cands:
        a, n = _pl_mle(x, xm)
        if not np.isfinite(a) or n < 30:
            continue
        d = _ks_distance(x, xm, a)
        if d < best["ks"]:
            best = dict(xmin=float(xm), alpha=float(a), n_tail=int(n), ks=float(d))
    return best


def csn_gof(x, n_boot=300, rng=None, n_xmin=40):
    """Parametric-bootstrap p-value for the power-law hypothesis.

    Small p means the power law is rejected.  Following Clauset-Shalizi-Newman,
    each synthetic sample draws the body by resampling the empirical
    below-xmin data and the tail from the fitted power law, then is refitted
    from scratch.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    fit = csn_fit(x, n_xmin=n_xmin)
    if not np.isfinite(fit["ks"]) or fit["n_tail"] < 30:
        return dict(**fit, p=np.nan, n_boot=0)
    n = x.size
    body = x[x < fit["xmin"]]
    p_tail = fit["n_tail"] / n
    worse = 0
    for _ in range(n_boot):
        n_t = rng.binomial(n, p_tail)
        n_b = n - n_t
        syn_tail = fit["xmin"] * (1.0 - rng.random(n_t)) ** (-1.0 / fit["alpha"])
        syn_body = rng.choice(body, size=n_b, replace=True) if body.size else np.empty(0)
        syn = np.concatenate([syn_body, syn_tail])
        f2 = csn_fit(syn, n_xmin=n_xmin)
        if np.isfinite(f2["ks"]) and f2["ks"] >= fit["ks"]:
            worse += 1
    return dict(**fit, p=float(worse / n_boot), n_boot=int(n_boot))
