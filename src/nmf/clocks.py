"""Exact simulation of staleness clocks built from stable subordinators.

The clock vector is  S_j(u) = X_j(u) + a_j Z(u),  j = 1..k,  where X_1..X_k and
Z are independent standard stable subordinators (Laplace exponent lambda^alpha).

Jumps are drawn by the inverse-Levy-measure (Rosinski) series representation, so
the sample path is a *step function known exactly at its jump points* -- there is
no time discretization anywhere.  Consequently the first-passage time

    L_j(t) = inf{u : S_j(u) > t}

and the undershoot

    H_j(t) = S_j(L_j(t)-)

are computed exactly (up to series truncation, which is reported).

Series representation
---------------------
For a subordinator with Levy measure Pi on (0, inf), the jumps on the
operational interval [0, U] are

    {(V_i, Pibar^{-1}(Gamma_i / U))}_{i>=1},
    Gamma_i = E_1 + ... + E_i  (unit exponentials),   V_i ~ iid U(0, U).

For the standard alpha-stable subordinator,
    Pi(dz) = alpha / Gamma(1-alpha) * z^{-1-alpha} dz,
    Pibar(z) = z^{-alpha} / Gamma(1-alpha),
    Pibar^{-1}(y) = (y * Gamma(1-alpha))^{-1/alpha},
so the i-th largest jump on [0, U] is

    J_i = (U / (Gamma_i * Gamma(1-alpha)))^{1/alpha}.

Truncating after N terms omits a total mass with expectation

    tail_mass(N) ~ (U / Gamma(1-alpha))^{1/alpha} * alpha/(1-alpha) * N^{-(1-alpha)/alpha},

which every routine here returns as a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as _gamma


# ---------------------------------------------------------------------------
# Series-representation primitives
# ---------------------------------------------------------------------------


def truncation_mass(alpha: float, horizon: float, n_jumps: int) -> float:
    """Expected total size of the jumps omitted by truncating the series."""
    c = (horizon / _gamma(1.0 - alpha)) ** (1.0 / alpha)
    return c * alpha / (1.0 - alpha) * n_jumps ** (-(1.0 - alpha) / alpha)


def required_jumps(alpha: float, horizon: float, tol: float) -> int:
    """Smallest N whose truncation mass is below `tol`."""
    c = (horizon / _gamma(1.0 - alpha)) ** (1.0 / alpha) * alpha / (1.0 - alpha)
    return int(np.ceil((c / tol) ** (alpha / (1.0 - alpha))))


def stable_jumps(alpha, horizon, n_paths, n_jumps, rng):
    """Jump times and sizes of `n_paths` independent standard stable subordinators.

    Returns
    -------
    times : (n_paths, n_jumps) float array, iid Uniform(0, horizon), unsorted.
    sizes : (n_paths, n_jumps) float array, decreasing along axis 1.
    """
    e = rng.exponential(size=(n_paths, n_jumps))
    gam = np.cumsum(e, axis=1)
    sizes = (horizon / (gam * _gamma(1.0 - alpha))) ** (1.0 / alpha)
    times = rng.uniform(0.0, horizon, size=(n_paths, n_jumps))
    return times, sizes


# ---------------------------------------------------------------------------
# Common-shock clock vector
# ---------------------------------------------------------------------------


@dataclass
class ClockSpec:
    """S_j = X_j + a_j Z, with X_j standard alpha_j-stable and Z standard alpha_c-stable."""

    alphas: tuple          # (alpha_1, ..., alpha_k) for the idiosyncratic parts
    alpha_c: float         # index of the common shock Z
    loadings: tuple        # (a_1, ..., a_k), a_j >= 0

    @property
    def k(self) -> int:
        return len(self.alphas)


@dataclass
class ClockPaths:
    """Merged event representation of `n_paths` realizations of the clock vector.

    For each path the events (jumps of any component) are sorted in operational
    time.  `cum[p, j, i]` is S_j just after event i on path p.
    """

    times: np.ndarray      # (n_paths, n_events) sorted operational times
    cum: np.ndarray        # (n_paths, k, n_events) cumulative clock values
    horizon: float
    trunc_mass: float      # worst-case expected omitted mass across components


def simulate_clocks(spec: ClockSpec, horizon, n_paths, n_jumps, rng) -> ClockPaths:
    """Simulate the common-shock clock vector by exact jump enumeration."""
    k = spec.k
    blocks_t, blocks_s = [], []
    trunc = 0.0

    # idiosyncratic components: X_j contributes only to clock j
    for j in range(k):
        t, s = stable_jumps(spec.alphas[j], horizon, n_paths, n_jumps, rng)
        full = np.zeros((n_paths, k, n_jumps))
        full[:, j, :] = s
        blocks_t.append(t)
        blocks_s.append(full)
        trunc = max(trunc, truncation_mass(spec.alphas[j], horizon, n_jumps))

    # common shock: Z contributes a_j * dZ to every clock j
    if any(a > 0 for a in spec.loadings):
        t, s = stable_jumps(spec.alpha_c, horizon, n_paths, n_jumps, rng)
        full = np.empty((n_paths, k, n_jumps))
        for j in range(k):
            full[:, j, :] = spec.loadings[j] * s
        blocks_t.append(t)
        blocks_s.append(full)
        trunc = max(trunc, truncation_mass(spec.alpha_c, horizon, n_jumps)
                    * max(spec.loadings))

    times = np.concatenate(blocks_t, axis=1)                 # (P, E)
    sizes = np.concatenate(blocks_s, axis=2)                 # (P, k, E)

    order = np.argsort(times, axis=1, kind="stable")
    times = np.take_along_axis(times, order, axis=1)
    sizes = np.take_along_axis(sizes, order[:, None, :], axis=2)
    cum = np.cumsum(sizes, axis=2)
    return ClockPaths(times=times, cum=cum, horizon=horizon, trunc_mass=trunc)


# ---------------------------------------------------------------------------
# First passage, undershoot, simultaneity
# ---------------------------------------------------------------------------


def crossing_index(paths: ClockPaths, t: float) -> np.ndarray:
    """Index of the event at which each clock first exceeds level t.

    Returns an (n_paths, k) integer array.  Entries equal to n_events mean the
    clock never reached t within the operational horizon (caller must check).
    """
    return (paths.cum <= t).sum(axis=2)


def undershoot(paths: ClockPaths, t: float):
    """Exact undershoot H_j(t) = S_j(L_j(t)-) and the crossing indices.

    Returns
    -------
    H    : (n_paths, k) float array
    idx  : (n_paths, k) int array of crossing event indices
    ok   : (n_paths,) bool array, True where every clock crossed t in time
    """
    idx = crossing_index(paths, t)
    n_events = paths.cum.shape[2]
    ok = (idx < n_events).all(axis=1)
    safe = np.minimum(idx, n_events - 1)
    # value just before the crossing event; 0 if the crossing is the first event
    prev = np.maximum(safe - 1, 0)
    H = np.take_along_axis(paths.cum, prev[:, :, None], axis=2)[:, :, 0]
    H = np.where(safe == 0, 0.0, H)
    return H, idx, ok


def first_passage_time(paths: ClockPaths, t: float) -> np.ndarray:
    """Operational first-passage time L_j(t), as an (n_paths, k) float array."""
    idx = crossing_index(paths, t)
    n_events = paths.times.shape[1]
    safe = np.minimum(idx, n_events - 1)
    return np.take_along_axis(paths.times, safe, axis=1) if safe.ndim == 2 else None


def simultaneity(paths: ClockPaths, t: float, j: int = 0, h: int = 1):
    """Indicator of the event {L_j(t) = L_h(t)} -- simultaneous staleness.

    Two clocks cross level t at the same operational time iff they cross at the
    same *event*, which (a.s.) requires a common jump of Z.  This is exact: no
    tolerance and no time grid are involved.
    """
    idx = crossing_index(paths, t)
    n_events = paths.cum.shape[2]
    ok = (idx < n_events).all(axis=1)
    return (idx[:, j] == idx[:, h]), ok


# ---------------------------------------------------------------------------
# Bivariate potential (renewal) measure
# ---------------------------------------------------------------------------


def potential_histogram(paths: ClockPaths, edges_x, edges_y, j: int = 0, h: int = 1):
    """Monte Carlo estimate of the bivariate potential measure

        U(B) = int_0^inf P[(S_j(u), S_h(u)) in B] du

    accumulated on the rectangular grid `edges_x` x `edges_y`.

    Because the path is a step function, the occupation integral is exact: the
    pair sits at (cum[j, i], cum[h, i]) for exactly (times[i+1] - times[i]) of
    operational time.  Only the horizon-truncated tail is approximated, and it
    is excluded rather than extrapolated.
    """
    t = paths.times
    dwell = np.diff(t, axis=1)                       # (P, E-1)
    x = paths.cum[:, j, :-1]
    y = paths.cum[:, h, :-1]
    inside = (
        (x >= edges_x[0]) & (x < edges_x[-1]) & (y >= edges_y[0]) & (y < edges_y[-1])
    )
    hist, _, _ = np.histogram2d(
        x[inside].ravel(), y[inside].ravel(),
        bins=[edges_x, edges_y], weights=dwell[inside].ravel(),
    )
    n_paths = t.shape[0]
    cell_area = np.outer(np.diff(edges_x), np.diff(edges_y))
    density = hist / (n_paths * cell_area)
    return density, hist / n_paths
