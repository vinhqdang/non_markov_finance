# Coupled staleness clocks in factor models

Replication package for the manuscript *Coupled Staleness Clocks in Factor
Models: Multivariate Inverse Subordination and Its Empirical Boundary*, prepared
for the IJTAF special issue on non-Markovian dynamics in finance.

Every number, table and figure in `paper/main.pdf` is generated from the code
here. Nothing is transcribed by hand: `experiments/make_tables.py` writes
`paper/results_macros.tex`, and the manuscript reads its numbers from there.

## Layout

```
src/nmf/clocks.py       exact (non-discretized) simulation of the clock vector
src/nmf/theory.py       closed forms: potential density, atom, Kummer transform
src/nmf/durations.py    Hill, running-mean, burst aggregation, deseasonalizing, CSN
data/fetch_binance.py   Binance spot aggTrades collector (order + fill level)
data/fetch_hose.py      HOSE tick collector via vnstock
experiments/exp01..06   theory experiments
experiments/exp10,11    empirical experiments (crypto, HOSE)
experiments/make_tables.py, make_figures.py   results -> LaTeX / PDF
results/                JSON + CSV outputs, one file per experiment
paper/                  manuscript, generated macros and tables
```

## Build

```
make paper      # regenerate macros + figures, compile paper/main.pdf
make theory     # re-run all simulations (~40 min on 10 cores)
make data       # re-collect both data sets (network required)
make empirics   # re-run the duration analyses
make all
```

Python: conda env `py313` (numpy, scipy, pandas, pyarrow, matplotlib, requests,
vnstock). LaTeX: any TeX Live with the bundled `ws-ijtaf.cls`.

## Method notes worth knowing

**The simulator never discretizes time.** Clock paths are built by enumerating
jumps through the inverse-Lévy-measure series, so the undershoot `H(t)` and the
first-passage time `L(t)` are exact. The simultaneity event `{L_1 = L_2}` is
decided by comparing *event indices*, which is why the atom in Section 3.4
cannot be a discretization artifact.

**Series truncation is chosen from measured bias, not from the bound.** The
analytic bound on the omitted jump mass is over `[0, U]`, but what matters is the
mass omitted over `[0, L(t)]`, which is far smaller. `exp01` measures the
realized bias against the exactly known Beta marginal and picks `N` from that;
the bound overstates the requirement by roughly a factor of four.

**Zero durations are kept.** They count toward `n` and therefore toward the Hill
threshold `k = frac * n`. That is exactly the channel through which fill bursts
bias the tail index downward, so discarding them would hide the effect being
measured.

**Correlation standard errors are bootstrapped.** The conditional variances are
heavy-tailed, so the usual `(1-r^2)/sqrt(n-3)` is invalid and inflates
z-statistics by a factor of several.

## Corrections to earlier working notes

Three claims carried in the project's earlier planning notes did not survive
being recomputed, and the manuscript reports the corrected versions:

1. **`rho(0+)` is not mean-reversion-free in general.** It is exactly free — and
   the correlation term structure exactly flat at *every* horizon — when the
   factors are driftless Brownian motions. With Ornstein–Uhlenbeck factors the
   closed form requires `kappa * t << 1`; the relevant scale is `kappa * t`, not
   `kappa * Delta`, so the approximation does *not* improve as the horizon
   shrinks. See `exp06` and Table 5.2.
2. **The slope-sign rule is refuted.** The conjecture that the initial slope of
   the correlation term structure is set by `alpha_j kappa_j` versus
   `alpha_c kappa_0` classifies 52.6% of 139,968 parameter cells — chance.
3. **The diagonal discrepancy in the potential density is a quadrature
   failure, not a derivation error.** Splitting each diagonal cell along
   `s_1 = s_2` and absorbing the logarithmic ridge moves the closed-form /
   simulation ratio from a median of 1.13 to 1.02. See `exp03`.

The empirical picture also changed on recomputation. The apparent sub-unit tail
indices are driven by counting individual *fills* as events and by the diurnal
activity cycle, and the effect is strongest in the most liquid pairs — not, as
previously supposed, in illiquid ones.

## Data

- **Binance**: 21 USDT spot pairs, one full UTC day, 1,332,610 aggregated trades
  built from 3,732,027 fills, millisecond timestamps. Each record carries the
  first and last raw trade id, so both event definitions come from one download.
- **HOSE**: 30 symbols, one continuous morning session, 28,696 prints,
  one-second timestamp resolution.

Both samples are archived under `data/` because the public endpoints do not
serve deep history. The two material limitations are stated in the manuscript:
the HOSE session is short, and its one-second resolution caps the finest event
definition available in that market.

## Before submission

- Insert the ORCID iD (marked in red in `paper/main.tex`).
- Verify every DOI in the bibliography against Crossref.
- Confirm the IJTAF APC waiver policy.
