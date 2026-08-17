# Project handoff: IJTAF paper

**Author:** Quang-Vinh Dang (BUV / SmartOSC)
**Date of handoff:** 17 Aug 2026
**Purpose:** pick up work toward a submission to IJTAF. Read §1 and §2 before proposing anything.

---

## 1. Read this first: a direction already died here

Three weeks of work went into a direction that was killed by data. Do **not** resurrect it.
It is documented here so it is not re-litigated, and because the negative results are real
findings worth keeping.

### 1.1 The dead direction

Target was the IJTAF Special Issue on **"Non-Markovian Dynamics in Finance: Factor Modelling
and Beyond"** (guest editors Enrico Scalas, Lorenzo Torricelli).

Plan: build a multi-asset factor model where each asset's price is time-changed by the
*undershooting* of a subordinator (price staleness), extending:

- **ASTT** — Ascione, Scalas, Toaldo, Torricelli, *Time-changed Markov processes and coupled
  non-local equations*, arXiv 2412.14956. One asset, one staleness clock.
- **DMS** — D'Onofrio, Mutti, Semeraro, *Additive subordination of multiparameter Markov
  processes*, arXiv 2507.20863. k factors, k clocks, but Markov (no staleness).
  Their "Future Works" section explicitly names time-changing by the **inverse** of an
  additive subordinator as the open next step.

### 1.2 What was actually derived (all verified numerically, all still mathematically valid)

| Result | Status |
|---|---|
| H(t)/t ~ Beta(α, 1−α) for α-stable clocks | **Classical** — Dynkin–Lamperti generalised arcsine law. Not a contribution. |
| E[e^(−cH(t))] = ₁F₁(α;1;−ct) | Verified to 1e-11; trivial Beta integral |
| Kummer solution satisfies ASTT's non-local governing equation | Verified 1e-16 to 1e-10 |
| **k clocks do NOT decouple** — sum of marginal operators fails | Verified; ratios 0.26–0.66, drifting with t |
| Beta-tilted local evolution equation ∂ₜq = Σⱼ αⱼ G⁽ʲ⁾q⁽ʲ⁾ | Verified to 1e-4 (MC s.e.) on a non-product model |
| Dependent clocks (Sⱼ = Xⱼ + aⱼZ): joint law of (H₁/t, H₂/t) is t-free | Verified over 64× range of T |
| Marginals stay Beta(α,1−α) under common shocks | KS p = 0.06–0.92 |
| **Atom P[L₁(t)=L₂(t)] > 0** (simultaneous staleness) | Real, not discretisation: survives 50× dt refinement (0.286→0.279) |
| Bivariate potential density homogeneous of degree α−2 | Verified, CV 5–11% over 46× range in r |
| Closed form for that density (α=1/2, common shock) | **Derived**, verified off-diagonal to 3–7% |

Derived formula (α=1/2, Sⱼ = Xⱼ + aZ, all standard ½-stable), with xⱼ = sⱼ − az:

```
u(s₁,s₂) = π^(−3/2) ∫₀^{min(s₁,s₂)/a} √(z·x₁·x₂) / (x₁x₂ + z(x₁+x₂))² dz
```

Log singularity on the diagonal, confirmed analytically (∫dx/x) and numerically
(u(1,1+d) = 0.131 → 0.240 → 0.345 for d = 0.1 → 0.01 → 0.001; constant increment per decade).

### 1.3 Why it died — three strikes

**Strike 1 — the correlation term-structure claim was false.**
Swept 54,000 parameter combinations: rising 20.2%, falling 26.0%, hump 41.6%, dip 12.1%.
The shape is free, not predicted. What survives is weaker: slope sign is set by αⱼkⱼ vs α_c,
and ρ(0+) = a_j a_h α_c / √[(σ_j²α_j + a_j²α_c)(σ_h²α_h + a_h²α_c)] is invariant to
mean-reversion speeds (max spread 3.8e-07 over a 4096× range in k).

**Strike 2 — the "new" equation was a classical identity in disguise.**
For k=1 the Beta-tilted equation reduces exactly to the Kummer derivative identity
d/dz ₁F₁(a;b;z) = (a/b)₁F₁(a+1;b+1;z). Textbook. The atom mechanism is also known —
Marshall–Olkin common shocks, see Sun–Mendoza-Arriaga–Linetsky (Adv. Appl. Prob. 2017),
who build MO distributions from first-passage times of multivariate subordinators.

**Strike 3 (fatal) — no empirical support for α ∈ (0,1) anywhere.**

The whole framework requires the inter-trade duration tail index α < 1 (infinite mean
waiting time). Tested on real data:

*Vietnamese equities* (HOSE, 09:15–10:30 on 17 Aug 2026, 12 symbols, 10,179 ticks):

| Symbol | median gap | Hill α |
|---|---|---|
| SSI | 2.0s | 2.52 |
| VNM | 3.0s | 1.73 |
| VCB | 6.0s | 2.33 |
| DXG | 9.0s | 2.42 |
| PVD | 13.0s | 2.56 |
| HSG | 20.0s | 2.70 |

- corr(log median gap, α) = **+0.198, p = 0.638**. The model predicts a strong *negative*
  correlation. There is none. Illiquid names have longer typical gaps but the *same* tail
  index — staleness in VN equities is a scale effect, not a tail effect.
- Assumption-light test: running mean of durations converges for all 8 symbols
  (ratios 0.88–1.07). Finite mean. α > 1 unambiguously.

*Crypto* (Gate.io, 13 pairs spanning 8 orders of magnitude in 24h volume):

Raw Hill suggested α < 1 for illiquid pairs (DORA 0.47, STIX 0.37). But:

- **Burst aggregation kills it.** Aggregating trades within W seconds into one trade event
  (the economically correct treatment — a burst of fills in one second is one order being
  filled, not many events): DORA 0.47 → 0.83 → **1.20** → 1.25 for W = raw/1s/5s/30s;
  STIX 0.37 → 0.62 → 0.59 → **1.03**; ZIL 0.81 → 1.31 → 1.54 → 1.67.
- **Power-law goodness of fit (Clauset–Shalizi–Newman KS, 300 bootstrap reps) rejects the
  best candidate.** DORA p = 0.00 — so its α is meaningless, the tail is not a power law.
  BTC 0.02, ETH 0.02, SOL 0.00, LTC 0.00 also rejected. Only ATOM (0.52) and ZIL (0.32)
  pass, and both have α > 1 after aggregation. STIX p = 0.09, marginal, n_tail = 79.

**Conclusion: the inverse-stable-subordinator staleness mechanism — precisely the motivation
the Non-Markovian CFP cites — has no empirical anchor in modern electronic markets.**
This negative result is itself publishable as a short note and would save others months.

### 1.4 Unresolved loose end (if anyone ever revives this)

The α=1/2 potential-density formula matches MC off-diagonal (3–7%) but **not** on the
diagonal (ratios 0.63, 0.61, 0.81 after box-averaging). Cause not established — likely
Gauss–Legendre failing on the log singularity cutting through the box interior, but a
derivation error not visible off-diagonal has **not** been ruled out. Would need the box
split along the diagonal with each half quadratured separately.

---

## 2. Current direction

**Target: IJTAF Special Issue on Decentralized Finance.**

- Guest editors: **Fayçal Drissi** (Oxford; Oxford-Man Institute), **Sebastian Jaimungal**
  (Toronto; Oxford-Man).
- Deadline: **1 November 2026**.
- Scope: "the mathematics, economics, and data science of decentralized finance and
  blockchain-based financial systems." Theoretical *and* empirical welcome; interdisciplinary
  work bridging mathematical finance, OR, CS and economics explicitly invited.

### 2.1 Journal facts

- Scimago 2025: **Q2** in Economics/Econometrics/Finance (misc.), **Q3** in Finance.
  SJR 0.366, H-index 41. JCR IF 0.5 (sources disagree on WoS quartile — verify with
  Clarivate directly if it matters for appraisal).
- Acceptance rate **7%**. Submission → first decision 35.5 days; acceptance → online 26 days.
- Hybrid OA. **Check APC waiver before submitting** — standing preference is to avoid fees.
- Note: **Alexander Mijatović is an Associate Editor** and works directly on
  undershoot/overshoot joint laws. Relevant if §1's math is ever revived.

### 2.2 Assets that are NOT available

- **VEIN** (causal SCM, on-chain crypto systemic risk) — under review at JRF. Off limits.
- **CAIRN** (causal world model, DeFi + robotics) — under review. Off limits.

Anything new must be clearly distinct from both. If a future paper reuses VEIN's on-chain
data pipeline, the *contribution* must be sharply separated (VEIN = observational SCM for
risk measurement; anything new = something else) and VEIN cited once its status resolves.

### 2.3 Where the openings are

The CFP text itself is short and generic — no stated open problems. The signal comes from
the guest editors' own work. Drissi's core paper:

> **Cartea, Drissi, Monga**, *Decentralised Finance and Automated Market Making: Predictable
> Loss and Optimal Liquidity Provision*, arXiv **2309.08431** v3, forthcoming/published
> SIAM J. Financial Mathematics (DOI 10.1137/23m1602103).

They derive a closed-form optimal liquidity-provision strategy for concentrated-liquidity
pools, balancing fee revenue, predictable loss (PL), concentration risk, and rebalancing
costs. Their **model-extensions section** is an explicit list of what they did not do:

| Limitation | Their words | Assessment |
|---|---|---|
| **Gas fees** | assumed flat and constant; "in practice, gas fees are stochastic"; strategy "requires a large initial wealth to overcome gas fees from continuous trading"; they **expect** it to stay profitable in discrete time "when the stochastic drift μ and the stochastic profitability π remain stable, so the LP only rebalances her liquidity position when either the drift μ or the pool fee rate π undergo **large changes**" | **A change-detection problem stated but not solved.** No rule given for what "large" means. This is the most direct opening — but see §2.4, the area is crowded. |
| **Rebalancing costs** | "the nonlinearity in the CL constant product formula **complicates the mathematical modelling** of this aspect of trading costs" | They admit this is hard. Appears untouched. **Possibly the best remaining lead.** |
| **Asymmetry** | fixed relation between position asymmetry and drift; "future work will consider a richer characterization ... as a function of other state variables or as a controlled variable" | Open |
| **Fee dynamics** | fees assumed uncorrelated with price; "future work will consider more complex relations" | Open |
| **Blockchain delays** | 13s blocks, random intra-block queue, sandwich attacks; "our model can be extended to include delays" | Open |
| **Market impact** | "our analysis does not take into account the impact of liquidity provision on liquidity taking activity" | Open |
| **Constant volatility** | "**It is straightforward to extend** our strategy to this type of models" | **AVOID** — they declared it trivial |

Second relevant paper: Cartea, Drissi, Monga, *Execution and Speculation*, arXiv **2307.03499**.
Its future-work section points to strategic liquidity provision and AMM design.

### 2.4 Prior art — the gas/rebalancing lead is crowded

Checked 17 Aug 2026. Do not propose anything in this space without reading these first:

| Paper | What it already does |
|---|---|
| **RAmmStein**, arXiv **2602.19419** (Mar 2026) — *Regime Adaptation in Mean-reverting Markets with Stein thresholds: Optimal Impulse Control in Concentrated AMMs* | Double DQN for exactly the LP-rebalancing impulse control problem, **with regime indicators**. Biggest threat. **NOT YET READ — read this first.** |
| **arXiv 2606.21769** (Jun 2026) — *Optimal Dynamic Fees for AMMs: A Stochastic Control Approach to Loss-Versus-Rebalancing* | Ergodic HJB; "treats gas costs through an **impulse-control dead-band**" |
| Fan et al., AFT 2023 (LIPIcs 282, 25) — *Strategic Liquidity Provision in Uniswap v3* | τ-reset strategies, context-aware dynamic LP via NNs |
| arXiv 2309.10129 | Deep RL for adaptive LP in Uniswap v3 |
| arXiv 2411.12375 | LP position pricing via **stopping time** |
| Fan et al., ICAIF 2022, arXiv 2204.00464 | Differential liquidity provision, gas-fee modelling |

**Honest read:** impulse control for LP rebalancing is done; gas dead-bands are done; RL is
done; regime adaptation is done. The only obvious residue is the *anytime-valid* angle —
RAmmStein uses Stein thresholds + DQN (no statistical guarantee), the dead-band in 2606.21769
is a fixed HJB threshold, not a detection rule. **But "replace a fixed threshold with an
e-process" is thin** and matches the incremental pattern that has drawn desk rejections
before. Treat it as a candidate to be disproved, not a plan.

### 2.5 Open questions to answer before designing anything

1. Read RAmmStein (2602.19419) fully. Specifically: (a) does its trigger carry any
   statistical guarantee, or is it a learned threshold? (b) does it model stochastic gas, or
   constant gas as in CDM? (c) what limitations does it declare?
2. If the gap after (1) is thin, pivot to the **rebalancing-costs nonlinearity** limitation —
   the one CDM themselves call mathematically complicating, and which nobody appears to have
   attacked.
3. Only then design. **Do not build before validating** — see §4.

---

## 3. Data pipeline

### 3.1 Verified working (tested 17 Aug 2026)

- **DeFiLlama**, no key required: `api.llama.fi/protocols`, `/v2/chains`,
  `coins.llama.fi/prices/current/...`, `api.llama.fi/summary/fees/{protocol}` — all 200.
- **Gate.io** public spot API — works, used for the §1.3 crypto tests. Caps at ~1000 most
  recent trades per pair; `last_id` pagination did **not** yield deeper history.
- **Kraken, Coinbase, MEXC** public APIs reachable. **Binance 451, Bybit 403** from
  datacenter IPs.
- Ethereum RPC: `ethereum-rpc.publicnode.com` and `eth.drpc.org` answer `eth_blockNumber`.
  `rpc.ankr.com` now requires a key; `cloudflare-eth.com` refuses.

### 3.2 Not verified

`eth_getLogs` could not be confirmed end-to-end — public nodes returned 403 after a few dozen
requests from the sandbox IP. **Get a free Alchemy or Infura key and put it first in
`RPC_URLS` before any long backfill.** That removes block-range limits entirely.

### 3.3 Collector script: `uni_collect.py`

Pulls Swap / Mint / Burn events + block timestamps + base fee for a target pool
(default: USDC/WETH 0.05%, `0x88e6a0c2ddd26feeb64f039a2c41296fbcb3d5aa` — the pool CDM
study), writes parquet, resumes from `uni_state.json`, auto-halves chunk size on range
errors, round-robins across RPC endpoints.

**Two silent-failure bugs were found and fixed while writing it. Both are worth knowing:**

1. **Topic hash.** The Swap `topic0` was mis-transcribed from memory
   (`...818eb64fced9d1bf8d3f96b0d0`; correct is `...818eb64fed8004e115fbcca67`). A wrong
   topic hash returns **zero logs forever with no error**. The script now computes all
   topic hashes at runtime via keccak256 of the event signature. Never hand-copy these.
2. **Sign extension.** Solidity sign-extends narrow signed types (`int24 tick`, `tickLower`,
   `tickUpper`) to the full 32-byte word, in both `data` and indexed topics. Decoding at
   24 bits yields a 77-digit garbage tick. Decode at 256 bits.

All three decoders are unit-tested offline against synthetic logs.

Deps: `pip install pandas pyarrow requests pycryptodome`

### 3.4 Other data collected

`vn_ticks.py` (resumable HOSE tick collector via vnstock) and the resulting `vn_ticks.csv`
exist from §1.3. Keep them — they are the evidence base for the negative result, and the
collector is reusable if VN market data is ever needed again. Note HOSE timestamps have
**1-second resolution** with 31–56% zero-duration ties, which caps what any
simultaneity statistic can measure.

---

## 4. Working rules for this project

These come from what went wrong in §1. They are not optional.

1. **Validate the empirical premise before deriving anything.** The α < 1 test should have
   been run on day one; it would have saved three weeks. Any new direction gets its
   load-bearing assumption tested against real data *first*.
2. **Search prior art before building, not after.** Two "new" results in §1 turned out to be
   a textbook Kummer identity and a known Marshall–Olkin mechanism.
3. **Compute, don't recall.** Hashes, constants, formulas — derive or compute them in code.
   Two of the bugs in §3.3 were memory errors that would have failed silently.
4. **Report negative results.** The 54,000-parameter sweep that killed the term-structure
   claim belongs in any paper written from this material, stated up front. Referees find
   these things; self-disclosure reads as rigour, discovery reads as concealment.
5. **No incremental combinations.** "Known method A applied to domain B" is the pattern that
   draws desk rejections. If the contribution can be described as a wrapper, it is not ready.
6. **Citations must be real** — full author lists and DOIs, verified, no hallucinated
   references.

---

## 5. Immediate next actions

- [ ] Read arXiv 2602.19419 (RAmmStein) in full; answer the three questions in §2.5(1)
- [ ] Get an Alchemy/Infura key; run `uni_collect.py` for the USDC/WETH 0.05% pool
- [ ] Read arXiv 2606.21769 to pin down exactly what its gas dead-band does
- [ ] Decide: anytime-valid rebalancing trigger, or the rebalancing-cost nonlinearity
- [ ] Check IJTAF APC waiver policy before committing to submission
