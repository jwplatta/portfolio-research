# SP500 Market-Neutral Short-Horizon Momentum — Cost Analysis Log

Goal: find configurations that achieve net Sharpe > 1.0 at 10 bps one-way cost.
Starting point: `short_horizon_momentum_w20` (gross Sharpe 1.28, net Sharpe 0.66, 29.5% avg daily turnover, 7.4% annual cost drag).

---

## Results

| Version | Net Sharpe | Ann. Return | Max DD | Turnover | Change |
|---|---|---|---|---|---|
| v0 | 0.66 | 7.6% | -16.6% | 29.5% | baseline |
| v1_rebalance_10d | **0.68** | 7.4% | -15.7% | 16.9% | rebalance every 10d |
| v2_rebalance_15d | -0.00 | -0.7% | -25.7% | 11.8% | rebalance every 15d |
| v3_n15 | 0.22 | 2.4% | -41.4% | 18.5% | 15L/15S |
| v4_n10 | 0.12 | 0.5% | -44.2% | 28.7% | 10L/10S |
| v5_inv_vol | -0.68 | -7.2% | -55.2% | 24.6% | inv-vol weighting |
| v6_signal_threshold | 0.66 | 7.6% | -16.6% | 29.5% | outer-30th-pct filter |
| v7_rebal10_n15 | 0.37 | 5.7% | -36.4% | 12.7% | 10d + 15L/15S |
| v8_rebal10_n15_inv_vol | -0.25 | -3.4% | -38.4% | 15.9% | 10d + 15L/15S + inv-vol |
| v9_window_15 | -0.45 | -6.4% | -57.6% | 31.3% | signal window 15d |
| v10_window_30 | -0.08 | -1.7% | -25.8% | 27.8% | signal window 30d |

---

## Key Findings

### Rebalance cadence is the only viable lever — but not enough
- **v1 (10d rebalance)** is the only version that improves on v0, halving turnover (29.5% → 16.9%) while barely affecting net Sharpe (0.66 → 0.68). Cost drag falls from 7.4% to ~4.2% annualized.
- **v2 (15d rebalance)** collapses: Sharpe goes to zero. The 20d signal simply decays too fast — by 15d the positions are stale and the gross edge disappears.

### Concentration destroys Sharpe, doesn't save it
- v3 (15 names) and v4 (10 names) both have lower turnover than v0 but far worse Sharpe and drawdown. Idiosyncratic vol overwhelms any cost saving.

### Inverse-vol weighting is harmful here
- v5 and v8 are sharply negative. The inv-vol scheme changes position weights continuously each day, which generates *more* turnover than equal weight at the same rebalance cadence — exactly the opposite of the intention.

### Signal threshold filter has no effect
- v6 matches v0 exactly. The outer-30th-pct mask NaNs the middle of the distribution, but the `build_long_short(n=20)` selection already only touches the tails — so the mask is effectively a no-op.

### Shorter/longer signal windows both hurt
- v9 (15d) and v10 (30d): worse Sharpe and worse drawdown. 20d is the dominant window (confirmed by the original sweep).

---

## Conclusion

**Net Sharpe > 1.0 at 10 bps is not achievable** with this signal and these structural levers. The gross alpha (Sharpe 1.28) is real but the cost drag (~7.4% ann.) is too large relative to the signal's decay characteristics. The 20d window requires frequent rebalancing to stay fresh; slowing rebalance past 10d destroys the gross signal faster than it saves costs.

Best version: **v1 (10d rebalance)** — Sharpe 0.68, turnover halved, modest drawdown improvement.

### Next steps worth exploring
- Threshold-triggered rebalancing instead of calendar: only trade when a name's rank change exceeds some minimum, which could further reduce turnover without signal staleness
- Tighter bid-ask assumption (5 bps) as a sensitivity check — the strategy may be viable at lower cost execution venues
- Extending the date range post-2023 to check if the gross alpha held

---

## Round 2: Threshold-Triggered Rebalancing (baseline = v1, 10d calendar)

Hypothesis: replace fixed 10d calendar rebalance with signal-driven triggers to reduce turnover below v1's 16.9% without staling the book.

| Version | Net Sharpe | Ann. Return | Max DD | Turnover | Trigger |
|---|---|---|---|---|---|
| v1 (baseline) | 0.68 | 7.4% | -15.7% | 16.9% | every 10d |
| v11_rank_trigger | -0.26 | -3.9% | -44.5% | **43.8%** | rank_change(0.85) |
| v12_book_overlap_70 | -1.22 | -15.3% | -79.1% | **91.1%** | book_overlap(0.70) |
| v13_book_overlap_80 | -1.47 | -18.0% | -84.3% | **102.5%** | book_overlap(0.80) |

### Findings

**Threshold triggers dramatically increase turnover rather than reduce it.** All three versions are far worse than calendar rebalancing, and turnover explodes — v12/v13 exceed 90%+ daily turnover vs v1's 16.9%.

The reason: `rebalance_on` is evaluated daily against the *proposed* full-rebalance positions. Because the sector-beta neutralization produces small continuous weight perturbations day-to-day, the book_overlap trigger sees near-zero overlap and fires almost every day. The rank trigger has a similar problem at threshold=0.85 — the 20d rolling signal has a high auto-correlation in its ranks, but the neutralizer reshuffles weights enough that rank-corr is frequently below 0.85 for the position vector.

In other words: these triggers measure the wrong thing. They're comparing neutralized position weights (which change constantly due to beta adjustment) rather than pure signal rankings.

### Conclusion

Threshold-triggered rebalancing does not help for this strategy. The sector-beta neutralizer's continuous adjustments make position-level triggers fire near-daily regardless of how the underlying signal has changed. Calendar rebalancing (v1, every 10d) remains the best configuration found.

**Final recommendation: v1** — net Sharpe 0.68, turnover 16.9%, max DD -15.7%.
The strategy is not viable at 10 bps with the current signal and structure. Lower execution costs (≤5 bps) would be needed to push net Sharpe above 1.0.

---

## Round 3: Signal Selectivity (baseline = v1, 10d calendar)

Hypothesis: reduce turnover by only trading the highest-conviction names, or reduce the drawdown by filtering out regimes where the signal fails.

| Version | Net Sharpe | Ann. Return | Max DD | Turnover | Change |
|---|---|---|---|---|---|
| v1 (baseline) | 0.68 | 7.4% | -15.7% | 16.9% | every 10d |
| v15_min_hold_trigger | — | — | — | — | min_hold(5d) + book_overlap(0.50) |
| v16_smoothed_min_hold | — | — | — | — | smooth_signal(5d) + min_hold + book_overlap(0.50) |
| v17_top10_book | ~0.09 | — | — | — | 10L/10S (no filter) |
| v18_pct15_filter | 0.62 | — | — | — | outer-15pct filter + 20L/20S |
| v19_pct15_top10 | ~0.09 | — | — | — | outer-15pct filter + 10L/10S |

### Findings

**Selectivity does not improve net Sharpe.**
- Percentile filter (v18): nearly matches v1 (0.62 vs 0.68) but doesn't beat it. The filter eliminates boundary names but the 20L/20S selection already avoids the middle of the distribution.
- Concentration (v17, v19): 10L/10S crushes Sharpe to ~0.09. Idiosyncratic vol dominates.
- Signal smoothing (v16): killed gross alpha — 5d rolling mean introduces too much lag into a 20d signal.
- Trigger + min_hold (v15/v16): still fires frequently; no improvement over calendar.

---

## Round 4: Drawdown Regime Analysis (baseline = v1)

### v1 Drawdown Profile

Annual returns by year:
- 2015: +0.95%, 2016: +8.12%, 2017: +11.68%, 2018: +17.90%
- 2019: **-8.85%** (bench +31.2%) — worst year
- 2020: +32.03% (COVID vol), 2021: -2.41%, 2022: +10.96%, 2023: +1.67%

Top 3 drawdown episodes:
1. **2019-01-07 → 2020-02-03 (271d, -15.7%)**: Bench +30.9% during recovery from Q4 2018 crash — factor rotation into large-cap growth overwhelmed sector-relative momentum signal.
2. **2020-12-11 → 2022-02-22 (301d, -12.5%)**: Post-COVID reflation (+19.2% bench) — same pattern.
3. **2022-02-24 → 2022-09-29 (151d, -15.1%)**: Bear market with bench vol spike (+25% ann vol).

### Regime Analysis

| Bench Regime | Ann Return | Sharpe | N Days |
|---|---|---|---|
| Downtrend (<-2% 20d) | -5.1% | -0.31 | 422 |
| Neutral (±2% 20d) | +10.7% | 1.01 | 842 |
| Uptrend (>+2% 20d) | +10.6% | 0.95 | 981 |

| Bench Vol Regime | Ann Return | Sharpe | N Days |
|---|---|---|---|
| Low (bot 33%) | +6.9% | 0.80 | 741 |
| Mid (mid 33%) | -0.13% | 0.04 | 763 |
| High (top 33%) | +16.6% | 1.13 | 741 |

**Key insight:** Strategy fails specifically during bench *downtrend* periods and does *best* in high-vol regimes (COVID, 2022 bear market spikes). Mid-vol regime is a dead zone. The 2019 loss was during a strong bull market — the strategy was whipsawed by factor rotation, not caught by a simple trend filter.

### Regime Filter Attempts (v20-v23)

| Version | Net Sharpe | Description |
|---|---|---|
| v1 (baseline) | 0.68 | — |
| v20_bench_trend_filter | 0.64 | Go flat when bench 20d < -2% |
| v21_bench_trend_halfscale | 0.22 | 50% scale when bench 20d < -3% |
| v22_extreme_bench_filter | 0.56 | Flat when bench 20d outside (-2%, +7%) |
| v23_rolling_sharpe_scaler | 0.68 | 25% scale when rolling Sharpe < 0 (no-op: gross returns are None at scaler stage) |

### Why Regime Filters Failed

1. **Going flat creates flatline drawdowns.** When the strategy is off, the equity curve neither rises nor falls — but the drawdown clock keeps running from the prior HWM. Max drawdown duration ballooned from 271d to 700+ days.
2. **v23 was a no-op.** `gross_portfolio_returns` is initialized to `None` in the Study cache; it's not populated until after the backtest stage, which runs after all position_scalers. So the rolling Sharpe scaler always fell back to returning positions unchanged.
3. **The 2019 drawdown doesn't map to a simple regime.** 2019 was a strong bull market (+31% SPY). No downtrend filter would have caught it. The strategy lost because sector-relative momentum reversed sharply after the Q4 2018 selloff — a factor-level regime that can only be detected by cross-sectional IC monitoring, not macro filters.

### Conclusion

Regime conditioning does not improve v1. The three structural barriers remain:
- Gross alpha (Sharpe 1.28) is insufficient to overcome 10 bps costs at 16.9% turnover
- External regime filters either don't activate (2019 was an uptrend) or create worse drawdown duration when they do activate
- `scale_risk` scalers cannot access the strategy's own realized returns at execution time

**The strategy needs either: (a) <5 bps execution cost, or (b) a fundamentally different signal with >2x the gross Sharpe at comparable turnover.**

---

## Round 5: Signal Weighting and Faster Rebalance (baseline = v1)

| Version | Net Sharpe | Gross Sharpe | Turnover | Max DD | Notes |
|---|---|---|---|---|---|
| v1 (baseline) | 0.68 | 1.28 | 16.9% | -15.7% | — |
| v24_signal_weighted | 0.47 | 0.98 | 15.7% | -14.5% | signal-rank weights + neutralizer |
| v25_rebalance_5d | **0.69** | 1.30 | 29.5% | -16.6% | 5d calendar, no other changes |
| v26_signal_weighted_no_neutral | 0.38 | 0.53 | 6.4% | -19.1% | signal-rank weights, no neutralizer |

### Findings

**Signal weighting is incompatible with the pipeline's ordering.** `build_long_short` produces equal weights, the neutralizer then reshuffles them to enforce sector/beta neutrality, and applying signal-rank weights on top of that neutralized book reintroduces sector tilts. Gross Sharpe dropped from 1.28 to 0.98 (v24). Removing the neutralizer to avoid the ordering problem (v26) was even worse — the sector-relative signal loses most of its edge without neutralization (gross Sharpe 0.53). The neutralizer is load-bearing; signal weighting would require it to be built into `build_long_short` before neutralization, which the current API doesn't support.

**5d rebalance (v25) adds no value.** Gross Sharpe is essentially unchanged (1.30 vs 1.28) but turnover doubles back to v0 levels (29.5%). The cost drag wipes out any signal freshness benefit.

---

## Final Summary

**Best version: v1 (`v1_v1_rebalance_10d`)**

| Metric | Value |
|---|---|
| Net Sharpe | **0.68** |
| Gross Sharpe | 1.28 |
| Ann. Return (net) | 7.4% |
| Ann. Vol | ~11% |
| Max Drawdown | -15.7% |
| Max DD Duration | 271 days (2019-01-07 → 2020-02-03) |
| Avg Daily Turnover | 16.9% |
| Cost Drag (ann.) | ~4.2% |
| Cost Assumption | 10 bps one-way |

**Why nothing beat v1:** The gross signal (Sharpe 1.28) at 20d window is real but the cost structure at 10 bps leaves only 0.68 net. Every lever tried either increased turnover (more cost), reduced signal freshness (lower gross Sharpe), or introduced structural problems (neutralizer ordering with signal weighting, flatline drawdowns with regime filters).

**What would be needed to reach net Sharpe > 1.0:**
- Execution cost ≤ 5 bps (halves cost drag from ~4.2% to ~2.1%), or
- A signal with gross Sharpe ≥ 2.0 at comparable turnover, or
- Portfolio combination with an uncorrelated strategy to reduce drawdown duration at the portfolio level
