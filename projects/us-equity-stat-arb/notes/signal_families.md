# Signal Families in the qstudy Portfolio

## Overview

The portfolio uses 9 distinct signal families, organized by their economic mechanisms and market regimes. These families are designed to be economically orthogonal, providing diversified sources of edge across various market conditions. Portfolio construction prioritizes consistency (positive in most years), robustness to regime change, and genuine diversification—not just statistical correlation reduction.

---

## Tier 1: Consistent (Positive in almost every year)

### 1. Distance Pairs Mean Reversion

**Economic Motivation:**
Statistical mean reversion of cross-sectional pricing dislocation. Pairs stocks with similar characteristics (defined by nearest-neighbor distance in a latent space) and trades mean-reverting spreads. This is a market-neutral mechanism: when paired stocks diverge in price, they're expected to reconverge. Works best when there's sufficient cross-sectional dispersion—fails in "narrow bull" regimes where one group persistently outperforms.

**Signal Construction:**
- Core: normalized price spread between a stock and its k-nearest neighbors (k=1, 3, or 5)
- Signal = -(spread - rolling_mean) / rolling_std, clipped to [-2, 2]
- Lookback windows: z10, z20, z60 (rolling standardization window)
- Rebalance frequency: r5 (weekly), r10 (biweekly), r21 (monthly)

**Key Parameters & Variants Tested:**
- **k variants:** k=1 (closest peer, fastest/noisiest), k=3 (balanced), k=5 (smoother)
- **Standardization windows:** z10 (high sensitivity), z20 (balance), z60 (stable but slow)
- **Rebalance:** r21 is most cost-effective (0.26–0.34% daily TO); r5/r10 are more responsive
- **Regime gates:** `narrow_bull_off_40/50` disables signal when breadth < 40%/50% AND SPY is bullish—the regime where spread persistence breaks down

**Pool Sleeves:**
- `dist_mr_k1_z60__r21__none__cond__none` — ungated baseline (avg Sharpe ~0.63)
- `dist_mr_k1_z60__r5__none__cond__narrow_bull_off_50` — NBO50 variant; gated, weekly rebalance; highest single-sleeve IS Sharpe
- `dist_mr_k1_z20__r21__none__cond__none` — faster dislocation capture (Sharpe ~0.60)
- `dist_mr_k1_z60__r10__none__cond__none` — biweekly rebalance (Sharpe ~0.53)
- `dist_mr_k3_z20__r10__none__cond__low_disp_off_q30` — k3 with dispersion gate
- `dist_mr_k3_z60__r21__none__cond__none` — multi-partner, structurally distinct (Sharpe ~0.39)

**Final Portfolio Roles:**
- **Sequential FINAL:** `_Z60` = `dist_mr_k1_z60__r21__none__cond__none` (ungated baseline)
- **GREEDY_CORE:** `_NBO50` = `dist_mr_k1_z60__r5__none__cond__narrow_bull_off_50` (seed sleeve)

**Main Weakness:** 2023 (−0.35 on k1_z60). The narrow-bull gate is essential: removes regimes where market leaders persistently diverge from laggards.

---

### 2. Bear Reversal (Crisis Alpha)

**Economic Motivation:**
Short-term mean reversion conditional on market stress. When breadth narrows (many stocks underwater) and SPY is in downtrend, prices overshoot lower and snap back quickly. A "crisis alpha" signal—dormant in normal markets but explosive in sharp selloffs. Captures panic-driven reversals.

**Signal Construction:**
- **`bear_reversal`:** `-rolling_mean(returns, 20d)` — negated cumulative return (pure MR)
- **`vol_accel`:** `-(fast_vol - slow_vol)` — positive when volatility compresses after a spike

Conditioning filters:
- **`bear_narrow_lt40`:** Active only when (breadth < 40%) AND (SPY 50d MA < 200d MA)
- **`breadth_lt40/lt50`:** Looser breadth-only gates used by vol_accel variants

**Key Parameters & Variants Tested:**
- **Window:** bear_reversal at 5d, 20d; vol_accel at fast 5/10/20d, slow 60/90/120d
- **Rebalance:** r10, r21 (weekly not effective—too much noise in stressed markets)
- **Filter strictness:** bear_narrow_lt40 (strict, better Sharpe) vs breadth_lt40 (looser)

**Pool Sleeves:**
- `bear_reversal_20d__r21__trend_20_100_mr__cond__bear_narrow_lt40` — best overall (avg Sharpe ~1.08)
- `vol_accel_20_120d__r10__vol_10_60_up__cond__breadth_lt40` — vol compression variant (Sharpe ~1.05)
- `vol_accel_10_90d__r10__trend_50_200_mr__cond__bear_narrow_lt40` — strict bear + vol variant

**Final Portfolio Roles:**
- **Sequential FINAL & GREEDY_CORE:** `_BEAR` = `bear_reversal_20d__r21__trend_20_100_mr__cond__bear_narrow_lt40`
- Portfolio narrative: "zero negative IS years" — performs in every single OOS year tested

**Main Weakness:** Dormant in bull markets (by design). Not a return contributor in calm years; earns its place through crisis periods like 2022.

---

### 3. Monotonicity / Return Consistency

**Economic Motivation:**
Stocks with consistent (monotonic) daily returns signal persistent drift. Rewards sign-consistency over a year: if daily returns have mostly been positive and the rolling mean is positive, the stock is in a persistent upward drift and is expected to continue. Acts as a breadth-filtered momentum signal—turns off when the market is broadly up (reducing momentum crowding).

**Signal Construction:**
- **`monoton_skip_252d`:** fraction of daily returns in same sign as 252d mean × |mean|, shifted 5 days
  - The 5-day "skip" removes short-term reversal noise (Grundy & Martin 2001)
- **`monoton_w252d`:** Unshifted variant; structurally distinct annual profile

Conditioning filters:
- **`breadth_40_off`:** Turn OFF when breadth > 40% (broad rallies; momentum crowding risk)
- **`breadth_35_off`:** Tighter gate; more selective; reduces 2022 drawdown

**Key Parameters & Variants Tested:**
- **Window:** 60d, 120d, 252d (year-long consistency dominates)
- **Type:** skip (5-day offset) vs w (weighted, no offset)
- **Gate tightness:** breadth_40 vs breadth_35 (35 is tighter → fewer active days but more selective)
- **Rebalance:** r21 (monthly; high enough for this slow signal)

**Pool Sleeves:**
- `monoton_skip_252d__r21__breadth_40_off__cond__none` — baseline (avg Sharpe ~0.84)
- `monoton_skip_252d__r21__breadth_35_off__cond__none` — tighter gate (Sharpe ~0.84)
- `monoton_skip_252d__r21__trend_50_200_off__cond__none` — trend-gated variant
- `monoton_w252d__r21__breadth_35_off__cond__none` — unshifted; distinct 2017/2020 profile
- `monoton_skip_252d__r21__vol_10_60__cond__none` — vol-scaler variant (Sharpe ~0.75)

**Final Portfolio Roles:**
- **Sequential FINAL:** `_MONO` = `monoton_skip_252d__r21__breadth_40_off__cond__none`
- **GREEDY_CORE:** `_MONO35` = `monoton_skip_252d__r21__breadth_35_off__cond__none` (tighter gate selected by greedy process)

**Main Weakness:** 2022 (−1.04 on vol_10_60 variant). The breadth_off gate dampens this: signal turns off in broad, strong uptrends where momentum fades.

---

## Tier 2: Robust-Regime (Strong average, one meaningful bad year; benefit from gating)

### 4. Cross-Sectional Mean Reversion (Cumulative Return Spread)

**Economic Motivation:**
Classical cross-sectional mean reversion: stocks that are very positive relative to their recent baseline tend to revert. This is *global* cross-sectional MR—distinct from pairs MR because it doesn't match stocks locally, it uses a universal benchmark. Fails in extended rallies (2015) and sustained bear markets (2022), but a vol scaler dampens both.

**Signal Construction:**
- **`cumret_spread`:** `-(20d_rolling_mean - 252d_rolling_mean)` — how far recent drift is from annual baseline
- **`zscore_rev`:** Normalized version: `-(zscore of 20d_mean vs 252d_baseline)`

Scalers:
- **`vol_20_60, vol_20_100`:** Scale signal down in high-volatility (compress in stress)
- **`trend_20_100, trend_50_200`:** Scale down in uptrends (when MR is weakest)

**Key Parameters & Variants Tested:**
- **Windows:** (20, 252) dominant; (5, 252) and (10, 120) also tested
- **Rebalance:** r5 (highest Sharpe, ~2.2% daily TO), r10, r21
- **Scalers:** vol filters add ~0.07 Sharpe; trend gates also beneficial
- **Construction:** raw vs zscore (different distribution of signals across universe)

**Pool Sleeves:**
- `cumret_spread_20_252__r5__vol_20_60__cond__none` — best full-period (Sharpe ~1.03)
- `cumret_spread_20_252__r10__vol_20_60__cond__none` — lower TO (Sharpe ~0.96)
- `cumret_spread_20_252__r5__none__cond__none` — unscaled baseline (Sharpe ~0.95)
- `cumret_spread_20_252__r5__trend_50_200__cond__none` — trend-scaler variant
- `zscore_rev_20_252__r5__vol_20_60__cond__none` — alternative construction (Sharpe ~0.87)

**Final Portfolio Roles:**
- **Sequential FINAL & GREEDY_CORE:** `_CUMRET` = `cumret_spread_20_252__r5__vol_20_60__cond__none`
- Portfolio narrative: "cross-sectional style/sector MR — different mechanism from pairs"

**Main Weakness:** 2015 (extended rally) and 2022 (extended bear). Vol scaler dampens both but doesn't eliminate. Complements pairs MR (global vs local).

---

### 5. Gap Accumulation (Event-Driven)

**Economic Motivation:**
Stocks that have gapped up (spiked to new highs on a single day) tend to mean-revert. Signal is `-(rolling_max(returns, window))`—the strongest single-day move in the last 3 days is a sell signal. Works best in choppy/rotating markets; 2022 is the standout year (+2.76 on best config) because gap-reversions dominate in volatile bear markets with rapid relief bounces.

**Signal Construction:**
- **`gap_accum_3d`:** `-(rolling_max(returns, 3))` — 3-day lookback for the spike
- **`gap_accum_2d`:** `-(rolling_max(returns, 2))` — more sensitive
- **`resid_gap_accum_5d`:** Same logic on factor-residualized returns

Filters:
- **`trend_20_100_off`:** Turn off when trend is strongly bullish—avoids shorting persistent leaders in uptrends
- **`vol_10_60_off`:** Turn off in low-volatility regimes where gaps have weaker mean-reversion

**Key Parameters & Variants Tested:**
- **Window:** 2d, 3d, 5d (shorter windows more reactive)
- **Rebalance:** r10 (most common), r21
- **Residualized:** Full-period slightly underperforms raw, but 2022 is robust

**Pool Sleeves:**
- `gap_accum_3d__r21__trend_20_100_off__cond__none` — best overall (Sharpe ~0.81, 2022 = +2.76)
- `gap_accum_3d__r21__trend_20_100__cond__none` — slightly different trend gate
- `gap_accum_2d__r10__breadth_20_q30__cond__none` — 2-day variant; distinct 2020/2021 profile
- `resid_gap_accum_5d__r10__vol_10_60_off__cond__none` — residualized; lower Sharpe but 2022 hedging

**Final Portfolio Roles:**
- **Sequential FINAL:** Not included in CORE5 or FINAL
- **GREEDY_CORE extension:** `_GAP` = `gap_accum_3d__r21__trend_20_100_off__cond__none`
  - Portfolio narrative: "fixes the 2022 bear-market fold (SR 0.11 → 1.34); strongest single addition"

**Main Weakness:** Trend gate dependency—raw signal (no gate) has negative full-period Sharpe (−0.21). Gate does all the work; signal is exposed if trend gate misfires.

---

### 6. Sector ETF Momentum (Cross-Asset Signal)

**Economic Motivation:**
Sector leadership drives stock returns. When a sector ETF is outperforming SPY, all liquid names within that sector tend to do well. This is a "sector rotation" signal: identify which sectors are leading, then go long all liquid names in those sectors. Works best when sector dispersion is high (some sectors diverging sharply from SPY). Distinct from all other signals—only cross-asset factor in the pool.

**Signal Construction:**
- **`sector_spy_mom`:** `sector_etf.rolling(window).mean() - spy.rolling(window).mean()` — each stock receives its sector's momentum score vs market
- **`sector_rel_cumlog`:** Proper compounding: cumulative log excess return of sector vs SPY
- **`sector_spy_sharpe`:** Sharpened variant: momentum normalized by sector vol stability

Filters:
- **`sector_disp_20d_q70`:** Active only when sector dispersion (std across sector returns) > 70th percentile
- **`stress_disp_20d_q70`:** Active only when BOTH SPY vol is elevated AND sector dispersion is high
- **Long-only variants:** Short leg (betting sectors will underperform) is toxic post-2008 and excluded

**Key Parameters & Variants Tested:**
- **Window:** 20d, 60d, 120d (longer windows smooth sector momentum)
- **Skip:** skip5 variant (Grundy & Martin: skip 1 month to avoid reversal noise)
- **Dispersion threshold:** q60 vs q70 (stricter = fewer opportunities but higher quality)
- **Long-only:** Net positive vs short legs; short legs consistently detract

**Pool Sleeves:**
- `sector_spy_mom_20d__r5__none__cond__sector_disp_20d_q70` — dispersion-gated (Sharpe ~0.76)
- `sector_spy_mom_20d__r5__none__cond__stress_disp_20d_q70` — stress+disp gate (Sharpe ~0.76)
- `sector_spy_mom_120d_skip5__r21__trend_50_200_mom__cond__sector_disp_20d_q60` — slow skip variant
- `sector_rel_cumlog_20d__r5__long__cond__sector_disp_20d_q70` — cumlog, long-only
- Several cumlog_sharpe variants at 20d, 120d with dispersion gates

**Final Portfolio Roles:**
- **GREEDY_CORE:** `_SSM20` = `sector_spy_mom_20d__r5__none__cond__stress_disp_20d_q70`
  - Portfolio narrative: "cross-asset signal absent from sequential baseline; adds genuine diversification"
  - Selected consistently in 4+/5 folds across 3 greedy seed variants

**Main Weakness:** Dispersion-dependent—signal is turned off most of the time (only active when sectors diverge sharply). Adds capacity constraints.

---

## Tier 3: Regime-Dependent (Kept selectively or excluded)

### 7. Residual Mean Reversion (Factor-Model-Residualized)

**Economic Motivation:**
Mean reversion of idiosyncratic returns after removing systematic factor exposure (Barra-lite: size, sector, momentum, volatility). Captures stock-specific pricing dislocations missed by factor investing. Works when factor returns diverge sharply from stock idiosyncratic returns (2016, 2021, 2022). Blows up if factors reverse sharply (2019 loss: −1.98 without trend gate).

**Signal Construction:**
- **`factor_model_resid_mr`:** `-(residual_returns.rolling(window).mean())`
- **`etf_factor_resid_mr`:** Same logic using sector ETF factor model instead of Barra
- **`resid_zscore_w15_w10`:** Tight-winsorized z-score (±1.5 clip) on residual returns—prevents over-trading during noise spikes

Filters (mandatory):
- **`trend_20_100, trend_50_200`:** Without these, 2019-like factor reversals are catastrophic

**Key Parameters & Variants Tested:**
- **Window:** 3d, 5d, 10d, 20d (shorter = higher frequency MR)
- **Factor model:** Barra-lite (4-factor) vs ETF-factor (sector ETF as factor)
- **Winsorization:** Full clip [-2, 2] vs tight [±1.5] (tight clip reduces tail risk)
- **Rebalance:** r5, r10 (fast mean reversion requires higher frequency)

**Pool Sleeves:**
- `factor_model_resid_mr_10d__r10__trend_20_100__cond__none` — Sharpe ~0.60, walkforward ~0.73
- `factor_model_resid_mr_10d__r5__trend_20_100__cond__none` — weekly rebalance, Sharpe ~0.60
- `etf_factor_resid_mr_5d__r10__trend_50_200__cond__none` — ETF-based factors, walkforward ~0.93
- `resid_zscore_w15_w10__r10__trend_20_100__cond__none` — winsorized variant; Sharpe ~0.57, best 2022 OOS resilience (+1.02)

**Final Portfolio Roles:**
- **GREEDY_CORE extension:** `_RZSCORE` = `resid_zscore_w15_w10__r10__trend_20_100__cond__none`
  - Portfolio narrative: "consistent low-vol diversifier; modest per-fold gains, never hurts, lowest max drawdown"

**Main Weakness:** 2019 blowup (−1.98) without trend gate. The trend gate is mandatory, not optional.

---

### 8. Residual Momentum (252-Day Idiosyncratic)

**Economic Motivation:**
Factor residuals exhibit momentum—stocks beating their sector tend to continue beating it for ~252 days. This is a factor-timing signal: profit from the persistence of stock-specific outperformance (the idiosyncratic component of cross-sectional momentum). Positive in years when systematic momentum reverses (2015, 2018, 2020), providing an offset to raw momentum losses.

**Signal Construction:**
- **`resid_mom_252d`:** `residual_returns.rolling(252).mean()` — rolling mean of factor-residualized returns
- **`sharpe_resid_mom_252d_skip5`:** `(residual_returns.shift(5)).rolling(252).mean() / std` — Sharpe-scaled, with 5-day skip

Scalers:
- **`vol_20_60, vol_10_60`:** Compress signal in high-vol regimes (when momentum degrades)
- **`trend_10_60`:** Scale down in downtrends

**Key Parameters & Variants Tested:**
- **Window:** 252d dominates (year-long persistence); shorter windows underperform
- **Skip:** 0d vs 5d (skip 1 week, reduces microstructure reversal noise)
- **Sharpe-scaling:** Prevents over-trading when momentum is noisy
- **Rebalance:** r21 (low cost, ~0.16% daily TO)

**Pool Sleeves:**
- `resid_mom_252d__r21__vol_20_60__cond__none` — best variant (Sharpe ~0.65)
- `sharpe_resid_mom_252d_skip5__r21__vol_20_60__cond__none` — Sharpe-scaled, skip5 (Sharpe ~0.54)
- `sharpe_resid_mom_252d_skip5__r21__vol_20_100__cond__none` — different vol baseline (Sharpe ~0.51)

**Final Portfolio Roles:**
- Not included in FINAL or GREEDY_CORE
- Positive in 2015/2018/2020 but negative in 2016/2021/2023; inconsistency ruled it out

**Main Weakness:** 2016 (−0.68) and 2021 (−0.42). Skip5 and Sharpe-scaling help but don't eliminate the inconsistency.

---

### 9. Volatility Acceleration / IVOL Explosion

**Economic Motivation:**
Volatility regime changes precede return reversals. When vol compresses (fast < slow baseline), it's expected to snap back. The IVOL-explosion variant targets the cross-sectional tails: stocks with the most explosive idiosyncratic vol jumps exhibit subsequent mean reversion. High potential but high turnover—costs are the main obstacle.

**Signal Construction:**
- **`vol_accel`:** `-(fast_vol - slow_vol)` — positive when volatility is compressing
- **`vol_regime_ret`:** `sign(vol_accel) * |returns.rolling(ret_window).mean()|` — compound vol regime with return direction
- **`ivol_explosion`:** Same vol_accel logic but active only for top 10% of vol-ratio in the cross-section

Filters:
- **`breadth_lt40`:** Breadth gate (high vol often coincides with narrow breadth)
- **`vol_10_60_up`:** Scale up when realized vol is elevated

**Pool Sleeves (from vol-trend sweep):**
- `vol_accel_20_120d__r10__vol_10_60_up__cond__breadth_lt40` — Sharpe ~1.05 gross (but high TO)
- `vol_accel_10_90d__r10__trend_50_200_mr__cond__bear_narrow_lt40` — strict crisis variant
- `vol_regime_ret_10_90_r5__r10__none` — cost-robust variant (Sharpe ~0.52 net)
- `ivol_explosion_5_60_p90__r10__none` — distinct IVOL regime (Sharpe ~0.40 net)

**Final Portfolio Roles:**
- Not in FINAL or GREEDY_CORE
- Tested as extensions to GREEDY_CORE: "raises avg drawdown without proportional Sharpe gain"
- High gross-to-net drag (~3.4% daily TO; costs consume ~0.5–0.6 Sharpe units)

**Main Weakness:** Turnover and cost. Vol-accel at r10 = ~6% daily TO → severe cost impact even at 10 bps. Better as a small satellite (5–10% of portfolio weight) rather than a full allocation.

---

### 10. Cross-Sectional Momentum (Raw)

**Economic Motivation:**
Stocks with strong 252-day rolling returns tend to continue outperforming. Classical price momentum (Jegadeesh & Titman 1993) in a long-short cross-sectional setup. The 252-day window captures intermediate-horizon persistence; shorter windows (< 120d) are contaminated by mean-reversion noise at the frequencies where the MR signals operate. Full-period net Sharpe is modest (~0.47) but the signal earns positively in 6 of 9 years and provides regime-complementary exposure to the MR-heavy portfolio.

**Signal Construction:**
Three variants tested:
- **`mom_252d`:** `returns.rolling(252).mean()` — raw 252-day rolling mean (active returns vs benchmark)
- **`sharpe_mom_120d/252d`:** `mu / sigma` on active returns — Sharpe-ratio scaled; rewards consistent winners over high-vol high-mean names
- **`low_vol_mom_120d/252d`:** `mu / sigma^2` — double-penalizes volatility; explicitly rewards low-vol consistent winners over high-vol names with equal raw momentum

The `sharpe_mom` and `low_vol_mom` variants are not just scaled versions of `mom`—they rank stocks differently. A high-return, high-vol stock that tops raw momentum rankings will rank lower under `sharpe_mom` and lower still under `low_vol_mom`. This gives `low_vol_mom` a tilt toward the low-volatility anomaly.

Filters:
- **`breadth_50_off`, `breadth_40`:** Scale down when broad rally collapses breadth (momentum loses in narrow markets)
- **`vol_20_60`, `trend_50_200`:** Vol-spike and trend scalers
- **`trend_50_200_mom`:** Momentum-specific trend filter: scale UP in uptrend (momentum strongest when trend persists)

**Key Parameters & Variants Tested:**
- **Window:** 120d and 252d; 60d and shorter are comprehensively negative (contaminated by reversal noise)
- **Signal type:** raw `mom` vs `sharpe_mom` (mu/sigma) vs `low_vol_mom` (mu/sigma²)
- **Skip:** skip1 (5-day lag, Grundy & Martin) tested; modest benefit at 120d, minimal at 252d
- **Rebalance:** r21 strongly dominant; r5 flips negative due to cost drag (~1.24% daily TO vs 0.44% at r21)

**Pool Sleeves:**
- `sharpe_mom_252d__r21__none__cond__none` — Sharpe-scaled 252d (in greedy pool)
- `sharpe_mom_120d__r5__trend_50_200_mom__cond__breadth_lt50` — selected in gap_accum seed run (2019, 2023 folds)
- `low_vol_mom_120d__r21__vol_20_60__cond__breadth_lt50` — best low-vol variant (comment: "8/9 positive, 2022 only −0.42")
- `low_vol_mom_120d__r21__none__cond__breadth_lt50` — unscaled baseline
- `low_vol_mom_120d__r5__trend_50_200_mom__cond__breadth_lt50` — weekly rebalance variant

**Full-Sweep Results (momentum sweep OBSERVATIONS.md):**
- Best full-period: `mom_252d__r21__breadth_50` (net Sharpe 0.495, ann return 1.2%, vol 2.4%)
- Best vol-exposed: `mom_252d__r21__vol_20_60` (net Sharpe 0.470, ann return 3.8%, max DD −20.1%)
- 2022 is universally bad (−1.27 to −1.31) for all top-performing full-period configs
- Residualized momentum (`resid_mom`) has zero positive configs across all 256 combinations—removing factor exposure eliminates the sector-momentum component that drives persistence

**Final Portfolio Roles:**
- Not in FINAL or GREEDY_CORE
- `low_vol_mom_120d` selected once each in the NBO50, NBO40, and r10_NBO40 greedy seed runs (2021 fold)
- `sharpe_mom_120d` selected twice in the gap_accum seed run (2019, 2023 folds)
- Consistent enough to appear across multiple seeds but not strong enough to compete with MR-family sleeves head-to-head

**Main Weakness:** 2022 (−1.27 to −1.31 for raw mom_252d). The 252-day signal can't adapt to a rapid trend reversal within a single year. Monthly rebalance is required for cost efficiency but prevents timely repositioning. The `low_vol_mom` variant is more resilient (−0.42 in 2022 per sig_fam_utils comment) because the vol-penalty reduces exposure to the high-beta names that crashed hardest.

---

### 11. Beta Momentum (Low-Beta Momentum Tilt)

**Economic Motivation:**
Combines the low-beta anomaly with momentum: stocks that have recently exhibited *decreasing beta* to SPY (their correlation with the market has fallen) while maintaining positive returns. The intuition is that a stock decoupling from the market (beta falling from 60d to 252d) while still rising is exhibiting idiosyncratic strength—it's winning on its own fundamentals, not riding the market. These stocks tend to continue outperforming because they represent genuine alpha, not market exposure.

**Signal Construction:**
- **`beta_momentum_fast_slow`:** `-(beta_fast - beta_slow) * mom_sign`
  - `beta_fast` = 60-day rolling beta to SPY
  - `beta_slow` = 252-day rolling beta to SPY
  - Signal is positive when beta is compressing (fast < slow) AND returns are positive
  - Variants: `beta_momentum_60_252d` (main), `beta_momentum_20_120d` (faster)

Filters tested:
- **`vol_10_60`:** Vol-spike scaling; best single scaler for this family
- **`trend_50_200`:** Trend filter; reduces exposure in bear markets
- **`breadth_50`, `breadth_40`:** Breadth gate; modest improvement

**Key Parameters & Variants Tested:**
- **Fast/slow windows:** (60, 252) dominant; (20, 120) also tested
- **Rebalance:** r21 outperforms r10 (same as raw momentum; holding period matters)
- **Best config:** `beta_momentum_60_252d__r21__vol_10_60` (net Sharpe 0.349, avg annual 0.305)

**Pool Sleeves:**
- `beta_momentum_60_252d__r21__trend_50_200__cond__none` — top avg_annual Sharpe (Sharpe ~0.30)
- `beta_momentum_20_120d__r21__vol_10_60__cond__none` — faster variant (Sharpe ~0.30, min −1.40)
- `beta_momentum_60_252d__r21__vol_10_60__cond__none` — best risk-adjusted (Sharpe ~0.35, min −0.47)
- `beta_momentum_60_252d__r21__vol_20_60__cond__none` — vol_20_60 variant (Sharpe ~0.28)

**Final Portfolio Roles:**
- Not in FINAL or GREEDY_CORE
- Tested as an extension to GREEDY_CORE (`_BETAMOM`): portfolio narrative notes "increases vol and drawdown without proportional Sharpe gain"
- Full-period Sharpe (~0.30–0.35) is lower than the MR and event families; doesn't earn its allocation

**Main Weakness:** Low full-period Sharpe (~0.30) relative to the bar set by better sleeves. The `beta_momentum_20_120d` variant has a worst year of −1.40, making it risky for a portfolio already carrying 2022 exposure. The beta-compression signal is mechanistically interesting but empirically too weak at the tested parameter ranges to justify inclusion over stronger alternatives.

---

## Tier 4: Weak / Excluded (Screened Out During Signal Sweep)

These families were evaluated during the signal sweep but excluded from the sleeve pool entirely. Documented here to record why they didn't make the cut.

### Vol-Compression

**Best tested:** `vol_compression_resid_10_120__r21__none`

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | avg | pct_neg |
|------|------|------|------|------|------|------|------|------|-----|---------|
| −1.11 | −0.72 | 0.95 | 0.24 | 1.54 | 0.90 | 1.38 | 0.85 | −0.43 | **0.40** | 33% |

**Why excluded:** Multi-year bad stretch in 2015–2016 and again negative in 2023. The good run of 2019–2021 is the exception. No reliable regime fingerprint — losses appear in both up (2023) and down (2015–2016) markets. Marginal avg Sharpe (0.40) doesn't justify inclusion alongside better alternatives.

---

### Sector-Relative MR

**Best tested:** `sector_rel_mr_5d__r21__trend_20_100`

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | avg | pct_neg |
|------|------|------|------|------|------|------|------|------|-----|---------|
| −0.30 | 1.52 | −0.15 | 1.25 | 0.31 | −0.39 | 0.72 | −0.68 | 1.24 | **0.39** | 44% |

**Why excluded:** Four negative years with no consistent regime pattern — equally bad in up markets (2017, 2020) and down markets (2015, 2022). The alternating on/off profile suggests high sensitivity to sector rotation timing that isn't capturable with simple gates. Only 15% of all sector-relative configs across the sweep are positive-Sharpe, which is a red flag for the family as a whole.

---

### Sector-Rel-Momentum *(Avoid)*

**Best tested:** `sector_rel_sharpe_252d_skip5__r21__breadth_50`

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | avg | pct_neg |
|------|------|------|------|------|------|------|------|------|-----|---------|
| −0.56 | −0.41 | 1.44 | −0.78 | 1.31 | 0.84 | −0.33 | −1.43 | 1.33 | **0.16** | 44% |

**Why excluded:** Five of nine years negative, including a severe −1.43 in 2022. No reliable regime fingerprint — loses in both up (2015, 2016) and down (2018, 2022) markets. Full-period net Sharpe of 0.16 is entirely driven by three strong years (2017, 2019, 2023) masking persistent losses elsewhere. The breadth scaler helps modestly but is insufficient. Do not include in portfolio construction.

---

### Volume-Momentum

**Best tested:** `vol_weighted_mom_60d__r21__trend_50_200`

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | avg | pct_neg |
|------|------|------|------|------|------|------|------|------|-----|---------|
| 0.95 | −0.77 | 0.36 | 0.78 | 0.07 | 0.26 | −0.56 | 1.66 | −0.88 | **0.21** | 33% |

**Why excluded:** No persistent edge. Occasional bright spots (2015, 2018, 2022) with no identifiable regime. Near-zero in 2019 (the year almost every other family earns); negative in 2016, 2021, and 2023. The 2022 strength (+1.66) looks like noise given the surrounding weakness. Too noisy to rely on.

---

### Volume-MR *(Avoid)*

**Best tested:** `vol_adjusted_mr_20d__r21__trend_20_100`

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | avg | pct_neg |
|------|------|------|------|------|------|------|------|------|-----|---------|
| −0.41 | −0.18 | −0.21 | 0.06 | 0.01 | 0.11 | 0.58 | 0.80 | −0.57 | **0.02** | 44% |

**Why excluded:** Near-zero average across nine years. The best years (2021–2022) barely compensate for negative years elsewhere. No meaningful edge has been found in the volume-MR family across any scaler or rebalance combination tested. Do not include in portfolio construction.

---

## Cross-Family Year Summary

Which years were broadly good or bad across all families evaluated:

| Year | Market | Broadly Good For | Broadly Bad For |
|------|--------|-----------------|-----------------|
| 2015 | Choppy bull | Breadth signals, distance-pairs | Mean-reversion, vol-trend |
| 2016 | Post-Brexit rally | Mean-reversion, monoton, sector-rel | Momentum families |
| 2017 | Low-vol bull | Bull-breadth, sharpe-mom, resid-MR | Vol-compression |
| 2018 | Rate hike / Q4 crash | Distance-pairs, resid-MR | Momentum, monoton, sector-ETF |
| 2019 | Strong bull | **Almost everything** — best year for most | Distance-pairs (mild) |
| 2020 | COVID crash + melt-up | Event, distance-pairs, momentum | Mean-reversion (brief dip) |
| 2021 | Meme/factor churn | Resid-MR, monoton, vol-compression | Momentum, event (off), sector-rel |
| 2022 | Bear market | **Event gap signals**, distance-pairs | **Momentum families** (worst year universally) |
| 2023 | AI-driven bull | **Almost everything** — broadest recovery | Vol-compression, volume-MR |

**Key observations:**
- **2022 is the universal stress test.** Every momentum family takes a large loss (−1.0 to −1.8). Mean-reversion and event signals hold up. Portfolio-level crash protection in 2022 is the single highest-value risk control.
- **2019 is universally strong.** If a sleeve can't make money in 2019, it probably has no edge.
- **2016 discriminates MR from momentum.** MR excels, raw momentum struggles. A portfolio diversified across both smooths this out.
- **Event signals are the only family with a good 2022 AND a good 2019.** Natural portfolio anchor alongside breadth-conditioned signals.

---

## Summary Table

| Family | Economic Mechanism | Tier | Pool Size | In FINAL | In GREEDY_CORE | Key Weakness |
|---|---|---|---|---|---|---|
| Distance Pairs MR | Local pairs mean reversion | 1 | 7+ | _Z60 | _NBO50 (seed) | 2023 (−0.35) |
| Bear Reversal | Crisis short-term MR | 1 | 3 | _BEAR | _BEAR | Dormant in bull mkts (by design) |
| Monotonicity | Consistency-weighted momentum | 1 | 5 | _MONO | _MONO35 | 2022 (breadth gate dampens) |
| Cross-Sectional MR | Global cumulative return reversal | 2 | 5 | _CUMRET | _CUMRET | 2015/2022 (vol scaler dampens) |
| Gap Accumulation | Event-driven gap reversals | 2 | 4 | — | _GAP (extension) | Requires trend gate |
| Sector ETF Momentum | Sector leadership, cross-asset | 2 | 6+ | — | _SSM20 | Active only when dispersion is high |
| Residual MR | Factor-residual mean reversion | 3 | 4 | — | _RZSCORE (extension) | 2019 blowup; needs trend gate |
| Residual Momentum | Idiosyncratic 252d momentum | 3 | 3 | — | — | 2016/2021 losses; inconsistent |
| Vol Acceleration | Volatility regime changes | 3 | 4 | — | — | High turnover (~3–6% daily TO) |
| Cross-Sectional Momentum | Raw 252d / Sharpe-scaled / low-vol mom | 3 | 5 | — | — | 2022 universally bad (−1.27); monthly rebalance only |
| Beta Momentum | Beta-compression + momentum tilt | 3 | 4 | — | — | Low full-period Sharpe (~0.30–0.35) |
| Vol-Compression | Residualized vol-compression timing | 4 | — | — | — | No regime fingerprint; avg 0.40, bad 2015–2016 and 2023 |
| Sector-Relative MR | Sector-vs-sector mean reversion | 4 | — | — | — | 44% neg years, no consistent regime pattern |
| Sector-Rel-Momentum | Sector-relative Sharpe momentum | 4 | — | — | — | 44% neg years, avg 0.16; do not use |
| Volume-Momentum | Volume-weighted price momentum | 4 | — | — | — | No persistent edge; avg 0.21 |
| Volume-MR | Volume-adjusted mean reversion | 4 | — | — | — | Near-zero avg (0.02) across all configs; do not use |

---

## Portfolio Construction Narrative

### Sequential FINAL (Baseline)
- **Composition:** gap_accum + dist_mr_k1_z60 + monoton_skip + dist_mr_k1_z20 + cumret_spread + bear_reversal
- **Sleeve Count:** 6
- **Avg Validation Sharpe:** ~1.47
- **Strategy:** Three orthogonal short-term MR mechanisms added sequentially; all always-on
- **Weakness:** Naive to regime; no cross-asset signal; 2022 fold is the weak link

### GREEDY_CORE (Data-Driven)
- **Composition:** NBO50 + bear_reversal + cumret_spread + monoton_35 + sector_spy_mom_20d
- **Sleeve Count:** 5
- **Avg Validation Sharpe:** ~1.67
- **Strategy:** Greedy walkforward construction—seed with best single sleeve, iteratively add best IS improvement
- **Key differences from FINAL:** Regime-gated pairs (NBO50 vs ungated Z60); tighter monoton gate (35 vs 40); sector momentum replaces gap accumulation
- **Weakness:** 2022 fold is weak (SR ~0.11)—no active signal in trending bear market

### GREEDY_CORE + gap + rzscore (Best Candidate)
- **Composition:** GREEDY_CORE + gap_accum_3d + resid_zscore_w15_w10
- **Sleeve Count:** 7
- **Avg Validation Sharpe:** ~1.98 (best tested)
- **Key improvements:** Gap fixes 2022 (0.11 → 1.34); rzscore adds consistent marginal gains without drawdown
- **Profile:** Highest avg Sharpe AND lowest max drawdown of all portfolios tested

---

## Key Principles

1. **Regime gates do most of the work.** GREEDY_CORE (gated, 5 sleeves) outperforms FINAL (ungated, 6 sleeves) by +0.20 Sharpe. The narrow-bull-off, breadth, and dispersion filters are worth 60+ bps of net Sharpe.

2. **2022 is the discriminator.** Every portfolio struggles with 2022 (trending bear market) unless it holds gap accumulation or residual MR. Classical mean-reversion signals lose; event-driven and vol signals win. The best final portfolios combine MR + vol + event.

3. **Cost discipline eliminates vol-accel from core.** Vol-acceleration signals look strong gross (~1.0+ Sharpe), but 3–6% daily turnover consumes 0.5–0.6 Sharpe units at 10 bps. The best core sleeves have 0.2–0.5% daily TO.

4. **Diversification is structural, not statistical.** Distance pairs MR, gap accumulation, and monotonicity fail in different regimes: pairs fail in narrow bull (persistent leaders diverge), gap fails in smooth trends (no spikes), monoton fails in reversing rallies. Together they cover all regimes without requiring statistical decorrelation.

5. **Sector momentum is the only cross-asset signal that clears the bar.** The sector ETF momentum family as a whole averages ~0.25 net Sharpe, but the `20d stress+disp q70` variant reliably earns ~0.76 by activating only when sectors meaningfully diverge. Unique diversification value—adds an information source absent from all other signals.
