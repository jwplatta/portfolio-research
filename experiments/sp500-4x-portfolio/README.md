# SP500 4x Portfolio

Cross-sectional long/short equity portfolio combining four independent alpha strategies on SP500 constituents. Backtested 2015–2023.

## Portfolio Versions

### v11 — Five Strategies, Optimal Weighting + Sector Neutral

**File:** `v11_five_strat_optimal.py`

Builds a portfolio of five strategies (adds `short_horizon_momentum_w20` to the core four), combines them with rolling mean-variance optimal weighting, and enforces zero net sector exposure at the portfolio level.

```
PortfolioStudy
  .weight_optimal(window=126, gamma=1.0)
  .neutralize_positions({"sector": 0}, sector_map=sector_map, beta_window=60)
```

- **Weighting:** Rolling 126-day mean-variance optimization with ridge regularization (`gamma=1.0`)
- **Neutralization:** Portfolio-level sector neutralization, 60-day beta window

---

### v12 — Turnover Reduction at 10bps

**File:** `v12_turnover_reduction.py`

Uses the four-strategy portfolio (no event-driven) with equal-vol weighting and sector neutralization, and sweeps rebalance frequency for the two highest-turnover strategies (`resid_mr_v10` and `short_horizon_momentum_w20`) to find cost-optimal schedules at 10bps.

```
PortfolioStudy
  .weight_equal_vol(window=30)
  .neutralize_positions({"sector": 0}, sector_map=sector_map, beta_window=60)
  .with_transaction_costs(10)
```

**Variants tested:**

| Variant | resid_mr rebalance | shm rebalance |
|---|---|---|
| v12a (baseline) | every=1 | every=5 |
| v12b | every=3 | every=5 |
| v12c | every=5 | every=5 |
| v12d | every=5 | every=10 |
| v12e | every=3 | every=10 |
| v12f | every=1 | every=10 |

Results written to `v12_turnover_results.csv`.

---

## Strategies

### 1. Event-Driven Volume Shock v26 (`event_driven_v26`)

Trades continuation of price moves confirmed by abnormal volume.

**Signal:** Volume-shock z-score × price-move z-score. For each stock, compute 10-day cumulative return z-scored over a 60-day rolling window, multiplied by log relative volume (recent vs 30-day average). Only stocks in the top 90th percentile of volume shock AND top 80th percentile of price move magnitude are eligible.

**Filters:**
- Residualized returns (sector ETF factors)
- Cross-sectional demean
- Realized vol filter: bottom 70th percentile (5-day window) — prefer low-vol names

**Universe:** Liquidity top 150 by rolling dollar volume (60-day window)

**Positions:** Long/short 20/20

**Pipeline:**
```
.weight_equal_vol(vol_window=60)
.scale_risk(delay_entry(1))     ← 1-day entry delay
.rebalance(every=10)
```

---

### 2. Cross-Sectional Momentum v19 (`momentum_v19`)

Long-only benchmark-relative momentum with volume confirmation and defensive scaling.

**Signal:** 252-day cumulative return minus SPY return, skipped 21 days and shifted 1 day. Sector-neutralized via cross-sectional orthogonalization against a market + sector factor model.

**Filters:**
- Volume confirmation: recent 5-day volume ratio ≥ 70th percentile cross-sectionally; 63-day trailing return ≥ 0
- Relative volume strength: 63-day average volume ≥ 80th percentile
- Positive momentum gate: 126-day trailing return > 0
- Minimum breadth: ≥ 15 eligible names required to trade

**Universe:** Min ADV $30M; liquidity top 90 (63-day window)

**Positions:** Long-only, top 15

**Pipeline:**
```
.weight_equal_sharpe(window=126)
.scale_risk(benchmark_trend_scale(fast=80, slow=220, defensive_scale=0.6))
                                 ← scale to 60% when SPY 80d MA < 220d MA
.scale_risk(vol_target=0.16)    ← target 16% annualized vol
.rebalance(every=10)
```

---

### 3. Residual Momentum v29 (`momentum_v29`)

Market-neutral residual momentum, short-horizon.

**Signal:** 30-day rolling sum of market-residualized returns, skipped 20 days (avoids short-term mean reversion), shifted 1 day. Residualized via a single-factor market model.

**Universe:** Min price $5; min ADV $20M; liquidity top 150 (60-day)

**Positions:** Long/short 20/20

**Pipeline:**
```
.weight_equal()
.neutralize_positions({"market": 0})  ← zero net market beta
.rebalance(every=10)
```

---

### 4. Cointegration MR v28 (`coint_mr_v28`)

Two-factor residual mean reversion: strips both market beta and sector beta before computing the signal.

**Signal:** For each stock, compute the residual after removing market beta (via OLS) and sector beta (via OLS on the market-residualized stock vs sector ETF returns). This double-residualized return series is summed over 8 days, z-scored over an 8-day window, and negated (mean-reversion). Signal clipped at ±3. Beta windows: 126 days.

**Signal transforms:**
- Cross-sectional demean
- Within-sector demean (sector-relative z-score)

**Filters:**
- Volume z-score: bottom 20th percentile excluded (30-day window)

**Universe:** Min price $5; min ADV $5M; liquidity top 150 (60-day)

**Positions:** Long/short 10/10

**Pipeline:**
```
.weight_equal_vol(vol_window=60)
.scale_risk(beta_neutralize_positions(window=20))  ← neutralize net market beta
.rebalance(every=10)
```

---

### 5. Short-Horizon Sector-Relative Momentum w20 (`short_horizon_momentum_w20`)

20-day sector-relative momentum, high-frequency rebalance.

**Signal:** 20-day cumulative return minus the sector average 20-day return for each stock. No skip (skip=0), shifted 1 day. Captures recent sector-relative outperformers.

**Universe:** Min price $5; min ADV $20M; liquidity top 150 (60-day)

**Positions:** Long/short 20/20

**Pipeline:**
```
.weight_equal()
.scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
              ← 2-pass: (1) demean within each sector, (2) remove net market beta
.rebalance(every=5)
```

The `sector_beta_neutralize_positions` scaler runs two passes per date: first demeaning weights within each GICS sector, then subtracting the net market beta projection from the weight vector and renormalizing.

---

## Data

| Parameter | Value |
|---|---|
| Universe | S&P 500 constituents |
| Benchmark | SPY |
| Sector factors | SPY + 11 GICS sector ETFs |
| Start | 2015-01-01 |
| End | 2023-12-31 |

## Running

```bash
uv run python experiments/sp500-4x-portfolio/v11_five_strat_optimal.py
uv run python experiments/sp500-4x-portfolio/v12_turnover_reduction.py
```
