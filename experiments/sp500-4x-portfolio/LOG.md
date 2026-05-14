# SP500 Four-Strategy Portfolio Log

Combined portfolio of four independently developed SP500 strategies:
- **event_driven_v26** — volume shock z-score continuation, long/short 20/20, liquidity top 150
- **momentum_v19** — benchmark-relative momentum, volume confirmed, sector-neutral, long-only 15
- **resid_mr_v10** — residual mean reversion, proportional positions, dual regime scale
- **coint_mr_v28** — two-factor residual MR (market + sector beta stripped), beta-neutral, long/short 10/10

All strategy pipelines are self-contained in `v0_equal_weight.py`. Versions v1–v3 import from v0
and vary only the portfolio-level sleeve weighting scheme.

Date range: 2015-01-01 – 2023-12-31 | Universe: SP500 | Benchmark: SPY

---

## Weighting Experiments

### v0 — Equal weight
- 25% sleeve weight to each strategy, static.
- Baseline from `docs/combined_portfolio_study.py`.
- Expected: Sharpe ~1.37, Ann Return ~21.6%, Max DD ~-22.4%

### v1 — Equal-vol weighting
- Sleeve weights inversely proportional to trailing 126-day realized volatility of each strategy's return stream.
- Hypothesis: resid_mr_v10 has much lower vol than the other three; equal-vol should tilt toward it and reduce overall drawdown at the cost of some return.

### v2 — Equal-Sharpe weighting
- Sleeve weights proportional to trailing 126-day rolling Sharpe of each strategy's return stream (floored at zero).
- Hypothesis: coint_mr_v28 consistently has the highest Sharpe; this scheme should tilt toward it and lift portfolio Sharpe above the equal-weight baseline of 1.37.

### v3 — Mean-variance optimal weighting
- Max-Sharpe portfolio weights computed via mean-variance optimization over the trailing 126-day strategy return covariance matrix (gamma=1.0).
- Hypothesis: optimal weights exploit the near-zero correlation between resid_mr_v10 and the other three strategies, potentially achieving the best risk-adjusted return. Slower to run due to per-rebalance optimization.

---

## Portfolio-Level Neutralization Experiments

All built on v1 (equal-vol, best weighting). Neutralization applied to combined positions
after sleeve weighting via `PortfolioStudy.neutralize_positions()` — a new method added
to `PortfolioStudy` alongside `scale_risk()`.

### v4 — Beta neutralization
- `neutralize_positions({"market": 0}, beta_window=60)` on combined positions.
- Result: Sharpe 1.16, Ann Return +9.4%, Ann Vol 8.0%, Max DD -8.7%, benchmark_corr 0.13.
- Interpretation: stripping all market beta effectively removes the long-only momentum sleeve's return engine. Nearly market-neutral. Useful only if market-neutral exposure is explicitly desired.

### v5 — Sector neutralization
- `neutralize_positions({"sector": 0}, sector_map=..., beta_window=60)` on combined positions.
- Result: Sharpe 1.43, Ann Return +19.2%, Ann Vol 12.9%, Max DD -16.7%, benchmark_corr 0.78.
- Best neutralization result. Removes the residual sector tilt introduced by combining momentum (tech-heavy) with mean-reversion strategies, without meaningfully penalizing returns.

### v6 — Momentum neutralization
- `neutralize_positions({"momentum": 0}, momentum_window=126)` on combined positions.
- Result: Sharpe 1.32, Ann Return +15.4%, Ann Vol 11.3%, Max DD -12.8%, benchmark_corr 0.66.
- Interpretation: removing momentum factor exposure penalizes momentum_v19 significantly. Return reduction outweighs the drawdown improvement.

### v7 — Equal-vol with momentum_v29 sleeve
- Replace the long-only `momentum_v19` sleeve with market-neutral residual momentum `v29`.
- Keep the best sleeve weighting from v1: equal-vol at the portfolio level.
- Result: Sharpe 1.49, Ann Return +14.9%, Ann Vol 9.6%, Max DD -9.9%, benchmark_corr 0.64.
- Interpretation: swapping in the lower-beta `momentum_v29` sleeve materially reduces portfolio volatility and drawdown while improving Sharpe above the original equal-vol construction.

### v8 — Equal-vol + sector neutral with momentum_v29 sleeve
- Replace the long-only `momentum_v19` sleeve with market-neutral residual momentum `v29`.
- Keep the best neutralized construction from v5: equal-vol sleeve weighting plus portfolio-level sector neutralization.
- Result: Sharpe 1.52, Ann Return +15.2%, Ann Vol 9.6%, Max DD -9.9%, benchmark_corr 0.64.
- Interpretation: this is the best portfolio result so far. The `momentum_v29` sleeve improves risk-adjusted returns and the added sector neutralization no longer penalizes Sharpe.

### v9 — Equal-vol + sector neutral + no event sleeve
- Remove `event_driven_v26` from v8 and keep the other three sleeves.
- Goal: test whether the event sleeve is contributing meaningfully to benchmark correlation.

---

## Results Summary

| Version               | Sharpe | Ann Return | Ann Vol | Max DD  | BM Corr |
|-----------------------|--------|-----------|---------|---------|---------|
| v0 equal weight       |  1.34  |  +21.1%   |  15.1%  | -20.9%  |  0.83   |
| **v1 equal vol**      |  **1.43**  |  +19.0%   |  12.8%  | -15.2%  |  0.78   |
| v2 equal Sharpe       |  1.29  |  +20.6%   |  15.4%  | -21.7%  |  0.84   |
| v3 optimal            |  1.35  |  +20.8%   |  14.8%  | -19.2%  |  0.83   |
| v4 beta neutral       |  1.16  |   +9.4%   |   8.0%  |  -8.7%  |  0.13   |
| **v5 sector neutral** |  **1.43**  |  +19.2%   |  12.9%  | -16.7%  |  0.78   |
| v6 momentum neutral   |  1.32  |  +15.4%   |  11.3%  | -12.8%  |  0.66   |
| v7 eq-vol + mom v29   |  1.49  |  +14.9%   |   9.6%  |  -9.9%  |  0.64   |
| **v8 sector + mom v29** |  **1.52**  |  +15.2%   |   9.6%  |  -9.9%  |  0.64   |

**Best overall**: v8 (equal-vol + sector neutral + `momentum_v29`) is now the strongest portfolio variant, with Sharpe ~1.52 and dramatically lower drawdown than the original `momentum_v19` portfolio family.
v7 is a close second and shows that most of the improvement comes from replacing the old momentum sleeve with `momentum_v29`.
v4 remains the lowest-correlation portfolio variant if benchmark correlation needs to be near zero.
