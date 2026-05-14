# SP500 Market Neutral Cross-Sectional Momentum Log

The portfolio’s net returns should come primarily from relative pricing differences between securities, not from the market going up.
A genuine market-neutral momentum strategy usually requires:
- residual momentum
- industry-relative momentum
- factor-relative momentum
- very careful risk balancing
- much larger universes
- more diversification
- often lower raw returns but lower vol
 
## Baseline

### `v0`
- Baseline raw cross-sectional momentum on the `SP500`.
- Signal: trailing 12-month momentum excluding the most recent month.
- Construction: equal-weight top `25` longs and bottom `25` shorts, monthly rebalance.

## Single-Change Iterations

### `v1`
- Add basic tradeability filters.
- Keep only names above minimum price and ADV, and restrict to the most liquid subset.

### `v2`
- Switch to benchmark-relative momentum.
- Rank stocks on momentum relative to `SPY` rather than absolute trailing momentum.

### `v3`
- Switch to volatility-adjusted momentum.
- Normalize the momentum score by realized volatility.

### `v4`
- Keep only the strongest absolute signals.
- Filter the cross-section to the top momentum-magnitude cohort each rebalance.

### `v5`
- Rebalance faster.
- Move from monthly to 10-trading-day rebalancing.

### `v6`
- Concentrate the book.
- Reduce the portfolio to `15` longs and `15` shorts.

### `v7`
- Use proportional long/short sizing.
- Weight names by cross-sectional z-scored signal strength instead of equal-weight buckets.

## Composite Follow-Ups

### `v8`
- Combine the strongest early ideas into a cleaner relative-momentum branch.
- Use benchmark-relative momentum, stronger liquidity screens, concentration, and equal-Sharpe weighting.

### `v9`
- Add volume confirmation to the composite branch.
- Require stronger current volume support and enough eligible breadth.

### `v10`
- Add light risk control to the composite branch.
- Apply benchmark trend scaling plus portfolio volatility targeting.

## Neutrality Follow-Ups

### `v11`
- Add explicit portfolio-level market neutralization to `v9`.
- Keep the same signal and filters, then constrain the final book to zero market exposure.

### `v12`
- Use rolling beta neutralization instead of the built-in factor-model projection.
- Neutralize the final book using historical betas to `SPY`.

### `v13`
- Neutralize both the signal and the final positions to market exposure.
- Test whether removing market tilt earlier in the pipeline preserves more Sharpe after final neutralization.

### `v14`
- Extend the neutralization to sector as well as market exposure.
- Test whether sector drift is part of what still drives the `v9` market correlation.

### `v15`
- Pair market neutralization with proportional sizing and a narrower book.
- Test whether a more selective neutralized portfolio retains more edge than equal-weight buckets.

### `v16`
- Basic market-neutral residual momentum.
- Residualize returns to `SPY`, rank on 12-1 residual momentum, keep only basic tradeability screens, and neutralize final market exposure.

### `v17`
- Industry-relative residual momentum.
- Residualize returns to market and sector, then rank residual momentum within industries before final market neutralization.

### `v18`
- Shorter-horizon residual momentum: `60` day lookback, `5` day skip.

### `v19`
- Shorter-horizon residual momentum: `60` day lookback, `10` day skip.

### `v20`
- Shorter-horizon residual momentum: `60` day lookback, `20` day skip.

### `v21`
- Shorter-horizon residual momentum: `90` day lookback, `5` day skip.

### `v22`
- Shorter-horizon residual momentum: `90` day lookback, `10` day skip.

### `v23`
- Shorter-horizon residual momentum: `90` day lookback, `20` day skip.

### `v24`
- Shorter-horizon residual momentum: `120` day lookback, `5` day skip.

### `v25`
- Shorter-horizon residual momentum: `120` day lookback, `10` day skip.

### `v26`
- Shorter-horizon residual momentum: `120` day lookback, `20` day skip.

### `v27`
- Shorter-horizon residual momentum: `30` day lookback, `5` day skip.

### `v28`
- Shorter-horizon residual momentum: `30` day lookback, `10` day skip.

### `v29`
- Shorter-horizon residual momentum: `30` day lookback, `20` day skip.

### `v30`
- Stability-adjusted residual momentum.
- Rank on residual mean divided by residual standard error instead of raw cumulative residual return.

### `v31`
- Sector-balanced residual momentum.
- Demean the signal within sectors and take top `2` / bottom `2` names per sector with equal sector contribution.

### `v32`
- Volatility-conditioned residual momentum.
- Only trade when benchmark realized volatility and average cross-sectional correlation are below their trailing regime thresholds.

### `v33`
- Residual event continuation.
- Rank on short-horizon residual return shock times volume shock to isolate stock-specific continuation after abnormal participation.

### `v34`
- Volatility-based residual stability.
- Rank on residual mean divided by residual volatility instead of residual standard error.

### `v35`
- Fixed-book sector-balanced residual momentum.
- Keep sector-relative ranking but restore a fixed total `20` / `20` book instead of letting the name count float by sector.

### `v36`
- Residual-correlation regime filter.
- Only trade when benchmark realized volatility is calm and residual cross-sectional correlation is below its trailing threshold.

### `v37`
- Thresholded residual event continuation.
- Only rank names after statistically large residual moves, then weight the event by abnormal volume.
