# SP500 Cross-Sectional Momentum - Volume Confirmed

This directory tracks the momentum branch built with `qstudy` under a volume-confirmed filter.
The main goal was to push Sharpe above `1.0` while keeping drawdown relatively contained.

The aggregate metrics are written to [`results.csv`](./results.csv).

## Baseline

### `v1`
- Baseline long-only momentum with volume confirmation, positive-momentum gating, monthly rebalance, and benchmark trend scaling.
- Result: Sharpe `0.718`, ann return `0.132`, max drawdown `-0.302`.

## Signal / Filter Iterations

### `v2`
- Change: longer lookback, stricter volume confirmation, tighter breadth filter, and stronger liquidity / ADV gates.
- Result: Sharpe collapsed to `0.280`; too restrictive.

### `v3`
- Change: residualized momentum against `SPY` before applying the same confirmation structure.
- Result: Sharpe `0.401`, drawdown improved to `-0.277`, but return was too weak.

### `v4`
- Change: industry-relative transform plus volume and relative volume filters.
- Result: Sharpe `0.525`; not competitive with `v1`.

### `v5`
- Change: volatility-adjusted momentum with confirmation filters.
- Result: Sharpe `0.715`, close to baseline but not better.

### `v6`
- Change: benchmark-relative momentum, tighter confirmation, faster trend overlay, and `equal_sharpe` weighting.
- Result: Sharpe `0.778`, ann return `0.170`, max drawdown `-0.326`.
- Outcome: best early branch and the first one worth extending.

### `v7`
- Change: harder regime gating, narrower breadth, and more defensive cash overlay.
- Result: Sharpe `0.294`; too much signal suppression.

### `v8`
- Change: narrower position set with the same `v6` structure.
- Result: Sharpe `0.704`, drawdown worsened to `-0.438`.

### `v9`
- Change: longer confirmation windows and slower rebalance.
- Result: Sharpe `0.652`, slightly better drawdown than `v8`, but weaker overall.

### `v10`
- Change: industry-relative normalization on top of the `v6` structure.
- Result: Sharpe `0.497`; rejected.

## Construction Iterations

### `v11`
- Change: proportional long-only sizing instead of equal-weight selection.
- Result: Sharpe `0.425`; this weakened the branch.

### `v12`
- Change: softer risk control using `scale_risk(vol_target=0.16)` instead of hard regime gating.
- Result: Sharpe `0.892`, ann return `0.202`, max drawdown `-0.315`.
- Outcome: new best branch.

### `v13`
- Change: blend benchmark-relative and volatility-adjusted momentum.
- Result: Sharpe `0.442`; blend diluted the edge.

### `v14`
- Change: same `v12` structure but rebalance every 15 days.
- Result: Sharpe `0.817`, higher drawdown than `v12`.

### `v15`
- Change: `v12` with lower vol target (`0.14`).
- Result: identical to `v12` in practice.

### `v16`
- Change: `v12` with higher vol target (`0.18`).
- Result: identical to `v12` in practice.

### `v17`
- Change: `v12` with a milder benchmark trend overlay and slower rebalance.
- Result: Sharpe `0.964`, ann return `0.217`, max drawdown `-0.411`.
- Outcome: best Sharpe so far, but drawdown widened materially.

## Residual Follow-Up

### `v18`
- Change: `v12` converted to residual momentum using `residualize_returns()` against `SPY`, while keeping the same volume filters, sizing, and vol-target structure.
- Result: Sharpe `0.602`, ann return `0.139`, max drawdown `-0.372`.
- Outcome: residualization hurt both Sharpe and drawdown relative to `v12`; reject.

### `v19`
- Change: `v12` with sector-neutral signal construction using `add_factor_model(..., factors=["market", "sector"])` and `neutralize_signal(["sector"])`.
- Result: Sharpe `0.901`, ann return `0.204`, max drawdown `-0.310`.
- Outcome: slight improvement over `v12` on both Sharpe and drawdown, but turnover jumped materially.

## Summary

- The cleanest improvement came from replacing the hard regime gate with smoother `vol_target` scaling.
- The best Sharpe is currently `v17`, but it paid for that with worse drawdown.
- Residual momentum did not help this branch; it reduced Sharpe materially versus `v12`.
- The best balance is now `v19`, which slightly improves Sharpe and drawdown versus `v12` but at much higher turnover.
