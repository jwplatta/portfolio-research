# SP500 Market Neutral Event Driven Log

## Baseline

### `v0`
- Baseline market-neutral event-driven strategy, built to stay close to the `event_driven_v26` sleeve used in the four-strategy portfolio experiment.
- Signal: residual 10-day price move z-score times log relative volume.
- Residualization: daily returns residualized to `SPY + sector ETFs`.
- Signal gating:
  - relative volume in the top `10%` cross-sectionally
  - absolute move z-score in the top `20%` cross-sectionally
- Portfolio:
  - equal-vol weighted long/short `20` / `20`
  - rebalance every `10` trading days
  - low-vol filter with `30` day window, keep the lowest `70%`
  - liquidity top `150`
  - explicit portfolio-level market neutralization
  - `1` day entry delay

Goal for this experiment family:
- rebuild the event-driven strategy as a genuinely market-neutral branch
- start with a close analog to `v26`
- then iterate while keeping market exposure controlled

### `v1`
- Keep the residual event signal from `v0`.
- Neutralize final positions to both `market` and `sector`.

### `v2`
- Replace raw residual 10-day move with abnormal residual move relative to residual volatility.
- Keep market-neutral final positions.

### `v3`
- Convert the event ranking to a sector-relative version.
- Demean the signal within sector before global ranking, as a first pass on the sector/industry/cluster axis.

### `v4`
- Tighten the event window to `5` days.
- Keep the baseline residual event signal, volume conditioning, and market neutralization unchanged.

### `v5`
- Tighten the event window to `3` days.
- Keep the baseline residual event signal, volume conditioning, and market neutralization unchanged.
