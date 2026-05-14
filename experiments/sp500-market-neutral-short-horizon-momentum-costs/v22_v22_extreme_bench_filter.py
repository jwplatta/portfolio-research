"""v22: filter both extreme bench downtrend (<-2%) AND extreme uptrend (>+7%).

Baseline: v1 (20L/20S, 10d calendar, Sharpe 0.68, turnover 16.9%).

v20 (zero-out on <-2% downtrend) gave Sharpe 0.64 — filtering downtrend alone
didn't help because the 2019 drawdown was during a strong *uptrend* (bench +30%
while strategy lost). The strategy suffers in two regimes:
  1. Bench downtrend (<-2% 20d): Sharpe -0.31 — signal breaks during selloffs
  2. Strong bench uptrend (>+7% 20d): factor rotation into large-cap growth
     pollutes sector-relative signals

This version scales to 0 in BOTH tails, holding only when bench is in a
moderate range (-2% to +7% 20d return).
"""

from __future__ import annotations

import json

import qstudy as qs
from qstudy import Study

from shared import (
    LIQUIDITY_TOP_N,
    LIQUIDITY_WINDOW,
    MIN_ADV,
    MIN_PRICE,
    N_LONG,
    N_SHORT,
    load_benchmark,
    load_universe,
    sector_beta_neutralize_positions,
    short_horizon_sector_relative_signal,
    study_name,
)


def bench_moderate_regime_scaler(window: int = 20, low: float = -0.02, high: float = 0.07):
    """Hold only when bench 20d return is in [low, high]. Go flat otherwise."""
    benchmark = load_benchmark()
    bench_returns = benchmark.returns.iloc[:, 0]
    bench_20d = bench_returns.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=False)

    def scaler(positions, **cache):
        in_regime = (bench_20d.shift(1) >= low) & (bench_20d.shift(1) <= high)
        scale = in_regime.astype(float)
        scale = scale.reindex(positions.index).fillna(0.0)
        return positions.mul(scale, axis=0)

    scaler.__name__ = f"bench_moderate_regime(w={window},low={low},high={high})"
    return scaler


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v22_extreme_bench_filter"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .scale_risk(bench_moderate_regime_scaler(window=20, low=-0.02, high=0.07))
        .rebalance(every=10)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
