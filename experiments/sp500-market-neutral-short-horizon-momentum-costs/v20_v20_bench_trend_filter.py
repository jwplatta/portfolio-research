"""v20: bench-downtrend regime filter — go flat when SPY 20d return < -2%.

Baseline: v1 (20L/20S, 10d calendar, Sharpe 0.68, turnover 16.9%).

Drawdown analysis of v1 shows strategy loses -5.1% ann (Sharpe -0.31) when
SPY 20d return < -2%, vs +10.6% ann when above that threshold. The three
largest drawdowns (2019 bull recovery, post-COVID reflation, 2022 bear) all
coincide with bench downtrend periods. Scaling to zero during downtrends should
eliminate these episodes and improve net Sharpe.

Regime scaler: multiply positions by 0 when bench_20d_return < -0.02,
else multiply by 1. Applied as position_scaler (after scale_risk).
"""

from __future__ import annotations

import json

import pandas as pd

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


def bench_trend_scaler(window: int = 20, threshold: float = -0.02):
    """Scale positions to 0 when benchmark 20d return is below threshold."""
    benchmark = load_benchmark()
    bench_returns = benchmark.returns.iloc[:, 0]
    bench_20d = bench_returns.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=False)

    def scaler(positions, **cache):
        scale = (bench_20d.shift(1) >= threshold).astype(float)
        scale = scale.reindex(positions.index).fillna(0.0)
        return positions.mul(scale, axis=0)

    scaler.__name__ = f"bench_trend_scaler(window={window}, threshold={threshold})"
    return scaler


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v20_bench_trend_filter"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .scale_risk(bench_trend_scaler(window=20, threshold=-0.02))
        .rebalance(every=10)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
