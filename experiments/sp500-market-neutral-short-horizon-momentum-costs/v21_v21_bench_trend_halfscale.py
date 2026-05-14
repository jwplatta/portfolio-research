"""v21: bench-downtrend half-scale — reduce to 50% when SPY 20d return < -2%.

Baseline: v1 (20L/20S, 10d calendar, Sharpe 0.68, turnover 16.9%).

v20 (full zero-out during downtrend) hurt Sharpe: 0.64 vs v1's 0.68.
Going fully flat misses rapid recoveries. This version reduces position size
to 50% during bench downtrend instead of fully exiting. The goal is to dampen
losses without abandoning the strategy entirely.

Also tries a slightly tighter threshold: -3% instead of -2% to be more
selective about when to reduce, since -2% filters out too many trading days.
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


def bench_trend_halfscale(window: int = 20, threshold: float = -0.03, scale_down: float = 0.5):
    """Scale positions to scale_down when benchmark 20d return is below threshold."""
    benchmark = load_benchmark()
    bench_returns = benchmark.returns.iloc[:, 0]
    bench_20d = bench_returns.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=False)

    def scaler(positions, **cache):
        in_regime = bench_20d.shift(1) < threshold
        scale = in_regime.map({True: scale_down, False: 1.0}).astype(float)
        scale = scale.reindex(positions.index).fillna(1.0)
        return positions.mul(scale, axis=0)

    scaler.__name__ = f"bench_trend_halfscale(w={window},thr={threshold},scale={scale_down})"
    return scaler


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v21_bench_trend_halfscale"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .scale_risk(bench_trend_halfscale(window=20, threshold=-0.03, scale_down=0.5))
        .rebalance(every=10)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
