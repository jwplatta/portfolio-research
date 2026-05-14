"""v14: smoothed signal (5d rolling mean) + 10d calendar rebalance.

Baseline: v1 (10d calendar rebalance, Sharpe 0.68, turnover 16.9%).

The raw 20d sector-relative signal is a point-in-time snapshot — a single day's
return can reshuffle names near the book boundary. Taking a 5d rolling mean of
the signal damps that daily noise so names need consistent sector outperformance
to rank highly, not just a one-day spike.

This isolates the signal smoothing effect from the trigger change.
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


def smooth_signal(smooth_window: int = 5):
    """Rolling mean of the signal to stabilize daily rank ordering."""

    def transform(signal, **cache):
        return signal.rolling(smooth_window).mean()

    transform.__name__ = f"smooth_signal_{smooth_window}"
    return transform


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v14_smoothed_signal"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_filter(smooth_signal(smooth_window=5))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance(every=10)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
