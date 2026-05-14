"""v16: smoothed signal (5d rolling mean) + min-hold (5d) + book_overlap(0.50).

Baseline: v1 (10d calendar rebalance, Sharpe 0.68, turnover 16.9%).

Combines both improvements from v14 and v15:
- Signal smoothing stabilizes daily rank ordering (reduces churn at book boundary)
- Min-hold floor prevents trigger from firing more than once per week
- Lower book_overlap threshold (0.50) only fires when the book has genuinely changed
"""

from __future__ import annotations

import json
from typing import Callable

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


def smooth_signal(smooth_window: int = 5):
    def transform(signal, **cache):
        return signal.rolling(smooth_window).mean()

    transform.__name__ = f"smooth_signal_{smooth_window}"
    return transform


def min_hold_trigger(inner: Callable, min_hold_days: int = 5):
    """Wrap a trigger to enforce a minimum hold period between rebalances."""
    days_since = [0]

    def trigger(current: pd.Series, proposed: pd.Series) -> bool:
        days_since[0] += 1
        if days_since[0] < min_hold_days:
            return False
        if inner(current, proposed):
            days_since[0] = 0
            return True
        return False

    trigger.__name__ = f"min_hold({min_hold_days})__{getattr(inner, '__name__', 'trigger')}"
    return trigger


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    trigger = min_hold_trigger(
        qs.book_overlap_trigger(n=N_LONG, min_overlap=0.50),
        min_hold_days=5,
    )

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v16_smoothed_min_hold"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_filter(smooth_signal(smooth_window=5))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance_on(trigger)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
