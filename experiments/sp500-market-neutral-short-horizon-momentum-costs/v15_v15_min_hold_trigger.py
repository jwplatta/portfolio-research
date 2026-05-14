"""v15: min-hold floor (5d) + book_overlap_trigger (min_overlap=0.50).

Baseline: v1 (10d calendar rebalance, Sharpe 0.68, turnover 16.9%).

Fixes the problem in v12/v13 where the trigger fired almost daily. Two changes:
1. min_overlap lowered to 0.50 — only rebalance when >50% of names would change,
   not just >20-30% as in v12/v13.
2. 5-day minimum hold period enforced via a closure — the trigger cannot fire
   more than once per week regardless of signal movement.
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
        Study(universe=universe, benchmark=benchmark, name=study_name("v15_min_hold_trigger"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
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
