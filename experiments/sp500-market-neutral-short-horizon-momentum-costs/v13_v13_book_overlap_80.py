"""v13: threshold-triggered rebalance using book_overlap_trigger (min_overlap=0.80).

Baseline: v1 (10d calendar rebalance, Sharpe 0.68, turnover 16.9%).

Stricter than v12 (0.70 overlap). Rebalance only when more than 20% of the book
would change. Fewer rebalances = lower turnover, but risks holding stale positions
longer and missing the signal.
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


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v13_book_overlap_80"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance_on(qs.book_overlap_trigger(n=N_LONG, min_overlap=0.80))
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
