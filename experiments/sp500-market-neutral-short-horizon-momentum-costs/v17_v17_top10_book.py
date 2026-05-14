"""v17: concentrated 10L/10S book, 10d calendar rebalance.

Baseline: v1 (20L/20S, 10d calendar, Sharpe 0.68, turnover 16.9%).

Reduces from 20 to 10 names per side, trading only the strongest signal names.
Hypothesis: the top-10 by sector-relative return are more stable members than
names at rank 15-20, so the book churns less at each rebalance. Idiosyncratic
vol will be higher but fewer names = less absolute turnover per period.
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
        Study(universe=universe, benchmark=benchmark, name=study_name("v17_top10_book"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=10, n_short=10)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance(every=10)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
