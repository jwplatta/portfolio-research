"""v25: rebalance every 5 days (accepting higher turnover for fresher signal).

Baseline: v1 (20L/20S, 10d calendar, Sharpe 0.68, turnover 16.9%).

v0 rebalanced every 5d (the default) and had Sharpe 0.66 at 29.5% turnover.
v1 slowed to 10d and got 0.68 at 16.9% — the cost saving outweighed the
signal staleness. This version re-tests 5d rebalance but with the full
sector-beta neutralization (v0 was the pre-neutralizer baseline). The
neutralizer adds alpha recovery between rebalances, so the gross edge may
hold better than in v0, making the higher turnover cost worthwhile.
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
        Study(universe=universe, benchmark=benchmark, name=study_name("v25_rebalance_5d"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance(every=5)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
