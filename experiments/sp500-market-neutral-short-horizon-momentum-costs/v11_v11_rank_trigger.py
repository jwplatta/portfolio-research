"""v11: threshold-triggered rebalance using rank_change_trigger (threshold=0.85).

Baseline: v1 (10d calendar rebalance, Sharpe 0.68, turnover 16.9%).

Instead of a fixed 10d cadence, rebalance only when the Spearman rank-correlation
between the current signal and today's proposed signal drops below 0.85. A high
threshold means we're sensitive to re-rankings — fires more often than a low threshold
but only when the ordering has genuinely shifted enough to justify paying turnover.
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
        Study(universe=universe, benchmark=benchmark, name=study_name("v11_rank_trigger"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance_on(qs.rank_change_trigger(threshold=0.85))
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
