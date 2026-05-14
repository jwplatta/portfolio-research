"""v4: concentrate to 10 long / 10 short.

Most concentrated book — highest conviction, lowest turnover per name.
Risk: idiosyncratic vol increases; may hurt Sharpe even if costs improve.
"""

from __future__ import annotations

import json

from shared import (
    LIQUIDITY_TOP_N,
    LIQUIDITY_WINDOW,
    MIN_ADV,
    MIN_PRICE,
    REBALANCE_EVERY,
    load_benchmark,
    load_universe,
    sector_beta_neutralize_positions,
    short_horizon_sector_relative_signal,
    study_name,
)

import qstudy as qs
from qstudy import Study


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v4_n10"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=10, n_short=10)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance(every=REBALANCE_EVERY)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
