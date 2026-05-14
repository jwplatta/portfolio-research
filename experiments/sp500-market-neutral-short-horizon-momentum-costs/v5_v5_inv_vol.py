"""v5: inverse-vol weighting.

Down-weights high-vol names that tend to churn more and have wider spreads.
The sweep showed inv_vol reduced drawdown from -15.3% to -11.5% at the cost of
~3pp return gross — curious whether net Sharpe improves given the cost profile.
"""

from __future__ import annotations

import json

from shared import (
    LIQUIDITY_TOP_N,
    LIQUIDITY_WINDOW,
    MIN_ADV,
    MIN_PRICE,
    N_LONG,
    N_SHORT,
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
        Study(universe=universe, benchmark=benchmark, name=study_name("v5_inv_vol"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .weight_equal_vol(vol_window=63)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance(every=REBALANCE_EVERY)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
