"""v6: only trade names in the top/bottom 30th percentile of signal strength.

Masking out the middle of the signal distribution trades only high-conviction
names, which should reduce churn on marginal re-rankings at the portfolio edges.
"""

from __future__ import annotations

import json

import pandas as pd

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


def signal_threshold_filter(signal, **cache):
    """Only trade names in the outer 30th percentile of signal strength."""
    ranks = signal.rank(axis=1, pct=True)
    mask = (ranks <= 0.30) | (ranks >= 0.70)
    return signal.where(mask)


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v6_signal_threshold"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_filter(signal_threshold_filter)
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance(every=REBALANCE_EVERY)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
