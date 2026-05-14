"""v19: outer-15th-percentile filter + 10L/10S + 10d calendar rebalance.

Baseline: v1 (20L/20S, 10d calendar, Sharpe 0.68, turnover 16.9%).

Stacks v17 (concentrated book) and v18 (percentile filter):
- Only the outer 15% of signal eligible — eliminates marginal boundary names
- 10L/10S — trades only the very strongest conviction names from that filtered pool
- Hypothesis: highest-conviction + fewest names = most stable book = lowest churn
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


def outer_percentile_filter(pct: float = 0.15):
    """Keep only names in the outer pct of the signal distribution each day."""

    def filt(signal, **cache):
        ranks = signal.rank(axis=1, pct=True)
        mask = (ranks <= pct) | (ranks >= 1.0 - pct)
        return signal.where(mask)

    filt.__name__ = f"outer_percentile_filter({pct})"
    return filt


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v19_pct15_top10"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_filter(outer_percentile_filter(pct=0.15))
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
