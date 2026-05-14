"""v18: outer-15th-percentile signal filter, then 20L/20S, 10d calendar rebalance.

Baseline: v1 (20L/20S, 10d calendar, Sharpe 0.68, turnover 16.9%).

NaN-out the middle 70% of the signal distribution before position selection.
Only names in the strongest 15% or weakest 15% of the sector-relative signal
are eligible to be held. This means build_long_short selects from a pre-filtered
pool where every candidate has genuinely strong conviction — the marginal boundary
names that drive churn are excluded before the book is formed.
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
        Study(universe=universe, benchmark=benchmark, name=study_name("v18_pct15_filter"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_filter(outer_percentile_filter(pct=0.15))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance(every=10)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
