"""v24: signal-proportional weighting within the selected 20L/20S book.

Baseline: v1 (20L/20S, 10d calendar, Sharpe 0.68, turnover 16.9%).

Instead of equal-weighting the top/bottom 20 names, weight each position
proportional to its absolute signal rank within the selected book. The #1
long gets ~2x the weight of the #20 long, concentrating capital on the
highest-conviction names without changing which names are held or when we
rebalance. Turnover should be nearly identical to v1.
"""

from __future__ import annotations

import json

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


def signal_rank_weight(positions, **cache):
    """Re-weight positions proportional to signal rank within the book.

    For each day, longs are re-weighted by their descending signal rank and
    shorts by their ascending rank (strongest signal = largest weight). The
    book stays dollar-neutral: sum(longs) = 0.5, sum(shorts) = -0.5.
    """
    signal = cache["signal"]

    weighted = positions.copy()
    for date in positions.index:
        row = positions.loc[date]
        sig_row = signal.loc[date] if date in signal.index else None
        if sig_row is None:
            continue

        longs = row[row > 0].index
        shorts = row[row < 0].index

        if len(longs) > 0 and sig_row is not None:
            sig_l = sig_row.reindex(longs).fillna(0)
            # Rank descending: highest signal = rank 1 = most weight
            ranks_l = sig_l.rank(ascending=False)
            # Linear weight: rank 1 gets N weight, rank N gets 1 weight
            weights_l = (len(longs) + 1 - ranks_l)
            weights_l = weights_l / weights_l.sum() * 0.5
            weighted.loc[date, longs] = weights_l

        if len(shorts) > 0 and sig_row is not None:
            sig_s = sig_row.reindex(shorts).fillna(0)
            # Rank ascending: most negative signal = rank 1 = largest short
            ranks_s = sig_s.rank(ascending=True)
            weights_s = (len(shorts) + 1 - ranks_s)
            weights_s = weights_s / weights_s.sum() * 0.5
            weighted.loc[date, shorts] = -weights_s

    return weighted


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v24_signal_weighted"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .scale_risk(signal_rank_weight)
        .rebalance(every=10)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
