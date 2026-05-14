"""v26: signal-proportional weighting, no sector-beta neutralizer.

Baseline: v1 (20L/20S, 10d calendar, Sharpe 0.68, turnover 16.9%).

v24 showed signal weighting hurt (0.47 net Sharpe) because the neutralizer
runs after build_long_short and scrambles the weights before the signal-rank
scaler can apply them cleanly. This version removes the neutralizer entirely
so signal ranks are applied directly to the equal-weight book from
build_long_short — the highest-conviction names get the most capital.
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
    short_horizon_sector_relative_signal,
    study_name,
)


def signal_rank_weight(positions, **cache):
    """Re-weight positions proportional to signal rank within the book.

    Longs: highest signal = largest weight. Shorts: most negative = largest.
    Book stays dollar-neutral: sum(longs) = 0.5, sum(shorts) = -0.5.
    """
    signal = cache["signal"]
    weighted = positions.copy()

    for date in positions.index:
        row = positions.loc[date]
        if date not in signal.index:
            continue
        sig_row = signal.loc[date]

        longs = row[row > 0].index
        shorts = row[row < 0].index

        if len(longs) > 0:
            sig_l = sig_row.reindex(longs).fillna(0)
            ranks_l = sig_l.rank(ascending=False)
            weights_l = (len(longs) + 1 - ranks_l)
            weighted.loc[date, longs] = weights_l / weights_l.sum() * 0.5

        if len(shorts) > 0:
            sig_s = sig_row.reindex(shorts).fillna(0)
            ranks_s = sig_s.rank(ascending=True)
            weights_s = (len(shorts) + 1 - ranks_s)
            weighted.loc[date, shorts] = -(weights_s / weights_s.sum() * 0.5)

    return weighted


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v26_signal_weighted_no_neutral"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(signal_rank_weight)
        .rebalance(every=10)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
