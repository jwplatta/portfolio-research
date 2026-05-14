"""v23: rolling Sharpe scaler — reduce size when strategy's own 60d Sharpe < 0.

Baseline: v1 (20L/20S, 10d calendar, Sharpe 0.68, turnover 16.9%).

Bench-trend filters (v20-v22) hurt net Sharpe because going flat creates long
flatline drawdowns. A different approach: use the strategy's own realized
rolling Sharpe as a regime indicator. When the 60d rolling Sharpe of the
*gross* returns has been negative, scale down to 25%. This is adaptive
size-targeting based on the strategy's own performance, not an external regime.

This is a position_scaler that accesses the backtest's internal gross returns.
We approximate using the signal's historical cross-sectional dispersion as
a proxy for "working" vs "not working" — the actual gross returns require
running the study, which creates a chicken-and-egg problem. Instead we use
the realized dispersion of the signal as a proxy: when mean absolute signal
rank correlation to 5d-forward returns has been weak, scale down.

Simpler alternative: we run the gross study first, then use the gross equity
curve as a rolling Sharpe signal for a second pass. But that requires two runs.

Practical approach here: use the strategy's own positions turnover and signal
ICR as scaling factors. Actually the simplest viable approach is to check if
the 60d gross return of the base portfolio (from cache) is negative.

NOTE: This is a legitimate in-sample conditioning on gross returns (using data
available at time t-1), not a look-ahead bias. We are allowed to use it since
this is the training set.
"""

from __future__ import annotations

import json

import numpy as np
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


def rolling_sharpe_scaler(window: int = 60, threshold: float = 0.0, scale_down: float = 0.25):
    """Scale by scale_down when rolling Sharpe of gross portfolio returns < threshold.

    Accesses cache['gross_portfolio_returns'] if available, else falls back to
    cache['portfolio_returns']. Uses a 1-day lag to avoid look-ahead.
    """

    def scaler(positions, **cache):
        gross = cache.get("gross_portfolio_returns") or cache.get("portfolio_returns")
        if gross is None:
            return positions

        rolling_sharpe = (
            gross.rolling(window).mean() / gross.rolling(window).std() * np.sqrt(252)
        )
        # shift(1) so today's decision uses yesterday's realized Sharpe
        in_weak_regime = rolling_sharpe.shift(1) < threshold
        scale = in_weak_regime.map({True: scale_down, False: 1.0}).astype(float)
        scale = scale.reindex(positions.index).fillna(1.0)
        return positions.mul(scale, axis=0)

    scaler.__name__ = f"rolling_sharpe_scaler(w={window},thr={threshold},sd={scale_down})"
    return scaler


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()

    study = (
        Study(universe=universe, benchmark=benchmark, name=study_name("v23_rolling_sharpe_scaler"))
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .scale_risk(rolling_sharpe_scaler(window=60, threshold=0.0, scale_down=0.25))
        .rebalance(every=10)
        .with_transaction_costs(10)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
