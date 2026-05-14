"""Baseline market-neutral cross-sectional momentum study."""

from __future__ import annotations

import json

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
LOOKBACK = 252
SKIP = 21
SHIFT = 1
N_LONG = 25
N_SHORT = 25
REBALANCE_EVERY = 21

universe = qs.download(SP500, START_DATE, END_DATE)
benchmark = qs.download("SPY", START_DATE, END_DATE)


def momentum_signal(**cache):
    log_returns = cache["log_returns"]
    return log_returns.shift(SKIP).rolling(LOOKBACK).sum().shift(SHIFT)


study = (
    Study(
        universe=universe,
        benchmark=benchmark,
        name="sp500_market_neutral_xsect_momentum_v0",
    )
    .base_signal(momentum_signal)
    .build_long_short(n_long=N_LONG, n_short=N_SHORT)
    .rebalance(every=REBALANCE_EVERY)
    .run()
)


if __name__ == "__main__":
    print(json.dumps(study.metrics_dict(), default=str, sort_keys=True))
