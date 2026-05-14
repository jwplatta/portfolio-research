"""v0 - baseline market-neutral event-driven volume shock continuation."""

from __future__ import annotations

import json
from functools import cache

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SECTOR_ETFS, SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
STUDY_NAME = "sp500_market_neutral_event_driven_v0"


@cache
def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download(["SPY"], START_DATE, END_DATE)


@cache
def load_sector_factors():
    return qs.download(["SPY", *SECTOR_ETFS], START_DATE, END_DATE)


def volume_shock_move_zscore_signal(
    event_window: int = 10,
    volume_window: int = 30,
    volume_quantile: float = 0.9,
    move_quantile: float = 0.8,
    zscore_window: int = 60,
):
    def signal_fn(**cache):
        returns = cache["residual_returns"]
        volume = cache["volume"]
        price_move = returns.rolling(event_window).sum()
        move_mean = price_move.rolling(zscore_window).mean()
        move_std = price_move.rolling(zscore_window).std().replace(0.0, np.nan)
        move_z = price_move.sub(move_mean).div(move_std)
        rel_vol = volume.div(volume.rolling(volume_window).mean().replace(0.0, np.nan))
        volume_shock = np.log(rel_vol.replace(0.0, np.nan))
        signal = move_z.mul(volume_shock)
        mask = rel_vol.ge(rel_vol.quantile(volume_quantile, axis=1), axis=0)
        mask &= move_z.abs().ge(move_z.abs().quantile(move_quantile, axis=1), axis=0)
        return signal.where(mask)

    signal_fn.__name__ = "volume_shock_move_zscore_signal"
    return signal_fn


def demean_signal(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    return signal.sub(signal.mean(axis=1), axis=0)


def delay_entry_scale(days: int = 1):
    def scaler(positions, **cache):
        return positions.shift(days).fillna(0.0)

    scaler.__name__ = f"delay_entry_scale_{days}"
    return scaler


study = (
    Study(
        universe=load_universe(),
        benchmark=load_benchmark(),
        factors=load_sector_factors(),
        name=STUDY_NAME,
    )
    .residualize_returns()
    .base_signal(
        volume_shock_move_zscore_signal(
            event_window=10,
            volume_window=30,
            volume_quantile=0.9,
            move_quantile=0.8,
            zscore_window=60,
        )
    )
    .transform_signal(demean_signal)
    .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
    .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
    .build_long_short(n_long=20, n_short=20)
    .neutralize_positions({"market": 0})
    .scale_risk(delay_entry_scale(1))
    .weight_equal_vol(vol_window=60)
    .rebalance(every=10)
    .run()
)


if __name__ == "__main__":
    print(json.dumps(study.metrics_dict(), default=str, sort_keys=True))
