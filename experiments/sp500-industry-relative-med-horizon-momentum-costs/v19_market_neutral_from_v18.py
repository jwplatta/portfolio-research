from __future__ import annotations

import json
from functools import cache

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

EXPERIMENT_NAME = "sp500-industry-relative-med-horizon-momentum-costs"
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
BENCHMARK_TICKER = "SPY"
N_LONG = 10
N_SHORT = 10
MIN_PRICE_THRESHOLD = 5.0
MIN_ADV_THRESHOLD = 20_000_000.0
LIQUIDITY_TOP_N = 200
LIQUIDITY_WINDOW = 60
REBALANCE_EVERY = 15
COST_BPS = 10.0


@cache
def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download(BENCHMARK_TICKER, START_DATE, END_DATE)


def study_name(version: str) -> str:
    return f"{EXPERIMENT_NAME}:{version}"


def medium_horizon_relative_momentum_signal(
    long_lookback: int = 60,
    short_lookback: int = 20,
    shift: int = 1,
):
    def signal_fn(**cache):
        log_returns = cache["log_returns"]
        long_leg = log_returns.rolling(long_lookback).sum()
        short_leg = log_returns.rolling(short_lookback).sum()
        signal = long_leg.sub(short_leg)
        return signal.replace([np.inf, -np.inf], np.nan).shift(shift)

    signal_fn.__name__ = (
        f"medium_horizon_relative_momentum_signal_{long_lookback}_{short_lookback}_{shift}"
    )
    return signal_fn


@cache
def load_sector_map():
    return qs.get_sector_map(load_universe().tickers)


def sector_relative_transform(signal, **cache):
    sectors = pd.Series(load_sector_map()).reindex(signal.columns).fillna("Unknown")
    adjusted = signal.copy()
    for _, tickers in sectors.groupby(sectors):
        cols = tickers.index.tolist()
        adjusted.loc[:, cols] = signal.loc[:, cols].sub(signal.loc[:, cols].mean(axis=1), axis=0)
    return adjusted


def blended_raw_sector_signal(
    raw_weight: float = 0.75,
    sector_weight: float = 0.25,
    long_lookback: int = 60,
    short_lookback: int = 20,
    shift: int = 1,
):
    def signal_fn(**cache):
        raw_signal = medium_horizon_relative_momentum_signal(
            long_lookback=long_lookback,
            short_lookback=short_lookback,
            shift=shift,
        )(**cache)
        sector_signal = sector_relative_transform(raw_signal)
        return raw_signal.mul(raw_weight).add(sector_signal.mul(sector_weight))

    signal_fn.__name__ = (
        f"blended_raw_sector_signal_{raw_weight}_{sector_weight}_"
        f"{long_lookback}_{short_lookback}_{shift}"
    )
    return signal_fn


def signal_abs_quantile_filter(min_quantile: float = 0.55):
    def filter_fn(signal, **cache):
        threshold = signal.abs().quantile(min_quantile, axis=1)
        mask = signal.abs().ge(threshold, axis=0)
        return signal.where(mask)

    filter_fn.__name__ = f"signal_abs_quantile_filter_{min_quantile}"
    return filter_fn


def relative_volume_strength_filter(volume_window: int = 63, min_quantile: float = 0.35):
    def filter_fn(signal, **cache):
        volume = cache["volume"]
        ratio = volume.div(volume.rolling(volume_window).mean())
        threshold = ratio.quantile(min_quantile, axis=1)
        mask = ratio.ge(threshold, axis=0)
        return signal.where(mask)

    filter_fn.__name__ = f"relative_volume_strength_filter_{volume_window}_{min_quantile}"
    return filter_fn


def min_universe_breadth_filter(min_names: int = 80):
    def filter_fn(signal, **cache):
        breadth = signal.notna().sum(axis=1)
        return signal.where(breadth.ge(min_names), axis=0)

    filter_fn.__name__ = f"min_universe_breadth_filter_{min_names}"
    return filter_fn


def benchmark_trend_scale(fast: int = 100, slow: int = 200, defensive_scale: float = 0.75):
    def scaler(positions, **cache):
        benchmark = cache["benchmark"].fillna(0.0)
        benchmark_price = (1.0 + benchmark).cumprod()
        fast_ma = benchmark_price.rolling(fast).mean()
        slow_ma = benchmark_price.rolling(slow).mean()
        scale = pd.Series(
            np.where(fast_ma.ge(slow_ma), 1.0, defensive_scale),
            index=benchmark_price.index,
        )
        return positions.mul(scale.shift(1).fillna(1.0), axis=0)

    scaler.__name__ = f"benchmark_trend_scale_{fast}_{slow}_{defensive_scale}"
    return scaler


def run_study() -> dict:
    study = (
        Study(
            universe=load_universe(),
            benchmark=load_benchmark(),
            name=study_name("v19_market_neutral_from_v19"),
        )
        .base_signal(
            blended_raw_sector_signal(
                raw_weight=0.75,
                sector_weight=0.25,
                long_lookback=60,
                short_lookback=20,
                shift=1,
            )
        )
        .add_filter(signal_abs_quantile_filter(0.55))
        .add_filter(relative_volume_strength_filter(volume_window=63, min_quantile=0.35))
        .add_filter(min_universe_breadth_filter(80))
        .add_tradeable_constraint(qs.min_price(MIN_PRICE_THRESHOLD))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV_THRESHOLD))
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .weight_equal_vol(vol_window=63)
        .scale_risk(benchmark_trend_scale(fast=100, slow=200, defensive_scale=0.75))
        .rebalance(every=REBALANCE_EVERY)
        .with_transaction_costs(cost_bps=COST_BPS)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
