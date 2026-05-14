from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

EXPERIMENT_NAME = "sp500-residual-momentum-costs"
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
BENCHMARK_TICKER = "SPY"
N_LONG = 20
N_SHORT = 20
MIN_PRICE_THRESHOLD = 5.0
MIN_ADV_THRESHOLD = 20_000_000.0
LIQUIDITY_TOP_N = 150
LIQUIDITY_WINDOW = 60
REBALANCE_EVERY = 10
COST_BPS = 10.0


@cache
def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download(BENCHMARK_TICKER, START_DATE, END_DATE)


@cache
def load_sector_map():
    return qs.get_sector_map(load_universe().tickers)


def study_name(version: str) -> str:
    return f"{EXPERIMENT_NAME}:{version}"


def residual_momentum_signal(lookback: int = 30, skip: int = 20, shift: int = 1):
    def signal_fn(**cache):
        residual_returns = cache["residual_returns"]
        return residual_returns.shift(skip).rolling(lookback).sum().shift(shift)

    signal_fn.__name__ = f"residual_momentum_signal_{lookback}_{skip}_{shift}"
    return signal_fn


def residual_stability_signal(lookback: int = 30, skip: int = 20, shift: int = 1):
    def signal_fn(**cache):
        residual_returns = cache["residual_returns"].shift(skip)
        mean = residual_returns.rolling(lookback).mean()
        std = residual_returns.rolling(lookback).std().replace(0.0, np.nan)
        std_error = std.div(np.sqrt(lookback))
        return mean.div(std_error).replace([np.inf, -np.inf], np.nan).shift(shift)

    signal_fn.__name__ = f"residual_stability_signal_{lookback}_{skip}_{shift}"
    return signal_fn


def residual_vol_adjusted_signal(lookback: int = 30, skip: int = 20, shift: int = 1):
    def signal_fn(**cache):
        residual_returns = cache["residual_returns"].shift(skip)
        mean = residual_returns.rolling(lookback).mean()
        vol = residual_returns.rolling(lookback).std().replace(0.0, np.nan)
        return mean.div(vol).replace([np.inf, -np.inf], np.nan).shift(shift)

    signal_fn.__name__ = f"residual_vol_adjusted_signal_{lookback}_{skip}_{shift}"
    return signal_fn


def sector_relative_transform(signal, **cache):
    sectors = pd.Series(load_sector_map()).reindex(signal.columns).fillna("Unknown")
    adjusted = signal.copy()
    for _, tickers in sectors.groupby(sectors):
        cols = tickers.index.tolist()
        adjusted.loc[:, cols] = signal.loc[:, cols].sub(signal.loc[:, cols].mean(axis=1), axis=0)
    return adjusted


def signal_abs_quantile_filter(min_quantile: float = 0.7):
    def filter_fn(signal, **cache):
        threshold = signal.abs().quantile(min_quantile, axis=1)
        mask = signal.abs().ge(threshold, axis=0)
        return signal.where(mask)

    filter_fn.__name__ = f"signal_abs_quantile_filter_{min_quantile}"
    return filter_fn


def min_universe_breadth_filter(min_names: int = 30):
    def filter_fn(signal, **cache):
        breadth = signal.notna().sum(axis=1)
        active_dates = breadth >= min_names
        return signal.where(active_dates, other=np.nan, axis=0)

    filter_fn.__name__ = f"min_universe_breadth_filter_{min_names}"
    return filter_fn


def relative_volume_strength_filter(volume_window: int = 63, volume_quantile: float = 0.7):
    def filter_fn(signal, **cache):
        volume = cache["volume"]
        ratio = volume.div(volume.rolling(volume_window).mean())
        mask = ratio.gt(ratio.quantile(volume_quantile, axis=1), axis=0)
        return signal.where(mask)

    filter_fn.__name__ = f"relative_volume_strength_filter_{volume_window}_{volume_quantile}"
    return filter_fn


def favorable_residual_regime_filter(
    vol_window: int = 20,
    regime_window: int = 126,
    max_vol_quantile: float = 0.75,
    max_corr_quantile: float = 0.75,
):
    def filter_fn(signal, **cache):
        benchmark = cache["benchmark"].fillna(0.0)
        returns = cache["returns"]

        realized_vol = benchmark.rolling(vol_window).std()
        vol_threshold = realized_vol.rolling(regime_window).quantile(max_vol_quantile)
        vol_ok = realized_vol.lt(vol_threshold).shift(1).fillna(False).astype(bool)

        n_assets = returns.notna().sum(axis=1).replace(0, np.nan)
        avg_var = returns.rolling(vol_window).var().mean(axis=1)
        ew_return = returns.mean(axis=1)
        ew_var = ew_return.rolling(vol_window).var()
        avg_corr = ((n_assets * ew_var) - avg_var).div((n_assets - 1.0) * avg_var)
        avg_corr = avg_corr.clip(-1.0, 1.0)
        corr_threshold = avg_corr.rolling(regime_window).quantile(max_corr_quantile)
        corr_ok = avg_corr.lt(corr_threshold).shift(1).fillna(False).astype(bool)

        return signal.where(vol_ok & corr_ok, axis=0)

    filter_fn.__name__ = (
        f"favorable_residual_regime_filter_{vol_window}_{regime_window}_{max_vol_quantile}"
        f"_{max_corr_quantile}"
    )
    return filter_fn


def favorable_residual_corr_regime_filter(
    vol_window: int = 20,
    regime_window: int = 126,
    max_vol_quantile: float = 0.75,
    max_corr_quantile: float = 0.75,
):
    def filter_fn(signal, **cache):
        benchmark = cache["benchmark"].fillna(0.0)
        residual_returns = cache["residual_returns"]

        realized_vol = benchmark.rolling(vol_window).std()
        vol_threshold = realized_vol.rolling(regime_window).quantile(max_vol_quantile)
        vol_ok = realized_vol.lt(vol_threshold).shift(1).fillna(False).astype(bool)

        n_assets = residual_returns.notna().sum(axis=1).replace(0, np.nan)
        avg_var = residual_returns.rolling(vol_window).var().mean(axis=1)
        ew_return = residual_returns.mean(axis=1)
        ew_var = ew_return.rolling(vol_window).var()
        avg_corr = ((n_assets * ew_var) - avg_var).div((n_assets - 1.0) * avg_var)
        avg_corr = avg_corr.clip(-1.0, 1.0)
        corr_threshold = avg_corr.rolling(regime_window).quantile(max_corr_quantile)
        corr_ok = avg_corr.lt(corr_threshold).shift(1).fillna(False).astype(bool)

        return signal.where(vol_ok & corr_ok, axis=0)

    filter_fn.__name__ = (
        "favorable_residual_corr_regime_filter_"
        f"{vol_window}_{regime_window}_{max_vol_quantile}_{max_corr_quantile}"
    )
    return filter_fn


def dynamic_breadth_regime_filter(
    breadth_window: int = 126,
    min_breadth_quantile: float = 0.35,
    vol_window: int = 20,
    regime_window: int = 126,
    max_vol_quantile: float = 0.8,
    max_corr_quantile: float = 0.8,
):
    def filter_fn(signal, **cache):
        benchmark = cache["benchmark"].fillna(0.0)
        residual_returns = cache["residual_returns"]

        breadth = signal.notna().sum(axis=1)
        breadth_threshold = breadth.rolling(breadth_window).quantile(min_breadth_quantile)
        breadth_ok = breadth.ge(breadth_threshold).shift(1).fillna(False).astype(bool)

        realized_vol = benchmark.rolling(vol_window).std()
        vol_threshold = realized_vol.rolling(regime_window).quantile(max_vol_quantile)
        vol_ok = realized_vol.le(vol_threshold).shift(1).fillna(False).astype(bool)

        n_assets = residual_returns.notna().sum(axis=1).replace(0, np.nan)
        avg_var = residual_returns.rolling(vol_window).var().mean(axis=1)
        ew_return = residual_returns.mean(axis=1)
        ew_var = ew_return.rolling(vol_window).var()
        avg_corr = ((n_assets * ew_var) - avg_var).div((n_assets - 1.0) * avg_var)
        avg_corr = avg_corr.clip(-1.0, 1.0)
        corr_threshold = avg_corr.rolling(regime_window).quantile(max_corr_quantile)
        corr_ok = avg_corr.le(corr_threshold).shift(1).fillna(False).astype(bool)

        active = breadth_ok & vol_ok & corr_ok
        return signal.where(active, axis=0)

    filter_fn.__name__ = (
        "dynamic_breadth_regime_filter_"
        f"{breadth_window}_{min_breadth_quantile}_{vol_window}_{regime_window}_"
        f"{max_vol_quantile}_{max_corr_quantile}"
    )
    return filter_fn


def build_study(
    version: str,
    *,
    signal_fn=None,
    transforms=None,
    filters=None,
    n_long: int = N_LONG,
    n_short: int = N_SHORT,
    liquidity_top_n: int = LIQUIDITY_TOP_N,
    rebalance_every: int = REBALANCE_EVERY,
    factor_model_factors=None,
    weighting: str = "equal",
):
    factor_model_factors = factor_model_factors or ["market"]

    study = Study(
        universe=load_universe(),
        benchmark=load_benchmark(),
        name=study_name(version),
    )

    factor_model_kwargs = {"factors": factor_model_factors}
    if "sector" in factor_model_factors:
        factor_model_kwargs["sector_map"] = load_sector_map()

    study = (
        study.add_factor_model(**factor_model_kwargs)
        .residualize_returns()
        .base_signal(signal_fn or residual_momentum_signal(lookback=30, skip=20, shift=1))
    )

    for transform_fn in transforms or []:
        study = study.transform_signal(transform_fn)

    for filter_fn in filters or []:
        study = study.add_filter(filter_fn)

    study = (
        study.add_tradeable_constraint(qs.min_price(MIN_PRICE_THRESHOLD))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV_THRESHOLD))
        .add_tradeable_constraint(qs.liquidity(top_n=liquidity_top_n, window=LIQUIDITY_WINDOW))
        .build_long_short(n_long=n_long, n_short=n_short)
    )

    if weighting == "equal_sharpe":
        study = study.weight_equal_sharpe(window=126)
    elif weighting == "equal_vol":
        study = study.weight_equal_vol(vol_window=63)
    else:
        study = study.weight_equal()

    study = (
        study.neutralize_positions({"market": 0})
        .rebalance(every=rebalance_every)
        .with_transaction_costs(cost_bps=COST_BPS)
    )

    return study.run()
