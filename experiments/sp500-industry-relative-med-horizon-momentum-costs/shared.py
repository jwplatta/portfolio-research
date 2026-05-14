from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

EXPERIMENT_NAME = "sp500-industry-relative-med-horizon-momentum-costs"
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
BENCHMARK_TICKER = "SPY"
INDUSTRY_MAP_PATH = Path.home() / ".qstudy" / "sp500_xsect_momentum" / "industry_map.json"
N_LONG = 30
N_SHORT = 30
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


@cache
def load_sector_map():
    return qs.get_sector_map(load_universe().tickers)


@cache
def load_industry_map():
    tickers = load_universe().tickers
    if not INDUSTRY_MAP_PATH.exists():
        return {ticker: "Unknown" for ticker in tickers}

    with INDUSTRY_MAP_PATH.open() as handle:
        cached = json.load(handle)

    return {ticker: cached.get(ticker, "Unknown") for ticker in tickers}


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


def industry_relative_transform(signal, **cache):
    industries = pd.Series(load_industry_map()).reindex(signal.columns).fillna("Unknown")
    adjusted = signal.copy()
    for _, tickers in industries.groupby(industries):
        cols = tickers.index.tolist()
        adjusted.loc[:, cols] = signal.loc[:, cols].sub(signal.loc[:, cols].mean(axis=1), axis=0)
    return adjusted


def sector_relative_transform(signal, **cache):
    sectors = pd.Series(load_sector_map()).reindex(signal.columns).fillna("Unknown")
    adjusted = signal.copy()
    for _, tickers in sectors.groupby(sectors):
        cols = tickers.index.tolist()
        adjusted.loc[:, cols] = signal.loc[:, cols].sub(signal.loc[:, cols].mean(axis=1), axis=0)
    return adjusted


def hybrid_sector_industry_transform(sector_weight: float = 0.5):
    industry_weight = 1.0 - sector_weight

    def transform_fn(signal, **cache):
        sector_signal = sector_relative_transform(signal, **cache)
        industry_signal = industry_relative_transform(signal, **cache)
        return sector_signal.mul(sector_weight).add(industry_signal.mul(industry_weight))

    transform_fn.__name__ = f"hybrid_sector_industry_transform_{sector_weight}"
    return transform_fn


def signal_abs_quantile_filter(min_quantile: float = 0.55):
    def filter_fn(signal, **cache):
        threshold = signal.abs().quantile(min_quantile, axis=1)
        mask = signal.abs().ge(threshold, axis=0)
        return signal.where(mask)

    filter_fn.__name__ = f"signal_abs_quantile_filter_{min_quantile}"
    return filter_fn


def min_universe_breadth_filter(min_names: int = 80):
    def filter_fn(signal, **cache):
        breadth = signal.notna().sum(axis=1)
        return signal.where(breadth.ge(min_names), axis=0)

    filter_fn.__name__ = f"min_universe_breadth_filter_{min_names}"
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


def blended_raw_sector_signal(
    raw_weight: float = 0.5,
    sector_weight: float = 0.5,
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


def benchmark_trend_gate_filter(fast: int = 100, slow: int = 200):
    def filter_fn(signal, **cache):
        benchmark = cache["benchmark"].fillna(0.0)
        benchmark_price = (1.0 + benchmark).cumprod()
        fast_ma = benchmark_price.rolling(fast).mean()
        slow_ma = benchmark_price.rolling(slow).mean()
        regime_ok = fast_ma.ge(slow_ma).shift(1).astype("boolean").fillna(False).astype(bool)
        return signal.where(regime_ok, axis=0)

    filter_fn.__name__ = f"benchmark_trend_gate_filter_{fast}_{slow}"
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


def proportional_long_short_positions(signal, clip: float = 3.0):
    signal_z = signal.sub(signal.mean(axis=1), axis=0)
    signal_z = signal_z.div(signal_z.std(axis=1), axis=0).clip(-clip, clip)
    signal_z = signal_z.sub(signal_z.mean(axis=1), axis=0)
    gross = signal_z.abs().sum(axis=1).replace(0.0, np.nan)
    return signal_z.div(gross, axis=0).fillna(0.0)


def industry_balanced_long_short_positions(top_k: int = 1, bottom_k: int = 1):
    industries = pd.Series(load_industry_map())

    def position_fn(signal, **cache):
        positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        aligned_industries = industries.reindex(signal.columns).fillna("Unknown")

        for date in signal.index:
            row = signal.loc[date].dropna()
            if row.empty:
                continue

            industry_books: list[tuple[list[str], list[str]]] = []
            for _, tickers in aligned_industries.groupby(aligned_industries):
                names = [ticker for ticker in tickers.index if ticker in row.index]
                if len(names) < top_k + bottom_k:
                    continue

                ranked = row.loc[names].sort_values()
                shorts = ranked.index[:bottom_k].tolist()
                longs = ranked.index[-top_k:].tolist()
                if longs and shorts:
                    industry_books.append((longs, shorts))

            if not industry_books:
                continue

            industry_weight = 1.0 / len(industry_books)
            for longs, shorts in industry_books:
                long_weight = 0.5 * industry_weight / len(longs)
                short_weight = -0.5 * industry_weight / len(shorts)
                positions.loc[date, longs] = long_weight
                positions.loc[date, shorts] = short_weight

        return positions

    position_fn.__name__ = f"industry_balanced_long_short_positions_{top_k}_{bottom_k}"
    return position_fn


def sector_balanced_long_short_positions(top_k: int = 1, bottom_k: int = 1):
    sectors = pd.Series(load_sector_map())

    def position_fn(signal, **cache):
        positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        aligned_sectors = sectors.reindex(signal.columns).fillna("Unknown")

        for date in signal.index:
            row = signal.loc[date].dropna()
            if row.empty:
                continue

            sector_books: list[tuple[list[str], list[str]]] = []
            for _, tickers in aligned_sectors.groupby(aligned_sectors):
                names = [ticker for ticker in tickers.index if ticker in row.index]
                if len(names) < top_k + bottom_k:
                    continue

                ranked = row.loc[names].sort_values()
                shorts = ranked.index[:bottom_k].tolist()
                longs = ranked.index[-top_k:].tolist()
                if longs and shorts:
                    sector_books.append((longs, shorts))

            if not sector_books:
                continue

            sector_weight = 1.0 / len(sector_books)
            for longs, shorts in sector_books:
                long_weight = 0.5 * sector_weight / len(longs)
                short_weight = -0.5 * sector_weight / len(shorts)
                positions.loc[date, longs] = long_weight
                positions.loc[date, shorts] = short_weight

        return positions

    position_fn.__name__ = f"sector_balanced_long_short_positions_{top_k}_{bottom_k}"
    return position_fn


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
    weighting: str = "equal_vol",
    factor_model_factors=None,
    position_neutralization_constraints=None,
    position_builder_fn=None,
    factor_model_beta_window: int = 60,
    risk_scalers=None,
    neutralize_after_risk_scalers: bool = False,
):
    study = Study(
        universe=load_universe(),
        benchmark=load_benchmark(),
        name=study_name(version),
    )

    factor_model_factors = factor_model_factors or []
    position_neutralization_constraints = position_neutralization_constraints or {}

    if factor_model_factors:
        factor_model_kwargs = {"factors": factor_model_factors}
        if "sector" in factor_model_factors:
            factor_model_kwargs["sector_map"] = load_sector_map()
        study = study.add_factor_model(beta_window=factor_model_beta_window, **factor_model_kwargs)

    study = study.base_signal(
        signal_fn or medium_horizon_relative_momentum_signal(long_lookback=60, short_lookback=20)
    )

    for transform_fn in transforms or []:
        study = study.transform_signal(transform_fn)

    for filter_fn in filters or []:
        study = study.add_filter(filter_fn)

    study = (
        study.add_tradeable_constraint(qs.min_price(MIN_PRICE_THRESHOLD))
        .add_tradeable_constraint(qs.min_adv(MIN_ADV_THRESHOLD))
        .add_tradeable_constraint(qs.liquidity(top_n=liquidity_top_n, window=LIQUIDITY_WINDOW))
    )

    if position_builder_fn is not None:
        study = study.build_positions(position_builder_fn)
    else:
        study = study.build_long_short(n_long=n_long, n_short=n_short)

    if weighting == "equal_vol":
        study = study.weight_equal_vol(vol_window=63)
    elif weighting == "equal":
        study = study.weight_equal()
    elif weighting == "proportional":
        pass
    else:
        raise ValueError(f"Unsupported weighting mode: {weighting}")

    if position_neutralization_constraints and not neutralize_after_risk_scalers:
        study = study.neutralize_positions(position_neutralization_constraints)

    for risk_scaler in risk_scalers or []:
        study = study.scale_risk(risk_scaler)

    if position_neutralization_constraints and neutralize_after_risk_scalers:
        study = study.neutralize_positions(position_neutralization_constraints)

    study = study.rebalance(every=rebalance_every).with_transaction_costs(cost_bps=COST_BPS)

    return study.run()
