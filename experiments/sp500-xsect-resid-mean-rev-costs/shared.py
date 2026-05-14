from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy.constants import SP500

EXPERIMENT_NAME = "sp500-xsect-resid-mean-rev-costs"
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
BENCHMARK_TICKER = "SPY"
FACTOR_TICKERS = ["SPY", "XLK"]
COST_BPS = 10.0
LIQUIDITY_TOP_N = 250
LIQUIDITY_WINDOW = 60


@cache
def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download(BENCHMARK_TICKER, START_DATE, END_DATE)


@cache
def load_factors():
    return qs.download(FACTOR_TICKERS, START_DATE, END_DATE)


def study_name(version: str) -> str:
    return f"{EXPERIMENT_NAME}:{version}"


def residual_mean_reversion_signal(window=5, shift=1, vol_window=None):
    def signal_fn(**cache):
        signal = -cache["residual_returns"].rolling(window).mean()
        if shift:
            signal = signal.shift(shift)
        if vol_window is not None:
            vol = cache["residual_returns"].rolling(vol_window).std().replace(0.0, np.nan)
            signal = signal.div(vol)
        return signal.replace([np.inf, -np.inf], np.nan)

    signal_fn.__name__ = f"residual_mean_reversion_signal_{window}_{shift}_{vol_window}"
    return signal_fn


def proportional_positions(signal, **cache):
    centered = signal.sub(signal.mean(axis=1), axis=0)
    gross = centered.abs().sum(axis=1).replace(0.0, np.nan)
    return centered.div(gross, axis=0)


def zscore_clipped_positions(signal, **cache):
    signal_z = signal.sub(signal.mean(axis=1), axis=0)
    signal_z = signal_z.div(signal_z.std(axis=1), axis=0)
    signal_z = signal_z.clip(-3, 3)
    signal_z = signal_z.sub(signal_z.mean(axis=1), axis=0)
    gross = signal_z.abs().sum(axis=1).replace(0.0, np.nan)
    return signal_z.div(gross, axis=0)


def equity_curve_regime_scale(lookback=20, defensive_scale=0.25):
    def scaler(positions, **cache):
        returns = cache["returns"]
        mask = cache.get("_tradeable_mask")
        if mask is not None:
            returns = returns.where(mask)

        raw_ret = (positions.shift(1) * returns).sum(axis=1)
        equity = (1 + raw_ret).cumprod()
        equity_ma = equity.rolling(lookback).mean()
        scale = pd.Series(np.where(equity > equity_ma, 1.0, defensive_scale), index=equity.index)
        return positions.mul(scale.shift(1), axis=0)

    scaler.__name__ = f"equity_curve_regime_scale_{lookback}_{defensive_scale}"
    return scaler


def benchmark_regime_scale(fast=150, slow=250, defensive_scale=0.75):
    def scaler(positions, **cache):
        benchmark = cache["benchmark"]
        price = (1 + benchmark.fillna(0.0)).cumprod()
        fast_ma = price.rolling(fast).mean()
        slow_ma = price.rolling(slow).mean()
        scale = pd.Series(np.where(fast_ma >= slow_ma, 1.0, defensive_scale), index=price.index)
        return positions.mul(scale.shift(1), axis=0)

    scaler.__name__ = f"benchmark_regime_scale_{fast}_{slow}_{defensive_scale}"
    return scaler


def residual_shock_filter(shock_threshold=2.5, vol_window=20):
    def filter_fn(signal, **cache):
        residuals = cache["residual_returns"]
        vol = residuals.rolling(vol_window).std().replace(0.0, np.nan)
        shock = residuals.abs().div(vol)
        return signal.where(shock < shock_threshold)

    filter_fn.__name__ = f"residual_shock_filter_{shock_threshold}_{vol_window}"
    return filter_fn


def short_term_confirmation_filter(fast_window=3, slow_window=5):
    def filter_fn(signal, **cache):
        residuals = cache["residual_returns"]
        fast = -residuals.rolling(fast_window).mean()
        slow = -residuals.rolling(slow_window).mean()
        mask = np.sign(fast).eq(np.sign(slow))
        return signal.where(mask)

    filter_fn.__name__ = f"short_term_confirmation_filter_{fast_window}_{slow_window}"
    return filter_fn


def position_change_buffer(threshold=0.1):
    def scaler(positions, **cache):
        adjusted = positions.fillna(0.0).copy()
        if adjusted.empty:
            return adjusted

        for idx in range(1, len(adjusted.index)):
            prev = adjusted.iloc[idx - 1]
            target = adjusted.iloc[idx]
            delta = target - prev
            adjusted.iloc[idx] = target.where(delta.abs() > threshold, prev)

        return adjusted

    scaler.__name__ = f"position_change_buffer_{threshold}"
    return scaler


def partial_rebalance(scale=0.5):
    def scaler(positions, **cache):
        adjusted = positions.fillna(0.0).copy()
        if adjusted.empty:
            return adjusted

        for idx in range(1, len(adjusted.index)):
            prev = adjusted.iloc[idx - 1]
            target = adjusted.iloc[idx]
            adjusted.iloc[idx] = prev + scale * (target - prev)

        return adjusted

    scaler.__name__ = f"partial_rebalance_{scale}"
    return scaler


def final_weight_trade_threshold(epsilon=0.01):
    def scaler(positions, **cache):
        adjusted = positions.fillna(0.0).copy()
        if adjusted.empty:
            return adjusted

        for idx in range(1, len(adjusted.index)):
            prev = adjusted.iloc[idx - 1]
            target = adjusted.iloc[idx]
            delta = target - prev
            mixed = target.where(delta.abs() > epsilon, prev)
            mixed = mixed - mixed.mean()
            gross = mixed.abs().sum()
            if gross > 0:
                mixed = mixed / gross
            adjusted.iloc[idx] = mixed

        return adjusted

    scaler.__name__ = f"final_weight_trade_threshold_{epsilon}"
    return scaler
