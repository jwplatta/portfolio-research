import json
from functools import cache

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SECTOR_ETFS, SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"


@cache
def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download("SPY", START_DATE, END_DATE)


@cache
def load_baseline_factors():
    return qs.download(["SPY", "XLK"], START_DATE, END_DATE)


@cache
def load_sector_factors():
    return qs.download(["SPY", *SECTOR_ETFS], START_DATE, END_DATE)


def demean_signal(signal, **cache):
    return signal.sub(signal.mean(axis=1), axis=0)


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
        if mask is None:
            mask = cache.get("_liquidity_mask")
        if mask is not None:
            returns = returns.where(mask)

        raw_ret = (positions.shift(1) * returns).sum(axis=1)
        equity = (1 + raw_ret).cumprod()
        equity_ma = equity.rolling(lookback).mean()
        scale = pd.Series(np.where(equity > equity_ma, 1.0, defensive_scale), index=equity.index)
        return positions.mul(scale.shift(1), axis=0)

    scaler.__name__ = f"equity_curve_regime_scale_{lookback}_{defensive_scale}"
    return scaler


def benchmark_regime_scale(fast=100, slow=200, defensive_scale=0.5):
    def scaler(positions, **cache):
        benchmark = cache["benchmark"]
        price = (1 + benchmark.fillna(0.0)).cumprod()
        fast_ma = price.rolling(fast).mean()
        slow_ma = price.rolling(slow).mean()
        scale = pd.Series(np.where(fast_ma >= slow_ma, 1.0, defensive_scale), index=price.index)
        return positions.mul(scale.shift(1), axis=0)

    scaler.__name__ = f"benchmark_regime_scale_{fast}_{slow}_{defensive_scale}"
    return scaler


def residual_vol_regime_scale(lookback=20, trigger_quantile=0.8, defensive_scale=0.5):
    def scaler(positions, **cache):
        residuals = cache["residual_returns"]
        xs_vol = residuals.rolling(lookback).std().median(axis=1)
        threshold = xs_vol.rolling(252).quantile(trigger_quantile)
        scale = pd.Series(np.where(xs_vol > threshold, defensive_scale, 1.0), index=xs_vol.index)
        return positions.mul(scale.shift(1), axis=0)

    scaler.__name__ = (
        f"residual_vol_regime_scale_{lookback}_{trigger_quantile}_{defensive_scale}"
    )
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


def beta_neutralize_positions(window=60):
    def scaler(positions, **cache):
        returns = cache["returns"]
        benchmark = cache["benchmark"]
        bench = benchmark.reindex(returns.index).fillna(0.0)

        mean_r = returns.rolling(window).mean()
        mean_b = bench.rolling(window).mean()
        mean_rb = returns.mul(bench, axis=0).rolling(window).mean()
        cov = mean_rb.sub(mean_r.mul(mean_b, axis=0))
        var_b = bench.rolling(window).var().replace(0.0, np.nan)
        betas = cov.div(var_b, axis=0).shift(1)

        adjusted = positions.copy()
        for date in positions.index:
            active = positions.loc[date]
            active = active[active != 0.0]
            if active.empty:
                continue

            beta_slice = betas.loc[date, active.index].dropna()
            if len(beta_slice) < 2:
                continue

            weights = active.reindex(beta_slice.index)
            beta_exposure = float((weights * beta_slice).sum())
            beta_norm = float((beta_slice**2).sum())
            if beta_norm == 0.0:
                continue

            neutralized = weights - (beta_exposure / beta_norm) * beta_slice
            neutralized = neutralized - neutralized.mean()
            gross = neutralized.abs().sum()
            if gross == 0.0 or pd.isna(gross):
                continue
            adjusted.loc[date, beta_slice.index] = neutralized / gross

        return adjusted.fillna(0.0)

    scaler.__name__ = f"beta_neutralize_positions_{window}"
    return scaler


def build_study(
    *,
    name,
    factors_loader=load_baseline_factors,
    signal_window=5,
    signal_shift=1,
    signal_vol_window=None,
    vol_window=5,
    vol_quantile=0.6,
    volume_window=30,
    volume_quantile=0.8,
    momentum_window=60,
    momentum_quantile=0.7,
    liquidity_top_n=250,
    liquidity_window=60,
    min_price_threshold=None,
    min_adv_threshold=None,
    extra_filters=None,
    rebalance_every=1,
    risk_scalers=None,
):
    study = (
        Study(
            universe=load_universe(),
            benchmark=load_benchmark(),
            factors=factors_loader(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            residual_mean_reversion_signal(
                window=signal_window,
                shift=signal_shift,
                vol_window=signal_vol_window,
            )
        )
        .transform_signal(demean_signal)
        .add_vol_filter(vol_window=vol_window, quantile=vol_quantile)
        .add_volume_zscore_filter(window=volume_window, min_zscore_quantile=volume_quantile)
        .add_momentum_context_filter(window=momentum_window, max_abs_quantile=momentum_quantile)
    )

    for fn in extra_filters or []:
        study = study.add_filter(fn)

    if min_price_threshold is not None:
        study = study.add_tradeable_constraint(qs.min_price(min_price_threshold))

    if min_adv_threshold is not None:
        study = study.add_tradeable_constraint(qs.min_adv(min_adv_threshold))

    study = study.add_tradeable_constraint(
        qs.liquidity(top_n=liquidity_top_n, window=liquidity_window)
    )
    study = study.build_positions(proportional_positions)

    for fn in risk_scalers or []:
        study = study.scale_risk(fn)

    study = study.rebalance(every=rebalance_every)

    return study.run()


def baseline_study(name="residual_mean_reversion"):
    return build_study(
        name=name,
        risk_scalers=[equity_curve_regime_scale()],
    )


def emit_metrics(study):
    print(json.dumps(study.metrics_dict(), default=str, sort_keys=True))
