from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
INDUSTRY_MAP_PATH = Path.home() / ".qstudy" / "sp500_xsect_momentum" / "industry_map.json"


@cache
def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download("SPY", START_DATE, END_DATE)


@cache
def load_sector_map():
    return qs.get_sector_map(load_universe().tickers)


@cache
def load_industry_map():
    with INDUSTRY_MAP_PATH.open() as handle:
        cached = json.load(handle)
    return {ticker: cached.get(ticker, "Unknown") for ticker in load_universe().tickers}


def emit_metrics(study):
    print(json.dumps(study.metrics_dict(), default=str, sort_keys=True))


def classic_momentum_signal(lookback: int = 252, skip: int = 21, shift: int = 1):
    def signal_fn(**cache):
        log_returns = cache["log_returns"]
        return log_returns.shift(skip).rolling(lookback).sum().shift(shift)

    signal_fn.__name__ = f"classic_momentum_signal_{lookback}_{skip}_{shift}"
    return signal_fn


def benchmark_relative_momentum_signal(lookback: int = 252, skip: int = 21, shift: int = 1):
    def signal_fn(**cache):
        log_returns = cache["log_returns"]
        benchmark = cache["benchmark"].fillna(0.0)
        stock_signal = log_returns.shift(skip).rolling(lookback).sum()
        benchmark_signal = benchmark.shift(skip).rolling(lookback).sum()
        return stock_signal.sub(benchmark_signal, axis=0).shift(shift)

    signal_fn.__name__ = f"benchmark_relative_momentum_signal_{lookback}_{skip}_{shift}"
    return signal_fn


def volatility_adjusted_momentum_signal(
    lookback: int = 252,
    skip: int = 21,
    vol_window: int = 63,
    shift: int = 1,
):
    def signal_fn(**cache):
        log_returns = cache["log_returns"]
        signal = log_returns.shift(skip).rolling(lookback).sum()
        vol = cache["returns"].rolling(vol_window).std().replace(0.0, np.nan)
        return signal.div(vol).replace([np.inf, -np.inf], np.nan).shift(shift)

    signal_fn.__name__ = (
        f"volatility_adjusted_momentum_signal_{lookback}_{skip}_{vol_window}_{shift}"
    )
    return signal_fn


def residual_momentum_signal(lookback: int = 252, skip: int = 21, shift: int = 1):
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


def residual_event_continuation_signal(
    move_window: int = 5,
    move_z_window: int = 30,
    volume_window: int = 30,
    shift: int = 1,
):
    def signal_fn(**cache):
        residual_returns = cache["residual_returns"]
        volume = cache["volume"]

        residual_move = residual_returns.rolling(move_window).sum()
        residual_move_mean = residual_move.rolling(move_z_window).mean()
        residual_move_std = residual_move.rolling(move_z_window).std().replace(0.0, np.nan)
        residual_move_z = residual_move.sub(residual_move_mean).div(residual_move_std)

        volume_mean = volume.rolling(volume_window).mean()
        volume_std = volume.rolling(volume_window).std().replace(0.0, np.nan)
        volume_z = volume.sub(volume_mean).div(volume_std)

        return residual_move_z.mul(volume_z).replace([np.inf, -np.inf], np.nan).shift(shift)

    signal_fn.__name__ = (
        f"residual_event_continuation_signal_{move_window}_{move_z_window}_{volume_window}_{shift}"
    )
    return signal_fn


def residual_threshold_event_signal(
    move_window: int = 5,
    move_z_window: int = 30,
    volume_window: int = 30,
    min_move_z: float = 1.5,
    shift: int = 1,
):
    def signal_fn(**cache):
        residual_returns = cache["residual_returns"]
        volume = cache["volume"]

        residual_move = residual_returns.rolling(move_window).sum()
        residual_move_mean = residual_move.rolling(move_z_window).mean()
        residual_move_std = residual_move.rolling(move_z_window).std().replace(0.0, np.nan)
        residual_move_z = residual_move.sub(residual_move_mean).div(residual_move_std)

        volume_mean = volume.rolling(volume_window).mean()
        volume_std = volume.rolling(volume_window).std().replace(0.0, np.nan)
        volume_z = volume.sub(volume_mean).div(volume_std)

        signal = residual_move_z.mul(volume_z)
        event_mask = residual_move_z.abs().ge(min_move_z)
        return signal.where(event_mask).replace([np.inf, -np.inf], np.nan).shift(shift)

    signal_fn.__name__ = (
        "residual_threshold_event_signal_"
        f"{move_window}_{move_z_window}_{volume_window}_{min_move_z}_{shift}"
    )
    return signal_fn


def sector_relative_transform(signal, **cache):
    sectors = pd.Series(load_sector_map()).reindex(signal.columns).fillna("Unknown")
    adjusted = signal.copy()
    for sector, tickers in sectors.groupby(sectors):
        cols = tickers.index.tolist()
        adjusted.loc[:, cols] = signal.loc[:, cols].sub(signal.loc[:, cols].mean(axis=1), axis=0)
    return adjusted


def industry_relative_transform(signal, **cache):
    industries = pd.Series(load_industry_map()).reindex(signal.columns).fillna("Unknown")
    adjusted = signal.copy()
    for industry, tickers in industries.groupby(industries):
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


def volume_confirmation_filter_factory(
    volume_window: int = 30,
    volume_quantile: float = 0.65,
    trailing_window: int = 21,
):
    def filter_fn(signal, **cache):
        volume = cache["volume"]
        volume_z = (
            volume - volume.rolling(volume_window).mean()
        ) / volume.rolling(volume_window).std()
        trailing = cache["returns"].rolling(trailing_window).mean()
        direction_ok = np.sign(signal).eq(np.sign(trailing))
        volume_ok = volume_z.gt(volume_z.quantile(volume_quantile, axis=1), axis=0)
        return signal.where(direction_ok & volume_ok)

    filter_fn.__name__ = (
        f"volume_confirmation_filter_{volume_window}_{volume_quantile}_{trailing_window}"
    )
    return filter_fn


def relative_volume_strength_filter(
    volume_window: int = 63,
    volume_quantile: float = 0.7,
):
    def filter_fn(signal, **cache):
        volume = cache["volume"]
        ratio = volume.div(volume.rolling(volume_window).mean())
        mask = ratio.gt(ratio.quantile(volume_quantile, axis=1), axis=0)
        return signal.where(mask)

    filter_fn.__name__ = f"relative_volume_strength_filter_{volume_window}_{volume_quantile}"
    return filter_fn


def min_universe_breadth_filter(min_names: int = 30):
    def filter_fn(signal, **cache):
        breadth = signal.notna().sum(axis=1)
        active_dates = breadth >= min_names
        return signal.where(active_dates, other=np.nan, axis=0)

    filter_fn.__name__ = f"min_universe_breadth_filter_{min_names}"
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
        vol_ok = realized_vol.lt(vol_threshold).shift(1).fillna(False)

        n_assets = returns.notna().sum(axis=1).replace(0, np.nan)
        avg_var = returns.rolling(vol_window).var().mean(axis=1)
        ew_return = returns.mean(axis=1)
        ew_var = ew_return.rolling(vol_window).var()
        avg_corr = ((n_assets * ew_var) - avg_var).div((n_assets - 1.0) * avg_var)
        avg_corr = avg_corr.clip(-1.0, 1.0)
        corr_threshold = avg_corr.rolling(regime_window).quantile(max_corr_quantile)
        corr_ok = avg_corr.lt(corr_threshold).shift(1).fillna(False)

        active = vol_ok & corr_ok
        return signal.where(active, axis=0)

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

        active = vol_ok & corr_ok
        return signal.where(active, axis=0)

    filter_fn.__name__ = (
        "favorable_residual_corr_regime_filter_"
        f"{vol_window}_{regime_window}_{max_vol_quantile}_{max_corr_quantile}"
    )
    return filter_fn


def proportional_long_short_positions(signal, clip: float = 3.0):
    signal_z = signal.sub(signal.mean(axis=1), axis=0)
    signal_z = signal_z.div(signal_z.std(axis=1), axis=0).clip(-clip, clip)
    signal_z = signal_z.sub(signal_z.mean(axis=1), axis=0)
    gross = signal_z.abs().sum(axis=1).replace(0.0, np.nan)
    return signal_z.div(gross, axis=0).fillna(0.0)


def sector_balanced_long_short_positions(top_k: int = 2, bottom_k: int = 2):
    sectors = pd.Series(load_sector_map())

    def position_fn(signal, **cache):
        positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        aligned_sectors = sectors.reindex(signal.columns).fillna("Unknown")

        for date in signal.index:
            row = signal.loc[date].dropna()
            if row.empty:
                continue

            sector_books: list[tuple[list[str], list[str]]] = []
            for sector, tickers in aligned_sectors.groupby(aligned_sectors):
                names = [ticker for ticker in tickers.index if ticker in row.index]
                if len(names) < top_k + bottom_k:
                    continue

                ranked = row.loc[names].sort_values()
                shorts = ranked.index[:bottom_k].tolist()
                longs = ranked.index[-top_k:].tolist()
                if not longs or not shorts:
                    continue
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


def fixed_book_sector_balanced_positions(total_longs: int = 20, total_shorts: int = 20):
    sectors = pd.Series(load_sector_map())

    def position_fn(signal, **cache):
        positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
        aligned_sectors = sectors.reindex(signal.columns).fillna("Unknown")

        for date in signal.index:
            row = signal.loc[date].dropna()
            if row.empty:
                continue

            sector_scores = {}
            sector_longs = {}
            sector_shorts = {}
            for sector, tickers in aligned_sectors.groupby(aligned_sectors):
                names = [ticker for ticker in tickers.index if ticker in row.index]
                if len(names) < 2:
                    continue
                ranked = row.loc[names].sort_values()
                sector_longs[sector] = ranked.index[::-1].tolist()
                sector_shorts[sector] = ranked.index.tolist()
                sector_scores[sector] = ranked.abs().mean()

            if not sector_scores:
                continue

            sector_order = sorted(
                sector_scores,
                key=lambda sector: (float(sector_scores[sector]), sector),
                reverse=True,
            )

            selected_longs: list[str] = []
            selected_shorts: list[str] = []
            long_counts = {sector: 0 for sector in sector_order}
            short_counts = {sector: 0 for sector in sector_order}

            while len(selected_longs) < total_longs:
                added = False
                for sector in sector_order:
                    candidates = sector_longs[sector]
                    idx = long_counts[sector]
                    while idx < len(candidates) and candidates[idx] in selected_longs:
                        idx += 1
                    if idx >= len(candidates):
                        long_counts[sector] = idx
                        continue
                    selected_longs.append(candidates[idx])
                    long_counts[sector] = idx + 1
                    added = True
                    if len(selected_longs) >= total_longs:
                        break
                if not added:
                    break

            while len(selected_shorts) < total_shorts:
                added = False
                for sector in sector_order:
                    candidates = sector_shorts[sector]
                    idx = short_counts[sector]
                    while idx < len(candidates) and candidates[idx] in selected_shorts:
                        idx += 1
                    if idx >= len(candidates):
                        short_counts[sector] = idx
                        continue
                    selected_shorts.append(candidates[idx])
                    short_counts[sector] = idx + 1
                    added = True
                    if len(selected_shorts) >= total_shorts:
                        break
                if not added:
                    break

            if not selected_longs or not selected_shorts:
                continue

            positions.loc[date, selected_longs] = 0.5 / len(selected_longs)
            positions.loc[date, selected_shorts] = -0.5 / len(selected_shorts)

        return positions

    position_fn.__name__ = f"fixed_book_sector_balanced_positions_{total_longs}_{total_shorts}"
    return position_fn


def benchmark_trend_scale(fast: int = 100, slow: int = 200, defensive_scale: float = 0.5):
    def scaler(positions, **cache):
        benchmark = cache["benchmark"].fillna(0.0)
        benchmark_price = (1.0 + benchmark).cumprod()
        fast_ma = benchmark_price.rolling(fast).mean()
        slow_ma = benchmark_price.rolling(slow).mean()
        scale = pd.Series(
            np.where(fast_ma >= slow_ma, 1.0, defensive_scale),
            index=benchmark_price.index,
        )
        return positions.mul(scale.shift(1).fillna(1.0), axis=0)

    scaler.__name__ = f"benchmark_trend_scale_{fast}_{slow}_{defensive_scale}"
    return scaler


def beta_neutralize_positions(window: int = 60):
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
    name: str,
    *,
    signal_fn=None,
    transforms=None,
    filters=None,
    n_long: int = 25,
    n_short: int = 25,
    position_builder_fn=None,
    min_price_threshold: float | None = None,
    min_adv_threshold: float | None = None,
    liquidity_top_n: int | None = None,
    liquidity_window: int = 60,
    rebalance_every: int = 21,
    factor_model_factors: list[str] | None = None,
    residualize: bool = False,
    signal_neutralization_factors: list[str] | None = None,
    position_neutralization_constraints: dict[str, float] | None = None,
    risk_scalers=None,
    vol_target: float | None = None,
    weighting: str = "equal",
):
    if signal_fn is None:
        signal_fn = classic_momentum_signal()

    study = Study(
        universe=load_universe(),
        benchmark=load_benchmark(),
        name=name,
    )

    if factor_model_factors is not None:
        factor_model_kwargs = {"factors": factor_model_factors}
        if "sector" in factor_model_factors:
            factor_model_kwargs["sector_map"] = load_sector_map()
        study = study.add_factor_model(**factor_model_kwargs)

    if residualize:
        study = study.residualize_returns()

    study = study.base_signal(signal_fn)

    if signal_neutralization_factors is not None:
        study = study.neutralize_signal(signal_neutralization_factors)

    for fn in transforms or []:
        study = study.transform_signal(fn)

    for fn in filters or []:
        study = study.add_filter(fn)

    if min_price_threshold is not None:
        study = study.add_tradeable_constraint(qs.min_price(min_price_threshold))

    if min_adv_threshold is not None:
        study = study.add_tradeable_constraint(qs.min_adv(min_adv_threshold))

    if liquidity_top_n is not None:
        study = study.add_tradeable_constraint(
            qs.liquidity(top_n=liquidity_top_n, window=liquidity_window)
        )

    if position_builder_fn is not None:
        study = study.build_positions(position_builder_fn)
    else:
        study = study.build_long_short(n_long=n_long, n_short=n_short)

    if position_neutralization_constraints is not None:
        study = study.neutralize_positions(position_neutralization_constraints)

    for fn in risk_scalers or []:
        study = study.scale_risk(fn)

    if vol_target is not None:
        study = study.scale_risk(vol_target=vol_target)

    if weighting == "equal_vol":
        study = study.weight_equal_vol(vol_window=63)
    elif weighting == "equal_sharpe":
        study = study.weight_equal_sharpe(window=126)
    else:
        study = study.weight_equal()

    study = study.rebalance(every=rebalance_every)

    return study.run()
