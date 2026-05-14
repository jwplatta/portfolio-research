from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SECTOR_ETFS, SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
CACHE_DIR = Path.home() / ".qstudy" / "sp500_xsect_momentum"


@cache
def load_sp500_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_sector_universe():
    return qs.download(SECTOR_ETFS, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download("SPY", START_DATE, END_DATE)


@cache
def load_spy_factor():
    return qs.download(["SPY"], START_DATE, END_DATE)


@cache
def load_sector_factors():
    return qs.download(["SPY", *SECTOR_ETFS], START_DATE, END_DATE)


@cache
def load_sector_map():
    return qs.get_sector_map(load_sp500_universe().tickers)


@cache
def load_industry_map():
    path = CACHE_DIR / "industry_map.json"
    if not path.exists():
        return {ticker: "Unknown" for ticker in load_sp500_universe().tickers}

    with open(path) as f:
        cached = json.load(f)

    return {ticker: cached.get(ticker, "Unknown") for ticker in load_sp500_universe().tickers}


def emit_metrics(study):
    print(json.dumps(study.metrics_dict(), default=str, sort_keys=True))


def save_study(study, name: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    study.save(CACHE_DIR / f"{name}.pkl")


def benchmark_trend_scale(fast: int = 100, slow: int = 200, defensive_scale: float = 0.25):
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


def sector_relative_strength_scale(defensive_scale: float = 0.5):
    def scaler(positions, **cache):
        benchmark = cache["benchmark"].fillna(0.0)
        benchmark_trailing = benchmark.rolling(126).mean()
        scale = pd.Series(
            np.where(benchmark_trailing > 0.0, 1.0, defensive_scale),
            index=benchmark.index,
        )
        return positions.mul(scale.shift(1).fillna(1.0), axis=0)

    scaler.__name__ = f"sector_relative_strength_scale_{defensive_scale}"
    return scaler


def _rolling_momentum_frame(
    returns: pd.DataFrame,
    lookback: int,
    skip: int,
) -> pd.DataFrame:
    recent = returns.rolling(skip).mean() if skip > 0 else 0.0
    signal = returns.rolling(lookback).mean() - recent
    return signal.replace([np.inf, -np.inf], np.nan)


def classic_momentum_signal(lookback: int = 252, skip: int = 21, shift: int = 1):
    def signal_fn(**cache):
        signal = _rolling_momentum_frame(cache["_active_returns"], lookback=lookback, skip=skip)
        return signal.shift(shift)

    signal_fn.__name__ = f"classic_momentum_signal_{lookback}_{skip}_{shift}"
    return signal_fn


def benchmark_relative_momentum_signal(lookback: int = 126, skip: int = 21, shift: int = 1):
    def signal_fn(**cache):
        benchmark = cache["benchmark"].fillna(0.0)
        signal = _rolling_momentum_frame(cache["_active_returns"], lookback=lookback, skip=skip)
        benchmark_signal = _rolling_momentum_frame(
            benchmark.to_frame("SPY"),
            lookback=lookback,
            skip=skip,
        )["SPY"]
        relative = signal.sub(benchmark_signal, axis=0)
        return relative.shift(shift)

    signal_fn.__name__ = f"benchmark_relative_momentum_signal_{lookback}_{skip}_{shift}"
    return signal_fn


def residual_momentum_signal(lookback: int = 126, skip: int = 21, shift: int = 1):
    def signal_fn(**cache):
        residual_returns = cache["residual_returns"]
        signal = _rolling_momentum_frame(residual_returns, lookback=lookback, skip=skip)
        return signal.shift(shift)

    signal_fn.__name__ = f"residual_momentum_signal_{lookback}_{skip}_{shift}"
    return signal_fn


def volatility_adjusted_momentum_signal(
    lookback: int = 252,
    skip: int = 21,
    vol_window: int = 63,
    shift: int = 1,
):
    def signal_fn(**cache):
        signal = _rolling_momentum_frame(cache["_active_returns"], lookback=lookback, skip=skip)
        vol = cache["returns"].rolling(vol_window).std().replace(0.0, np.nan)
        signal = signal.div(vol)
        return signal.replace([np.inf, -np.inf], np.nan).shift(shift)

    signal_fn.__name__ = (
        f"volatility_adjusted_momentum_signal_{lookback}_{skip}_{vol_window}_{shift}"
    )
    return signal_fn


def cross_sectional_demean(signal, **cache):
    return signal.sub(signal.mean(axis=1), axis=0)


def sector_relative_transform(signal, **cache):
    sector_map = pd.Series(load_sector_map())
    aligned = sector_map.reindex(signal.columns).fillna("Unknown")
    adjusted = signal.copy()
    for sector, tickers in aligned.groupby(aligned):
        cols = tickers.index.tolist()
        adjusted.loc[:, cols] = signal.loc[:, cols].sub(signal.loc[:, cols].mean(axis=1), axis=0)
    return adjusted


def industry_relative_transform(signal, **cache):
    aligned = pd.Series(load_industry_map()).reindex(signal.columns).fillna("Unknown")
    adjusted = signal.copy()
    for industry, tickers in aligned.groupby(aligned):
        cols = tickers.index.tolist()
        adjusted.loc[:, cols] = signal.loc[:, cols].sub(signal.loc[:, cols].mean(axis=1), axis=0)
    return adjusted


def positive_momentum_filter(signal, **cache):
    return signal.where(signal > 0.0)


def volume_confirmation_filter(
    signal,
    **cache,
):
    volume = cache["volume"]
    volume_z = (volume - volume.rolling(30).mean()) / volume.rolling(30).std()
    trailing = cache["returns"].rolling(21).mean()
    mask = volume_z.gt(volume_z.quantile(0.6, axis=1), axis=0) & trailing.gt(0.0)
    return signal.where(mask)


def volume_confirmation_filter_factory(
    volume_window: int = 30,
    volume_quantile: float = 0.6,
    trailing_window: int = 21,
    min_trailing_return: float = 0.0,
):
    def filter_fn(signal, **cache):
        volume = cache["volume"]
        volume_z = (
            volume - volume.rolling(volume_window).mean()
        ) / volume.rolling(volume_window).std()
        trailing = cache["_active_returns"].rolling(trailing_window).mean()
        mask = volume_z.gt(volume_z.quantile(volume_quantile, axis=1), axis=0)
        mask &= trailing.gt(min_trailing_return)
        return signal.where(mask)

    filter_fn.__name__ = (
        f"volume_confirmation_filter_{volume_window}_{volume_quantile}_"
        f"{trailing_window}_{min_trailing_return}"
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


def trend_regime_filter(
    lookback: int = 126,
    min_benchmark_return: float = 0.0,
):
    def filter_fn(signal, **cache):
        benchmark = cache["benchmark"].fillna(0.0)
        benchmark_trend = benchmark.rolling(lookback).mean()
        active_dates = benchmark_trend > min_benchmark_return
        return signal.where(active_dates, other=np.nan, axis=0)

    filter_fn.__name__ = f"trend_regime_filter_{lookback}_{min_benchmark_return}"
    return filter_fn


def blended_signal(*signal_fns, weights=None):
    if weights is None:
        weights = [1.0 / len(signal_fns)] * len(signal_fns)

    def signal_fn(**cache):
        combined = None
        for weight, fn in zip(weights, signal_fns, strict=False):
            value = fn(**cache)
            combined = value.mul(weight) if combined is None else combined.add(value.mul(weight))
        return combined

    signal_fn.__name__ = "blended_signal"
    return signal_fn


def min_universe_breadth_filter(min_names: int = 25):
    def filter_fn(signal, **cache):
        breadth = signal.notna().sum(axis=1)
        active_dates = breadth >= min_names
        return signal.where(active_dates, other=np.nan, axis=0)

    filter_fn.__name__ = f"min_universe_breadth_filter_{min_names}"
    return filter_fn


def proportional_long_only_positions(top_n: int = 15):
    def builder(signal):
        rank = signal.rank(axis=1, ascending=False, na_option="bottom")
        selected = signal.where(rank <= top_n).clip(lower=0.0)
        gross = selected.sum(axis=1).replace(0.0, np.nan)
        return selected.div(gross, axis=0).fillna(0.0)

    builder.__name__ = f"proportional_long_only_positions_{top_n}"
    return builder


def build_long_only_momentum_study(
    *,
    name: str,
    universe_loader=load_sp500_universe,
    base_signal_fn=None,
    transforms=None,
    filters=None,
    n_positions: int = 40,
    position_builder_fn=None,
    liquidity_top_n: int | None = 200,
    liquidity_window: int = 60,
    min_price_threshold: float = 5.0,
    min_adv_threshold: float = 10_000_000.0,
    rebalance_every: int = 21,
    risk_scalers=None,
    vol_target: float | None = None,
    weighting: str = "equal_vol",
    residualize: bool = False,
    factors_loader=load_spy_factor,
):
    universe = universe_loader()
    benchmark = load_benchmark()
    factors = factors_loader() if residualize else None

    study = Study(
        universe=universe,
        benchmark=benchmark,
        factors=factors,
        name=name,
    )

    if residualize:
        study = study.residualize_returns()

    study = study.base_signal(base_signal_fn)

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
        study = study.build_long_only(n=n_positions)

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


def build_sector_rotation_study(name: str):
    return build_long_only_momentum_study(
        name=name,
        universe_loader=load_sector_universe,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=126, skip=21, shift=1),
        filters=[positive_momentum_filter, min_universe_breadth_filter(min_names=3)],
        n_positions=3,
        liquidity_top_n=None,
        min_price_threshold=None,
        min_adv_threshold=None,
        rebalance_every=21,
        risk_scalers=[sector_relative_strength_scale(defensive_scale=0.35)],
        weighting="equal",
    )


def build_cross_sectional_momentum_study(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=classic_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[positive_momentum_filter],
        n_positions=50,
        liquidity_top_n=200,
        rebalance_every=21,
        risk_scalers=[benchmark_trend_scale(fast=100, slow=200, defensive_scale=0.35)],
        weighting="equal_vol",
    )


def build_residual_momentum_study(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=residual_momentum_signal(lookback=126, skip=21, shift=1),
        transforms=[cross_sectional_demean],
        filters=[positive_momentum_filter],
        n_positions=35,
        liquidity_top_n=175,
        rebalance_every=21,
        risk_scalers=[benchmark_trend_scale(fast=100, slow=200, defensive_scale=0.4)],
        weighting="equal_vol",
        residualize=True,
        factors_loader=load_spy_factor,
    )


def build_industry_relative_momentum_study(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=classic_momentum_signal(lookback=252, skip=21, shift=1),
        transforms=[industry_relative_transform],
        filters=[positive_momentum_filter],
        n_positions=40,
        liquidity_top_n=200,
        rebalance_every=21,
        risk_scalers=[benchmark_trend_scale(fast=100, slow=200, defensive_scale=0.35)],
        weighting="equal_vol",
    )


def build_volatility_adjusted_momentum_study(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=volatility_adjusted_momentum_signal(
            lookback=252,
            skip=21,
            vol_window=63,
            shift=1,
        ),
        filters=[positive_momentum_filter],
        n_positions=45,
        liquidity_top_n=225,
        rebalance_every=21,
        risk_scalers=[benchmark_trend_scale(fast=100, slow=200, defensive_scale=0.4)],
        weighting="equal_vol",
    )


def build_volume_confirmed_momentum_study(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=classic_momentum_signal(lookback=189, skip=21, shift=1),
        filters=[volume_confirmation_filter, positive_momentum_filter],
        n_positions=35,
        liquidity_top_n=175,
        rebalance_every=21,
        risk_scalers=[benchmark_trend_scale(fast=100, slow=200, defensive_scale=0.35)],
        weighting="equal_vol",
    )


def build_volume_confirmed_iteration_v2(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=classic_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=40,
                volume_quantile=0.7,
                trailing_window=21,
                min_trailing_return=0.0005,
            ),
            positive_momentum_filter,
            min_universe_breadth_filter(min_names=20),
        ],
        n_positions=25,
        liquidity_top_n=125,
        min_adv_threshold=20_000_000.0,
        rebalance_every=21,
        risk_scalers=[benchmark_trend_scale(fast=80, slow=180, defensive_scale=0.25)],
        weighting="equal_vol",
    )


def build_volume_confirmed_iteration_v3(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=residual_momentum_signal(lookback=189, skip=21, shift=1),
        transforms=[cross_sectional_demean],
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.65,
                trailing_window=21,
                min_trailing_return=0.0,
            ),
            positive_momentum_filter,
            min_universe_breadth_filter(min_names=20),
        ],
        n_positions=25,
        liquidity_top_n=150,
        min_adv_threshold=15_000_000.0,
        rebalance_every=21,
        risk_scalers=[benchmark_trend_scale(fast=100, slow=200, defensive_scale=0.25)],
        weighting="equal_sharpe",
        residualize=True,
        factors_loader=load_spy_factor,
    )


def build_volume_confirmed_iteration_v4(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=classic_momentum_signal(lookback=252, skip=21, shift=1),
        transforms=[industry_relative_transform],
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.65,
                trailing_window=42,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.7),
            positive_momentum_filter,
            min_universe_breadth_filter(min_names=20),
        ],
        n_positions=20,
        liquidity_top_n=125,
        min_adv_threshold=20_000_000.0,
        rebalance_every=21,
        risk_scalers=[benchmark_trend_scale(fast=100, slow=200, defensive_scale=0.2)],
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v5(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=volatility_adjusted_momentum_signal(
            lookback=252,
            skip=21,
            vol_window=63,
            shift=1,
        ),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.6,
                trailing_window=21,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.75),
            positive_momentum_filter,
            trend_regime_filter(lookback=126, min_benchmark_return=0.0),
            min_universe_breadth_filter(min_names=15),
        ],
        n_positions=20,
        liquidity_top_n=100,
        min_adv_threshold=25_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=80, slow=200, defensive_scale=0.15)],
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v6(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            trend_regime_filter(lookback=126, min_benchmark_return=0.0002),
            min_universe_breadth_filter(min_names=15),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=60, slow=180, defensive_scale=0.1)],
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v7(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            trend_regime_filter(lookback=126, min_benchmark_return=0.0005),
            min_universe_breadth_filter(min_names=12),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=50, slow=200, defensive_scale=0.0)],
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v8(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            trend_regime_filter(lookback=126, min_benchmark_return=0.0002),
            min_universe_breadth_filter(min_names=10),
        ],
        n_positions=10,
        liquidity_top_n=75,
        min_adv_threshold=35_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=60, slow=180, defensive_scale=0.1)],
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v9(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=45,
                volume_quantile=0.7,
                trailing_window=84,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=84, volume_quantile=0.8),
            positive_momentum_filter,
            trend_regime_filter(lookback=126, min_benchmark_return=0.0002),
            min_universe_breadth_filter(min_names=12),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=21,
        risk_scalers=[benchmark_trend_scale(fast=60, slow=180, defensive_scale=0.1)],
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v10(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        transforms=[industry_relative_transform],
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            trend_regime_filter(lookback=126, min_benchmark_return=0.0002),
            min_universe_breadth_filter(min_names=12),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=60, slow=180, defensive_scale=0.1)],
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v11(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            trend_regime_filter(lookback=126, min_benchmark_return=0.0002),
            min_universe_breadth_filter(min_names=15),
        ],
        position_builder_fn=proportional_long_only_positions(top_n=15),
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=60, slow=180, defensive_scale=0.1)],
        weighting="equal",
    )


def build_volume_confirmed_iteration_v12(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            min_universe_breadth_filter(min_names=15),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=80, slow=220, defensive_scale=0.6)],
        vol_target=0.16,
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v13(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=blended_signal(
            benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
            volatility_adjusted_momentum_signal(
                lookback=252,
                skip=21,
                vol_window=63,
                shift=1,
            ),
            weights=[0.65, 0.35],
        ),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            trend_regime_filter(lookback=126, min_benchmark_return=0.0002),
            min_universe_breadth_filter(min_names=15),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=60, slow=180, defensive_scale=0.1)],
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v14(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            trend_regime_filter(lookback=126, min_benchmark_return=0.0002),
            min_universe_breadth_filter(min_names=15),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=15,
        risk_scalers=[benchmark_trend_scale(fast=60, slow=180, defensive_scale=0.1)],
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v15(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            min_universe_breadth_filter(min_names=15),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=80, slow=220, defensive_scale=0.6)],
        vol_target=0.14,
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v16(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            min_universe_breadth_filter(min_names=15),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=80, slow=220, defensive_scale=0.6)],
        vol_target=0.18,
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v17(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            min_universe_breadth_filter(min_names=15),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=15,
        risk_scalers=[benchmark_trend_scale(fast=60, slow=180, defensive_scale=0.75)],
        vol_target=0.16,
        weighting="equal_sharpe",
    )


def build_volume_confirmed_iteration_v18(name: str):
    return build_long_only_momentum_study(
        name=name,
        base_signal_fn=residual_momentum_signal(lookback=252, skip=21, shift=1),
        filters=[
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            ),
            relative_volume_strength_filter(volume_window=63, volume_quantile=0.8),
            positive_momentum_filter,
            min_universe_breadth_filter(min_names=15),
        ],
        n_positions=15,
        liquidity_top_n=90,
        min_adv_threshold=30_000_000.0,
        rebalance_every=10,
        risk_scalers=[benchmark_trend_scale(fast=80, slow=220, defensive_scale=0.6)],
        vol_target=0.16,
        weighting="equal_sharpe",
        residualize=True,
        factors_loader=load_spy_factor,
    )


def build_volume_confirmed_iteration_v19(name: str):
    universe = load_sp500_universe()
    benchmark = load_benchmark()
    sector_map = load_sector_map()

    study = (
        Study(
            universe=universe,
            benchmark=benchmark,
            name=name,
        )
        .add_factor_model(factors=["market", "sector"], sector_map=sector_map)
        .base_signal(benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1))
        .neutralize_signal(["sector"])
        .add_filter(
            volume_confirmation_filter_factory(
                volume_window=30,
                volume_quantile=0.7,
                trailing_window=63,
                min_trailing_return=0.0,
            )
        )
        .add_filter(relative_volume_strength_filter(volume_window=63, volume_quantile=0.8))
        .add_filter(positive_momentum_filter)
        .add_filter(min_universe_breadth_filter(min_names=15))
        .add_tradeable_constraint(qs.min_adv(30_000_000.0))
        .add_tradeable_constraint(qs.liquidity(top_n=90, window=63))
        .build_long_only(n=15)
        .scale_risk(benchmark_trend_scale(fast=80, slow=220, defensive_scale=0.6))
        .scale_risk(vol_target=0.16)
        .weight_equal_sharpe(window=126)
        .rebalance(every=10)
        .run()
    )
    return study
