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
CACHE_DIR = Path.home() / ".qstudy" / "sp500_event_driven"
INTRADAY_INTERVAL = "5m"
INTRADAY_LOOKBACK_DAYS = 60
INTRADAY_BATCH_SIZE = 25


@cache
def load_sp500_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download(["SPY"], START_DATE, END_DATE)


@cache
def load_sector_factors():
    return qs.download(["SPY", *SECTOR_ETFS], START_DATE, END_DATE)


@cache
def load_sector_map():
    return qs.get_sector_map(load_sp500_universe().tickers)


def intraday_window(lookback_days: int = INTRADAY_LOOKBACK_DAYS) -> tuple[str, str]:
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    return start.strftime("%Y-%m-%d"), (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def combine_study_data(parts: list[qs.StudyData], interval: str) -> qs.StudyData:
    if not parts:
        raise ValueError("combine_study_data() requires at least one StudyData object.")

    def concat_field(field: str) -> pd.DataFrame:
        return pd.concat([getattr(part, field) for part in parts], axis=1).sort_index()

    open_ = concat_field("open")
    high = concat_field("high")
    low = concat_field("low")
    close = concat_field("close")
    volume = concat_field("volume")
    returns = close.pct_change().fillna(0.0)
    log_returns = np.log(close / close.shift(1))
    return qs.StudyData(
        tickers=close.columns.tolist(),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        returns=returns,
        log_returns=log_returns,
        interval=interval,
    )


@cache
def load_sp500_intraday_universe(
    start: str,
    end: str,
    interval: str = INTRADAY_INTERVAL,
    batch_size: int = INTRADAY_BATCH_SIZE,
):
    parts = []
    for offset in range(0, len(SP500), batch_size):
        batch = SP500[offset : offset + batch_size]
        parts.append(qs.download(batch, start, end, interval=interval))
    return combine_study_data(parts, interval=interval)


@cache
def load_intraday_benchmark(
    start: str,
    end: str,
    interval: str = INTRADAY_INTERVAL,
):
    return qs.download(["SPY"], start, end, interval=interval)


def emit_metrics(study: Study) -> None:
    print(json.dumps(study.metrics_dict(), default=str, sort_keys=True))


def save_study(study: Study, name: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    study.save(CACHE_DIR / f"{name}.pkl")


def demean(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    return signal.sub(signal.mean(axis=1), axis=0)


def gap_return(open_prices: pd.DataFrame, close_prices: pd.DataFrame) -> pd.DataFrame:
    return open_prices.div(close_prices.shift(1)).sub(1.0)


def intraday_return(open_prices: pd.DataFrame, close_prices: pd.DataFrame) -> pd.DataFrame:
    safe_open = open_prices.replace(0.0, np.nan)
    return close_prices.div(safe_open).sub(1.0)


def relative_volume(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return volume.div(volume.rolling(window).mean().replace(0.0, np.nan))


def range_fraction(
    high_prices: pd.DataFrame,
    low_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
) -> pd.DataFrame:
    safe_open = open_prices.replace(0.0, np.nan)
    return high_prices.sub(low_prices).div(safe_open)


def close_location_value(
    high_prices: pd.DataFrame,
    low_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> pd.DataFrame:
    denom = high_prices.sub(low_prices).replace(0.0, np.nan)
    return close_prices.sub(low_prices).div(denom)


def top_abs_quantile_mask(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    threshold = frame.abs().quantile(quantile, axis=1)
    return frame.abs().ge(threshold, axis=0)


def benchmark_trend_scale(fast: int = 100, slow: int = 200, defensive_scale: float = 0.35):
    def scaler(positions, **cache):
        benchmark = cache["benchmark"].fillna(0.0)
        benchmark_price = (1.0 + benchmark).cumprod()
        fast_ma = benchmark_price.rolling(fast).mean()
        slow_ma = benchmark_price.rolling(slow).mean()
        scale = pd.Series(
            np.where(fast_ma >= slow_ma, 1.0, defensive_scale),
            index=benchmark.index,
        )
        return positions.mul(scale.shift(1).fillna(1.0), axis=0)

    scaler.__name__ = f"benchmark_trend_scale_{fast}_{slow}_{defensive_scale}"
    return scaler


def delay_entry_scale(days: int = 1):
    def scaler(positions, **cache):
        return positions.shift(days).fillna(0.0)

    scaler.__name__ = f"delay_entry_scale_{days}"
    return scaler


def staggered_rebalance_scale(lags: tuple[int, ...] = (0, 5, 10)):
    def scaler(positions, **cache):
        blended = sum(positions.shift(lag).fillna(0.0) for lag in lags) / len(lags)
        gross = blended.abs().sum(axis=1).replace(0.0, np.nan)
        return blended.div(gross, axis=0).fillna(0.0)

    scaler.__name__ = "staggered_rebalance_scale_" + "_".join(str(lag) for lag in lags)
    return scaler


def _session_date_index(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        return index.tz_convert("America/New_York").normalize().tz_localize(None)
    return pd.DatetimeIndex(index).normalize()


def _grouped_by_session(frame: pd.DataFrame):
    return frame.groupby(_session_date_index(frame.index))


def aggregate_intraday_to_daily(data: qs.StudyData) -> qs.StudyData:
    open_ = _grouped_by_session(data.open).first()
    high = _grouped_by_session(data.high).max()
    low = _grouped_by_session(data.low).min()
    close = _grouped_by_session(data.close).last()
    volume = _grouped_by_session(data.volume).sum(min_count=1)
    returns = close.pct_change().fillna(0.0)
    log_returns = np.log(close / close.shift(1))
    return qs.StudyData(
        tickers=close.columns.tolist(),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        returns=returns,
        log_returns=log_returns,
        interval="1d_from_intraday",
    )


def build_intraday_exhaustion_features(data: qs.StudyData) -> dict[str, pd.DataFrame]:
    session_groups = _session_date_index(data.close.index)
    day_open = _grouped_by_session(data.open).first()
    day_high = _grouped_by_session(data.high).max()
    day_low = _grouped_by_session(data.low).min()
    day_close = _grouped_by_session(data.close).last()
    day_volume = _grouped_by_session(data.volume).sum(min_count=1)

    minute_returns = data.close.groupby(session_groups).pct_change()
    realized_vol = minute_returns.groupby(session_groups).std().fillna(0.0)
    abs_flow = minute_returns.abs().groupby(session_groups).sum().fillna(0.0)

    intraday_idx = (
        data.close.index.tz_convert("America/New_York")
        if getattr(data.close.index, "tz", None) is not None
        else data.close.index
    )
    last_hour_mask = intraday_idx.time >= pd.Timestamp("15:00").time()
    late_close = _grouped_by_session(data.close.loc[last_hour_mask]).last().reindex(day_close.index)
    late_open = _grouped_by_session(data.open.loc[last_hour_mask]).first().reindex(day_open.index)
    late_move = late_close.div(late_open.replace(0.0, np.nan)).sub(1.0)

    day_move = day_close.div(day_open.replace(0.0, np.nan)).sub(1.0)
    day_range = day_high.sub(day_low).div(day_open.replace(0.0, np.nan))
    close_location = close_location_value(day_high, day_low, day_close)
    rel_volume = relative_volume(day_volume, window=20)

    return {
        "day_move": day_move,
        "day_range": day_range,
        "close_location": close_location,
        "rel_volume": rel_volume,
        "realized_vol": realized_vol,
        "abs_flow": abs_flow,
        "late_move": late_move,
    }


def gap_fade_signal(
    gap_quantile: float = 0.9,
    rel_volume_window: int = 20,
    rel_volume_quantile: float = 0.55,
):
    def signal_fn(**cache):
        open_prices = cache["open"]
        close_prices = cache["close"]
        volume = cache["volume"]
        gap = gap_return(open_prices, close_prices)
        rel_volume = relative_volume(volume, window=rel_volume_window)

        signal = -gap
        mask = top_abs_quantile_mask(gap, gap_quantile)
        mask &= rel_volume.le(rel_volume.quantile(rel_volume_quantile, axis=1), axis=0)
        return signal.where(mask)

    signal_fn.__name__ = f"gap_fade_signal_{gap_quantile}_{rel_volume_window}_{rel_volume_quantile}"
    return signal_fn


def volume_shock_continuation_signal(
    event_window: int = 5,
    volume_window: int = 20,
    volume_quantile: float = 0.9,
    move_quantile: float = 0.75,
    use_active_returns: bool = False,
):
    def signal_fn(**cache):
        returns = cache["_active_returns"] if use_active_returns else cache["returns"]
        volume = cache["volume"]
        price_move = returns.rolling(event_window).sum()
        rel_volume = relative_volume(volume, window=volume_window)
        volume_shock = np.log(rel_volume.replace(0.0, np.nan))
        signal = price_move.mul(volume_shock)

        mask = rel_volume.ge(rel_volume.quantile(volume_quantile, axis=1), axis=0)
        mask &= top_abs_quantile_mask(price_move, move_quantile)
        return signal.where(mask)

    signal_fn.__name__ = (
        "volume_shock_continuation_signal_"
        f"{event_window}_{volume_window}_{volume_quantile}_{move_quantile}_{use_active_returns}"
    )
    return signal_fn


def volume_shock_move_zscore_signal(
    event_window: int = 10,
    volume_window: int = 30,
    volume_quantile: float = 0.9,
    move_quantile: float = 0.8,
    zscore_window: int = 60,
):
    def signal_fn(**cache):
        returns = cache["_active_returns"]
        volume = cache["volume"]
        price_move = returns.rolling(event_window).sum()
        move_mean = price_move.rolling(zscore_window).mean()
        move_std = price_move.rolling(zscore_window).std().replace(0.0, np.nan)
        move_z = price_move.sub(move_mean).div(move_std)
        rel_volume = relative_volume(volume, window=volume_window)
        volume_shock = np.log(rel_volume.replace(0.0, np.nan))
        signal = move_z.mul(volume_shock)

        mask = rel_volume.ge(rel_volume.quantile(volume_quantile, axis=1), axis=0)
        mask &= top_abs_quantile_mask(move_z, move_quantile)
        return signal.where(mask)

    signal_fn.__name__ = (
        "volume_shock_move_zscore_signal_"
        f"{event_window}_{volume_window}_{volume_quantile}_{move_quantile}_{zscore_window}"
    )
    return signal_fn


def volume_shock_vol_adjusted_signal(
    event_window: int = 10,
    volume_window: int = 30,
    volume_quantile: float = 0.9,
    move_quantile: float = 0.8,
    realized_vol_window: int = 20,
):
    def signal_fn(**cache):
        returns = cache["_active_returns"]
        volume = cache["volume"]
        price_move = returns.rolling(event_window).sum()
        realized_vol = returns.rolling(realized_vol_window).std().replace(0.0, np.nan)
        abnormal_move = price_move.div(realized_vol * np.sqrt(event_window))
        rel_volume = relative_volume(volume, window=volume_window)
        volume_shock = np.log(rel_volume.replace(0.0, np.nan))
        signal = abnormal_move.mul(volume_shock)

        mask = rel_volume.ge(rel_volume.quantile(volume_quantile, axis=1), axis=0)
        mask &= top_abs_quantile_mask(abnormal_move, move_quantile)
        return signal.where(mask)

    signal_fn.__name__ = (
        "volume_shock_vol_adjusted_signal_"
        f"{event_window}_{volume_window}_{volume_quantile}_{move_quantile}_{realized_vol_window}"
    )
    return signal_fn


def positive_short_term_trend_filter(window: int = 20):
    def filter_fn(signal, **cache):
        trend = cache["_active_returns"].rolling(window).mean()
        return signal.where(trend > 0.0)

    filter_fn.__name__ = f"positive_short_term_trend_filter_{window}"
    return filter_fn


def intraday_exhaustion_reversal_signal(
    features: dict[str, pd.DataFrame],
    range_window: int = 20,
    range_quantile: float = 0.9,
    volume_window: int = 20,
    volume_quantile: float = 0.7,
    move_quantile: float = 0.8,
):
    def signal_fn(**cache):
        day_move = features["day_move"].reindex(cache["close"].index)
        day_range = features["day_range"].reindex(cache["close"].index)
        close_location = features["close_location"].reindex(cache["close"].index)
        rel_volume = features["rel_volume"].reindex(cache["close"].index)
        realized_vol = features["realized_vol"].reindex(cache["close"].index)
        abs_flow = features["abs_flow"].reindex(cache["close"].index)
        late_move = features["late_move"].reindex(cache["close"].index)

        range_threshold = day_range.quantile(range_quantile, axis=1)
        move_threshold = day_move.abs().quantile(move_quantile, axis=1)
        exhaustion_side = -np.sign(day_move)
        signal = exhaustion_side.mul(day_move.abs()).mul(realized_vol).mul(abs_flow)

        mask = day_range.ge(range_threshold, axis=0)
        mask &= day_move.abs().ge(move_threshold, axis=0)
        mask &= rel_volume.ge(rel_volume.quantile(volume_quantile, axis=1), axis=0)
        mask &= np.sign(day_move).eq(np.sign(late_move))
        mask &= close_location.le(0.2) | close_location.ge(0.8)

        # A second range filter vs each stock's own recent history avoids ranking quiet names.
        rolling_range_median = day_range.rolling(range_window).median()
        mask &= day_range.ge(rolling_range_median)
        return signal.where(mask)

    signal_fn.__name__ = (
        "intraday_exhaustion_reversal_signal_"
        f"{range_window}_{range_quantile}_{volume_window}_{volume_quantile}_{move_quantile}"
    )
    return signal_fn


def build_gap_fade_study(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(gap_fade_signal())
        .transform_signal(demean)
        .add_tradeable_constraint(qs.liquidity(top_n=250, window=60))
        .build_long_short(n_long=25, n_short=25)
        .run()
    )


def build_gap_fade_iteration_v2(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(
            gap_fade_signal(
                gap_quantile=0.95,
                rel_volume_window=30,
                rel_volume_quantile=0.45,
            )
        )
        .transform_signal(demean)
        .add_tradeable_constraint(qs.liquidity(top_n=200, window=60))
        .build_long_short(n_long=20, n_short=20)
        .run()
    )


def build_gap_fade_iteration_v3(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(
            gap_fade_signal(
                gap_quantile=0.9,
                rel_volume_window=20,
                rel_volume_quantile=0.5,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=20, quantile=0.6, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=200, window=60))
        .build_long_short(n_long=20, n_short=20)
        .run()
    )


def build_gap_fade_iteration_v4(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(
            gap_fade_signal(
                gap_quantile=0.92,
                rel_volume_window=20,
                rel_volume_quantile=0.5,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=20, quantile=0.6, keep="low")
        .add_momentum_context_filter(window=60, max_abs_quantile=0.7)
        .add_tradeable_constraint(qs.liquidity(top_n=200, window=60))
        .build_long_short(n_long=20, n_short=20)
        .run()
    )


def build_gap_fade_iteration_v5(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(
            gap_fade_signal(
                gap_quantile=0.92,
                rel_volume_window=20,
                rel_volume_quantile=0.5,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=20, quantile=0.6, keep="low")
        .add_momentum_context_filter(window=60, max_abs_quantile=0.7)
        .add_tradeable_constraint(qs.liquidity(top_n=200, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal_vol(vol_window=40)
        .run()
    )


def build_gap_fade_iteration_v6(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(
            gap_fade_signal(
                gap_quantile=0.92,
                rel_volume_window=20,
                rel_volume_quantile=0.5,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=20, quantile=0.6, keep="low")
        .add_momentum_context_filter(window=60, max_abs_quantile=0.7)
        .add_tradeable_constraint(qs.liquidity(top_n=200, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal_vol(vol_window=40)
        .scale_risk(vol_target=0.18)
        .run()
    )


def build_volume_shock_continuation_study(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(volume_shock_continuation_signal())
        .transform_signal(demean)
        .add_tradeable_constraint(qs.liquidity(top_n=250, window=60))
        .weight_equal_vol(vol_window=60)
        .build_long_short(n_long=25, n_short=25)
        .run()
    )


def build_volume_shock_iteration_v2(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(
            volume_shock_continuation_signal(
                event_window=5,
                volume_window=30,
                volume_quantile=0.95,
                move_quantile=0.85,
            )
        )
        .transform_signal(demean)
        .add_tradeable_constraint(qs.liquidity(top_n=200, window=60))
        .weight_equal_vol(vol_window=60)
        .build_long_short(n_long=20, n_short=20)
        .run()
    )


def build_volume_shock_iteration_v3(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .base_signal(
            volume_shock_continuation_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
            )
        )
        .transform_signal(demean)
        .add_tradeable_constraint(qs.liquidity(top_n=200, window=60))
        .weight_equal_vol(vol_window=60)
        .build_long_short(n_long=20, n_short=20)
        .rebalance(every=5)
        .run()
    )


def build_volume_shock_iteration_v4(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_continuation_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                use_active_returns=True,
            )
        )
        .transform_signal(demean)
        .add_tradeable_constraint(qs.liquidity(top_n=200, window=60))
        .weight_equal_vol(vol_window=60)
        .build_long_short(n_long=20, n_short=20)
        .rebalance(every=5)
        .run()
    )


def build_volume_shock_iteration_v5(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_continuation_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                use_active_returns=True,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .weight_equal_vol(vol_window=60)
        .build_long_short(n_long=20, n_short=20)
        .rebalance(every=5)
        .run()
    )


def build_volume_shock_iteration_v6(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_continuation_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                use_active_returns=True,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal_vol(vol_window=60)
        .scale_risk(benchmark_trend_scale(100, 200, 0.35))
        .rebalance(every=5)
        .run()
    )


def build_volume_shock_iteration_v7(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
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
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal_vol(vol_window=60)
        .rebalance(every=5)
        .run()
    )


def build_volume_shock_iteration_v8(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_vol_adjusted_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                realized_vol_window=20,
            )
        )
        .transform_signal(demean)
        .add_filter(positive_short_term_trend_filter(window=20))
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal_vol(vol_window=60)
        .rebalance(every=5)
        .run()
    )


def build_volume_shock_iteration_v9(name: str) -> Study:
    return (
        Study(universe=load_sp500_universe(), benchmark=load_benchmark(), name=name)
        .add_factor_model(
            "barra-lite",
            factors=["market", "sector"],
            sector_map=load_sector_map(),
            beta_window=60,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_vol_adjusted_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                realized_vol_window=20,
            )
        )
        .transform_signal(demean)
        .neutralize_signal(["market", "sector"])
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal_vol(vol_window=60)
        .rebalance(every=5)
        .run()
    )


def build_volume_shock_iteration_v10(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_vol_adjusted_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                realized_vol_window=20,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal_vol(vol_window=60)
        .rebalance(every=3)
        .run()
    )


def build_volume_shock_iteration_v11(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_vol_adjusted_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                realized_vol_window=20,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal_vol(vol_window=60)
        .rebalance(every=10)
        .run()
    )


def build_volume_shock_iteration_v12(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_vol_adjusted_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                realized_vol_window=20,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .build_long_short(n_long=20, n_short=20)
        .scale_risk(staggered_rebalance_scale((0, 5, 10)))
        .weight_equal_vol(vol_window=60)
        .rebalance(every=5)
        .run()
    )


def build_volume_shock_iteration_v13(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_vol_adjusted_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                realized_vol_window=20,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .build_long_short(n_long=20, n_short=20)
        .scale_risk(delay_entry_scale(1))
        .weight_equal_vol(vol_window=60)
        .rebalance(every=5)
        .run()
    )


def build_volume_shock_iteration_v14(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
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
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal_vol(vol_window=60)
        .rebalance(every=10)
        .run()
    )


def build_volume_shock_iteration_v15(name: str) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_vol_adjusted_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                realized_vol_window=20,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=175, window=60))
        .build_long_short(n_long=20, n_short=20)
        .scale_risk(delay_entry_scale(1))
        .weight_equal_vol(vol_window=60)
        .rebalance(every=10)
        .run()
    )


def build_volume_shock_iteration_v16(name: str) -> Study:
    return build_volume_shock_v21_variant(name)


def build_volume_shock_v21_variant(
    name: str,
    *,
    zscore_window: int = 60,
    liquidity_top_n: int = 175,
    rebalance_every: int = 10,
    entry_delay_days: int = 1,
) -> Study:
    return (
        Study(
            universe=load_sp500_universe(),
            benchmark=load_benchmark(),
            factors=load_sector_factors(),
            name=name,
        )
        .residualize_returns()
        .base_signal(
            volume_shock_move_zscore_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                zscore_window=zscore_window,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=liquidity_top_n, window=60))
        .build_long_short(n_long=20, n_short=20)
        .scale_risk(delay_entry_scale(entry_delay_days))
        .weight_equal_vol(vol_window=60)
        .rebalance(every=rebalance_every)
        .run()
    )


def build_volume_shock_iteration_v17(name: str) -> Study:
    return build_volume_shock_v21_variant(name, rebalance_every=8)


def build_volume_shock_iteration_v18(name: str) -> Study:
    return build_volume_shock_v21_variant(name, rebalance_every=12)


def build_volume_shock_iteration_v19(name: str) -> Study:
    return build_volume_shock_v21_variant(name, rebalance_every=15)


def build_volume_shock_iteration_v20(name: str) -> Study:
    return build_volume_shock_v21_variant(name, zscore_window=40)


def build_volume_shock_iteration_v21(name: str) -> Study:
    return build_volume_shock_v21_variant(name, liquidity_top_n=150)


def build_intraday_exhaustion_study(name: str) -> Study:
    intraday_start, intraday_end = intraday_window()
    minute_universe = load_sp500_intraday_universe(intraday_start, intraday_end)
    minute_benchmark = load_intraday_benchmark(intraday_start, intraday_end)
    daily_universe = aggregate_intraday_to_daily(minute_universe)
    daily_benchmark = aggregate_intraday_to_daily(minute_benchmark)
    features = build_intraday_exhaustion_features(minute_universe)
    return (
        Study(universe=daily_universe, benchmark=daily_benchmark, name=name)
        .base_signal(intraday_exhaustion_reversal_signal(features=features))
        .transform_signal(demean)
        .add_tradeable_constraint(qs.liquidity(top_n=100, window=20))
        .build_long_short(n_long=25, n_short=25)
        .run()
    )
