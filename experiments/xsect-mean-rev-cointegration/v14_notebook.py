import json
from pathlib import Path

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SECTOR_ETF_MAP, SECTOR_ETFS, SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
SECTOR_CACHE_PATH = Path.home() / ".qstudy" / "sector_map.json"


def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)


def load_benchmark():
    return qs.download(["SPY"], START_DATE, END_DATE)


def load_sector_factors():
    return qs.download(["SPY", *SECTOR_ETFS], START_DATE, END_DATE)


def load_sector_map():
    if SECTOR_CACHE_PATH.exists():
        with SECTOR_CACHE_PATH.open() as handle:
            cached = json.load(handle)
        return {ticker: cached.get(ticker, "Unknown") for ticker in SP500}
    return qs.get_sector_map(SP500)


def load_sector_etf_by_ticker():
    sector_map = load_sector_map()
    by_ticker = {}
    for ticker in SP500:
        sector = sector_map.get(ticker, "Unknown")
        by_ticker[ticker] = SECTOR_ETF_MAP.get(sector, "SPY")
    return by_ticker


def compute_cointegration_zscore(
    close: pd.DataFrame,
    factor_close: pd.DataFrame,
    *,
    lookback: int = 252,
    z_window: int = 6,
) -> pd.DataFrame:
    log_close = np.log(close.replace(0.0, np.nan))
    available_factor_cols = set(factor_close.columns)
    ticker_to_etf = load_sector_etf_by_ticker()

    sector_close = pd.DataFrame(
        {
            ticker: factor_close[
                ticker_to_etf[ticker] if ticker_to_etf[ticker] in available_factor_cols else "SPY"
            ]
            for ticker in close.columns
        },
        index=close.index,
    )
    sector_log = np.log(sector_close.replace(0.0, np.nan))

    sector_var = sector_log.rolling(lookback).var().replace(0.0, np.nan)
    beta_sector = log_close.rolling(lookback).cov(sector_log).div(sector_var)
    spread = log_close - beta_sector * sector_log

    spread_mean = spread.rolling(z_window).mean()
    spread_std = spread.rolling(z_window).std().replace(0.0, np.nan)
    zscore = spread.sub(spread_mean).div(spread_std)
    return zscore.replace([np.inf, -np.inf], np.nan)


def cointegration_signal(**cache):
    factor_close = load_sector_factors().close.reindex(cache["close"].index)
    zscore = compute_cointegration_zscore(
        cache["close"],
        factor_close,
        lookback=252,
        z_window=6,
    )
    return -zscore


def demean_signal(signal, **cache):
    return signal.sub(signal.mean(axis=1), axis=0)


def benchmark_regime_scaler(fast=100, slow=200, defensive_scale=0.8):
    def scaler(positions, **cache):
        benchmark = cache["benchmark"]
        if benchmark is None:
            return positions

        price = (1.0 + benchmark.fillna(0.0)).cumprod()
        fast_ma = price.rolling(fast).mean()
        slow_ma = price.rolling(slow).mean()
        scale = pd.Series(1.0, index=positions.index)
        scale = scale.where(fast_ma >= slow_ma, defensive_scale).shift(1).fillna(1.0)
        return positions.mul(scale, axis=0)

    return scaler


def build_v14_study():
    universe = load_universe()
    benchmark = load_benchmark()
    factors = load_sector_factors()

    return (
        Study(
            universe=universe,
            benchmark=benchmark,
            factors=factors,
            name="xsect_coint_mr_v14_notebook",
        )
        .residualize_returns()
        .base_signal(cointegration_signal)
        .transform_signal(demean_signal)
        .add_tradeable_constraint(qs.min_price(5.0))
        .add_tradeable_constraint(qs.min_adv(5_000_000.0))
        .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
        .build_long_short(n_long=10, n_short=10)
        .scale_risk(vol_target=0.16)
        .rebalance(every=10)
        .run()
    )


study = build_v14_study()


if __name__ == "__main__":
    print(json.dumps(study.metrics_dict(), default=str, indent=2))
