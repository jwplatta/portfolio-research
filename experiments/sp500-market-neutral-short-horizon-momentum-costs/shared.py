from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy.constants import SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
BENCHMARK_TICKER = "SPY"
N_LONG = 20
N_SHORT = 20
MIN_PRICE = 5.0
MIN_ADV = 20_000_000.0
LIQUIDITY_TOP_N = 150
LIQUIDITY_WINDOW = 60
REBALANCE_EVERY = 5


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
    return f"sp500-market-neutral-short-horizon-momentum-costs:{version}"


def short_horizon_sector_relative_signal(window: int = 20, skip: int = 0):
    """Signal = stock_window_return - sector_window_return."""

    def signal_fn(**cache):
        returns = cache["returns"]
        sector_map = pd.Series(load_sector_map()).reindex(returns.columns).fillna("Unknown")
        stock_ret = returns.rolling(window).sum().shift(skip + 1)
        sector_ret = stock_ret.copy()
        for _, tickers in sector_map.groupby(sector_map):
            cols = tickers.index.tolist()
            sector_ret.loc[:, cols] = stock_ret.loc[:, cols].mean(axis=1).values[:, None]
        return stock_ret - sector_ret

    signal_fn.__name__ = f"short_horizon_sector_relative_signal_{window}_{skip}"
    return signal_fn


def sector_beta_neutralize_positions(window: int = 60, passes: int = 2):
    sector_map = load_sector_map()

    def scaler(positions, **cache):
        returns = cache["returns"]
        benchmark = cache["benchmark"].reindex(returns.index).fillna(0.0)

        mean_r = returns.rolling(window).mean()
        mean_b = benchmark.rolling(window).mean()
        mean_rb = returns.mul(benchmark, axis=0).rolling(window).mean()
        cov = mean_rb.sub(mean_r.mul(mean_b, axis=0))
        var_b = benchmark.rolling(window).var().replace(0.0, np.nan)
        betas = cov.div(var_b, axis=0).shift(1)

        adjusted = positions.copy()
        for date in positions.index:
            active = positions.loc[date]
            active = active[active != 0.0]
            if active.empty:
                continue

            weights = active.copy()
            for _ in range(passes):
                sectors = pd.Series(
                    {ticker: sector_map.get(ticker, "Unknown") for ticker in weights.index}
                )
                for _, tickers in sectors.groupby(sectors):
                    cols = tickers.index.tolist()
                    sector_weights = weights.loc[cols]
                    weights.loc[cols] = sector_weights - sector_weights.mean()

                beta_slice = betas.loc[date, weights.index].dropna()
                if len(beta_slice) >= 2:
                    w = weights.reindex(beta_slice.index).fillna(0.0)
                    beta_exposure = float((w * beta_slice).sum())
                    beta_norm = float((beta_slice**2).sum())
                    if beta_norm > 0.0:
                        weights.loc[beta_slice.index] = w - (beta_exposure / beta_norm) * beta_slice

            weights = weights - weights.mean()
            gross = weights.abs().sum()
            if gross > 0.0 and not pd.isna(gross):
                adjusted.loc[date, weights.index] = weights / gross

        return adjusted.fillna(0.0)

    scaler.__name__ = f"sector_beta_neutralize_positions_{window}_{passes}"
    return scaler
