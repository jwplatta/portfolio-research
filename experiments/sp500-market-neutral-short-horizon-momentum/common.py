from __future__ import annotations

import json
import os
from functools import cache

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"


@cache
def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download("SPY", START_DATE, END_DATE)


@cache
def load_sector_map():
    return qs.get_sector_map(load_universe().tickers)


def emit_metrics(study):
    print(json.dumps(study.metrics_dict(), default=str, sort_keys=True))


def short_horizon_sector_relative_signal(window: int = 5, skip: int = 0):
    """Signal = stock_window_return - sector_window_return.

    The skip period shifts the lookback window back, so skip=3 means
    the most recent 3 days are excluded from the return window.
    """

    def signal_fn(**cache):
        returns = cache["returns"]
        sector_map = pd.Series(load_sector_map()).reindex(returns.columns).fillna("Unknown")
        # sum of returns over window, shifted back by (skip + 1) to lag 1 day for execution
        stock_ret = returns.rolling(window).sum().shift(skip + 1)
        # compute equal-weighted sector return for each stock's sector
        sector_ret = stock_ret.copy()
        for sector, tickers in sector_map.groupby(sector_map):
            cols = tickers.index.tolist()
            sector_ret.loc[:, cols] = stock_ret.loc[:, cols].mean(axis=1).values[:, None]
        return stock_ret - sector_ret

    signal_fn.__name__ = f"short_horizon_sector_relative_signal_{window}_{skip}"
    return signal_fn


def vol_normalize_transform(vol_window: int = 20):
    """Divide signal by rolling realized vol to normalize magnitude across vol regimes."""

    def transform(signal, **cache):
        returns = cache["returns"]
        vol = returns.rolling(vol_window).std().shift(1)
        vol = vol.replace(0.0, np.nan)
        return signal.div(vol)

    transform.__name__ = f"vol_normalize_transform_{vol_window}"
    return transform


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
                for sector, tickers in sectors.groupby(sectors):
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


def build_study(
    name: str,
    *,
    window: int = 5,
    skip: int = 0,
    residualize: bool = False,
    vol_normalize: bool = False,
    weighting: str = "equal",
    n_long: int = 20,
    n_short: int = 20,
    min_price_threshold: float = 5.0,
    min_adv_threshold: float = 20_000_000.0,
    liquidity_top_n: int = 150,
    rebalance_every: int = 5,
) -> Study:
    universe = load_universe()
    benchmark = load_benchmark()

    study = Study(
        universe=universe,
        benchmark=benchmark,
        factors=benchmark if residualize else None,
        name=name,
    )

    if residualize:
        study = study.residualize_returns()

    study = study.base_signal(short_horizon_sector_relative_signal(window=window, skip=skip))

    if vol_normalize:
        study = study.transform_signal(vol_normalize_transform(vol_window=20))

    study = (
        study.add_tradeable_constraint(qs.min_price(min_price_threshold))
        .add_tradeable_constraint(qs.min_adv(min_adv_threshold))
        .add_tradeable_constraint(qs.liquidity(top_n=liquidity_top_n, window=60))
        .build_long_short(n_long=n_long, n_short=n_short)
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
    )

    if weighting == "inv_vol":
        study = study.weight_equal_vol(vol_window=63)
    elif weighting == "proportional":
        study = study.weight_equal_sharpe(window=126)

    study = study.rebalance(every=rebalance_every)

    return study.run()
