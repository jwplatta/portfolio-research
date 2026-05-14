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


def load_sector_returns_by_ticker():
    factors = load_sector_factors()
    ticker_to_etf = load_sector_etf_by_ticker()
    available_factor_cols = set(factors.returns.columns)
    return pd.DataFrame(
        {
            ticker: factors.returns[
                ticker_to_etf[ticker] if ticker_to_etf[ticker] in available_factor_cols else "SPY"
            ]
            for ticker in load_universe().returns.columns
        },
        index=factors.returns.index,
    )


def compute_two_factor_residual_zscore(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    sector_returns: pd.DataFrame,
    *,
    beta_window: int = 126,
    signal_window: int = 8,
    clip: float = 3.0,
) -> pd.DataFrame:
    market = pd.DataFrame(
        np.repeat(benchmark_returns.to_numpy()[:, None], returns.shape[1], axis=1),
        index=returns.index,
        columns=returns.columns,
    )
    sector = sector_returns.reindex_like(returns).fillna(0.0)

    market_var = market.rolling(beta_window).var().replace(0.0, np.nan)
    stock_market_beta = returns.rolling(beta_window).cov(market).div(market_var)
    sector_market_beta = sector.rolling(beta_window).cov(market).div(market_var)

    stock_ex_market = returns - stock_market_beta.mul(market)
    sector_ex_market = sector - sector_market_beta.mul(market)
    sector_var = sector_ex_market.rolling(beta_window).var().replace(0.0, np.nan)
    sector_beta = stock_ex_market.rolling(beta_window).cov(sector_ex_market).div(sector_var)

    residual_returns = stock_ex_market - sector_beta.mul(sector_ex_market)
    residual_signal = residual_returns.rolling(signal_window).sum()
    signal_mean = residual_signal.rolling(signal_window).mean()
    signal_std = residual_signal.rolling(signal_window).std().replace(0.0, np.nan)
    zscore = residual_signal.sub(signal_mean).div(signal_std)
    return zscore.clip(-clip, clip).replace([np.inf, -np.inf], np.nan)


def residual_signal(**cache):
    benchmark_returns = load_benchmark().returns["SPY"].reindex(cache["returns"].index).fillna(0.0)
    sector_returns = load_sector_returns_by_ticker().reindex_like(cache["returns"]).fillna(0.0)
    zscore = compute_two_factor_residual_zscore(
        cache["returns"],
        benchmark_returns,
        sector_returns,
        beta_window=126,
        signal_window=8,
    )
    return -zscore


def demean_signal(signal, **cache):
    return signal.sub(signal.mean(axis=1), axis=0)


def sector_demean_signal(signal, **cache):
    adjusted = signal.copy()
    sector_map = load_sector_map()
    sector_members = {}
    for ticker, sector in sector_map.items():
        sector_members.setdefault(sector, []).append(ticker)

    for members in sector_members.values():
        cols = [ticker for ticker in members if ticker in adjusted.columns]
        if not cols:
            continue
        adjusted.loc[:, cols] = adjusted[cols].sub(adjusted[cols].mean(axis=1), axis=0)
    return adjusted


def beta_neutralize_positions(window=20):
    def scaler(positions, **cache):
        returns = cache["returns"]
        benchmark = cache["benchmark"]
        if benchmark is None:
            return positions

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
            gross = neutralized.abs().sum()
            if gross == 0.0 or pd.isna(gross):
                continue
            adjusted.loc[date, beta_slice.index] = neutralized / gross

        return adjusted.fillna(0.0)

    scaler.__name__ = f"beta_neutralize_positions_{window}"
    return scaler


def build_v28_study():
    universe = load_universe()
    benchmark = load_benchmark()
    factors = load_sector_factors()

    return (
        Study(
            universe=universe,
            benchmark=benchmark,
            factors=factors,
            name="xsect_coint_mr_v28_notebook",
        )
        .base_signal(residual_signal)
        .transform_signal(demean_signal)
        .transform_signal(sector_demean_signal)
        .add_volume_zscore_filter(window=30, min_zscore_quantile=0.8)
        .add_tradeable_constraint(qs.min_price(5.0))
        .add_tradeable_constraint(qs.min_adv(5_000_000.0))
        .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
        .build_long_short(n_long=10, n_short=10)
        .scale_risk(beta_neutralize_positions(20))
        .weight_equal_vol(vol_window=60)
        .rebalance(every=10)
        .run()
    )


study = build_v28_study()


if __name__ == "__main__":
    print(json.dumps(study.metrics_dict(), default=str, indent=2))
