"""sp500_four_strat_port — v0: Equal weight baseline.

Four strategies combined with equal (25%) sleeve weights.
All strategy code is self-contained; no imports from other experiments dirs.

Results target (from docs/combined_portfolio_study.py):
    Sharpe ~1.37 | Ann Return ~21.6% | Max DD ~-22.4%

Usage:
    uv run python experiments/sp500-four-strat-port/v0_equal_weight.py
"""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import PortfolioStudy, Study
from qstudy.constants import SECTOR_ETF_MAP, SECTOR_ETFS, SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@cache
def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@cache
def load_benchmark():
    return qs.download(["SPY"], START_DATE, END_DATE)


@cache
def load_sector_factors():
    return qs.download(["SPY", *SECTOR_ETFS], START_DATE, END_DATE)


@cache
def load_sector_map():
    return qs.get_sector_map(load_universe().tickers)


# ---------------------------------------------------------------------------
# Strategy 1: Event-Driven Volume Shock v26
# ---------------------------------------------------------------------------


def build_event_driven_v26(sector_factors) -> Study:
    """Volume shock z-score continuation, low-vol filtered, liquidity top 150."""

    def volume_shock_zscore_signal(
        event_window=10,
        volume_window=30,
        volume_quantile=0.9,
        move_quantile=0.8,
        zscore_window=60,
    ):
        def signal_fn(**cache):
            returns = cache["_active_returns"]
            volume = cache["volume"]
            price_move = returns.rolling(event_window).sum()
            move_mean = price_move.rolling(zscore_window).mean()
            move_std = price_move.rolling(zscore_window).std().replace(0.0, np.nan)
            move_z = price_move.sub(move_mean).div(move_std)
            rel_vol = volume.div(volume.rolling(volume_window).mean().replace(0.0, np.nan))
            volume_shock = np.log(rel_vol.replace(0.0, np.nan))
            signal = move_z.mul(volume_shock)
            mask = rel_vol.ge(rel_vol.quantile(volume_quantile, axis=1), axis=0)
            mask &= move_z.abs().ge(move_z.abs().quantile(move_quantile, axis=1), axis=0)
            return signal.where(mask)

        signal_fn.__name__ = "volume_shock_zscore_signal"
        return signal_fn

    def demean(signal, **cache):
        return signal.sub(signal.mean(axis=1), axis=0)

    def delay_entry(days=1):
        def scaler(positions, **cache):
            return positions.shift(days).fillna(0.0)

        scaler.__name__ = f"delay_entry_{days}"
        return scaler

    return (
        Study(name="event_driven_v26", factors=sector_factors)
        .residualize_returns()
        .base_signal(
            volume_shock_zscore_signal(
                event_window=10,
                volume_window=30,
                volume_quantile=0.9,
                move_quantile=0.8,
                zscore_window=60,
            )
        )
        .transform_signal(demean)
        .add_vol_filter(vol_window=30, quantile=0.7, keep="low")
        .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal_vol(vol_window=60)
        .scale_risk(delay_entry(1))
        .rebalance(every=10)
    )


# ---------------------------------------------------------------------------
# Strategy 2: Cross-Sectional Momentum v19
# ---------------------------------------------------------------------------


def build_momentum_v19(sector_map: dict) -> Study:
    """Benchmark-relative momentum with volume confirmation, sector-neutral, long-only."""

    def benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1):
        def signal_fn(**cache):
            returns = cache["returns"]
            benchmark = cache["benchmark"]
            cum_ret = (1 + returns).rolling(lookback).apply(lambda x: x.prod(), raw=True) - 1
            bm_rolling = (
                (1 + benchmark.fillna(0.0))
                .rolling(lookback)
                .apply(lambda x: x.prod(), raw=True)
                - 1
            )
            return cum_ret.sub(bm_rolling, axis=0).shift(skip).shift(shift)

        signal_fn.__name__ = f"benchmark_relative_momentum_{lookback}_{skip}_{shift}"
        return signal_fn

    def volume_confirmation_filter(
        volume_window=30, volume_quantile=0.7, trailing_window=63, min_trailing_return=0.0
    ):
        def filter_fn(signal, **cache):
            volume = cache["volume"]
            returns = cache["returns"]
            avg_vol = volume.rolling(volume_window).mean().replace(0.0, np.nan)
            recent_vol = volume.rolling(5).mean()
            vol_ratio = recent_vol.div(avg_vol)
            trailing_ret = returns.rolling(trailing_window).sum()
            mask = vol_ratio.ge(vol_ratio.quantile(volume_quantile, axis=1), axis=0)
            mask &= trailing_ret >= min_trailing_return
            return signal.where(mask)

        filter_fn.__name__ = "volume_confirmation_filter"
        return filter_fn

    def relative_volume_strength_filter(volume_window=63, volume_quantile=0.8):
        def filter_fn(signal, **cache):
            volume = cache["volume"]
            avg_vol = volume.rolling(volume_window).mean()
            mask = avg_vol.ge(avg_vol.quantile(volume_quantile, axis=1), axis=0)
            return signal.where(mask)

        filter_fn.__name__ = "relative_volume_strength_filter"
        return filter_fn

    def positive_momentum_filter(signal, **cache):
        trailing = cache["returns"].rolling(126).sum()
        return signal.where(trailing > 0)

    positive_momentum_filter.__name__ = "positive_momentum_filter"

    def min_universe_breadth_filter(min_names=15):
        def filter_fn(signal, **cache):
            eligible = signal.notna().sum(axis=1)
            return signal.where(eligible >= min_names, other=np.nan)

        filter_fn.__name__ = f"min_universe_breadth_{min_names}"
        return filter_fn

    def benchmark_trend_scale(fast=80, slow=220, defensive_scale=0.6):
        def scaler(positions, **cache):
            price = (1 + cache["benchmark"].fillna(0.0)).cumprod()
            scale = pd.Series(
                np.where(price.rolling(fast).mean() >= price.rolling(slow).mean(), 1.0, defensive_scale),
                index=price.index,
            )
            return positions.mul(scale.shift(1).fillna(1.0), axis=0)

        scaler.__name__ = f"benchmark_trend_scale_{fast}_{slow}"
        return scaler

    return (
        Study(name="momentum_v19")
        .add_factor_model(factors=["market", "sector"], sector_map=sector_map)
        .base_signal(benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1))
        .neutralize_signal(["sector"])
        .add_filter(
            volume_confirmation_filter(
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
        .weight_equal_sharpe(window=126)
        .scale_risk(benchmark_trend_scale(fast=80, slow=220, defensive_scale=0.6))
        .scale_risk(vol_target=0.16)
        .rebalance(every=10)
    )


# ---------------------------------------------------------------------------
# Strategy 2b: Market-Neutral Residual Momentum v29
# ---------------------------------------------------------------------------


def build_momentum_v29(sector_factors) -> Study:
    """Residual momentum v29: 30d lookback, 20d skip, market-neutral, long/short 20/20."""

    def residual_momentum_signal(lookback=30, skip=20, shift=1):
        def signal_fn(**cache):
            residual_returns = cache["residual_returns"]
            return residual_returns.shift(skip).rolling(lookback).sum().shift(shift)

        signal_fn.__name__ = f"residual_momentum_signal_{lookback}_{skip}_{shift}"
        return signal_fn

    return (
        Study(name="momentum_v29", factors=sector_factors)
        .add_factor_model(factors=["market"])
        .residualize_returns()
        .base_signal(residual_momentum_signal(lookback=30, skip=20, shift=1))
        .add_tradeable_constraint(qs.min_price(5.0))
        .add_tradeable_constraint(qs.min_adv(20_000_000.0))
        .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal()
        .neutralize_positions({"market": 0})
        .rebalance(every=10)
    )


# ---------------------------------------------------------------------------
# Strategy 3: Residual Mean Reversion v10
# ---------------------------------------------------------------------------


def build_resid_mr_v10(sector_factors) -> Study:
    """Residual mean reversion, proportional positions, dual equity/benchmark regime scale."""

    def residual_mean_reversion_signal(**cache):
        return -cache["residual_returns"].rolling(5).mean().shift(1)

    def demean_signal(signal, **cache):
        return signal.sub(signal.mean(axis=1), axis=0)

    def proportional_positions(signal, **cache):
        z = signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1), axis=0).clip(-3, 3)
        z = z.sub(z.mean(axis=1), axis=0)
        gross = z.abs().sum(axis=1).replace(0.0, np.nan)
        return z.div(gross, axis=0)

    def equity_curve_regime_scale(positions, **cache):
        returns = cache["returns"]
        mask = cache.get("_tradeable_mask")
        if mask is None:
            mask = cache.get("_liquidity_mask")
        if mask is not None:
            returns = returns.where(mask)
        raw_ret = (positions.shift(1) * returns).sum(axis=1)
        equity = (1 + raw_ret).cumprod()
        scale = pd.Series(
            np.where(equity > equity.rolling(20).mean(), 1.0, 0.25), index=equity.index
        )
        return positions.mul(scale.shift(1), axis=0)

    def benchmark_regime_scale(positions, **cache):
        price = (1 + cache["benchmark"].fillna(0.0)).cumprod()
        scale = pd.Series(
            np.where(price.rolling(150).mean() >= price.rolling(250).mean(), 1.0, 0.75),
            index=price.index,
        )
        return positions.mul(scale.shift(1), axis=0)

    equity_curve_regime_scale.__name__ = "equity_curve_regime_scale"
    benchmark_regime_scale.__name__ = "benchmark_regime_scale"

    return (
        Study(name="resid_mr_v10", factors=sector_factors)
        .residualize_returns()
        .base_signal(residual_mean_reversion_signal)
        .transform_signal(demean_signal)
        .add_vol_filter(vol_window=5, quantile=0.6)
        .add_volume_zscore_filter(window=30, min_zscore_quantile=0.8)
        .add_momentum_context_filter(window=60, max_abs_quantile=0.7)
        .add_tradeable_constraint(qs.liquidity(top_n=250, window=60))
        .build_positions(proportional_positions)
        .scale_risk(benchmark_regime_scale)
        .scale_risk(equity_curve_regime_scale)
        .rebalance(every=1)
    )


# ---------------------------------------------------------------------------
# Strategy 4: Cointegration MR v28 (two-factor residual, inlined signal)
# ---------------------------------------------------------------------------


def build_coint_mr_v28(sector_factors) -> Study:
    """Two-factor residual MR (market + sector beta stripped), beta-neutral.

    Signal logic inlined from experiments/xsect-mean-rev-cointegration/common.py:
    compute_two_factor_residual_features + residual_mean_reversion_signal.
    v28 params: beta_window=126, signal_window=8, signal_mode="returns", beta_neutral_window=20.
    """
    from qstudy.constants import SECTOR_ETF_MAP

    def _build_sector_returns_by_ticker(returns, sector_factors_data, sector_map):
        """Map each stock ticker to its sector ETF return series."""
        available = set(sector_factors_data.returns.columns)
        return pd.DataFrame(
            {
                ticker: sector_factors_data.returns[
                    SECTOR_ETF_MAP.get(sector_map.get(ticker, ""), "SPY")
                    if SECTOR_ETF_MAP.get(sector_map.get(ticker, ""), "SPY") in available
                    else "SPY"
                ]
                for ticker in returns.columns
            },
            index=sector_factors_data.returns.index,
        )

    def two_factor_residual_signal(beta_window=126, signal_window=8, clip=3.0):
        def signal_fn(**cache):
            returns = cache["returns"]
            benchmark = cache["benchmark"]
            sector_ret_by_ticker = cache.get("_sector_returns_by_ticker")
            if sector_ret_by_ticker is None:
                # fallback: use SPY as sector proxy for all tickers
                bench_vals = benchmark.reindex(returns.index).fillna(0.0).to_numpy()
                sector_ret_by_ticker = pd.DataFrame(
                    np.repeat(bench_vals[:, None], returns.shape[1], axis=1),
                    index=returns.index,
                    columns=returns.columns,
                )

            bench = benchmark.reindex(returns.index).fillna(0.0)
            market = pd.DataFrame(
                np.repeat(bench.to_numpy()[:, None], returns.shape[1], axis=1),
                index=returns.index,
                columns=returns.columns,
            )
            sector = sector_ret_by_ticker.reindex_like(returns).fillna(0.0)

            market_var = market.rolling(beta_window).var().replace(0.0, np.nan)
            stock_mkt_beta = returns.rolling(beta_window).cov(market).div(market_var)
            sector_mkt_beta = sector.rolling(beta_window).cov(market).div(market_var)

            stock_ex_mkt = returns - stock_mkt_beta.mul(market)
            sector_ex_mkt = sector - sector_mkt_beta.mul(market)
            sector_var = sector_ex_mkt.rolling(beta_window).var().replace(0.0, np.nan)
            sector_beta = stock_ex_mkt.rolling(beta_window).cov(sector_ex_mkt).div(sector_var)

            residual = (stock_ex_mkt - sector_beta.mul(sector_ex_mkt)).replace(
                [np.inf, -np.inf], np.nan
            )

            signal_source = residual.rolling(signal_window).sum()
            z_mean = signal_source.rolling(signal_window).mean()
            z_std = signal_source.rolling(signal_window).std().replace(0.0, np.nan)
            zscore = signal_source.sub(z_mean).div(z_std).clip(-clip, clip)
            return -zscore.replace([np.inf, -np.inf], np.nan)

        signal_fn.__name__ = f"two_factor_residual_signal_{beta_window}_{signal_window}"
        return signal_fn

    def demean_signal(signal, **cache):
        return signal.sub(signal.mean(axis=1), axis=0)

    def sector_demean_signal(signal, **cache):
        # Group columns by sector using cache sector map if available
        sector_map = cache.get("_sector_map", {})
        adjusted = signal.copy()
        by_sector: dict[str, list[str]] = {}
        for ticker in adjusted.columns:
            s = sector_map.get(ticker, "Unknown")
            by_sector.setdefault(s, []).append(ticker)
        for members in by_sector.values():
            cols = [t for t in members if t in adjusted.columns]
            if not cols:
                continue
            adjusted.loc[:, cols] = adjusted[cols].sub(adjusted[cols].mean(axis=1), axis=0)
        return adjusted

    sector_demean_signal.__name__ = "sector_demean_signal"

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

    return (
        Study(name="coint_mr_v28", factors=sector_factors)
        .base_signal(two_factor_residual_signal(beta_window=126, signal_window=8))
        .transform_signal(demean_signal)
        .transform_signal(sector_demean_signal)
        .add_volume_zscore_filter(window=30, min_zscore_quantile=0.8)
        .add_tradeable_constraint(qs.min_price(5.0))
        .add_tradeable_constraint(qs.min_adv(5_000_000.0))
        .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
        .build_long_short(n_long=10, n_short=10)
        .weight_equal_vol(vol_window=60)
        .scale_risk(beta_neutralize_positions(20))
        .rebalance(every=10)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_strategies(sector_factors, sector_map):
    return [
        build_event_driven_v26(sector_factors),
        build_momentum_v19(sector_map),
        build_resid_mr_v10(sector_factors),
        build_coint_mr_v28(sector_factors),
    ]


def build_strategies_with_momentum_v29(sector_factors, sector_map):
    return [
        build_event_driven_v26(sector_factors),
        build_momentum_v29(sector_factors),
        build_resid_mr_v10(sector_factors),
        build_coint_mr_v28(sector_factors),
    ]


def build_strategies_without_event_with_momentum_v29(sector_factors, sector_map):
    return [
        build_momentum_v29(sector_factors),
        build_resid_mr_v10(sector_factors),
        build_coint_mr_v28(sector_factors),
    ]


# ---------------------------------------------------------------------------
# Strategy 5: Short-Horizon Sector-Relative Momentum (w20_s0_equal)
# ---------------------------------------------------------------------------


def build_short_horizon_momentum_w20(sector_map: dict) -> Study:
    """20-day sector-relative momentum, equal weight, long/short 20/20, weekly rebalance.

    Best configuration from sp500-market-neutral-short-horizon-momentum sweep:
    window=20, skip=0, equal weight — Sharpe 1.25, Ann. Return 15.8%, Max DD -15.3%.
    """

    def short_horizon_sector_relative_signal(window: int = 20, skip: int = 0):
        def signal_fn(**cache):
            returns = cache["returns"]
            sector_map_local = pd.Series(sector_map).reindex(returns.columns).fillna("Unknown")
            stock_ret = returns.rolling(window).sum().shift(skip + 1)
            sector_ret = stock_ret.copy()
            for sector, tickers in sector_map_local.groupby(sector_map_local):
                cols = tickers.index.tolist()
                sector_ret.loc[:, cols] = stock_ret.loc[:, cols].mean(axis=1).values[:, None]
            return stock_ret - sector_ret

        signal_fn.__name__ = f"short_horizon_sector_relative_signal_{window}_{skip}"
        return signal_fn

    def sector_beta_neutralize_positions(window: int = 60, passes: int = 2):
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
                            weights.loc[beta_slice.index] = (
                                w - (beta_exposure / beta_norm) * beta_slice
                            )
                weights = weights - weights.mean()
                gross = weights.abs().sum()
                if gross > 0.0 and not pd.isna(gross):
                    adjusted.loc[date, weights.index] = weights / gross

            return adjusted.fillna(0.0)

        scaler.__name__ = f"sector_beta_neutralize_positions_{window}_{passes}"
        return scaler

    return (
        Study(name="short_horizon_momentum_w20")
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(5.0))
        .add_tradeable_constraint(qs.min_adv(20_000_000.0))
        .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal()
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance(every=5)
    )


def build_strategies_five(sector_factors, sector_map):
    return [
        build_event_driven_v26(sector_factors),
        build_momentum_v29(sector_factors),
        build_resid_mr_v10(sector_factors),
        build_coint_mr_v28(sector_factors),
        build_short_horizon_momentum_w20(sector_map),
    ]


def print_results(portfolio, strategies):
    print("\n--- Strategy Return Correlations ---")
    print(portfolio.strategy_corr.to_string())

    print("\n--- Individual Strategy Metrics ---")
    for study in strategies:
        m = study.metrics
        print(
            f"  {study._name:25s}  sharpe={m.sharpe_ratio:+.2f}"
            f"  ret={m.ann_return:+.1%}  dd={m.max_drawdown:.1%}"
        )

    print("\n--- Portfolio Metrics ---")
    m = portfolio.metrics
    print(f"  Sharpe:      {m.sharpe_ratio:+.2f}")
    print(f"  Ann Return:  {m.ann_return:+.1%}")
    print(f"  Ann Vol:     {m.ann_vol:.1%}")
    print(f"  Max DD:      {m.max_drawdown:.1%}")
    print(f"  DD Duration: {m.drawdown_duration} days")
    if m.information_ratio is not None:
        print(f"  Info Ratio:  {m.information_ratio:+.2f}")
    if m.benchmark_corr is not None:
        print(f"  Benchmark Corr: {m.benchmark_corr:.2f}")


def main():
    print("Loading data...")
    universe = load_universe()
    benchmark = load_benchmark()
    sector_factors = load_sector_factors()
    sector_map = load_sector_map()

    print(
        f"Universe: {len(universe.tickers)} tickers | "
        f"{universe.returns.index[0].date()} – {universe.returns.index[-1].date()}"
    )

    strategies = build_strategies(sector_factors, sector_map)

    print("\n[v0] Equal weight (25% each)...")
    portfolio = (
        PortfolioStudy(
            strategies=strategies,
            universe=universe,
            benchmark=benchmark,
            name="sp500_four_strat_port_v0_equal",
        )
        .weight_equal()
        .run()
    )
    print_results(portfolio, strategies)
    return portfolio


if __name__ == "__main__":
    main()
