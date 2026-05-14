import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

import qstudy as qs
from qstudy import Study
from qstudy.constants import SECTOR_ETF_MAP, SECTOR_ETFS, SP500

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
SECTOR_CACHE_PATH = Path.home() / ".qstudy" / "sector_map.json"


@lru_cache(maxsize=1)
def load_universe():
    return qs.download(SP500, START_DATE, END_DATE)


@lru_cache(maxsize=1)
def load_benchmark():
    return qs.download(["SPY"], START_DATE, END_DATE)


@lru_cache(maxsize=1)
def load_sector_factors():
    return qs.download(["SPY", *SECTOR_ETFS], START_DATE, END_DATE)


@lru_cache(maxsize=1)
def load_sector_map():
    if SECTOR_CACHE_PATH.exists():
        with SECTOR_CACHE_PATH.open() as handle:
            cached = json.load(handle)
        return {ticker: cached.get(ticker, "Unknown") for ticker in SP500}
    return qs.get_sector_map(SP500)


@lru_cache(maxsize=1)
def load_sector_etf_by_ticker():
    sector_map = load_sector_map()
    by_ticker = {}
    for ticker in SP500:
        sector = sector_map.get(ticker, "Unknown")
        by_ticker[ticker] = SECTOR_ETF_MAP.get(sector, "SPY")
    return by_ticker


@lru_cache(maxsize=1)
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


def make_study(*, name=None):
    return Study(
        universe=load_universe(),
        benchmark=load_benchmark(),
        factors=load_sector_factors(),
        name=name,
    )


def _safe_zscore(frame):
    mean = frame.mean()
    std = frame.std()
    if pd.isna(std) or std == 0.0:
        return (frame - mean) * np.nan
    return (frame - mean) / std


def _sector_members():
    by_sector = {}
    sector_map = load_sector_map()
    for ticker, sector in sector_map.items():
        by_sector.setdefault(sector, []).append(ticker)
    return by_sector


@lru_cache(maxsize=64)
def compute_cointegration_features(
    lookback=252,
    z_window=15,
    use_market=False,
    adf_window=252,
    adf_stride=21,
    half_life_window=126,
    smooth_window=1,
):
    universe = load_universe()
    factors = load_sector_factors()
    close = universe.close.copy()
    log_close = np.log(close.replace(0.0, np.nan))
    market_log = np.log(factors.close["SPY"].replace(0.0, np.nan))
    available_factor_cols = set(factors.close.columns)

    ticker_to_etf = load_sector_etf_by_ticker()
    sector_close = pd.DataFrame(
        {
            ticker: factors.close[
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

    if use_market:
        market_var = market_log.rolling(lookback).var().replace(0.0, np.nan)
        beta_market = spread.rolling(lookback).cov(market_log).div(market_var, axis=0)
        spread = spread.sub(beta_market.mul(market_log, axis=0), axis=0)

    if smooth_window > 1:
        spread = spread.rolling(smooth_window).mean()

    spread_mean = spread.rolling(z_window).mean()
    spread_std = spread.rolling(z_window).std().replace(0.0, np.nan)
    zscore = spread.sub(spread_mean).div(spread_std)

    lagged = spread.shift(1)
    phi = (
        spread.rolling(half_life_window)
        .cov(lagged)
        .div(lagged.rolling(half_life_window).var().replace(0.0, np.nan))
    )
    phi = phi.clip(lower=1e-4, upper=0.999)
    half_life = (-np.log(2.0) / np.log(phi)).replace([np.inf, -np.inf], np.nan)

    adf_pvalue = pd.DataFrame(np.nan, index=spread.index, columns=spread.columns)
    for ticker in spread.columns:
        series = spread[ticker]
        for end_idx in range(adf_window, len(series), adf_stride):
            window = series.iloc[end_idx - adf_window : end_idx].dropna()
            if len(window) < int(adf_window * 0.8):
                continue
            try:
                adf_pvalue.iloc[end_idx, adf_pvalue.columns.get_loc(ticker)] = adfuller(
                    window, maxlag=1, regression="c", autolag=None
                )[1]
            except Exception:
                continue
    adf_pvalue = adf_pvalue.ffill()

    return {
        "spread": spread,
        "zscore": zscore.replace([np.inf, -np.inf], np.nan),
        "half_life": half_life,
        "adf_pvalue": adf_pvalue,
    }


@lru_cache(maxsize=64)
def compute_two_factor_residual_features(
    *,
    beta_window=126,
    signal_window=10,
    signal_mode="returns",
    clip=3.0,
    smooth_window=1,
):
    returns = load_universe().returns.copy()
    benchmark = load_benchmark().returns["SPY"].reindex(returns.index).fillna(0.0)
    market = pd.DataFrame(
        np.repeat(benchmark.to_numpy()[:, None], returns.shape[1], axis=1),
        index=returns.index,
        columns=returns.columns,
    )
    sector = load_sector_returns_by_ticker().reindex_like(returns).fillna(0.0)

    market_var = market.rolling(beta_window).var().replace(0.0, np.nan)
    stock_market_beta = returns.rolling(beta_window).cov(market).div(market_var)
    sector_market_beta = sector.rolling(beta_window).cov(market).div(market_var)

    stock_ex_market = returns - stock_market_beta.mul(market)
    sector_ex_market = sector - sector_market_beta.mul(market)
    sector_var = sector_ex_market.rolling(beta_window).var().replace(0.0, np.nan)
    sector_beta = stock_ex_market.rolling(beta_window).cov(sector_ex_market).div(sector_var)

    residual_returns = stock_ex_market - sector_beta.mul(sector_ex_market)
    if smooth_window > 1:
        residual_returns = residual_returns.rolling(smooth_window).mean()
    residual_returns = residual_returns.replace([np.inf, -np.inf], np.nan)

    if signal_mode == "returns":
        signal_source = residual_returns.rolling(signal_window).sum()
    elif signal_mode == "price_path":
        residual_path = residual_returns.fillna(0.0).cumsum()
        path_mean = residual_path.rolling(signal_window).mean()
        path_std = residual_path.rolling(signal_window).std().replace(0.0, np.nan)
        signal_source = residual_path.sub(path_mean).div(path_std)
    else:
        raise ValueError(f"Unsupported signal_mode: {signal_mode}")

    z_mean = signal_source.rolling(signal_window).mean()
    z_std = signal_source.rolling(signal_window).std().replace(0.0, np.nan)
    zscore = signal_source.sub(z_mean).div(z_std).clip(-clip, clip)

    return {
        "residual_returns": residual_returns,
        "signal_source": signal_source,
        "zscore": zscore.replace([np.inf, -np.inf], np.nan),
    }


def cointegration_signal(
    *,
    lookback=252,
    z_window=15,
    use_market=False,
    adf_window=252,
    adf_stride=21,
    half_life_window=126,
    smooth_window=1,
    clip=3.0,
    score_mode="zscore",
):
    def signal_fn(**cache):
        features = compute_cointegration_features(
            lookback=lookback,
            z_window=z_window,
            use_market=use_market,
            adf_window=adf_window,
            adf_stride=adf_stride,
            half_life_window=half_life_window,
            smooth_window=smooth_window,
        )
        signal = -features["zscore"].clip(-clip, clip)
        if score_mode == "adf_weighted":
            quality = 1.0 - features["adf_pvalue"].clip(0.0, 1.0)
            signal = signal * quality
        elif score_mode == "hl_weighted":
            half_life = features["half_life"]
            weight = (20.0 / half_life).clip(lower=0.25, upper=1.5)
            signal = signal * weight
        return signal.replace([np.inf, -np.inf], np.nan)

    signal_fn.__name__ = (
        f"cointegration_signal_{lookback}_{z_window}_{use_market}_{smooth_window}_{score_mode}"
    )
    return signal_fn


def residual_mean_reversion_signal(
    *,
    beta_window=126,
    signal_window=10,
    signal_mode="returns",
    smooth_window=1,
    clip=3.0,
):
    def signal_fn(**cache):
        features = compute_two_factor_residual_features(
            beta_window=beta_window,
            signal_window=signal_window,
            signal_mode=signal_mode,
            clip=clip,
            smooth_window=smooth_window,
        )
        return -features["zscore"]

    signal_fn.__name__ = (
        f"residual_mean_reversion_signal_{beta_window}_{signal_window}_{signal_mode}"
    )
    return signal_fn


def demean_signal(signal, **cache):
    return signal.sub(signal.mean(axis=1), axis=0)


def sector_demean_signal(signal, **cache):
    adjusted = signal.copy()
    for members in _sector_members().values():
        cols = [ticker for ticker in members if ticker in adjusted.columns]
        if not cols:
            continue
        adjusted.loc[:, cols] = adjusted[cols].sub(adjusted[cols].mean(axis=1), axis=0)
    return adjusted


def sector_balanced_positions(signal, n_long=10, n_short=10):
    positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    sector_map = load_sector_map()

    for date in signal.index:
        row = signal.loc[date].dropna()
        if row.empty:
            continue

        long_names = row.nlargest(n_long).index
        short_names = row.nsmallest(n_short).index

        long_sectors = {}
        for ticker in long_names:
            long_sectors.setdefault(sector_map.get(ticker, "Unknown"), []).append(ticker)

        short_sectors = {}
        for ticker in short_names:
            short_sectors.setdefault(sector_map.get(ticker, "Unknown"), []).append(ticker)

        if long_sectors:
            sector_weight = 0.5 / len(long_sectors)
            for tickers in long_sectors.values():
                weight = sector_weight / len(tickers)
                positions.loc[date, tickers] = weight

        if short_sectors:
            sector_weight = 0.5 / len(short_sectors)
            for tickers in short_sectors.values():
                weight = sector_weight / len(tickers)
                positions.loc[date, tickers] = -weight

    return positions


def adf_filter(
    *,
    max_pvalue=0.15,
    lookback=252,
    z_window=15,
    use_market=False,
    adf_window=252,
    adf_stride=21,
    half_life_window=126,
    smooth_window=1,
):
    def filter_fn(signal, **cache):
        features = compute_cointegration_features(
            lookback=lookback,
            z_window=z_window,
            use_market=use_market,
            adf_window=adf_window,
            adf_stride=adf_stride,
            half_life_window=half_life_window,
            smooth_window=smooth_window,
        )
        mask = features["adf_pvalue"].le(max_pvalue)
        return signal.where(mask)

    filter_fn.__name__ = f"adf_filter_{max_pvalue}"
    return filter_fn


def half_life_filter(
    *,
    min_days=5.0,
    max_days=20.0,
    lookback=252,
    z_window=15,
    use_market=False,
    adf_window=252,
    adf_stride=21,
    half_life_window=126,
    smooth_window=1,
):
    def filter_fn(signal, **cache):
        features = compute_cointegration_features(
            lookback=lookback,
            z_window=z_window,
            use_market=use_market,
            adf_window=adf_window,
            adf_stride=adf_stride,
            half_life_window=half_life_window,
            smooth_window=smooth_window,
        )
        half_life = features["half_life"]
        mask = half_life.ge(min_days) & half_life.le(max_days)
        return signal.where(mask)

    filter_fn.__name__ = f"half_life_filter_{min_days}_{max_days}"
    return filter_fn


def dislocation_filter(
    *,
    min_abs_z=1.0,
    lookback=252,
    z_window=15,
    use_market=False,
    adf_window=252,
    adf_stride=21,
    half_life_window=126,
    smooth_window=1,
):
    def filter_fn(signal, **cache):
        features = compute_cointegration_features(
            lookback=lookback,
            z_window=z_window,
            use_market=use_market,
            adf_window=adf_window,
            adf_stride=adf_stride,
            half_life_window=half_life_window,
            smooth_window=smooth_window,
        )
        mask = features["zscore"].abs().ge(min_abs_z)
        return signal.where(mask)

    filter_fn.__name__ = f"dislocation_filter_{min_abs_z}"
    return filter_fn


def spread_shock_filter(
    *,
    max_abs_z=3.5,
    lookback=252,
    z_window=15,
    use_market=False,
    adf_window=252,
    adf_stride=21,
    half_life_window=126,
    smooth_window=1,
):
    def filter_fn(signal, **cache):
        features = compute_cointegration_features(
            lookback=lookback,
            z_window=z_window,
            use_market=use_market,
            adf_window=adf_window,
            adf_stride=adf_stride,
            half_life_window=half_life_window,
            smooth_window=smooth_window,
        )
        mask = features["zscore"].abs().le(max_abs_z)
        return signal.where(mask)

    filter_fn.__name__ = f"spread_shock_filter_{max_abs_z}"
    return filter_fn


def proportional_positions(signal, **cache):
    standardized = signal.sub(signal.mean(axis=1), axis=0)
    std = standardized.std(axis=1).replace(0.0, np.nan)
    standardized = standardized.div(std, axis=0).clip(-3.0, 3.0)
    gross = standardized.abs().sum(axis=1).replace(0.0, np.nan)
    return standardized.div(gross, axis=0).fillna(0.0)


def beta_neutralize_positions(window=60):
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


def benchmark_regime_scaler(fast=100, slow=200, defensive_scale=0.7):
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

    scaler.__name__ = f"benchmark_regime_scaler_{fast}_{slow}_{defensive_scale}"
    return scaler


def dispersion_regime_scaler(
    *,
    lookback=126,
    signal_lookback=15,
    low_scale=0.75,
    high_scale=1.1,
    use_market=False,
):
    def scaler(positions, **cache):
        features = compute_cointegration_features(
            lookback=252,
            z_window=signal_lookback,
            use_market=use_market,
            adf_window=252,
            adf_stride=21,
            half_life_window=126,
            smooth_window=1,
        )
        dispersion = features["zscore"].abs().median(axis=1)
        baseline = dispersion.rolling(lookback).median().replace(0.0, np.nan)
        scale = dispersion.div(baseline).clip(lower=low_scale, upper=high_scale)
        return positions.mul(scale.shift(1).fillna(1.0), axis=0)

    scaler.__name__ = f"dispersion_regime_scaler_{lookback}_{signal_lookback}"
    return scaler


def build_experiment(
    *,
    name,
    lookback=252,
    z_window=15,
    use_market=False,
    adf_window=252,
    adf_stride=21,
    half_life_window=126,
    smooth_window=1,
    signal_clip=3.0,
    score_mode="zscore",
    add_sector_demean=False,
    adf_max_pvalue=None,
    half_life_bounds=None,
    min_abs_z=None,
    max_abs_z=None,
    vol_quantile=None,
    volume_quantile=None,
    momentum_window=None,
    momentum_quantile=None,
    min_price_threshold=5.0,
    min_adv_threshold=5_000_000.0,
    liquidity_top_n=250,
    liquidity_window=60,
    n_long=25,
    n_short=25,
    use_proportional_positions=False,
    rebalance_every=5,
    beta_neutral_window=None,
    benchmark_regime=None,
    dispersion_regime=None,
    equal_vol_window=None,
    vol_target=None,
    study_residualize=True,
    signal_family="cointegration",
    residual_beta_window=126,
    residual_signal_window=10,
    residual_signal_mode="returns",
    position_builder="long_short",
):
    study = make_study(name=name)
    if study_residualize:
        study = study.residualize_returns()

    if signal_family == "cointegration":
        signal_fn = cointegration_signal(
            lookback=lookback,
            z_window=z_window,
            use_market=use_market,
            adf_window=adf_window,
            adf_stride=adf_stride,
            half_life_window=half_life_window,
            smooth_window=smooth_window,
            clip=signal_clip,
            score_mode=score_mode,
        )
    elif signal_family == "residual":
        signal_fn = residual_mean_reversion_signal(
            beta_window=residual_beta_window,
            signal_window=residual_signal_window,
            signal_mode=residual_signal_mode,
            smooth_window=smooth_window,
            clip=signal_clip,
        )
    else:
        raise ValueError(f"Unsupported signal_family: {signal_family}")

    study = study.base_signal(signal_fn).transform_signal(demean_signal)

    if add_sector_demean:
        study = study.transform_signal(sector_demean_signal)

    if adf_max_pvalue is not None:
        study = study.add_filter(
            adf_filter(
                max_pvalue=adf_max_pvalue,
                lookback=lookback,
                z_window=z_window,
                use_market=use_market,
                adf_window=adf_window,
                adf_stride=adf_stride,
                half_life_window=half_life_window,
                smooth_window=smooth_window,
            )
        )

    if half_life_bounds is not None:
        study = study.add_filter(
            half_life_filter(
                min_days=half_life_bounds[0],
                max_days=half_life_bounds[1],
                lookback=lookback,
                z_window=z_window,
                use_market=use_market,
                adf_window=adf_window,
                adf_stride=adf_stride,
                half_life_window=half_life_window,
                smooth_window=smooth_window,
            )
        )

    if min_abs_z is not None:
        study = study.add_filter(
            dislocation_filter(
                min_abs_z=min_abs_z,
                lookback=lookback,
                z_window=z_window,
                use_market=use_market,
                adf_window=adf_window,
                adf_stride=adf_stride,
                half_life_window=half_life_window,
                smooth_window=smooth_window,
            )
        )

    if max_abs_z is not None:
        study = study.add_filter(
            spread_shock_filter(
                max_abs_z=max_abs_z,
                lookback=lookback,
                z_window=z_window,
                use_market=use_market,
                adf_window=adf_window,
                adf_stride=adf_stride,
                half_life_window=half_life_window,
                smooth_window=smooth_window,
            )
        )

    if vol_quantile is not None:
        study = study.add_vol_filter(vol_window=max(z_window, 10), quantile=vol_quantile)

    if volume_quantile is not None:
        study = study.add_volume_zscore_filter(window=30, min_zscore_quantile=volume_quantile)

    if momentum_window is not None and momentum_quantile is not None:
        study = study.add_momentum_context_filter(
            window=momentum_window, max_abs_quantile=momentum_quantile
        )

    study = study.add_tradeable_constraint(qs.min_price(min_price_threshold))
    study = study.add_tradeable_constraint(qs.min_adv(min_adv_threshold))
    study = study.add_tradeable_constraint(
        qs.liquidity(top_n=liquidity_top_n, window=liquidity_window)
    )

    if position_builder == "sector_balanced":
        study = study.build_positions(
            lambda signal: sector_balanced_positions(signal, n_long=n_long, n_short=n_short)
        )
    elif use_proportional_positions:
        study = study.build_positions(proportional_positions)
    else:
        study = study.build_long_short(n_long=n_long, n_short=n_short)

    if beta_neutral_window is not None:
        study = study.scale_risk(beta_neutralize_positions(beta_neutral_window))

    if benchmark_regime is not None:
        study = study.scale_risk(benchmark_regime_scaler(**benchmark_regime))

    if dispersion_regime is not None:
        study = study.scale_risk(
            dispersion_regime_scaler(
                lookback=dispersion_regime["lookback"],
                signal_lookback=z_window,
                low_scale=dispersion_regime["low_scale"],
                high_scale=dispersion_regime["high_scale"],
                use_market=use_market,
            )
        )

    if equal_vol_window is not None:
        study = study.weight_equal_vol(vol_window=equal_vol_window)

    if vol_target is not None:
        study = study.scale_risk(vol_target=vol_target)

    study = study.rebalance(every=rebalance_every)

    return study.run()


def emit_metrics(study):
    print(json.dumps(study.metrics_dict(), default=str, sort_keys=True))
