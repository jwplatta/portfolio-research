"""Signal sweep: sector ETF momentum vs SPY — v4 focused on tie-breaking and long-only.

Motivation
----------
Narrow breadth at the market level misses *which* sector is driving it. In 2024,
XLK dominated while most other sectors lagged — a market-level breadth filter gates
the whole strategy off, but the right move is to be long tech stocks and short stocks
in lagging sectors.

V1 results: market breadth gates don't work; almost everything negative.
V2 results: sector dispersion gate (q60/q70) helped substantially. Best avg SR ~0.58.
  - Best signals: sector_spy_mom_20d, sector_spy_sharpe_60d, sector_spy_mom_x_stock_120d
  - Best gate: sector_disp_20d_q60 and sector_disp_20d_q70
  - Worst: always-on (none), leadership persistence filter, trend_20_100_mom
  - Core problem: 2017/2019/2021 (calm bull) are weak; 2018/2022 (macro shock) are strong

V3 changes (literature-informed)
---------------------------------
1. Long-only version (Beluška & Vojtko 2024): the long leg is consistently profitable.
2. Skip-5 signal variant (Grundy & Martin 2001): avoids short-term reversal contamination.
3. Dispersion acceleration gate (Stivers & Sun 2010).
4. Stress + dispersion gate: SPY vol elevated AND high sector dispersion.

V3 results
----------
- Best avg_annual_sr: sector_spy_mom_x_stock_120d (0.88), sector_spy_mom_within_20d r21 (0.85),
  sector_rel_cumlog_120d long (0.84), sector_rel_cumlog_sharpe_20d long (0.81)
- Dropped: stress_disp_60d_q60 (weak), accel gates (noisy)
- sector_rel_cumlog and sector_rel_cumlog_sharpe are promising but still suffer from
  arbitrary within-sector stock selection (all stocks in a sector share the same signal value)

V4 changes
----------
1. Add within-sector tie-breaking variants of cumlog signals: sector_rel_cumlog_within_{w}d
   and sector_rel_cumlog_sharpe_within_{w}d — same sector score but multiplied by inverse
   distance from sector mean return, so the most representative stocks are selected.

2. Add contrarian-within-sector signal: sector_spy_mom_contrarian_{w}d — long the laggards
   within the leading sector, short the winners within the losing sector. Tests whether
   mean reversion within a momentum sector adds value over pure momentum selection.

3. Drop: stress_disp_60d_q60, all accel gate configs, sector_disp_20d_q60_t50 and
   sector_disp_20d_q70_t50 for long-short (only helped within_20d r21 which is in the pool).
   Drop 60d window for main signals (dominated by 20d and 120d).

Signal constructions
--------------------
- sector_spy_mom_{w}d: ETF.rolling(w).mean() - SPY.rolling(w).mean()
- sector_spy_sharpe_{w}d: (ETF-SPY).rolling(w).mean() / ETF.rolling(w).std()
- sector_spy_mom_x_stock_{w}d: sector rank x intra-sector stock rank (120d only)
- sector_spy_mom_within_{w}d: sector score x closeness to sector mean return
- sector_spy_mom_contrarian_{w}d: sector score x farthest-from-sector-mean (NEW)
- sector_rel_cumlog_{w}d: log(ETF_cum / SPY_cum) — proper compounding
- sector_rel_cumlog_sharpe_{w}d: cumlog / rolling std of daily log excess
- sector_rel_cumlog_within_{w}d: cumlog x closeness to sector mean return (NEW)
- sector_rel_cumlog_sharpe_within_{w}d: cumlog_sharpe x closeness (NEW)
- skip5 variants of mom and sharpe at [60, 120]d

Windows: [20, 120] for main signals (60d dropped — dominated in v3)

Usage:
    uv run python examples/signal_sweeps/signal_sweep_sector_etf_momentum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study

sys.path.insert(0, str(Path(__file__).parent.parent))
from portfolio_utils import make_equity_curve_regime_scale
from signal_sweep_utils import (
    COST_BPS,
    N_LONG,
    N_SHORT,
    TRAIN_START,
    load_data,
    run_sweep,
)

GROUP = "sector-etf-momentum"
OUT_DIR = Path(__file__).resolve().parent / "out" / "sector-etf-momentum"

REBALANCE_PERIODS = [5, 21]
WINDOWS = [20, 120]
SKIP5_WINDOWS = [60, 120]

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB"]

GICS_TO_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLK",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_etf_frame(
    r: pd.DataFrame,
    factor_returns: pd.DataFrame,
    sector_etf_map: dict[str, str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for ticker in r.columns:
        etf = sector_etf_map.get(ticker, "SPY")
        src = etf if etf in factor_returns.columns else "SPY"
        out[ticker] = factor_returns[src].reindex(r.index).fillna(0.0)
    return out


def _sector_etf_df(factor_returns: pd.DataFrame) -> pd.DataFrame:
    cols = [e for e in SECTOR_ETFS if e in factor_returns.columns]
    return factor_returns[cols]


# ---------------------------------------------------------------------------
# Signal factories
# ---------------------------------------------------------------------------


def _make_sector_spy_sharpe(window: int, sector_etf_map: dict, skip: int = 0) -> dict:
    name = f"sector_spy_sharpe_{window}d" + (f"_skip{skip}" if skip else "")

    def signal_fn(window=window, skip=skip, sector_etf_map=sector_etf_map, **cache):
        r = cache["_active_returns"]
        fr = cache["factor_returns"]
        spy = fr["SPY"].reindex(r.index).fillna(0.0)
        etf_frame = _build_etf_frame(r, fr, sector_etf_map)
        if skip:
            etf_frame = etf_frame.shift(skip)
            spy = spy.shift(skip)
        excess = etf_frame.sub(spy, axis=0)
        mu = excess.rolling(window).mean()
        sigma = etf_frame.rolling(window).std().clip(lower=1e-8)
        return mu / sigma

    signal_fn.__name__ = name
    return {"name": name, "fn": signal_fn, "filters": ""}


def _make_sector_spy_mom(window: int, sector_etf_map: dict, skip: int = 0) -> dict:
    name = f"sector_spy_mom_{window}d" + (f"_skip{skip}" if skip else "")

    def signal_fn(window=window, skip=skip, sector_etf_map=sector_etf_map, **cache):
        r = cache["_active_returns"]
        fr = cache["factor_returns"]
        spy = fr["SPY"].reindex(r.index).fillna(0.0)
        etf_frame = _build_etf_frame(r, fr, sector_etf_map)
        if skip:
            etf_frame = etf_frame.shift(skip)
            spy = spy.shift(skip)
        return etf_frame.rolling(window).mean().sub(spy.rolling(window).mean(), axis=0)

    signal_fn.__name__ = name
    return {"name": name, "fn": signal_fn, "filters": ""}


def _make_sector_spy_mom_within(window: int, sector_etf_map: dict) -> dict:
    """Sector ETF momentum selecting most representative (closest-to-mean) stocks.

    Same sector-level score as _make_sector_spy_mom, but breaks within-sector ties
    by how close each stock's rolling mean return is to its sector's average.
    Stocks nearest the sector mean score highest within that sector.
    """
    name = f"sector_spy_mom_within_{window}d"

    # Pre-compute ETF -> tickers mapping from the sector_etf_map
    etf_to_tickers: dict[str, list[str]] = {}
    for ticker, etf in sector_etf_map.items():
        etf_to_tickers.setdefault(etf, []).append(ticker)

    def signal_fn(window=window, sector_etf_map=sector_etf_map, etf_to_tickers=etf_to_tickers, **cache):
        r = cache["_active_returns"]
        fr = cache["factor_returns"]
        spy = fr["SPY"].reindex(r.index).fillna(0.0)
        etf_frame = _build_etf_frame(r, fr, sector_etf_map)

        # 1. Sector-level score
        sector_score = etf_frame.rolling(window).mean().sub(spy.rolling(window).mean(), axis=0)

        # 2. Within-sector closeness to sector mean return
        stock_mean = r.rolling(window).mean()
        closeness = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
        for etf, tickers in etf_to_tickers.items():
            # Only use tickers present in r
            present = [t for t in tickers if t in r.columns]
            if not present:
                continue
            sector_slice = stock_mean[present]
            sector_mean = sector_slice.mean(axis=1)
            distance = sector_slice.sub(sector_mean, axis=0).abs()
            closeness[present] = 1.0 / (distance + 1e-8)

        return sector_score * closeness

    signal_fn.__name__ = name
    return {"name": name, "fn": signal_fn, "filters": ""}


def _make_sector_spy_mom_contrarian(window: int, sector_etf_map: dict) -> dict:
    """Contrarian within sector: long laggards in the leading sector, short winners in the losing sector.

    Sector score is the same as sector_spy_mom — sectors ranked by ETF vs SPY momentum.
    Within each sector, stocks are ranked by distance from the sector mean return, but
    inverted: the farthest below the sector mean score highest (laggards in a leading sector),
    the farthest above the sector mean score lowest (winners in a lagging sector).

    Tests whether intra-sector mean reversion adds value on top of cross-sector momentum.
    """
    name = f"sector_spy_mom_contrarian_{window}d"

    etf_to_tickers: dict[str, list[str]] = {}
    for ticker, etf in sector_etf_map.items():
        etf_to_tickers.setdefault(etf, []).append(ticker)

    def signal_fn(window=window, sector_etf_map=sector_etf_map, etf_to_tickers=etf_to_tickers, **cache):
        r = cache["_active_returns"]
        fr = cache["factor_returns"]
        spy = fr["SPY"].reindex(r.index).fillna(0.0)
        etf_frame = _build_etf_frame(r, fr, sector_etf_map)

        sector_score = etf_frame.rolling(window).mean().sub(spy.rolling(window).mean(), axis=0)

        stock_mean = r.rolling(window).mean()
        signed_distance = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
        for etf, tickers in etf_to_tickers.items():
            present = [t for t in tickers if t in r.columns]
            if not present:
                continue
            sector_slice = stock_mean[present]
            sector_mean = sector_slice.mean(axis=1)
            # Negative: stock below sector mean (laggard) gets positive value in leading sector
            signed_distance[present] = -sector_slice.sub(sector_mean, axis=0)

        return sector_score * signed_distance

    signal_fn.__name__ = name
    return {"name": name, "fn": signal_fn, "filters": ""}


def _make_sector_rel_cumlog(window: int, sector_etf_map: dict, skip: int = 0) -> dict:
    """Cumulative log return of each stock's sector ETF relative to SPY over the window.

    log_excess_t = log(1 + r_etf_t) - log(1 + r_spy_t)
    signal = rolling(window).sum(log_excess)  ≈ log(ETF_cum / SPY_cum)

    This directly measures how far the sector has drifted from the market, compounding
    properly — unlike mean(r_etf) - mean(r_spy) which ignores path and variance drag.
    """
    name = f"sector_rel_cumlog_{window}d" + (f"_skip{skip}" if skip else "")

    def signal_fn(window=window, skip=skip, sector_etf_map=sector_etf_map, **cache):
        r = cache["_active_returns"]
        fr = cache["factor_returns"]
        spy = fr["SPY"].reindex(r.index).fillna(0.0)
        etf_frame = _build_etf_frame(r, fr, sector_etf_map)
        if skip:
            etf_frame = etf_frame.shift(skip)
            spy = spy.shift(skip)
        log_excess = np.log1p(etf_frame).sub(np.log1p(spy), axis=0)
        return log_excess.rolling(window).sum()

    signal_fn.__name__ = name
    return {"name": name, "fn": signal_fn, "filters": ""}


def _make_sector_rel_cumlog_sharpe(window: int, sector_etf_map: dict, skip: int = 0) -> dict:
    """Cumulative log excess return divided by rolling std of daily log excess.

    Normalizes the dislocation signal by its own volatility — how many standard
    deviations has this sector drifted from SPY? Accounts for regimes where all
    sectors are more volatile (e.g. 2022) vs. calm years (2017, 2021).
    """
    name = f"sector_rel_cumlog_sharpe_{window}d" + (f"_skip{skip}" if skip else "")

    def signal_fn(window=window, skip=skip, sector_etf_map=sector_etf_map, **cache):
        r = cache["_active_returns"]
        fr = cache["factor_returns"]
        spy = fr["SPY"].reindex(r.index).fillna(0.0)
        etf_frame = _build_etf_frame(r, fr, sector_etf_map)
        if skip:
            etf_frame = etf_frame.shift(skip)
            spy = spy.shift(skip)
        log_excess = np.log1p(etf_frame).sub(np.log1p(spy), axis=0)
        cum_excess = log_excess.rolling(window).sum()
        sigma = log_excess.rolling(window).std().clip(lower=1e-8)
        return cum_excess / sigma

    signal_fn.__name__ = name
    return {"name": name, "fn": signal_fn, "filters": ""}


def _make_sector_rel_cumlog_within(window: int, sector_etf_map: dict) -> dict:
    """Cumulative log return vs SPY, with within-sector tie-breaking by closeness to sector mean.

    Same sector score as sector_rel_cumlog but multiplied by inverse distance from
    sector mean return — selects the most representative stocks in leading/lagging sectors
    rather than picking arbitrarily by column order.
    """
    name = f"sector_rel_cumlog_within_{window}d"

    etf_to_tickers: dict[str, list[str]] = {}
    for ticker, etf in sector_etf_map.items():
        etf_to_tickers.setdefault(etf, []).append(ticker)

    def signal_fn(window=window, sector_etf_map=sector_etf_map, etf_to_tickers=etf_to_tickers, **cache):
        r = cache["_active_returns"]
        fr = cache["factor_returns"]
        spy = fr["SPY"].reindex(r.index).fillna(0.0)
        etf_frame = _build_etf_frame(r, fr, sector_etf_map)
        log_excess = np.log1p(etf_frame).sub(np.log1p(spy), axis=0)
        sector_score = log_excess.rolling(window).sum()

        stock_mean = r.rolling(window).mean()
        closeness = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
        for etf, tickers in etf_to_tickers.items():
            present = [t for t in tickers if t in r.columns]
            if not present:
                continue
            sector_slice = stock_mean[present]
            sector_mean = sector_slice.mean(axis=1)
            distance = sector_slice.sub(sector_mean, axis=0).abs()
            closeness[present] = 1.0 / (distance + 1e-8)

        return sector_score * closeness

    signal_fn.__name__ = name
    return {"name": name, "fn": signal_fn, "filters": ""}


def _make_sector_rel_cumlog_sharpe_within(window: int, sector_etf_map: dict) -> dict:
    """Cumulative log return / rolling std, with within-sector tie-breaking by closeness to sector mean."""
    name = f"sector_rel_cumlog_sharpe_within_{window}d"

    etf_to_tickers: dict[str, list[str]] = {}
    for ticker, etf in sector_etf_map.items():
        etf_to_tickers.setdefault(etf, []).append(ticker)

    def signal_fn(window=window, sector_etf_map=sector_etf_map, etf_to_tickers=etf_to_tickers, **cache):
        r = cache["_active_returns"]
        fr = cache["factor_returns"]
        spy = fr["SPY"].reindex(r.index).fillna(0.0)
        etf_frame = _build_etf_frame(r, fr, sector_etf_map)
        log_excess = np.log1p(etf_frame).sub(np.log1p(spy), axis=0)
        cum_excess = log_excess.rolling(window).sum()
        sigma = log_excess.rolling(window).std().clip(lower=1e-8)
        sector_score = cum_excess / sigma

        stock_mean = r.rolling(window).mean()
        closeness = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
        for etf, tickers in etf_to_tickers.items():
            present = [t for t in tickers if t in r.columns]
            if not present:
                continue
            sector_slice = stock_mean[present]
            sector_mean = sector_slice.mean(axis=1)
            distance = sector_slice.sub(sector_mean, axis=0).abs()
            closeness[present] = 1.0 / (distance + 1e-8)

        return sector_score * closeness

    signal_fn.__name__ = name
    return {"name": name, "fn": signal_fn, "filters": ""}


def _make_sector_spy_mom_x_stock(window: int, sector_etf_map: dict, skip: int = 0) -> dict:
    name = f"sector_spy_mom_x_stock_{window}d" + (f"_skip{skip}" if skip else "")

    def signal_fn(window=window, skip=skip, sector_etf_map=sector_etf_map, **cache):
        r = cache["_active_returns"]
        fr = cache["factor_returns"]
        spy = fr["SPY"].reindex(r.index).fillna(0.0)
        etf_frame = _build_etf_frame(r, fr, sector_etf_map)
        r_shifted = r.shift(skip) if skip else r
        if skip:
            etf_frame = etf_frame.shift(skip)
            spy = spy.shift(skip)
        sector_score = etf_frame.rolling(window).mean().sub(
            spy.rolling(window).mean(), axis=0
        )
        stock_score = r_shifted.rolling(window).mean() - etf_frame.rolling(window).mean()
        sector_rank = sector_score.rank(axis=1, pct=True, na_option="keep")
        stock_rank = stock_score.rank(axis=1, pct=True, na_option="keep")
        return sector_rank * stock_rank

    signal_fn.__name__ = name
    return {"name": name, "fn": signal_fn, "filters": ""}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _make_sector_disp_filter(window: int, quantile: float):
    """ON when sector ETF dispersion > trailing quantile (absolute level gate)."""
    name = f"sector_disp_{window}d_q{int(quantile * 100)}"

    def filt(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        fr = cache["factor_returns"]
        etfs = _sector_etf_df(fr)
        disp = etfs.rolling(window).mean().std(axis=1)
        threshold = disp.rolling(252, min_periods=126).quantile(quantile)
        active = disp.gt(threshold).reindex(signal.index).fillna(False)
        return signal.where(active, other=np.nan)

    filt.__name__ = name
    return filt


def _make_disp_acceleration_filter(fast: int, slow: int):
    """ON when sector dispersion is accelerating: fast_disp > slow_disp.

    Catches early-to-mid sector rotation rather than the peak.
    Motivated by Stivers & Sun (2010): rising dispersion is a better entry
    point than peak dispersion, which often precedes momentum reversals.
    """
    name = f"sector_disp_accel_{fast}d_vs_{slow}d"

    def filt(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        fr = cache["factor_returns"]
        etfs = _sector_etf_df(fr)
        etf_rolling = etfs.rolling(1).mean()  # daily returns
        disp_fast = etf_rolling.rolling(fast).std(ddof=0)  # std of daily returns over fast window...
        # Actually: dispersion = cross-sectional std of ETF rolling mean returns
        disp_fast = etfs.rolling(fast).mean().std(axis=1)
        disp_slow = etfs.rolling(slow).mean().std(axis=1)
        active = disp_fast.gt(disp_slow).reindex(signal.index).fillna(False)
        return signal.where(active, other=np.nan)

    filt.__name__ = name
    return filt


def _make_stress_and_disp_filter(vol_window: int, disp_window: int, disp_quantile: float):
    """ON when market vol is elevated (SPY rolling vol > its median) AND sector
    dispersion is above its trailing quantile.

    The years where sector ETF momentum works best (2018, 2022) have both:
    high realized market vol AND large sector divergence. Calm bull years
    (2017, 2019, 2021) fail this gate.
    """
    name = f"stress_and_disp_{vol_window}d__sector_disp_{disp_window}d_q{int(disp_quantile * 100)}"

    def filt(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        bm = cache["benchmark"]
        fr = cache["factor_returns"]

        # Market stress: SPY rolling vol above its trailing median
        spy_vol = bm.rolling(vol_window).std()
        high_vol = spy_vol.gt(spy_vol.rolling(252, min_periods=126).quantile(0.50))

        # Sector dispersion above quantile
        etfs = _sector_etf_df(fr)
        disp = etfs.rolling(disp_window).mean().std(axis=1)
        high_disp = disp.gt(disp.rolling(252, min_periods=126).quantile(disp_quantile))

        active = (high_vol & high_disp).reindex(signal.index).fillna(False)
        return signal.where(active, other=np.nan)

    filt.__name__ = name
    return filt


def _make_trend_scaler_mom(fast: int, slow: int):
    """Scale DOWN to 0.25x in SPY downtrend. Momentum-style."""
    def scaler(positions: pd.DataFrame, **cache) -> pd.DataFrame:
        bm = cache["benchmark"]
        equity = (1 + bm).cumprod()
        uptrend = equity.rolling(fast).mean().gt(equity.rolling(slow).mean())
        scale = pd.Series(
            np.where(uptrend.reindex(positions.index).fillna(False), 1.0, 0.25),
            index=positions.index,
        )
        return positions.mul(scale.shift(1), axis=0)

    scaler.__name__ = f"trend_{fast}_{slow}_mom"
    return scaler


# ---------------------------------------------------------------------------
# Scaler / filter configs
# ---------------------------------------------------------------------------

SCALER_CONFIGS = [
    # --- Absolute dispersion gates (long-short) ---
    {"tag": "sector_disp_20d_q60",     "filter_fn": _make_sector_disp_filter(20, 0.60),         "scaler_fns": [],                               "filters": "sector_disp_20d_q60", "long_only": False},
    {"tag": "sector_disp_20d_q70",     "filter_fn": _make_sector_disp_filter(20, 0.70),         "scaler_fns": [],                               "filters": "sector_disp_20d_q70", "long_only": False},
    # t50 scaler only for long-short (helped within_20d r21 in v3)
    {"tag": "sector_disp_20d_q70_t50", "filter_fn": _make_sector_disp_filter(20, 0.70),         "scaler_fns": [_make_trend_scaler_mom(50, 200)], "filters": "sector_disp_20d_q70", "long_only": False},

    # --- Stress + dispersion gate (long-short) ---
    {"tag": "stress_disp_20d_q60",     "filter_fn": _make_stress_and_disp_filter(20, 20, 0.60), "scaler_fns": [],                               "filters": "stress_disp_20d_q60", "long_only": False},
    {"tag": "stress_disp_20d_q70",     "filter_fn": _make_stress_and_disp_filter(20, 20, 0.70), "scaler_fns": [],                               "filters": "stress_disp_20d_q70", "long_only": False},

    # --- Long-only versions ---
    {"tag": "sector_disp_20d_q60_long", "filter_fn": _make_sector_disp_filter(20, 0.60),        "scaler_fns": [],                               "filters": "sector_disp_20d_q60", "long_only": True},
    {"tag": "sector_disp_20d_q70_long", "filter_fn": _make_sector_disp_filter(20, 0.70),        "scaler_fns": [],                               "filters": "sector_disp_20d_q70", "long_only": True},
    {"tag": "stress_disp_20d_q60_long", "filter_fn": _make_stress_and_disp_filter(20, 20, 0.60),"scaler_fns": [],                               "filters": "stress_disp_20d_q60", "long_only": True},
    {"tag": "stress_disp_20d_q70_long", "filter_fn": _make_stress_and_disp_filter(20, 20, 0.70),"scaler_fns": [],                               "filters": "stress_disp_20d_q70", "long_only": True},
]


# ---------------------------------------------------------------------------
# Build signal list
# ---------------------------------------------------------------------------


def _build_signals(sector_etf_map: dict) -> list[dict]:
    signals = []
    # Main signals at 20d and 120d (60d dropped — dominated in v3)
    for w in WINDOWS:
        signals.append(_make_sector_spy_sharpe(w, sector_etf_map))
        signals.append(_make_sector_spy_mom(w, sector_etf_map))
    # x_stock only at 120d (best in v3)
    signals.append(_make_sector_spy_mom_x_stock(120, sector_etf_map))
    # Within-sector closeness variant at 20d
    signals.append(_make_sector_spy_mom_within(20, sector_etf_map))
    # Contrarian within sector at 20d and 120d (NEW v4)
    for w in WINDOWS:
        signals.append(_make_sector_spy_mom_contrarian(w, sector_etf_map))
    # Skip-5 variants at 60d and 120d (Grundy & Martin)
    for w in SKIP5_WINDOWS:
        signals.append(_make_sector_spy_sharpe(w, sector_etf_map, skip=5))
        signals.append(_make_sector_spy_mom(w, sector_etf_map, skip=5))
    # Cumulative log relative return — proper compounding vs. mean-return proxy
    for w in WINDOWS:
        signals.append(_make_sector_rel_cumlog(w, sector_etf_map))
        signals.append(_make_sector_rel_cumlog_sharpe(w, sector_etf_map))
    # Cumlog within-sector tie-breaking variants (NEW v4)
    for w in WINDOWS:
        signals.append(_make_sector_rel_cumlog_within(w, sector_etf_map))
        signals.append(_make_sector_rel_cumlog_sharpe_within(w, sector_etf_map))
    return signals


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------


def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    fn = entry["fn"]
    filter_fn = scaler_cfg.get("filter_fn")
    scaler_fns = scaler_cfg.get("scaler_fns", [])
    long_only = scaler_cfg.get("long_only", False)
    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)

    builder = Study(
        universe=universe, benchmark=benchmark, factors=factors, verbose=verbose
    ).base_signal(fn)

    if filter_fn is not None:
        builder = builder.add_filter(filter_fn)

    builder = builder.add_tradeable_constraint(qs.liquidity(top_n=300)).rank_transform()

    if long_only:
        builder = builder.build_long_only(n=N_LONG)
    else:
        builder = builder.build_long_short(n_long=N_LONG, n_short=N_SHORT)

    builder = builder.fully_invest().scale_risk(fn=equity_curve_scaler)

    for scaler_fn in scaler_fns:
        builder = builder.scale_risk(fn=scaler_fn)

    return builder.rebalance(every=rebalance).with_transaction_costs(cost_bps=COST_BPS).run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    universe, benchmark, factors = load_data()
    sector_map = qs.get_sector_map(list(universe.returns.columns))
    sector_etf_map = {t: GICS_TO_ETF.get(s, "SPY") for t, s in sector_map.items()}

    signals = _build_signals(sector_etf_map)
    n_total = len(signals) * len(REBALANCE_PERIODS) * len(SCALER_CONFIGS)
    print(f"Total configs: {len(signals)} signals × {len(REBALANCE_PERIODS)} rebalance × {len(SCALER_CONFIGS)} scalers = {n_total}")

    run_sweep(
        group=GROUP,
        signals=signals,
        scaler_configs=SCALER_CONFIGS,
        rebalance_periods=REBALANCE_PERIODS,
        build_study_fn=build_study_fn,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
