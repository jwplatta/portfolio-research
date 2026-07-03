"""Signal sweep: bull-regime narrow-breadth / high-dispersion signals.

Target regime: strong or moderate bull market with narrow breadth and high
cross-sectional dispersion — 2023/2024/2025-style. A few sectors/stocks
dominate, trends are persistent, winners keep winning.

Signal families:
  1. Cross-sectional momentum (raw, Sharpe, skip-5) gated on high dispersion
  2. MA distance momentum — stocks far above long-term MA in a trend-dominated market
  3. 52-week high proximity — stocks near highs keep going in a bull
  4. Sector ETF momentum gated on high dispersion — sector leaders compound
  5. Within-sector relative momentum — sub-leaders within winning sectors
  6. Low-vol momentum — consistent grinders outperform high-beta names in narrow bull
  7. Residual momentum gated on dispersion — factor-stripped trend alpha

Conditioning filters:
  - disp_60_q75: 60d smoothed cross-sectional dispersion > 75th percentile
  - disp_60_q60: same but > 60th percentile (wider gate)
  - breadth_lt50: pct stocks above 200d MA < 50%
  - disp_q60_and_breadth_lt50: BOTH high dispersion AND narrow breadth
  - uptrend_50_200: SPY 50d MA > 200d MA (bull confirmation)
  - disp_q60_and_uptrend: high dispersion AND SPY in uptrend

Scaler configs:
  - none
  - trend_50_200_mom: scale down in downtrend (momentum-style)
  - trend_20_100_mom: faster momentum trend scaler
  - vol_20_60: scale down in high vol (momentum crashes in stress)

Usage:
    uv run python examples/signal_sweeps/signal_sweep_bull_narrow_breadth.py
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
    REBALANCE_PERIODS,
    TRAIN_START,
    load_data,
    run_sweep,
)

GROUP = "bull-narrow-breadth"
OUT_DIR = Path(__file__).resolve().parent / "out" / "bull-narrow-breadth"

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


def _sector_etf_returns(
    r: pd.DataFrame, factor_returns: pd.DataFrame, sector_etf_map: dict
) -> pd.DataFrame:
    out = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for ticker in r.columns:
        etf = sector_etf_map.get(ticker, "SPY")
        col = etf if etf in factor_returns.columns else "SPY"
        out[ticker] = factor_returns[col].reindex(r.index).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def make_signals(sector_etf_map: dict[str, str]) -> list[dict]:
    signals = []

    # --- 1. Cross-sectional momentum ---
    # In high-dispersion bull markets, momentum is strong and persistent.
    for w in [120, 252]:
        window = w

        def _mom(window=window, **cache):
            return cache["_active_returns"].rolling(window).mean()

        _mom.__name__ = f"mom_{w}d"
        signals.append({"name": f"mom_{w}d", "fn": _mom})

    # --- 2. Sharpe momentum ---
    # Risk-adjusted momentum — rewards consistent trends over noisy spikes.
    for w in [120, 252]:
        window = w

        def _sharpe_mom(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            sigma = r.rolling(window).std().clip(lower=1e-8)
            return mu / sigma

        _sharpe_mom.__name__ = f"sharpe_mom_{w}d"
        signals.append({"name": f"sharpe_mom_{w}d", "fn": _sharpe_mom})

    # --- 3. Skip-5 momentum ---
    # Standard academic momentum avoiding short-term reversal.
    for w in [120, 252]:
        window = w

        def _skip_mom(window=window, **cache):
            return cache["_active_returns"].shift(5).rolling(window).mean()

        _skip_mom.__name__ = f"skip_mom_{w}d"
        signals.append({"name": f"skip_mom_{w}d", "fn": _skip_mom})

    # --- 4. MA distance momentum ---
    # Price / rolling MA - 1. Stocks far above long-term MA in a bull keep going.
    for fast, slow in [(50, 200), (20, 200), (20, 100)]:
        f, s = fast, slow

        def _ma_dist(f=f, s=s, **cache):
            r = cache["_active_returns"]
            price = (1 + r).cumprod()
            return price.rolling(f).mean() / price.rolling(s).mean().clip(lower=1e-8) - 1

        _ma_dist.__name__ = f"ma_dist_{f}_{s}d"
        signals.append({"name": f"ma_dist_{f}_{s}d", "fn": _ma_dist})

    # --- 5. 52-week high proximity ---
    # Stocks near 52-week highs in a bull market continue to outperform.
    # Signal: price / rolling_252d_max. High value = near 52w high → long.
    def _high_prox(**cache):
        r = cache["_active_returns"]
        price = (1 + r).cumprod()
        high_252 = price.rolling(252).max().clip(lower=1e-8)
        return price / high_252  # near 1.0 = near 52w high → long

    _high_prox.__name__ = "high_prox_252d"
    signals.append({"name": "high_prox_252d", "fn": _high_prox})

    # --- 6. Sector ETF momentum (bull-regime) ---
    # Long stocks in leading sectors, short laggards. In a narrow bull,
    # sector dispersion is high and sector leadership is persistent.
    for w in [120, 252]:
        window = w

        def _etf_mom(window=window, sector_etf_map=sector_etf_map, **cache):
            r = cache["_active_returns"]
            sec_r = _sector_etf_returns(r, cache["factor_returns"], sector_etf_map)
            return sec_r.rolling(window).mean()

        _etf_mom.__name__ = f"etf_mom_{w}d"
        signals.append({"name": f"etf_mom_{w}d", "fn": _etf_mom})

    # --- 7. Sector ETF momentum blend (sector + within-sector) ---
    for w in [120, 252]:
        window = w

        def _etf_blend(window=window, sector_etf_map=sector_etf_map, **cache):
            r = cache["_active_returns"]
            sec_r = _sector_etf_returns(r, cache["factor_returns"], sector_etf_map)
            sector_signal = sec_r.rolling(window).mean()
            rel_signal = (r - sec_r).rolling(window).mean()
            return 0.7 * sector_signal + 0.3 * rel_signal

        _etf_blend.__name__ = f"etf_blend_{w}d"
        signals.append({"name": f"etf_blend_{w}d", "fn": _etf_blend})

    # --- 8. Within-sector relative momentum ---
    # In a narrow bull, sub-leaders within winning sectors compound further.
    for w in [60, 120]:
        window = w

        def _sector_rel_mom(window=window, sector_etf_map=sector_etf_map, **cache):
            r = cache["_active_returns"]
            sec_r = _sector_etf_returns(r, cache["factor_returns"], sector_etf_map)
            return (r - sec_r).rolling(window).mean()

        _sector_rel_mom.__name__ = f"sector_rel_mom_{w}d"
        signals.append({"name": f"sector_rel_mom_{w}d", "fn": _sector_rel_mom})

    # --- 9. Low-vol momentum ---
    # In a narrow bull, consistent low-vol grinders outperform high-beta names.
    # Signal: Sharpe / realized_vol. Rewards high return per unit of risk.
    for w in [120, 252]:
        window = w

        def _low_vol_mom(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            sigma = r.rolling(window).std().clip(lower=1e-8)
            # Weight Sharpe by inverse vol: reward consistent low-vol winners
            return (mu / sigma) / sigma

        _low_vol_mom.__name__ = f"low_vol_mom_{w}d"
        signals.append({"name": f"low_vol_mom_{w}d", "fn": _low_vol_mom})

    # --- 10. Residual momentum ---
    # Factor-stripped momentum. In a narrow bull, idiosyncratic trend
    # (beyond market/sector beta) is concentrated in a few winners.
    for w in [120, 252]:
        window = w

        def _resid_mom(window=window, **cache):
            resid = cache.get("residual_returns")
            if resid is None:
                return cache["_active_returns"].rolling(window).mean()
            return resid.rolling(window).mean()

        _resid_mom.__name__ = f"resid_mom_{w}d"
        signals.append({"name": f"resid_mom_{w}d", "fn": _resid_mom})

    return signals


# ---------------------------------------------------------------------------
# Conditioning filters
# ---------------------------------------------------------------------------


def _make_disp_filter(window: int, quantile: float) -> object:
    """Active when cross-sectional dispersion > quantile threshold."""

    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        disp = cache["returns"].std(axis=1)
        disp_smooth = disp.rolling(window, min_periods=window // 2).mean()
        thresh = disp_smooth.rolling(252, min_periods=126).quantile(quantile)
        mask = disp_smooth.gt(thresh).reindex(signal.index).fillna(False)
        return signal.where(mask, other=np.nan)

    _filter.__name__ = f"disp_{window}_q{int(quantile * 100)}"
    return _filter


def _make_breadth_filter(threshold: float) -> object:
    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        prices = (1 + cache["returns"]).cumprod()
        ma_200 = prices.rolling(200).mean()
        pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
        mask = pct_above.lt(threshold).reindex(signal.index).fillna(False)
        return signal.where(mask, other=np.nan)

    _filter.__name__ = f"breadth_lt{int(threshold * 100)}"
    return _filter


def _make_uptrend_filter(fast: int, slow: int) -> object:
    """Active when SPY fast MA > slow MA (bull confirmation)."""

    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        bm = cache.get("benchmark")
        if bm is None:
            return signal
        spy_price = (1 + bm).cumprod()
        uptrend = spy_price.rolling(fast).mean().gt(spy_price.rolling(slow).mean())
        mask = uptrend.reindex(signal.index).fillna(False)
        return signal.where(mask, other=np.nan)

    _filter.__name__ = f"uptrend_{fast}_{slow}"
    return _filter


def _make_disp_and_uptrend_filter(disp_window: int, disp_q: float, fast: int, slow: int) -> object:
    """Active when BOTH dispersion is high AND SPY is in uptrend."""

    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        disp = cache["returns"].std(axis=1)
        disp_smooth = disp.rolling(disp_window, min_periods=disp_window // 2).mean()
        thresh = disp_smooth.rolling(252, min_periods=126).quantile(disp_q)
        high_disp = disp_smooth.gt(thresh).reindex(signal.index).fillna(False)

        bm = cache.get("benchmark")
        if bm is not None:
            spy_price = (1 + bm).cumprod()
            uptrend = spy_price.rolling(fast).mean().gt(spy_price.rolling(slow).mean())
            uptrend = uptrend.reindex(signal.index).fillna(False)
        else:
            uptrend = pd.Series(True, index=signal.index)

        return signal.where(high_disp & uptrend, other=np.nan)

    _filter.__name__ = f"disp_{disp_window}_q{int(disp_q * 100)}_and_uptrend_{fast}_{slow}"
    return _filter


def _make_disp_and_breadth_filter(
    disp_window: int, disp_q: float, breadth_threshold: float
) -> object:
    """Active when BOTH dispersion is high AND breadth is narrow."""

    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        disp = cache["returns"].std(axis=1)
        disp_smooth = disp.rolling(disp_window, min_periods=disp_window // 2).mean()
        thresh = disp_smooth.rolling(252, min_periods=126).quantile(disp_q)
        high_disp = disp_smooth.gt(thresh).reindex(signal.index).fillna(False)

        prices = (1 + cache["returns"]).cumprod()
        ma_200 = prices.rolling(200).mean()
        pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
        narrow = pct_above.lt(breadth_threshold).reindex(signal.index).fillna(False)

        return signal.where(high_disp & narrow, other=np.nan)

    _filter.__name__ = (
        f"disp_{disp_window}_q{int(disp_q * 100)}_and_breadth_lt{int(breadth_threshold * 100)}"
    )
    return _filter


CONDITIONING_FILTERS = {
    # Pure dispersion gates
    "disp_60_q75": _make_disp_filter(60, 0.75),
    "disp_60_q60": _make_disp_filter(60, 0.60),
    # Pure breadth gate
    "breadth_lt50": _make_breadth_filter(0.50),
    # Pure uptrend gate
    "uptrend_50_200": _make_uptrend_filter(50, 200),
    "uptrend_20_100": _make_uptrend_filter(20, 100),
    # Combined: high dispersion AND uptrend (bull narrow breadth confirmation)
    "disp_q60_and_uptrend_50_200": _make_disp_and_uptrend_filter(60, 0.60, 50, 200),
    "disp_q75_and_uptrend_50_200": _make_disp_and_uptrend_filter(60, 0.75, 50, 200),
    # Combined: high dispersion AND narrow breadth
    "disp_q60_and_breadth_lt50": _make_disp_and_breadth_filter(60, 0.60, 0.50),
}


# ---------------------------------------------------------------------------
# Scaler configs — momentum-style
# ---------------------------------------------------------------------------


def make_scaler_configs() -> list[dict]:
    return [
        {"tag": "none", "trend": None, "vol": None},
        {"tag": "trend_50_200_mom", "trend": {"fast": 50, "slow": 200, "mr": False}, "vol": None},
        {"tag": "trend_20_100_mom", "trend": {"fast": 20, "slow": 100, "mr": False}, "vol": None},
        {"tag": "vol_20_60", "trend": None, "vol": {"fast": 20, "slow": 60}},
    ]


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------


def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    fn = entry["fn"]
    cond_filter = entry["cond_filter"]
    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)

    builder = (
        Study(universe=universe, benchmark=benchmark, factors=factors, verbose=verbose)
        .base_signal(fn)
        .add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
    )

    if cond_filter is not None:
        builder = builder.add_filter(cond_filter)

    builder = (
        builder.build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=equity_curve_scaler)
    )

    trend_cfg = scaler_cfg.get("trend")
    if trend_cfg is not None:
        fast = trend_cfg["fast"]
        slow = trend_cfg["slow"]
        mr_style = trend_cfg.get("mr", False)

        def _trend(positions, fast=fast, slow=slow, mr_style=mr_style, **cache):
            bm = cache.get("benchmark")
            if bm is None:
                return positions
            spy_price = (1 + bm).cumprod()
            in_uptrend = spy_price.rolling(fast).mean() > spy_price.rolling(slow).mean()
            scale_vals = np.where(
                in_uptrend.reindex(positions.index).fillna(False),
                1.0 if not mr_style else 0.25,
                0.25 if not mr_style else 1.0,
            )
            return positions.mul(pd.Series(scale_vals, index=positions.index).shift(1), axis=0)

        _trend.__name__ = f"trend_{fast}_{slow}"
        builder = builder.scale_risk(fn=_trend)

    vol_cfg = scaler_cfg.get("vol")
    if vol_cfg is not None:
        fast = vol_cfg["fast"]
        slow = vol_cfg["slow"]

        def _vol(positions, fast=fast, slow=slow, **cache):
            bm = cache.get("benchmark")
            if bm is None:
                return positions
            fv = bm.rolling(fast).std()
            sv = bm.rolling(slow).std()
            in_spike = (fv > sv).reindex(positions.index).fillna(False)
            scale_vals = np.where(in_spike, 0.25, 1.0)
            return positions.mul(pd.Series(scale_vals, index=positions.index).shift(1), axis=0)

        _vol.__name__ = f"vol_{fast}_{slow}"
        builder = builder.scale_risk(fn=_vol)

    return builder.rebalance(every=rebalance).run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    universe, _, _ = load_data()
    sector_map = qs.get_sector_map(list(universe.returns.columns))
    sector_etf_map = {t: GICS_TO_ETF.get(s, "SPY") for t, s in sector_map.items()}

    base_signals = make_signals(sector_etf_map)
    scaler_configs = make_scaler_configs()

    signals: list[dict] = []
    for sig in base_signals:
        for filter_name, cond_filter in CONDITIONING_FILTERS.items():
            name = f"{sig['name']}__cond__{filter_name}"
            signals.append(
                {
                    "name": name,
                    "fn": sig["fn"],
                    "cond_filter": cond_filter,
                    "filters": filter_name,
                }
            )

    total = len(signals) * len(scaler_configs) * len(REBALANCE_PERIODS)
    print(
        f"Bull-narrow sweep: {len(base_signals)} signals × {len(CONDITIONING_FILTERS)} filters "
        f"× {len(scaler_configs)} scalers × {len(REBALANCE_PERIODS)} rebalance = {total} configs"
    )

    run_sweep(
        group=GROUP,
        signals=signals,
        scaler_configs=scaler_configs,
        rebalance_periods=REBALANCE_PERIODS,
        build_study_fn=build_study_fn,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
