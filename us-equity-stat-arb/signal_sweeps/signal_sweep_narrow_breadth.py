"""Full-period signal sweep: narrow-breadth regime signals.

Targets years where market breadth is narrow (2022, 2023, 2024, 2025):
  - 2022: bear + narrow + high vol
  - 2023+: bull + narrow + high dispersion

The key design principle: every config here uses a breadth or dispersion
gate as a CONDITIONING FILTER (not a scaler) so the signal goes completely
flat (NaN positions) when breadth is not narrow. This means it contributes
nothing in broad-market years and only acts when the regime matches.

Signal families:
  1. Sector ETF momentum (252d blend, 120d) — captures sector concentration
  2. High-dispersion momentum — cross-sectional winner concentration
  3. Sharpe momentum gated by dispersion — risk-adjusted momentum in narrow markets
  4. Beta momentum gated by breadth — defensive tilt in narrow bear regimes
  5. Within-sector relative momentum — stock selection within leading sectors

Conditioning filters (applied as signal gates, not scalers):
  - breadth_weak_40: pct stocks above 200d MA < 40% (narrow bear, 2022)
  - breadth_weak_50: pct stocks above 200d MA < 50% (wider narrow gate)
  - dispersion_high_q75: cross-sectional dispersion > 75th percentile (narrow bull, 2023+)
  - dispersion_high_q60: cross-sectional dispersion > 60th percentile (wider gate)

Scaler configs: minimal — equity curve scaler only, plus optional trend/vol
overlays. The conditioning filter already does the heavy lifting.

Usage:
    uv run python examples/signal_sweeps/walkforward_signal_sweep_narrow_breadth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import qstudy as qs
from qstudy import Study

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
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

GROUP = "narrow-breadth"
OUT_DIR = Path(__file__).resolve().parent / "out" / "narrow-breadth"

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


def _rolling_beta_df(r: pd.DataFrame, mkt: pd.Series, window: int) -> pd.DataFrame:
    mkt_var = mkt.rolling(window).var().clip(lower=1e-10)
    betas = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for col in r.columns:
        s = r[col].fillna(0.0)
        cov = (s * mkt).rolling(window).mean() - s.rolling(window).mean() * mkt.rolling(
            window
        ).mean()
        betas[col] = cov / mkt_var
    return betas


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def make_signals(sector_etf_map: dict[str, str]) -> list[dict]:
    signals = []

    # --- 1. Sector ETF momentum (252d blend) ---
    # Buys stocks in strong sectors, shorts stocks in weak sectors.
    # Works in 2023+ bull narrow-breadth where sector leadership is concentrated.
    def _etf_mom_blend_252d(sector_etf_map=sector_etf_map, **cache):
        r = cache["_active_returns"]
        sec_r = _sector_etf_returns(r, cache["factor_returns"], sector_etf_map)
        sector_signal = sec_r.rolling(252).mean()
        rel_signal = (r - sec_r).rolling(252).mean()
        return 0.7 * sector_signal + 0.3 * rel_signal

    _etf_mom_blend_252d.__name__ = "etf_mom_blend_252d"
    signals.append({"name": "etf_mom_blend_252d", "fn": _etf_mom_blend_252d})

    # --- 2. Sector ETF momentum (120d) ---
    # Faster-responding variant — better for 2022 where sector rotation happened quickly.
    def _etf_mom_120d(sector_etf_map=sector_etf_map, **cache):
        r = cache["_active_returns"]
        sec_r = _sector_etf_returns(r, cache["factor_returns"], sector_etf_map)
        return sec_r.rolling(120).mean()

    _etf_mom_120d.__name__ = "etf_mom_120d"
    signals.append({"name": "etf_mom_120d", "fn": _etf_mom_120d})

    # --- 3. Sector ETF Sharpe momentum (120d) ---
    # Risk-adjusted sector momentum — rewards consistent sector trends.
    def _etf_sharpe_mom_120d(sector_etf_map=sector_etf_map, **cache):
        r = cache["_active_returns"]
        sec_r = _sector_etf_returns(r, cache["factor_returns"], sector_etf_map)
        mu = sec_r.rolling(120).mean()
        sigma = sec_r.rolling(120).std().clip(lower=1e-8)
        return mu / sigma

    _etf_sharpe_mom_120d.__name__ = "etf_sharpe_mom_120d"
    signals.append({"name": "etf_sharpe_mom_120d", "fn": _etf_sharpe_mom_120d})

    # --- 4. High-dispersion momentum ---
    # Pure cross-sectional momentum on raw returns. In narrow-breadth regimes
    # the cross-section is highly dispersed and winners/losers are persistent.
    for w in [120, 252]:
        window = w

        def _disp_mom(window=window, **cache):
            return cache["_active_returns"].rolling(window).mean()

        _disp_mom.__name__ = f"mom_{w}d"
        signals.append({"name": f"mom_{w}d", "fn": _disp_mom})

    # --- 5. Sharpe momentum ---
    # Risk-adjusted momentum. In high-dispersion narrow markets, Sharpe
    # ranking picks up consistent winners more cleanly than raw return.
    for w in [120, 252]:
        window = w

        def _sharpe_mom(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            sigma = r.rolling(window).std().clip(lower=1e-8)
            return mu / sigma

        _sharpe_mom.__name__ = f"sharpe_mom_{w}d"
        signals.append({"name": f"sharpe_mom_{w}d", "fn": _sharpe_mom})

    # --- 6. Skip-5 momentum ---
    # Avoids short-term reversal noise. Standard academic momentum.
    for w in [120, 252]:
        window = w

        def _skip_mom(window=window, **cache):
            return cache["_active_returns"].shift(5).rolling(window).mean()

        _skip_mom.__name__ = f"skip_mom_{w}d"
        signals.append({"name": f"skip_mom_{w}d", "fn": _skip_mom})

    # --- 7. Beta momentum ---
    # Stocks whose beta is falling (becoming defensive) in narrow-bear (2022).
    # In narrow-bull (2023+) high-beta growth leaders have rising beta — short them.
    def _beta_mom(sector_etf_map=sector_etf_map, **cache):
        r = cache["_active_returns"]
        bm = cache.get("benchmark")
        if bm is None:
            return pd.DataFrame(np.nan, index=r.index, columns=r.columns)
        mkt = bm.reindex(r.index).fillna(0.0)
        beta_fast = _rolling_beta_df(r, mkt, 60)
        beta_slow = _rolling_beta_df(r, mkt, 252)
        return -(beta_fast - beta_slow)

    _beta_mom.__name__ = "beta_momentum_60_252d"
    signals.append({"name": "beta_momentum_60_252d", "fn": _beta_mom})

    # --- 8. Within-sector relative momentum ---
    # Stock return vs its sector ETF, rolled 120d. When breadth is narrow,
    # picking within-sector winners captures the concentrated alpha.
    for w in [60, 120]:
        window = w

        def _sector_rel_mom(window=window, sector_etf_map=sector_etf_map, **cache):
            r = cache["_active_returns"]
            sec_r = _sector_etf_returns(r, cache["factor_returns"], sector_etf_map)
            return (r - sec_r).rolling(window).mean()

        _sector_rel_mom.__name__ = f"sector_rel_mom_{w}d"
        signals.append({"name": f"sector_rel_mom_{w}d", "fn": _sector_rel_mom})

    return signals


# ---------------------------------------------------------------------------
# Conditioning filters — these are the core regime gates
# ---------------------------------------------------------------------------


def _make_breadth_filter(threshold: float) -> object:
    """Signal goes NaN when pct stocks above 200d MA >= threshold."""

    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        prices = (1 + cache["returns"]).cumprod()
        ma_200 = prices.rolling(200).mean()
        pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
        mask = pct_above.lt(threshold).reindex(signal.index).fillna(False)
        return signal.where(mask, other=np.nan)

    _filter.__name__ = f"breadth_lt_{int(threshold * 100)}"
    return _filter


def _make_dispersion_filter(window: int, quantile: float) -> object:
    """Signal goes NaN when cross-sectional dispersion < quantile threshold."""

    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        disp = cache["returns"].std(axis=1)
        disp_smooth = disp.rolling(window, min_periods=window // 2).mean()
        thresh = disp_smooth.rolling(252, min_periods=126).quantile(quantile)
        mask = disp_smooth.gt(thresh).reindex(signal.index).fillna(False)
        return signal.where(mask, other=np.nan)

    _filter.__name__ = f"disp_{window}_q{int(quantile * 100)}"
    return _filter


def _make_breadth_or_dispersion_filter(
    breadth_threshold: float, disp_window: int, disp_quantile: float
) -> object:
    """Signal active when EITHER breadth is narrow OR dispersion is high.
    Covers both 2022 (narrow bear) and 2023+ (narrow bull with high dispersion).
    """

    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        prices = (1 + cache["returns"]).cumprod()
        ma_200 = prices.rolling(200).mean()
        pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
        narrow = pct_above.lt(breadth_threshold).reindex(signal.index).fillna(False)

        disp = cache["returns"].std(axis=1)
        disp_smooth = disp.rolling(disp_window, min_periods=disp_window // 2).mean()
        thresh = disp_smooth.rolling(252, min_periods=126).quantile(disp_quantile)
        high_disp = disp_smooth.gt(thresh).reindex(signal.index).fillna(False)

        mask = narrow | high_disp
        return signal.where(mask, other=np.nan)

    _filter.__name__ = (
        f"breadth_lt{int(breadth_threshold * 100)}_or_disp_q{int(disp_quantile * 100)}"
    )
    return _filter


CONDITIONING_FILTERS = {
    # Pure breadth gates
    "breadth_lt40": _make_breadth_filter(0.40),  # strict narrow: < 40% above 200d MA
    "breadth_lt50": _make_breadth_filter(0.50),  # wider narrow gate
    # Pure dispersion gates
    "disp_60_q75": _make_dispersion_filter(60, 0.75),  # top quartile dispersion
    "disp_60_q60": _make_dispersion_filter(60, 0.60),  # top 40% dispersion
    # Combined: active in EITHER narrow breadth OR high dispersion
    # This covers 2022 (breadth < 40%) AND 2023+ (high dispersion bull)
    "breadth_lt50_or_disp_q60": _make_breadth_or_dispersion_filter(0.50, 60, 0.60),
    "breadth_lt50_or_disp_q75": _make_breadth_or_dispersion_filter(0.50, 60, 0.75),
    "breadth_lt40_or_disp_q75": _make_breadth_or_dispersion_filter(0.40, 60, 0.75),
}


# ---------------------------------------------------------------------------
# Scaler configs — minimal; conditioning filter does the regime work
# ---------------------------------------------------------------------------


def make_scaler_configs() -> list[dict]:
    return [
        {"tag": "none", "trend": None},
        # Momentum-style: scale down in downtrend
        {"tag": "trend_50_200_mom", "trend": {"fast": 50, "slow": 200, "mr": False}},
        # MR-style: scale down in uptrend (for beta_momentum / defensive signals)
        {"tag": "trend_50_200_mr", "trend": {"fast": 50, "slow": 200, "mr": True}},
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
            if mr_style:
                scale_vals = np.where(in_uptrend.reindex(positions.index).fillna(False), 0.25, 1.0)
            else:
                scale_vals = np.where(in_uptrend.reindex(positions.index).fillna(False), 1.0, 0.25)
            return positions.mul(pd.Series(scale_vals, index=positions.index).shift(1), axis=0)

        _trend.__name__ = f"trend_{fast}_{slow}"
        builder = builder.scale_risk(fn=_trend)

    return builder.rebalance(every=rebalance).run()


# ---------------------------------------------------------------------------
# Main — cross product of signals × conditioning filters × scalers × rebalance
# ---------------------------------------------------------------------------


def main() -> None:
    universe, _, _ = load_data()
    sector_map = qs.get_sector_map(list(universe.returns.columns))
    sector_etf_map = {t: GICS_TO_ETF.get(s, "SPY") for t, s in sector_map.items()}

    base_signals = make_signals(sector_etf_map)
    scaler_configs = make_scaler_configs()

    # Expand: each base signal × each conditioning filter = one sweep entry
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
        f"Narrow-breadth sweep: {len(base_signals)} signals × {len(CONDITIONING_FILTERS)} filters "
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
