"""Signal sweep: bear-regime narrow-breadth signals.

Target regime: persistent bear market with narrow breadth (2022-style).
  - Pct stocks above 200d MA < 40% (strict narrow bear gate)
  - SPY in downtrend (fast MA < slow MA)

Signal families:
  1. Below-200d MA gap momentum — stocks furthest below their 200d MA (short them)
  2. Sector ETF bear momentum — short stocks in weakest sectors (MR-style: scale up in downtrend)
  3. High-beta in downtrend — short rising-beta stocks in a bear
  4. Realized vol acceleration — short stocks with accelerating realized vol
  5. Bear short-term reversal — in a bear, short recent 5d/10d winners (they get sold)
  6. 52-week low proximity — short stocks nearest their 52-week low (momentum continuation)
  7. Down-trend strength — rank by slope of price decline (faster falling = short)

Conditioning filters:
  - bear_narrow_lt40: breadth < 40% AND SPY in downtrend (50d < 200d MA)
  - bear_narrow_lt50: breadth < 50% AND SPY in downtrend
  - breadth_lt40: pure breadth gate (no trend requirement)
  - downtrend_only: SPY 50d < 200d MA, no breadth gate

Scaler configs:
  - none
  - trend_50_200_mr: scale down in uptrend (MR-style — these are bear signals)
  - vol_20_60_up: scale UP in high vol (bear signals benefit from stress)

Usage:
    uv run python examples/signal_sweeps/signal_sweep_bear_narrow_breadth.py
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
    N_LONG,
    N_SHORT,
    REBALANCE_PERIODS,
    TRAIN_START,
    load_data,
    run_sweep,
)

GROUP = "bear-narrow-breadth"
OUT_DIR = Path(__file__).resolve().parent / "out" / "bear-narrow-breadth"

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

    # --- 1. Below-200d MA gap momentum ---
    # Stocks furthest below their 200d MA tend to keep falling in a bear.
    # Signal: -(price / ma_200 - 1). Negative gap = below MA → positive signal → long.
    # We want to SHORT these (furthest below), so we negate: most negative gap → highest rank → long.
    # Actually: in a bear we want to SHORT the weakest, so signal = price/ma200 - 1
    # (highest = furthest above MA → long; lowest = furthest below → short). That's momentum.
    for w in [100, 200]:
        window = w

        def _ma_gap(window=window, **cache):
            r = cache["_active_returns"]
            price = (1 + r).cumprod()
            ma = price.rolling(window).mean()
            return (
                price / ma.clip(lower=1e-8) - 1
            )  # positive = above MA (long); negative = below (short)

        _ma_gap.__name__ = f"ma_gap_{w}d"
        signals.append({"name": f"ma_gap_{w}d", "fn": _ma_gap})

    # --- 2. Sector ETF bear momentum ---
    # In a bear, weak sectors keep falling. Short stocks in weak sectors.
    # Same as sector ETF momentum but intended to be used in a bear gate.
    for w in [60, 120]:
        window = w

        def _sector_bear_mom(window=window, sector_etf_map=sector_etf_map, **cache):
            r = cache["_active_returns"]
            sec_r = _sector_etf_returns(r, cache["factor_returns"], sector_etf_map)
            return sec_r.rolling(window).mean()  # long strong sectors, short weak

        _sector_bear_mom.__name__ = f"sector_bear_mom_{w}d"
        signals.append({"name": f"sector_bear_mom_{w}d", "fn": _sector_bear_mom})

    # --- 3. High-beta in downtrend (short rising-beta stocks) ---
    # Beta momentum: stocks with rising beta are becoming more correlated with a falling market.
    # Short them. Negate so falling beta → long.
    for fast, slow in [(20, 120), (60, 252)]:
        f, s = fast, slow

        def _beta_accel(f=f, s=s, **cache):
            r = cache["_active_returns"]
            bm = cache.get("benchmark")
            if bm is None:
                return pd.DataFrame(np.nan, index=r.index, columns=r.columns)
            mkt = bm.reindex(r.index).fillna(0.0)
            beta_fast = _rolling_beta_df(r, mkt, f)
            beta_slow = _rolling_beta_df(r, mkt, s)
            return -(beta_fast - beta_slow)  # falling beta → positive → long

        _beta_accel.__name__ = f"beta_accel_{f}_{s}d"
        signals.append({"name": f"beta_accel_{f}_{s}d", "fn": _beta_accel})

    # --- 4. Realized vol acceleration (short stocks with accelerating vol) ---
    # In a bear, stocks with fast-rising realized vol are in distress.
    # Short them: high vol acceleration → negative signal → short.
    for fast, slow in [(5, 60), (10, 90), (20, 120)]:
        f, s = fast, slow

        def _vol_accel(f=f, s=s, **cache):
            r = cache["_active_returns"]
            return -(r.rolling(f).std() - r.rolling(s).std())  # vol compressing → long

        _vol_accel.__name__ = f"vol_accel_{f}_{s}d"
        signals.append({"name": f"vol_accel_{f}_{s}d", "fn": _vol_accel})

    # --- 5. Bear short-term reversal (short recent winners) ---
    # In a bear market, stocks that bounce get sold. Short recent 5d/10d winners.
    # Signal: -(recent return). Negative recent return → positive signal → long.
    for w in [5, 10, 20]:
        window = w

        def _bear_reversal(window=window, **cache):
            return -cache["_active_returns"].rolling(window).mean()

        _bear_reversal.__name__ = f"bear_reversal_{w}d"
        signals.append({"name": f"bear_reversal_{w}d", "fn": _bear_reversal})

    # --- 6. 52-week low proximity (momentum continuation) ---
    # In a bear, stocks near their 52-week low keep falling.
    # Signal: price / rolling_252d_min - 1. Low value = near 52w low → short.
    def _low_proximity(**cache):
        r = cache["_active_returns"]
        price = (1 + r).cumprod()
        low_252 = price.rolling(252).min().clip(lower=1e-8)
        return price / low_252 - 1  # near 0 = near 52w low → short; high = far above → long

    _low_proximity.__name__ = "low_proximity_252d"
    signals.append({"name": "low_proximity_252d", "fn": _low_proximity})

    # --- 7. Down-trend strength (price slope) ---
    # Rank by steepness of price decline over trailing window.
    # Computed as linear regression slope of log-price (normalized).
    for w in [60, 120]:
        window = w

        def _trend_slope(window=window, **cache):
            r = cache["_active_returns"]
            log_price = np.log((1 + r).cumprod().clip(lower=1e-8))
            # Approximate slope via (end - start) / window as a fast proxy
            return log_price - log_price.shift(window)  # positive = uptrend → long

        _trend_slope.__name__ = f"trend_slope_{w}d"
        signals.append({"name": f"trend_slope_{w}d", "fn": _trend_slope})

    return signals


# ---------------------------------------------------------------------------
# Conditioning filters — bear regime gates
# ---------------------------------------------------------------------------


def _make_bear_narrow_filter(breadth_threshold: float) -> object:
    """Active when breadth < threshold AND SPY is in downtrend (50d < 200d MA)."""

    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        # Breadth gate
        prices = (1 + cache["returns"]).cumprod()
        ma_200 = prices.rolling(200).mean()
        pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
        narrow = pct_above.lt(breadth_threshold).reindex(signal.index).fillna(False)

        # Downtrend gate: SPY 50d MA < 200d MA
        bm = cache.get("benchmark")
        if bm is not None:
            spy_price = (1 + bm).cumprod()
            downtrend = spy_price.rolling(50).mean().lt(spy_price.rolling(200).mean())
            downtrend = downtrend.reindex(signal.index).fillna(False)
        else:
            downtrend = pd.Series(True, index=signal.index)

        mask = narrow & downtrend
        return signal.where(mask, other=np.nan)

    _filter.__name__ = f"bear_narrow_lt{int(breadth_threshold * 100)}"
    return _filter


def _make_breadth_filter(threshold: float) -> object:
    """Pure breadth gate: active when pct above 200d MA < threshold."""

    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        prices = (1 + cache["returns"]).cumprod()
        ma_200 = prices.rolling(200).mean()
        pct_above = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
        mask = pct_above.lt(threshold).reindex(signal.index).fillna(False)
        return signal.where(mask, other=np.nan)

    _filter.__name__ = f"breadth_lt{int(threshold * 100)}"
    return _filter


def _make_downtrend_filter(fast: int, slow: int) -> object:
    """Pure downtrend gate: active when SPY fast MA < slow MA."""

    def _filter(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        bm = cache.get("benchmark")
        if bm is None:
            return signal
        spy_price = (1 + bm).cumprod()
        downtrend = spy_price.rolling(fast).mean().lt(spy_price.rolling(slow).mean())
        mask = downtrend.reindex(signal.index).fillna(False)
        return signal.where(mask, other=np.nan)

    _filter.__name__ = f"downtrend_{fast}_{slow}"
    return _filter


CONDITIONING_FILTERS = {
    # Strict bear + narrow: requires BOTH downtrend AND narrow breadth
    "bear_narrow_lt40": _make_bear_narrow_filter(0.40),
    "bear_narrow_lt50": _make_bear_narrow_filter(0.50),
    # Pure breadth (no trend requirement) — fires in narrow bear AND narrow bull
    "breadth_lt40": _make_breadth_filter(0.40),
    "breadth_lt50": _make_breadth_filter(0.50),
    # Pure downtrend (no breadth requirement) — fires whenever SPY is below its MA
    "downtrend_50_200": _make_downtrend_filter(50, 200),
    "downtrend_20_100": _make_downtrend_filter(20, 100),
}


# ---------------------------------------------------------------------------
# Scaler configs
# ---------------------------------------------------------------------------


def make_scaler_configs() -> list[dict]:
    return [
        {"tag": "none", "trend": None, "vol": None},
        # MR-style: scale down in uptrend (bear signals work in downtrends)
        {"tag": "trend_50_200_mr", "trend": {"fast": 50, "slow": 200, "mr": True}, "vol": None},
        {"tag": "trend_20_100_mr", "trend": {"fast": 20, "slow": 100, "mr": True}, "vol": None},
        # Vol: scale UP in high vol (bear + stress = best environment for these signals)
        {"tag": "vol_20_60_up", "trend": None, "vol": {"fast": 20, "slow": 60, "invert": True}},
        {"tag": "vol_10_60_up", "trend": None, "vol": {"fast": 10, "slow": 60, "invert": True}},
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
        mr_style = trend_cfg.get("mr", True)

        def _trend(positions, fast=fast, slow=slow, mr_style=mr_style, **cache):
            bm = cache.get("benchmark")
            if bm is None:
                return positions
            spy_price = (1 + bm).cumprod()
            in_uptrend = spy_price.rolling(fast).mean() > spy_price.rolling(slow).mean()
            scale_vals = np.where(
                in_uptrend.reindex(positions.index).fillna(False),
                0.25 if mr_style else 1.0,
                1.0 if mr_style else 0.25,
            )
            return positions.mul(pd.Series(scale_vals, index=positions.index).shift(1), axis=0)

        _trend.__name__ = f"trend_{fast}_{slow}"
        builder = builder.scale_risk(fn=_trend)

    vol_cfg = scaler_cfg.get("vol")
    if vol_cfg is not None:
        fast = vol_cfg["fast"]
        slow = vol_cfg["slow"]
        invert = vol_cfg.get("invert", False)

        def _vol(positions, fast=fast, slow=slow, invert=invert, **cache):
            bm = cache.get("benchmark")
            if bm is None:
                return positions
            fv = bm.rolling(fast).std()
            sv = bm.rolling(slow).std()
            in_spike = (fv > sv).reindex(positions.index).fillna(False)
            scale_vals = np.where(in_spike, 1.0 if invert else 0.25, 0.25 if invert else 1.0)
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
        f"Bear-narrow sweep: {len(base_signals)} signals x {len(CONDITIONING_FILTERS)} filters "
        f"x {len(scaler_configs)} scalers x {len(REBALANCE_PERIODS)} rebalance = {total} configs"
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
