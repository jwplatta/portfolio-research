"""Signal sweep: momentum zoo (~50 signals).

Signals cover: cumulative, risk-adjusted, residual, sector-relative, trend-quality,
regression-slope, R²-weighted, exponential, multi-horizon, acceleration, persistence,
volatility-conditioned, participation-confirmed, breadth, dispersion, low-vol,
trend-consistency, MA-distance, and composite momentum.

Only equity_curve_regime_scale is applied (no optional scaler sweep).

         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_momentum_zoo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import qstudy as qs

sys.path.insert(0, str(Path(__file__).parent.parent))
from portfolio_utils import make_equity_curve_regime_scale

from signal_sweep_utils import (
    TRAIN_START,
    COST_BPS,
    N_LONG,
    N_SHORT,
    REBALANCE_PERIODS,
    run_sweep,
    build_study_generic,
)

GROUP = "momentum-zoo"
OUT_DIR = Path(__file__).resolve().parent / "out" / "momentum-zoo"

# Module-level cache: universe id -> sector_map
_sector_map_cache: dict[int, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sector_map(universe) -> dict:
    key = id(universe)
    if key not in _sector_map_cache:
        _sector_map_cache[key] = qs.get_sector_map(list(universe.returns.columns))
    return _sector_map_cache[key]


def _rolling_ols(y: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (slope, r2) DataFrames from rolling OLS of log_price ~ time."""
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    x_ss = (x**2).sum()

    def _slope_r2(col: pd.Series) -> tuple[pd.Series, pd.Series]:
        slopes, r2s = [], []
        arr = col.values
        for i in range(len(arr)):
            if i < window - 1:
                slopes.append(np.nan)
                r2s.append(np.nan)
            else:
                y_win = arr[i - window + 1 : i + 1].astype(float)
                if np.isnan(y_win).any():
                    slopes.append(np.nan)
                    r2s.append(np.nan)
                else:
                    y_win = y_win - y_win.mean()
                    slope = (x * y_win).sum() / x_ss
                    y_hat = x * slope
                    ss_res = ((y_win - y_hat) ** 2).sum()
                    ss_tot = (y_win**2).sum()
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                    slopes.append(slope)
                    r2s.append(max(r2, 0.0))
        return (
            pd.Series(slopes, index=col.index),
            pd.Series(r2s, index=col.index),
        )

    slope_cols, r2_cols = {}, {}
    for ticker in y.columns:
        s, r = _slope_r2(y[ticker])
        slope_cols[ticker] = s
        r2_cols[ticker] = r
    return pd.DataFrame(slope_cols), pd.DataFrame(r2_cols)


def _build_sector_df(returns: pd.DataFrame, factor_returns: pd.DataFrame, sector_map: dict | None = None) -> pd.DataFrame:
    """Broadcast sector ETF returns to the stock universe via sector_map."""
    if sector_map is None:
        sector_map = {}
    etf_cols = factor_returns.columns.tolist()
    rows = {}
    for ticker in returns.columns:
        sector = sector_map.get(ticker, "Unknown")
        matched = [e for e in etf_cols if sector.lower() in e.lower()]
        etf = matched[0] if matched else etf_cols[0]
        rows[ticker] = factor_returns[etf]
    return pd.DataFrame(rows, index=returns.index)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:  # noqa: PLR0915
    signals = []

    def add(name: str, fn, use_residual: bool = False):
        fn.__name__ = name
        signals.append({"name": name, "use_residual": use_residual, "fn": fn})

    # ------------------------------------------------------------------
    # 1. Cumulative Return Momentum
    # ------------------------------------------------------------------
    for w in [20, 60, 120, 252]:
        window = w

        def _mom(window=window, **cache):
            return cache["_active_returns"].rolling(window).sum()

        add(f"mom_{window}d", _mom)

    # skip-1 month
    for w in [60, 120, 252]:
        window = w

        def _skip1_mom(window=window, **cache):
            return cache["_active_returns"].shift(21).rolling(window).sum()

        add(f"skip1_mom_{window}d", _skip1_mom)

    # log-return momentum
    for w in [60, 120]:
        window = w

        def _log_mom(window=window, **cache):
            return cache["log_returns"].rolling(window).sum()

        add(f"log_mom_{window}d", _log_mom)

    # ------------------------------------------------------------------
    # 2. Risk-Adjusted Momentum (rolling Sharpe)
    # ------------------------------------------------------------------
    for w in [60, 120, 252]:
        window = w

        def _sharpe_mom(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            sigma = r.rolling(window).std().clip(lower=1e-8)
            return mu / sigma

        add(f"sharpe_mom_{window}d", _sharpe_mom)

    # ------------------------------------------------------------------
    # 3. Residual Momentum
    # ------------------------------------------------------------------
    for w in [60, 120, 252]:
        window = w

        def _resid_mom(window=window, **cache):
            return cache["residual_returns"].rolling(window).mean()

        add(f"resid_mom_{window}d", _resid_mom, use_residual=True)

    # Residual Sharpe Momentum
    for w in [60, 120, 252]:
        for skip in [0, 5]:
            window, skip_ = w, skip

            def _resid_sharpe(window=window, skip=skip_, **cache):
                r = cache["residual_returns"]
                mu = r.shift(skip).rolling(window).mean()
                sigma = r.shift(skip).rolling(window).std().clip(lower=1e-8)
                return mu / sigma

            suffix = "" if skip_ == 0 else f"_skip{skip_}"
            name = f"resid_sharpe_{window}d{suffix}"
            add(name, _resid_sharpe, use_residual=True)

    # ------------------------------------------------------------------
    # 4. Sector-Relative Momentum
    # ------------------------------------------------------------------
    for w in [60, 120]:
        window = w

        def _sector_rel_mom(window=window, **cache):
            r = cache["_active_returns"]
            sector_df = _build_sector_df(r, cache["factor_returns"], cache.get("_sector_map"))
            return (r - sector_df).rolling(window).mean()

        add(f"sector_rel_mom_{window}d", _sector_rel_mom)

    # ------------------------------------------------------------------
    # 5. Relative-Strength Momentum (vs universe mean)
    # ------------------------------------------------------------------
    for w in [60, 120]:
        window = w

        def _rel_strength(window=window, **cache):
            r = cache["_active_returns"]
            excess = r.sub(r.mean(axis=1), axis=0)
            return excess.rolling(window).mean()

        add(f"rel_strength_{window}d", _rel_strength)

    # ------------------------------------------------------------------
    # 6 & 7. Trend-Quality / Regression-Slope Momentum
    # ------------------------------------------------------------------
    for w in [60, 120]:
        window = w

        def _slope_mom(window=window, **cache):
            log_price = cache["log_returns"].cumsum()
            slope, _ = _rolling_ols(log_price, window)
            return slope

        add(f"slope_mom_{window}d", _slope_mom)

        def _slope_vol(window=window, **cache):
            log_price = cache["log_returns"].cumsum()
            slope, _ = _rolling_ols(log_price, window)
            vol = cache["_active_returns"].rolling(window).std().clip(lower=1e-8)
            return slope / vol

        add(f"slope_vol_{window}d", _slope_vol)

    # ------------------------------------------------------------------
    # 8. R²-Weighted Momentum (slope * R²)
    # ------------------------------------------------------------------
    for w in [60, 120]:
        window = w

        def _slope_r2(window=window, **cache):
            log_price = cache["log_returns"].cumsum()
            slope, r2 = _rolling_ols(log_price, window)
            return slope * r2

        add(f"slope_r2_{window}d", _slope_r2)

    # ------------------------------------------------------------------
    # 9. Exponential Momentum
    # ------------------------------------------------------------------
    for alpha in [0.03, 0.05, 0.1]:
        a = alpha

        def _ewm_mom(a=a, **cache):
            return cache["_active_returns"].ewm(alpha=a, adjust=False).mean()

        add(f"ewm_mom_a{int(a*100):02d}", _ewm_mom)

    # ------------------------------------------------------------------
    # 10. Multi-Horizon Momentum
    # ------------------------------------------------------------------
    def _multi_horizon(**cache):
        r = cache["_active_returns"]
        m20 = r.rolling(20).mean()
        m60 = r.rolling(60).mean()
        m252 = r.rolling(252).mean()
        return 0.2 * m20 + 0.3 * m60 + 0.5 * m252

    add("multi_horizon_mom", _multi_horizon)

    # ------------------------------------------------------------------
    # 11. Momentum Acceleration (fast - slow)
    # ------------------------------------------------------------------
    for fast, slow in [(20, 60), (60, 120)]:
        f, s = fast, slow

        def _accel(f=f, s=s, **cache):
            r = cache["_active_returns"]
            return r.rolling(f).mean() - r.rolling(s).mean()

        add(f"mom_accel_{f}_{s}", _accel)

    # ------------------------------------------------------------------
    # 12. Momentum Persistence (hit rate — fraction of positive days)
    # ------------------------------------------------------------------
    for w in [60, 120]:
        window = w

        def _hit_rate(window=window, **cache):
            r = cache["_active_returns"]
            return (r > 0).rolling(window).mean()

        add(f"hit_rate_{window}d", _hit_rate)

    # ------------------------------------------------------------------
    # 13. Volatility-Conditioned Momentum (momentum / realized vol)
    # ------------------------------------------------------------------
    for w in [60, 120]:
        window = w

        def _vol_cond(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            vol = r.rolling(window).std().clip(lower=1e-8)
            return mu / vol

        add(f"vol_cond_mom_{window}d", _vol_cond)

    # ------------------------------------------------------------------
    # 14. Participation-Confirmed Momentum (return * relative volume)
    # ------------------------------------------------------------------
    for w in [60, 120]:
        window = w

        def _part_mom(window=window, **cache):
            r = cache["_active_returns"]
            volume = cache["volume"].reindex(columns=r.columns)
            rel_vol = volume / volume.rolling(window).mean().clip(lower=1e-8)
            return (r * rel_vol).rolling(window).mean()

        add(f"part_mom_{window}d", _part_mom)

    # ------------------------------------------------------------------
    # 15. Breadth Momentum (% positive days within window)
    # ------------------------------------------------------------------
    for w in [60, 120]:
        window = w

        def _breadth_mom(window=window, **cache):
            r = cache["_active_returns"]
            return (r > 0).rolling(window).mean() - 0.5

        add(f"breadth_mom_{window}d", _breadth_mom)

    # ------------------------------------------------------------------
    # 16. Dispersion-Scaled Momentum
    # ------------------------------------------------------------------
    for w in [60, 120]:
        window = w

        def _disp_mom(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            disp = r.std(axis=1).rolling(window).mean()
            return mu.mul(disp, axis=0)

        add(f"disp_mom_{window}d", _disp_mom)

    # ------------------------------------------------------------------
    # 17. Low-Vol Momentum
    # ------------------------------------------------------------------
    def _low_vol_mom(**cache):
        r = cache["_active_returns"]
        mu = r.rolling(252).mean()
        vol = r.rolling(252).std().clip(lower=1e-8)
        return mu / vol

    add("low_vol_mom_252d", _low_vol_mom)

    # ------------------------------------------------------------------
    # 19. Trend Consistency — sign consistency
    # ------------------------------------------------------------------
    for w in [60, 120]:
        window = w

        def _sign_consist(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            same_sign = ((r > 0) == (mu > 0)).rolling(window).mean()
            return same_sign

        add(f"sign_consist_{window}d", _sign_consist)

    for w in [60, 120]:
        window = w

        def _monoton(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            same_sign = (r.gt(0) == mu.gt(0)).astype(float)
            return same_sign.rolling(window).mean() * mu.abs()

        add(f"monoton_{window}d", _monoton)

    # ------------------------------------------------------------------
    # 22. Moving Average Distance
    # ------------------------------------------------------------------
    for fast, slow in [(20, 200), (50, 200)]:
        f, s = fast, slow

        def _ma_dist(f=f, s=s, **cache):
            r = cache["_active_returns"]
            price = (1 + r).cumprod()
            return price.rolling(f).mean() / price.rolling(s).mean() - 1

        add(f"ma_dist_{f}_{s}", _ma_dist)

    # ------------------------------------------------------------------
    # 23. Composite Momentum
    # ------------------------------------------------------------------
    def _composite(**cache):
        r = cache["_active_returns"]
        rr = cache["residual_returns"]
        sharpe = r.rolling(120).mean() / r.rolling(120).std().clip(lower=1e-8)
        resid = rr.rolling(120).mean()
        hit = (r > 0).rolling(60).mean()
        price = (1 + r).cumprod()
        ma = price.rolling(50).mean() / price.rolling(200).mean() - 1

        def _z(df):
            m, s = df.mean(axis=1), df.std(axis=1).clip(lower=1e-8)
            return df.sub(m, axis=0).div(s, axis=0)

        return (_z(sharpe) + _z(resid) + _z(hit) + _z(ma)) / 4

    add("composite_mom", _composite, use_residual=True)

    return signals


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------

def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    fn_orig = entry["fn"]
    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)

    # Inject sector_map into the signal fn via closure — sector_rel_mom signals need this.
    sector_map = _get_sector_map(universe)

    def fn(_sector_map=sector_map, **cache):
        return fn_orig(_sector_map=_sector_map, **cache)

    fn.__name__ = fn_orig.__name__

    # Build a modified entry with the wrapped fn and the sector_map for factor model.
    # use_factor_model=True triggers add_factor_model(..., sector_map=...) inside
    # build_study_generic, which matches what the original code did for residual signals.
    wrapped_entry = {**entry, "fn": fn}

    return build_study_generic(
        wrapped_entry, rebalance, scaler_cfg, universe, benchmark, factors,
        equity_curve_scaler=equity_curve_scaler, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_sweep(
        group=GROUP,
        signals=make_signals(),
        scaler_configs=[{"tag": "none"}],
        rebalance_periods=REBALANCE_PERIODS,
        build_study_fn=build_study_fn,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
