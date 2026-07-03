"""Full-period signal sweep: low-beta / defensive tilt signals.

Signal concept: rank stocks by their rolling market beta — long low-beta
(defensive) stocks, short high-beta (aggressive) stocks. In persistent bear
markets with narrow breadth (2022), low-beta stocks fall less and the
long-short spread widens. Variants include raw beta, idiosyncratic vol
adjusted beta, and a beta-momentum blend.

Signals:
  - rolling_beta_{window}d: OLS beta vs SPY over trailing window
  - beta_resid_vol_{window}d: beta / idiosyncratic vol (risk-adjusted)
  - beta_vs_sector_{window}d: stock beta relative to its sector ETF beta

Scaler configs: MR-style trend (scale down in uptrend — low-beta works
in downtrends, not uptrends), breadth, vol.

Usage:
    uv run python examples/signal_sweeps/signal_sweep_low_beta.py
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
    TRAIN_START,
    COST_BPS,
    N_LONG,
    N_SHORT,
    REBALANCE_PERIODS,
    load_data,
    run_sweep,
)

GROUP = "low-beta"
OUT_DIR = Path(__file__).resolve().parent / "out" / "low-beta"

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


def _rolling_beta(stock_returns: pd.DataFrame, market_returns: pd.Series, window: int) -> pd.DataFrame:
    """Compute rolling OLS beta of each stock vs market over trailing window."""
    mkt = market_returns.reindex(stock_returns.index).fillna(0.0)
    mkt_var = mkt.rolling(window).var().clip(lower=1e-10)
    # Covariance via rolling mean of product minus product of means
    betas = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)
    for col in stock_returns.columns:
        s = stock_returns[col].fillna(0.0)
        cov = (s * mkt).rolling(window).mean() - s.rolling(window).mean() * mkt.rolling(window).mean()
        betas[col] = cov / mkt_var
    return betas


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals(sector_etf_map: dict[str, str]) -> list[dict]:
    signals = []

    # --- Raw rolling beta vs SPY ---
    # Long low-beta (< 1), short high-beta (> 1).
    # Negate so that low-beta → positive signal (long) as in standard ranking.
    for w in [60, 120, 252]:
        window = w

        def _beta(window=window, **cache):
            r = cache["_active_returns"]
            bm = cache.get("benchmark")
            if bm is None:
                return None
            betas = _rolling_beta(r, bm, window)
            return -betas  # negate: low beta → long

        _beta.__name__ = f"rolling_beta_{w}d"
        signals.append({"name": f"rolling_beta_{w}d", "fn": _beta})

    # --- Beta adjusted by idiosyncratic vol ---
    # beta / idio_vol: penalizes stocks that are both high-beta AND volatile.
    # Rewards stocks with low systematic risk relative to their total risk.
    for w in [120, 252]:
        window = w

        def _beta_resid_vol(window=window, **cache):
            r = cache["_active_returns"]
            bm = cache.get("benchmark")
            if bm is None:
                return None
            betas = _rolling_beta(r, bm, window)
            mkt = bm.reindex(r.index).fillna(0.0)
            # Idiosyncratic vol: std of (stock_return - beta * mkt_return)
            idio = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
            for col in r.columns:
                resid = r[col].fillna(0.0) - betas[col] * mkt
                idio[col] = resid.rolling(window).std().clip(lower=1e-8)
            # Low beta/idio_vol → defensive, low total risk relative to market
            return -(betas / idio)

        _beta_resid_vol.__name__ = f"beta_resid_vol_{w}d"
        signals.append({"name": f"beta_resid_vol_{w}d", "fn": _beta_resid_vol})

    # --- Beta relative to sector ETF ---
    # Computes each stock's beta vs its own sector ETF rather than vs SPY.
    # Captures within-sector defensiveness — within energy, which stocks
    # are the most defensive relative to XLE?
    for w in [120, 252]:
        window = w

        def _beta_vs_sector(window=window, sector_etf_map=sector_etf_map, **cache):
            r = cache["_active_returns"]
            factor_returns = cache.get("factor_returns")
            if factor_returns is None:
                return None
            betas = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
            for col in r.columns:
                etf = sector_etf_map.get(col, "SPY")
                sec_ret = (
                    factor_returns[etf].reindex(r.index).fillna(0.0)
                    if etf in factor_returns.columns
                    else factor_returns["SPY"].reindex(r.index).fillna(0.0)
                )
                sec_var = sec_ret.rolling(window).var().clip(lower=1e-10)
                s = r[col].fillna(0.0)
                cov = (s * sec_ret).rolling(window).mean() - s.rolling(window).mean() * sec_ret.rolling(window).mean()
                betas[col] = cov / sec_var
            return -betas  # negate: low beta within sector → long

        _beta_vs_sector.__name__ = f"beta_vs_sector_{w}d"
        signals.append({"name": f"beta_vs_sector_{w}d", "fn": _beta_vs_sector})

    # --- Beta momentum: change in beta (rising beta → short) ---
    # Stocks whose beta is rising are becoming more risky; short them.
    # Stocks whose beta is falling are becoming more defensive; long them.
    for fast, slow in [(20, 120), (60, 252)]:
        f, s = fast, slow

        def _beta_momentum(f=f, s=s, **cache):
            r = cache["_active_returns"]
            bm = cache.get("benchmark")
            if bm is None:
                return None
            beta_fast = _rolling_beta(r, bm, f)
            beta_slow = _rolling_beta(r, bm, s)
            # Negative: rising beta (fast > slow) → short
            return -(beta_fast - beta_slow)

        _beta_momentum.__name__ = f"beta_momentum_{f}_{s}d"
        signals.append({"name": f"beta_momentum_{f}_{s}d", "fn": _beta_momentum})

    return signals


# ---------------------------------------------------------------------------
# Scaler configs
# ---------------------------------------------------------------------------

def make_scaler_configs() -> list[dict]:
    return [
        {"tag": "none",         "trend": None,                             "vol": None,                    "breadth": None},
        # MR-style trend: scale DOWN in uptrend (low-beta works in downtrends)
        {"tag": "trend_20_100", "trend": {"fast": 20, "slow": 100, "mr": True}, "vol": None,              "breadth": None},
        {"tag": "trend_50_200", "trend": {"fast": 50, "slow": 200, "mr": True}, "vol": None,              "breadth": None},
        {"tag": "trend_20_100_h", "trend": {"fast": 20, "slow": 100, "mr": True, "scale_down": 0.50}, "vol": None, "breadth": None},
        # Vol scalers: scale UP in high-vol (low-beta outperforms most in stress)
        {"tag": "vol_10_60",    "trend": None, "vol": {"fast": 10, "slow": 60,  "invert": True},  "breadth": None},
        {"tag": "vol_20_60",    "trend": None, "vol": {"fast": 20, "slow": 60,  "invert": True},  "breadth": None},
        {"tag": "vol_20_100",   "trend": None, "vol": {"fast": 20, "slow": 100, "invert": True},  "breadth": None},
        # Breadth: scale UP when breadth is weak (narrow breadth = defensive environment)
        {"tag": "breadth_40",   "trend": None, "vol": None, "breadth": {"threshold": 0.40, "invert": True}},
        {"tag": "breadth_50",   "trend": None, "vol": None, "breadth": {"threshold": 0.50, "invert": True}},
    ]


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------

def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    fn = entry["fn"]
    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)

    builder = (
        Study(universe=universe, benchmark=benchmark, factors=factors, verbose=verbose)
        .base_signal(fn)
        .add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=equity_curve_scaler)
    )

    trend_cfg = scaler_cfg.get("trend")
    if trend_cfg is not None:
        fast = trend_cfg["fast"]
        slow = trend_cfg["slow"]
        mr_style = trend_cfg.get("mr", True)
        sd = trend_cfg.get("scale_down", 0.25)

        def _trend(positions, fast=fast, slow=slow, mr_style=mr_style, sd=sd, **cache):
            bm = cache.get("benchmark")
            if bm is None:
                return positions
            spy_price = (1 + bm).cumprod()
            in_uptrend = spy_price.rolling(fast).mean() > spy_price.rolling(slow).mean()
            if mr_style:
                # MR-style: scale down IN uptrend (low-beta works in downtrends)
                scale_vals = np.where(in_uptrend.reindex(positions.index).fillna(False), sd, 1.0)
            else:
                scale_vals = np.where(in_uptrend.reindex(positions.index).fillna(False), 1.0, sd)
            return positions.mul(pd.Series(scale_vals, index=positions.index).shift(1), axis=0)

        _trend.__name__ = f"trend_{fast}_{slow}"
        builder = builder.scale_risk(fn=_trend)

    vol_cfg = scaler_cfg.get("vol")
    if vol_cfg is not None:
        fast = vol_cfg["fast"]
        slow = vol_cfg["slow"]
        invert = vol_cfg.get("invert", False)  # invert=True: scale UP in high-vol

        def _vol(positions, fast=fast, slow=slow, invert=invert, **cache):
            bm = cache.get("benchmark")
            if bm is None:
                return positions
            fv = bm.rolling(fast).std()
            sv = bm.rolling(slow).std()
            in_spike = (fv > sv).reindex(positions.index).fillna(False)
            if invert:
                # Scale UP in high-vol (low-beta benefits from stress)
                scale_vals = np.where(in_spike, 1.0, 0.25)
            else:
                scale_vals = np.where(in_spike, 0.25, 1.0)
            return positions.mul(pd.Series(scale_vals, index=positions.index).shift(1), axis=0)

        _vol.__name__ = f"vol_{fast}_{slow}"
        builder = builder.scale_risk(fn=_vol)

    breadth_cfg = scaler_cfg.get("breadth")
    if breadth_cfg is not None:
        thresh = breadth_cfg["threshold"]
        invert = breadth_cfg.get("invert", False)

        def _breadth(positions, thresh=thresh, invert=invert, **cache):
            returns = cache.get("returns")
            if returns is None:
                return positions
            prices = (1 + returns).cumprod()
            pct_above = (prices > prices.rolling(200).mean()).mean(axis=1)
            weak_breadth = pct_above.reindex(positions.index).fillna(0) < thresh
            if invert:
                # Scale UP when breadth is weak (low-beta defensive environment)
                scale_vals = np.where(weak_breadth, 1.0, 0.25)
            else:
                scale_vals = np.where(weak_breadth, 0.25, 1.0)
            return positions.mul(pd.Series(scale_vals, index=positions.index).shift(1), axis=0)

        _breadth.__name__ = f"breadth_{thresh}"
        builder = builder.scale_risk(fn=_breadth)

    return builder.rebalance(every=rebalance).run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    universe, _, _ = load_data()
    sector_map = qs.get_sector_map(list(universe.returns.columns))
    sector_etf_map = {t: GICS_TO_ETF.get(s, "SPY") for t, s in sector_map.items()}

    run_sweep(
        group=GROUP,
        signals=make_signals(sector_etf_map),
        scaler_configs=make_scaler_configs(),
        rebalance_periods=REBALANCE_PERIODS,
        build_study_fn=build_study_fn,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
