"""GGR Pairs Trading Backtest
=================================
Implements the distance-based pairs trading strategy from:
  Gatev, Goetzmann & Rouwenhorst (2006) — "Pairs Trading: Performance of a
  Relative-Value Arbitrage Rule" (Review of Financial Studies 19(3))

Strategy mechanics
------------------
  Formation period  : 12-month rolling window to rank pairs by SSD of normalized
                      cumulative return paths.  Top-20 pairs selected.
  Trading signal    : For each stock, track spread vs its matched partner.
                      Signal = -z_score(spread, window=formation_window).
                      Mean reversion: buy the underperformer, short the outperformer.
  Liquidity filter  : Top-300 S&P 500 stocks by rolling 60-day avg dollar volume.
  Transaction costs : 10 bps one-way.

Backtest windows
----------------
  In-sample  : 2015-01-01 to 2023-12-31  (with 1-year warmup from 2014-01-01)
  Out-of-sample : 2024-01-01 to 2026-05-31

Usage
-----
    uv run python pairs-trading/ggr_pairs.py
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import qstudy as qs
from qstudy import Study

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_START = "2014-01-01"  # 1 year before in-sample start for indicator warmup
IS_START = "2015-01-01"  # in-sample start (backtest)
IS_END = "2023-12-31"  # in-sample end
OOS_START = "2024-01-01"  # out-of-sample start
OOS_END = "2026-05-31"  # out-of-sample end
OOS_WARMUP_START = "2023-01-01"  # 1 year of warmup before OOS (for signal initialization)

BENCHMARK_TICKER = "SPY"
COST_BPS = 5.0

# GGR parameters
FORMATION_WINDOW = 252  # trading days for pair formation (~12 months)
ZSCORE_WINDOW = 60  # rolling window for spread z-score (3 months)
TOP_N_PAIRS = 20  # top pairs per stock to consider
LIQUIDITY_TOP_N = 300  # liquid universe size

OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------


@cache
def load_is_data() -> tuple:
    """Load in-sample + warmup data (2014-2023)."""
    print(f"Loading in-sample data ({WARMUP_START} to {IS_END}) ...")
    universe = qs.download(index_code="SP500", start=WARMUP_START, end=IS_END)
    benchmark = qs.download([BENCHMARK_TICKER], start=WARMUP_START, end=IS_END)
    print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")
    return universe, benchmark


@cache
def load_oos_data() -> tuple:
    """Load out-of-sample data (with 1-year warmup from 2023-01-01)."""
    print(f"Loading out-of-sample data ({OOS_WARMUP_START} to {OOS_END}) ...")
    universe = qs.download(index_code="SP500", start=OOS_WARMUP_START, end=OOS_END)
    benchmark = qs.download([BENCHMARK_TICKER], start=OOS_WARMUP_START, end=OOS_END)
    print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")
    return universe, benchmark


# ---------------------------------------------------------------------------
# GGR signal
# ---------------------------------------------------------------------------


def make_ggr_pairs_signal(
    formation_window: int = FORMATION_WINDOW, zscore_window: int = ZSCORE_WINDOW
):
    """Return a GGR distance-based pairs signal function.

    Formation: for each date, look back `formation_window` days and find each
    stock's nearest partner by minimum SSD of normalized cumulative returns.

    Signal: negative z-score of the spread (stock price - partner price),
    normalized over `zscore_window` days. Mean-reversion: buy stocks that have
    fallen relative to their partner.

    This is a rolling-window implementation — pairs are re-formed every day
    using the most recent `formation_window` days, which is the continuous
    analogue of GGR's 6-month non-overlapping trading periods.
    """

    def ggr_signal(**cache) -> pd.DataFrame:
        returns = cache["_active_returns"]

        # Cumulative return index (rebased to 1.0 at each formation window start)
        # Use log-price path for distance computation (matches GGR normalization)
        log_price = np.log1p(returns).cumsum()

        # Normalize each stock's price path: (P - mean) / std over the full history
        # This is the GGR "normalization" step — makes price levels comparable
        norm_price = (log_price - log_price.rolling(formation_window).mean()) / (
            log_price.rolling(formation_window).std().clip(lower=1e-8)
        )

        # Find pairs using the most recent formation window (rolling SSD)
        # For efficiency, compute SSD via: SSD(i,j) = sum((P_i - P_j)^2)
        # This equals var(P_i - P_j) * (n-1), so minimum SSD ~ minimum spread variance
        # We compute the correlation of normalized prices as a proxy (high corr = low SSD)
        signal = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for i, date in enumerate(returns.index):
            if i < formation_window:
                continue

            window_slice = norm_price.iloc[i - formation_window : i]
            # Only use tickers with full data in this window
            valid = window_slice.columns[window_slice.notna().all()]
            if len(valid) < 2:
                continue

            window_data = window_slice[valid]

            # --- Pair formation: find nearest partner by SSD ---
            # SSD(i, j) = sum_t (P_i_t - P_j_t)^2
            # Equivalent to squared L2 norm; compute via pairwise variance of differences.
            # Use vectorized approach: var(P_i - P_j) * (n-1)
            # We approximate with correlation (monotone transform of SSD for standardized paths)
            corr_arr = window_data.corr().values.copy()
            np.fill_diagonal(corr_arr, np.nan)
            corr_matrix = pd.DataFrame(corr_arr, index=valid, columns=valid)

            # For each stock, find its best partner (highest correlation = lowest SSD)
            best_partner = corr_matrix.idxmax(axis=1)

            # --- Spread computation for today ---
            # Spread = normalized_price[stock] - normalized_price[partner]
            today_norm = norm_price.loc[date]
            spread_today = today_norm - today_norm[best_partner.values].values

            # --- Rolling spread z-score for the signal ---
            # Use last zscore_window days of spreads for this stock-partner pair
            # Build spread series over the lookback window
            for ticker in valid:
                partner = best_partner.get(ticker)
                if partner is None or pd.isna(partner):
                    continue
                if partner not in norm_price.columns:
                    continue

                spread_history = (
                    norm_price[ticker].iloc[max(0, i - zscore_window) : i + 1]
                    - norm_price[partner].iloc[max(0, i - zscore_window) : i + 1]
                )
                mu = spread_history.mean()
                sigma = spread_history.std()
                if sigma < 1e-8 or pd.isna(sigma):
                    continue

                z = (spread_today[ticker] - mu) / sigma
                # Mean reversion: negative z-score (buy the laggard, short the leader)
                signal.loc[date, ticker] = -float(np.clip(z, -3.0, 3.0))

        return signal

    ggr_signal.__name__ = f"ggr_pairs_f{formation_window}_z{zscore_window}"
    return ggr_signal


# ---------------------------------------------------------------------------
# Vectorized GGR signal (faster implementation)
# ---------------------------------------------------------------------------


def make_ggr_pairs_signal_fast(
    formation_window: int = FORMATION_WINDOW,
    zscore_window: int = ZSCORE_WINDOW,
):
    """Vectorized GGR pairs signal.

    Avoids per-row Python loops by:
    1. Computing the rolling normalized price matrix once.
    2. Using rolling correlation (or rolling SSD via matrix ops) at checkpoints.
    3. Interpolating partner assignments between checkpoints.

    Pair assignments are updated monthly (every 21 trading days) rather than
    daily, which is closer to GGR's original non-overlapping periods and much
    faster to compute.
    """
    recompute_every = 21  # re-form pairs monthly

    def ggr_signal_fast(**cache) -> pd.DataFrame:
        returns = cache["_active_returns"]
        n_dates = len(returns)

        log_price = np.log1p(returns.fillna(0)).cumsum()
        norm_price = (log_price - log_price.rolling(formation_window).mean()) / (
            log_price.rolling(formation_window).std().clip(lower=1e-8)
        )

        # Identify checkpoint dates for re-forming pairs
        checkpoints = list(range(formation_window, n_dates, recompute_every))
        if not checkpoints:
            return pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        # Map: checkpoint_idx -> {ticker: partner}
        pair_maps: dict[int, pd.Series] = {}
        for cp in checkpoints:
            window_data = norm_price.iloc[cp - formation_window : cp]
            valid_cols = window_data.columns[window_data.notna().all()]
            if len(valid_cols) < 2:
                continue
            corr_arr = window_data[valid_cols].corr().values.copy()
            np.fill_diagonal(corr_arr, np.nan)
            corr = pd.DataFrame(corr_arr, index=valid_cols, columns=valid_cols)
            # idxmax raises on all-NaN rows; skip tickers with no valid correlations
            has_any = corr.notna().any(axis=1)
            pair_maps[cp] = corr.loc[has_any].idxmax(axis=1)

        if not pair_maps:
            return pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        # Build spread DataFrame using rolling pair assignments
        spread = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        sorted_cps = sorted(pair_maps)

        for seg_idx, cp in enumerate(sorted_cps):
            partners = pair_maps[cp]
            # This pair assignment is valid from cp to the next checkpoint
            next_cp = sorted_cps[seg_idx + 1] if seg_idx + 1 < len(sorted_cps) else n_dates
            date_slice = returns.index[cp:next_cp]

            for ticker in partners.index:
                partner = partners[ticker]
                if pd.isna(partner) or partner not in norm_price.columns:
                    continue
                spread.loc[date_slice, ticker] = (
                    norm_price[ticker].loc[date_slice].values
                    - norm_price[partner].loc[date_slice].values
                )

        # Z-score the spread and negate for mean reversion
        mu = spread.rolling(zscore_window, min_periods=max(5, zscore_window // 4)).mean()
        sigma = (
            spread.rolling(zscore_window, min_periods=max(5, zscore_window // 4))
            .std()
            .clip(lower=1e-8)
        )
        z = (spread - mu) / sigma
        return -z.clip(-3.0, 3.0)

    ggr_signal_fast.__name__ = f"ggr_pairs_fast_f{formation_window}_z{zscore_window}"
    return ggr_signal_fast


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------


def build_study(
    universe,
    benchmark,
    *,
    label: str = "GGR Pairs",
    rebalance_every: int = 21,
) -> Study:
    """Build and run the GGR pairs trading study.

    GGR hold pairs for the full 6-month trading period (low turnover by design).
    We approximate this with a monthly rebalance (every=21), which keeps turnover
    in line with the paper's intent.  Daily rebalancing with a signal that continuously
    re-ranks would generate excessive transaction costs.
    """
    signal_fn = make_ggr_pairs_signal_fast(
        formation_window=FORMATION_WINDOW,
        zscore_window=ZSCORE_WINDOW,
    )

    study = (
        Study(universe=universe, benchmark=benchmark, name=label)
        .base_signal(signal_fn)
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=60))
        .winsorize(lower=0.05, upper=0.95)
        .build_long_short(n_long=25, n_short=25)
        .fully_invest()
        .rebalance(every=rebalance_every)
        .with_transaction_costs(cost_bps=COST_BPS)
        .run()
    )
    return study


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_is() -> dict:
    """Run in-sample backtest (2015–2023)."""
    universe, benchmark = load_is_data()

    print(f"\n=== In-Sample Backtest: {IS_START} to {IS_END} ===")
    study = build_study(universe, benchmark, label="GGR Pairs (IS)")
    metrics = study.metrics_dict()
    print("\nIn-Sample Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
    return metrics


def run_oos() -> dict:
    """Run out-of-sample evaluation (2024–May 2026)."""
    universe, benchmark = load_oos_data()

    print(f"\n=== Out-of-Sample Evaluation: {OOS_START} to {OOS_END} ===")
    study = build_study(universe, benchmark, label="GGR Pairs (OOS)")
    metrics = study.metrics_dict()
    print("\nOut-of-Sample Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
    return metrics


def save_results(is_metrics: dict, oos_metrics: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "strategy": "GGR Pairs Trading",
        "paper": "Gatev, Goetzmann & Rouwenhorst (2006)",
        "parameters": {
            "formation_window_days": FORMATION_WINDOW,
            "zscore_window_days": ZSCORE_WINDOW,
            "liquidity_top_n": LIQUIDITY_TOP_N,
            "n_long": 25,
            "n_short": 25,
            "cost_bps": COST_BPS,
        },
        "in_sample": {"period": f"{IS_START} to {IS_END}", "metrics": is_metrics},
        "out_of_sample": {"period": f"{OOS_START} to {OOS_END}", "metrics": oos_metrics},
    }
    out_path = OUT_DIR / "ggr_pairs_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    is_metrics = run_is()
    oos_metrics = run_oos()
    save_results(is_metrics, oos_metrics)
