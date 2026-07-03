"""Signal sweep: dist_mr_k3 regime filter development.

Motivation
----------
dist_mr_k3_z10 and dist_mr_k3_z20 are among the strongest IS performers in the pool
(SR 1.0-4.1 every year 2015-2023, no negative years). Out-of-sample (2024-2026) they
get killed: SR -1.4, -1.7, -2.0 for k3z10 and similar for k3z20.

The culprit is not breadth per se — 2023 (breadth=0.39, spy_up=90%) was the best year,
while 2024 (breadth=0.49, spy_up=100%) was the worst. The real differentiator is
*leadership persistence*: when the same stocks keep winning (Mag7 2024, tariff
winners/losers 2025), pairs that diverged stay diverged. When leadership rotates
(2023 had multiple sector rotations), pairs naturally reconverge.

Approaches tested
-----------------
1. Narrow-bull off gate: active when NOT (breadth < threshold AND SPY in uptrend).
   Targets the exact regime that hurts — narrow breadth combined with directional
   bull market (persistent leadership). 2023 survives (breadth low but not paired
   with a stable uptrend at 50d/200d level all year). Implemented as a conditioning
   filter that returns NaN when in narrow-bull.

2. Dispersion gate: active only when cross-sectional return dispersion is above its
   rolling percentile. Low dispersion = stocks moving together = pairs have less
   spread to capture, more noise. Uses existing disp_60_q30 scaler or a filter variant.

3. Vol regime gate (vol_10_60): scale down in calm vol (MR pairs benefit from
   dislocations; calm = pairs diverge but trend rather than revert). Already in pool.

4. Trend scaler (trend_20_100_mr): scale down in SPY uptrend. Blunt but tests whether
   general uptrend exposure is the problem.

5. Combined: narrow-bull off + vol scaler. Belt-and-suspenders: gate off the worst
   regime, then also scale down in calm vol within active periods.

Base signals swept
------------------
- dist_mr_k3_z10 (3-nearest, 10d z-score, faster signal)
- dist_mr_k3_z20 (3-nearest, 20d z-score, slower signal)
- dist_mr_k5_z10 (5-nearest, 10d — more diversified pairs, potentially smoother)
- dist_mr_k5_z20 (5-nearest, 20d)
- dist_mr_k1_z20 (k=1 nearest partner, 20d, reference)
- dist_mr_k1_z60 (k=1 nearest partner, 60d, reference)

Rebalance periods: [5, 10] (r10 is the standard for k3 sleeves)

Usage:
    uv run python examples/signal_sweeps/signal_sweep_dist_mr_k3_regime_filter.py
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
    TRAIN_END,
    TRAIN_START,
    load_data,
    run_sweep,
)

GROUP = "dist-mr-k3-regime-filter"
OUT_DIR = Path(__file__).resolve().parent / "out" / "dist-mr-k3-regime-filter"

REBALANCE_PERIODS = [5, 10]

# ---------------------------------------------------------------------------
# Regime filters
# ---------------------------------------------------------------------------


def filter_narrow_bull_off_40(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    """Gate OFF when breadth < 40% AND SPY is in an uptrend (50d MA > 200d MA).
    This is the narrow-bull regime that kills k3 pairs: persistent directional
    leadership with few rotation opportunities.
    Leaves the signal active during:
      - broad markets (breadth >= 40%), both bull and bear
      - narrow bear markets (breadth < 40% + SPY downtrend) — pairs MR still works
    """
    prices = (1 + cache["returns"]).cumprod()
    ma_200 = prices.rolling(200, min_periods=100).mean()
    breadth = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
    narrow = breadth.lt(0.40).reindex(signal.index).fillna(False)

    bm = cache.get("benchmark")
    if bm is not None:
        spy_price = (1 + bm).cumprod()
        uptrend = (
            spy_price.rolling(50, min_periods=25)
            .mean()
            .gt(spy_price.rolling(200, min_periods=100).mean())
        )
        uptrend = uptrend.reindex(signal.index).fillna(False)
    else:
        uptrend = pd.Series(False, index=signal.index)

    # Active when NOT in narrow bull
    active = ~(narrow & uptrend)
    return signal.where(active, other=np.nan)


filter_narrow_bull_off_40.__name__ = "narrow_bull_off_40"


def filter_narrow_bull_off_50(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    """Gate OFF when breadth < 50% AND SPY is in an uptrend (50d MA > 200d MA).
    Looser breadth threshold than the 40% variant — catches 2024 more aggressively
    (avg breadth 0.49 in 2024) at the cost of potentially gating some good periods.
    """
    prices = (1 + cache["returns"]).cumprod()
    ma_200 = prices.rolling(200, min_periods=100).mean()
    breadth = prices.gt(ma_200).where(ma_200.notna()).mean(axis=1)
    narrow = breadth.lt(0.50).reindex(signal.index).fillna(False)

    bm = cache.get("benchmark")
    if bm is not None:
        spy_price = (1 + bm).cumprod()
        uptrend = (
            spy_price.rolling(50, min_periods=25)
            .mean()
            .gt(spy_price.rolling(200, min_periods=100).mean())
        )
        uptrend = uptrend.reindex(signal.index).fillna(False)
    else:
        uptrend = pd.Series(False, index=signal.index)

    active = ~(narrow & uptrend)
    return signal.where(active, other=np.nan)


filter_narrow_bull_off_50.__name__ = "narrow_bull_off_50"


def filter_low_disp_off(signal: pd.DataFrame, **cache) -> pd.DataFrame:
    """Gate OFF when cross-sectional return dispersion is in the bottom 30th percentile
    of its trailing 252-day distribution. Low dispersion = stocks moving together =
    pairs have less spread and more noise.
    """
    r = cache["returns"]
    disp = r.std(axis=1)
    low_disp = disp.lt(disp.rolling(252, min_periods=126).quantile(0.30))
    low_disp = low_disp.reindex(signal.index).fillna(False)
    return signal.where(~low_disp, other=np.nan)


filter_low_disp_off.__name__ = "low_disp_off_q30"


# ---------------------------------------------------------------------------
# Vol scalers (position-level, not signal-level)
# ---------------------------------------------------------------------------


def make_vol_scaler_down(
    fast: int, slow: int
) -> qs.ScalerFn if hasattr(qs, "ScalerFn") else object:  # type: ignore[valid-type]
    """Scale DOWN to 0.25 in calm vol (fast_vol < slow_vol). MR pairs benefit from
    volatility spikes that create dislocations; scale down when vol is compressed."""

    def vol_scaler(positions: pd.DataFrame, **cache) -> pd.DataFrame:
        bm = cache["benchmark"]
        calm = bm.rolling(fast).std().lt(bm.rolling(slow).std())
        calm = calm.reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(calm, 0.25, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    vol_scaler.__name__ = f"vol_{fast}_{slow}_down"
    return vol_scaler


def make_trend_scaler_mr(fast: int, slow: int) -> object:
    """Scale DOWN to 0.25 when SPY fast MA > slow MA (uptrend). MR-style: reduce
    exposure in trending bull markets where pairs may not revert."""

    def trend_scaler(positions: pd.DataFrame, **cache) -> pd.DataFrame:
        bm = cache["benchmark"]
        equity = (1 + bm).cumprod()
        uptrend = equity.rolling(fast).mean().gt(equity.rolling(slow).mean())
        uptrend = uptrend.reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(uptrend, 0.25, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    trend_scaler.__name__ = f"trend_{fast}_{slow}_mr"
    return trend_scaler


# ---------------------------------------------------------------------------
# Scaler configs
#
# tag: used in the sleeve name and CSV output
# filter_fn: optional conditioning filter applied at signal level (before positions)
# scaler_fns: list of position-level scalers applied after build_long_short
# ---------------------------------------------------------------------------

SCALER_CONFIGS = [
    # --- Baseline ---
    {
        "tag": "none",
        "filter_fn": None,
        "scaler_fns": [],
        "filters": "",
    },
    # --- Approach 1: narrow-bull off gate (signal level) ---
    {
        "tag": "narrow_bull_off_40",
        "filter_fn": filter_narrow_bull_off_40,
        "scaler_fns": [],
        "filters": "narrow_bull_off_40",
    },
    {
        "tag": "narrow_bull_off_50",
        "filter_fn": filter_narrow_bull_off_50,
        "scaler_fns": [],
        "filters": "narrow_bull_off_50",
    },
    # --- Approach 2: dispersion gate (signal level) ---
    {
        "tag": "low_disp_off_q30",
        "filter_fn": filter_low_disp_off,
        "scaler_fns": [],
        "filters": "low_disp_off_q30",
    },
    # --- Approach 3: vol regime scaler (position level) ---
    {
        "tag": "vol_10_60_down",
        "filter_fn": None,
        "scaler_fns": [make_vol_scaler_down(10, 60)],
        "filters": "",
    },
    # --- Approach 4: trend scaler (position level) ---
    {
        "tag": "trend_20_100_mr",
        "filter_fn": None,
        "scaler_fns": [make_trend_scaler_mr(20, 100)],
        "filters": "",
    },
    # --- Approach 5: combined narrow-bull off + vol scaler ---
    {
        "tag": "narrow_bull_off_40__vol_10_60",
        "filter_fn": filter_narrow_bull_off_40,
        "scaler_fns": [make_vol_scaler_down(10, 60)],
        "filters": "narrow_bull_off_40",
    },
    {
        "tag": "narrow_bull_off_50__vol_10_60",
        "filter_fn": filter_narrow_bull_off_50,
        "scaler_fns": [make_vol_scaler_down(10, 60)],
        "filters": "narrow_bull_off_50",
    },
]

# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------

# Partners computed once per universe instance: (universe_id, k) -> {ticker: [peers]}
_partners_cache: dict[tuple, dict[str, list[str]]] = {}


def _get_partners(universe, k: int) -> dict[str, list[str]]:
    key = (id(universe), k)
    if key not in _partners_cache:
        print(f"  Computing distance partners k={k} (train {TRAIN_START}..{TRAIN_END}) ...")
        log_price = universe.log_returns.loc[TRAIN_START:TRAIN_END].cumsum()
        norm = (log_price - log_price.mean()) / log_price.std().clip(lower=1e-8)
        dist = 1 - norm.corr()
        pairs: dict[str, list[str]] = {}
        for ticker in dist.columns:
            pairs[ticker] = dist[ticker].drop(ticker).nsmallest(k).index.tolist()
        _partners_cache[key] = pairs
    return _partners_cache[key]


def _make_dist_mr_signal(k: int, zw: int) -> dict:
    """Return a signal dict for dist_mr_k{k}_z{zw}."""

    def signal_fn(k=k, zw=zw, **cache):
        r = cache["_active_returns"]
        # _universe_ref is injected by build_study_fn so partners can be precomputed
        universe = cache.get("_universe_ref")
        if universe is not None:
            pair_map = _get_partners(universe, k)
        else:
            pair_map = {}

        price = (1 + r).cumprod()
        norm = price / price.bfill().iloc[0].clip(lower=1e-8)
        spread = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
        for ticker in r.columns:
            peers = [p for p in pair_map.get(ticker, []) if p in r.columns]
            if not peers:
                spread[ticker] = np.nan
                continue
            spread[ticker] = norm[ticker] - norm[peers].mean(axis=1)

        mu = spread.rolling(zw, min_periods=zw // 2).mean()
        sigma = spread.rolling(zw, min_periods=zw // 2).std().clip(lower=1e-8)
        return -((spread - mu) / sigma).clip(-2, 2)

    name = f"dist_mr_k{k}_z{zw}"
    signal_fn.__name__ = name
    return {"name": name, "fn": signal_fn, "use_residual": False, "filters": ""}


SIGNALS = [
    _make_dist_mr_signal(3, 10),
    _make_dist_mr_signal(3, 20),
    _make_dist_mr_signal(5, 10),  # k5: 5-nearest-neighbor — more diversified pairs
    _make_dist_mr_signal(5, 20),
    _make_dist_mr_signal(1, 20),  # reference: k1 included for comparison
    _make_dist_mr_signal(1, 60),  # reference: k1 included for comparison
]

# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------


def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    original_fn = entry["fn"]
    filter_fn = scaler_cfg.get("filter_fn")
    scaler_fns = scaler_cfg.get("scaler_fns", [])

    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)

    # Inject universe reference via closure so signal_fn can call _get_partners
    def patched_signal(**cache):
        return original_fn(**cache, _universe_ref=universe)

    patched_signal.__name__ = original_fn.__name__

    builder = Study(
        universe=universe, benchmark=benchmark, factors=factors, verbose=verbose
    ).base_signal(patched_signal)

    if filter_fn is not None:
        builder = builder.add_filter(filter_fn)

    builder = (
        builder.add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=equity_curve_scaler)
    )

    for scaler_fn in scaler_fns:
        builder = builder.scale_risk(fn=scaler_fn)

    return builder.rebalance(every=rebalance).with_transaction_costs(cost_bps=COST_BPS).run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    run_sweep(
        group=GROUP,
        signals=SIGNALS,
        scaler_configs=SCALER_CONFIGS,
        rebalance_periods=REBALANCE_PERIODS,
        build_study_fn=build_study_fn,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
