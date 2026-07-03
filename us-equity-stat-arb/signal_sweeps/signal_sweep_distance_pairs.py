"""Signal sweep: distance-based pairs mean reversion.

Signals: dist_mr_k{1,3}_z{10,20,60} (6 signals)

Usage:
    uv run python examples/signal_sweeps/signal_sweep_distance_pairs.py
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
    run_sweep,
)

GROUP = "distance-pairs-mr"
OUT_DIR = Path(__file__).resolve().parent / "out" / "distance-pairs-mr"

# Module-level cache: universe id -> {signal_name -> signal_dict}
_cached_signals: dict[int, dict[str, dict]] = {}


# ---------------------------------------------------------------------------
# Signals (computed from universe data)
# ---------------------------------------------------------------------------


def make_signals(universe) -> list[dict]:
    """Compute distance-based pair partners from universe history and return signal list."""
    log_price = universe.log_returns.cumsum()
    norm_price = (log_price - log_price.mean()) / log_price.std().clip(lower=1e-8)
    dist = 1 - norm_price.corr()

    signals = []
    for k in [1, 2, 3, 5]:
        for zw in [10, 20, 60]:
            partners: dict[str, list[str]] = {}
            for ticker in dist.columns:
                row = dist[ticker].drop(ticker).nsmallest(k)
                partners[ticker] = row.index.tolist()

            def _dist_mr(partners=partners, zw=zw, **cache):
                r = cache["_active_returns"]
                price = (1 + r).cumprod()
                norm = price / price.bfill().iloc[0].clip(lower=1e-8)
                spread = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
                for ticker in r.columns:
                    p = [x for x in partners.get(ticker, []) if x in r.columns]
                    if not p:
                        spread[ticker] = np.nan
                        continue
                    spread[ticker] = norm[ticker] - norm[p].mean(axis=1)
                mu = spread.rolling(zw).mean()
                sigma = spread.rolling(zw).std().clip(lower=1e-8)
                return -((spread - mu) / sigma).clip(-2, 2)

            name = f"dist_mr_k{k}_z{zw}"
            _dist_mr.__name__ = name
            signals.append({"name": name, "fn": _dist_mr, "use_residual": False})

    return signals


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------


def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    # Compute signals from universe on first call, then cache by universe id
    key = id(universe)
    if key not in _cached_signals:
        _cached_signals[key] = {s["name"]: s for s in make_signals(universe)}

    actual_entry = _cached_signals[key][entry["name"]]
    fn = actual_entry["fn"]
    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)

    return (
        Study(universe=universe, benchmark=benchmark, factors=factors, verbose=verbose)
        .base_signal(fn)
        .add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=equity_curve_scaler)
        .rebalance(every=rebalance)
        .with_transaction_costs(cost_bps=COST_BPS)
        .run()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Placeholder signal list with correct names; actual fns are baked in build_study_fn
    placeholder_signals = [
        {"name": f"dist_mr_k{k}_z{zw}", "fn": lambda **cache: None, "use_residual": False}
        for k in [1, 3]
        for zw in [10, 20, 60]
    ]

    run_sweep(
        group=GROUP,
        signals=placeholder_signals,
        scaler_configs=[{"tag": "none"}],
        rebalance_periods=REBALANCE_PERIODS,
        build_study_fn=build_study_fn,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
