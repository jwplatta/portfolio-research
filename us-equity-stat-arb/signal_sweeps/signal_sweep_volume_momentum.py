"""Signal sweep: volume-weighted momentum.

Signals: vol_weighted_mom_60d, vol_weighted_mom_120d

Usage:
    uv run python examples/signal_sweeps/signal_sweep_volume_momentum.py
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
from sweep_scalers import apply_scalers

GROUP = "volume-momentum"
OUT_DIR = Path(__file__).resolve().parent / "out" / "volume-momentum"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:
    signals = []

    for w in [60, 120]:
        window = w

        def vol_weighted_mom(window=window, **cache):
            r = cache["_active_returns"]
            volume = cache["volume"].reindex(columns=r.columns)
            weight = volume / volume.rolling(window).mean().clip(lower=1e-8)
            return (r * weight).rolling(window).mean()

        vol_weighted_mom.__name__ = f"vol_weighted_mom_{w}d"
        signals.append({"name": f"vol_weighted_mom_{w}d", "use_residual": False, "fn": vol_weighted_mom})

    return signals


# ---------------------------------------------------------------------------
# Scaler configs
# ---------------------------------------------------------------------------

def make_scaler_configs() -> list[dict]:
    return [
        {"tag": "none", "trend": None, "vol_shock": None, "breadth": None, "crash": None},
        # trend (momentum polarity: full in uptrend)
        {"tag": "trend_20_100", "trend": {"fast": 20, "slow": 100}, "vol_shock": None, "breadth": None, "crash": None},
        {"tag": "trend_50_200", "trend": {"fast": 50, "slow": 200}, "vol_shock": None, "breadth": None, "crash": None},
        {"tag": "trend_20_200", "trend": {"fast": 20, "slow": 200}, "vol_shock": None, "breadth": None, "crash": None},
        # vol_shock
        {"tag": "vol_shock_10_60", "trend": None, "vol_shock": {"fast": 10, "slow": 60}, "breadth": None, "crash": None},
        {"tag": "vol_shock_20_100", "trend": None, "vol_shock": {"fast": 20, "slow": 100}, "breadth": None, "crash": None},
        {"tag": "vol_shock_20_60", "trend": None, "vol_shock": {"fast": 20, "slow": 60}, "breadth": None, "crash": None},
        # breadth
        {"tag": "breadth_20_q25", "trend": None, "vol_shock": None, "breadth": {"window": 20, "low_q": 0.25}, "crash": None},
        {"tag": "breadth_60_q25", "trend": None, "vol_shock": None, "breadth": {"window": 60, "low_q": 0.25}, "crash": None},
        {"tag": "breadth_20_q30", "trend": None, "vol_shock": None, "breadth": {"window": 20, "low_q": 0.30}, "crash": None},
        # crash
        {"tag": "crash_20_m10", "trend": None, "vol_shock": None, "breadth": None, "crash": {"window": 20, "threshold": -0.10}},
        {"tag": "crash_60_m10", "trend": None, "vol_shock": None, "breadth": None, "crash": {"window": 60, "threshold": -0.10}},
        {"tag": "crash_20_m15", "trend": None, "vol_shock": None, "breadth": None, "crash": {"window": 20, "threshold": -0.15}},
    ]


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------

def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    fn = entry["fn"]
    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)

    # Note: apply_scalers() handles trend, vol_shock, and breadth (low_q variant).
    # The crash scaler here uses drawdown semantics (scale down when SPY is IN a crash),
    # which differs from make_crash_scaler (rebound after crash). Kept inline.
    builder = (
        qs.Study(universe=universe, benchmark=benchmark, factors=factors, verbose=verbose)
        .base_signal(fn)
        .add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=equity_curve_scaler)
    )

    builder = apply_scalers(builder, {k: v for k, v in scaler_cfg.items() if k != "crash"})

    crash_cfg = scaler_cfg.get("crash")
    if crash_cfg is not None:
        window, threshold = crash_cfg["window"], crash_cfg["threshold"]

        def _crash(positions, window=window, threshold=threshold, **cache):
            bm = cache.get("benchmark")
            if bm is None:
                return positions
            drawdown = (1 + bm).cumprod()
            drawdown = drawdown / drawdown.rolling(window).max() - 1
            in_crash = (drawdown < threshold).reindex(positions.index).fillna(False)
            scale = pd.Series(np.where(in_crash, 0.25, 1.0), index=positions.index)
            return positions.mul(scale.shift(1), axis=0)

        _crash.__name__ = f"crash_{window}_{threshold}"
        builder = builder.scale_risk(fn=_crash)

    return builder.rebalance(every=rebalance).with_transaction_costs(cost_bps=COST_BPS).run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_sweep(
        group=GROUP,
        signals=make_signals(),
        scaler_configs=make_scaler_configs(),
        rebalance_periods=REBALANCE_PERIODS,
        build_study_fn=build_study_fn,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
