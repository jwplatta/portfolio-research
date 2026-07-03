"""Signal sweep: idiosyncratic volatility expansion.

Signals: ivol_expansion_5d, ivol_expansion_10d

Usage:
    uv run python examples/signal_sweeps/signal_sweep_ivol_expansion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from sweep_scalers import SCALER_PRESETS_MR

GROUP = "ivol-expansion"
OUT_DIR = Path(__file__).resolve().parent / "out" / "ivol-expansion"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:
    signals = []

    for w in [5, 10]:
        fast = w

        def ivol_expansion(fast=fast, **cache):
            r = cache.get("residual_returns", cache["_active_returns"])
            return -(r.rolling(fast).std() / r.rolling(60).std().clip(lower=1e-8))

        ivol_expansion.__name__ = f"ivol_expansion_{w}d"
        signals.append({"name": f"ivol_expansion_{w}d", "use_residual": True, "fn": ivol_expansion})

    return signals


# ---------------------------------------------------------------------------
# Scaler configs — MR-style
# ---------------------------------------------------------------------------

def make_scaler_configs() -> list[dict]:
    return SCALER_PRESETS_MR


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------

def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)
    return build_study_generic(
        entry, rebalance, scaler_cfg, universe, benchmark, factors,
        equity_curve_scaler=equity_curve_scaler, verbose=verbose,
    )


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
