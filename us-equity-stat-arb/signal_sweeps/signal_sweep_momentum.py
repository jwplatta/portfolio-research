"""Signal sweep: momentum signals.

Signals: mom, resid_mom, skip1_mom, high_52w
         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_momentum.py
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
from sweep_scalers import SCALER_PRESETS_MOM

GROUP = "momentum"
OUT_DIR = Path(__file__).resolve().parent / "out" / "momentum"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:
    signals = []

    for w in [20, 40, 60, 120, 252]:
        window = w

        def mom(window=window, **cache):
            return cache["_active_returns"].rolling(window).mean()

        mom.__name__ = f"mom_{w}d"
        signals.append({"name": f"mom_{w}d", "use_residual": False, "fn": mom})

    for w in [20, 40, 60, 120]:
        window = w

        def resid_mom(window=window, **cache):
            return cache["residual_returns"].rolling(window).mean()

        resid_mom.__name__ = f"resid_mom_{w}d"
        signals.append({"name": f"resid_mom_{w}d", "use_residual": True, "fn": resid_mom})

    for w in [60, 120, 252]:
        window = w

        def skip1_mom(window=window, **cache):
            return cache["_active_returns"].shift(5).rolling(window).mean()

        skip1_mom.__name__ = f"skip1_mom_{w}d"
        signals.append({"name": f"skip1_mom_{w}d", "use_residual": False, "fn": skip1_mom})

    def high_52w(**cache):
        prices = cache.get("close", (1 + cache["_active_returns"]).cumprod())
        return prices / prices.rolling(252).max().clip(lower=1e-8)

    high_52w.__name__ = "high_52w"
    signals.append({"name": "high_52w", "use_residual": False, "fn": high_52w})

    return signals


# ---------------------------------------------------------------------------
# Scaler configs
# ---------------------------------------------------------------------------

def make_scaler_configs() -> list[dict]:
    return SCALER_PRESETS_MOM


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
