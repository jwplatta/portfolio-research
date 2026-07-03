"""Signal sweep: residual momentum signals.

Signals: resid_mom, skip1_resid_mom (factor-model residualized returns)
         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_resid_momentum.py
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

GROUP = "resid-momentum"
OUT_DIR = Path(__file__).resolve().parent / "out" / "resid-momentum"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:
    signals = []

    for w in [20, 40, 60, 120, 252]:
        window = w

        def resid_mom(window=window, **cache):
            return cache["residual_returns"].rolling(window).mean()

        resid_mom.__name__ = f"resid_mom_{w}d"
        signals.append({"name": f"resid_mom_{w}d", "use_residual": True, "fn": resid_mom})

    for w in [60, 120, 252]:
        window = w

        def skip1_resid_mom(window=window, **cache):
            return cache["residual_returns"].shift(21).rolling(window).mean()

        skip1_resid_mom.__name__ = f"skip1_resid_mom_{w}d"
        signals.append({"name": f"skip1_resid_mom_{w}d", "use_residual": True, "fn": skip1_resid_mom})

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
    # All signals in this sweep use factor-model residualization.
    # Override use_factor_model since signal entries don't carry that flag.
    entry_with_fm = {**entry, "use_factor_model": True}
    return build_study_generic(
        entry_with_fm, rebalance, scaler_cfg, universe, benchmark, factors,
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
