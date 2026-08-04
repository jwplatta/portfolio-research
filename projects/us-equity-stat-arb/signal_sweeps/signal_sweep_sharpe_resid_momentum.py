"""Signal sweep: Sharpe-scaled residual momentum signals.

Signals: sharpe_resid_mom with window x skip combinations
         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_sharpe_resid_momentum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import qstudy as qs

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
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

GROUP = "sharpe-resid-momentum"
OUT_DIR = Path(__file__).resolve().parent / "out" / "sharpe-resid-momentum"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:
    signals = []

    for w in [60, 120, 252]:
        for s in [0, 5]:
            window, skip = w, s

            def sharpe_resid_mom(window=window, skip=skip, **cache):
                r = cache["residual_returns"]
                mu = r.shift(skip).rolling(window).mean()
                sigma = r.shift(skip).rolling(window).std().clip(lower=1e-8)
                return mu / sigma

            if skip == 0:
                name = f"sharpe_resid_mom_{w}d"
            else:
                name = f"sharpe_resid_mom_{w}d_skip{s}"

            sharpe_resid_mom.__name__ = name
            signals.append({"name": name, "use_residual": True, "fn": sharpe_resid_mom})

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
    # All signals use factor-model residualization; override since entries don't carry that flag.
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
