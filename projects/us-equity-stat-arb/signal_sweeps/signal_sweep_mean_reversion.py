"""Signal sweep: mean reversion (raw returns).

Signals: mr, zscore_rev, cumret_spread
         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_mean_reversion.py
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
from sweep_scalers import SCALER_PRESETS_MR

GROUP = "mean-reversion"
OUT_DIR = Path(__file__).resolve().parent / "out" / "mean-reversion"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:
    signals = []

    for w in [2, 3, 5, 10, 15, 20]:
        window = w

        def mr(window=window, **cache):
            return -cache["_active_returns"].rolling(window).mean()

        mr.__name__ = f"mr_{window}d"
        signals.append({"name": f"mr_{window}d", "use_residual": False, "fn": mr})

    for fast, slow in [(5, 60), (10, 120), (5, 252), (20, 252)]:
        f, s = fast, slow

        def zscore_rev(fast=f, slow=s, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(slow).mean()
            sigma = r.rolling(slow).std().clip(lower=1e-8)
            return -(r.rolling(fast).mean() - mu) / sigma

        zscore_rev.__name__ = f"zscore_rev_{f}_{s}"
        signals.append({"name": f"zscore_rev_{f}_{s}", "use_residual": False, "fn": zscore_rev})

    for short_w, long_w in [(5, 60), (10, 120), (20, 252)]:
        sw, lw = short_w, long_w

        def cumret_spread(sw=sw, lw=lw, **cache):
            r = cache["_active_returns"]
            return -(r.rolling(sw).mean() - r.rolling(lw).mean())

        cumret_spread.__name__ = f"cumret_spread_{sw}_{lw}"
        signals.append(
            {"name": f"cumret_spread_{sw}_{lw}", "use_residual": False, "fn": cumret_spread}
        )

    return signals


# ---------------------------------------------------------------------------
# Scaler configs
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
