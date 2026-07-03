"""Signal sweep: residual mean reversion (factor-stripped returns).

Signals: factor_model_resid_mr, ivol, resid_zscore, etf_factor_resid_mr,
         resid_zscore_w15 (winsorized)
         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_resid_mean_reversion.py
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

GROUP = "resid-mean-reversion"
OUT_DIR = Path(__file__).resolve().parent / "out" / "resid-mean-reversion"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:
    signals = []

    for w in [2, 3, 5, 10, 15, 20]:
        window = w

        def factor_model_resid_mr(window=window, **cache):
            return -cache["residual_returns"].rolling(window).mean()

        factor_model_resid_mr.__name__ = f"factor_model_resid_mr_{window}d"
        signals.append({
            "name": f"factor_model_resid_mr_{window}d",
            "use_residual": True,
            "use_factor_model": True,
            "fn": factor_model_resid_mr,
        })

    for w in [20, 60]:
        window = w

        def ivol(window=window, **cache):
            return -cache["residual_returns"].rolling(window).std()

        ivol.__name__ = f"ivol_{window}d"
        signals.append({
            "name": f"ivol_{window}d",
            "use_residual": True,
            "use_factor_model": True,
            "fn": ivol,
        })

    for w in [3, 5, 10, 20, 60]:
        window = w

        def resid_zscore(window=window, **cache):
            r = cache["residual_returns"]
            mu = r.rolling(window).mean()
            sigma = r.rolling(window).std().clip(lower=1e-8)
            return -((r - mu) / sigma).clip(-2, 2)

        resid_zscore.__name__ = f"resid_zscore_w{window}"
        signals.append({
            "name": f"resid_zscore_w{window}",
            "use_residual": True,
            "use_factor_model": False,
            "fn": resid_zscore,
        })

    for w in [2, 5, 10]:
        window = w

        def etf_factor_resid_mr(window=window, **cache):
            return -cache["residual_returns"].rolling(window).mean()

        etf_factor_resid_mr.__name__ = f"etf_factor_resid_mr_{window}d"
        signals.append({
            "name": f"etf_factor_resid_mr_{window}d",
            "use_residual": True,
            "use_factor_model": False,
            "fn": etf_factor_resid_mr,
        })

    for w in [5, 10, 20]:
        window = w

        def resid_zscore_w15(window=window, **cache):
            r = cache["residual_returns"]
            mu = r.rolling(window).mean()
            sigma = r.rolling(window).std().clip(lower=1e-8)
            return -((r - mu) / sigma).clip(-1.5, 1.5)

        resid_zscore_w15.__name__ = f"resid_zscore_w15_w{window}"
        signals.append({
            "name": f"resid_zscore_w15_w{window}",
            "use_residual": True,
            "use_factor_model": False,
            "fn": resid_zscore_w15,
        })

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
