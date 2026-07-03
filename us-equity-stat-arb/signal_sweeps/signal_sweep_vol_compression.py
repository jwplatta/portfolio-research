"""Signal sweep: vol compression signals.

Signals: vol_compression, vol_compression_z, vol_compression_resid

Usage:
    uv run python examples/signal_sweeps/signal_sweep_vol_compression.py
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

GROUP = "vol-compression"
OUT_DIR = Path(__file__).resolve().parent / "out" / "vol-compression"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:
    signals = []

    for short_w, long_w in [(5, 60), (10, 120)]:
        sw, lw = short_w, long_w

        # Raw ratio: -(short_vol / long_vol)
        def vol_compression(sw=sw, lw=lw, **cache):
            r = cache["_active_returns"]
            short_vol = r.rolling(sw).std()
            long_vol = r.rolling(lw).std().clip(lower=1e-8)
            return -(short_vol / long_vol)

        vol_compression.__name__ = f"vol_compression_{sw}_{lw}"
        signals.append({"name": f"vol_compression_{sw}_{lw}", "use_residual": False, "fn": vol_compression})

        # Z-scored ratio
        def vol_compression_z(sw=sw, lw=lw, **cache):
            r = cache["_active_returns"]
            ratio = r.rolling(sw).std() / r.rolling(lw).std().clip(lower=1e-8)
            mu = ratio.rolling(60).mean()
            sigma = ratio.rolling(60).std().clip(lower=1e-8)
            return -((ratio - mu) / sigma)

        vol_compression_z.__name__ = f"vol_compression_z_{sw}_{lw}"
        signals.append({"name": f"vol_compression_z_{sw}_{lw}", "use_residual": False, "fn": vol_compression_z})

        # Residualized: same ratio on market/sector-residual returns
        def vol_compression_resid(sw=sw, lw=lw, **cache):
            r = cache["residual_returns"]
            short_vol = r.rolling(sw).std()
            long_vol = r.rolling(lw).std().clip(lower=1e-8)
            return -(short_vol / long_vol)

        vol_compression_resid.__name__ = f"vol_compression_resid_{sw}_{lw}"
        signals.append({"name": f"vol_compression_resid_{sw}_{lw}", "use_residual": True, "fn": vol_compression_resid})

    return signals


# ---------------------------------------------------------------------------
# Scaler configs
# ---------------------------------------------------------------------------

def make_scaler_configs() -> list[dict]:
    return [{"tag": "none"}]


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
