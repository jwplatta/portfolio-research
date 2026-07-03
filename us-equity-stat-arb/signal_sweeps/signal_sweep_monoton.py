"""Signal sweep: monotonicity signals.

Signals: weighted consistency, unweighted, signed, skip-1-week variants
         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_monoton.py
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
)
from sweep_scalers import apply_scalers

GROUP = "monoton"
OUT_DIR = Path(__file__).resolve().parent / "out" / "monoton"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:
    signals = []

    # Weighted: consistency score weighted by mean magnitude.
    # Short windows (w20d, w40d) showed poor pool-candidate performance — dropped.
    for w in [60, 120, 252]:
        window = w

        def monoton_weighted(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            return (r.gt(0) == mu.gt(0)).rolling(window).mean() * mu.abs()

        monoton_weighted.__name__ = f"monoton_w{w}d"
        signals.append({"name": f"monoton_w{w}d", "use_residual": False, "fn": monoton_weighted})

    # Unweighted: consistency score only.
    # u20d and u60d dropped; u120d kept as the one unweighted variant with meaningful edge.
    for w in [120]:
        window = w

        def monoton_unweighted(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            return (r.gt(0) == mu.gt(0)).rolling(window).mean()

        monoton_unweighted.__name__ = f"monoton_u{w}d"
        signals.append({"name": f"monoton_u{w}d", "use_residual": False, "fn": monoton_unweighted})

    # Signed: consistency weighted by direction of mean.
    # s40d dropped; s60d and s120d kept.
    for w in [60, 120]:
        window = w

        def monoton_signed(window=window, **cache):
            r = cache["_active_returns"]
            mu = r.rolling(window).mean()
            return (r.gt(0) == mu.gt(0)).rolling(window).mean() * np.sign(mu)

        monoton_signed.__name__ = f"monoton_s{w}d"
        signals.append({"name": f"monoton_s{w}d", "use_residual": False, "fn": monoton_signed})

    # Skip-1-week: shift returns by 5 days before computing.
    # Best overall family — all three windows kept.
    for w in [60, 120, 252]:
        window = w

        def monoton_skip(window=window, **cache):
            r = cache["_active_returns"].shift(5)
            mu = r.rolling(window).mean()
            return (r.gt(0) == mu.gt(0)).rolling(window).mean() * mu.abs()

        monoton_skip.__name__ = f"monoton_skip_{w}d"
        signals.append({"name": f"monoton_skip_{w}d", "use_residual": False, "fn": monoton_skip})

    return signals


# ---------------------------------------------------------------------------
# Scaler configs
# ---------------------------------------------------------------------------

def make_scaler_configs() -> list[dict]:
    # Removed scalers:
    #   breadth_40/50/60 — formula was buggy (cumprod vs rolling MA fired almost never)
    #   disp_*           — over-filters: kills 2018-2023 activity entirely
    #   crash_*          — no meaningful improvement over baseline
    #   trend_10_60      — too noisy, dominated by trend_20_100
    #   vol_20_60        — redundant with vol_10_60 and vol_20_100
    #
    # Added scalers:
    #   trend_*_off      — scale to 0 (not 0.25) in downtrend; filter_dev showed this
    #                      cleanly fixes 2015 without hurting other years
    #   breadth_35_off / breadth_40_off — fixed breadth (prices > prices.shift(200));
    #                      threshold ~35-40% softens 2015 with minimal false positives
    #   dd_guard_5pct    — turns off when SPY drops >5% from rolling peak (21-day hold);
    #                      more responsive to early selloffs than MA crossover
    return [
        {"tag": "none",             "trend": None,                                        "vol_spike": None,                                "breadth": None,                                "dd_guard": None},
        {"tag": "trend_20_100",     "trend": {"fast": 20, "slow": 100, "scale_down": 0.25}, "vol_spike": None,                                "breadth": None,                                "dd_guard": None},
        {"tag": "trend_50_200",     "trend": {"fast": 50, "slow": 200, "scale_down": 0.25}, "vol_spike": None,                                "breadth": None,                                "dd_guard": None},
        {"tag": "trend_20_100_off", "trend": {"fast": 20, "slow": 100, "scale_down": 0.0},  "vol_spike": None,                                "breadth": None,                                "dd_guard": None},
        {"tag": "trend_50_200_off", "trend": {"fast": 50, "slow": 200, "scale_down": 0.0},  "vol_spike": None,                                "breadth": None,                                "dd_guard": None},
        {"tag": "vol_10_60",        "trend": None,                                        "vol_spike": {"fast": 10, "slow": 60,  "scale_down": 0.25}, "breadth": None,                                "dd_guard": None},
        {"tag": "vol_20_100",       "trend": None,                                        "vol_spike": {"fast": 20, "slow": 100, "scale_down": 0.25}, "breadth": None,                                "dd_guard": None},
        {"tag": "breadth_35_off",   "trend": None,                                        "vol_spike": None,                                "breadth": {"threshold": 0.35, "scale_down": 0.0}, "dd_guard": None},
        {"tag": "breadth_40_off",   "trend": None,                                        "vol_spike": None,                                "breadth": {"threshold": 0.40, "scale_down": 0.0}, "dd_guard": None},
        {"tag": "dd_guard_5pct",    "trend": None,                                        "vol_spike": None,                                "breadth": None,                                "dd_guard": {"threshold": -0.05, "recovery": 21}},
    ]


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------

def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    fn = entry["fn"]
    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)

    # Monoton signals never use residuals.
    # Scaler dispatch via apply_scalers handles: trend (MR polarity with scale_down key),
    # vol_spike (with scale_down key), breadth (with scale_down key, uses shift(200) method),
    # and dd_guard (SPY drawdown guard).
    builder = (
        qs.Study(universe=universe, benchmark=benchmark, factors=factors, verbose=verbose)
        .base_signal(fn)
        .add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=equity_curve_scaler)
    )

    builder = apply_scalers(builder, scaler_cfg)

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
