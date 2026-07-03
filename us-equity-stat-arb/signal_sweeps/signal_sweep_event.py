"""Signal sweep: event-driven / gap reversion signals.

Signals: gap_reversion, resid_gap_reversion, gap_accum_{2,3,5,10,20}d,
         resid_gap_accum_{2,3,5,10,20}d  (12 total)
         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_event.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import qstudy as qs

sys.path.insert(0, str(Path(__file__).parent.parent))
from portfolio_utils import make_equity_curve_regime_scale
from signal_sweep_utils import (
    COST_BPS,
    N_LONG,
    N_SHORT,
    REBALANCE_PERIODS,
    TRAIN_START,
    run_sweep,
    build_study_generic,
)

GROUP = "event"
OUT_DIR = Path(__file__).resolve().parent / "out" / "event"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _returns_for_signal(cache_dict: dict, use_residual: bool) -> pd.DataFrame:
    return cache_dict["residual_returns"] if use_residual else cache_dict["_active_returns"]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def make_signals() -> list[dict]:
    signals = []

    # Signals kept and rationale (informed by full-period sweep + walkforward 2020-2023):
    #
    #   gap_accum_2d   — #1 in walkforward (SR 1.15 with breadth scaler); strong 2020-2022
    #   resid_gap_accum_2d — walkforward SR 0.93; complementary year pattern to raw 2d
    #   resid_gap_accum_5d — most consistent walkforward config (SR 1.00, positive all 4 years);
    #                         scaler-agnostic; genuine all-weather edge
    #   gap_accum_3d   — walkforward SR 0.83 with trend_20_100; 2022 dominant (+2.88);
    #                     only viable at r=21 with MR-polarity trend scaler
    #
    # Signals dropped:
    #   gap_reversion / resid_gap_reversion — catastrophic at any rebalance; avg WF SR < -1.0
    #   gap_accum_5d (raw) — underperforms residualized 5d in WF; avg SR < 0
    #   gap_accum_10d      — consistently negative in walkforward (avg WF SR -0.60)
    #   gap_accum_20d      — no edge; WF avg SR -0.77
    #   resid_gap_accum_3d — worse than raw gap_accum_3d in walkforward; dropped
    #   resid_gap_accum_10d/20d — negative across the board

    # Raw gap accumulation
    for w in [2, 3]:
        window = w

        def gap_accum(window=window, **cache):
            r = cache["_active_returns"]
            return -r.rolling(window).max()

        sig_name = f"gap_accum_{w}d"
        gap_accum.__name__ = sig_name
        signals.append({"name": sig_name, "use_residual": False, "fn": gap_accum})

    # Residualized gap accumulation (strips market/factor component)
    for w in [2, 5]:
        ur, window = True, w

        def resid_gap_accum(window=window, use_residual=ur, **cache):
            r = _returns_for_signal(cache, use_residual)
            return -r.rolling(window).max()

        sig_name = f"resid_gap_accum_{w}d"
        resid_gap_accum.__name__ = sig_name
        signals.append({"name": sig_name, "use_residual": ur, "fn": resid_gap_accum})

    return signals


# ---------------------------------------------------------------------------
# Scaler configs — event-style (vol_exp first, MR-polarity trend, corr, breadth, dispersion)
# ---------------------------------------------------------------------------


def make_scaler_configs() -> list[dict]:
    # MR polarity: gap reversion works best in choppy/downtrending markets.
    # scale_down applies to uptrend; gap reversion is scaled up (or left at 1.0) in downtrend.
    #
    # Removed:
    #   vol_20_60       — redundant with vol_10_60 and vol_20_100
    #   trend_20_100_h  — 0.5x variant dominated by 0.25x and 0.0x
    #   corr_20_q80     — near-identical to corr_20_q75
    #   breadth_60_q25  — near-identical to breadth_20_q25
    #   disp_*          — minimal differentiation from baseline in yearly breakdown
    #
    # Added:
    #   trend_20_100_off — scale to 0 in uptrend (filter_dev winner for gap_accum_3d:
    #                      eliminates bad 2017/2021 entirely, dramatically cuts 2020)
    #   trend_50_200_off — same concept, slower MA (captures more of 2015 improvement)
    #   vol_10_60_calm_off — scale to 0 when vol is NOT spiking (gap reversion thrives
    #                        in elevated-vol environments; off when markets are calm)
    return [
        {"tag": "none",              "vol_exp": None,                      "trend": None,                                        "corr": None,                      "breadth": None,                    "dispersion": None},
        # vol expansion (scale down when calm, run when vol is elevated)
        {"tag": "vol_10_60",         "vol_exp": {"fast": 10, "slow": 60},  "trend": None,                                        "corr": None,                      "breadth": None,                    "dispersion": None},
        {"tag": "vol_20_100",        "vol_exp": {"fast": 20, "slow": 100}, "trend": None,                                        "corr": None,                      "breadth": None,                    "dispersion": None},
        {"tag": "vol_10_60_off",     "vol_exp": {"fast": 10, "slow": 60, "scale_no_spike": 0.0}, "trend": None,                  "corr": None,                      "breadth": None,                    "dispersion": None},
        # MR-polarity trend: scale DOWN in uptrend (0.25x → reduce, 0.0 → fully off)
        {"tag": "trend_20_100",      "vol_exp": None,                      "trend": {"fast": 20, "slow": 100, "scale_down": 0.25}, "corr": None,                    "breadth": None,                    "dispersion": None},
        {"tag": "trend_50_200",      "vol_exp": None,                      "trend": {"fast": 50, "slow": 200, "scale_down": 0.25}, "corr": None,                    "breadth": None,                    "dispersion": None},
        {"tag": "trend_20_100_off",  "vol_exp": None,                      "trend": {"fast": 20, "slow": 100, "scale_down": 0.0},  "corr": None,                    "breadth": None,                    "dispersion": None},
        {"tag": "trend_50_200_off",  "vol_exp": None,                      "trend": {"fast": 50, "slow": 200, "scale_down": 0.0},  "corr": None,                    "breadth": None,                    "dispersion": None},
        # corr (high correlation = crowded markets = larger gaps)
        {"tag": "corr_20_q75",       "vol_exp": None,                      "trend": None,                                        "corr": {"window": 20, "high_q": 0.75}, "breadth": None,              "dispersion": None},
        {"tag": "corr_60_q75",       "vol_exp": None,                      "trend": None,                                        "corr": {"window": 60, "high_q": 0.75}, "breadth": None,              "dispersion": None},
        # breadth (low breadth = stressed market = good for gap reversion)
        {"tag": "breadth_20_q25",    "vol_exp": None,                      "trend": None,                                        "corr": None,                      "breadth": {"window": 20, "low_q": 0.25}, "dispersion": None},
        {"tag": "breadth_20_q30",    "vol_exp": None,                      "trend": None,                                        "corr": None,                      "breadth": {"window": 20, "low_q": 0.30}, "dispersion": None},
        # dispersion
        {"tag": "disp_60_q75",       "vol_exp": None,                      "trend": None,                                        "corr": None,                      "breadth": None,                    "dispersion": {"window": 60, "high_q": 0.75}},
    ]


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------


def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    # Event sweep uses the same standard pipeline as other MR sweeps.
    # The scaler_cfg keys (vol_exp with scale_no_spike, breadth with low_q,
    # dispersion with high_q) are handled by apply_scalers() in sweep_scalers.py.
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
