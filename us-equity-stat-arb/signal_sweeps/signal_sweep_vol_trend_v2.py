"""Signal sweep: vol trend v2 signals.

Signals: ivol_accel family, vol_ewm_cross, vol_trend_consistency,
         downside_vol_trend, ret_vol_signal, ivol_accel_explosion, vol_regime_ret
         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_vol_trend_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import qstudy as qs

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from portfolio_utils import make_equity_curve_regime_scale

from signal_sweep_utils import (
    TRAIN_START,
    COST_BPS,
    N_LONG,
    N_SHORT,
    run_sweep,
    build_study_generic,
)

GROUP = "vol-trend-v2"
OUT_DIR = Path(__file__).resolve().parent / "out" / "vol-trend-v2"
REBALANCE_PERIODS = [5, 10, 21]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals() -> list[dict]:
    signals = []

    # -----------------------------------------------------------------------
    # ivol_accel family — -(fast_resid_vol - slow_resid_vol)
    # -----------------------------------------------------------------------
    for fast, slow in [(5, 20), (5, 60), (10, 60), (10, 90), (10, 120), (20, 90), (20, 120)]:
        f, s = fast, slow

        def ivol_accel(f=f, s=s, **cache):
            r = cache.get("residual_returns", cache["_active_returns"])
            return -(r.rolling(f).std() - r.rolling(s).std())

        ivol_accel.__name__ = f"ivol_accel_{f}_{s}"
        signals.append({"name": f"ivol_accel_{f}_{s}", "fn": ivol_accel, "use_residual": True})

    # Z-scored ivol_accel
    for fast, slow, zw in [(5, 60, 60), (10, 90, 90), (10, 120, 120)]:
        f, s, z = fast, slow, zw

        def ivol_accel_z(f=f, s=s, z=z, **cache):
            r = cache.get("residual_returns", cache["_active_returns"])
            accel = -(r.rolling(f).std() - r.rolling(s).std())
            mu = accel.rolling(z).mean()
            sigma = accel.rolling(z).std().clip(lower=1e-8)
            return (accel - mu) / sigma

        ivol_accel_z.__name__ = f"ivol_accel_z_{f}_{s}_{z}"
        signals.append({"name": f"ivol_accel_z_{f}_{s}_{z}", "fn": ivol_accel_z, "use_residual": True})

    # -----------------------------------------------------------------------
    # vol_ewm_cross family
    # -----------------------------------------------------------------------
    for fast_span, slow_span in [(5, 20), (5, 60), (10, 60), (20, 90), (20, 120)]:
        fs, ss = fast_span, slow_span

        def vol_ewm_cross(fs=fs, ss=ss, **cache):
            r = cache["_active_returns"]
            return -(r.ewm(span=fs).std() - r.ewm(span=ss).std())

        vol_ewm_cross.__name__ = f"vol_ewm_cross_{fs}_{ss}"
        signals.append({"name": f"vol_ewm_cross_{fs}_{ss}", "fn": vol_ewm_cross})

    # EWM cross on residual returns
    for fast_span, slow_span in [(5, 60), (10, 90), (20, 120)]:
        fs, ss = fast_span, slow_span

        def ivol_ewm_cross(fs=fs, ss=ss, **cache):
            r = cache.get("residual_returns", cache["_active_returns"])
            return -(r.ewm(span=fs).std() - r.ewm(span=ss).std())

        ivol_ewm_cross.__name__ = f"ivol_ewm_cross_{fs}_{ss}"
        signals.append({"name": f"ivol_ewm_cross_{fs}_{ss}", "fn": ivol_ewm_cross, "use_residual": True})

    # -----------------------------------------------------------------------
    # vol_trend_consistency
    # -----------------------------------------------------------------------
    for vol_w, cons_w in [(10, 40), (10, 60), (20, 60), (20, 90), (20, 120)]:
        vw, cw = vol_w, cons_w

        def vol_trend_consistency(vw=vw, cw=cw, **cache):
            r = cache["_active_returns"]
            vol = r.rolling(vw).std()
            decreased = (vol.diff() < 0).astype(float)
            return decreased.rolling(cw).mean()

        vol_trend_consistency.__name__ = f"vol_consistency_{vw}_{cw}"
        signals.append({"name": f"vol_consistency_{vw}_{cw}", "fn": vol_trend_consistency})

    # Consistency on residual vol
    for vol_w, cons_w in [(10, 60), (20, 90), (20, 120)]:
        vw, cw = vol_w, cons_w

        def ivol_consistency(vw=vw, cw=cw, **cache):
            r = cache.get("residual_returns", cache["_active_returns"])
            vol = r.rolling(vw).std()
            decreased = (vol.diff() < 0).astype(float)
            return decreased.rolling(cw).mean()

        ivol_consistency.__name__ = f"ivol_consistency_{vw}_{cw}"
        signals.append({"name": f"ivol_consistency_{vw}_{cw}", "fn": ivol_consistency, "use_residual": True})

    # -----------------------------------------------------------------------
    # downside_vol_trend
    # -----------------------------------------------------------------------
    for fast, slow in [(5, 60), (10, 60), (10, 90), (10, 120), (20, 90), (20, 120)]:
        f, s = fast, slow

        def downside_vol_trend(f=f, s=s, **cache):
            r = cache["_active_returns"]
            neg_r = r.where(r < 0, 0.0)
            return -(neg_r.rolling(f).std() - neg_r.rolling(s).std())

        downside_vol_trend.__name__ = f"downside_vol_{f}_{s}"
        signals.append({"name": f"downside_vol_{f}_{s}", "fn": downside_vol_trend})

    # Downside vol on residual returns
    for fast, slow in [(10, 90), (10, 120), (20, 120)]:
        f, s = fast, slow

        def idownside_vol_trend(f=f, s=s, **cache):
            r = cache.get("residual_returns", cache["_active_returns"])
            neg_r = r.where(r < 0, 0.0)
            return -(neg_r.rolling(f).std() - neg_r.rolling(s).std())

        idownside_vol_trend.__name__ = f"idownside_vol_{f}_{s}"
        signals.append({"name": f"idownside_vol_{f}_{s}", "fn": idownside_vol_trend, "use_residual": True})

    # -----------------------------------------------------------------------
    # Return-conditioned vol signals: -(return * short_vol / long_vol)
    # -----------------------------------------------------------------------
    for ret_w, fast, slow in [(1, 5, 60), (5, 10, 90), (5, 10, 120)]:
        rw, f, s = ret_w, fast, slow

        def ret_vol_signal(rw=rw, f=f, s=s, **cache):
            r = cache["_active_returns"]
            recent_ret = r.rolling(rw).mean()
            vol_ratio = r.rolling(f).std() / r.rolling(s).std().clip(lower=1e-8)
            return -(recent_ret * vol_ratio)

        ret_vol_signal.__name__ = f"ret_vol_{rw}_{f}_{s}"
        signals.append({"name": f"ret_vol_{rw}_{f}_{s}", "fn": ret_vol_signal})

    # Same on residual returns
    for ret_w, fast, slow in [(1, 5, 60), (5, 10, 120)]:
        rw, f, s = ret_w, fast, slow

        def iret_vol_signal(rw=rw, f=f, s=s, **cache):
            r = cache.get("residual_returns", cache["_active_returns"])
            recent_ret = r.rolling(rw).mean()
            vol_ratio = r.rolling(f).std() / r.rolling(s).std().clip(lower=1e-8)
            return -(recent_ret * vol_ratio)

        iret_vol_signal.__name__ = f"iret_vol_{rw}_{f}_{s}"
        signals.append({"name": f"iret_vol_{rw}_{f}_{s}", "fn": iret_vol_signal, "use_residual": True})

    # -----------------------------------------------------------------------
    # Vol explosion filter: ivol_accel active only in top N% of vol ratio
    # -----------------------------------------------------------------------
    for fast, slow, pct in [(5, 60, 0.90), (10, 90, 0.90), (10, 120, 0.95)]:
        f, s, p = fast, slow, pct

        def ivol_accel_explosion(f=f, s=s, p=p, **cache):
            r = cache.get("residual_returns", cache["_active_returns"])
            fast_vol = r.rolling(f).std()
            slow_vol = r.rolling(s).std().clip(lower=1e-8)
            vol_ratio = fast_vol / slow_vol
            threshold = vol_ratio.quantile(p, axis=1)
            in_explosion = vol_ratio.ge(threshold, axis=0)
            accel = -(fast_vol - slow_vol)
            return accel.where(in_explosion)

        pct_str = str(int(p * 100))
        ivol_accel_explosion.__name__ = f"ivol_explosion_{f}_{s}_p{pct_str}"
        signals.append({"name": f"ivol_explosion_{f}_{s}_p{pct_str}", "fn": ivol_accel_explosion, "use_residual": True})

    # -----------------------------------------------------------------------
    # Vol regime × residual return: sign(ivol_accel) * |residual_return|
    # -----------------------------------------------------------------------
    for fast, slow, ret_w in [(5, 60, 5), (10, 90, 5), (10, 120, 10)]:
        f, s, rw = fast, slow, ret_w

        def vol_regime_ret(f=f, s=s, rw=rw, **cache):
            r = cache.get("residual_returns", cache["_active_returns"])
            accel = -(r.rolling(f).std() - r.rolling(s).std())
            direction = accel.apply(np.sign)
            magnitude = r.rolling(rw).mean().abs()
            return direction * magnitude

        vol_regime_ret.__name__ = f"vol_regime_ret_{f}_{s}_r{rw}"
        signals.append({"name": f"vol_regime_ret_{f}_{s}_r{rw}", "fn": vol_regime_ret, "use_residual": True})

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
