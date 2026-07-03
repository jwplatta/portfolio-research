"""Shared scaler utilities for signal sweep scripts.

Provides:
  - SCALER_PRESETS_MR  — preset configs for mean-reversion-style sweeps
  - SCALER_PRESETS_MOM — preset configs for momentum-style sweeps
  - apply_scalers(builder, scaler_cfg) — dispatch function that chains .scale_risk() calls

Scaler config dicts use these keys (all optional, None = inactive):
  tag        : str  — human-readable label used in output filenames
  trend      : dict — {"fast", "slow", "scale_down"=0.25}
                       MR polarity if scale_down applies in uptrend;
                       momentum polarity if scale_down applies in downtrend.
                       When scale_down=0.0, fully turns off in the condition.
  vol_exp    : dict — {"fast", "slow"}  (also accepted: vol_spike, vol_shock)
  corr       : dict — {"window", "high_q"}
  breadth    : dict — {"threshold"} uses pct above 200d MA, OR
                       {"window", "low_q"} uses rolling daily-breadth quantile
  dispersion : dict — {"window", "quantile"}  (also accepted: disp)
  crash      : dict — {"window", "threshold"}
  dd_guard   : dict — {"threshold", "recovery"=21}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from sig_fam_utils import (
    make_breadth_scaler,
    make_crash_scaler,
    make_disp_scaler,
    make_dd_guard_scaler,
    make_high_disp_scaler,
    make_trend_scaler,
    make_vol_scaler,
    make_vol_scaler_up,
    make_vol_off_scaler,
)

# ---------------------------------------------------------------------------
# Preset config lists
# ---------------------------------------------------------------------------

SCALER_PRESETS_MR: list[dict] = [
    {"tag": "none"},
    # Trend — MR polarity: scale_down applies in uptrend
    {"tag": "trend_20_100",   "trend": {"fast": 20, "slow": 100, "scale_down": 0.25}},
    {"tag": "trend_50_200",   "trend": {"fast": 50, "slow": 200, "scale_down": 0.25}},
    {"tag": "trend_20_100_h", "trend": {"fast": 20, "slow": 100, "scale_down": 0.5}},
    # Vol expansion — scale down when vol is spiking
    {"tag": "vol_10_60",  "vol_exp": {"fast": 10, "slow": 60}},
    {"tag": "vol_20_100", "vol_exp": {"fast": 20, "slow": 100}},
    {"tag": "vol_20_60",  "vol_exp": {"fast": 20, "slow": 60}},
    # Correlation — scale down when avg pairwise correlation is high
    {"tag": "corr_20_q75", "corr": {"window": 20, "high_q": 0.75}},
    {"tag": "corr_20_q80", "corr": {"window": 20, "high_q": 0.80}},
    {"tag": "corr_60_q75", "corr": {"window": 60, "high_q": 0.75}},
]

SCALER_PRESETS_MOM: list[dict] = [
    {"tag": "none"},
    # Trend — momentum polarity: scale_down applies in downtrend (no scale_down key)
    {"tag": "trend_20_100", "trend": {"fast": 20, "slow": 100}},
    {"tag": "trend_50_200", "trend": {"fast": 50, "slow": 200}},
    {"tag": "trend_10_60",  "trend": {"fast": 10, "slow": 60}},
    # Vol spike — scale down in high-vol
    {"tag": "vol_10_60",  "vol_spike": {"fast": 10, "slow": 60}},
    {"tag": "vol_20_100", "vol_spike": {"fast": 20, "slow": 100}},
    {"tag": "vol_20_60",  "vol_spike": {"fast": 20, "slow": 60}},
    # Breadth — scale down when breadth is weak
    {"tag": "breadth_40", "breadth": {"threshold": 0.40}},
    {"tag": "breadth_50", "breadth": {"threshold": 0.50}},
    {"tag": "breadth_60", "breadth": {"threshold": 0.60}},
    # Crash rebound — scale down on violent up-move after a crash
    {"tag": "crash_5_3pct",  "crash": {"window": 5,  "threshold": 0.03}},
    {"tag": "crash_5_5pct",  "crash": {"window": 5,  "threshold": 0.05}},
    {"tag": "crash_10_5pct", "crash": {"window": 10, "threshold": 0.05}},
    # Dispersion — scale down when dispersion is low
    {"tag": "disp_60_q20", "dispersion": {"window": 60, "quantile": 0.20}},
    {"tag": "disp_60_q30", "dispersion": {"window": 60, "quantile": 0.30}},
    {"tag": "disp_40_q25", "dispersion": {"window": 40, "quantile": 0.25}},
]


# ---------------------------------------------------------------------------
# Corr scaler (not in sig_fam_utils — defined here)
# ---------------------------------------------------------------------------

def make_corr_scaler(window: int, high_q: float, scale_down: float = 0.25):
    """Scale down when rolling avg pairwise correlation exceeds its high_q percentile."""

    def corr_scaler(positions: pd.DataFrame, **cache):
        rets = cache["returns"].dropna(axis=1, how="all")
        sample = rets.iloc[:, :50]
        avg_corr = (
            sample.rolling(window)
            .corr()
            .groupby(level=0)
            .apply(lambda m: (m.values.sum() - len(m)) / max(len(m) * (len(m) - 1), 1))
        )
        threshold = avg_corr.rolling(252).quantile(high_q)
        high_corr = avg_corr.gt(threshold).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(high_corr, scale_down, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    corr_scaler.__name__ = f"corr_{window}_{high_q}"
    return corr_scaler


# ---------------------------------------------------------------------------
# Breadth (quantile-based, e.g. event and volume sweeps)
# ---------------------------------------------------------------------------

def make_breadth_quantile_scaler(window: int, low_q: float, scale_down: float = 0.25):
    """Scale down when rolling daily breadth (% positive returns) is below its low_q percentile.

    Different from make_breadth_scaler (which compares vs a fixed threshold on pct above 200d MA).
    """

    def breadth_q_scaler(positions: pd.DataFrame, **cache):
        r = cache["returns"].dropna(axis=1, how="all")
        breadth = (r > 0).mean(axis=1)
        threshold = breadth.rolling(252).quantile(low_q)
        low_breadth = breadth.lt(threshold).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(low_breadth, scale_down, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    breadth_q_scaler.__name__ = f"breadth_{window}_q{int(low_q * 100)}"
    return breadth_q_scaler


# ---------------------------------------------------------------------------
# Monoton-style breadth (compares price vs shift(200), not rolling mean)
# ---------------------------------------------------------------------------

def make_breadth_shift_scaler(threshold: float, scale_down: float = 0.0):
    """Scale down when % stocks with price > price-200d-ago is below threshold.

    Uses prices.shift(200) rather than rolling(200).mean() to avoid cumprod drift.
    Default scale_down=0.0 (fully off when breadth weak).
    """

    def breadth_shift_scaler(positions: pd.DataFrame, **cache):
        returns = cache.get("returns")
        if returns is None:
            return positions
        prices = (1 + returns).cumprod()
        pct_above = (prices > prices.shift(200)).mean(axis=1)
        below = pct_above.lt(threshold).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(below, scale_down, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    breadth_shift_scaler.__name__ = f"breadth_shift_{int(threshold * 100)}"
    return breadth_shift_scaler


# ---------------------------------------------------------------------------
# DD guard (SPY drawdown — monoton uses this, slightly different from sig_fam_utils version)
# ---------------------------------------------------------------------------

def make_spy_dd_guard_scaler(threshold: float, recovery: int = 21):
    """Scale to 0 when SPY is in drawdown below threshold (not the sleeve's equity curve).

    Different from make_dd_guard_scaler which guards on the sleeve's own returns.
    """

    def spy_dd_guard(positions: pd.DataFrame, **cache):
        bm = cache.get("benchmark")
        if bm is None:
            return positions
        spy_price = (1 + bm).cumprod()
        rolling_peak = spy_price.cummax()
        drawdown = (spy_price / rolling_peak) - 1
        in_dd = drawdown.lt(threshold).reindex(positions.index).fillna(False)
        in_dd_extended = in_dd.rolling(recovery, min_periods=1).max().astype(bool)
        scale = pd.Series(np.where(in_dd_extended, 0.0, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)

    spy_dd_guard.__name__ = f"spy_dd_guard_{int(abs(threshold) * 100)}pct"
    return spy_dd_guard


# ---------------------------------------------------------------------------
# apply_scalers — main dispatch function
# ---------------------------------------------------------------------------

def apply_scalers(builder, scaler_cfg: dict):
    """Apply all scalers encoded in scaler_cfg to the study builder.

    Reads keys: trend, vol_exp/vol_spike/vol_shock, corr, breadth, dispersion/disp,
                crash, dd_guard.

    Returns the updated builder (for chaining).
    """
    # ------------------------------------------------------------------
    # Trend scaler
    # ------------------------------------------------------------------
    trend = scaler_cfg.get("trend")
    if trend is not None:
        fast, slow = trend["fast"], trend["slow"]
        scale_down = trend.get("scale_down")
        if scale_down is None:
            # No scale_down key → momentum polarity (scale down in downtrend, full in uptrend)
            scaler = make_trend_scaler(fast, slow, mr_style=False, scale_down=0.25)
        else:
            # Has scale_down → MR polarity (scale_down applies in uptrend)
            scaler = make_trend_scaler(fast, slow, mr_style=True, scale_down=scale_down)
        builder = builder.scale_risk(fn=scaler)

    # ------------------------------------------------------------------
    # Vol scaler — accepts vol_exp, vol_spike, or vol_shock keys
    # ------------------------------------------------------------------
    vol = scaler_cfg.get("vol_exp") or scaler_cfg.get("vol_spike") or scaler_cfg.get("vol_shock")
    if vol is not None:
        fast, slow = vol["fast"], vol["slow"]
        scale_no_spike = vol.get("scale_no_spike")
        if scale_no_spike is not None and scale_no_spike == 0.0:
            # vol_10_60_off: fully turn off when vol is NOT spiking
            scaler = make_vol_off_scaler(fast, slow)
        else:
            sd = vol.get("scale_down", 0.25)
            scaler = make_vol_scaler(fast, slow, scale_down=sd)
        builder = builder.scale_risk(fn=scaler)

    # ------------------------------------------------------------------
    # Correlation scaler
    # ------------------------------------------------------------------
    corr = scaler_cfg.get("corr")
    if corr is not None:
        builder = builder.scale_risk(fn=make_corr_scaler(corr["window"], corr["high_q"]))

    # ------------------------------------------------------------------
    # Breadth scaler — two variants:
    #   "threshold" key → pct stocks above 200d rolling MA (standard momentum breadth)
    #   "low_q" key     → rolling daily breadth quantile (event/volume sweep style)
    #   "scale_down" key in breadth dict → monoton-style breadth with shift(200)
    # ------------------------------------------------------------------
    breadth = scaler_cfg.get("breadth")
    if breadth is not None:
        if "low_q" in breadth:
            window = breadth.get("window", 20)
            scaler = make_breadth_quantile_scaler(window, breadth["low_q"])
        elif "scale_down" in breadth:
            scaler = make_breadth_shift_scaler(breadth["threshold"], breadth["scale_down"])
        else:
            scaler = make_breadth_scaler(breadth["threshold"])
        builder = builder.scale_risk(fn=scaler)

    # ------------------------------------------------------------------
    # Dispersion scaler — accepts dispersion or disp keys
    # ------------------------------------------------------------------
    disp = scaler_cfg.get("dispersion") or scaler_cfg.get("disp")
    if disp is not None:
        if "quantile" in disp:
            # Standard: scale down when dispersion is LOW
            builder = builder.scale_risk(fn=make_disp_scaler(disp["window"], disp["quantile"]))
        elif "high_q" in disp:
            # Event-style: scale down when dispersion is HIGH
            builder = builder.scale_risk(fn=make_high_disp_scaler(disp["window"], disp["high_q"]))

    # ------------------------------------------------------------------
    # Crash scaler
    # ------------------------------------------------------------------
    crash = scaler_cfg.get("crash")
    if crash is not None:
        builder = builder.scale_risk(fn=make_crash_scaler(crash["window"], crash["threshold"]))

    # ------------------------------------------------------------------
    # DD guard scaler (SPY drawdown)
    # ------------------------------------------------------------------
    dd_guard = scaler_cfg.get("dd_guard")
    if dd_guard is not None:
        builder = builder.scale_risk(
            fn=make_spy_dd_guard_scaler(dd_guard["threshold"], dd_guard.get("recovery", 21))
        )

    return builder
