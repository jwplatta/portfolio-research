"""Shared filter re-exports for signal sweep scripts.

All filter functions come from sig_fam_utils. This module re-exports them
so sweep scripts can use a single local import instead of a long sys.path chain.

Each filter takes (signal: pd.DataFrame, **cache) -> pd.DataFrame and sets
signal to NaN on dates where the regime is inactive.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sig_fam_utils import (
    # Breadth / market-width filters
    filter_breadth_weak_40,
    filter_breadth_lt50,
    filter_breadth_strong_55,
    filter_bear_narrow_lt40,
    filter_bear_narrow_lt40_no_recovery,
    filter_narrow_bull_off_40,
    filter_narrow_bull_off_50,
    filter_narrow_bear_off_40,
    filter_narrow_bear_off_50,
    # Trend filters
    filter_uptrend_50_200,
    filter_downtrend_50_200,
    filter_market_trend_down_20_100,
    # Dispersion / vol filters
    filter_dispersion_high_60_q75,
    filter_dispersion_high_60_q60,
    filter_residual_dispersion_high_20_q75,
    filter_low_disp_off_q30,
    filter_vol_contraction_10_60,
    filter_vol_expansion_10_60,
    # Sector / stress filters
    filter_sector_dislocation_5_q80,
    filter_panic_10d_minus5,
    filter_sector_disp_20d_q70,
    filter_sector_disp_20d_q60,
    filter_stress_disp_20d_q70,
    filter_stress_disp_20d_q60,
)

# Named mapping for sweeps that iterate over a set of conditioning filters.
CONDITIONING_FILTERS: dict[str, object] = {
    "none":                     None,
    "breadth_lt40":             filter_breadth_weak_40,
    "breadth_lt50":             filter_breadth_lt50,
    "breadth_strong_55":        filter_breadth_strong_55,
    "bear_narrow_lt40":         filter_bear_narrow_lt40,
    "narrow_bull_off_40":       filter_narrow_bull_off_40,
    "narrow_bull_off_50":       filter_narrow_bull_off_50,
    "narrow_bear_off_40":       filter_narrow_bear_off_40,
    "narrow_bear_off_50":       filter_narrow_bear_off_50,
    "uptrend_50_200":           filter_uptrend_50_200,
    "downtrend_50_200":         filter_downtrend_50_200,
    "trend_down_20_100":        filter_market_trend_down_20_100,
    "disp_high_60_q75":         filter_dispersion_high_60_q75,
    "disp_high_60_q60":         filter_dispersion_high_60_q60,
    "resid_disp_high_20_q75":   filter_residual_dispersion_high_20_q75,
    "low_disp_off_q30":         filter_low_disp_off_q30,
    "vol_contraction_10_60":    filter_vol_contraction_10_60,
    "vol_expansion_10_60":      filter_vol_expansion_10_60,
    "sector_disp_20d_q70":      filter_sector_disp_20d_q70,
    "sector_disp_20d_q60":      filter_sector_disp_20d_q60,
    "stress_disp_20d_q70":      filter_stress_disp_20d_q70,
    "stress_disp_20d_q60":      filter_stress_disp_20d_q60,
    "panic_10d_minus5":         filter_panic_10d_minus5,
    "sector_dislocation_5_q80": filter_sector_dislocation_5_q80,
}

__all__ = [
    "CONDITIONING_FILTERS",
    # Breadth
    "filter_breadth_weak_40",
    "filter_breadth_lt50",
    "filter_breadth_strong_55",
    "filter_bear_narrow_lt40",
    "filter_bear_narrow_lt40_no_recovery",
    "filter_narrow_bull_off_40",
    "filter_narrow_bull_off_50",
    "filter_narrow_bear_off_40",
    "filter_narrow_bear_off_50",
    # Trend
    "filter_uptrend_50_200",
    "filter_downtrend_50_200",
    "filter_market_trend_down_20_100",
    # Dispersion / vol
    "filter_dispersion_high_60_q75",
    "filter_dispersion_high_60_q60",
    "filter_residual_dispersion_high_20_q75",
    "filter_low_disp_off_q30",
    "filter_vol_contraction_10_60",
    "filter_vol_expansion_10_60",
    # Sector / stress
    "filter_sector_dislocation_5_q80",
    "filter_panic_10d_minus5",
    "filter_sector_disp_20d_q70",
    "filter_sector_disp_20d_q60",
    "filter_stress_disp_20d_q70",
    "filter_stress_disp_20d_q60",
]
