"""v3 - volatility-adjusted momentum signal."""

from common import build_study, emit_metrics, volatility_adjusted_momentum_signal

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v3_vol_adjusted"
study = build_study(
    STUDY_NAME,
    signal_fn=volatility_adjusted_momentum_signal(lookback=252, skip=21, vol_window=63, shift=1),
)

if __name__ == "__main__":
    emit_metrics(study)
