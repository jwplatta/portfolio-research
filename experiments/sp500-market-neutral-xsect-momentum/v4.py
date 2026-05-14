"""v4 - keep only the strongest absolute momentum names."""

from common import build_study, emit_metrics, signal_abs_quantile_filter

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v4_strength"
study = build_study(
    STUDY_NAME,
    filters=[signal_abs_quantile_filter(min_quantile=0.7)],
)

if __name__ == "__main__":
    emit_metrics(study)
