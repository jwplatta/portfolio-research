"""v7 - proportional long/short sizing from z-scored momentum."""

from common import build_study, emit_metrics, proportional_long_short_positions

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v7_proportional"
study = build_study(
    STUDY_NAME,
    position_builder_fn=proportional_long_short_positions,
)

if __name__ == "__main__":
    emit_metrics(study)
