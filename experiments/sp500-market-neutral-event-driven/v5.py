"""v5 - 3-day event window."""

from common import build_study, emit_metrics, volume_shock_move_zscore_signal

STUDY_NAME = "sp500_market_neutral_event_driven_v5_event_window_3"
study = build_study(
    STUDY_NAME,
    signal_fn=volume_shock_move_zscore_signal(
        event_window=3,
        volume_window=30,
        volume_quantile=0.9,
        move_quantile=0.8,
        zscore_window=60,
    ),
    position_neutralization_constraints={"market": 0},
)

if __name__ == "__main__":
    emit_metrics(study)
