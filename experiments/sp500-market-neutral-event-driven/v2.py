"""v2 - abnormal residual move relative to residual volatility."""

from common import build_study, emit_metrics, residual_vol_adjusted_event_signal

STUDY_NAME = "sp500_market_neutral_event_driven_v2_residual_vol_adjusted"
study = build_study(
    STUDY_NAME,
    signal_fn=residual_vol_adjusted_event_signal(
        event_window=10,
        volume_window=30,
        volume_quantile=0.9,
        move_quantile=0.8,
        vol_window=60,
    ),
    position_neutralization_constraints={"market": 0},
)

if __name__ == "__main__":
    emit_metrics(study)
