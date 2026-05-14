"""v37 - thresholded residual event continuation."""

from common import build_study, emit_metrics, residual_threshold_event_signal

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v37_threshold_event_continuation"
study = build_study(
    STUDY_NAME,
    signal_fn=residual_threshold_event_signal(
        move_window=5,
        move_z_window=30,
        volume_window=30,
        min_move_z=1.5,
        shift=1,
    ),
    residualize=True,
    n_long=20,
    n_short=20,
    min_price_threshold=5.0,
    min_adv_threshold=20_000_000.0,
    liquidity_top_n=150,
    rebalance_every=10,
    factor_model_factors=["market"],
    position_neutralization_constraints={"market": 0},
    weighting="equal",
)

if __name__ == "__main__":
    emit_metrics(study)
