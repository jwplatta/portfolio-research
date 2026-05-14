"""v13 - neutralize the signal and positions to market exposure."""

from common import (
    benchmark_relative_momentum_signal,
    build_study,
    emit_metrics,
    min_universe_breadth_filter,
    relative_volume_strength_filter,
    signal_abs_quantile_filter,
    volume_confirmation_filter_factory,
)

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v13_signal_and_position_neutral"
study = build_study(
    STUDY_NAME,
    signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
    filters=[
        signal_abs_quantile_filter(min_quantile=0.7),
        volume_confirmation_filter_factory(
            volume_window=30,
            volume_quantile=0.7,
            trailing_window=63,
        ),
        relative_volume_strength_filter(volume_window=63, volume_quantile=0.75),
        min_universe_breadth_filter(min_names=20),
    ],
    n_long=15,
    n_short=15,
    min_price_threshold=5.0,
    min_adv_threshold=30_000_000.0,
    liquidity_top_n=100,
    rebalance_every=10,
    factor_model_factors=["market"],
    signal_neutralization_factors=["market"],
    position_neutralization_constraints={"market": 0},
    weighting="equal_sharpe",
)

if __name__ == "__main__":
    emit_metrics(study)
