"""v31 - sector-balanced residual momentum."""

from common import (
    build_study,
    emit_metrics,
    residual_momentum_signal,
    sector_balanced_long_short_positions,
    sector_relative_transform,
)

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v31_sector_balanced_residual"
study = build_study(
    STUDY_NAME,
    signal_fn=residual_momentum_signal(lookback=30, skip=20, shift=1),
    transforms=[sector_relative_transform],
    residualize=True,
    position_builder_fn=sector_balanced_long_short_positions(top_k=2, bottom_k=2),
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
