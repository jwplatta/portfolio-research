"""v35 - fixed-book sector-balanced residual momentum."""

from common import (
    build_study,
    emit_metrics,
    fixed_book_sector_balanced_positions,
    residual_momentum_signal,
    sector_relative_transform,
)

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v35_fixed_book_sector_balanced"
study = build_study(
    STUDY_NAME,
    signal_fn=residual_momentum_signal(lookback=30, skip=20, shift=1),
    transforms=[sector_relative_transform],
    residualize=True,
    position_builder_fn=fixed_book_sector_balanced_positions(total_longs=20, total_shorts=20),
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
