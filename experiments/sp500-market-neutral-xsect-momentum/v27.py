"""v27 - residual momentum with 30-day lookback and 5-day skip."""

from common import build_study, emit_metrics, residual_momentum_signal

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v27_residual_30_5"
study = build_study(
    STUDY_NAME,
    signal_fn=residual_momentum_signal(lookback=30, skip=5, shift=1),
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
