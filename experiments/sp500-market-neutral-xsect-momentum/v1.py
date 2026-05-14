"""v1 - add liquidity and ADV screens."""

from common import build_study, emit_metrics

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v1_liquid"
study = build_study(
    STUDY_NAME,
    min_price_threshold=5.0,
    min_adv_threshold=20_000_000.0,
    liquidity_top_n=200,
    liquidity_window=60,
)

if __name__ == "__main__":
    emit_metrics(study)
