"""v5 - faster 10-day rebalance cadence."""

from common import build_study, emit_metrics

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v5_faster_rebalance"
study = build_study(
    STUDY_NAME,
    rebalance_every=10,
)

if __name__ == "__main__":
    emit_metrics(study)
