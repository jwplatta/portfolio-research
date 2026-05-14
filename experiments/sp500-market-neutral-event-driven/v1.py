"""v1 - residual event signals with market and sector neutralization."""

from common import build_study, emit_metrics

STUDY_NAME = "sp500_market_neutral_event_driven_v1_market_sector_neutral"
study = build_study(
    STUDY_NAME,
    position_neutralization_constraints={"market": 0, "sector": 0},
)

if __name__ == "__main__":
    emit_metrics(study)
