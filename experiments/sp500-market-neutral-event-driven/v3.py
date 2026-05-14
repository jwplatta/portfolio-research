"""v3 - sector-relative residual event ranking with market neutralization."""

from common import (
    build_study,
    emit_metrics,
    sector_relative_transform,
)

STUDY_NAME = "sp500_market_neutral_event_driven_v3_sector_relative"
study = build_study(
    STUDY_NAME,
    transforms=[sector_relative_transform,],
    position_neutralization_constraints={"market": 0},
)

if __name__ == "__main__":
    emit_metrics(study)
