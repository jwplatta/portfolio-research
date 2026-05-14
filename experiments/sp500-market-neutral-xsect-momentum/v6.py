"""v6 - concentrate the long and short books."""

from common import build_study, emit_metrics

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v6_concentrated"
study = build_study(
    STUDY_NAME,
    n_long=15,
    n_short=15,
)

if __name__ == "__main__":
    emit_metrics(study)
