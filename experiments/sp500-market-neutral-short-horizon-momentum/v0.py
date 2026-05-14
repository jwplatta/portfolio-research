from common import build_study, emit_metrics

STUDY_NAME = "sp500_market_neutral_short_horizon_momentum_v0"
study = build_study(STUDY_NAME)

if __name__ == "__main__":
    emit_metrics(study)
