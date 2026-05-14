from common import build_volatility_adjusted_momentum_study, emit_metrics, save_study

STUDY_NAME = "sp500_xsect_momentum_v1_volatility_adjusted"
study = build_volatility_adjusted_momentum_study(STUDY_NAME)

if __name__ == "__main__":
    save_study(study, STUDY_NAME)
    emit_metrics(study)
