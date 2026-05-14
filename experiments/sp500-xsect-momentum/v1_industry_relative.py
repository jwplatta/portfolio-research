from common import build_industry_relative_momentum_study, emit_metrics, save_study

STUDY_NAME = "sp500_xsect_momentum_v1_industry_relative"
study = build_industry_relative_momentum_study(STUDY_NAME)

if __name__ == "__main__":
    save_study(study, STUDY_NAME)
    emit_metrics(study)
