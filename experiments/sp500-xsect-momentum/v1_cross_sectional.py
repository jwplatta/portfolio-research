from common import build_cross_sectional_momentum_study, emit_metrics, save_study

STUDY_NAME = "sp500_xsect_momentum_v1_cross_sectional"
study = build_cross_sectional_momentum_study(STUDY_NAME)

if __name__ == "__main__":
    save_study(study, STUDY_NAME)
    emit_metrics(study)
