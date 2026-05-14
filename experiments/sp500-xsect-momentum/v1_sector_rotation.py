from common import build_sector_rotation_study, emit_metrics, save_study

STUDY_NAME = "sp500_xsect_momentum_v1_sector_rotation"
study = build_sector_rotation_study(STUDY_NAME)

if __name__ == "__main__":
    save_study(study, STUDY_NAME)
    emit_metrics(study)
