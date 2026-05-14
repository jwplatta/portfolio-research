from common import build_volume_confirmed_iteration_v12, emit_metrics, save_study

STUDY_NAME = "sp500_xsect_momentum_v12_volume_confirmed"
study = build_volume_confirmed_iteration_v12(STUDY_NAME)

if __name__ == "__main__":
    save_study(study, STUDY_NAME)
    emit_metrics(study)
