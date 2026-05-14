from common import build_study, emit_metrics, equity_curve_regime_scale, load_sector_factors

study = build_study(
    name="resid_mr_v1_sector_factors",
    factors_loader=load_sector_factors,
    risk_scalers=[equity_curve_regime_scale()],
)

if __name__ == "__main__":
    emit_metrics(study)
