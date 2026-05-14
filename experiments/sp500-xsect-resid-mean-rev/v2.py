from common import (
    build_study,
    emit_metrics,
    equity_curve_regime_scale,
    load_sector_factors,
)

study = build_study(
    name="resid_mr_v2_signal_vol_scaled",
    factors_loader=load_sector_factors,
    signal_vol_window=20,
    risk_scalers=[equity_curve_regime_scale()],
)

if __name__ == "__main__":
    emit_metrics(study)
