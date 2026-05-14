from common import (
    beta_neutralize_positions,
    build_study,
    emit_metrics,
    equity_curve_regime_scale,
    load_sector_factors,
    residual_shock_filter,
)

study = build_study(
    name="resid_mr_v4_beta_neutral",
    factors_loader=load_sector_factors,
    extra_filters=[residual_shock_filter(shock_threshold=2.5, vol_window=20)],
    risk_scalers=[beta_neutralize_positions(window=60), equity_curve_regime_scale()],
)

if __name__ == "__main__":
    emit_metrics(study)
