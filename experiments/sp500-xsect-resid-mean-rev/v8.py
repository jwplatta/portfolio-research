from common import (
    build_study,
    emit_metrics,
    equity_curve_regime_scale,
    residual_vol_regime_scale,
)

study = build_study(
    name="resid_mr_v8_residual_vol_regime",
    risk_scalers=[
        residual_vol_regime_scale(lookback=20, trigger_quantile=0.8, defensive_scale=0.6),
        equity_curve_regime_scale(),
    ],
)

if __name__ == "__main__":
    emit_metrics(study)
