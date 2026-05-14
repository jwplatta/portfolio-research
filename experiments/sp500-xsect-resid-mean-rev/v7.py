from common import (
    benchmark_regime_scale,
    build_study,
    emit_metrics,
    equity_curve_regime_scale,
)

study = build_study(
    name="resid_mr_v7_benchmark_regime",
    risk_scalers=[
        benchmark_regime_scale(fast=100, slow=200, defensive_scale=0.6),
        equity_curve_regime_scale(),
    ],
)

if __name__ == "__main__":
    emit_metrics(study)
