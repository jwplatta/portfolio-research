from common import (
    benchmark_regime_scale,
    build_study,
    emit_metrics,
    equity_curve_regime_scale,
)

study = build_study(
    name="resid_mr_v10_slow_benchmark_regime",
    risk_scalers=[
        benchmark_regime_scale(fast=150, slow=250, defensive_scale=0.75),
        equity_curve_regime_scale(),
    ],
)

if __name__ == "__main__":
    emit_metrics(study)
