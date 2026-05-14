from common import (
    build_study,
    emit_metrics,
    equity_curve_regime_scale,
    load_sector_factors,
    short_term_confirmation_filter,
)

study = build_study(
    name="resid_mr_v9_confirmation",
    factors_loader=load_sector_factors,
    extra_filters=[short_term_confirmation_filter(fast_window=3, slow_window=5)],
    risk_scalers=[equity_curve_regime_scale()],
)

if __name__ == "__main__":
    emit_metrics(study)
