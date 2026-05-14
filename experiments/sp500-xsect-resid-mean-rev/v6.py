from common import build_study, emit_metrics, equity_curve_regime_scale

study = build_study(
    name="resid_mr_v6_two_day_rebalance",
    rebalance_every=2,
    risk_scalers=[equity_curve_regime_scale()],
)

if __name__ == "__main__":
    emit_metrics(study)
