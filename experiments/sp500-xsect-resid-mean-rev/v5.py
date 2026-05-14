from common import build_study, emit_metrics, equity_curve_regime_scale

study = build_study(
    name="resid_mr_v5_tighter_liquidity",
    liquidity_top_n=200,
    risk_scalers=[equity_curve_regime_scale()],
)

if __name__ == "__main__":
    emit_metrics(study)
