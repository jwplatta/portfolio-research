"""v8 - benchmark-relative momentum with liquid, concentrated construction."""

from common import benchmark_relative_momentum_signal, build_study, emit_metrics, signal_abs_quantile_filter

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v8_relative_liquid"
study = build_study(
    STUDY_NAME,
    signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
    filters=[signal_abs_quantile_filter(min_quantile=0.7)],
    n_long=15,
    n_short=15,
    min_price_threshold=5.0,
    min_adv_threshold=20_000_000.0,
    liquidity_top_n=150,
    rebalance_every=10,
    weighting="equal_sharpe",
)

if __name__ == "__main__":
    emit_metrics(study)
