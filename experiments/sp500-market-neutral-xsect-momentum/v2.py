"""v2 - benchmark-relative momentum signal."""

from common import benchmark_relative_momentum_signal, build_study, emit_metrics

STUDY_NAME = "sp500_market_neutral_xsect_momentum_v2_relative"
study = build_study(
    STUDY_NAME,
    signal_fn=benchmark_relative_momentum_signal(lookback=252, skip=21, shift=1),
)

if __name__ == "__main__":
    emit_metrics(study)
