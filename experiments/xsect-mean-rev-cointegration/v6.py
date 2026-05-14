from common import build_experiment, emit_metrics

study = build_experiment(
    name="xsect_coint_mr_v6_eight_by_eight",
    lookback=252,
    z_window=8,
    use_market=False,
    liquidity_top_n=150,
    n_long=8,
    n_short=8,
    rebalance_every=5,
)

if __name__ == "__main__":
    emit_metrics(study)
