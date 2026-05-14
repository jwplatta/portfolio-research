from common import build_experiment, emit_metrics

study = build_experiment(
    name="xsect_coint_mr_v1_baseline",
    lookback=252,
    z_window=15,
    use_market=False,
    liquidity_top_n=200,
    n_long=20,
    n_short=20,
    rebalance_every=5,
)

if __name__ == "__main__":
    emit_metrics(study)
