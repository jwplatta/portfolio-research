from common import build_experiment, emit_metrics

study = build_experiment(
    name="xsect_coint_mr_v2_shorter_signal",
    lookback=126,
    z_window=5,
    use_market=False,
    liquidity_top_n=150,
    n_long=10,
    n_short=10,
    rebalance_every=5,
)

if __name__ == "__main__":
    emit_metrics(study)
