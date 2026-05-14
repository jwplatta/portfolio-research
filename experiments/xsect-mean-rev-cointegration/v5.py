from common import build_experiment, emit_metrics

study = build_experiment(
    name="xsect_coint_mr_v5_tighter_basket",
    lookback=252,
    z_window=8,
    use_market=False,
    liquidity_top_n=150,
    n_long=5,
    n_short=5,
    rebalance_every=5,
)

if __name__ == "__main__":
    emit_metrics(study)
