from common import build_experiment, emit_metrics

study = build_experiment(
    name="xsect_coint_mr_v15_slower_rebalance_tighter_universe",
    lookback=252,
    z_window=6,
    use_market=False,
    liquidity_top_n=125,
    n_long=10,
    n_short=10,
    rebalance_every=10,
)

if __name__ == "__main__":
    emit_metrics(study)
