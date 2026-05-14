from common import build_experiment, emit_metrics

study = build_experiment(
    name="xsect_coint_mr_v12_slower_rebalance_eqvol",
    lookback=252,
    z_window=6,
    use_market=False,
    liquidity_top_n=150,
    n_long=10,
    n_short=10,
    rebalance_every=10,
    equal_vol_window=60,
)

if __name__ == "__main__":
    emit_metrics(study)
