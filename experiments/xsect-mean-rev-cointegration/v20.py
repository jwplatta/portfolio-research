from common import build_experiment, emit_metrics

study = build_experiment(
    name="xsect_coint_mr_v20_tighter_universe_equal_vol_slower_signal",
    lookback=252,
    z_window=7,
    use_market=False,
    liquidity_top_n=125,
    n_long=10,
    n_short=10,
    rebalance_every=10,
    equal_vol_window=60,
)

if __name__ == "__main__":
    emit_metrics(study)
