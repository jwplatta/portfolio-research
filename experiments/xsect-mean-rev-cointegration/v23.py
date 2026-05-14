from common import build_experiment, emit_metrics

study = build_experiment(
    name="xsect_coint_mr_v23_tighter_universe_vol_target_12",
    lookback=252,
    z_window=6,
    use_market=False,
    liquidity_top_n=125,
    n_long=10,
    n_short=10,
    rebalance_every=10,
    vol_target=0.12,
    benchmark_regime={"fast": 100, "slow": 200, "defensive_scale": 0.7},
)

if __name__ == "__main__":
    emit_metrics(study)
