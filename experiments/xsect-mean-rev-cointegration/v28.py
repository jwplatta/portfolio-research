from common import build_experiment, emit_metrics

study = build_experiment(
    name="xsect_coint_mr_v28_shorter_residual_window_hits_targets",
    signal_family="residual",
    study_residualize=False,
    residual_beta_window=126,
    residual_signal_window=8,
    residual_signal_mode="returns",
    add_sector_demean=True,
    liquidity_top_n=150,
    volume_quantile=0.8,
    n_long=10,
    n_short=10,
    rebalance_every=10,
    equal_vol_window=60,
    beta_neutral_window=20,
)

if __name__ == "__main__":
    emit_metrics(study)
