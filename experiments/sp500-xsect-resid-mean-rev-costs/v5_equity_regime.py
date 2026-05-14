from __future__ import annotations

import json

from shared import (
    COST_BPS,
    LIQUIDITY_TOP_N,
    LIQUIDITY_WINDOW,
    equity_curve_regime_scale,
    load_benchmark,
    load_factors,
    load_universe,
    residual_mean_reversion_signal,
    study_name,
    zscore_clipped_positions,
)

import qstudy as qs
from qstudy import Study


def run_study() -> dict:
    universe = load_universe()
    benchmark = load_benchmark()
    factors = load_factors()

    study = (
        Study(
            universe=universe,
            benchmark=benchmark,
            factors=factors,
            name=study_name("v5_equity_regime"),
        )
        .residualize_returns()
        .base_signal(residual_mean_reversion_signal(window=5, shift=1))
        .add_vol_filter(vol_window=5, quantile=0.6)
        .add_volume_zscore_filter(window=30, min_zscore_quantile=0.8)
        .add_momentum_context_filter(window=60, max_abs_quantile=0.7)
        .add_tradeable_constraint(qs.liquidity(top_n=LIQUIDITY_TOP_N, window=LIQUIDITY_WINDOW))
        .build_positions(zscore_clipped_positions)
        .scale_risk(equity_curve_regime_scale())
        .with_transaction_costs(cost_bps=COST_BPS)
        .rebalance(every=1)
        .run()
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
