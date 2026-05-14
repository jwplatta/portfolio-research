from __future__ import annotations

import json

from shared import (
    benchmark_trend_scale,
    blended_raw_sector_signal,
    build_study,
    min_universe_breadth_filter,
    relative_volume_strength_filter,
    signal_abs_quantile_filter,
)


def run_study() -> dict:
    study = build_study(
        "v18",
        signal_fn=blended_raw_sector_signal(
            raw_weight=0.75,
            sector_weight=0.25,
            long_lookback=60,
            short_lookback=20,
            shift=1,
        ),
        filters=[
            signal_abs_quantile_filter(0.55),
            relative_volume_strength_filter(volume_window=63, min_quantile=0.35),
            min_universe_breadth_filter(80),
        ],
        n_long=10,
        n_short=10,
        liquidity_top_n=200,
        rebalance_every=15,
        weighting="equal_vol",
        risk_scalers=[benchmark_trend_scale(fast=100, slow=200, defensive_scale=0.75)],
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
