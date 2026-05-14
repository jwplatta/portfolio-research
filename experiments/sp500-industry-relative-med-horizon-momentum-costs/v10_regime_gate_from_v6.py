from __future__ import annotations

import json

from shared import (
    benchmark_trend_gate_filter,
    build_study,
    min_universe_breadth_filter,
    relative_volume_strength_filter,
    sector_relative_transform,
    signal_abs_quantile_filter,
)


def run_study() -> dict:
    study = build_study(
        "v10",
        transforms=[sector_relative_transform],
        filters=[
            signal_abs_quantile_filter(0.55),
            relative_volume_strength_filter(volume_window=63, min_quantile=0.35),
            min_universe_breadth_filter(80),
            benchmark_trend_gate_filter(fast=100, slow=200),
        ],
        n_long=10,
        n_short=10,
        liquidity_top_n=200,
        rebalance_every=15,
        weighting="equal_vol",
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
