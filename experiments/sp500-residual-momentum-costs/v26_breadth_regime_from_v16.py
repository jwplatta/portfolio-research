from __future__ import annotations

import json

from shared import dynamic_breadth_regime_filter, build_study, signal_abs_quantile_filter


def run_study() -> dict:
    study = build_study(
        "v26_breadth_regime_from_v16",
        filters=[
            signal_abs_quantile_filter(min_quantile=0.75),
            dynamic_breadth_regime_filter(),
        ],
        rebalance_every=15,
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
