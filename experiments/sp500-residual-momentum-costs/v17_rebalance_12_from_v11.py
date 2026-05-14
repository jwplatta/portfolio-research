from __future__ import annotations

import json

from shared import build_study, signal_abs_quantile_filter


def run_study() -> dict:
    study = build_study(
        "v17_rebalance_12_from_v11",
        filters=[signal_abs_quantile_filter(min_quantile=0.7)],
        rebalance_every=12,
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
