from __future__ import annotations

import json

from shared import build_study


def run_study() -> dict:
    study = build_study(
        "v14_liquidity_top_125_from_v9",
        liquidity_top_n=125,
        rebalance_every=15,
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
