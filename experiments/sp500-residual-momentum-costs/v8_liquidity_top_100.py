from __future__ import annotations

import json

from shared import build_study


def run_study() -> dict:
    study = build_study(
        "v8_liquidity_top_100",
        liquidity_top_n=100,
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
