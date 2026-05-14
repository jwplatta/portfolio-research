from __future__ import annotations

import json

from shared import build_study


def run_study() -> dict:
    study = build_study(
        "v15_book_25_from_v9",
        n_long=25,
        n_short=25,
        rebalance_every=15,
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
