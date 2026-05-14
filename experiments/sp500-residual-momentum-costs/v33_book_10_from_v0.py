from __future__ import annotations

import json

from shared import build_study, signal_abs_quantile_filter


def run_study() -> dict:
    study = build_study(
        "v33_book_10_from_v0",
        filters=[signal_abs_quantile_filter(min_quantile=0.75)],
        n_long=10,
        n_short=10,
        rebalance_every=15,
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
