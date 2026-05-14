from __future__ import annotations

import json

from shared import build_study, residual_momentum_signal, signal_abs_quantile_filter


def run_study() -> dict:
    study = build_study(
        "v21_signal_60_20_from_v16",
        signal_fn=residual_momentum_signal(lookback=60, skip=20, shift=1),
        filters=[signal_abs_quantile_filter(min_quantile=0.75)],
        rebalance_every=15,
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
