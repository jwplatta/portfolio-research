from __future__ import annotations

import json

from shared import build_study, residual_vol_adjusted_signal


def run_study() -> dict:
    study = build_study(
        "v2_vol_adjusted_signal",
        signal_fn=residual_vol_adjusted_signal(lookback=30, skip=20, shift=1),
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
