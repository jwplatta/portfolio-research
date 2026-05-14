from __future__ import annotations

import json

from shared import build_study, favorable_residual_regime_filter


def run_study() -> dict:
    study = build_study(
        "v5_residual_regime",
        filters=[favorable_residual_regime_filter()],
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
