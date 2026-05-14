from __future__ import annotations

import json

from shared import build_study, sector_relative_transform, signal_abs_quantile_filter


def run_study() -> dict:
    study = build_study(
        "v32_sector_relative_sector_model_from_v0",
        transforms=[sector_relative_transform],
        filters=[signal_abs_quantile_filter(min_quantile=0.75)],
        rebalance_every=15,
        factor_model_factors=["market", "sector"],
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
