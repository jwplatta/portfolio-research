from __future__ import annotations

import json

from shared import build_study, sector_relative_transform


def run_study() -> dict:
    study = build_study(
        "v7_sector_relative",
        transforms=[sector_relative_transform],
    )
    return study.metrics_dict()


if __name__ == "__main__":
    print(json.dumps(run_study(), default=str, indent=2, sort_keys=True))
