from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results.csv"
VERSIONS = ["v0"]


def load_study(version: str):
    path = HERE / f"{version}.py"
    spec = importlib.util.spec_from_file_location(version, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.study


if __name__ == "__main__":
    rows = []
    for version in VERSIONS:
        study = load_study(version)
        metrics = study.metrics_dict()
        metrics["version"] = version
        rows.append(metrics)
        print(json.dumps(metrics, default=str, sort_keys=True))

    df = pd.DataFrame(rows).set_index("version")
    df.transpose().to_csv(CSV_PATH)
    print(df.round(4).to_string())
    print(f"\nWrote CSV: {CSV_PATH}")
