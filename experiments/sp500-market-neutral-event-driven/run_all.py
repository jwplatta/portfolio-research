from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results.csv"
VERSIONS = ["v0", "v1", "v2", "v3", "v4", "v5"]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def preload_shared_data():
    common = load_module("common", HERE / "common.py")
    for loader_name in ("load_universe", "load_benchmark", "load_sector_factors", "load_sector_map"):
        loader = getattr(common, loader_name, None)
        if callable(loader):
            loader()
    return common


def load_study(version: str):
    module = load_module(version, HERE / f"{version}.py")
    return module.study


if __name__ == "__main__":
    preload_shared_data()

    rows = []
    for version in VERSIONS:
        study = load_study(version)
        metrics = study.metrics_dict()
        metrics["version"] = version
        rows.append(metrics)
        print(json.dumps(metrics, default=str, sort_keys=True))

    df = pd.DataFrame(rows).set_index("version")
    df.transpose().to_csv(CSV_PATH)
    preferred_cols = [
        "sharpe",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "max_drawdown_duration",
        "avg_daily_turnover",
        "benchmark_corr",
        "information_ratio",
    ]
    existing_cols = [col for col in preferred_cols if col in df.columns]
    print(df[existing_cols].sort_index().round(4).to_string())
    print(f"\nWrote CSV: {CSV_PATH}")
