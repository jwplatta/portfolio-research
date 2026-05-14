"""Run all sp500-four-strat-port versions and write results.csv.

Each version script exposes a main() that returns the PortfolioStudy.
We call main() for each version, collect metrics_dict(), and write the CSV.

Usage:
    uv run python experiments/sp500-four-strat-port/run_all.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results.csv"

VERSIONS = [
    "v0_equal_weight",
    "v1_equal_vol",
    "v2_equal_sharpe",
    "v3_optimal",
    "v4_beta_neutral",
    "v5_sector_neutral",
    "v6_momentum_neutral",
    "v7_equal_vol_momentum_v29",
    "v8_sector_neutral_momentum_v29",
    "v9_sector_neutral_no_event_momentum_v29",
]


def load_version(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    rows = []
    for version in VERSIONS:
        print(f"\n{'='*60}")
        print(f"Running {version}...")
        print("=" * 60)
        module = load_version(version)
        portfolio = module.main()
        metrics = portfolio.metrics_dict()
        metrics["version"] = version
        rows.append(metrics)
        print(json.dumps(metrics, default=str, sort_keys=True))

    df = pd.DataFrame(rows).set_index("version")
    df.to_csv(CSV_PATH)

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
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df[existing_cols].sort_index().round(4).to_string())
    print(f"\nWrote CSV: {CSV_PATH}")
    print("\nJSON:")
    print(json.dumps(rows, default=str, indent=2))
