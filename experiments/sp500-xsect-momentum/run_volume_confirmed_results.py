import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results.csv"


def load_study(version: str):
    path = HERE / f"{version}.py"
    spec = importlib.util.spec_from_file_location(version, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.study


def main():
    versions = [f"v{i}_volume_confirmed" for i in range(1, 20)]
    rows = []
    for version in versions:
        study = load_study(version)
        metrics = study.metrics_dict()
        metrics["version"] = version
        rows.append(metrics)

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
    print("\nJSON:")
    print(json.dumps(rows, default=str, indent=2))


if __name__ == "__main__":
    main()
