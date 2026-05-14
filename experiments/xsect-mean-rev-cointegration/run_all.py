import importlib.util
import json
from pathlib import Path

import pandas as pd
from common import RESULTS_PATH

HERE = Path(__file__).resolve().parent


def load_study(version):
    path = HERE / f"{version}.py"
    spec = importlib.util.spec_from_file_location(version, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.study


def main():
    versions = [f"v{i}" for i in range(1, 31)]
    rows = []
    for version in versions:
        study = load_study(version)
        metrics = study.metrics_dict()
        metrics["version"] = version
        metrics["strategy_title"] = getattr(study, "_name", version)
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("version")
    df.to_csv(RESULTS_PATH)
    preferred_cols = [
        "strategy_title",
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
    print(f"\nWrote CSV: {RESULTS_PATH}")
    print("\nJSON:")
    print(json.dumps(rows, default=str, indent=2))


if __name__ == "__main__":
    main()
