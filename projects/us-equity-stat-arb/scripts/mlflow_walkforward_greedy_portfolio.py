"""
Run the walkforward greedy portfolio validation and record it in MLflow.

This script intentionally keeps the research calculation in
walkforward_greedy_portfolio.py as the source of truth. It runs that workflow,
then logs the generated CSV and chart artifacts plus fold-level train/validation
metrics to MLflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import walkforward_greedy_portfolio as wf

DEFAULT_EXPERIMENT_NAME = "us-equity-stat-arb/walkforward-greedy-portfolio"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and log the US equity stat-arb walkforward portfolio experiment."
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"MLflow experiment name. Default: {DEFAULT_EXPERIMENT_NAME}",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional MLflow parent run name.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI or local server.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only log existing artifacts in the output directory; do not rerun the backtest.",
    )
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log_param_json(key: str, value: Any) -> None:
    mlflow.log_param(key, json.dumps(value, sort_keys=True, default=str))


def log_metric_if_number(key: str, value: Any) -> None:
    if pd.isna(value):
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(numeric):
        mlflow.log_metric(key, numeric)


def metric_name(label: str, prefix: str) -> str:
    label = label.removeprefix("Train ").removeprefix("Validation ")
    normalized = (
        label.lower()
        .replace("(net)", "net")
        .replace("%", "pct")
        .replace(".", "")
        .replace(" ", "_")
    )
    return f"{prefix}_{normalized}"


def log_parent_params() -> None:
    folds = [
        {
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
        }
        for train_start, train_end, val_start, val_end in wf.FOLDS
    ]
    log_param_json("folds", folds)
    log_param_json("weighting_schemes", wf.WEIGHTING_SCHEMES)
    mlflow.log_param("cost_bps", wf.COST_BPS)
    mlflow.log_param("seed_sleeve", wf.SEED_SLEEVE or "")
    mlflow.log_param("max_pairwise_abs_corr", wf.MAX_PAIRWISE_ABS_CORR)
    mlflow.log_param("min_net_sharpe_delta", wf.MIN_NET_SHARPE_DELTA)
    mlflow.log_param("max_dd_regression", wf.MAX_DD_REGRESSION)
    mlflow.log_param("max_turnover_regression", wf.MAX_TURNOVER_REGRESSION)
    mlflow.log_param("min_dd_improvement", wf.MIN_DD_IMPROVEMENT)
    mlflow.log_param("min_turnover_improvement", wf.MIN_TURNOVER_IMPROVEMENT)
    mlflow.log_param("min_sleeves", wf.MIN_SLEEVES)

    commit = git_commit()
    if commit:
        mlflow.set_tag("git_commit", commit)


def log_fold_runs(fold_results_path: Path, weights_path: Path) -> None:
    fold_results = pd.read_csv(fold_results_path)
    weights = pd.read_csv(weights_path)

    for _, row in fold_results.iterrows():
        fold = str(row["Fold"])
        fold_weights = weights.loc[weights["fold"].astype(str) == fold]
        selected_sleeves = fold_weights["sleeve"].tolist()

        with mlflow.start_run(run_name=f"fold-{fold}", nested=True):
            mlflow.set_tag("fold", fold)
            mlflow.log_param("n_selected", len(selected_sleeves))
            mlflow.log_param("portfolio_sleeves", "|".join(selected_sleeves))
            if not fold_weights.empty:
                mlflow.log_param("final_scheme", str(fold_weights["scheme"].iloc[0]))

            for _, weight_row in fold_weights.iterrows():
                sleeve = str(weight_row["sleeve"])
                mlflow.log_metric(f"weight.{sleeve}", float(weight_row["weight"]))

            for column in fold_results.columns:
                if column == "Fold":
                    continue
                prefix = "validation" if column.startswith("Validation") else "train"
                log_metric_if_number(metric_name(column, prefix), row[column])


def log_output_artifacts(out_dir: Path) -> None:
    artifact_names = [
        "walkforward_fold_results.csv",
        "walkforward_sleeve_selection_frequency.csv",
        "walkforward_selected_weights.csv",
        "walkforward_validation_returns.csv",
        "walkforward_validation_returns_stitched.csv",
        "walkforward_selection_detail.csv",
        "walkforward_train_sharpe_buildout.png",
    ]
    for artifact_name in artifact_names:
        path = out_dir / artifact_name
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="walkforward_outputs")


def log_aggregate_metrics(fold_results_path: Path) -> None:
    fold_results = pd.read_csv(fold_results_path)
    aggregate_columns = [
        "Train SR (net)",
        "Train Return",
        "Train Vol",
        "Train Drawdown",
        "Train Turnover",
        "Validation SR (net)",
        "Validation Return",
        "Validation Vol",
        "Validation Drawdown",
        "Validation Turnover",
    ]
    for column in aggregate_columns:
        if column in fold_results:
            period_prefix = "validation" if column.startswith("Validation") else "train"
            log_metric_if_number(
                metric_name(column, f"avg_{period_prefix}"),
                fold_results[column].mean(),
            )
            log_metric_if_number(
                metric_name(column, f"min_{period_prefix}"),
                fold_results[column].min(),
            )


def main() -> None:
    args = parse_args()

    if not args.skip_run:
        wf.main()

    out_dir = wf.OUT_DIR
    fold_results_path = out_dir / "walkforward_fold_results.csv"
    weights_path = out_dir / "walkforward_selected_weights.csv"
    if not fold_results_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            "Expected walkforward output files are missing. "
            f"Need {fold_results_path} and {weights_path}. "
            "Run without --skip-run first."
        )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("project", "us-equity-stat-arb")
        mlflow.set_tag("workflow", "walkforward-greedy-portfolio")
        log_parent_params()
        log_aggregate_metrics(fold_results_path)
        log_output_artifacts(out_dir)
        log_fold_runs(fold_results_path, weights_path)

        print(f"Logged MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
