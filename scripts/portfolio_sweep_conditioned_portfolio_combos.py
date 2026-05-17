"""Exhaustive portfolio-combo sweep from strong conditioned 2-strategy portfolios.

Workflow:
1. Read ``portfolio_sweep_conditioned_winner_pairs.csv``.
2. Keep the 2-strategy portfolios whose ``net_sharpe`` exceeds a threshold.
3. Rebuild and pre-run each unique underlying conditioned sleeve once, with no
   strategy-level costs.
4. For every ordered pair of selected 2-strategy portfolios, combine their
   underlying sleeves, remove duplicate strategies, and run a new
   ``PortfolioStudy`` with equal weights and portfolio-level transaction costs.

Usage:
    uv run python scripts/portfolio_sweep_conditioned_portfolio_combos.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from qstudy.study.PortfolioStudy import PortfolioStudy

import signal_sweep_conditioned_winners as scw

PAIR_PATH = Path("portfolio_sweep_conditioned_winner_pairs.csv")
TOP25_PATH = Path("top25_conditioned_winners.csv")
OUTPUT_PATH = Path("portfolio_sweep_conditioned_portfolio_combos.csv")
SELECTED_PAIRS_PATH = Path("selected_conditioned_winner_pairs_over_1.csv")

MIN_NET_SHARPE = 1.35
COST_BPS = 10.0
PORTFOLIO_VERBOSE = False


def load_selected_pair_ports(min_net_sharpe: float | None = None) -> pd.DataFrame:
    if min_net_sharpe is None:
        min_net_sharpe = MIN_NET_SHARPE
    if not PAIR_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PAIR_PATH}. Run scripts/portfolio_sweep_conditioned_winner_pairs.py first."
        )

    df = pd.read_csv(PAIR_PATH)
    if "net_sharpe" not in df.columns and "sharpe" in df.columns:
        df["net_sharpe"] = df["sharpe"]

    selected = df[df["net_sharpe"] > min_net_sharpe].copy().reset_index(drop=True)
    selected.to_csv(SELECTED_PAIRS_PATH, index=False)
    return selected


def load_sleeve_lookup() -> dict[str, dict[str, object]]:
    if not TOP25_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TOP25_PATH}. Run scripts/signal_sweep_conditioned_winners.py first."
        )

    df = pd.read_csv(TOP25_PATH)
    return {
        str(row["name"]): {
            "name": str(row["name"]),
            "base_label": str(row["base_label"]),
            "conditioning_filter": str(row["conditioning_filter"]),
            "group": str(row["group"]),
            "rebalance": int(row["rebalance"]),
            "legacy_scaler": str(row["legacy_scaler"]),
        }
        for _, row in df.iterrows()
    }


def ordered_unique(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def run_unique_sleeves(selected_pairs_df: pd.DataFrame) -> dict[str, object]:
    sleeve_lookup = load_sleeve_lookup()
    strategy_lookup = {str(item["label"]): item for item in scw.TOP_STRATEGIES}
    filter_lookup = {item["tag"]: item for item in scw.make_conditioning_filters()}

    labels = ordered_unique(
        selected_pairs_df["strategy_a"].astype(str).tolist()
        + selected_pairs_df["strategy_b"].astype(str).tolist()
    )

    studies: dict[str, object] = {}
    for label in labels:
        if label not in sleeve_lookup:
            raise KeyError(
                f"Missing sleeve metadata for {label} in {TOP25_PATH}. "
                "Re-run scripts/signal_sweep_conditioned_winners.py first."
            )
        meta = sleeve_lookup[label]
        strategy = strategy_lookup[meta["base_label"]]
        conditioning_filter = filter_lookup[meta["conditioning_filter"]]
        print(f"Running sleeve {label} ...", end=" ", flush=True)
        studies[label] = scw._run_study(strategy, conditioning_filter, include_strategy_costs=False)
        print("ok")

    return studies


def run_combo_portfolio(
    pair_1: pd.Series,
    pair_2: pd.Series,
    studies: dict[str, object],
) -> dict[str, object] | None:
    all_labels = ordered_unique(
        [
            str(pair_1["strategy_a"]),
            str(pair_1["strategy_b"]),
            str(pair_2["strategy_a"]),
            str(pair_2["strategy_b"]),
        ]
    )

    universe = scw.load_universe()
    benchmark = scw.load_benchmark()

    try:
        portfolio = (
            PortfolioStudy(
                strategies=[studies[label] for label in all_labels],
                universe=universe,
                benchmark=benchmark,
                name=f"{pair_1['pair']} || {pair_2['pair']}",
                cost_bps=COST_BPS,
                verbose=PORTFOLIO_VERBOSE,
            )
            .weight_equal()
            .with_transaction_costs(COST_BPS)
            .run()
        )
    except Exception as exc:
        print(f"    [SKIP] {pair_1['pair']} || {pair_2['pair']}: {exc}")
        return None

    result = portfolio.metrics_dict()
    result["portfolio_1"] = str(pair_1["pair"])
    result["portfolio_2"] = str(pair_2["pair"])
    result["combined_portfolio"] = f"{pair_1['pair']} || {pair_2['pair']}"
    result["weighting"] = "equal"
    result["strategy_labels"] = " | ".join(all_labels)
    result["n_unique_strategies"] = len(all_labels)
    result["p1_net_sharpe"] = float(pair_1["net_sharpe"])
    result["p2_net_sharpe"] = float(pair_2["net_sharpe"])
    return result


def main() -> pd.DataFrame | None:
    selected_pairs_df = load_selected_pair_ports(min_net_sharpe=MIN_NET_SHARPE)
    print(
        f"\nSelected {len(selected_pairs_df)} two-strategy portfolios with "
        f"net_sharpe > {MIN_NET_SHARPE:.2f}"
    )
    print(
        selected_pairs_df[["pair", "strategy_a", "strategy_b", "net_sharpe"]]
        .head(20)
        .to_string(index=False, float_format="{:.4f}".format)
    )
    print(f"\nSaved selected source portfolios to {SELECTED_PAIRS_PATH}")

    print("\nPre-running unique conditioned sleeves ...")
    studies = run_unique_sleeves(selected_pairs_df)

    total = len(selected_pairs_df) * len(selected_pairs_df)
    print(f"\nRunning {total} ordered portfolio-combo configurations ...")

    results: list[dict[str, object]] = []
    idx = 0
    for _, pair_1 in selected_pairs_df.iterrows():
        for _, pair_2 in selected_pairs_df.iterrows():
            idx += 1
            label = f"{pair_1['pair']} || {pair_2['pair']}"
            print(f"[{idx:>6}/{total}] {label} ...", end=" ", flush=True)
            result = run_combo_portfolio(pair_1, pair_2, studies)
            if result is None:
                print("skipped")
                continue
            results.append(result)
            sharpe = result.get("net_sharpe", result.get("sharpe", float("nan")))
            print(f"net_sharpe={sharpe:.3f}")

    if not results:
        print("No portfolio-combo results.")
        return None

    result_df = pd.DataFrame(results).sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    display_cols = [
        "combined_portfolio",
        "n_unique_strategies",
        "net_sharpe",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "avg_daily_turnover",
        "benchmark_corr",
        "information_ratio",
        "p1_net_sharpe",
        "p2_net_sharpe",
    ]
    available = [col for col in display_cols if col in result_df.columns]

    print("\n" + "=" * 140)
    print(f"{'Conditioned Portfolio Combo Sweep Results':^140}")
    print("=" * 140)
    print(result_df[available].head(20).to_string(index=False, float_format="{:.4f}".format))
    print("=" * 140)
    print(f"Saved portfolio combo results to {OUTPUT_PATH}")
    return result_df


if __name__ == "__main__":
    main()
