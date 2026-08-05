"""Random search for asymmetric binary trade parameters ranked by a chosen metric."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

from kelly_asymmetric_bet_sim import SimulationConfig, full_kelly_fraction, simulate


@dataclass(frozen=True)
class UniformRange:
    low: float
    high: float

    def rvs(self, random_state=None) -> float:
        if random_state is None:
            import random

            return random.uniform(self.low, self.high)
        return float(random_state.uniform(self.low, self.high))


@dataclass(frozen=True)
class CandidateResult:
    win_prob: float
    breakeven_win_prob: float
    edge_over_breakeven: float
    win_payout: float
    loss_amount: float
    payout_loss_ratio: float
    full_kelly: float
    actual_bet_fraction: float
    ev_per_unit: float
    ev_per_dollar_risked: float
    mean_return: float
    median_return: float
    return_std: float
    sharpe: float
    median_final_bankroll: float
    worst_final_bankroll: float
    p95_max_drawdown: float
    mean_max_drawdown: float
    worst_max_drawdown: float
    prob_drawdown_over_20: float

RANK_METRICS = {
    "edge": ("Edge vs BE", lambda result: result.edge_over_breakeven, True, True),
    "edge_per_drawdown": (
        "Edge/P95DD",
        lambda result: result.edge_over_breakeven / result.p95_max_drawdown
        if result.p95_max_drawdown
        else 0.0,
        True,
        False,
    ),
    "sharpe": ("Sharpe", lambda result: result.sharpe, True, False),
    "p95_drawdown": ("P95 DD", lambda result: result.p95_max_drawdown, False, True),
    "median_final_bankroll": (
        "Median $",
        lambda result: result.median_final_bankroll,
        True,
        False,
    ),
    "mean_final_bankroll": (
        "Mean $",
        lambda result: result.mean_return,
        True,
        False,
    ),
    "bet_fraction": ("Bet %", lambda result: result.actual_bet_fraction, False, True),
}


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite value greater than 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite value between 0 and 1")
    return parsed


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot calculate quantile for an empty list")
    ordered = sorted(values)
    index = math.ceil(q * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random-search asymmetric bet parameters and rank by a chosen metric.",
    )
    parser.add_argument("--initial-bankroll", type=positive_float, default=10_000.0)
    parser.add_argument("--trades", type=positive_int, default=1_000)
    parser.add_argument("--samples", type=positive_int, default=250)
    parser.add_argument("--trials", type=positive_int, default=50)
    parser.add_argument("--kelly-scalar", type=probability, default=0.25)
    parser.add_argument(
        "--max-bet-fraction",
        type=probability,
        default=None,
        help="Optional filter on the actual bankroll fraction risked per trade.",
    )
    parser.add_argument("--top", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--rank-by",
        choices=sorted(RANK_METRICS),
        default="edge",
        help="Metric used to rank candidates.",
    )
    parser.add_argument("--min-win-prob", type=probability, default=0.55)
    parser.add_argument("--max-win-prob", type=probability, default=0.95)
    parser.add_argument("--min-win-payout", type=positive_float, required=True)
    parser.add_argument("--max-win-payout", type=positive_float, required=True)
    parser.add_argument("--min-loss-amount", type=positive_float, default=500.0)
    parser.add_argument("--max-loss-amount", type=positive_float, default=5_000.0)
    parser.add_argument("--min-payout-loss-ratio", type=positive_float, default=0.01)
    parser.add_argument("--max-payout-loss-ratio", type=positive_float, default=0.30)
    parser.add_argument(
        "--max-p95-drawdown",
        type=probability,
        default=None,
        help="Optional drawdown filter before ranking.",
    )
    args = parser.parse_args()

    if args.min_win_prob > args.max_win_prob:
        parser.error("--min-win-prob must be less than or equal to --max-win-prob")
    if args.min_win_payout > args.max_win_payout:
        parser.error("--min-win-payout must be less than or equal to --max-win-payout")
    if args.min_loss_amount > args.max_loss_amount:
        parser.error("--min-loss-amount must be less than or equal to --max-loss-amount")
    if args.min_payout_loss_ratio > args.max_payout_loss_ratio:
        parser.error(
            "--min-payout-loss-ratio must be less than or equal to --max-payout-loss-ratio",
        )

    return args


def evaluate_candidate(
    params: dict[str, float],
    args: argparse.Namespace,
) -> CandidateResult | None:
    win_prob = params["win_prob"]
    win_payout = params["win_payout"]
    loss_amount = params["loss_amount"]
    payout_loss_ratio = win_payout / loss_amount

    if not args.min_payout_loss_ratio <= payout_loss_ratio <= args.max_payout_loss_ratio:
        return None

    full_kelly = full_kelly_fraction(win_prob, win_payout, loss_amount)
    if full_kelly <= 0:
        return None
    actual_bet_fraction = full_kelly * args.kelly_scalar
    if args.max_bet_fraction is not None and actual_bet_fraction > args.max_bet_fraction:
        return None

    final_bankrolls = []
    trial_returns = []
    max_drawdowns = []
    for _ in range(args.trials):
        config = SimulationConfig(
            initial_bankroll=args.initial_bankroll,
            win_prob=win_prob,
            win_payout=win_payout,
            loss_amount=loss_amount,
            kelly_fraction=full_kelly,
            kelly_scalar=args.kelly_scalar,
            trades=args.trades,
            target_bankroll=None,
            ruin_bankroll=0.0,
            seed=None,
        )
        result = simulate(config, show_progress=False)
        final_bankrolls.append(result.final_bankroll)
        trial_returns.append((result.final_bankroll / args.initial_bankroll) - 1.0)
        max_drawdowns.append(result.max_drawdown)

    p95_max_drawdown = quantile(max_drawdowns, 0.95)
    if args.max_p95_drawdown is not None and p95_max_drawdown > args.max_p95_drawdown:
        return None

    return_std = statistics.stdev(trial_returns) if len(trial_returns) > 1 else 0.0
    mean_return = statistics.fmean(trial_returns)
    sharpe = mean_return / return_std if return_std else 0.0
    ev_per_unit = (win_prob * win_payout) - ((1.0 - win_prob) * loss_amount)
    breakeven_win_prob = loss_amount / (win_payout + loss_amount)
    drawdowns_over_20 = sum(drawdown > 0.20 for drawdown in max_drawdowns)

    return CandidateResult(
        win_prob=win_prob,
        breakeven_win_prob=breakeven_win_prob,
        edge_over_breakeven=win_prob - breakeven_win_prob,
        win_payout=win_payout,
        loss_amount=loss_amount,
        payout_loss_ratio=payout_loss_ratio,
        full_kelly=full_kelly,
        actual_bet_fraction=actual_bet_fraction,
        ev_per_unit=ev_per_unit,
        ev_per_dollar_risked=ev_per_unit / loss_amount,
        mean_return=mean_return,
        median_return=statistics.median(trial_returns),
        return_std=return_std,
        sharpe=sharpe,
        median_final_bankroll=statistics.median(final_bankrolls),
        worst_final_bankroll=min(final_bankrolls),
        p95_max_drawdown=p95_max_drawdown,
        mean_max_drawdown=statistics.fmean(max_drawdowns),
        worst_max_drawdown=max(max_drawdowns),
        prob_drawdown_over_20=drawdowns_over_20 / len(max_drawdowns),
    )


def rank_label(args) -> str:
    return RANK_METRICS[args.rank_by][0]


def rank_value(result: CandidateResult, args) -> float:
    return RANK_METRICS[args.rank_by][1](result)


def rank_descending(args) -> bool:
    return RANK_METRICS[args.rank_by][2]


def rank_is_percent(args) -> bool:
    return RANK_METRICS[args.rank_by][3]


def format_rank_value(result: CandidateResult, args) -> str:
    value = rank_value(result, args)
    if args.rank_by in {"median_final_bankroll", "mean_final_bankroll"}:
        if args.rank_by == "mean_final_bankroll":
            value = args.initial_bankroll * (1.0 + result.mean_return)
        return f"${value:,.2f}"
    if rank_is_percent(args):
        return f"{value:.2%}"
    return f"{value:.2f}"


def print_results(results: list[CandidateResult], args, elapsed: float) -> None:
    console = Console(width=160)
    console.print()
    console.print("[bold]Asymmetric Kelly Parameter Search[/bold]")
    console.print(f"Rank by: {rank_label(args)}")
    console.print(f"Samples: {args.samples:,}")
    console.print(f"Trials/sample: {args.trials:,}")
    console.print(f"Trades/trial: {args.trades:,}")
    console.print(f"Kelly scalar: {args.kelly_scalar:.2%}")
    if args.max_p95_drawdown is not None:
        console.print(f"Max allowed p95 drawdown: {args.max_p95_drawdown:.2%}")
    console.print(f"Feasible candidates: {len(results):,}")
    console.print(f"Elapsed: {elapsed:.2f}s")

    if not results:
        console.print()
        console.print("No candidates passed the EV and optional drawdown filters.")
        return

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Rank", justify="right")
    show_rank_metric_column = args.rank_by not in {"edge", "p95_drawdown"}
    if show_rank_metric_column:
        table.add_column(rank_label(args), justify="right")
    table.add_column("Edge vs BE", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("BE Win %", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Mean $", justify="right")
    table.add_column("Median $", justify="right")
    table.add_column("P95 DD", justify="right")
    table.add_column("Payout", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Kelly", justify="right")
    table.add_column("Bet %", justify="right")

    for rank, result in enumerate(results[: args.top], start=1):
        row = [
            str(rank),
            f"{result.edge_over_breakeven:.2%}",
            f"{result.win_prob:.2%}",
            f"{result.breakeven_win_prob:.2%}",
            f"{result.sharpe:.2f}",
            f"${args.initial_bankroll * (1.0 + result.mean_return):,.2f}",
            f"${result.median_final_bankroll:,.2f}",
            f"{result.p95_max_drawdown:.2%}",
            f"${result.win_payout:,.2f}",
            f"${result.loss_amount:,.2f}",
            f"{result.payout_loss_ratio:.2%}",
            f"{result.full_kelly:.2%}",
            f"{result.actual_bet_fraction:.2%}",
        ]
        if show_rank_metric_column:
            row.insert(1, format_rank_value(result, args))
        table.add_row(*row)
    console.print()
    console.print(table)


def write_csv(results: list[CandidateResult], args) -> Path:
    output_path = Path("out/kelly-simulation/parameter-search.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "rank_by",
        "rank_value",
        "edge_over_breakeven",
        "win_prob",
        "breakeven_win_prob",
        "sharpe",
        "mean_return",
        "median_return",
        "return_std",
        "median_final_bankroll",
        "worst_final_bankroll",
        "p95_max_drawdown",
        "mean_max_drawdown",
        "worst_max_drawdown",
        "prob_drawdown_over_20",
        "win_payout",
        "loss_amount",
        "payout_loss_ratio",
        "full_kelly",
        "actual_bet_fraction",
        "ev_per_unit",
        "ev_per_dollar_risked",
        "kelly_scalar",
    ]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "rank_by": args.rank_by,
                    "rank_value": rank_value(result, args),
                    "edge_over_breakeven": result.edge_over_breakeven,
                    "win_prob": result.win_prob,
                    "breakeven_win_prob": result.breakeven_win_prob,
                    "sharpe": result.sharpe,
                    "mean_return": result.mean_return,
                    "median_return": result.median_return,
                    "return_std": result.return_std,
                    "median_final_bankroll": result.median_final_bankroll,
                    "worst_final_bankroll": result.worst_final_bankroll,
                    "p95_max_drawdown": result.p95_max_drawdown,
                    "mean_max_drawdown": result.mean_max_drawdown,
                    "worst_max_drawdown": result.worst_max_drawdown,
                    "prob_drawdown_over_20": result.prob_drawdown_over_20,
                    "win_payout": result.win_payout,
                    "loss_amount": result.loss_amount,
                    "payout_loss_ratio": result.payout_loss_ratio,
                    "full_kelly": result.full_kelly,
                    "actual_bet_fraction": result.actual_bet_fraction,
                    "ev_per_unit": result.ev_per_unit,
                    "ev_per_dollar_risked": result.ev_per_dollar_risked,
                    "kelly_scalar": args.kelly_scalar,
                }
            )

    return output_path


def show_plots(results: list[CandidateResult], args) -> Path | None:
    if not results:
        return None

    plot_results = results[:50]
    if rank_is_percent(args):
        y_values = [rank_value(result, args) * 100 for result in plot_results]
        y_label = f"{rank_label(args)} %"
    else:
        y_values = [rank_value(result, args) for result in plot_results]
        y_label = rank_label(args)
    plot_specs = [
        ("Payout", [result.win_payout for result in plot_results]),
        ("Sharpe", [result.sharpe for result in plot_results]),
        ("P95 DD %", [result.p95_max_drawdown * 100 for result in plot_results]),
        ("Bet %", [result.actual_bet_fraction * 100 for result in plot_results]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"{rank_label(args)} Diagnostics: Top 50 Candidates", fontsize=14)

    for axis, (x_label, x_values) in zip(axes.ravel(), plot_specs, strict=True):
        axis.scatter(x_values, y_values, alpha=0.7, s=24)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)

    output_path = Path("out/kelly-simulation/parameter-search-diagnostics.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.show()
    return output_path


def main() -> int:
    args = parse_args()
    if args.max_bet_fraction is None:
        args.max_bet_fraction = 0.02
    started_at = time.perf_counter()
    distributions = {
        "win_prob": UniformRange(args.min_win_prob, args.max_win_prob),
        "win_payout": UniformRange(args.min_win_payout, args.max_win_payout),
        "loss_amount": UniformRange(args.min_loss_amount, args.max_loss_amount),
    }
    sampler = ParameterSampler(
        distributions,
        n_iter=args.samples,
        random_state=args.seed,
    )

    results = []
    for params in tqdm(sampler, total=args.samples, unit="candidate", dynamic_ncols=True):
        result = evaluate_candidate(params, args)
        if result is not None:
            results.append(result)

    results.sort(key=lambda result: rank_value(result, args), reverse=rank_descending(args))
    output_path = write_csv(results, args)
    print_results(results, args, time.perf_counter() - started_at)
    Console().print(f"\nFull results CSV: [bold]{output_path}[/bold]")
    plot_path = show_plots(results, args)
    if plot_path is not None:
        Console().print(f"Diagnostics chart: [bold]{plot_path}[/bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
