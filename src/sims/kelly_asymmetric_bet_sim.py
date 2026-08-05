"""Simulate repeated asymmetric binary bets with Kelly sizing."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

from tqdm import tqdm


@dataclass(frozen=True)
class SimulationConfig:
    initial_bankroll: float
    win_prob: float
    win_payout: float
    loss_amount: float
    kelly_fraction: float
    kelly_scalar: float
    trades: int | None
    target_bankroll: float | None
    ruin_bankroll: float
    seed: int | None


@dataclass(frozen=True)
class SimulationResult:
    final_bankroll: float
    max_drawdown: float
    wins: int
    losses: int
    trades: int
    kelly_fraction: float
    kelly_scalar: float
    expected_return_per_unit: float
    expected_return_per_dollar_risked: float
    stop_reason: str

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0


def full_kelly_fraction(win_prob: float, win_payout: float, loss_amount: float) -> float:
    """Return the Kelly fraction of bankroll to risk for fixed binary dollar payoffs."""
    loss_prob = 1.0 - win_prob
    edge = (win_prob * win_payout) - (loss_prob * loss_amount)
    return edge / win_payout


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description="Simulate repeated binary bets with asymmetric win and loss payoffs.",
    )
    parser.add_argument("--initial-bankroll", type=positive_float, default=10_000.0)
    parser.add_argument("--win-prob", type=probability, default=0.70)
    parser.add_argument(
        "--win-payout",
        type=positive_float,
        default=1.0,
        help="Dollar profit per risk unit/contract on a winning trade.",
    )
    parser.add_argument(
        "--loss-amount",
        type=positive_float,
        default=3.0,
        help="Dollar loss per risk unit/contract on a losing trade.",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=non_negative_float,
        default=None,
        help="Fraction of bankroll to bet before --kelly-scalar. Defaults to clamped full Kelly.",
    )
    parser.add_argument("--kelly-scalar", type=non_negative_float, default=1.0)
    parser.add_argument(
        "--trades",
        type=non_negative_int,
        default=10_000,
        help="Finite trade count. Use 0 for an unbounded run with target/ruin stopping.",
    )
    parser.add_argument("--target-bankroll", type=positive_float, default=None)
    parser.add_argument("--ruin-bankroll", type=non_negative_float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    computed_kelly = max(full_kelly_fraction(args.win_prob, args.win_payout, args.loss_amount), 0.0)
    kelly_fraction = computed_kelly if args.kelly_fraction is None else args.kelly_fraction

    trades = None if args.trades == 0 else args.trades
    actual_fraction = kelly_fraction * args.kelly_scalar
    if trades is None and args.target_bankroll is None and args.ruin_bankroll <= 0:
        parser.error("unbounded runs require --target-bankroll or a positive --ruin-bankroll")
    if trades is None and actual_fraction == 0 and args.ruin_bankroll <= 0:
        parser.error("unbounded zero-fraction runs require a positive --ruin-bankroll")
    if args.ruin_bankroll >= args.initial_bankroll:
        parser.error("--ruin-bankroll must be below --initial-bankroll")
    if args.target_bankroll is not None and args.target_bankroll <= args.initial_bankroll:
        parser.error("--target-bankroll must be above --initial-bankroll")

    return SimulationConfig(
        initial_bankroll=args.initial_bankroll,
        win_prob=args.win_prob,
        win_payout=args.win_payout,
        loss_amount=args.loss_amount,
        kelly_fraction=kelly_fraction,
        kelly_scalar=args.kelly_scalar,
        trades=trades,
        target_bankroll=args.target_bankroll,
        ruin_bankroll=args.ruin_bankroll,
        seed=args.seed,
    )


def trade_indices(trades: int | None) -> Iterator[int]:
    if trades is None:
        i = 0
        while True:
            yield i
            i += 1
    else:
        yield from range(trades)


def simulate(config: SimulationConfig, show_progress: bool = True) -> SimulationResult:
    rng = random.Random(config.seed)
    bankroll = config.initial_bankroll
    peak = bankroll
    max_drawdown = 0.0
    wins = 0
    losses = 0
    stop_reason = "trade count reached"
    actual_fraction = config.kelly_fraction * config.kelly_scalar

    progress = tqdm(
        trade_indices(config.trades),
        total=config.trades,
        unit="trade",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    with progress:
        for trade_count, _ in enumerate(progress, start=1):
            if actual_fraction > 0:
                dollars_risked = bankroll * actual_fraction
                units = dollars_risked / config.loss_amount
                if rng.random() < config.win_prob:
                    bankroll += units * config.win_payout
                    wins += 1
                else:
                    bankroll -= dollars_risked
                    losses += 1
            else:
                wins += int(rng.random() < config.win_prob)
                losses = trade_count - wins

            peak = max(peak, bankroll)
            drawdown = (peak - bankroll) / peak if peak else 0.0
            max_drawdown = max(max_drawdown, drawdown)

            if show_progress:
                progress.set_postfix(bankroll=f"{bankroll:,.2f}", refresh=False)

            if config.target_bankroll is not None and bankroll >= config.target_bankroll:
                stop_reason = "target bankroll reached"
                break
            if bankroll <= config.ruin_bankroll:
                stop_reason = "ruin threshold reached"
                break
        else:
            trade_count = config.trades or 0

    expected_return_per_unit = (config.win_prob * config.win_payout) - (
        (1.0 - config.win_prob) * config.loss_amount
    )
    expected_return_per_dollar_risked = expected_return_per_unit / config.loss_amount
    return SimulationResult(
        final_bankroll=bankroll,
        max_drawdown=max_drawdown,
        wins=wins,
        losses=losses,
        trades=trade_count,
        kelly_fraction=config.kelly_fraction,
        kelly_scalar=config.kelly_scalar,
        expected_return_per_unit=expected_return_per_unit,
        expected_return_per_dollar_risked=expected_return_per_dollar_risked,
        stop_reason=stop_reason,
    )


def money(value: float) -> str:
    return f"${value:,.2f}"


def percent(value: float) -> str:
    return f"{value:.2%}"


def print_summary(
    config: SimulationConfig,
    result: SimulationResult,
    elapsed_seconds: float,
) -> None:
    total_return = (result.final_bankroll / config.initial_bankroll) - 1.0
    actual_fraction = result.kelly_fraction * result.kelly_scalar

    print()
    print("Asymmetric Kelly Bet Simulation")
    print("-------------------------------")
    print(f"Initial bankroll: {money(config.initial_bankroll)}")
    print(f"Final bankroll:   {money(result.final_bankroll)}")
    print(f"Total return:     {percent(total_return)}")
    print(f"Max drawdown:     {percent(result.max_drawdown)}")
    print(f"Win rate:         {percent(result.win_rate)}")
    print(f"Wins / losses:    {result.wins:,} / {result.losses:,}")
    print(f"Trades:           {result.trades:,}")
    print(f"Kelly fraction:   {percent(result.kelly_fraction)}")
    print(f"Kelly scalar:     {result.kelly_scalar:.4g}")
    print(f"Actual bet frac:  {percent(actual_fraction)}")
    print(f"Expected $/unit:  {result.expected_return_per_unit:.6g}")
    print(f"Expected $/$risk: {result.expected_return_per_dollar_risked:.6g}")
    print(f"Stop reason:      {result.stop_reason}")
    print(f"Elapsed:          {elapsed_seconds:.2f}s")


def main() -> int:
    config = parse_args()
    started_at = time.perf_counter()
    result = simulate(config)
    print_summary(config, result, time.perf_counter() - started_at)
    return 0


if __name__ == "__main__":
    sys.exit(main())
