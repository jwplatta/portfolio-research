"""sp500_four_strat_port — v2: Equal-Sharpe sleeve weighting.

Weights each strategy proportionally to its trailing 126-day rolling Sharpe ratio
(floored at zero so poor performers don't go short).

Usage:
    uv run python experiments/sp500-four-strat-port/v2_equal_sharpe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v0_equal_weight import (  # noqa: E402
    build_strategies,
    load_benchmark,
    load_sector_factors,
    load_sector_map,
    load_universe,
    print_results,
)

import qstudy as qs
from qstudy import PortfolioStudy


def main():
    print("Loading data...")
    universe = load_universe()
    benchmark = load_benchmark()
    sector_factors = load_sector_factors()
    sector_map = load_sector_map()

    print(
        f"Universe: {len(universe.tickers)} tickers | "
        f"{universe.returns.index[0].date()} – {universe.returns.index[-1].date()}"
    )

    strategies = build_strategies(sector_factors, sector_map)

    print("\n[v2] Equal-Sharpe sleeve weighting (window=126)...")
    portfolio = (
        PortfolioStudy(
            strategies=strategies,
            universe=universe,
            benchmark=benchmark,
            name="sp500_four_strat_port_v2_equal_sharpe",
        )
        .weight_equal_sharpe(window=126)
        .run()
    )
    print_results(portfolio, strategies)
    return portfolio


if __name__ == "__main__":
    main()
