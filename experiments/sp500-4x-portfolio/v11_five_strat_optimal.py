"""sp500_four_strat_port — v11: Five strategies, optimal weighting + sector neutral."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v0_equal_weight import (  # noqa: E402
    build_strategies_five,
    load_benchmark,
    load_sector_factors,
    load_sector_map,
    load_universe,
    print_results,
)

from qstudy import PortfolioStudy


def main():
    print("Loading data...")
    universe = load_universe()
    benchmark = load_benchmark()
    sector_factors = load_sector_factors()
    sector_map = load_sector_map()

    print(
        f"Universe: {len(universe.tickers)} tickers | "
        f"{universe.returns.index[0].date()} - {universe.returns.index[-1].date()}"
    )

    strategies = build_strategies_five(sector_factors, sector_map)

    print("\n[v11] Five strategies, optimal weighting + sector neutralization...")
    portfolio = (
        PortfolioStudy(
            strategies=strategies,
            universe=universe,
            benchmark=benchmark,
            name="sp500_five_strat_port_v11_optimal",
        )
        .weight_optimal(window=126, gamma=1.0)
        .neutralize_positions({"sector": 0}, sector_map=sector_map, beta_window=60)
        .run()
    )
    print_results(portfolio, strategies)
    return portfolio


if __name__ == "__main__":
    main()
