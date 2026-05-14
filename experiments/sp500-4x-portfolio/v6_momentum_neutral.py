"""sp500_four_strat_port — v6: Equal-vol weighting + portfolio-level momentum neutralization.

Builds on v1 (best weighting) and adds neutralize_positions({"momentum": 0}) on the
combined portfolio. Removes residual momentum factor exposure — particularly relevant
since momentum_v19 is an explicit momentum strategy that may bleed factor loading into
the combined book.

Usage:
    uv run python experiments/sp500-four-strat-port/v6_momentum_neutral.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v0_equal_weight import (  # noqa: E402
    build_strategies,
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

    strategies = build_strategies(sector_factors, sector_map)

    print("\n[v6] Equal-vol weighting + momentum neutralization...")
    portfolio = (
        PortfolioStudy(
            strategies=strategies,
            universe=universe,
            benchmark=benchmark,
            name="sp500_four_strat_port_v6_momentum_neutral",
        )
        .weight_equal_vol(window=126)
        .neutralize_positions({"momentum": 0}, momentum_window=126)
        .run()
    )
    print_results(portfolio, strategies)
    return portfolio


if __name__ == "__main__":
    main()
