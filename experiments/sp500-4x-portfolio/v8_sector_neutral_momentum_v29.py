"""sp500_four_strat_port — v8: Equal-vol + sector neutral with momentum_v29 replacing momentum_v19."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v0_equal_weight import (  # noqa: E402
    build_strategies_with_momentum_v29,
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

    strategies = build_strategies_with_momentum_v29(sector_factors, sector_map)

    print("\n[v8] Equal-vol weighting + sector neutralization with momentum_v29...")
    portfolio = (
        PortfolioStudy(
            strategies=strategies,
            universe=universe,
            benchmark=benchmark,
            name="sp500_four_strat_port_v8_sector_neutral_momentum_v29",
        )
        .weight_equal_vol(window=126)
        .neutralize_positions({"sector": 0}, sector_map=sector_map, beta_window=60)
        .run()
    )
    print_results(portfolio, strategies)
    return portfolio


if __name__ == "__main__":
    main()
