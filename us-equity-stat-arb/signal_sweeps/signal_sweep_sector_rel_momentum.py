"""Signal sweep: sector-relative momentum signals.

Signals: sector_rel_mom, sector_rel_sharpe, sector_rank_mom
         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_sector_rel_momentum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import qstudy as qs

sys.path.insert(0, str(Path(__file__).parent.parent))
from portfolio_utils import make_equity_curve_regime_scale

from signal_sweep_utils import (
    TRAIN_START,
    COST_BPS,
    N_LONG,
    N_SHORT,
    REBALANCE_PERIODS,
    run_sweep,
)
from sweep_scalers import SCALER_PRESETS_MOM, apply_scalers

GROUP = "sector-rel-momentum"
OUT_DIR = Path(__file__).resolve().parent / "out" / "sector-rel-momentum"

# GICS sector name → sector ETF ticker
GICS_TO_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLK",
}


# ---------------------------------------------------------------------------
# Sector returns helper
# ---------------------------------------------------------------------------

def _build_sector_returns(r: pd.DataFrame, factor_returns: pd.DataFrame, sector_etf_map: dict) -> pd.DataFrame:
    """Broadcast each stock's own-sector ETF return into a stock-shaped DataFrame."""
    sector_df = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for ticker in r.columns:
        etf = sector_etf_map.get(ticker, "SPY")
        sector_df[ticker] = (
            factor_returns[etf].reindex(r.index).fillna(0.0)
            if etf in factor_returns.columns
            else 0.0
        )
    return sector_df


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def make_signals(sector_etf_map: dict[str, str]) -> list[dict]:
    signals = []

    # Sector-relative momentum: stock mom - sector mom
    for w in [20, 40, 60, 120, 252]:
        for skip in [0, 5, 21]:
            window, skip_ = w, skip

            def _sector_rel_mom(window=window, skip=skip_, sector_etf_map=sector_etf_map, **cache):
                r = cache["residual_returns"]
                sector_r = _build_sector_returns(r, cache["factor_returns"], sector_etf_map or {})
                stock_mom = r.shift(skip).rolling(window).mean()
                sector_mom = sector_r.shift(skip).rolling(window).mean()
                return stock_mom - sector_mom

            suffix = "" if skip_ == 0 else f"_skip{skip_}"
            name = f"sector_rel_mom_{window}d{suffix}"
            _sector_rel_mom.__name__ = name
            signals.append({"name": name, "use_residual": True, "fn": _sector_rel_mom})

    # Sharpe-normalized: (stock - sector) / rolling std of the spread
    for w in [60, 120, 252]:
        for skip in [0, 5]:
            window, skip_ = w, skip

            def _sector_rel_sharpe(window=window, skip=skip_, sector_etf_map=sector_etf_map, **cache):
                r = cache["residual_returns"]
                sector_r = _build_sector_returns(r, cache["factor_returns"], sector_etf_map or {})
                spread = r - sector_r
                mu = spread.shift(skip).rolling(window).mean()
                sigma = spread.shift(skip).rolling(window).std().clip(lower=1e-8)
                return mu / sigma

            name = (
                f"sector_rel_sharpe_{window}d"
                if skip_ == 0
                else f"sector_rel_sharpe_{window}d_skip{skip_}"
            )
            _sector_rel_sharpe.__name__ = name
            signals.append({"name": name, "use_residual": True, "fn": _sector_rel_sharpe})

    # Cross-sectional rank within sector
    for w in [60, 120]:
        window = w

        def _sector_rank_mom(window=window, sector_etf_map=sector_etf_map, **cache):
            r = cache["residual_returns"]
            mom = r.rolling(window).mean()
            ranked = mom.copy()
            sectors: dict[str, list] = {}
            for ticker in r.columns:
                etf = sector_etf_map.get(ticker, "SPY")
                sectors.setdefault(etf, []).append(ticker)
            for tickers in sectors.values():
                ranked[tickers] = mom[tickers].rank(axis=1, pct=True)
            return ranked

        _sector_rank_mom.__name__ = f"sector_rank_mom_{window}d"
        signals.append({"name": f"sector_rank_mom_{window}d", "use_residual": True, "fn": _sector_rank_mom})

    return signals


# ---------------------------------------------------------------------------
# Scaler configs — momentum-style (full in uptrend, scale down otherwise)
# ---------------------------------------------------------------------------

def make_scaler_configs() -> list[dict]:
    return SCALER_PRESETS_MOM


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------

def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    # Sector-relative signals close over sector_etf_map derived from the live universe,
    # so we re-derive the map and signal fn here rather than using build_study_generic.
    sector_map = qs.get_sector_map(list(universe.returns.columns))
    sector_etf_map = {t: GICS_TO_ETF.get(s, "SPY") for t, s in sector_map.items()}
    signals = make_signals(sector_etf_map)
    entry_fn_orig = next(e["fn"] for e in signals if e["name"] == entry["name"])

    def fn(**cache):
        return entry_fn_orig(**cache)

    fn.__name__ = entry_fn_orig.__name__
    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)

    builder = (
        qs.Study(universe=universe, benchmark=benchmark, factors=factors, verbose=verbose)
        .residualize_returns(fit_start=TRAIN_START)
        .base_signal(fn)
        .add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=equity_curve_scaler)
    )
    builder = apply_scalers(builder, scaler_cfg)
    return builder.rebalance(every=rebalance).with_transaction_costs(cost_bps=COST_BPS).run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Build a representative sector_etf_map for signal enumeration; the actual
    # map used per-run is recomputed inside build_study_fn from the live universe.
    from signal_sweep_utils import load_data
    universe, _, _ = load_data()
    sector_map = qs.get_sector_map(list(universe.returns.columns))
    sector_etf_map = {t: GICS_TO_ETF.get(s, "SPY") for t, s in sector_map.items()}

    run_sweep(
        group=GROUP,
        signals=make_signals(sector_etf_map),
        scaler_configs=make_scaler_configs(),
        rebalance_periods=REBALANCE_PERIODS,
        build_study_fn=build_study_fn,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
