"""Signal sweep: sector-relative signals (MR-style).

Signals: sector_rel_mr, sector_rel_mom, sector_rel_zscore
         train 2015-2021 val 2022, train 2015-2022 val 2023

Usage:
    uv run python examples/signal_sweeps/signal_sweep_sector_relative.py
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
from sweep_scalers import SCALER_PRESETS_MR, apply_scalers

GROUP = "sector-relative"
OUT_DIR = Path(__file__).resolve().parent / "out" / "sector-relative"

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

    # MR on (stock - sector_etf)
    for w in [5, 20, 60]:
        window = w

        def _sector_rel_mr(window=window, sector_etf_map=sector_etf_map, **cache):
            r = cache["_active_returns"]
            sector_r = _build_sector_returns(r, cache["factor_returns"], sector_etf_map or {})
            return -(r - sector_r).rolling(window).mean()

        _sector_rel_mr.__name__ = f"sector_rel_mr_{window}d"
        signals.append({"name": f"sector_rel_mr_{window}d", "use_residual": False, "fn": _sector_rel_mr})

    # Momentum on spread
    for w in [60, 120]:
        window = w

        def _sector_rel_mom(window=window, sector_etf_map=sector_etf_map, **cache):
            r = cache["_active_returns"]
            sector_r = _build_sector_returns(r, cache["factor_returns"], sector_etf_map or {})
            return (r - sector_r).rolling(window).mean()

        _sector_rel_mom.__name__ = f"sector_rel_mom_{window}d"
        signals.append({"name": f"sector_rel_mom_{window}d", "use_residual": False, "fn": _sector_rel_mom})

    # Zscore of 5-day spread vs 60-day baseline
    def _sector_rel_zscore(sector_etf_map=sector_etf_map, **cache):
        r = cache["_active_returns"]
        sector_r = _build_sector_returns(r, cache["factor_returns"], sector_etf_map or {})
        spread = r - sector_r
        mu5 = spread.rolling(5).mean()
        mu60 = spread.rolling(60).mean()
        sigma60 = spread.rolling(60).std().clip(lower=1e-8)
        return -((mu5 - mu60) / sigma60)

    _sector_rel_zscore.__name__ = "sector_rel_zscore_5_60"
    signals.append({"name": "sector_rel_zscore_5_60", "use_residual": False, "fn": _sector_rel_zscore})

    return signals


# ---------------------------------------------------------------------------
# Scaler configs — MR-style (scale down in uptrend)
# ---------------------------------------------------------------------------

def make_scaler_configs() -> list[dict]:
    return SCALER_PRESETS_MR


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------

def build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False):
    # Re-derive sector_etf_map and signal fn here because sector-relative signals
    # close over sector_etf_map which varies per universe call.
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
