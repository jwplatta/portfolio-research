"""sp500_four_strat_port — v12: Turnover reduction experiments at 10bps.

Baseline: four strategies (no event_driven_v26), equal-vol + sector neutral, 10bps costs.
    resid_mr_v10:              rebalance(every=1)  <- dominant turnover source
    short_horizon_momentum_w20: rebalance(every=5)
    momentum_v29:              rebalance(every=10)
    coint_mr_v28:              rebalance(every=10)

Variants tested (all at 10bps):
    v12a — baseline: resid_mr every=1,  shm every=5   (current notebook state)
    v12b — resid_mr every=3,  shm every=5
    v12c — resid_mr every=5,  shm every=5
    v12d — resid_mr every=5,  shm every=10
    v12e — resid_mr every=3,  shm every=10
    v12f — resid_mr every=1,  shm every=10  (isolate shm contribution)

Usage:
    uv run python experiments/sp500-four-strat-port/v12_turnover_reduction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v0_equal_weight import (  # noqa: E402
    build_coint_mr_v28,
    build_momentum_v29,
    build_resid_mr_v10,
    build_short_horizon_momentum_w20,
    load_benchmark,
    load_sector_factors,
    load_sector_map,
    load_universe,
)

from qstudy import PortfolioStudy
from qstudy.study.metrics import turnover

COST_BPS = 10
PORTFOLIO_NAME = "sp500_four_strat_port_v12_turnover"


def build_portfolio(
    resid_mr_every: int,
    shm_every: int,
    sector_factors,
    sector_map,
    universe,
    benchmark,
    label: str,
) -> PortfolioStudy:
    """Build and run the four-strategy portfolio with the given rebalance schedules."""
    mr = build_resid_mr_v10(sector_factors)
    shm = build_short_horizon_momentum_w20(sector_map)
    mom = build_momentum_v29(sector_factors)
    coint = build_coint_mr_v28(sector_factors)

    # Patch rebalance schedules — replace the existing rebalance step by appending
    # a new one (the pipeline accepts the last rebalance(every=N) as the effective one
    # since rebalance() appends a position_scaler step and all scalers run in order).
    # Simpler: rebuild with explicit every= by accessing _steps directly is fragile,
    # so we use .rebalance() which always appends — but resid_mr_v10 already has
    # rebalance(every=1) baked in. We need to reconstruct with the desired schedule.
    #
    # Clean approach: rebuild the studies with patched rebalance values.
    mr = _build_resid_mr_patched(sector_factors, every=resid_mr_every)
    shm = _build_shm_patched(sector_map, every=shm_every)

    strategies = [mom, mr, coint, shm]

    portfolio = (
        PortfolioStudy(
            strategies=strategies,
            universe=universe,
            benchmark=benchmark,
            name=f"{PORTFOLIO_NAME}_{label}",
        )
        .weight_equal_vol(window=30)
        .neutralize_positions({"sector": 0}, sector_map=sector_map, beta_window=60)
        .with_transaction_costs(COST_BPS)
        .run()
    )
    return portfolio, strategies


def _build_resid_mr_patched(sector_factors, every: int):
    """resid_mr_v10 with a configurable rebalance schedule."""
    import numpy as np

    from qstudy import Study
    import qstudy as qs

    def residual_mean_reversion_signal(**cache):
        return -cache["residual_returns"].rolling(5).mean().shift(1)

    def demean_signal(signal, **cache):
        return signal.sub(signal.mean(axis=1), axis=0)

    def proportional_positions(signal, **cache):
        z = signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1), axis=0).clip(-3, 3)
        z = z.sub(z.mean(axis=1), axis=0)
        gross = z.abs().sum(axis=1).replace(0.0, float("nan"))
        return z.div(gross, axis=0)

    def equity_curve_regime_scale(positions, **cache):
        import numpy as np
        returns = cache["returns"]
        mask = cache.get("_tradeable_mask")
        if mask is None:
            mask = cache.get("_liquidity_mask")
        if mask is not None:
            returns = returns.where(mask)
        raw_ret = (positions.shift(1) * returns).sum(axis=1)
        equity = (1 + raw_ret).cumprod()
        scale = pd.Series(
            np.where(equity > equity.rolling(20).mean(), 1.0, 0.25), index=equity.index
        )
        return positions.mul(scale.shift(1), axis=0)

    def benchmark_regime_scale(positions, **cache):
        import numpy as np
        price = (1 + cache["benchmark"].fillna(0.0)).cumprod()
        scale = pd.Series(
            np.where(price.rolling(150).mean() >= price.rolling(250).mean(), 1.0, 0.75),
            index=price.index,
        )
        return positions.mul(scale.shift(1), axis=0)

    equity_curve_regime_scale.__name__ = "equity_curve_regime_scale"
    benchmark_regime_scale.__name__ = "benchmark_regime_scale"

    return (
        Study(name=f"resid_mr_v10_rb{every}", factors=sector_factors)
        .residualize_returns()
        .base_signal(residual_mean_reversion_signal)
        .transform_signal(demean_signal)
        .add_vol_filter(vol_window=5, quantile=0.6)
        .add_volume_zscore_filter(window=30, min_zscore_quantile=0.8)
        .add_momentum_context_filter(window=60, max_abs_quantile=0.7)
        .add_tradeable_constraint(qs.liquidity(top_n=250, window=60))
        .build_positions(proportional_positions)
        .scale_risk(benchmark_regime_scale)
        .scale_risk(equity_curve_regime_scale)
        .rebalance(every=every)
    )


def _build_shm_patched(sector_map: dict, every: int):
    """short_horizon_momentum_w20 with a configurable rebalance schedule."""
    import numpy as np

    from qstudy import Study
    import qstudy as qs

    def short_horizon_sector_relative_signal(window: int = 20, skip: int = 0):
        def signal_fn(**cache):
            returns = cache["returns"]
            sector_map_local = pd.Series(sector_map).reindex(returns.columns).fillna("Unknown")
            stock_ret = returns.rolling(window).sum().shift(skip + 1)
            sector_ret = stock_ret.copy()
            for s, tickers in sector_map_local.groupby(sector_map_local):
                cols = tickers.index.tolist()
                sector_ret.loc[:, cols] = stock_ret.loc[:, cols].mean(axis=1).values[:, None]
            return stock_ret - sector_ret

        signal_fn.__name__ = f"short_horizon_sector_relative_signal_{window}_{skip}"
        return signal_fn

    def sector_beta_neutralize_positions(window: int = 60, passes: int = 2):
        def scaler(positions, **cache):
            returns = cache["returns"]
            benchmark = cache["benchmark"].reindex(returns.index).fillna(0.0)
            mean_r = returns.rolling(window).mean()
            mean_b = benchmark.rolling(window).mean()
            cov = returns.mul(benchmark, axis=0).rolling(window).mean().sub(
                mean_r.mul(mean_b, axis=0)
            )
            var_b = benchmark.rolling(window).var().replace(0.0, float("nan"))
            betas = cov.div(var_b, axis=0).shift(1)

            adjusted = positions.copy()
            for date in positions.index:
                active = positions.loc[date]
                active = active[active != 0.0]
                if active.empty:
                    continue
                weights = active.copy()
                for _ in range(passes):
                    sectors = pd.Series(
                        {t: sector_map.get(t, "Unknown") for t in weights.index}
                    )
                    for s, tickers in sectors.groupby(sectors):
                        cols = tickers.index.tolist()
                        sw = weights.loc[cols]
                        weights.loc[cols] = sw - sw.mean()
                    beta_slice = betas.loc[date, weights.index].dropna()
                    if len(beta_slice) >= 2:
                        w = weights.reindex(beta_slice.index).fillna(0.0)
                        beta_exp = float((w * beta_slice).sum())
                        beta_norm = float((beta_slice**2).sum())
                        if beta_norm > 0.0:
                            weights.loc[beta_slice.index] = (
                                w - (beta_exp / beta_norm) * beta_slice
                            )
                weights = weights - weights.mean()
                gross = weights.abs().sum()
                if gross > 0.0 and not pd.isna(gross):
                    adjusted.loc[date, weights.index] = weights / gross

            return adjusted.fillna(0.0)

        scaler.__name__ = f"sector_beta_neutralize_positions_{window}_{passes}"
        return scaler

    return (
        Study(name=f"shm_w20_rb{every}")
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(5.0))
        .add_tradeable_constraint(qs.min_adv(20_000_000.0))
        .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal()
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance(every=every)
    )


def print_variant_results(label, portfolio, strategies):
    s = portfolio.cache["metrics_summary"]
    port_to = s["avg_daily_turnover"]
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  Net  Sharpe:     {s['sharpe']:+.2f}   Gross Sharpe: {s.get('gross_sharpe', float('nan')):+.2f}")
    print(f"  Net  Ann Ret:    {s['ann_return']:+.1%}  Gross Ann Ret: {s.get('gross_ann_return', float('nan')):+.1%}")
    print(f"  Ann Vol:         {s['ann_vol']:.1%}")
    print(f"  Max DD:          {s['max_drawdown']:.1%}  ({s['max_drawdown_duration']} days)")
    print(f"  Portfolio TO:    {port_to:.3f}  → {port_to * COST_BPS / 10_000 * 252:.1%} drag")
    print()
    print("  Per-strategy turnover:")
    for study in strategies:
        to = turnover(study.cache["positions"]).mean()
        drag = to * COST_BPS / 10_000 * 252
        print(f"    {study._name:35s}  TO={to:.3f}  drag={drag:.1%}")


VARIANTS = [
    # (label,          resid_mr_every, shm_every)
    ("v12a  baseline   mr=1  shm=5",   1,  5),
    ("v12b             mr=3  shm=5",   3,  5),
    ("v12c             mr=5  shm=5",   5,  5),
    ("v12d             mr=5  shm=10",  5, 10),
    ("v12e             mr=3  shm=10",  3, 10),
    ("v12f  isolate mr mr=1  shm=10",  1, 10),
]


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

    rows = []
    for label, mr_every, shm_every in VARIANTS:
        print(f"\nRunning {label}...")
        portfolio, strategies = build_portfolio(
            resid_mr_every=mr_every,
            shm_every=shm_every,
            sector_factors=sector_factors,
            sector_map=sector_map,
            universe=universe,
            benchmark=benchmark,
            label=label.split()[0],
        )
        print_variant_results(label, portfolio, strategies)
        s = portfolio.cache["metrics_summary"]
        rows.append({
            "variant": label,
            "resid_mr_every": mr_every,
            "shm_every": shm_every,
            "net_sharpe": round(s["sharpe"], 3),
            "gross_sharpe": round(s.get("gross_sharpe", float("nan")), 3),
            "net_ann_return": round(s["ann_return"], 4),
            "gross_ann_return": round(s.get("gross_ann_return", float("nan")), 4),
            "ann_vol": round(s["ann_vol"], 4),
            "max_drawdown": round(s["max_drawdown"], 4),
            "avg_daily_turnover": round(s["avg_daily_turnover"], 4),
            "cost_drag_ann": round(s.get("cost_drag_ann", float("nan")), 4),
        })

    summary = pd.DataFrame(rows).set_index("variant")
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    cols = ["resid_mr_every", "shm_every", "net_sharpe", "gross_sharpe",
            "avg_daily_turnover", "cost_drag_ann", "net_ann_return"]
    print(summary[cols].to_string())

    out_path = Path(__file__).parent / "v12_turnover_results.csv"
    summary.to_csv(out_path)
    print(f"\nWrote: {out_path}")

    return summary


if __name__ == "__main__":
    main()
