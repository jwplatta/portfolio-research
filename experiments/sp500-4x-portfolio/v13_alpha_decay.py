"""sp500_four_strat_port — v13: Alpha decay vs rebalance frequency.

For resid_mr_v10 and short_horizon_momentum_w20, sweep rebalance(every=N)
across N in [1, 2, 3, 5, 10, 15, 21] and measure:
  - gross Sharpe (no costs) — pure signal decay
  - net Sharpe at 5, 10, 15 bps
  - avg daily turnover
  - cost-optimal rebalance frequency (highest net Sharpe) per cost level

Usage:
    uv run python experiments/sp500-four-strat-port/v13_alpha_decay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v0_equal_weight import (  # noqa: E402
    load_benchmark,
    load_sector_factors,
    load_sector_map,
    load_universe,
)

import qstudy as qs
from qstudy import Study
from qstudy.study.metrics import turnover

REBALANCE_FREQS = [1, 2, 3, 5, 10, 15, 21]
COST_LEVELS_BPS = [5, 10, 15]


# ---------------------------------------------------------------------------
# resid_mr standalone (no portfolio, run directly with universe/benchmark)
# ---------------------------------------------------------------------------


def build_resid_mr_standalone(sector_factors, every: int) -> Study:
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
        price = (1 + cache["benchmark"].fillna(0.0)).cumprod()
        scale = pd.Series(
            np.where(price.rolling(150).mean() >= price.rolling(250).mean(), 1.0, 0.75),
            index=price.index,
        )
        return positions.mul(scale.shift(1), axis=0)

    equity_curve_regime_scale.__name__ = "equity_curve_regime_scale"
    benchmark_regime_scale.__name__ = "benchmark_regime_scale"

    return (
        Study(name=f"resid_mr_rb{every}", factors=sector_factors)
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


# ---------------------------------------------------------------------------
# short_horizon_momentum standalone
# ---------------------------------------------------------------------------


def build_shm_standalone(sector_map: dict, every: int) -> Study:
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

        signal_fn.__name__ = f"shm_signal_{window}_{skip}"
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

        scaler.__name__ = f"sector_beta_neutralize_{window}_{passes}"
        return scaler

    return (
        Study(name=f"shm_rb{every}")
        .base_signal(short_horizon_sector_relative_signal(window=20, skip=0))
        .add_tradeable_constraint(qs.min_price(5.0))
        .add_tradeable_constraint(qs.min_adv(20_000_000.0))
        .add_tradeable_constraint(qs.liquidity(top_n=150, window=60))
        .build_long_short(n_long=20, n_short=20)
        .weight_equal()
        .scale_risk(sector_beta_neutralize_positions(window=60, passes=2))
        .rebalance(every=every)
    )


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def sweep_strategy(builder_fn, freqs, universe, benchmark, strategy_label):
    """Run a strategy at each rebalance frequency. Returns a DataFrame of results."""
    rows = []
    for every in freqs:
        print(f"  {strategy_label} every={every}...", flush=True)
        study = builder_fn(every)
        study._inject_data(universe, benchmark)
        study.run()

        gross_ret = study.cache["gross_portfolio_returns"]
        positions = study.cache["positions"]
        to = turnover(positions).mean()
        gross_sharpe = study.cache["metrics_summary"]["sharpe"]  # no costs applied

        row = {
            "every": every,
            "gross_sharpe": gross_sharpe,
            "avg_daily_to": to,
            "ann_return_gross": study.cache["metrics_summary"]["ann_return"],
            "ann_vol": study.cache["metrics_summary"]["ann_vol"],
            "max_drawdown": study.cache["metrics_summary"]["max_drawdown"],
        }
        for bps in COST_LEVELS_BPS:
            cost_per_dollar = bps / 10_000
            cost_series = (
                turnover(positions).shift(1).reindex(gross_ret.index).fillna(0.0) * cost_per_dollar
            )
            net_ret = gross_ret - cost_series
            from qstudy.study.metrics import sharpe as sharpe_fn
            row[f"net_sharpe_{bps}bps"] = sharpe_fn(net_ret)
            row[f"drag_{bps}bps"] = to * cost_per_dollar * 252

        rows.append(row)
    return pd.DataFrame(rows).set_index("every")


def print_sweep(label, df):
    print(f"\n{'='*70}")
    print(f"  Alpha decay sweep: {label}")
    print(f"{'='*70}")
    cols = (
        ["gross_sharpe", "avg_daily_to"]
        + [f"net_sharpe_{b}bps" for b in COST_LEVELS_BPS]
        + [f"drag_{b}bps" for b in COST_LEVELS_BPS[:1]]
    )
    print(df[cols].round(3).to_string())

    # Optimal frequency per cost level
    print()
    print("  Cost-optimal rebalance frequency:")
    print(f"    gross (0bps):  every={df['gross_sharpe'].idxmax()}")
    for bps in COST_LEVELS_BPS:
        col = f"net_sharpe_{bps}bps"
        best = df[col].idxmax()
        print(f"    {bps}bps:          every={best}  (net sharpe={df.loc[best, col]:.2f})")


def plot_alpha_decay(mr_df, shm_df, out_path):
    matplotlib.use("Agg")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Alpha Decay vs Rebalance Frequency", fontsize=14)

    colors = {"gross": "black", 5: "#2ca02c", 10: "#1f77b4", 15: "#d62728"}

    for row_idx, (df, label) in enumerate([(mr_df, "resid_mr_v10"), (shm_df, "shm_w20")]):
        # Sharpe panel
        ax = axes[row_idx][0]
        ax.plot(df.index, df["gross_sharpe"], color=colors["gross"], marker="o",
                linewidth=2, label="gross (0bps)")
        for bps in COST_LEVELS_BPS:
            ax.plot(df.index, df[f"net_sharpe_{bps}bps"], color=colors[bps], marker="o",
                    linewidth=1.5, linestyle="--", label=f"net {bps}bps")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_title(f"{label} — Sharpe vs Rebalance Frequency")
        ax.set_xlabel("Rebalance every N days")
        ax.set_ylabel("Sharpe ratio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(df.index)

        # Turnover panel
        ax = axes[row_idx][1]
        drag_10 = df["avg_daily_to"] * 0.001 * 252
        ax.bar(df.index, df["avg_daily_to"], color="#1f77b4", alpha=0.7, label="avg daily TO")
        ax2 = ax.twinx()
        ax2.plot(df.index, drag_10, color="#d62728", marker="o", linewidth=1.5,
                 label="annualized drag @ 10bps")
        ax2.set_ylabel("Annualized drag @ 10bps", color="#d62728")
        ax.set_title(f"{label} — Turnover vs Rebalance Frequency")
        ax.set_xlabel("Rebalance every N days")
        ax.set_ylabel("Avg daily turnover")
        ax.set_xticks(df.index)
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved chart: {out_path}")


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

    # --- resid_mr sweep ---
    print("\nSweeping resid_mr_v10 rebalance frequency...")
    mr_df = sweep_strategy(
        lambda every: build_resid_mr_standalone(sector_factors, every),
        REBALANCE_FREQS,
        universe,
        benchmark,
        "resid_mr",
    )
    print_sweep("resid_mr_v10", mr_df)

    # --- shm sweep ---
    print("\nSweeping short_horizon_momentum_w20 rebalance frequency...")
    shm_df = sweep_strategy(
        lambda every: build_shm_standalone(sector_map, every),
        REBALANCE_FREQS,
        universe,
        benchmark,
        "shm_w20",
    )
    print_sweep("short_horizon_momentum_w20", shm_df)

    # --- Save results ---
    here = Path(__file__).parent
    mr_df.to_csv(here / "v13_alpha_decay_resid_mr.csv")
    shm_df.to_csv(here / "v13_alpha_decay_shm.csv")
    print(f"\nWrote CSVs to {here}/v13_alpha_decay_*.csv")

    plot_alpha_decay(mr_df, shm_df, here / "v13_alpha_decay.png")

    return mr_df, shm_df


if __name__ == "__main__":
    main()
