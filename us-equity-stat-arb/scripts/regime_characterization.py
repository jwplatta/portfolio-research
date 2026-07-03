"""
Regime characterization for validation years 2021, 2022, 2023.

Loads SPY + SP500 universe returns and computes statistics to classify
each year's market environment:
  - Trend / momentum strength
  - Mean-reversion tendency
  - Realized volatility
  - Market directionality (dispersion of returns vs flat grinding)
  - Cross-sectional dispersion (how spread out individual stock returns were)
  - Drawdown severity
  - Autocorrelation of daily returns (persistence vs reversal)
  - Up-day / down-day asymmetry
  - Rolling beta stability
  - Breadth (% stocks above their 200d MA)

Outputs:
  examples/out/regime_characterization/
    regime_stats.csv         — all metrics per year
    regime_heatmap.png       — normalized heatmap across years x metrics
    regime_timeseries.png    — key time series for visual inspection
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import qstudy as qs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

YEARS = list(range(2015, 2026))
START = "2013-01-01"
END = "2026-05-31"

OUT_DIR = Path(__file__).parent / "out" / "regime_characterization"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sharpe(rets: pd.Series) -> float:
    std = rets.std()
    return float(rets.mean() / std * (252**0.5)) if std > 0 else float("nan")


def max_drawdown(rets: pd.Series) -> float:
    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return float(dd.min())


def autocorr_lag1(rets: pd.Series) -> float:
    return float(rets.autocorr(lag=1))


def momentum_trend_strength(rets: pd.Series, window: int = 60) -> float:
    """Mean absolute rolling cumulative return over `window` days (annualized).
    High = market spent most of the year in a strong directional trend."""
    cum = rets.rolling(window).sum() * (252 / window)
    return float(cum.abs().mean())


def momentum_hit_rate(rets: pd.Series, signal_window: int = 20) -> float:
    """Fraction of days where the sign of the prior rolling return predicted
    the sign of the next day's return. >0.5 = momentum, <0.5 = mean-reversion."""
    signal = rets.rolling(signal_window).sum().shift(1)
    aligned = pd.concat([signal, rets], axis=1).dropna()
    if len(aligned) < 20:
        return float("nan")
    correct = (np.sign(aligned.iloc[:, 0]) == np.sign(aligned.iloc[:, 1])).mean()
    return float(correct)


def mean_reversion_score(rets: pd.Series, window: int = 5) -> float:
    """SPY-level: correlation of short-term momentum with next-day return.
    Negative = mean-reverting, positive = momentum-driven."""
    mom = rets.rolling(window).mean().shift(1)
    aligned = pd.concat([mom, rets], axis=1).dropna()
    if len(aligned) < 20:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def xs_mean_reversion_score(stock_rets: pd.DataFrame, window: int = 5) -> float:
    """Cross-sectional MR: mean daily rank-correlation between a stock's
    recent-return rank (signal) and its next-day return rank (outcome).
    Negative = cross-sectional mean-reversion, positive = cross-sectional momentum."""
    signal = stock_rets.rolling(window).mean().shift(1)
    daily_corrs = []
    for date in stock_rets.index:
        sig = signal.loc[date].dropna()
        ret = stock_rets.loc[date].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < 20:
            continue
        corr = sig[common].rank().corr(ret[common].rank(), method="spearman")
        daily_corrs.append(corr)
    return float(np.mean(daily_corrs)) if daily_corrs else float("nan")


def xs_momentum_score(stock_rets: pd.DataFrame, window: int = 60) -> float:
    """Cross-sectional momentum: mean daily rank-correlation between a stock's
    longer-term return rank and its next-day return rank.
    Positive = momentum persists, negative = reversal."""
    signal = stock_rets.rolling(window).mean().shift(1)
    daily_corrs = []
    for date in stock_rets.index:
        sig = signal.loc[date].dropna()
        ret = stock_rets.loc[date].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < 20:
            continue
        corr = sig[common].rank().corr(ret[common].rank(), method="spearman")
        daily_corrs.append(corr)
    return float(np.mean(daily_corrs)) if daily_corrs else float("nan")


def trend_consistency(rets: pd.Series) -> float:
    """Fraction of rolling 21d windows where market moved >1% net (directional)."""
    r21 = rets.rolling(21).sum()
    return float((r21.abs() > 0.01).mean())


def cross_sectional_dispersion(stock_rets: pd.DataFrame) -> float:
    """Mean cross-sectional std of daily stock returns — how spread out stocks are."""
    return float(stock_rets.std(axis=1).mean())


def pct_stocks_above_200ma(close: pd.DataFrame) -> pd.Series:
    """Daily breadth: fraction of stocks above their 200d MA."""
    # Forward-fill within each ticker so holidays don't break rolling window
    close_ffill = close.ffill()
    ma200 = close_ffill.rolling(200, min_periods=150).mean()
    above = (close_ffill > ma200).where(ma200.notna())
    return above.mean(axis=1)


def up_capture_ratio(spy_rets: pd.Series, year_mask: pd.Series) -> float:
    """Ratio of mean return on up-SPY days vs down-SPY days (magnitude)."""
    y = spy_rets[year_mask]
    up = y[y > 0].mean()
    dn = y[y < 0].mean()
    if np.isnan(up) or np.isnan(dn) or dn == 0:
        return float("nan")
    return float(up / abs(dn))


def vol_of_vol(rets: pd.Series, window: int = 21) -> float:
    """Volatility of rolling realized vol — high = vol regime changes frequently."""
    rolling_vol = rets.rolling(window).std() * (252**0.5)
    return float(rolling_vol.std())


def skewness(rets: pd.Series) -> float:
    return float(rets.skew())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    print(f"\nLoading data ({START} to {END}) ...")
    benchmark = qs.download(["SPY"], start=START, end=END)
    universe = qs.download(index_code="SP500", start=START, end=END)
    print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")

    spy_rets = benchmark.returns["SPY"]
    spy_close = benchmark.close["SPY"]
    stock_rets = universe.returns
    stock_close = universe.close

    # Breadth (computed on full history for the 200d MA to be valid)
    breadth = pct_stocks_above_200ma(stock_close)

    rows = []
    for year in YEARS:
        mask_spy = spy_rets.index.year == year
        yr_spy = spy_rets[mask_spy]
        yr_stocks = stock_rets[stock_rets.index.year == year]
        yr_breadth = breadth[breadth.index.year == year]
        mask = mask_spy  # keep for up_capture_ratio

        # SPY level stats
        ann_ret = float((1 + yr_spy).prod() ** (252 / len(yr_spy)) - 1)
        ann_vol = float(yr_spy.std() * (252**0.5))
        yr_sharpe = sharpe(yr_spy)
        mdd = max_drawdown(yr_spy)
        skew = skewness(yr_spy)
        ac1 = autocorr_lag1(yr_spy)
        vov = vol_of_vol(yr_spy)
        up_cap = up_capture_ratio(spy_rets, mask)
        trend_cons = trend_consistency(yr_spy)

        # SPY-level momentum vs mean-reversion
        spy_mr5 = mean_reversion_score(yr_spy, window=5)
        trend_strength = momentum_trend_strength(yr_spy, window=60)
        mom_hit_rate = momentum_hit_rate(yr_spy, signal_window=20)

        # Cross-sectional
        cs_disp = cross_sectional_dispersion(yr_stocks)
        breadth_mean = float(yr_breadth.mean())
        breadth_std = float(yr_breadth.std())

        # Cross-sectional MR/momentum (the ones that actually matter for L/S strategies)
        print(f"  Computing cross-sectional MR scores for {year} ...")
        xs_mr5 = xs_mean_reversion_score(yr_stocks, window=5)
        xs_mom60 = xs_momentum_score(yr_stocks, window=60)

        rows.append(
            {
                "year": year,
                # Return profile
                "spy_ann_return": ann_ret,
                "spy_ann_vol": ann_vol,
                "spy_sharpe": yr_sharpe,
                "spy_max_drawdown": mdd,
                "spy_skew": skew,
                # SPY-level trend vs mean-reversion
                "spy_mr_score_5d": spy_mr5,
                "trend_strength_60d": trend_strength,  # ann. abs. rolling 60d return — high = trending
                "mom_hit_rate_20d": mom_hit_rate,  # >0.5 = momentum, <0.5 = mean-reversion
                # Cross-sectional MR/momentum (rank-IC)
                "xs_mr_5d": xs_mr5,  # negative = stocks mean-revert cross-sectionally
                "xs_mom_60d": xs_mom60,  # positive = winners keep winning cross-sectionally
                "trend_consistency": trend_cons,  # frac of months with >1% net move
                # Regime texture
                "lag1_autocorr": ac1,
                "vol_of_vol": vov,  # high = volatile vol regime
                "up_vs_down_day_ratio": up_cap,
                # Cross-sectional
                "cs_dispersion": cs_disp,  # how spread out stock returns are (alpha opportunity)
                "breadth_mean": breadth_mean,  # avg % stocks above 200d MA
                "breadth_std": breadth_std,  # how much breadth moved around
            }
        )

        print(f"\n--- {year} ---")
        print(
            f"  SPY return: {ann_ret:+.1%}  vol: {ann_vol:.1%}  sharpe: {yr_sharpe:.2f}  MDD: {mdd:.1%}"
        )
        print(f"  Lag-1 autocorr: {ac1:+.3f}  (+ = momentum, - = mean-reversion)")
        print(
            f"  SPY 5d MR score: {spy_mr5:+.3f}  Trend strength 60d: {trend_strength:.3f}  Mom hit rate 20d: {mom_hit_rate:.3f}"
        )
        print(f"  XS MR 5d (rank-IC): {xs_mr5:+.4f}  XS Mom 60d (rank-IC): {xs_mom60:+.4f}")
        print(f"  Trend consistency (>1% monthly moves): {trend_cons:.1%}")
        print(f"  Vol of vol: {vov:.3f}  Skew: {skew:.2f}")
        print(f"  Cross-sectional dispersion: {cs_disp:.4f}")
        print(f"  Breadth (% above 200d MA): mean={breadth_mean:.1%}  std={breadth_std:.1%}")

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    stats_df = pd.DataFrame(rows).set_index("year")
    stats_df.to_csv(OUT_DIR / "regime_stats.csv")
    print("\nSaved regime_stats.csv")
    print(stats_df.T.to_string(float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------------------------
    # Heatmap: normalized metrics
    # ---------------------------------------------------------------------------
    numeric_cols = stats_df.columns.tolist()
    normed = stats_df[numeric_cols].copy().astype(float)
    # Normalize each metric to [0, 1] across years
    for col in numeric_cols:
        mn, mx = normed[col].min(), normed[col].max()
        if mx > mn:
            normed[col] = (normed[col] - mn) / (mx - mn)
        else:
            normed[col] = 0.5

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(normed.values.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalized (0=low, 1=high)")
    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels(YEARS, fontsize=11)
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=8)
    ax.set_title("Market Regime Characterization — Validation Years", pad=10)

    # Annotate with raw values
    for i, year in enumerate(YEARS):
        for j, col in enumerate(numeric_cols):
            val = stats_df.loc[year, col]
            fmt = f"{val:+.2f}" if abs(val) < 10 else f"{val:.1f}"
            ax.text(i, j, fmt, ha="center", va="center", fontsize=7, color="black")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "regime_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved regime_heatmap.png")

    # ---------------------------------------------------------------------------
    # Time series: rolling vol, breadth, cumulative return
    # ---------------------------------------------------------------------------
    palette = [
        "#4e79a7",
        "#f28e2b",
        "#59a14f",
        "#e15759",
        "#76b7b2",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
    ]
    year_colors = {year: palette[i % len(palette)] for i, year in enumerate(YEARS)}

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    # Panel 1: Cumulative return by year
    ax = axes[0]
    for year in YEARS:
        yr_spy = spy_rets[spy_rets.index.year == year]
        cum = (1 + yr_spy).cumprod() - 1
        cum.index = range(len(cum))
        ax.plot(cum.values, label=str(year), color=year_colors[year])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("SPY Cumulative Return")
    ax.set_title("SPY Path by Year")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Rolling 21d realized vol (annualized)
    ax = axes[1]
    for year in YEARS:
        yr_spy = spy_rets[spy_rets.index.year == year]
        rv = yr_spy.rolling(21).std() * (252**0.5)
        rv.index = range(len(rv))
        ax.plot(rv.values, label=str(year), color=year_colors[year])
    ax.set_ylabel("Rolling 21d Realized Vol (ann.)")
    ax.set_title("SPY Volatility Regime by Year")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Breadth (% above 200d MA)
    ax = axes[2]
    for year in YEARS:
        yr_breadth = breadth[breadth.index.year == year]
        yr_breadth.index = range(len(yr_breadth))
        ax.plot(yr_breadth.values, label=str(year), color=year_colors[year])
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Fraction Stocks > 200d MA")
    ax.set_title("Market Breadth by Year")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "regime_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved regime_timeseries.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
