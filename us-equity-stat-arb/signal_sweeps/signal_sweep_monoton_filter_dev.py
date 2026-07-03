"""
Monoton filter development script.

The top monoton sleeves (skip_252d, w252d) have bad years in 2015, 2018, 2022.
These correspond to downtrending / high-vol / low-dispersion market regimes where
cross-sectional monotonicity breaks down (stocks correlate and lose their idiosyncratic
trending behaviour).

This script tests filter combinations on the two best sleeves to find which regime
conditions most cleanly switch them off in bad years without hurting good years.

Filters under test:
  - market_trend: SPY 50d vs 200d MA — scale down when below
  - vol_spike: SPY 10d vs 60d vol — scale down when realised vol spikes
  - breadth_200ma: pct stocks above 200d MA — scale down when breadth collapses
  - dispersion_low: cross-sectional return dispersion — scale down when dispersion is low
    (when stocks are highly correlated, monoton ranks break down)
  - combined: trend + dispersion together

Outputs: examples/signal_sweeps/out/monoton_filter_dev/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import qstudy as qs
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics
from qstudy import Study
from portfolio_utils import make_equity_curve_regime_scale
from signal_sweep_utils import load_data, TRAIN_START, COST_BPS, eval_full_period

OUT_DIR = Path(__file__).resolve().parent / "out" / "monoton_filter_dev"

# Best sleeves to test
TARGET_SIGNAL_NAME = "monoton_skip_252d"
TARGET_SIGNAL_2_NAME = "monoton_w252d"
N_LONG = 20
N_SHORT = 20
REBALANCE = 21  # from the top configs: r21


# ---------------------------------------------------------------------------
# Signal definitions (matching signal_sweep_monoton.py)
# ---------------------------------------------------------------------------

def monoton_skip_252d(**cache):
    r = cache["_active_returns"].shift(5)
    mu = r.rolling(252).mean()
    return (r.gt(0) == mu.gt(0)).rolling(252).mean() * mu.abs()

monoton_skip_252d.__name__ = "monoton_skip_252d"


def monoton_w252d(**cache):
    r = cache["_active_returns"]
    mu = r.rolling(252).mean()
    return (r.gt(0) == mu.gt(0)).rolling(252).mean() * mu.abs()

monoton_w252d.__name__ = "monoton_w252d"


# ---------------------------------------------------------------------------
# Filter / scaler factories
# ---------------------------------------------------------------------------

def make_vol_spike_scaler(fast: int, slow: int, scale_down: float = 0.0):
    """Scale down (or off) when short-term vol > long-term vol (vol spike)."""
    def _scaler(positions, fast=fast, slow=slow, scale_down=scale_down, **cache):
        bm = cache.get("benchmark")
        if bm is None:
            return positions
        fv = bm.rolling(fast).std()
        sv = bm.rolling(slow).std()
        in_spike = (fv > sv).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(in_spike, scale_down, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"vol_spike_{fast}_{slow}_off{scale_down}"
    return _scaler


def make_trend_scaler(fast: int, slow: int, scale_down: float = 0.0):
    """Scale down when market is in downtrend (SPY fast MA < slow MA)."""
    def _scaler(positions, fast=fast, slow=slow, scale_down=scale_down, **cache):
        bm = cache.get("benchmark")
        if bm is None:
            return positions
        spy_price = (1 + bm).cumprod()
        in_uptrend = spy_price.rolling(fast).mean() > spy_price.rolling(slow).mean()
        scale = pd.Series(
            np.where(in_uptrend.reindex(positions.index).fillna(False), 1.0, scale_down),
            index=positions.index,
        )
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"trend_{fast}_{slow}_off{scale_down}"
    return _scaler


def make_breadth_scaler(threshold: float, scale_down: float = 0.0):
    """Scale down when pct stocks with price > 200d ago < threshold (breadth collapse).

    NOTE: uses prices.shift(200) — price today vs price 200 days ago — NOT the rolling
    mean of a cumulative price from inception (which is always > its MA after a few years).
    """
    def _scaler(positions, threshold=threshold, scale_down=scale_down, **cache):
        returns = cache.get("returns")
        if returns is None:
            return positions
        prices = (1 + returns).cumprod()
        pct_above = (prices > prices.shift(200)).mean(axis=1)
        scale = pd.Series(
            np.where(pct_above.reindex(positions.index).fillna(0) < threshold, scale_down, 1.0),
            index=positions.index,
        )
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"breadth_{threshold}_off{scale_down}"
    return _scaler


def make_dispersion_scaler(window: int, quantile: float, scale_down: float = 0.0):
    """Scale down when cross-sectional return dispersion is low.

    Low dispersion = high correlation = monoton rankings unreliable.
    Dispersion = mean of rolling std of individual stock returns.
    """
    def _scaler(positions, window=window, quantile=quantile, scale_down=scale_down, **cache):
        returns = cache.get("returns")
        if returns is None:
            return positions
        dispersion = returns.rolling(window).std().mean(axis=1)
        # dynamic threshold: rolling quantile of dispersion itself
        threshold = dispersion.rolling(252).quantile(quantile)
        low_disp = (dispersion < threshold).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(low_disp, scale_down, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"disp_{window}_{quantile}_off{scale_down}"
    return _scaler


def make_vol_x_breadth_scaler(
    vol_fast: int, vol_slow: int, breadth_thresh: float, scale_down: float = 0.0
):
    """Scale down when EITHER vol spikes OR breadth collapses."""
    def _scaler(
        positions, vol_fast=vol_fast, vol_slow=vol_slow,
        breadth_thresh=breadth_thresh, scale_down=scale_down, **cache
    ):
        bm = cache.get("benchmark")
        returns = cache.get("returns")
        if bm is None or returns is None:
            return positions

        fv = bm.rolling(vol_fast).std()
        sv = bm.rolling(vol_slow).std()
        in_vol_spike = (fv > sv).reindex(positions.index).fillna(False)

        prices = (1 + returns).cumprod()
        pct_above = (prices > prices.rolling(200).mean()).mean(axis=1)
        low_breadth = (pct_above.reindex(positions.index).fillna(0) < breadth_thresh)

        bad_regime = in_vol_spike | low_breadth
        scale = pd.Series(np.where(bad_regime, scale_down, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"vol{vol_fast}_{vol_slow}_x_breadth{breadth_thresh}_off{scale_down}"
    return _scaler


def make_drawdown_guard_scaler(dd_threshold: float = -0.05, recovery_window: int = 21):
    """Scale to 0 when SPY is more than dd_threshold below its rolling peak.

    More aggressive than the trend filter — triggers as soon as SPY drops X%
    from its trailing high, rather than waiting for MA crossover.
    """
    def _scaler(positions, dd_threshold=dd_threshold, recovery_window=recovery_window, **cache):
        bm = cache.get("benchmark")
        if bm is None:
            return positions
        spy_price = (1 + bm).cumprod()
        rolling_peak = spy_price.cummax()
        drawdown = (spy_price / rolling_peak) - 1
        # Turn off when in drawdown, stay off until recovery_window days after drawdown ends
        in_dd = (drawdown < dd_threshold).reindex(positions.index).fillna(False)
        # Extend the off signal by recovery_window days using rolling max
        in_dd_extended = in_dd.rolling(recovery_window, min_periods=1).max().astype(bool)
        scale = pd.Series(np.where(in_dd_extended, 0.0, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"dd_guard_{dd_threshold}_rec{recovery_window}"
    return _scaler


def make_trend_x_dispersion_scaler(
    trend_fast: int, trend_slow: int, disp_window: int, disp_q: float, scale_down: float = 0.0
):
    """Scale down when EITHER market is in downtrend OR dispersion is low."""
    def _scaler(
        positions, trend_fast=trend_fast, trend_slow=trend_slow,
        disp_window=disp_window, disp_q=disp_q, scale_down=scale_down, **cache
    ):
        bm = cache.get("benchmark")
        returns = cache.get("returns")
        if bm is None or returns is None:
            return positions

        spy_price = (1 + bm).cumprod()
        in_downtrend = ~(spy_price.rolling(trend_fast).mean() > spy_price.rolling(trend_slow).mean())

        dispersion = returns.rolling(disp_window).std().mean(axis=1)
        threshold = dispersion.rolling(252).quantile(disp_q)
        low_disp = dispersion < threshold

        bad_regime = (
            in_downtrend.reindex(positions.index).fillna(False) |
            low_disp.reindex(positions.index).fillna(False)
        )
        scale = pd.Series(np.where(bad_regime, scale_down, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"trend{trend_fast}_{trend_slow}_x_disp{disp_window}_{disp_q}_off{scale_down}"
    return _scaler


# ---------------------------------------------------------------------------
# Build and evaluate one study
# ---------------------------------------------------------------------------

def build_and_eval(
    signal_fn,
    scalers: list,
    universe,
    benchmark,
    factors,
    label: str,
) -> tuple[dict, dict[str, dict]]:
    equity_curve_scaler = make_equity_curve_regime_scale(scale_start=TRAIN_START)

    builder = (
        Study(universe=universe, benchmark=benchmark, factors=factors, verbose=False)
        .base_signal(signal_fn)
        .add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=equity_curve_scaler)
    )
    for scaler in scalers:
        builder = builder.scale_risk(fn=scaler)

    study = builder.rebalance(every=REBALANCE).with_transaction_costs(cost_bps=COST_BPS).run()
    full_m, yearly_m, _ = eval_full_period(study.cache["positions"], universe.returns)
    return full_m, yearly_m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    universe, benchmark, factors = load_data()

    # -------------------------------------------------------------------------
    # Configurations to test
    # Each entry: (label, signal_fn, [scalers])
    # -------------------------------------------------------------------------
    configs = []

    # NOTE ON 2022:
    # 2022 was a systemic bear market (SPY -19%, 81% of days below 50/200 MA).
    # The monoton signal loses edge here but so does almost every long/short equity
    # strategy — the cross-sectional signal gets overwhelmed by market-factor returns.
    # The trend 50/200 filter covers most of 2022 starting from April, but Jan-Mar
    # are still exposed (trend was still "up" while the market was already falling).
    # Full 2022 protection probably belongs at the PORTFOLIO level (bear-market regime
    # switch on the combined portfolio) rather than on individual sleeves.
    # The per-sleeve filters below are designed to improve 2015 and reduce the 2022
    # drag wherever possible, without over-filtering 2016/2017/2019/2023.

    for signal_fn, sig_name in [
        (monoton_skip_252d, "skip_252d"),
        (monoton_w252d, "w252d"),
    ]:
        # Baseline: equity curve scaler only (no regime filter)
        configs.append((f"{sig_name}__baseline", signal_fn, []))

        # Existing best: vol spike 10/60 at 0.25x (from sweep)
        configs.append((f"{sig_name}__vol10_60_x25", signal_fn,
                        [make_vol_spike_scaler(10, 60, scale_down=0.25)]))

        # Test: vol spike fully off
        configs.append((f"{sig_name}__vol10_60_off", signal_fn,
                        [make_vol_spike_scaler(10, 60, scale_down=0.0)]))

        # Test: wider vol window (20/100)
        configs.append((f"{sig_name}__vol20_100_off", signal_fn,
                        [make_vol_spike_scaler(20, 100, scale_down=0.0)]))

        # Test: trend 50/200 fully off
        configs.append((f"{sig_name}__trend50_200_off", signal_fn,
                        [make_trend_scaler(50, 200, scale_down=0.0)]))

        # Test: trend 50/200 at 0.25x
        configs.append((f"{sig_name}__trend50_200_x25", signal_fn,
                        [make_trend_scaler(50, 200, scale_down=0.25)]))

        # Breadth tests (fixed: uses prices.shift(200), not cumprod rolling mean)
        # Good years: 2021 avg=0.58. Bad years: 2015 avg=0.26, 2022 avg=0.30
        # threshold=0.40 fires ~75% of 2015 and ~71% of 2022 but also ~44% of good 2016
        # threshold=0.50 fires in almost every year — too broad

        # Test: breadth < 40% → off (fires mainly in weak-breadth regimes)
        configs.append((f"{sig_name}__breadth40_off", signal_fn,
                        [make_breadth_scaler(0.40, scale_down=0.0)]))

        # Test: breadth < 35% → off (tighter: only deep breadth collapses)
        configs.append((f"{sig_name}__breadth35_off", signal_fn,
                        [make_breadth_scaler(0.35, scale_down=0.0)]))

        # Test: breadth < 30% → off (very tight)
        configs.append((f"{sig_name}__breadth30_off", signal_fn,
                        [make_breadth_scaler(0.30, scale_down=0.0)]))

        # Test: breadth < 50% → off (was the original, now actually fires)
        configs.append((f"{sig_name}__breadth50_off", signal_fn,
                        [make_breadth_scaler(0.50, scale_down=0.0)]))

        # Test: dispersion low (20d/q25) → off
        configs.append((f"{sig_name}__disp20_q25_off", signal_fn,
                        [make_dispersion_scaler(20, 0.25, scale_down=0.0)]))

        # Test: dispersion low (60d/q30) → off
        configs.append((f"{sig_name}__disp60_q30_off", signal_fn,
                        [make_dispersion_scaler(60, 0.30, scale_down=0.0)]))

        # Test: dispersion low (60d/q40) → off
        configs.append((f"{sig_name}__disp60_q40_off", signal_fn,
                        [make_dispersion_scaler(60, 0.40, scale_down=0.0)]))

        # Test: vol OR breadth → off
        configs.append((f"{sig_name}__vol10_60_x_breadth50_off", signal_fn,
                        [make_vol_x_breadth_scaler(10, 60, 0.50, scale_down=0.0)]))

        # Test: trend OR dispersion → off
        configs.append((f"{sig_name}__trend50_200_x_disp60_q30_off", signal_fn,
                        [make_trend_x_dispersion_scaler(50, 200, 60, 0.30, scale_down=0.0)]))

        # Test: trend OR dispersion (tighter) → off
        configs.append((f"{sig_name}__trend50_200_x_disp60_q40_off", signal_fn,
                        [make_trend_x_dispersion_scaler(50, 200, 60, 0.40, scale_down=0.0)]))

        # Test: trend AND vol → off (only off when both conditions met, more permissive)
        configs.append((f"{sig_name}__trend50_200_off_then_vol10_60_off", signal_fn,
                        [
                            make_trend_scaler(50, 200, scale_down=0.0),
                            make_vol_spike_scaler(10, 60, scale_down=0.0),
                        ]))

        # Test: faster trend (20/100) — catches early-2022 drawdown sooner than 50/200
        configs.append((f"{sig_name}__trend20_100_off", signal_fn,
                        [make_trend_scaler(20, 100, scale_down=0.0)]))

        # Test: SPY drawdown guard -5% from peak → off for 21 days
        configs.append((f"{sig_name}__dd_guard_5pct", signal_fn,
                        [make_drawdown_guard_scaler(dd_threshold=-0.05, recovery_window=21)]))

        # Test: SPY drawdown guard -8% from peak → off for 21 days (looser)
        configs.append((f"{sig_name}__dd_guard_8pct", signal_fn,
                        [make_drawdown_guard_scaler(dd_threshold=-0.08, recovery_window=21)]))

        # Test: trend 20/100 (catches early selloffs) + vol spike 10/60
        configs.append((f"{sig_name}__trend20_100_x_vol10_60_off", signal_fn,
                        [make_vol_x_breadth_scaler(10, 60, 0.50, scale_down=0.0),
                         make_trend_scaler(20, 100, scale_down=0.0)]))

    print(f"\nRunning {len(configs)} configs ...\n")

    rows = []
    yearly_rows = []

    for label, signal_fn, scalers in configs:
        print(f"  {label}")
        try:
            full_m, yearly_m = build_and_eval(signal_fn, scalers, universe, benchmark, factors, label)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue

        if not full_m:
            print("    No metrics.")
            continue

        rows.append({"name": label, **full_m})
        for yr, m in yearly_m.items():
            yearly_rows.append({"name": label, "year": int(yr), **m})

    summary = pd.DataFrame(rows)
    yearly = pd.DataFrame(yearly_rows)

    # Compute avg/min annual net Sharpe
    avg_ann = yearly.groupby("name")["net_sharpe"].mean().rename("avg_annual_net_sharpe")
    min_ann = yearly.groupby("name")["net_sharpe"].min().rename("min_annual_net_sharpe")
    pct_neg = yearly.groupby("name")["net_sharpe"].apply(lambda s: (s < 0).mean()).rename("pct_negative_years")
    summary = summary.join(avg_ann, on="name").join(min_ann, on="name").join(pct_neg, on="name")
    summary = summary.sort_values("avg_annual_net_sharpe", ascending=False).reset_index(drop=True)

    # Save
    summary.to_csv(OUT_DIR / "monoton_filter_dev_summary.csv", index=False)
    yearly.to_csv(OUT_DIR / "monoton_filter_dev_yearly.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")

    # Print summary table
    cols = ["name", "net_sharpe", "avg_annual_net_sharpe", "min_annual_net_sharpe",
            "pct_negative_years", "ann_return", "max_drawdown"]
    print("\n" + summary[cols].to_string(index=False))

    # -------------------------------------------------------------------------
    # Yearly heatmap
    # -------------------------------------------------------------------------
    pivot = yearly.pivot_table(index="name", columns="year", values="net_sharpe", aggfunc="mean")
    # Sort by avg annual
    name_order = list(summary["name"])
    pivot = pivot.reindex([n for n in name_order if n in pivot.index])
    years = sorted(pivot.columns)
    pivot = pivot.reindex(columns=years)

    bound = max(abs(np.nanpercentile(pivot.values.astype(float), 5)),
                abs(np.nanpercentile(pivot.values.astype(float), 95)), 0.5)

    n_s = len(pivot)
    n_y = len(years)
    fig, ax = plt.subplots(figsize=(max(10, n_y * 1.5 + 4), max(8, n_s * 0.4 + 2)))
    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="RdYlGn", vmin=-bound, vmax=bound)
    plt.colorbar(im, ax=ax, label="Net Sharpe", fraction=0.03, pad=0.02)

    ax.set_xticks(range(n_y))
    ax.set_xticklabels(years, fontsize=10)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels(pivot.index, fontsize=8, ha="right")
    ax.set_title("Monoton filter dev — Net Sharpe by Year", pad=12, fontsize=11)

    for i in range(n_s):
        for j in range(n_y):
            val = pivot.values[i, j]
            if not np.isnan(val):
                brightness = (val - (-bound)) / (2 * bound) if bound > 0 else 0.5
                color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "monoton_filter_dev_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap.")

    # -------------------------------------------------------------------------
    # Print year-by-year comparison for easy reading
    # -------------------------------------------------------------------------
    print("\n=== Per-year net Sharpe ===")
    year_pivot = yearly.pivot_table(index="name", columns="year", values="net_sharpe").round(3)
    year_pivot = year_pivot.reindex([n for n in name_order if n in year_pivot.index])
    print(year_pivot.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
