"""
Event signal filter development script.

Focus sleeves (from pool candidates + yearly analysis):
  - gap_accum_3d__r21  — best overall; strong 2019/2022/2023, weak 2015/2016/2020/2021
  - gap_accum_10d__r21 — best in 2018; weak 2017/2020/2022
  - gap_accum_2d__r10  — inconsistent; kept for comparison

Key insight:
  gap_accum_3d works best in choppy/downtrending markets (2019, 2022, 2023) and worst in
  strong uptrends (2017, 2020, 2021). The existing trend_20_100 scaler scales DOWN in
  uptrend (MR polarity: reduce exposure when trend is up), which already helps but only
  scales to 0.25x. Testing fully-off variants and other regime conditions.

  gap_accum_10d is different — bad years are 2017 and 2020/2022. The 2018 strength may
  be a mean-reversion regime (high vol, Q4 crash). Testing vol-expansion filters for it.

  2020 is interesting: strong bull run but both 3d and 10d sleeves underperform.
  Likely due to the March COVID crash distorting the gap signal, and then the melt-up.

Outputs: examples/signal_sweeps/out/event_filter_dev/
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
sys.path.insert(0, str(Path(__file__).parent))

import qstudy as qs
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics
from qstudy import Study
from portfolio_utils import make_equity_curve_regime_scale
from signal_sweep_utils import load_data, TRAIN_START, COST_BPS, eval_full_period

OUT_DIR = Path(__file__).resolve().parent / "out" / "event_filter_dev"

N_LONG = 20
N_SHORT = 20
REBALANCE = 21


# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------

def gap_accum_3d(**cache):
    r = cache["_active_returns"]
    return -r.rolling(3).max()

gap_accum_3d.__name__ = "gap_accum_3d"


def gap_accum_10d(**cache):
    r = cache["_active_returns"]
    return -r.rolling(10).max()

gap_accum_10d.__name__ = "gap_accum_10d"


def gap_accum_2d(**cache):
    r = cache["_active_returns"]
    return -r.rolling(2).max()

gap_accum_2d.__name__ = "gap_accum_2d"


# ---------------------------------------------------------------------------
# Scaler factories
# ---------------------------------------------------------------------------

def make_trend_scaler(fast: int, slow: int, scale_in_uptrend: float, scale_in_downtrend: float = 1.0):
    """MR-polarity trend scaler: gap reversion works better in downtrends/chop.

    scale_in_uptrend: position scale when market is trending up (usually < 1.0)
    scale_in_downtrend: position scale when market is below trend (usually 1.0)
    """
    def _scaler(positions, fast=fast, slow=slow,
                su=scale_in_uptrend, sd=scale_in_downtrend, **cache):
        bm = cache.get("benchmark")
        if bm is None:
            return positions
        spy_price = (1 + bm).cumprod()
        in_uptrend = spy_price.rolling(fast).mean() > spy_price.rolling(slow).mean()
        scale = pd.Series(
            np.where(in_uptrend.reindex(positions.index).fillna(False), su, sd),
            index=positions.index,
        )
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"trend_{fast}_{slow}_up{scale_in_uptrend}_dn{scale_in_downtrend}"
    return _scaler


def make_vol_expansion_scaler(fast: int, slow: int, scale_in_spike: float = 1.0, scale_no_spike: float = 0.25):
    """Vol expansion scaler: gap reversion thrives when vol is elevated.

    scale_in_spike: scale when short vol > long vol (elevated vol regime)
    scale_no_spike: scale when vol is calm (reduce exposure)
    """
    def _scaler(positions, fast=fast, slow=slow,
                ss=scale_in_spike, sn=scale_no_spike, **cache):
        bm = cache.get("benchmark")
        if bm is None:
            return positions
        fv = bm.rolling(fast).std()
        sv = bm.rolling(slow).std()
        in_spike = (fv > sv).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(in_spike, ss, sn), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"vol_{fast}_{slow}_spike{scale_in_spike}_calm{scale_no_spike}"
    return _scaler


def make_breadth_daily_scaler(window: int, low_q: float, scale_low: float = 1.0, scale_high: float = 0.25):
    """Scale based on breadth of positive daily returns (rolling quantile).

    For gap reversion: low breadth (many stocks down) → good environment.
    """
    def _scaler(positions, window=window, low_q=low_q,
                sl=scale_low, sh=scale_high, **cache):
        returns = cache.get("returns")
        if returns is None:
            return positions
        r = returns.dropna(axis=1, how="all")
        breadth = (r > 0).rolling(window).mean().mean(axis=1)
        threshold = breadth.rolling(252).quantile(low_q)
        low_breadth = (breadth < threshold).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(low_breadth, sl, sh), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"breadth_{window}_q{low_q}_low{scale_low}_high{scale_high}"
    return _scaler


def make_corr_scaler(window: int, high_q: float, scale_high_corr: float = 1.0, scale_low_corr: float = 0.25):
    """Scale up when cross-sectional correlation is high (crowded markets → bigger gaps)."""
    def _scaler(positions, window=window, high_q=high_q,
                sh=scale_high_corr, sl=scale_low_corr, **cache):
        rets = cache.get("returns")
        if rets is None:
            return positions
        r = rets.dropna(axis=1, how="all")
        sample = r.iloc[:, :50]
        avg_corr = (
            sample.rolling(window)
            .corr()
            .groupby(level=0)
            .apply(lambda m: (m.values.sum() - len(m)) / max(len(m) * (len(m) - 1), 1))
        )
        threshold = avg_corr.rolling(252).quantile(high_q)
        high_corr = (avg_corr > threshold).reindex(positions.index).fillna(False)
        scale = pd.Series(np.where(high_corr, sh, sl), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"corr_{window}_q{high_q}_high{scale_high_corr}_low{scale_low_corr}"
    return _scaler


def make_dd_guard_scaler(threshold: float = -0.05, recovery: int = 21, scale_in_dd: float = 1.0):
    """During SPY drawdown >threshold from peak: use scale_in_dd (can be >1 to increase size)."""
    def _scaler(positions, threshold=threshold, recovery=recovery, sid=scale_in_dd, **cache):
        bm = cache.get("benchmark")
        if bm is None:
            return positions
        spy_price = (1 + bm).cumprod()
        rolling_peak = spy_price.cummax()
        drawdown = (spy_price / rolling_peak) - 1
        in_dd = (drawdown < threshold).reindex(positions.index).fillna(False)
        in_dd_extended = in_dd.rolling(recovery, min_periods=1).max().astype(bool)
        # out of drawdown = scale 1.0; in drawdown = scale_in_dd
        scale = pd.Series(np.where(in_dd_extended, sid, 1.0), index=positions.index)
        return positions.mul(scale.shift(1), axis=0)
    _scaler.__name__ = f"dd_guard_{threshold}_rec{recovery}_sid{scale_in_dd}"
    return _scaler


# ---------------------------------------------------------------------------
# Build and evaluate
# ---------------------------------------------------------------------------

def build_and_eval(signal_fn, scalers, universe, benchmark, factors):
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
    # Configs to test
    # Notation: MR polarity = scale DOWN in uptrend (gap reversion works in chop/down)
    # -------------------------------------------------------------------------
    configs = []

    for signal_fn, sig_name in [
        (gap_accum_3d, "gap3d"),
        (gap_accum_10d, "gap10d"),
        (gap_accum_2d, "gap2d"),
    ]:
        # Baseline
        configs.append((f"{sig_name}__baseline", signal_fn, []))

        # --- MR-polarity trend scalers (existing sweep, for reference) ---
        # 0.25x in uptrend (from sweep)
        configs.append((f"{sig_name}__trend20_100_x25", signal_fn,
                        [make_trend_scaler(20, 100, scale_in_uptrend=0.25)]))
        configs.append((f"{sig_name}__trend50_200_x25", signal_fn,
                        [make_trend_scaler(50, 200, scale_in_uptrend=0.25)]))
        configs.append((f"{sig_name}__trend20_100_x50", signal_fn,
                        [make_trend_scaler(20, 100, scale_in_uptrend=0.5)]))

        # Fully off in uptrend (gap reversion has no edge in trending markets)
        configs.append((f"{sig_name}__trend20_100_off", signal_fn,
                        [make_trend_scaler(20, 100, scale_in_uptrend=0.0)]))
        configs.append((f"{sig_name}__trend50_200_off", signal_fn,
                        [make_trend_scaler(50, 200, scale_in_uptrend=0.0)]))

        # Boost in downtrend, reduce in uptrend
        configs.append((f"{sig_name}__trend20_100_boost", signal_fn,
                        [make_trend_scaler(20, 100, scale_in_uptrend=0.25, scale_in_downtrend=1.5)]))

        # --- Vol expansion scalers (gap reversion thrives in high-vol regimes) ---
        # Existing sweep: scale down when calm (0.25x when no spike)
        configs.append((f"{sig_name}__vol10_60_calm25", signal_fn,
                        [make_vol_expansion_scaler(10, 60, scale_in_spike=1.0, scale_no_spike=0.25)]))
        # Fully off when vol is calm
        configs.append((f"{sig_name}__vol10_60_calm_off", signal_fn,
                        [make_vol_expansion_scaler(10, 60, scale_in_spike=1.0, scale_no_spike=0.0)]))
        # Boost in vol spike
        configs.append((f"{sig_name}__vol10_60_boost", signal_fn,
                        [make_vol_expansion_scaler(10, 60, scale_in_spike=1.5, scale_no_spike=0.5)]))

        # --- Breadth scalers (low breadth = stressed market = good for gap reversion) ---
        configs.append((f"{sig_name}__breadth20_q25", signal_fn,
                        [make_breadth_daily_scaler(20, 0.25, scale_low=1.0, scale_high=0.25)]))
        configs.append((f"{sig_name}__breadth20_q30", signal_fn,
                        [make_breadth_daily_scaler(20, 0.30, scale_low=1.0, scale_high=0.25)]))

        # --- DD guard: increase size during market drawdowns (gap reversion thrives then) ---
        configs.append((f"{sig_name}__dd_guard_boost", signal_fn,
                        [make_dd_guard_scaler(threshold=-0.05, recovery=10, scale_in_dd=1.5)]))

        # --- Combinations ---
        # trend off + vol boost: only trade when market is down AND vol is high
        configs.append((f"{sig_name}__trend20_off_x_vol10_calm25", signal_fn,
                        [make_trend_scaler(20, 100, scale_in_uptrend=0.0),
                         make_vol_expansion_scaler(10, 60, scale_in_spike=1.0, scale_no_spike=0.25)]))
        # trend x25 + breadth
        configs.append((f"{sig_name}__trend20_x25_breadth20_q30", signal_fn,
                        [make_trend_scaler(20, 100, scale_in_uptrend=0.25),
                         make_breadth_daily_scaler(20, 0.30, scale_low=1.0, scale_high=0.25)]))

    print(f"\nRunning {len(configs)} configs ...\n")

    rows = []
    yearly_rows = []

    for label, signal_fn, scalers in configs:
        print(f"  {label}")
        try:
            full_m, yearly_m = build_and_eval(signal_fn, scalers, universe, benchmark, factors)
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

    avg_ann = yearly.groupby("name")["net_sharpe"].mean().rename("avg_annual_net_sharpe")
    min_ann = yearly.groupby("name")["net_sharpe"].min().rename("min_annual_net_sharpe")
    pct_neg = yearly.groupby("name")["net_sharpe"].apply(lambda s: (s < 0).mean()).rename("pct_negative_years")
    summary = summary.join(avg_ann, on="name").join(min_ann, on="name").join(pct_neg, on="name")
    summary = summary.sort_values("avg_annual_net_sharpe", ascending=False).reset_index(drop=True)

    summary.to_csv(OUT_DIR / "event_filter_dev_summary.csv", index=False)
    yearly.to_csv(OUT_DIR / "event_filter_dev_yearly.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")

    cols = ["name", "net_sharpe", "avg_annual_net_sharpe", "min_annual_net_sharpe",
            "pct_negative_years", "ann_return", "max_drawdown"]
    print("\n" + summary[cols].to_string(index=False))

    # -------------------------------------------------------------------------
    # Heatmap
    # -------------------------------------------------------------------------
    pivot = yearly.pivot_table(index="name", columns="year", values="net_sharpe", aggfunc="mean")
    name_order = list(summary["name"])
    pivot = pivot.reindex([n for n in name_order if n in pivot.index])
    years = sorted(pivot.columns)
    pivot = pivot.reindex(columns=years)

    bound = max(abs(np.nanpercentile(pivot.values.astype(float), 5)),
                abs(np.nanpercentile(pivot.values.astype(float), 95)), 0.5)

    n_s, n_y = len(pivot), len(years)
    fig, ax = plt.subplots(figsize=(max(10, n_y * 1.5 + 4), max(8, n_s * 0.4 + 2)))
    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="RdYlGn", vmin=-bound, vmax=bound)
    plt.colorbar(im, ax=ax, label="Net Sharpe", fraction=0.03, pad=0.02)
    ax.set_xticks(range(n_y))
    ax.set_xticklabels(years, fontsize=10)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels(pivot.index, fontsize=8, ha="right")
    ax.set_title("Event filter dev — Net Sharpe by Year", pad=12, fontsize=11)
    for i in range(n_s):
        for j in range(n_y):
            val = pivot.values[i, j]
            if not np.isnan(val):
                brightness = (val - (-bound)) / (2 * bound) if bound > 0 else 0.5
                color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "event_filter_dev_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved heatmap.")

    print("\n=== Per-year net Sharpe ===")
    year_pivot = yearly.pivot_table(index="name", columns="year", values="net_sharpe").round(3)
    year_pivot = year_pivot.reindex([n for n in name_order if n in year_pivot.index])
    print(year_pivot.to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
