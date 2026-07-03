"""
Annual performance breakdown for all sleeves in sig_fam_utils.py.

Runs all 40 sleeves on the full 2015-2023 window, then slices each year
and computes net Sharpe, ann return, vol, max drawdown, and turnover.

Outputs (examples/out/sleeve_annual_performance/):
  - sleeve_annual_performance.csv  — one row per sleeve x year
  - sleeve_net_sharpe_heatmap.png  — heatmap: sleeves x years (net Sharpe)
  - sleeve_ann_return_heatmap.png  — heatmap: sleeves x years (ann return)
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

import portfolio_utils as pu
from sig_fam_utils import build_sleeve_specs, sleeve_short_name

import qstudy as qs
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COST_BPS = 10.0
# Warm-up period loaded before each fold start so rolling windows are filled
# from day 1 of the fold. OLS residualization and the equity-curve regime
# scaler are anchored to fold_start so warm-up data doesn't change in-sample
# beta estimates or regime decisions.
WARMUP_YEARS = 1
FOLDS = [
    ("2015-01-01", "2019-12-31"),
    ("2015-01-01", "2020-12-31"),
    ("2015-01-01", "2021-12-31"),
    ("2015-01-01", "2022-12-31"),
    ("2015-01-01", "2023-12-31"),
]

# Sleeves to analyse. Use all pool sleeves from sig_fam_utils.SIGNAL_POOL_SLEEVE_NAMES.
# Set to [] to analyse every sleeve in the pool, or list specific names to narrow the focus.
from sig_fam_utils import SIGNAL_POOL_SLEEVE_NAMES

SLEEVES: list[str] = [
    "bear_reversal_20d__r21__trend_20_100_mr__cond__bear_narrow_lt40",
    "cumret_spread_20_252__r5__vol_20_60__cond__none",
    "dist_mr_k1_z20__r21__none__cond__none",
    "dist_mr_k1_z60__r21__none__cond__none",
    "monoton_skip_252d__r21__breadth_40_off__cond__none",
    "dist_mr_k3_z20__r10__none__cond__none",
    "vol_accel_20_120d__r10__vol_10_60_up__cond__breadth_lt40",
    "dist_mr_k3_z10__r10__none__cond__none",
    "resid_gap_accum_5d__r10__vol_10_60_off__cond__none",
    "beta_momentum_60_252d__r21__none__cond__breadth_lt50",
    "dist_mr_k3_z20__r21__none__cond__none",
    "gap_accum_3d__r21__trend_20_100_off__cond__none",
    "ma_dist_20_200__r21__none__cond__none",
    "monoton_skip_252d__r21__breadth_35_off__cond__none",
    "sharpe_mom_120d__r5__trend_50_200_mom__cond__breadth_lt50",
    "sector_spy_mom_20d__r5__long__cond__sector_disp_20d_q70",
    "sector_spy_sharpe_20d__r5__long__cond__sector_disp_20d_q70",
    "sector_spy_mom_20d__r5__none__cond__stress_disp_20d_q70",
    "sector_spy_mom_120d__r5__long__cond__stress_disp_20d_q60",
    "sector_spy_sharpe_120d__r5__long__cond__sector_disp_20d_q60",
    "sector_spy_sharpe_120d__r5__long__cond__sector_disp_20d_q70",
    "sector_spy_mom_20d__r5__none__cond__sector_disp_20d_q70",
]  # SIGNAL_POOL_SLEEVE_NAMES

OUT_DIR = Path(__file__).parent / "out" / "sleeve_annual_performance"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _short(name: str) -> str:
    return sleeve_short_name(name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    all_rows: list[dict] = []
    # For correlation: use returns from the longest fold (last entry in FOLDS)
    last_fold_net_returns: dict[str, pd.Series] = {}
    last_fold_specs: dict = {}

    for fold_idx, (fold_start, fold_end) in enumerate(FOLDS):
        years = list(range(int(fold_start[:4]), int(fold_end[:4]) + 1))
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}: {fold_start} to {fold_end}")
        print("=" * 60)

        print("  Loading data ...")
        warmup_start = str(int(fold_start[:4]) - WARMUP_YEARS) + fold_start[4:]
        universe, benchmark, factors = pu.load_data(warmup_start, fold_end)
        print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")

        partners = pu.compute_distance_partners(
            universe, train_end=fold_end, train_start=fold_start
        )
        get_distance_partners = lambda: partners  # noqa: E731
        sector_etf_map = pu.get_sector_etf_map_for(universe)
        get_sector_etf_map = lambda: sector_etf_map  # noqa: E731
        sector_map = qs.get_sector_map(list(universe.returns.columns))

        all_specs = build_sleeve_specs(
            get_distance_partners=get_distance_partners,
            get_sector_etf_map=get_sector_etf_map,
        )

        if len(SLEEVES) >= 2:
            missing = [s for s in SLEEVES if s not in all_specs]
            if missing:
                raise ValueError(f"Sleeves not found in sig_fam_utils: {missing}")
            specs = {n: all_specs[n] for n in SLEEVES}
            if fold_idx == 0:
                print(f"  Using {len(specs)} specified sleeves (SLEEVES override active).")
        else:
            specs = all_specs

        print(f"  Running {len(specs)} sleeves ...")
        studies = pu.run_sleeve_pool(
            specs,
            universe,
            benchmark,
            factors,
            sector_map,
            residualize_fit_start=fold_start,
            scaler_start=fold_start,
        )
        print("  Done.")

        spy_returns = benchmark.returns["SPY"]

        for name, study in studies.items():
            positions = study.cache["positions"]
            univ_returns = universe.returns.reindex(columns=positions.columns).fillna(0)
            gross_returns = qs_engine.run(positions, univ_returns)
            to = qs_metrics.turnover(positions)
            net_returns = gross_returns - to * COST_BPS / 10_000

            if fold_idx == len(FOLDS) - 1:
                last_fold_net_returns[name] = net_returns
                last_fold_specs = specs

            full_m = study.metrics_dict()
            full_ns = float(full_m.get("net_sharpe", full_m.get("sharpe", float("nan"))))
            print(f"    {name}: net_sharpe={full_ns:.3f}")

            for year in years:
                yr_net = net_returns[net_returns.index.year == year]
                yr_to = to[to.index.year == year]
                yr_spy = spy_returns.reindex(yr_net.index).fillna(0)

                if yr_net.empty or yr_net.std() == 0:
                    all_rows.append(
                        {
                            "fold_end": fold_end[:4],
                            "sleeve": name,
                            "year": year,
                            "net_sharpe": float("nan"),
                            "ann_return": float("nan"),
                            "ann_vol": float("nan"),
                            "max_drawdown": float("nan"),
                            "avg_daily_turnover": float("nan"),
                            "spy_ann_return": float("nan"),
                        }
                    )
                    continue

                n = len(yr_net)
                ann_ret = float((1 + yr_net).prod() ** (252 / n) - 1)
                ann_vol = float(yr_net.std() * (252**0.5))
                net_sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
                cum = (1 + yr_net).cumprod()
                mdd = float((cum / cum.cummax() - 1).min())
                avg_to = float(yr_to.mean())
                spy_ret = float((1 + yr_spy).prod() ** (252 / n) - 1)

                all_rows.append(
                    {
                        "fold_end": fold_end[:4],
                        "sleeve": name,
                        "year": year,
                        "net_sharpe": net_sharpe,
                        "ann_return": ann_ret,
                        "ann_vol": ann_vol,
                        "max_drawdown": mdd,
                        "avg_daily_turnover": avg_to,
                        "spy_ann_return": spy_ret,
                    }
                )

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "sleeve_annual_performance.csv", index=False)
    print(f"\nSaved sleeve_annual_performance.csv ({len(df)} rows)")

    # ---------------------------------------------------------------------------
    # Heatmaps — one per fold (net Sharpe only)
    # ---------------------------------------------------------------------------
    def _heatmap(
        fold_df: pd.DataFrame,
        sleeve_order: list[str],
        years: list[int],
        metric: str,
        title: str,
        filename: str,
        fmt: str = ".2f",
        cmap: str = "RdBu",
    ) -> None:
        pivot = fold_df.pivot_table(index="sleeve", columns="year", values=metric, aggfunc="first")
        pivot = pivot.reindex(index=sleeve_order, columns=years)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(13, max(6, len(pivot) * 0.38 + 1.5)))
        vmax = np.nanpercentile(pivot.values.astype(float), 95)
        vmin = np.nanpercentile(pivot.values.astype(float), 5)
        bound = max(abs(vmin), abs(vmax))
        im = ax.imshow(
            pivot.values.astype(float), aspect="auto", cmap=cmap, vmin=-bound, vmax=bound
        )
        plt.colorbar(im, ax=ax, label=metric)

        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, fontsize=9)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title(title, pad=10)

        for i in range(len(pivot)):
            for j in range(len(years)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    brightness = (val - (-bound)) / (2 * bound) if bound > 0 else 0.5
                    color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=6, color=color)

        plt.tight_layout()
        plt.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {filename}")

    for fold_start, fold_end in FOLDS:
        fold_label = fold_end[:4]
        years = list(range(int(fold_start[:4]), int(fold_end[:4]) + 1))
        fold_df = df[df["fold_end"] == fold_label]
        sleeve_order = (
            list(last_fold_specs.keys()) if last_fold_specs else list(fold_df["sleeve"].unique())
        )

        _heatmap(
            fold_df,
            sleeve_order,
            years,
            "net_sharpe",
            f"Sleeve Annual Net Sharpe ({fold_start[:4]}–{fold_label})",
            f"sleeve_net_sharpe_heatmap_{fold_label}.png",
        )
        _heatmap(
            fold_df,
            sleeve_order,
            years,
            "ann_return",
            f"Sleeve Annual Net Return ({fold_start[:4]}–{fold_label})",
            f"sleeve_ann_return_heatmap_{fold_label}.png",
            fmt=".1%",
        )

    # ---------------------------------------------------------------------------
    # Correlation heatmap: from the longest fold
    # ---------------------------------------------------------------------------
    returns_df = pd.DataFrame(last_fold_net_returns)
    last_fold_df = df[df["fold_end"] == FOLDS[-1][1][:4]]
    avg_ns = last_fold_df.groupby("sleeve")["net_sharpe"].mean().sort_values(ascending=False)
    col_order = [s for s in avg_ns.index if s in returns_df.columns]
    returns_df = returns_df[col_order]

    corr = returns_df.corr()
    short_names = [_short(s) for s in col_order]

    n_sleeves = len(col_order)
    # Rectangular: fixed cell size so the chart doesn't balloon with many sleeves
    cell_w, cell_h = 0.65, 0.52
    fig_w = max(10, n_sleeves * cell_w + 4.0)  # extra width for y-labels + colorbar
    fig_h = max(4, n_sleeves * cell_h + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr.values, aspect="auto", cmap="RdBu", vmin=-0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Pearson Corr.", fraction=0.02, pad=0.02)

    fold_end_label = FOLDS[-1][1][:4]
    # x-axis: just tick marks with numbers 1..N to save horizontal space
    ax.set_xticks(range(n_sleeves))
    ax.set_xticklabels([str(i + 1) for i in range(n_sleeves)], fontsize=9)
    # y-axis: numbered label + abbreviated name
    ax.set_yticks(range(n_sleeves))
    plt.setp(
        ax.set_yticklabels([f"{i + 1}. {s}" for i, s in enumerate(short_names)], fontsize=9),
        rotation=30,
        ha="right",
    )
    ax.set_title(f"Net Return Correlations ({FOLDS[-1][0][:4]}-{fold_end_label})", pad=8)

    for i in range(n_sleeves):
        for j in range(n_sleeves):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "sleeve_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved sleeve_correlation_heatmap.png")

    # ---------------------------------------------------------------------------
    # Summary: avg net Sharpe per sleeve across all folds
    # ---------------------------------------------------------------------------
    summary = (
        df.groupby("sleeve")["net_sharpe"]
        .agg(["mean", "min", "max", "std"])
        .rename(columns={"mean": "avg_ns", "min": "min_ns", "max": "max_ns", "std": "std_ns"})
        .sort_values("avg_ns", ascending=False)
    )
    print("\nSleeve summary (sorted by avg net Sharpe across all folds):")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
