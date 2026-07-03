"""Shared utilities for signal sweep scripts.

Each signal-family script calls run_sweep() with its own
make_signals(), make_scaler_configs(), and build_study_fn() implementations.

build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False)
must return a completed Study (already .run()) or None on failure.

build_study_generic() provides a shared implementation of build_study_fn for sweeps
that follow the standard pipeline — see its docstring for details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import qstudy as qs
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
WARMUP_YEARS = 1

COST_BPS = 10.0
N_LONG = 20
N_SHORT = 20
REBALANCE_PERIODS = [1, 5, 10, 21]
BENCHMARK_TICKER = "SPY"
FACTOR_TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB"]

EVAL_START = TRAIN_START  # first year to include in per-year breakdown
EVAL_END = TRAIN_END


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple:
    """Load universe/benchmark/factors for the full period with warmup."""
    warmup_start = str(int(TRAIN_START[:4]) - WARMUP_YEARS) + TRAIN_START[4:]
    print(f"Loading data ({warmup_start} to {TRAIN_END}) ...")
    universe = qs.download(index_code="SP500", start=warmup_start, end=TRAIN_END)
    benchmark = qs.download([BENCHMARK_TICKER], start=warmup_start, end=TRAIN_END)
    factors = qs.download(FACTOR_TICKERS, start=warmup_start, end=TRAIN_END)
    print(f"Universe: {universe.returns.shape[0]} days × {universe.returns.shape[1]} tickers")
    return universe, benchmark, factors


# ---------------------------------------------------------------------------
# Evaluation: full period + per-year breakdown
# ---------------------------------------------------------------------------


def _compute_metrics(
    net_ret: pd.Series, gross_ret: pd.Series, to: pd.Series, period: str = ""
) -> dict:
    """Compute standard metrics for a returns series."""
    n_net = len(net_ret.dropna())
    n_gross = len(gross_ret.dropna())
    if n_net < 20 or net_ret.std() == 0:
        return {}

    ann_ret_net = float((1 + net_ret).prod() ** (252 / n_net) - 1)
    ann_vol_net = float(net_ret.std() * (252**0.5))
    net_sharpe = ann_ret_net / ann_vol_net if ann_vol_net > 0 else float("nan")

    ann_ret_gross = float((1 + gross_ret).prod() ** (252 / n_gross) - 1)
    ann_vol_gross = float(gross_ret.std() * (252**0.5))
    gross_sharpe = ann_ret_gross / ann_vol_gross if ann_vol_gross > 0 else float("nan")

    cum = (1 + net_ret).cumprod()
    mdd = float((cum / cum.cummax() - 1).min())
    avg_to = float(to.mean())

    return {
        "net_sharpe": net_sharpe,
        "gross_sharpe": gross_sharpe,
        "ann_return": ann_ret_net,
        "ann_vol": ann_vol_net,
        "max_drawdown": mdd,
        "avg_daily_turnover": avg_to,
    }


def eval_full_period(
    positions: pd.DataFrame,
    universe_returns: pd.DataFrame,
    eval_start: str = EVAL_START,
    eval_end: str = EVAL_END,
) -> tuple[dict, dict[str, dict], pd.Series | None]:
    """Evaluate positions over the full period and break down by calendar year.

    Returns:
        full_metrics: aggregate metrics over eval_start..eval_end
        yearly_metrics: {year_str: metrics_dict} for each calendar year
        net_ret: full net returns series (for correlation heatmap)
    """
    pos = positions.loc[eval_start:eval_end]
    if pos.empty:
        return {}, {}, None

    univ = universe_returns.loc[eval_start:eval_end].reindex(columns=pos.columns).fillna(0)
    gross_ret = qs_engine.run(pos, univ)
    to = qs_metrics.turnover(pos).reindex(gross_ret.index).fillna(0)
    net_ret = gross_ret - to * COST_BPS / 10_000

    full_metrics = _compute_metrics(net_ret, gross_ret, to)
    if not full_metrics:
        return {}, {}, None

    # Per-year breakdown
    yearly_metrics: dict[str, dict] = {}
    for year, grp in net_ret.groupby(net_ret.index.year):
        yr = str(year)
        g_yr = gross_ret.reindex(grp.index).fillna(0)
        to_yr = to.reindex(grp.index).fillna(0)
        m = _compute_metrics(grp, g_yr, to_yr)
        if m:
            yearly_metrics[yr] = m

    return full_metrics, yearly_metrics, net_ret


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------


def _write_run_log(group: str, out_dir: Path) -> None:
    """Append or update a row in the shared run log CSV (signal_sweeps/out/run_log.csv)."""
    from datetime import datetime
    import zoneinfo

    log_path = out_dir.parent / "run_log.csv"
    cst = zoneinfo.ZoneInfo("America/Chicago")
    completed_at = datetime.now(tz=cst).strftime("%Y-%m-%d %H:%M:%S CST")

    if log_path.exists():
        log_df = pd.read_csv(log_path)
    else:
        log_df = pd.DataFrame(columns=["script", "completed_at"])

    script = f"signal_sweep_{group}"
    log_df = log_df[log_df["script"] != script]  # remove old entry if exists
    log_df = pd.concat(
        [log_df, pd.DataFrame([{"script": script, "completed_at": completed_at}])],
        ignore_index=True,
    ).sort_values("script").reset_index(drop=True)
    log_df.to_csv(log_path, index=False)


def run_sweep(
    group: str,
    signals: list[dict],
    scaler_configs: list[dict],
    rebalance_periods: list[int],
    build_study_fn: Callable,
    out_dir: Path,
) -> pd.DataFrame:
    """Run a full-period sweep for one signal family.

    build_study_fn(entry, rebalance, scaler_cfg, universe, benchmark, factors, verbose=False)
    must return a completed Study or None on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(signals) * len(rebalance_periods) * len(scaler_configs)
    print(
        f"{group}: {len(signals)} signals × {len(rebalance_periods)} rebalance × "
        f"{len(scaler_configs)} scalers = {total} configs\n"
    )

    universe, benchmark, factors = load_data()

    summary_rows: list[dict] = []  # one row per config (full-period metrics)
    yearly_rows: list[dict] = []  # one row per config × year
    returns_by_name: dict[str, pd.Series] = {}

    with tqdm(total=total, desc=group, unit="cfg") as pbar:
        for entry in signals:
            for rebalance in rebalance_periods:
                for scaler_cfg in scaler_configs:
                    name = f"{entry['name']}__r{rebalance}__{scaler_cfg['tag']}"
                    filters_str = entry.get("filters", "")
                    pbar.set_postfix(cfg=name[:40])

                    try:
                        study = build_study_fn(
                            entry,
                            rebalance,
                            scaler_cfg,
                            universe,
                            benchmark,
                            factors,
                            verbose=False,
                        )
                    except Exception as exc:
                        pbar.write(f"  SKIP {name}: {exc}")
                        pbar.update(1)
                        continue

                    if study is None:
                        pbar.update(1)
                        continue

                    positions = study.cache.get("positions")
                    if positions is None or positions.empty:
                        pbar.update(1)
                        continue

                    full_m, yearly_m, net_ret = eval_full_period(positions, universe.returns)
                    pbar.update(1)

                    if not full_m:
                        continue

                    base = {
                        "name": name,
                        "base_signal": entry["name"],
                        "filters": filters_str,
                        "rebalance": rebalance,
                        "scaler_config": scaler_cfg["tag"],
                    }
                    summary_rows.append({**base, **full_m})

                    for yr, m in yearly_m.items():
                        yearly_rows.append({**base, "year": yr, **m})

                    if net_ret is not None:
                        returns_by_name[name] = net_ret

    summary_df = (
        pd.DataFrame(summary_rows).sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    )
    yearly_df = pd.DataFrame(yearly_rows)

    if summary_df.empty:
        print("No results.")
        return summary_df

    # Insert avg_annual_net_sharpe and min_annual_net_sharpe right after net_sharpe
    avg_annual = yearly_df.groupby("name")["net_sharpe"].mean().rename("avg_annual_net_sharpe")
    min_annual = yearly_df.groupby("name")["net_sharpe"].min().rename("min_annual_net_sharpe")
    ns_idx = summary_df.columns.get_loc("net_sharpe") + 1
    summary_df.insert(ns_idx, "min_annual_net_sharpe", summary_df["name"].map(min_annual))
    summary_df.insert(ns_idx, "avg_annual_net_sharpe", summary_df["name"].map(avg_annual))

    summary_path = out_dir / f"signal_sweep_{group}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    yearly_path = out_dir / f"signal_sweep_{group}_yearly.csv"
    yearly_df.to_csv(yearly_path, index=False)
    print(f"Saved {yearly_path}")

    if returns_by_name:
        returns_df = pd.DataFrame(returns_by_name)
        returns_path = out_dir / f"signal_sweep_{group}_returns.csv"
        returns_df.to_csv(returns_path)
        print(f"Saved {returns_path}")

    _write_heatmap(summary_df, yearly_df, group, out_dir)
    _write_corr_heatmap(summary_df, returns_by_name, group, out_dir)
    _write_pool_candidates(summary_df, yearly_df, returns_by_name, group, out_dir)
    _write_run_log(group, out_dir)
    return summary_df


# ---------------------------------------------------------------------------
# Pool candidates output
# ---------------------------------------------------------------------------

N_TOP_CANDIDATES = 15
MIN_BEST_YEAR_SHARPE = 1.0
MIN_BEST_MIN_SHARPE = 0.0
MIN_ANNUAL_FLOOR = -0.8  # worst active year floor: used for best_YYYY guard and top_min_annual
MIN_ACTIVE_YEARS = 4  # always-on sleeves with fewer non-NaN years are excluded as sparse
N_COMPLEMENTS = 2


def _write_pool_candidates(
    summary_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    returns_by_name: dict[str, pd.Series],
    group: str,
    out_dir: Path,
) -> None:
    rows: list[dict] = []
    selected: set[str] = set()

    def add(name: str, reason: str) -> bool:
        if name not in selected:
            selected.add(name)
            rows.append({"name": name, "reason": reason})
            return True
        return False

    # Exclude always-on sleeves with too few non-NaN years (sparse/failed runs).
    # Gated sleeves are exempt — NaN years just mean the regime wasn't active.
    always_on_names = yearly_df[yearly_df["filters"].fillna("").str.len() == 0]["name"].unique()
    active_year_counts = (
        yearly_df[yearly_df["name"].isin(always_on_names)]
        .groupby("name")["net_sharpe"]
        .apply(lambda s: s.notna().sum())
    )
    sparse = active_year_counts[active_year_counts < MIN_ACTIVE_YEARS].index
    summary_df = summary_df[~summary_df["name"].isin(sparse)].copy()

    if summary_df.empty:
        print("No pool candidates after sparse filter.")
        return

    avg_annual = summary_df.set_index("name")["avg_annual_net_sharpe"]
    family_avg_threshold = float(avg_annual.mean())

    # For gated sleeves (non-empty filters), only consider years where the sleeve
    # was active (net_sharpe != 0.0) when computing the min floor guard.
    # For always-on sleeves, use all years.
    gated = yearly_df.copy()
    is_gated = gated["filters"].fillna("").str.len() > 0
    active_rows = gated[~is_gated | (gated["net_sharpe"] != 0.0)]
    active_min = active_rows.groupby("name")["net_sharpe"].min().rename("active_min_net_sharpe")
    summary_df = summary_df.copy()
    summary_df["active_min_net_sharpe"] = summary_df["name"].map(active_min)

    def pick_with_group_coverage(df: pd.DataFrame, metric: str, n: int, reason: str) -> None:
        """Pick top 1 per base_signal group first, then fill remaining slots globally
        allowing at most 4 per group so dominant groups get their best scaler variants
        without crowding out weaker-but-distinct signal families entirely."""
        ranked = df.sort_values(metric, ascending=False).reset_index(drop=True)
        base_by_name = dict(zip(ranked["name"], ranked["base_signal"]))
        group_counts: dict[str, int] = {}

        # Pass 1: one per group, up to n
        for _, row in ranked.drop_duplicates(subset="base_signal").iterrows():
            if len([r for r in rows if r["reason"] == reason]) >= n:
                break
            add(row["name"], reason)
            group_counts[row["base_signal"]] = 1

        # Pass 2: fill to n, max 4 per group
        for _, row in ranked.iterrows():
            if len([r for r in rows if r["reason"] == reason]) >= n:
                break
            base = base_by_name[row["name"]]
            if group_counts.get(base, 0) < 4:
                if add(row["name"], reason):
                    group_counts[base] = group_counts.get(base, 0) + 1

    # Top N by avg_annual_net_sharpe: must clear a minimum threshold — crash handling is portfolio-level
    MIN_AVG_THRESHOLD = 0.15
    eligible_avg = summary_df[summary_df["avg_annual_net_sharpe"] > MIN_AVG_THRESHOLD]
    pick_with_group_coverage(
        eligible_avg, "avg_annual_net_sharpe", N_TOP_CANDIDATES, "top_avg_annual"
    )

    # Top N by min_annual_net_sharpe: 1 per base_signal group, then fill to N_TOP_CANDIDATES
    eligible_min = summary_df[
        (summary_df["active_min_net_sharpe"] > MIN_BEST_MIN_SHARPE)
        & (summary_df["avg_annual_net_sharpe"] > family_avg_threshold)
    ]
    pick_with_group_coverage(
        eligible_min, "min_annual_net_sharpe", N_TOP_CANDIDATES, "top_min_annual"
    )

    # Best sleeve per year (net_sharpe > 1.0 in that year), guarded by active min floor and avg threshold
    names_above_floor = summary_df[
        (summary_df["active_min_net_sharpe"] > MIN_ANNUAL_FLOOR)
        & (summary_df["avg_annual_net_sharpe"] > MIN_AVG_THRESHOLD)
    ]["name"]
    for year, grp in yearly_df.groupby("year"):
        eligible = grp[
            (grp["net_sharpe"] > MIN_BEST_YEAR_SHARPE) & (grp["name"].isin(names_above_floor))
        ]
        if not eligible.empty:
            best = eligible.nlargest(1, "net_sharpe").iloc[0]["name"]
            add(best, f"best_{year}")

    # Sleeve with highest min_annual_net_sharpe per year, where min > 0.0
    # i.e. for each year, find the sleeve whose worst year across all years is > 0
    # and had the best floor, among those that were active in this year
    all_min = yearly_df.groupby("name")["net_sharpe"].min()
    positive_floor = all_min[all_min > MIN_BEST_MIN_SHARPE].index
    positive_avg = summary_df[summary_df["avg_annual_net_sharpe"] > MIN_AVG_THRESHOLD]["name"]
    yearly_pos = yearly_df[
        yearly_df["name"].isin(positive_floor) & yearly_df["name"].isin(positive_avg)
    ]
    for year, grp in yearly_pos.groupby("year"):
        if not grp.empty:
            best = grp.nlargest(1, "net_sharpe").iloc[0]["name"]
            add(best, f"best_min_{year}")

    candidates_df = pd.DataFrame(rows)
    path = out_dir / f"signal_sweep_{group}_pool_candidates.csv"
    if candidates_df.empty:
        print("No pool candidates selected.")

        return
    # Join in key metrics
    metrics = summary_df[["name", "net_sharpe", "avg_annual_net_sharpe", "min_annual_net_sharpe"]]
    candidates_df = candidates_df.merge(metrics, on="name", how="left")

    candidates_df.to_csv(path, index=False)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Heatmap output
# ---------------------------------------------------------------------------

TOP_N_HEATMAP = 30
METRIC_COLS = [
    "net_sharpe",
    "gross_sharpe",
    "ann_return",
    "ann_vol",
    "max_drawdown",
    "avg_daily_turnover",
]


def _write_heatmap(
    summary_df: pd.DataFrame, yearly_df: pd.DataFrame, group: str, out_dir: Path
) -> None:
    """Net Sharpe heatmap: top N configs × calendar year."""
    top_names = list(summary_df.head(TOP_N_HEATMAP)["name"])

    pivot = yearly_df[yearly_df["name"].isin(top_names)].pivot_table(
        index="name", columns="year", values="net_sharpe", aggfunc="mean"
    )
    # Preserve summary ranking order
    pivot = pivot.reindex([n for n in top_names if n in pivot.index])
    years = sorted(pivot.columns)
    pivot = pivot.reindex(columns=years)

    bound = max(
        abs(np.nanpercentile(pivot.values.astype(float), 5)),
        abs(np.nanpercentile(pivot.values.astype(float), 95)),
        0.5,
    )

    n_sleeves = len(pivot)
    n_years = len(years)
    fig_w = max(10, n_years * 1.5 + 4)
    fig_h = max(8, n_sleeves * 0.35 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        pivot.values.astype(float),
        aspect="auto",
        cmap="RdYlGn",
        vmin=-bound,
        vmax=bound,
    )
    plt.colorbar(im, ax=ax, label="Net Sharpe", fraction=0.03, pad=0.02)

    ax.set_xticks(range(n_years))
    ax.set_xticklabels(years, fontsize=10)
    ax.set_yticks(range(n_sleeves))
    ax.set_yticklabels(pivot.index, fontsize=7, rotation=0, ha="right")
    ax.set_title(
        f"{group} — Net Sharpe by Year ({years[0]}–{years[-1]})\n"
        f"Full period, {COST_BPS:.0f} bps costs, top {n_sleeves} by full-period Sharpe",
        pad=12,
        fontsize=11,
    )

    for i in range(n_sleeves):
        for j in range(n_years):
            val = pivot.values[i, j]
            if not np.isnan(val):
                brightness = (val - (-bound)) / (2 * bound) if bound > 0 else 0.5
                color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.tight_layout()
    png_path = out_dir / f"signal_sweep_{group}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")


def _write_corr_heatmap(
    summary_df: pd.DataFrame,
    returns_by_name: dict[str, pd.Series],
    group: str,
    out_dir: Path,
) -> None:
    """Correlation heatmap of full-period net returns for top N configs."""
    top_names = [n for n in summary_df.head(TOP_N_HEATMAP)["name"] if n in returns_by_name]

    if len(top_names) < 2:
        return

    ret_df = pd.DataFrame({n: returns_by_name[n] for n in top_names})
    corr = ret_df.corr()

    n = len(corr)
    fig_size = max(10, n * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson correlation", fraction=0.03, pad=0.02)

    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, fontsize=6, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title(
        f"{group} — Return Correlations (top {n} by full-period net Sharpe)\n"
        f"Full period {EVAL_START[:4]}–{EVAL_END[:4]}",
        pad=12,
        fontsize=10,
    )

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            brightness = (val + 1) / 2
            color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

    plt.tight_layout()
    corr_path = out_dir / f"signal_sweep_{group}_corr.png"
    plt.savefig(corr_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {corr_path}")


# ---------------------------------------------------------------------------
# Generic study builder
# ---------------------------------------------------------------------------

def build_study_generic(
    entry: dict,
    rebalance: int,
    scaler_cfg: dict,
    universe,
    benchmark,
    factors,
    *,
    equity_curve_scaler,
    verbose: bool = False,
    include_transaction_costs: bool = True,
):
    """Generic build_study_fn implementation for standard-pipeline sweeps.

    Handles the common pattern shared by most sweep scripts:
      1. Optional factor-model residualization (entry["use_factor_model"] = True)
      2. Optional ETF-factor residualization (entry["use_residual"] = True)
      3. Optional conditioning filter (entry["cond_filter"] = <callable>)
      4. Standard pipeline: base_signal → liquidity → rank_transform →
         build_long_short → fully_invest → equity_curve_scaler
      5. Additional scalers from scaler_cfg via apply_scalers()

    Args:
        entry: Signal descriptor dict with keys "fn", optionally "use_residual",
               "use_factor_model", "cond_filter".
        rebalance: Rebalance period in days.
        scaler_cfg: Scaler config dict; passed to apply_scalers().
        universe, benchmark, factors: StudyData objects from load_data().
        equity_curve_scaler: Pre-built equity-curve regime scaler (always applied first).
        verbose: Passed to Study().
        include_transaction_costs: Set False for sweeps that omit costs.

    Returns:
        Completed Study (already .run()).
    """
    # Import here to avoid circular dependency (sweep_scalers imports from sig_fam_utils,
    # not from signal_sweep_utils).
    from sweep_scalers import apply_scalers

    fn = entry["fn"]
    use_residual = entry.get("use_residual", False)
    use_factor_model = entry.get("use_factor_model", False)
    cond_filter = entry.get("cond_filter")

    builder = qs.Study(universe=universe, benchmark=benchmark, factors=factors, verbose=verbose)

    if use_residual:
        if use_factor_model:
            builder = builder.add_factor_model(
                factors=["market", "sector"],
                sector_map=qs.get_sector_map(list(universe.returns.columns)),
            )
        builder = builder.residualize_returns(fit_start=TRAIN_START)

    builder = (
        builder
        .base_signal(fn)
        .add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
    )

    if cond_filter is not None:
        builder = builder.add_filter(cond_filter)

    builder = (
        builder
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=equity_curve_scaler)
    )

    builder = apply_scalers(builder, scaler_cfg)

    builder = builder.rebalance(every=rebalance)
    if include_transaction_costs:
        builder = builder.with_transaction_costs(cost_bps=COST_BPS)

    return builder.run()
