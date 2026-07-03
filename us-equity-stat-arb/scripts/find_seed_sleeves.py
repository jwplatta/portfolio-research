"""
Identify candidate seed sleeves for walkforward_greedy_portfolio.py.

Loads all signal-sweep yearly results, filters to pool-candidate sleeves,
then ranks by consistency: good average net Sharpe, few negative years,
no extreme drawdowns, reasonable turnover.

Outputs:
  examples/out/seed_sleeve_candidates.csv  — ranked table
  examples/out/seed_sleeve_candidates.png  — scatter of avg vs min annual SR
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from constants import OUT_ROOT

OUT_DIR = OUT_ROOT
SWEEP_OUT = Path(__file__).parent.parent / "signal_sweeps" / "out"

# ---------------------------------------------------------------------------
# Thresholds — tune these to taste
# ---------------------------------------------------------------------------
MIN_YEARS_COVERED = 7          # must have data for at least this many years
MIN_AVG_NET_SHARPE = 0.15      # average annual net Sharpe across all years
MIN_ANNUAL_NET_SHARPE = -0.50  # worst single year allowed
MAX_PCT_NEGATIVE_YEARS = 0.45  # at most this fraction of years can be negative
MAX_AVG_DRAWDOWN = -0.25       # average annual max drawdown (less negative = better)
MAX_WORST_DRAWDOWN = -0.50     # single-year worst drawdown allowed
MAX_AVG_TURNOVER = 2.0         # average daily turnover cap (round-trips)
# Minimum turnover: conditionally-active sleeves (regime-gated, breadth-conditioned) are
# off most of the time. When inactive, returns are ~0 — not negative — so they score
# artificially well on pct_negative_years and consistency. Require a meaningful minimum
# turnover to ensure the sleeve is actually doing work most of the time.
MIN_AVG_TURNOVER = 0.005       # ~0.5% daily one-way; filters out mostly-dormant sleeves


def load_yearly() -> pd.DataFrame:
    dfs = []
    for f in SWEEP_OUT.glob("*/signal_sweep_*_yearly.csv"):
        dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True)


def load_pool_candidates() -> set[str]:
    dfs = []
    for f in SWEEP_OUT.glob("*/signal_sweep_*_pool_candidates.csv"):
        dfs.append(pd.read_csv(f))
    combined = pd.concat(dfs, ignore_index=True)
    return set(combined["name"].unique())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    yearly = load_yearly()
    pool_names = load_pool_candidates()

    print(f"Total sleeves in yearly data : {yearly['name'].nunique()}")
    print(f"Pool-candidate sleeves       : {len(pool_names)}")

    # Restrict to pool candidates
    yearly = yearly[yearly["name"].isin(pool_names)].copy()
    print(f"Rows after pool-candidate filter: {len(yearly)}")

    # ---------------------------------------------------------------------------
    # Per-sleeve aggregate stats
    # ---------------------------------------------------------------------------
    grp = yearly.groupby("name")

    agg = pd.DataFrame(
        {
            "n_years": grp["year"].count(),
            "avg_net_sharpe": grp["net_sharpe"].mean(),
            "min_net_sharpe": grp["net_sharpe"].min(),
            "max_net_sharpe": grp["net_sharpe"].max(),
            "std_net_sharpe": grp["net_sharpe"].std(),
            "pct_negative_years": grp["net_sharpe"].apply(lambda s: (s < 0).mean()),
            "avg_max_drawdown": grp["max_drawdown"].mean(),
            "worst_max_drawdown": grp["max_drawdown"].min(),
            "avg_turnover": grp["avg_daily_turnover"].mean(),
        }
    ).reset_index()

    # Consistency score: information ratio of annual returns (mean/std)
    agg["consistency_score"] = agg["avg_net_sharpe"] / agg["std_net_sharpe"].replace(0, float("nan"))

    print(f"\nBefore filters: {len(agg)} sleeves")

    # ---------------------------------------------------------------------------
    # Apply filters
    # ---------------------------------------------------------------------------
    mask = (
        (agg["n_years"] >= MIN_YEARS_COVERED)
        & (agg["avg_net_sharpe"] >= MIN_AVG_NET_SHARPE)
        & (agg["min_net_sharpe"] >= MIN_ANNUAL_NET_SHARPE)
        & (agg["pct_negative_years"] <= MAX_PCT_NEGATIVE_YEARS)
        & (agg["avg_max_drawdown"] >= MAX_AVG_DRAWDOWN)
        & (agg["worst_max_drawdown"] >= MAX_WORST_DRAWDOWN)
        & (agg["avg_turnover"] >= MIN_AVG_TURNOVER)
        & (agg["avg_turnover"] <= MAX_AVG_TURNOVER)
    )
    candidates = agg[mask].copy()
    print(f"After filters : {len(candidates)} sleeves")

    # ---------------------------------------------------------------------------
    # Rank: primary = consistency_score, secondary = avg_net_sharpe
    # ---------------------------------------------------------------------------
    candidates = candidates.sort_values(
        ["consistency_score", "avg_net_sharpe"], ascending=False
    ).reset_index(drop=True)
    candidates.index += 1  # 1-based rank
    candidates.index.name = "rank"

    # Save
    out_csv = OUT_DIR / "seed_sleeve_candidates.csv"
    candidates.to_csv(out_csv)
    print(f"\nSaved {out_csv}")
    print(candidates[["name", "avg_net_sharpe", "min_net_sharpe", "pct_negative_years",
                       "avg_max_drawdown", "avg_turnover", "consistency_score"]].head(20).to_string())

    # ---------------------------------------------------------------------------
    # Chart: avg vs min annual Sharpe, coloured by pct_negative_years
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        candidates["avg_net_sharpe"],
        candidates["min_net_sharpe"],
        c=candidates["pct_negative_years"],
        cmap="RdYlGn_r",
        s=60,
        alpha=0.8,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Fraction of negative years")

    # Label top 10
    for _, row in candidates.head(10).iterrows():
        ax.annotate(
            row["name"],
            xy=(row["avg_net_sharpe"], row["min_net_sharpe"]),
            fontsize=6,
            xytext=(4, 2),
            textcoords="offset points",
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Average annual net Sharpe")
    ax.set_ylabel("Worst single-year net Sharpe")
    ax.set_title("Seed sleeve candidates — consistency view\n(top 10 labelled)")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_png = OUT_DIR / "seed_sleeve_candidates.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")

    # ---------------------------------------------------------------------------
    # Print the SEED_SLEEVE list ready to paste into walkforward_greedy_portfolio.py
    # Names in sweep CSVs omit __cond__none; look up the canonical pool name.
    # ---------------------------------------------------------------------------
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
    from sig_fam_utils import SIGNAL_POOL_SLEEVE_NAMES

    pool_set = set(SIGNAL_POOL_SLEEVE_NAMES)

    def _to_pool_name(sweep_name: str) -> str:
        """Convert a sweep CSV name to its pool-canonical form.

        Sweep CSV names use one of two formats:
          - No filter:   {signal}__{rebalance}__{scaler}
          - With filter: {signal}__{rebalance}__{filter}   (scaler segment omitted)

        Pool canonical format is always:
          {signal}__{rebalance}__{scaler}__cond__{filter}
        """
        if sweep_name in pool_set:
            return sweep_name
        # Try appending __cond__none (unconditioned sleeve missing the cond suffix)
        candidate = sweep_name + "__cond__none"
        if candidate in pool_set:
            return candidate
        # Try inserting __none__cond__ before the last __ segment — sweep names for
        # conditioned sleeves omit the scaler segment entirely, producing
        # {signal}__{rebalance}__{filter} instead of {signal}__{rebalance}__none__cond__{filter}
        parts = sweep_name.rsplit("__", 1)
        if len(parts) == 2:
            candidate = parts[0] + "__none__cond__" + parts[1]
            if candidate in pool_set:
                return candidate
        # Sweep names for long-only conditioned sleeves encode the scaler as a suffix on
        # the filter tag (e.g. sector_disp_20d_q70_long). Try stripping _long suffix and
        # inserting __long__cond__ before the filter.
        if sweep_name.endswith("_long"):
            base = sweep_name[:-5]  # strip _long
            parts2 = base.rsplit("__", 1)
            if len(parts2) == 2:
                candidate = parts2[0] + "__long__cond__" + parts2[1]
                if candidate in pool_set:
                    return candidate
        # Sweep tag combines filter+scaler (e.g. sector_disp_20d_q60_t50): try known
        # scaler suffixes and split them off to reconstruct the pool name.
        for scaler_suffix, scaler_key in [("_t50", "trend_50_200_mom"), ("_t20", "trend_20_100_mr")]:
            if scaler_suffix in sweep_name:
                without = sweep_name.replace(scaler_suffix, "")
                parts3 = without.rsplit("__", 1)
                if len(parts3) == 2:
                    candidate = parts3[0] + f"__{scaler_key}__cond__" + parts3[1]
                    if candidate in pool_set:
                        return candidate
        return sweep_name

    print("\n# Top 10 seed sleeve candidates — paste into SEED_SLEEVE:")
    print("# (names shown are the full pool names with __cond__ suffix)")
    for rank, row in candidates.head(10).iterrows():
        sweep_name = row["name"]
        pool_name = _to_pool_name(sweep_name)
        in_pool = pool_name in pool_set
        flag = "" if in_pool else "  [NOT IN CURRENT POOL]"
        print(
            f'  #{rank:2d}  avg_SR={row["avg_net_sharpe"]:.2f}  '
            f'min_SR={row["min_net_sharpe"]:.2f}  '
            f'neg_yrs={row["pct_negative_years"]:.0%}  '
            f'"{pool_name}"{flag}'
        )


if __name__ == "__main__":
    main()
