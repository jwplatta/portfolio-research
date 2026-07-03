"""
Walkforward validation of selected thematic portfolio combinations.

Two groups are evaluated:
  1. Top 5 portfolios from portfolio_combinations.py (by full-period net Sharpe)
  2. Portfolio 9 (Core + breadth + sector) and four variants each adding one
     candidate sleeve targeting the 2023 high-momentum regime gap.

For each portfolio x fold the sleeve positions are run on the full combined
window (train + val), then sliced to train/val periods for evaluation. Weights
are re-estimated from the training slice only (best of 4 schemes by training SR).

Folds:
  Fold 1: train 2015-2020, val 2021
  Fold 2: train 2015-2021, val 2022
  Fold 3: train 2015-2022, val 2023

Outputs (examples/out/walkforward_portfolio_combinations/):
  - walkforward_pc_fold_results.csv   — train/val metrics per portfolio x fold
  - walkforward_pc_val_returns.csv    — daily val returns per portfolio x fold
  - walkforward_pc_val_sharpe_bar.png — bar chart: val net Sharpe / drawdown / turnover per portfolio x fold
  - walkforward_pc_val_summary.csv    — avg val metrics per portfolio across all folds
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
from sig_fam_utils import build_sleeve_specs

import qstudy as qs
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FOLDS = [
    ("2015-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("2015-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
]

COST_BPS = 10.0
WARMUP_YEARS = 1
WEIGHTING_SCHEMES = ["equal", "equal_vol", "equal_sharpe"]
# cumret_spread_20_252__r5__vol_20_60__cond__none
# dist_mr_k3_z60__r21__none__cond__none
# etf_factor_resid_mr_5d__r10__trend_50_200__cond__none
# ivol_explosion_5_60_p90__r10__none__cond__none

# ---------------------------------------------------------------------------
# Portfolio construction narrative
#
# === Chapter 1: Early greedy run (small pool, gap_accum seed) ===
#
# Output of an earlier greedy walkforward run before the full sleeve pool was
# assembled. gap_accum was used as the fixed seed. The result is an ungated,
# MR-heavy portfolio that happens to tell a clean story about orthogonal MR
# mechanisms — but that story is post-hoc; the greedy process selected it.
#
#   CORE3: gap_accum (event-driven gap fill), dist_mr_z60 (pairs MR, slow mean),
#          monoton_skip (skip-period reversal, breadth-gated)
#   +z20:  same pairs framework, 20d window — faster dislocation capture
#   +cumret: cross-sectional style/sector MR — different mechanism from pairs
#   +bear:   crisis alpha, active only in bear+narrow regime — zero negative IS years
#
# === Chapter 2: Full greedy run (full pool, NBO50 seed) ===
#
# Seed: narrow_bull_off_50 (dist_mr_k1_z60 with narrow-bull gate) — selected as
# the single best seed across greedy walkforward runs. Then iteratively add
# the sleeve that most improves walkforward Sharpe, measured across 5 IS folds.
#
# The greedy process consistently selected the same 4 additions in 10+/15 slots
# across 3 different seed variants: bear_reversal, cumret_spread, monoton_35,
# sector_spy_mom_20d. Together these become GREEDY_CORE.
#
# GREEDY_CORE improves on the early greedy baseline (avg val SR 1.67 vs 1.47) but
# has a weak 2022 fold (SR 0.11) — no active sleeve during a trending bear market.
#
# === Chapter 3: Extending GREEDY_CORE — what helps, what doesn't ===
#
# Test single additions to GREEDY_CORE. Evaluated purely on IS walkforward.
#   + gap_accum:  strongest single addition; fixes 2022 fold (SR 0.11 → 1.34);
#                 avg val SR 1.95 — highest of any 6-sleeve portfolio tested
#   + rzscore:    consistent low-vol diversifier; modest per-fold gains but
#                 never hurts; lowest max drawdown of all portfolios tested
#   + gap + rzscore: best avg val SR overall (1.98); rzscore adds improvement
#                 over gap_accum alone without adding vol or drawdown
#   + rgap:       high IS Sharpe but driven by a single dominant fold (2020);
#                 inconsistent across other folds — not robust IS
#   + z20:        marginal gain over core; dominated by gap_accum
#   + betamom:    increases vol and drawdown without proportional Sharpe gain
#   + vol_accel:  strong in high-vol folds but raises avg drawdown and vol;
#                 Sharpe/vol tradeoff worse than gap_accum
#
# === Final candidates (IS-only basis) ===
#   A. GREEDY_CORE alone — defensible baseline, avg val SR 1.67
#   B. GREEDY_CORE + gap_accum — fixes 2022 regime gap, avg val SR 1.95
#   C. GREEDY_CORE + gap_accum + rzscore — best avg val SR 1.98, lowest drawdown
# ---------------------------------------------------------------------------

_Z20 = "dist_mr_k1_z20__r21__none__cond__none"
_Z60 = "dist_mr_k1_z60__r21__none__cond__none"
_GAP = "gap_accum_3d__r21__trend_20_100_off__cond__none"
_MONO = "monoton_skip_252d__r21__breadth_40_off__cond__none"
_CUMRET = "cumret_spread_20_252__r5__vol_20_60__cond__none"
_BEAR = "bear_reversal_20d__r21__trend_20_100_mr__cond__bear_narrow_lt40"
_K3Z20 = "dist_mr_k3_z20__r10__none__cond__low_disp_off_q30"
_SSM20 = "sector_spy_mom_20d__r5__none__cond__stress_disp_20d_q70"
_MONO35 = "monoton_skip_252d__r21__breadth_35_off__cond__none"
_NBO50 = "dist_mr_k1_z60__r5__none__cond__narrow_bull_off_50"
_RGAP = "resid_gap_accum_5d__r10__vol_10_60_off__cond__none"
_LOWVOLMOM = "low_vol_mom_120d__r5__trend_50_200_mom__cond__breadth_lt50"
_VOLACCEL = "vol_accel_20_120d__r10__vol_10_60_up__cond__breadth_lt40"
_K3Z20R21 = "dist_mr_k3_z20__r21__none__cond__none"
_SSM120S5 = "sector_spy_mom_120d_skip5__r21__trend_50_200_mom__cond__sector_disp_20d_q60"
_BETAMOM = "beta_momentum_60_252d__r21__vol_20_60__cond__breadth_lt50"
_RZSCORE = "resid_zscore_w15_w10__r10__trend_20_100__cond__none"
_SCL120W = "sector_rel_cumlog_within_120d__r21__long__cond__stress_disp_20d_q70"
_SCL120W_LS = "sector_rel_cumlog_within_120d__r5__none__cond__stress_disp_20d_q70"
_SCL20W = "sector_rel_cumlog_within_20d__r21__trend_50_200_mom__cond__sector_disp_20d_q70"
_SSMX120 = "sector_spy_mom_x_stock_120d__r5__long__cond__stress_disp_20d_q60"

CORE3 = [_GAP, _Z60, _MONO]
CORE4 = CORE3 + [_Z20]
CORE5 = CORE4 + [_CUMRET]
FINAL = CORE5 + [_BEAR]

# Greedy-derived core: seed (narrow_bull_off_50) + 4 sleeves selected in 10+/15
# slots across all 3 seed variants. Data-driven construction from walkforward results.
GREEDY_FOUR = [_NBO50, _BEAR, _CUMRET, _MONO35]
GREEDY_CORE = [_NBO50, _BEAR, _CUMRET, _MONO35, _SSM20]

ALL_PORTFOLIOS = [
    # -----------------------------------------------------------------------
    # Chapter 1: Early greedy run (small pool, gap_accum seed)
    # -----------------------------------------------------------------------
    # Output of an earlier greedy walkforward run using a smaller sleeve pool
    # with gap_accum as the fixed seed. Ungated MR-heavy result; included as a
    # baseline to show what the greedy process produces without the full pool
    # and without regime-gated seed candidates.
    (
        "1. Early greedy\n(ga3 seed, small pool)\n(ga3 + dp1z60 + ms252/b40x\n+ dp1z20 + cs20)",
        CORE5,
    ),
    ("1. Early greedy + brv20", FINAL),
    # -----------------------------------------------------------------------
    # Chapter 2: Full greedy run (full pool, NBO50 seed)
    # -----------------------------------------------------------------------
    # Output of a later greedy walkforward run with the full sleeve pool and
    # narrow_bull_off_50 as the fixed seed. Regime-gated nbo50 replaces ungated
    # z60+z20; tighter mono35 breadth gate; sector_spy_mom adds cross-asset
    # signal absent from early greedy result.
    # Bear, cumret, mono35, ssm20 each selected in 4/5 folds across 3 seed variants.
    ("2. Greedy-5\n(dp1z60|nbo50 + brv20 + cs20\n+ ms252/b35x + ssm20)", GREEDY_CORE),
    # -----------------------------------------------------------------------
    # Chapter 3: Extending GREEDY_CORE
    # -----------------------------------------------------------------------
    # ga3: the one sleeve that fixes the 2022 bear-market fold (SR 0.11 → 1.34)
    ("3a. Greedy-5\n+ ga3", GREEDY_CORE + [_GAP]),
    # ga3 + rz15: rz15 adds marginal improvement across folds and lowest drawdown
    ("3b. Greedy-5\n+ ga3 + rz15\n[FINAL CANDIDATE]", GREEDY_CORE + [_GAP, _RZSCORE]),
    # -----------------------------------------------------------------------
    # Chapter 3: Alternatives set aside
    # -----------------------------------------------------------------------
    # rga5: avg SR 1.78 but 2022 fold only +0.35 — improvement concentrated in 2020
    ("3x. Greedy-5\n+ rga5\n[2020-concentrated]", GREEDY_CORE + [_RGAP]),
    # va20: strong in high-vol folds but worst max drawdown of all portfolios tested
    ("3x. Greedy-5\n+ va20\n[high drawdown]", GREEDY_CORE + [_VOLACCEL]),
]

OUT_DIR = Path(__file__).parent / "out" / "walkforward_portfolio_combinations"

COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _best_weights(
    names: list[str], sleeve_returns_train: pd.DataFrame
) -> tuple[str, dict[str, float]]:
    """Try all weighting schemes on training returns, return (best_scheme, weights)."""
    best_scheme = "equal"
    best_ns = float("-inf")
    best_weights: dict[str, float] = pu.estimate_weights_equal(names)

    for scheme in WEIGHTING_SCHEMES:
        if scheme == "equal":
            w = pu.estimate_weights_equal(names)
        elif scheme == "equal_vol":
            w = pu.estimate_weights_equal_vol(names, sleeve_returns_train)
        elif scheme == "equal_sharpe":
            w = pu.estimate_weights_equal_sharpe(names, sleeve_returns_train)
        else:
            w = pu.estimate_weights_optimal(names, sleeve_returns_train)

        # Evaluate on training slice to pick best scheme
        # (use sleeve_returns_train proxy — no position data needed here)
        port_ret = sum(sleeve_returns_train[n] * w.get(n, 0.0) for n in names)
        std = float(port_ret.std())
        ns = float(port_ret.mean() / std * (252**0.5)) if std > 0 else float("-inf")
        if ns > best_ns:
            best_ns = ns
            best_scheme = scheme
            best_weights = w

    return best_scheme, best_weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    all_rows: list[dict] = []
    all_val_returns: list[pd.DataFrame] = []
    folds = [f"{val_start[:4]}" for _, _, val_start, _ in FOLDS]

    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(FOLDS):
        fold_label = f"{val_start[:4]}"
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}: train {train_start}..{train_end}, val {val_start}..{val_end}")
        print("=" * 60)

        # --- Load combined window (with warmup for signal initialization) ---
        warmup_start = str(int(train_start[:4]) - WARMUP_YEARS) + train_start[4:]
        print(f"  Loading data ({warmup_start} to {val_end}) ...")
        universe, benchmark, factors = pu.load_data(warmup_start, val_end)
        print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")

        # --- Distance partners on train slice ---
        print("  Computing distance partners on train slice ...")
        partners = pu.compute_distance_partners(
            universe, train_end=train_end, train_start=train_start
        )
        get_distance_partners = lambda: partners  # noqa: E731
        sector_etf_map = pu.get_sector_etf_map_for(universe)
        get_sector_etf_map = lambda: sector_etf_map  # noqa: E731
        sector_map = qs.get_sector_map(list(universe.returns.columns))

        # --- Run all unique sleeves needed across all portfolios ---
        all_needed = set()
        for _, sleeves in ALL_PORTFOLIOS:
            all_needed.update(sleeves)

        specs = build_sleeve_specs(
            get_distance_partners=get_distance_partners,
            get_sector_etf_map=get_sector_etf_map,
        )
        # Filter to only sleeves we need
        specs_needed = {n: s for n, s in specs.items() if n in all_needed}
        print(f"  Running {len(specs_needed)} sleeves on full window ...")
        studies = pu.run_sleeve_pool(
            specs_needed,
            universe,
            benchmark,
            factors,
            sector_map,
            verbose=False,
            residualize_fit_start=train_start,
            scaler_start=train_start,
        )
        print("  Done.")

        sleeve_returns_full = pd.DataFrame(
            {n: studies[n].cache["portfolio_returns"] for n in specs_needed}
        )
        sleeve_returns_train = sleeve_returns_full.loc[:train_end]

        val_universe_returns = universe.returns.loc[val_start:val_end]
        val_bm_series = benchmark.returns["SPY"].loc[val_start:val_end]
        train_universe_returns = universe.returns.loc[:train_end]
        train_bm_series = benchmark.returns["SPY"].loc[:train_end]

        # --- Evaluate each portfolio ---
        for port_label, names in ALL_PORTFOLIOS:
            print(f"  {port_label} ({len(names)} sleeves) ...", end=" ", flush=True)

            best_scheme, weights = _best_weights(names, sleeve_returns_train)

            # Build combined positions on full window
            full_combined = pu.combine_positions_fixed_weights(studies, weights, names)

            # --- Train eval ---
            train_combined = full_combined.loc[:train_end]
            train_metrics = pu.evaluate_fixed_weight_portfolio_raw(
                train_combined, train_universe_returns, train_bm_series, COST_BPS
            )
            train_ns = pu.get_net_sharpe(train_metrics)

            # --- Val eval ---
            val_combined = full_combined.loc[val_start:val_end]
            val_metrics = pu.evaluate_fixed_weight_portfolio_raw(
                val_combined, val_universe_returns, val_bm_series, COST_BPS
            )
            val_ns = pu.get_net_sharpe(val_metrics)

            print(f"train_ns={train_ns:.3f}  val_ns={val_ns:.3f}")

            all_rows.append(
                {
                    "portfolio": port_label,
                    "fold": fold_label,
                    "train_start": train_start,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "sleeve_count": len(names),
                    "best_scheme": best_scheme,
                    "train_net_sharpe": train_ns,
                    "train_ann_return": float(train_metrics.get("ann_return", float("nan"))),
                    "train_ann_vol": float(train_metrics.get("ann_vol", float("nan"))),
                    "train_max_drawdown": float(train_metrics.get("max_drawdown", float("nan"))),
                    "train_avg_daily_turnover": float(
                        train_metrics.get("avg_daily_turnover", float("nan"))
                    ),
                    "val_net_sharpe": val_ns,
                    "val_ann_return": float(val_metrics.get("ann_return", float("nan"))),
                    "val_ann_vol": float(val_metrics.get("ann_vol", float("nan"))),
                    "val_max_drawdown": float(val_metrics.get("max_drawdown", float("nan"))),
                    "val_avg_daily_turnover": float(
                        val_metrics.get("avg_daily_turnover", float("nan"))
                    ),
                    "sleeves": "|".join(names),
                }
            )

            # --- Daily val returns ---
            val_universe_aligned = val_universe_returns.reindex(
                columns=val_combined.columns
            ).fillna(0)
            val_gross = qs_engine.run(val_combined, val_universe_aligned)
            val_to = qs_metrics.turnover(val_combined)
            val_net_ret = val_gross - val_to * COST_BPS / 10_000
            val_bm_aligned = val_bm_series.reindex(val_gross.index).fillna(0)

            fold_ret_df = pd.DataFrame(
                {"gross": val_gross, "net": val_net_ret, "benchmark": val_bm_aligned}
            )
            fold_ret_df.index.name = "date"
            fold_ret_df["fold"] = fold_label
            fold_ret_df["portfolio"] = port_label
            all_val_returns.append(fold_ret_df)

    # ---------------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------------
    print("\nWriting outputs ...")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUT_DIR / "walkforward_pc_fold_results.csv", index=False)
    print("Saved walkforward_pc_fold_results.csv")

    val_returns_df = pd.concat(all_val_returns).reset_index()
    val_returns_df.to_csv(OUT_DIR / "walkforward_pc_val_returns.csv", index=False)
    print("Saved walkforward_pc_val_returns.csv")

    # --- Bar chart: val net Sharpe / max drawdown / turnover per portfolio x fold ---
    port_labels = [label for label, _ in ALL_PORTFOLIOS]
    x = np.arange(len(port_labels))
    n_folds = len(folds)
    width = 0.7 / n_folds
    x_tick_offset = width * (n_folds - 1) / 2
    tick_labels = [p.split(". ", 1)[-1] for p in port_labels]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(max(12, len(port_labels) * 1.2), 11), sharex=True
    )

    for fi, fold in enumerate(folds):
        fold_df = results_df[results_df["fold"] == fold].set_index("portfolio")
        offsets = x + fi * width

        def _vals(col):
            return [
                fold_df.loc[p, col] if p in fold_df.index else float("nan") for p in port_labels
            ]

        ax1.bar(
            offsets,
            _vals("val_net_sharpe"),
            width * 0.9,
            label=fold,
            color=COLORS[fi],
            alpha=0.85,
        )
        ax2.bar(
            offsets,
            [v * 100 for v in _vals("val_max_drawdown")],
            width * 0.9,
            color=COLORS[fi],
            alpha=0.85,
        )
        ax3.bar(
            offsets,
            [v * 100 for v in _vals("val_avg_daily_turnover")],
            width * 0.9,
            color=COLORS[fi],
            alpha=0.85,
        )

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Val Net Sharpe")
    # ax1.set_title("Walkforward Validation — Portfolio Combinations", pad=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Val Max Drawdown (%)")
    ax2.grid(True, axis="y", alpha=0.3)

    ax3.set_ylabel("Val Avg Daily Turnover (%)")
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.set_xticks(x + x_tick_offset)
    ax3.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=9, multialignment="left")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "walkforward_pc_val_sharpe_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved walkforward_pc_val_sharpe_bar.png")

    # --- Validation summary table ---
    val_cols = {
        "val_net_sharpe": "avg_net_sharpe",
        "val_ann_return": "avg_ann_return",
        "val_ann_vol": "avg_ann_vol",
        "val_max_drawdown": "avg_max_drawdown",
        "val_avg_daily_turnover": "avg_turnover",
    }
    summary = (
        results_df.groupby("portfolio")[list(val_cols.keys())]
        .mean()
        .rename(columns=val_cols)
        .reindex(port_labels)
    )
    print("\nAverage validation metrics by portfolio (across all folds):")
    print(
        summary.to_string(
            float_format=lambda x: f"{x:.3f}",
        )
    )
    summary.to_csv(OUT_DIR / "walkforward_pc_val_summary.csv")
    print("Saved walkforward_pc_val_summary.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
