"""
Annual alpha/beta regression of a sleeve portfolio against SPY.

Runs the specified sleeves, combines them with equal_sharpe weighting,
then regresses daily net returns against SPY for each calendar year
plus aggregate full-period summary.

Outputs (examples/out/portfolio_regression/):
  - portfolio_regression_results.csv  — alpha, beta, t-stats, CI per period
  - portfolio_regression_chart.png    — bar chart of annual alpha and beta
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import portfolio_utils as pu
from sig_fam_utils import build_sleeve_specs

import qstudy as qs
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics

# ---------------------------------------------------------------------------
# Configuration — edit these to change the portfolio being analysed
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

CORE3 = [_GAP, _Z60, _MONO]
CORE4 = CORE3 + [_Z20]
CORE5 = CORE4 + [_CUMRET]
FINAL = CORE5 + [_BEAR]

# Greedy-derived core: seed (narrow_bull_off_50) + 4 sleeves selected in 10+/15
# slots across all 3 seed variants. Data-driven construction from walkforward results.
GREEDY_CORE = [_NBO50, _BEAR, _CUMRET, _MONO35, _SSM20]

SLEEVES = GREEDY_CORE + [_RZSCORE, _GAP]

from constants import COST_BPS, OUT_ROOT, TRAIN_END, TRAIN_START

TEST_START = "2024-01-01"  # set to None to skip OOS regression
TEST_END = "2026-05-31"
WEIGHTING_SCHEME = "optimal"  # "equal" | "equal_vol" | "equal_sharpe" | "optimal" | "manual"

# Manual weights — only used when WEIGHTING_SCHEME = "manual".
# Must include an entry for every sleeve in SLEEVES; will be renormalized to sum to 1.
MANUAL_WEIGHTS: dict[str, float] = {
    "gap_accum_3d__r21__trend_20_100_off__cond__none": 1.0,
    "dist_mr_k1_z60__r21__none__cond__none": 1.0,
    "monoton_skip_252d__r21__breadth_40_off__cond__none": 1.0,
    "dist_mr_k1_z20__r21__none__cond__none": 1.0,
    "cumret_spread_20_252__r5__vol_20_60__cond__none": 1.0,
    "bear_reversal_20d__r21__trend_20_100_mr__cond__bear_narrow_lt40": 1.0,
    "sector_rel_mom_120d__r21__trend_50_200_mom__cond__breadth_lt50": 1.0,
}

OUT_DIR = OUT_ROOT / "portfolio_regression"


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------


def run_regression(net_returns: pd.Series, bm_returns: pd.Series):
    idx = net_returns.index.intersection(bm_returns.index)
    y = net_returns.loc[idx]
    x = bm_returns.loc[idx]
    valid = y.notna() & x.notna()
    y = y[valid]
    x = x[valid]
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    return model, y, x


def reg_stats(model, y: pd.Series, x: pd.Series, period: str, label: str) -> dict:
    alpha_daily = model.params["const"]
    ci = model.conf_int(alpha=0.05)
    bm_col = x.name if x.name else x.columns[0] if hasattr(x, "columns") else "SPY"
    beta = model.params.iloc[1]
    t_beta = model.tvalues.iloc[1]
    p_beta = model.pvalues.iloc[1]
    ann_ret = float((1 + y).prod() ** (252 / len(y)) - 1)
    ann_vol = float(y.std() * np.sqrt(252))
    return {
        "period": period,
        "label": label,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else float("nan"),
        "bm_ann_ret": float((1 + x).prod() ** (252 / len(x)) - 1),
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_daily * 252,
        "alpha_ci_lo": float(ci.iloc[0, 0]) * 252,
        "alpha_ci_hi": float(ci.iloc[0, 1]) * 252,
        "t_alpha": model.tvalues["const"],
        "p_alpha": model.pvalues["const"],
        "beta": beta,
        "t_beta": t_beta,
        "p_beta": p_beta,
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
        "skew": float(y.skew()),
        "kurtosis": float(y.kurt()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    data_end = TEST_END if TEST_START else TRAIN_END
    warmup_start = str(int(TRAIN_START[:4]) - 1) + TRAIN_START[4:]
    print(f"\nLoading data ({warmup_start} to {data_end}) ...")
    universe, benchmark, factors = pu.load_data(warmup_start, data_end)
    print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")

    partners = pu.compute_distance_partners(universe, train_end=TRAIN_END, train_start=TRAIN_START)
    get_distance_partners = lambda: partners  # noqa: E731
    sector_etf_map = pu.get_sector_etf_map_for(universe)
    get_sector_etf_map = lambda: sector_etf_map  # noqa: E731
    sector_map = qs.get_sector_map(list(universe.returns.columns))

    specs = build_sleeve_specs(
        get_distance_partners=get_distance_partners,
        get_sector_etf_map=get_sector_etf_map,
    )
    specs_needed = {n: s for n, s in specs.items() if n in set(SLEEVES)}
    missing = set(SLEEVES) - set(specs_needed)
    if missing:
        raise ValueError(f"Sleeves not found in spec registry: {missing}")

    print(f"\nRunning {len(specs_needed)} sleeves ...")
    studies = pu.run_sleeve_pool(
        specs_needed,
        universe,
        benchmark,
        factors,
        sector_map,
        residualize_fit_start=TRAIN_START,
        scaler_start=TRAIN_START,
    )
    print("Done.")

    all_sleeve_returns = pd.DataFrame({n: studies[n].cache["portfolio_returns"] for n in SLEEVES})
    # Estimate weights using training period only
    train_sleeve_returns = all_sleeve_returns.loc[TRAIN_START:TRAIN_END]

    if WEIGHTING_SCHEME == "equal":
        weights = pu.estimate_weights_equal(SLEEVES)
    elif WEIGHTING_SCHEME == "equal_vol":
        weights = pu.estimate_weights_equal_vol(SLEEVES, train_sleeve_returns)
    elif WEIGHTING_SCHEME == "equal_sharpe":
        weights = pu.estimate_weights_equal_sharpe(SLEEVES, train_sleeve_returns)
    elif WEIGHTING_SCHEME == "manual":
        missing_w = set(SLEEVES) - set(MANUAL_WEIGHTS)
        if missing_w:
            raise ValueError(f"MANUAL_WEIGHTS missing entries for: {missing_w}")
        total = sum(MANUAL_WEIGHTS[n] for n in SLEEVES)
        weights = {n: MANUAL_WEIGHTS[n] / total for n in SLEEVES}
    else:
        weights = pu.estimate_weights_optimal(SLEEVES, train_sleeve_returns)

    print(f"\nWeights ({WEIGHTING_SCHEME}):")
    for n, w in weights.items():
        print(f"  {n}: {w:.4f}")

    combined = pu.combine_positions_fixed_weights(studies, weights, SLEEVES)
    univ_returns = universe.returns.reindex(columns=combined.columns).fillna(0)
    gross_returns = qs_engine.run(combined, univ_returns)
    to = qs_metrics.turnover(combined)
    net_returns = gross_returns - to * COST_BPS / 10_000
    spy_returns = benchmark.returns["SPY"].reindex(net_returns.index).fillna(0)

    train_net = net_returns.loc[TRAIN_START:TRAIN_END]
    train_spy = spy_returns.loc[TRAIN_START:TRAIN_END]

    train_m = pu.evaluate_fixed_weight_portfolio(
        combined.loc[TRAIN_START:TRAIN_END], universe, benchmark, COST_BPS
    )
    print(f"\nTraining-period net Sharpe: {pu.get_net_sharpe(train_m):.3f}")

    if TEST_START:
        test_net = net_returns.loc[TEST_START:TEST_END]
        test_spy = spy_returns.loc[TEST_START:TEST_END]
        print(f"OOS period: {TEST_START} to {TEST_END}, {len(test_net)} days")

    # ---------------------------------------------------------------------------
    # Regressions
    # ---------------------------------------------------------------------------
    rows = []

    # IS annual
    years_is = sorted(train_net.index.year.unique())
    for year in years_is:
        yr_net = train_net[train_net.index.year == year]
        yr_spy = train_spy.reindex(yr_net.index).dropna()
        yr_net = yr_net.reindex(yr_spy.index)
        if len(yr_net) < 20:
            continue
        model, y, x = run_regression(yr_net, yr_spy)
        rows.append(reg_stats(model, y, x, str(year), "IS"))

    # IS full period
    model, y, x = run_regression(train_net, train_spy.reindex(train_net.index).fillna(0))
    rows.append(reg_stats(model, y, x, f"{TRAIN_START[:4]}–{TRAIN_END[:4]}", "IS-ALL"))

    # OOS regressions
    if TEST_START and len(test_net) >= 20:
        years_oos = sorted(test_net.index.year.unique())
        for year in years_oos:
            yr_net = test_net[test_net.index.year == year]
            yr_spy = test_spy.reindex(yr_net.index).dropna()
            yr_net = yr_net.reindex(yr_spy.index)
            if len(yr_net) < 20:
                continue
            model, y, x = run_regression(yr_net, yr_spy)
            rows.append(reg_stats(model, y, x, str(year), "OOS"))

        model, y, x = run_regression(test_net, test_spy.reindex(test_net.index).fillna(0))
        rows.append(reg_stats(model, y, x, f"{TEST_START[:4]}–{TEST_END[:4]}", "OOS-ALL"))

    # ---------------------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------------------
    table_rows = []
    for s in rows:
        table_rows.append(
            {
                "Period": s["period"],
                "Label": s["label"],
                "Ann Ret": f"{s['ann_return']:.1%}",
                "Ann Vol": f"{s['ann_vol']:.1%}",
                "Sharpe": round(s["sharpe"], 3),
                "SPY Ret": f"{s['bm_ann_ret']:.1%}",
                "Alpha (ann)": f"{s['alpha_annual']:.1%}",
                "CI lo": f"{s['alpha_ci_lo']:.1%}",
                "CI hi": f"{s['alpha_ci_hi']:.1%}",
                "t(alpha)": round(s["t_alpha"], 2),
                "p(alpha)": round(s["p_alpha"], 4),
                "Beta": round(s["beta"], 4),
                "t(beta)": round(s["t_beta"], 2),
                "p(beta)": round(s["p_beta"], 4),
                "R2": round(s["r_squared"], 4),
                "Skew": round(s["skew"], 2),
                "Kurt": round(s["kurtosis"], 1),
                "N": s["n_obs"],
            }
        )

    df = pd.DataFrame(table_rows).set_index("Period")
    df.to_csv(OUT_DIR / "portfolio_regression_results.csv")
    print("\nSaved portfolio_regression_results.csv")
    print(df.to_string())

    # ---------------------------------------------------------------------------
    # Chart: annual alpha and beta (IS + OOS shading)
    # ---------------------------------------------------------------------------
    annual = [s for s in rows if len(s["period"]) == 4]
    yr_labels = [s["period"] for s in annual]
    yr_ints = [int(s["period"]) for s in annual]
    alphas = [s["alpha_annual"] * 100 for s in annual]
    betas = [s["beta"] for s in annual]
    ci_lo = [s["alpha_ci_lo"] * 100 for s in annual]
    ci_hi = [s["alpha_ci_hi"] * 100 for s in annual]
    err_lo = [a - lo for a, lo in zip(alphas, ci_lo)]
    err_hi = [hi - a for a, hi in zip(alphas, ci_hi)]
    p_vals = [s["p_alpha"] for s in annual]
    labels = [s["label"] for s in annual]

    # IS = darker palette, OOS = orange tones
    is_sig = "#4e79a7"
    is_nosig = "#aec7e8"
    oos_sig = "#f28e2b"
    oos_nosig = "#ffcc99"

    bar_colors = []
    for lbl, p in zip(labels, p_vals):
        if lbl == "OOS":
            bar_colors.append(oos_sig if p < 0.05 else oos_nosig)
        else:
            bar_colors.append(is_sig if p < 0.05 else is_nosig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Shade OOS region
    if TEST_START:
        oos_years = [yi for yi, lbl in zip(yr_ints, labels) if lbl == "OOS"]
        if oos_years:
            ax1.axvspan(
                min(oos_years) - 0.5,
                max(oos_years) + 0.5,
                alpha=0.07,
                color="orange",
                zorder=0,
                label="OOS",
            )
            ax2.axvspan(
                min(oos_years) - 0.5, max(oos_years) + 0.5, alpha=0.07, color="orange", zorder=0
            )

    ax1.bar(yr_ints, alphas, color=bar_colors, alpha=0.85, width=0.6, zorder=3)
    ax1.errorbar(
        yr_ints,
        alphas,
        yerr=[err_lo, err_hi],
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1,
        zorder=4,
    )
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Annualized Alpha (%)")
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax1.set_title(
        f"Annual Alpha & Beta vs SPY — {WEIGHTING_SCHEME} weighted, {COST_BPS:.0f} bps costs\n"
        f"Sleeves: {', '.join(n.split('__cond__')[0] for n in SLEEVES)}",
        fontsize=9,
        pad=8,
    )
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Shade significant bars
    for xi, (a, p) in enumerate(zip(alphas, p_vals)):
        if p < 0.05:
            ax1.text(yr_ints[xi], a + (2 if a >= 0 else -3), "*", ha="center", fontsize=10)

    ax2.bar(yr_ints, betas, color=bar_colors, alpha=0.85, width=0.6, zorder=3)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Beta (SPY)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(yr_ints)
    ax2.set_xticklabels(yr_labels, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=is_sig, alpha=0.85, label="IS p(α) < 0.05"),
        plt.Rectangle((0, 0), 1, 1, color=is_nosig, alpha=0.85, label="IS p(α) ≥ 0.05"),
        plt.Rectangle((0, 0), 1, 1, color=oos_sig, alpha=0.85, label="OOS p(α) < 0.05"),
        plt.Rectangle((0, 0), 1, 1, color=oos_nosig, alpha=0.85, label="OOS p(α) ≥ 0.05"),
    ]
    ax1.legend(handles=legend_handles, fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "portfolio_regression_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved portfolio_regression_chart.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
