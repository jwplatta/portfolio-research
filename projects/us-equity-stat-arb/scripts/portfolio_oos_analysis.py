from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
import portfolio_utils as pu
from sig_fam_utils import build_sleeve_specs, sleeve_short_name

import qstudy as qs
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics
from qstudy.study.metrics import drawdown_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from constants import COST_BPS, OUT_ROOT, TRAIN_END, TRAIN_START, WARMUP_YEARS

OOS_START = "2015-01-01"
OOS_END = "2026-05-29"
OOS_SPLIT = pd.Timestamp("2024-01-01")
VOL_TARGET = 0.10
MAX_LEVERAGE = 15.0
ROLLING_WINDOW = 90
OUT_DIR = OUT_ROOT / "portfolio_oos_analysis"

# --- Portfolio-level regime scaler ---
# Scales down portfolio exposure when the market is in a narrow-breadth bull regime:
#   breadth (% stocks above 200d MA) is low AND SPY is in an uptrend (above 200d MA).
# This regime kills MR and event signals while the portfolio has no bull exposure to compensate.
REGIME_SCALER_ENABLED = False
BREADTH_MA_WINDOW = 200  # window for per-stock MA used in breadth computation
BREADTH_LOW_THRESHOLD = 0.50  # breadth < 50% = narrow market
SPY_MA_WINDOW = 200  # SPY MA window for trend filter
REGIME_SMOOTHING = 20  # smooth the regime indicator (days) to avoid rapid switching
REGIME_SCALE_DOWN = 0.25  # multiply positions by this in the narrow-breadth-bull regime

_Z20 = "dist_mr_k1_z20__r21__none__cond__none"
_Z60 = "dist_mr_k1_z60__r21__none__cond__none"
_GAP = "gap_accum_3d__r21__trend_20_100_off__cond__none"
_MONO = "monoton_skip_252d__r21__breadth_40_off__cond__none"
_CUMRET = "cumret_spread_20_252__r5__vol_20_60__cond__none"
_BEAR = "bear_reversal_20d__r21__trend_20_100_mr__cond__bear_narrow_lt40"
_K3Z20 = "dist_mr_k3_z20__r10__none__cond__low_disp_off_q30"
_SSM20 = "sector_spy_mom_20d__r5__none__cond__stress_disp_20d_q70"
# _SSM202 = (
#     "sector_spy_mom_within_20d__r21__trend_50_200_mom__cond__sector_disp_20d_q70"  # interesting
# )
# _SSM202 = (
#     "sector_spy_mom_within_20d__r21__trend_50_200_mom__cond__sector_disp_20d_q60"  # interesting
# )
# _SSM202 = "sector_spy_mom_x_stock_120d__r5__long__cond__stress_disp_20d_q60"
# _SSM202 = "sector_rel_cumlog_120d__r5__long__cond__sector_disp_20d_q60"
# _SSM202 = "sector_rel_cumlog_within_20d__r21__trend_50_200_mom__cond__sector_disp_20d_q70"
_SSM202 = "sector_rel_cumlog_within_120d__r21__long__cond__stress_disp_20d_q70"
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

ALL_PORTFOLIOS = [
    # --- Early greedy construction path (ga3 seed, small pool) ---
    ("1. Core-3\n(ga3 + dp1z60 + ms252/b40x)", CORE3),
    ("2. Core-3\n+ dp1z20", CORE4),
    ("3. Core-4\n+ cs20", CORE5),
    ("4. Core-5\n+ brv20", FINAL),
    # --- Full greedy construction path (dp1z60|nbo50 seed, full pool) ---
    # Seed + 4 core sleeves selected in 10+/15 slots across 3 seed variants
    ("5. Greedy-5\n(dp1z60|nbo50 + brv20 + cs20\n+ ms252/b35x + ssm20)", GREEDY_CORE),
    # Add rga5: selected 9/15 slots, strong IS Sharpe, low correlation to core
    ("6. Greedy-5\n+ rga5", GREEDY_CORE + [_RGAP]),
    # --- Greedy-5 + single additions (secondary sleeves from frequency table) ---
    ("7. Greedy-5\n+ dp1z20", GREEDY_CORE + [_Z20]),
    ("8. Greedy-5\n+ ga3", GREEDY_CORE + [_GAP]),
    ("9. Greedy-5\n+ dp3z20", GREEDY_CORE + [_K3Z20R21]),
    ("10. Greedy-5\n+ ssm120s5", GREEDY_CORE + [_SSM120S5]),
    # --- Greedy-5 + rga5 + single additions ---
    ("11. Greedy-5 + rga5\n+ dp1z20", GREEDY_CORE + [_RGAP, _Z20]),
    ("12. Greedy-5 + rga5\n+ bm60", GREEDY_CORE + [_RGAP, _BETAMOM]),
    ("13. Greedy-5 + rga5\n+ rz15", GREEDY_CORE + [_RGAP, _RZSCORE]),
    # --- Still exploring ---
    ("14. Core-5 + brv20\n+ lvm120", FINAL + [_LOWVOLMOM]),
    ("15. Core-5\n+ va20 + ssm20", CORE5 + [_VOLACCEL, _SSM20]),
]

SLEEVES = GREEDY_CORE + [_RZSCORE, _GAP]

SHORT_LABELS = {name: sleeve_short_name(name) for name in SLEEVES}

COST_VALS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25]

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
# Main
# ---------------------------------------------------------------------------


def compute_regime_scaler(
    universe_returns: pd.DataFrame,
    spy_returns: pd.Series,
    breadth_ma_window: int = BREADTH_MA_WINDOW,
    breadth_threshold: float = BREADTH_LOW_THRESHOLD,
    spy_ma_window: int = SPY_MA_WINDOW,
    smoothing: int = REGIME_SMOOTHING,
    scale_down: float = REGIME_SCALE_DOWN,
) -> pd.Series:
    """
    Returns a daily multiplier in [scale_down, 1.0].

    Multiplier = scale_down when:
      - breadth (fraction of stocks above their breadth_ma_window-day MA) < breadth_threshold
      - AND SPY is above its spy_ma_window-day MA (uptrend)

    Outside that regime the multiplier is 1.0.
    The raw regime flag is smoothed over `smoothing` days to avoid rapid switching.
    """
    # Breadth: fraction of stocks above their N-day MA
    prices = (1 + universe_returns.fillna(0)).cumprod()
    ma = prices.rolling(breadth_ma_window, min_periods=breadth_ma_window // 2).mean()
    above_ma = (prices > ma).astype(float)
    breadth = above_ma.mean(axis=1)

    # SPY trend
    spy_price = (1 + spy_returns.fillna(0)).cumprod()
    spy_ma = spy_price.rolling(spy_ma_window, min_periods=spy_ma_window // 2).mean()
    spy_uptrend = (spy_price > spy_ma).astype(float)

    # Narrow-breadth bull = breadth low AND spy in uptrend
    regime_flag = ((breadth < breadth_threshold) & (spy_uptrend > 0)).astype(float)

    # Smooth: use rolling mean so it ramps gradually
    regime_smooth = regime_flag.rolling(smoothing, min_periods=1).mean()

    # Scale: 1.0 - (1 - scale_down) * regime_smooth
    multiplier = 1.0 - (1.0 - scale_down) * regime_smooth
    return multiplier


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data (with warmup for signal initialization) ---
    warmup_start = str(int(TRAIN_START[:4]) - WARMUP_YEARS) + TRAIN_START[4:]
    print("Loading data ...")
    universe, benchmark, factors = pu.load_data(warmup_start, TRAIN_END)
    partners = pu.compute_distance_partners(universe, train_end=TRAIN_END, train_start=TRAIN_START)
    sector_etf_map = pu.get_sector_etf_map_for(universe)
    sector_map = qs.get_sector_map(list(universe.returns.columns))

    # --- Build sleeve specs ---
    all_specs = build_sleeve_specs(
        get_distance_partners=lambda: partners,
        get_sector_etf_map=lambda: sector_etf_map,
    )
    specs_needed = {n: all_specs[n] for n in SLEEVES}

    # --- Run sleeves ---
    print("Running sleeves ...")
    studies = pu.run_sleeve_pool(
        specs_needed,
        universe,
        benchmark,
        factors,
        sector_map,
        residualize_fit_start=TRAIN_START,
        scaler_start=TRAIN_START,
    )

    # --- Build sleeve returns DataFrame ---
    sleeve_returns = pd.DataFrame(
        {n: studies[n].cache["portfolio_returns"] for n in SLEEVES}
    ).dropna()

    # --- Optimal portfolio weights ---
    weights = pu.estimate_weights_optimal(SLEEVES, sleeve_returns)
    print("Optimal weights:")
    for n, w in weights.items():
        print(f"  {SHORT_LABELS[n]}: {w:.4f}")

    # --- Combined portfolio ---
    combined = pu.combine_positions_fixed_weights(studies, weights, SLEEVES)
    portfolio_metrics = pu.evaluate_fixed_weight_portfolio(
        combined, universe, benchmark, cost_bps=COST_BPS
    )
    net_sharpe_val = portfolio_metrics.get("net_sharpe", portfolio_metrics.get("sharpe"))
    print(f"\nPortfolio net Sharpe @ {COST_BPS:.0f} bps: {net_sharpe_val:.3f}")

    # -----------------------------------------------------------------------
    # Output 1: Sleeve Attribution CSV
    # -----------------------------------------------------------------------
    print("\nBuilding sleeve attribution ...")

    # Equal-vol weights for attribution
    vols = sleeve_returns[SLEEVES].std()
    inv_vol = 1.0 / vols.clip(lower=1e-12)
    ev_weights = (inv_vol / inv_vol.sum()).to_dict()

    w_arr = np.array([ev_weights[n] for n in SLEEVES])

    cov = sleeve_returns[SLEEVES].cov().values
    mu = sleeve_returns[SLEEVES].mean().values

    portfolio_mean = float(w_arr @ mu)
    portfolio_variance = float(w_arr @ cov @ w_arr)

    rows = []
    for i, n in enumerate(SLEEVES):
        m = studies[n].metrics_dict()
        wi = w_arr[i]
        sleeve_to = float(qs_metrics.turnover(studies[n].cache["positions"]).mean())
        portfolio_to = float(
            sum(
                w_arr[j] * qs_metrics.turnover(studies[j_name].cache["positions"]).mean()
                for j, j_name in enumerate(SLEEVES)
            )
        )

        ret_contrib = (
            (wi * float(mu[i]) / portfolio_mean * 100)
            if abs(portfolio_mean) > 1e-16
            else float("nan")
        )
        sigma_w = cov @ w_arr
        var_contrib = (
            (wi * float(sigma_w[i]) / portfolio_variance * 100)
            if abs(portfolio_variance) > 1e-16
            else float("nan")
        )
        efficiency = (ret_contrib / var_contrib) if abs(var_contrib) > 1e-16 else float("nan")
        to_contrib = (
            (wi * sleeve_to / portfolio_to * 100) if abs(portfolio_to) > 1e-16 else float("nan")
        )

        rows.append(
            {
                "sleeve": SHORT_LABELS[n],
                "weight_pct": round(wi * 100, 2),
                "sharpe": round(float(m.get("sharpe", float("nan"))), 3),
                "ann_return_pct": round(float(m.get("ann_return", float("nan"))) * 100, 2),
                "ret_contrib_pct": round(ret_contrib, 2),
                "var_contrib_pct": round(var_contrib, 2),
                "efficiency": round(efficiency, 3),
                "to_contrib_pct": round(to_contrib, 2),
            }
        )

    attr_df = pd.DataFrame(rows).set_index("sleeve")
    out_csv = OUT_DIR / "sleeve_attribution.csv"
    attr_df.to_csv(out_csv)
    print(f"Saved: {out_csv}")
    print(attr_df.to_string())

    # -----------------------------------------------------------------------
    # Output 2: Correlation Heatmap
    # -----------------------------------------------------------------------
    print("\nBuilding correlation heatmap ...")

    corr_df = sleeve_returns.rename(columns=SHORT_LABELS).corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 8},
    )
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out_heatmap = OUT_DIR / "sleeve_correlation_heatmap.png"
    fig.savefig(out_heatmap, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_heatmap}")

    # -----------------------------------------------------------------------
    # Output 3: Net Sharpe vs Transaction Cost
    # -----------------------------------------------------------------------
    print("\nBuilding cost sensitivity chart ...")

    # Align universe returns to combined position columns
    universe_returns_aligned = universe.returns.reindex(columns=combined.columns).fillna(0)

    # Portfolio gross returns and turnover
    port_gross = qs_engine.run(combined, universe_returns_aligned)
    port_to = qs_metrics.turnover(combined)

    # Portfolio net Sharpe at each cost
    port_net_sharpes = []
    for cost in COST_VALS:
        net = port_gross - port_to * cost / 10_000
        ns = float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else float("nan")
        port_net_sharpes.append(ns)

    # Per-sleeve net Sharpe at each cost
    sleeve_net_sharpes: dict[str, list[float]] = {n: [] for n in SLEEVES}
    for n in SLEEVES:
        gross = studies[n].cache["portfolio_returns"]
        to = qs_metrics.turnover(studies[n].cache["positions"])
        for cost in COST_VALS:
            net = gross - to * cost / 10_000
            ns = float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else float("nan")
            sleeve_net_sharpes[n].append(ns)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        COST_VALS,
        port_net_sharpes,
        linestyle="-",
        linewidth=1.5,
        color="#333333",
        marker="o",
        markersize=4,
        label="Portfolio",
    )
    for i, n in enumerate(SLEEVES):
        ax.plot(
            COST_VALS,
            sleeve_net_sharpes[n],
            linestyle="-",
            linewidth=1.0,
            marker="o",
            markersize=3,
            color=COLORS[i % len(COLORS)],
            label=SHORT_LABELS[n],
        )
    ax.axvline(x=10, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Net Sharpe")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_cost = OUT_DIR / "cost_sensitivity.png"
    fig.savefig(out_cost, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_cost}")

    # -----------------------------------------------------------------------
    # Output 4: OOS analysis (2015-2026)
    # -----------------------------------------------------------------------
    oos_warmup_start = str(int(OOS_START[:4]) - WARMUP_YEARS) + OOS_START[4:]
    print(f"\nLoading extended data ({oos_warmup_start} to {OOS_END}) ...")
    oos_universe, oos_benchmark, oos_factors = pu.load_data(oos_warmup_start, OOS_END)
    print(f"  Universe: {oos_universe.returns.shape}")

    # Distance partners computed on IS data only — do not refit on OOS window
    oos_partners = pu.compute_distance_partners(
        oos_universe, train_end=TRAIN_END, train_start=TRAIN_START
    )
    oos_sector_etf_map = pu.get_sector_etf_map_for(oos_universe)
    oos_sector_map = qs.get_sector_map(list(oos_universe.returns.columns))

    all_oos_specs = build_sleeve_specs(
        get_distance_partners=lambda: oos_partners,
        get_sector_etf_map=lambda: oos_sector_etf_map,
    )
    oos_specs_needed = {n: all_oos_specs[n] for n in SLEEVES}

    print("Running sleeves on full window (2014-2026) ...")
    oos_studies = pu.run_sleeve_pool(
        oos_specs_needed,
        oos_universe,
        oos_benchmark,
        oos_factors,
        oos_sector_map,
        residualize_fit_start=TRAIN_START,
        scaler_start=TRAIN_START,
    )

    # IS-calibrated optimal weights (from training returns)
    oos_sleeve_returns = pd.DataFrame(
        {n: oos_studies[n].cache["portfolio_returns"] for n in SLEEVES}
    )
    is_sleeve_returns = oos_sleeve_returns.loc[:TRAIN_END].dropna()
    oos_weights = pu.estimate_weights_optimal(SLEEVES, is_sleeve_returns)

    print("OOS weights (IS-calibrated):")
    for n, w in oos_weights.items():
        print(f"  {SHORT_LABELS[n]}: {w:.4f}")

    # Build combined positions on full window
    oos_combined = pu.combine_positions_fixed_weights(oos_studies, oos_weights, SLEEVES)

    oos_universe_aligned = oos_universe.returns.reindex(columns=oos_combined.columns).fillna(0)
    oos_gross = qs_engine.run(oos_combined, oos_universe_aligned)
    oos_to = qs_metrics.turnover(oos_combined)
    oos_net_unlev = oos_gross - oos_to * COST_BPS / 10_000

    # --- Portfolio-level regime scaler ---
    if REGIME_SCALER_ENABLED:
        print("\nComputing regime scaler (narrow-breadth bull) ...")
        spy_oos = oos_benchmark.returns["SPY"].reindex(oos_net_unlev.index).fillna(0)
        regime_mult = (
            compute_regime_scaler(
                universe_returns=oos_universe.returns.reindex(oos_net_unlev.index).fillna(0),
                spy_returns=spy_oos,
            )
            .reindex(oos_net_unlev.index)
            .fillna(1.0)
        )
        oos_net_unlev_raw = oos_net_unlev.copy()
        oos_net_unlev = oos_net_unlev * regime_mult
        print(
            f"  Mean regime multiplier: {regime_mult.mean():.3f}  "
            f"  Days at min scale: {(regime_mult < 0.5).sum()}"
        )
    else:
        oos_net_unlev_raw = oos_net_unlev

    # Vol-target leverage: compute IS vol, apply static lever
    is_vol = oos_net_unlev.loc[:TRAIN_END].std() * np.sqrt(252)
    lever_scale = min(VOL_TARGET / is_vol, MAX_LEVERAGE) if is_vol > 0 else 1.0
    oos_net_lev = oos_net_unlev * lever_scale

    bm_oos = oos_benchmark.returns["SPY"]

    # --- Annual performance table ---
    print("\nBuilding annual performance table ...")
    annual_rows = []
    for year in sorted(oos_net_lev.index.year.unique()):
        if year < int(TRAIN_START[:4]):
            continue  # skip warmup year(s)
        lev_yr = oos_net_lev[oos_net_lev.index.year == year]
        bm_yr = bm_oos.reindex(lev_yr.index)
        if len(lev_yr) < 20:
            continue
        s = qs_metrics.summary(lev_yr)
        bm_ret_yr = float((1 + bm_yr.fillna(0)).prod() - 1)
        annual_rows.append(
            {
                "Year": year,
                "Period": "OOS" if year >= OOS_SPLIT.year else "IS",
                "Lev. Ret.": f"{s['ann_return']:.1%}",
                "Lev. SR": round(float(s["sharpe"]), 3),
                "Lev. Vol": f"{s['ann_vol']:.1%}",
                "Lev. Max DD": f"{s['max_drawdown']:.1%}",
                "SPY Ret.": f"{bm_ret_yr:.1%}",
            }
        )

    annual_df = pd.DataFrame(annual_rows).set_index("Year")
    annual_df.to_csv(OUT_DIR / "annual_performance.csv")
    print("Saved annual_performance.csv")
    print(annual_df.to_string())

    # -----------------------------------------------------------------------
    # Output 5: Sleeve IS / OOS attribution table
    # Same calculation as Output 1 (sleeve_attribution.csv) but split by period.
    # -----------------------------------------------------------------------
    print("\nBuilding sleeve IS/OOS attribution table ...")

    oos_sleeve_net = pd.DataFrame(
        {
            n: oos_studies[n].cache["portfolio_returns"]
            - qs_metrics.turnover(oos_studies[n].cache["positions"]) * COST_BPS / 10_000
            for n in SLEEVES
        }
    )

    w_arr_oos = np.array([oos_weights[n] for n in SLEEVES])
    portfolio_to_oos = float(
        sum(
            w_arr_oos[j] * qs_metrics.turnover(oos_studies[n].cache["positions"]).mean()
            for j, n in enumerate(SLEEVES)
        )
    )

    is_oos_rows = []
    for period, start, end in [
        ("IS", OOS_START, TRAIN_END),
        ("OOS", str(OOS_SPLIT.date()), OOS_END),
    ]:
        sleeve_ret = oos_sleeve_net.loc[start:end].dropna()
        if len(sleeve_ret) < 5:
            continue

        mu = sleeve_ret.mean().values
        cov = sleeve_ret.cov().values
        portfolio_mean = float(w_arr_oos @ mu)
        portfolio_variance = float(w_arr_oos @ cov @ w_arr_oos)
        sigma_w = cov @ w_arr_oos

        for i, n in enumerate(SLEEVES):
            wi = w_arr_oos[i]
            s_ret = sleeve_ret.iloc[:, i]
            sleeve_to = float(
                qs_metrics.turnover(oos_studies[n].cache["positions"].loc[start:end]).mean()
            )

            ret_contrib = (
                wi * mu[i] / portfolio_mean * 100 if abs(portfolio_mean) > 1e-16 else float("nan")
            )
            var_contrib = (
                wi * sigma_w[i] / portfolio_variance * 100
                if abs(portfolio_variance) > 1e-16
                else float("nan")
            )
            efficiency = ret_contrib / var_contrib if abs(var_contrib) > 1e-16 else float("nan")
            to_contrib = (
                wi * sleeve_to / portfolio_to_oos * 100
                if abs(portfolio_to_oos) > 1e-16
                else float("nan")
            )
            sharpe = (
                float(s_ret.mean() / s_ret.std() * np.sqrt(252))
                if s_ret.std() > 0
                else float("nan")
            )

            is_oos_rows.append(
                {
                    "sleeve": SHORT_LABELS[n],
                    "period": period,
                    "weight_pct": round(wi * 100, 2),
                    "sharpe": round(sharpe, 3),
                    "ret_contrib_pct": round(ret_contrib, 2),
                    "var_contrib_pct": round(var_contrib, 2),
                    "efficiency": round(efficiency, 3),
                    "to_contrib_pct": round(to_contrib, 2),
                }
            )

    is_oos_df = pd.DataFrame(is_oos_rows).set_index(["sleeve", "period"])
    is_oos_df.to_csv(OUT_DIR / "sleeve_is_oos_attribution.csv")
    print("Saved sleeve_is_oos_attribution.csv")
    print(is_oos_df.to_string())

    # --- Figure: equity curve, drawdown, rolling Sharpe ---
    print("\nBuilding OOS summary chart ...")

    unlev_eq = (1 + oos_net_unlev.fillna(0)).cumprod()
    lev_eq = (1 + oos_net_lev.fillna(0)).cumprod()
    bm_eq = (1 + bm_oos.fillna(0)).cumprod().reindex(lev_eq.index)
    if REGIME_SCALER_ENABLED:
        unlev_raw_eq = (1 + oos_net_unlev_raw.fillna(0)).cumprod()

    def rolling_sharpe(ret: pd.Series, window: int) -> pd.Series:
        r = ret.fillna(0)
        mu = r.rolling(window).mean()
        sigma = r.rolling(window).std()
        return (mu / sigma.clip(lower=1e-10) * np.sqrt(252)).where(sigma > 1e-10)

    fig, axes = plt.subplots(
        3, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )

    ax = axes[0]
    if REGIME_SCALER_ENABLED:
        ax.plot(
            unlev_raw_eq.index, unlev_raw_eq, color=COLORS[0], lw=0.8, ls=":", label="No scaler"
        )
    ax.plot(
        unlev_eq.index,
        unlev_eq,
        color=COLORS[0],
        lw=1.0,
        label="Unlevered (regime scaled)" if REGIME_SCALER_ENABLED else "Unlevered",
    )
    ax.plot(
        lev_eq.index,
        lev_eq,
        color=COLORS[2],
        lw=1.0,
        label=f"Levered ({VOL_TARGET:.0%} Target Vol)",
    )
    ax.plot(bm_eq.index, bm_eq, color=COLORS[9], lw=1.0, ls="--", label="SPY", alpha=0.9)
    ax.axvline(OOS_SPLIT, color="gray", ls="--", lw=1.2, label="IS/OOS split")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    unlev_dd = drawdown_series(oos_net_unlev)
    lev_dd = drawdown_series(oos_net_lev)
    ax.plot(unlev_dd.index, unlev_dd, color=COLORS[0], lw=1.0, alpha=0.75, label="Unlevered")
    ax.plot(
        lev_dd.index,
        lev_dd,
        color=COLORS[2],
        lw=1.0,
        alpha=0.75,
        label=f"Levered ({VOL_TARGET:.0%} Target Vol)",
    )
    ax.axvline(OOS_SPLIT, color="gray", ls="--", lw=1.0)
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(
        rolling_sharpe(oos_net_unlev, ROLLING_WINDOW).index,
        rolling_sharpe(oos_net_unlev, ROLLING_WINDOW),
        color=COLORS[0],
        lw=0.9,
    )
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(OOS_SPLIT, color="gray", ls="--", lw=1.0)
    ax.set_ylabel(f"Rolling Sharpe ({ROLLING_WINDOW}d)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)

    # Annual x-axis ticks on all panels
    import matplotlib.dates as mdates

    for a in axes:
        a.xaxis.set_major_locator(mdates.YearLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "oos_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved oos_summary.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
