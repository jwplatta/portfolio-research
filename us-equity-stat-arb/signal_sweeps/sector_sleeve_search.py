"""Search for a principled long-short replacement for sector_spy_mom_20d__r5__none__cond__stress_disp_20d_q70.

The current sleeve assigns identical signal values to all stocks in a sector,
so build_long_short breaks ties by column order — arbitrary stock selection.

This script tests 10 variants that pick stocks in a principled way (within-sector
ranking by closeness, stock-vs-sector outperformance, etc.) and evaluates each
on IS 2015–2023. For each candidate we report:
  - Standalone IS net Sharpe (full period + per year)
  - Portfolio IS net Sharpe when swapped in for ssm20
  - Correlation with the original ssm20 sleeve

Usage:
    uv run python examples/signal_sweeps/sector_sleeve_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import portfolio_utils as pu
from sig_fam_utils import (
    make_sector_rel_cumlog_sharpe_within,
    make_sector_rel_cumlog_within,
    make_sector_spy_mom,
    make_sector_spy_mom_contrarian,
    make_sector_spy_mom_within,
    make_sector_spy_mom_x_stock,
)

import qstudy as qs
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics
from qstudy import Study

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
WARMUP_START = "2014-01-01"
COST_BPS = 10.0
N_LONG = N_SHORT = 20

# The sleeve we're trying to replace
SSM20_NAME = "sector_spy_mom_20d__r5__none__cond__stress_disp_20d_q70"

# Fixed base portfolio (6 sleeves, ssm20 excluded)
BASE_SLEEVES = [
    "dist_mr_k1_z60__r5__none__cond__narrow_bull_off_50",
    "bear_reversal_20d__r21__trend_20_100_mr__cond__bear_narrow_lt40",
    "cumret_spread_20_252__r5__vol_20_60__cond__none",
    "monoton_skip_252d__r21__breadth_35_off__cond__none",
    "resid_zscore_w15_w10__r10__trend_20_100__cond__none",
    "gap_accum_3d__r21__trend_20_100_off__cond__none",
]

# ---------------------------------------------------------------------------
# Gate / scaler factories (inline, matching sig_fam_utils filter_factories)
# ---------------------------------------------------------------------------


def _make_stress_disp_filter(vol_window: int, disp_window: int, disp_quantile: float):
    """stress_disp: SPY vol > median AND sector dispersion > quantile."""
    SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB"]

    def filt(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        bm = cache["benchmark"]
        fr = cache["factor_returns"]
        spy_vol = bm.rolling(vol_window).std()
        high_vol = spy_vol.gt(spy_vol.rolling(252, min_periods=126).quantile(0.50))
        etfs = fr[[c for c in SECTOR_ETFS if c in fr.columns]]
        disp = etfs.rolling(disp_window).mean().std(axis=1)
        high_disp = disp.gt(disp.rolling(252, min_periods=126).quantile(disp_quantile))
        active = (high_vol & high_disp).reindex(signal.index).fillna(False)
        return signal.where(active, other=np.nan)

    filt.__name__ = f"stress_disp_{vol_window}d_q{int(disp_quantile * 100)}"
    return filt


def _make_sector_disp_filter(window: int, quantile: float):
    """sector_disp: sector ETF dispersion > trailing quantile."""
    SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB"]

    def filt(signal: pd.DataFrame, **cache) -> pd.DataFrame:
        fr = cache["factor_returns"]
        etfs = fr[[c for c in SECTOR_ETFS if c in fr.columns]]
        disp = etfs.rolling(window).mean().std(axis=1)
        threshold = disp.rolling(252, min_periods=126).quantile(quantile)
        active = disp.gt(threshold).reindex(signal.index).fillna(False)
        return signal.where(active, other=np.nan)

    filt.__name__ = f"sector_disp_{window}d_q{int(quantile * 100)}"
    return filt


def _make_trend_scaler(fast: int, slow: int):
    """Scale to 0.25x in SPY downtrend."""
    def scaler(positions: pd.DataFrame, **cache) -> pd.DataFrame:
        bm = cache["benchmark"]
        equity = (1 + bm).cumprod()
        uptrend = equity.rolling(fast).mean().gt(equity.rolling(slow).mean())
        scale = pd.Series(
            np.where(uptrend.reindex(positions.index).fillna(False), 1.0, 0.25),
            index=positions.index,
        )
        return positions.mul(scale.shift(1), axis=0)
    scaler.__name__ = f"trend_{fast}_{slow}_mom"
    return scaler


# ---------------------------------------------------------------------------
# Candidate definitions
# ---------------------------------------------------------------------------
# Each candidate: (label, signal_factory_fn, gate_fn, extra_scalers, rebalance)

def _make_candidates(get_sector_etf_map):
    return [
        # --- within_20d: same gate/rebalance as ssm20, just principled stock pick ---
        # 1. exact same construction as ssm20 but with within-sector selection
        (
            "ssm20w/r5|strdq70",
            make_sector_spy_mom_within(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.70),
            [],
            5,
        ),
        # 2. same but trend scaler added on top
        (
            "ssm20w/r5/t50|strdq70",
            make_sector_spy_mom_within(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.70),
            [_make_trend_scaler(50, 200)],
            5,
        ),
        # 3. same gate, r5, stress q60 instead of q70 — fires more often
        (
            "ssm20w/r5|strdq60",
            make_sector_spy_mom_within(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.60),
            [],
            5,
        ),
        # 4. same gate r5, stress q80 — stricter, higher quality signals only
        (
            "ssm20w/r5|strdq80",
            make_sector_spy_mom_within(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.80),
            [],
            5,
        ),
        # --- cumlog_within: same gate r5 as ssm20 ---
        # 5. cumlog_within_20d, stress_disp_q70, r5 — cumlog sector score
        (
            "sclw20/r5|strdq70",
            make_sector_rel_cumlog_within(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.70),
            [],
            5,
        ),
        # 6. cumlog_within_20d, stress_disp_q70, r5, trend scaler
        (
            "sclw20/r5/t50|strdq70",
            make_sector_rel_cumlog_within(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.70),
            [_make_trend_scaler(50, 200)],
            5,
        ),
        # 7. cumlog_within_20d, stress_disp_q60, r5
        (
            "sclw20/r5|strdq60",
            make_sector_rel_cumlog_within(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.60),
            [],
            5,
        ),
        # 8. cumlog_sharpe_within_20d, stress_disp_q70, r5
        (
            "sclsw20/r5|strdq70",
            make_sector_rel_cumlog_sharpe_within(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.70),
            [],
            5,
        ),
        # 9. cumlog_sharpe_within_20d, stress_disp_q70, r5, trend scaler
        (
            "sclsw20/r5/t50|strdq70",
            make_sector_rel_cumlog_sharpe_within(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.70),
            [_make_trend_scaler(50, 200)],
            5,
        ),
        # 10. cumlog_sharpe_within_20d, stress_disp_q60, r5
        (
            "sclsw20/r5|strdq60",
            make_sector_rel_cumlog_sharpe_within(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.60),
            [],
            5,
        ),
        # --- x_stock: sector rank × stock-vs-sector rank ---
        # 11. x_stock_20d, stress_disp_q70, r5 — short window to match ssm20
        (
            "ssmx20/r5|strdq70",
            make_sector_spy_mom_x_stock(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.70),
            [],
            5,
        ),
        # 12. x_stock_20d, stress_disp_q70, r5, trend scaler
        (
            "ssmx20/r5/t50|strdq70",
            make_sector_spy_mom_x_stock(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.70),
            [_make_trend_scaler(50, 200)],
            5,
        ),
        # 13. x_stock_20d, stress_disp_q60, r5
        (
            "ssmx20/r5|strdq60",
            make_sector_spy_mom_x_stock(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.60),
            [],
            5,
        ),
        # 14. x_stock_120d, stress_disp_q70, r5
        (
            "ssmx120/r5|strdq70",
            make_sector_spy_mom_x_stock(120, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.70),
            [],
            5,
        ),
        # --- contrarian ---
        # 15. contrarian_20d, stress_disp_q70, r5
        (
            "ssmc20/r5|strdq70",
            make_sector_spy_mom_contrarian(20, get_sector_etf_map),
            _make_stress_disp_filter(20, 20, 0.70),
            [],
            5,
        ),
    ]


# ---------------------------------------------------------------------------
# Study builder
# ---------------------------------------------------------------------------


def _run_sleeve(signal_fn, gate_fn, extra_scalers, rebalance, universe, benchmark, factors):
    builder = Study(
        universe=universe, benchmark=benchmark, factors=factors, verbose=False
    ).base_signal(signal_fn)
    if gate_fn is not None:
        builder = builder.add_filter(gate_fn)
    builder = (
        builder
        .add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=pu.make_equity_curve_regime_scale(TRAIN_START))
    )
    for s in extra_scalers:
        builder = builder.scale_risk(fn=s)
    return builder.rebalance(every=rebalance).with_transaction_costs(cost_bps=COST_BPS).run()


def _net_sharpe(study, start=None, end=None):
    pr = study.cache["portfolio_returns"].loc[start:end]
    mu = pr.mean()
    sigma = pr.std()
    return float(mu / sigma * np.sqrt(252)) if sigma > 0 else float("nan")


def _yearly_sharpe(study, start, end):
    pr = study.cache["portfolio_returns"].loc[start:end]
    years = sorted(pr.index.year.unique())
    result = {}
    for y in years:
        yr = pr[pr.index.year == y]
        sigma = yr.std()
        result[y] = float(yr.mean() / sigma * np.sqrt(252)) if sigma > 0 else float("nan")
    return result


def _corr(study_a, study_b, start, end):
    pa = study_a.cache["portfolio_returns"].loc[start:end]
    pb = study_b.cache["portfolio_returns"].loc[start:end]
    aligned = pd.concat([pa, pb], axis=1).dropna()
    if aligned.shape[0] < 20:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def _portfolio_net_sharpe(base_studies, candidate_study, all_names, cand_name,
                           universe, benchmark, start, end):
    """Equal-vol weighted portfolio net Sharpe."""
    names = all_names + [cand_name]
    study_map = {**base_studies, cand_name: candidate_study}

    # Build combined returns for equal-vol weighting
    ret_df = pd.DataFrame({n: study_map[n].cache["portfolio_returns"] for n in names}).loc[start:end]
    vols = ret_df.std()
    inv_vol = 1.0 / vols.clip(lower=1e-8)
    weights = (inv_vol / inv_vol.sum()).to_dict()

    combined_pos = pu.combine_positions_fixed_weights(study_map, weights, names)
    combined_pos = combined_pos.loc[start:end]
    univ_ret = universe.returns.loc[start:end]
    bm_ret = benchmark.returns["SPY"].loc[start:end]

    univ_aligned = univ_ret.reindex(columns=combined_pos.columns).fillna(0)
    gross = qs_engine.run(combined_pos, univ_aligned)
    to = qs_metrics.turnover(combined_pos)
    net = gross - to * COST_BPS / 10_000
    sigma = net.std()
    return float(net.mean() / sigma * np.sqrt(252)) if sigma > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Loading data {WARMUP_START} to {TRAIN_END} ...")
    universe, benchmark, factors = pu.load_data(WARMUP_START, TRAIN_END)
    sector_etf_map = pu.get_sector_etf_map_for(universe)
    get_sector_etf_map = lambda: sector_etf_map  # noqa: E731
    sector_map = qs.get_sector_map(list(universe.returns.columns))
    partners = pu.compute_distance_partners(universe, train_end=TRAIN_END, train_start=TRAIN_START)
    get_distance_partners = lambda: partners  # noqa: E731

    # Run base sleeves + ssm20
    print("Running base sleeves + ssm20 ...")
    from sig_fam_utils import build_sleeve_specs
    all_specs = build_sleeve_specs(
        get_distance_partners=get_distance_partners,
        get_sector_etf_map=get_sector_etf_map,
    )
    needed = BASE_SLEEVES + [SSM20_NAME]
    base_studies = pu.run_sleeve_pool(
        {n: all_specs[n] for n in needed},
        universe, benchmark, factors, sector_map,
        verbose=True,
        residualize_fit_start=TRAIN_START,
        scaler_start=TRAIN_START,
    )
    ssm20_study = base_studies[SSM20_NAME]
    base_only = {n: base_studies[n] for n in BASE_SLEEVES}

    # Baseline portfolio (with ssm20)
    baseline_sr = _portfolio_net_sharpe(
        base_only, ssm20_study, BASE_SLEEVES, SSM20_NAME,
        universe, benchmark, TRAIN_START, TRAIN_END
    )
    ssm20_standalone = _net_sharpe(ssm20_study, TRAIN_START, TRAIN_END)
    ssm20_yearly = _yearly_sharpe(ssm20_study, TRAIN_START, TRAIN_END)

    print(f"\nBaseline (with ssm20): portfolio SR = {baseline_sr:.3f}")
    print(f"ssm20 standalone SR = {ssm20_standalone:.3f}")

    # Build and evaluate candidates
    candidates = _make_candidates(get_sector_etf_map)
    results = []

    years = sorted(ssm20_yearly.keys())

    for label, signal_fn, gate_fn, extra_scalers, rebalance in candidates:
        print(f"\nRunning {label} ...", end=" ", flush=True)
        try:
            study = _run_sleeve(signal_fn, gate_fn, extra_scalers, rebalance,
                                universe, benchmark, factors)
            standalone = _net_sharpe(study, TRAIN_START, TRAIN_END)
            yearly = _yearly_sharpe(study, TRAIN_START, TRAIN_END)
            corr = _corr(study, ssm20_study, TRAIN_START, TRAIN_END)
            port_sr = _portfolio_net_sharpe(
                base_only, study, BASE_SLEEVES, label,
                universe, benchmark, TRAIN_START, TRAIN_END
            )
            print(f"standalone={standalone:.3f}  port={port_sr:.3f}  corr={corr:.2f}")
            results.append({
                "label": label,
                "standalone_sr": standalone,
                "port_sr": port_sr,
                "corr_vs_ssm20": corr,
                "port_delta": port_sr - baseline_sr,
                **{f"yr_{y}": yearly.get(y, float("nan")) for y in years},
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"label": label, "standalone_sr": float("nan"),
                            "port_sr": float("nan"), "corr_vs_ssm20": float("nan"),
                            "port_delta": float("nan")})

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("SUMMARY — IS 2015–2023")
    print("=" * 110)

    yr_cols = [f"yr_{y}" for y in years]
    header = f"{'Candidate':<25} {'SA SR':>6} {'Port SR':>8} {'Delta':>7} {'Corr':>6}  " + \
             "  ".join(f"{y}" for y in years)
    print(header)
    print("-" * 110)

    # Print ssm20 baseline first
    yr_vals = "  ".join(f"{ssm20_yearly.get(y, float('nan')):5.2f}" for y in years)
    print(f"{'ssm20 [CURRENT]':<25} {ssm20_standalone:>6.3f} {baseline_sr:>8.3f} {'---':>7} {'---':>6}  {yr_vals}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: -x.get("port_sr", float("-inf"))):
        yr_vals = "  ".join(f"{r.get(f'yr_{y}', float('nan')):5.2f}" for y in years)
        delta_str = f"{r['port_delta']:+.3f}" if not np.isnan(r["port_delta"]) else "  nan"
        print(f"{r['label']:<25} {r['standalone_sr']:>6.3f} {r['port_sr']:>8.3f} {delta_str:>7} {r['corr_vs_ssm20']:>6.2f}  {yr_vals}")

    print("=" * 110)
    print("\nDelta = portfolio SR vs baseline (with ssm20). Positive = improvement.")


if __name__ == "__main__":
    main()
