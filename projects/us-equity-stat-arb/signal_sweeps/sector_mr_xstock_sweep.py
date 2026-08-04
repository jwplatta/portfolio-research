"""Sweep of double mean-reversion sector x_stock signal variants.

Background:
  The existing ssm20 sleeve (sector_spy_mom_20d__r5__none__cond__stress_disp_20d_q70)
  is a short-term sector mean-reversion signal. Within a lagging sector it picks stocks
  by arbitrary column-order tie-breaking.

  The principled alternative for a mean-reverting context is to pick stocks that
  deviated MOST from their sector — the deepest laggards within a lagging sector
  are the best reversion candidates.

  Formula: (1 - sector_pct_rank) * (1 - within_sector_pct_rank)
  - sector_pct_rank: rank of sector excess return vs SPY (low = sector lagged = mean-rev long)
  - within_sector_pct_rank: rank of stock excess return vs its sector ETF (low = stock lagged within sector)
  - Product is highest for stocks that lagged most within the most-lagging sector.
  - After rank_transform + build_long_short: these double-laggards go LONG.

  Compare to existing x_stock (momentum x momentum):
    sector_pct_rank * stock_pct_rank → long best-in-best-sector (momentum)

  We test: 20d and 60d windows, r5 rebalance (matching ssm20), stress_disp gates at
  q60/q70/q80, with and without a 50/200 trend scaler.

Usage:
    uv run python examples/signal_sweeps/sector_mr_xstock_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import portfolio_utils as pu
from sig_fam_utils import build_sleeve_specs

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

SSM20_NAME = "sector_spy_mom_20d__r5__none__cond__stress_disp_20d_q70"

# Base portfolio (6 sleeves, ssm20 excluded)
BASE_SLEEVES = [
    "dist_mr_k1_z60__r5__none__cond__narrow_bull_off_50",
    "bear_reversal_20d__r21__trend_20_100_mr__cond__bear_narrow_lt40",
    "cumret_spread_20_252__r5__vol_20_60__cond__none",
    "monoton_skip_252d__r21__breadth_35_off__cond__none",
    "resid_zscore_w15_w10__r10__trend_20_100__cond__none",
    "gap_accum_3d__r21__trend_20_100_off__cond__none",
]

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB"]


# ---------------------------------------------------------------------------
# Double mean-reversion x_stock signal
# ---------------------------------------------------------------------------


def make_sector_mr_xstock(window: int, get_sector_etf_map):
    """Double mean-reversion sector signal.

    Ranks the worst-performing sector AND the worst-performing stock within that
    sector highest, producing a long-short signal biased toward:
      LONG: biggest laggard within the most-lagging sector (double reversion)
      SHORT: biggest winner within the most-leading sector (double momentum short)

    Formula: (1 - sector_pct_rank) * (1 - within_sector_pct_rank)
    Both ranks are percentile ranks [0,1]; inverting makes low values (laggards) high.
    """

    def sector_mr_xstock(**cache):
        r = cache["_active_returns"]
        factor_returns = cache["factor_returns"]
        etf_map = get_sector_etf_map()

        spy = factor_returns["SPY"].reindex(r.index).fillna(0.0)

        # Map each stock to its sector ETF return
        sector_cols = {}
        for ticker in r.columns:
            etf = etf_map.get(ticker, "SPY")
            if etf in factor_returns.columns:
                sector_cols[ticker] = factor_returns[etf]
            else:
                sector_cols[ticker] = spy
        sector_df = pd.DataFrame(sector_cols, index=r.index)

        # Sector score: sector ETF rolling mean vs SPY (same as ssm20 raw signal)
        sector_score = sector_df.rolling(window).mean().sub(
            spy.rolling(window).mean(), axis=0
        )
        # Within-sector score: stock rolling mean vs its sector ETF mean
        within_score = r.rolling(window).mean() - sector_df.rolling(window).mean()

        # Percentile ranks [0,1]; high = outperformer
        sector_pct = sector_score.rank(axis=1, pct=True, na_option="keep")
        within_pct = within_score.rank(axis=1, pct=True, na_option="keep")

        # Double MR: invert both ranks so laggards score highest
        return (1.0 - sector_pct) * (1.0 - within_pct)

    sector_mr_xstock.__name__ = f"sector_mr_xstock_{window}d"
    return sector_mr_xstock


# ---------------------------------------------------------------------------
# Gate / scaler factories
# ---------------------------------------------------------------------------


def _make_stress_disp_filter(vol_window: int, disp_window: int, disp_quantile: float):
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


def _make_candidates(get_sector_etf_map):
    stress_q60 = _make_stress_disp_filter(20, 20, 0.60)
    stress_q70 = _make_stress_disp_filter(20, 20, 0.70)
    stress_q80 = _make_stress_disp_filter(20, 20, 0.80)
    sect_q60 = _make_sector_disp_filter(20, 0.60)
    sect_q70 = _make_sector_disp_filter(20, 0.70)
    trend = _make_trend_scaler(50, 200)

    signal_20 = make_sector_mr_xstock(20, get_sector_etf_map)
    signal_60 = make_sector_mr_xstock(60, get_sector_etf_map)
    signal_120 = make_sector_mr_xstock(120, get_sector_etf_map)

    return [
        # ---- 20d window: same rebalance (r5) + stress_disp gate as ssm20 ----
        ("mr_x20/r5|strdq70",        signal_20, stress_q70, [], 5),
        ("mr_x20/r5/t50|strdq70",    signal_20, stress_q70, [trend], 5),
        ("mr_x20/r5|strdq60",        signal_20, stress_q60, [], 5),
        ("mr_x20/r5/t50|strdq60",    signal_20, stress_q60, [trend], 5),
        ("mr_x20/r5|strdq80",        signal_20, stress_q80, [], 5),
        # sector_disp gate (no SPY vol requirement)
        ("mr_x20/r5|sectdq70",       signal_20, sect_q70, [], 5),
        ("mr_x20/r5|sectdq60",       signal_20, sect_q60, [], 5),
        # no gate — always active
        ("mr_x20/r5|none",           signal_20, None, [], 5),
        ("mr_x20/r5/t50|none",       signal_20, None, [trend], 5),
        # r21 rebalance for comparison
        ("mr_x20/r21|strdq70",       signal_20, stress_q70, [], 21),
        ("mr_x20/r21|strdq60",       signal_20, stress_q60, [], 21),
        # ---- 60d window ----
        ("mr_x60/r5|strdq70",        signal_60, stress_q70, [], 5),
        ("mr_x60/r5/t50|strdq70",    signal_60, stress_q70, [trend], 5),
        ("mr_x60/r5|strdq60",        signal_60, stress_q60, [], 5),
        ("mr_x60/r21|strdq70",       signal_60, stress_q70, [], 21),
        # ---- 120d window ----
        ("mr_x120/r5|strdq70",       signal_120, stress_q70, [], 5),
        ("mr_x120/r21|strdq70",      signal_120, stress_q70, [], 21),
        ("mr_x120/r21|strdq60",      signal_120, stress_q60, [], 21),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_sleeve(signal_fn, gate_fn, extra_scalers, rebalance, universe, benchmark, factors):
    builder = (
        Study(universe=universe, benchmark=benchmark, factors=factors, verbose=False)
        .base_signal(signal_fn)
    )
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
    mu, sigma = pr.mean(), pr.std()
    return float(mu / sigma * np.sqrt(252)) if sigma > 0 else float("nan")


def _yearly_sharpe(study, start, end):
    pr = study.cache["portfolio_returns"].loc[start:end]
    return {
        y: (lambda s: float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else float("nan"))(
            pr[pr.index.year == y]
        )
        for y in sorted(pr.index.year.unique())
    }


def _corr(study_a, study_b, start, end):
    pa = study_a.cache["portfolio_returns"].loc[start:end]
    pb = study_b.cache["portfolio_returns"].loc[start:end]
    aligned = pd.concat([pa, pb], axis=1).dropna()
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) >= 20 else float("nan")


def _portfolio_net_sharpe(base_studies, candidate_study, cand_name, universe, benchmark, start, end):
    names = list(base_studies) + [cand_name]
    study_map = {**base_studies, cand_name: candidate_study}
    ret_df = pd.DataFrame({n: study_map[n].cache["portfolio_returns"] for n in names}).loc[start:end]
    vols = ret_df.std()
    inv_vol = 1.0 / vols.clip(lower=1e-8)
    weights = (inv_vol / inv_vol.sum()).to_dict()

    combined_pos = pu.combine_positions_fixed_weights(study_map, weights, names).loc[start:end]
    univ_ret = universe.returns.loc[start:end].reindex(columns=combined_pos.columns).fillna(0)
    bm_ret = benchmark.returns["SPY"].loc[start:end]

    gross = qs_engine.run(combined_pos, univ_ret)
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

    baseline_sr = _portfolio_net_sharpe(
        base_only, ssm20_study, SSM20_NAME, universe, benchmark, TRAIN_START, TRAIN_END
    )
    ssm20_standalone = _net_sharpe(ssm20_study, TRAIN_START, TRAIN_END)
    ssm20_yearly = _yearly_sharpe(ssm20_study, TRAIN_START, TRAIN_END)

    print(f"\nBaseline (with ssm20): portfolio SR = {baseline_sr:.3f}")
    print(f"ssm20 standalone SR  = {ssm20_standalone:.3f}")

    # Evaluate candidates
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
                base_only, study, label, universe, benchmark, TRAIN_START, TRAIN_END
            )
            print(f"standalone={standalone:.3f}  port={port_sr:.3f}  corr={corr:.2f}")
            results.append({
                "label": label, "standalone_sr": standalone,
                "port_sr": port_sr, "corr_vs_ssm20": corr,
                "port_delta": port_sr - baseline_sr,
                **{f"yr_{y}": yearly.get(y, float("nan")) for y in years},
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "label": label, "standalone_sr": float("nan"),
                "port_sr": float("nan"), "corr_vs_ssm20": float("nan"),
                "port_delta": float("nan"),
            })

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("SUMMARY — IS 2015–2023 | Double MR x_stock variants")
    print("=" * 120)
    yr_cols = [f"yr_{y}" for y in years]
    header = (
        f"{'Candidate':<28} {'SA SR':>6} {'Port SR':>8} {'Delta':>7} {'Corr':>6}  "
        + "  ".join(str(y) for y in years)
    )
    print(header)
    print("-" * 120)

    yr_vals = "  ".join(f"{ssm20_yearly.get(y, float('nan')):5.2f}" for y in years)
    print(f"{'ssm20 [BASELINE]':<28} {ssm20_standalone:>6.3f} {baseline_sr:>8.3f} {'---':>7} {'---':>6}  {yr_vals}")
    print("-" * 120)

    for r in sorted(results, key=lambda x: -x.get("port_sr", float("-inf"))):
        yr_vals = "  ".join(f"{r.get(f'yr_{y}', float('nan')):5.2f}" for y in years)
        delta_str = f"{r['port_delta']:+.3f}" if not np.isnan(r.get("port_delta", float("nan"))) else "  nan"
        print(
            f"{r['label']:<28} {r['standalone_sr']:>6.3f} {r['port_sr']:>8.3f} "
            f"{delta_str:>7} {r['corr_vs_ssm20']:>6.2f}  {yr_vals}"
        )

    print("=" * 120)
    print("\nDelta = portfolio SR vs baseline (ssm20 in the 7-sleeve portfolio).")
    print("Positive delta = candidate improves on ssm20.")


if __name__ == "__main__":
    main()
