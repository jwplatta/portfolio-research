# Script Summary: 2026-05-17

This note summarizes the scripts worked on in this session, what each one does, and how they fit together.

## Signal Sweep Source Index

These are the signal sweep scripts that produced strategies which currently feed [signal_sweep_conditioning_best_strategies.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_conditioning_best_strategies.py):

- [signal_sweep_monoton.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_monoton.py)
- [signal_sweep_mean_reversion.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_mean_reversion.py)
- [signal_sweep_resid_mean_reversion.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_resid_mean_reversion.py)
- [signal_sweep_distance_pairs.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_distance_pairs.py)
- [signal_sweep_sector_relative.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_sector_relative.py)
- [signal_sweep_event.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_event.py)

Upstream discovery note:
- [signal_sweep_momentum_zoo.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_momentum_zoo.py) is not the direct source of shortlisted sleeves in the conditioning script, but it did contain monotonicity-style signals that appear to have motivated the dedicated `signal_sweep_monoton.py` follow-up.

## Risk Scaler Index

The rebuilt sleeves in [signal_sweep_conditioning_best_strategies.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_conditioning_best_strategies.py) use both conditioning filters and risk scalers.

Base scaler always applied:
- `equity_curve_regime_scale`

Additional legacy scaler families applied when a sleeve carries a non-`none` `legacy_scaler` tag:
- `trend_*`
- `trend_*_h`
- `vol_*`
- `corr_*`
- `breadth_*`
- `disp_*`
- `crash_*`
- `trend_plus_vol`

Concrete scaler tags currently referenced by shortlisted strategies:
- `none`
- `trend_20_100`
- `trend_20_100_h`
- `trend_50_200`
- `vol_10_60`
- `vol_20_60`
- `breadth_40`
- `breadth_60`
- `disp_60_q20`
- `disp_60_q30`
- `crash_10_5pct`
- `trend_plus_vol`

## 1. `scripts/signal_sweep_conditioning_best_strategies.py`

Purpose:
- Broad conditioning sweep built from the strongest strategies in `doc/strategy_log.md`.
- Rebuilds the winning sleeves from several groups:
  - monotonic momentum
  - mean reversion
  - residual mean reversion
  - distance pairs mean reversion
  - sector-relative
  - event
- Applies conditioning filters from `doc/conditioning_filters.md` as actual signal gates rather than just post-position risk scalers.

Important behavior:
- Preserves each shortlisted sleeve’s original signal, rebalance cadence, and legacy winning scaler.
- Can rebuild sleeves with or without strategy-level costs.
- Now correctly handles `trend_plus_vol`.
- Residual candidates now come from the merged `signal_sweep_resid_mean_reversion.py` output rather than a separate residual-xsection group.

Conditioning filters included:
- market trend up/down
- vol expansion/contraction
- breadth strong/weak
- dispersion high/low
- correlation high/low
- volume spike
- overnight gap
- residual vol spike
- residual dispersion high
- sector dislocation
- panic regime

Main outputs:
- `signal_sweep_conditioned_winners.csv`
- `top25_conditioned_winners.csv`
- `top25_conditioned_winner_return_correlations.csv`

## 2. `scripts/portfolio_sweep_conditioned_winner_pairs.py`

Purpose:
- Reads `top25_conditioned_winners.csv`.
- Rebuilds and pre-runs the selected conditioned sleeves once.
- Runs the full ordered `25 x 25` pair grid with `PortfolioStudy`.

Important behavior:
- Uses the exact same sleeve definitions as `signal_sweep_conditioning_best_strategies.py`.
- No strategy-level costs on the individual sleeves.
- Costs are applied only once at the portfolio level.
- Uses equal portfolio weighting.

Main output:
- `portfolio_sweep_conditioned_winner_pairs.csv`

## 3. `scripts/portfolio_sweep_conditioned_portfolio_combos.py`

Purpose:
- Reads `portfolio_sweep_conditioned_winner_pairs.csv`.
- Selects the 2-strategy portfolios above a Sharpe threshold.
- Rebuilds each unique underlying sleeve once.
- Runs an exhaustive ordered sweep of portfolio-of-portfolios:
  - takes one 2-strategy portfolio
  - takes a second 2-strategy portfolio
  - merges the four underlying sleeves
  - removes duplicates
  - runs a new equal-weight `PortfolioStudy`

Important behavior:
- Equal portfolio weighting.
- Portfolio-level costs only.
- Threshold is controlled by `MIN_NET_SHARPE`.

Main outputs:
- `selected_conditioned_winner_pairs_over_1.csv`
- `portfolio_sweep_conditioned_portfolio_combos.csv`

## 4. `scripts/portfolio_sweep_target_10_extensions.py`

Purpose:
- Builds the 10 target portfolios described in `doc/target_10_portfolios.md`.
- Runs the baseline target books.
- Extends each baseline with additional sleeves from the signal families that were explored but did not necessarily survive conditioning as standalone winners.

Extension families currently used:
- monotonic / persistent momentum
- event / residual gap
- sector-relative sleeves

Current extension search:
- baseline only
- baseline + every 1-extension combo
- baseline + every 2-extension combo
- baseline + every 3-extension combo

Current weighting sweep:
- `equal`
- `equal_vol`
- `equal_sharpe`
- `optimal`

Important behavior:
- Rebuilds each unique sleeve once.
- No strategy-level costs.
- Costs only at the portfolio level.
- Records `extension_count` and `weighting`.

Main output:
- `portfolio_sweep_target_10_extensions.csv`

## 5. `scripts/portfolio_sweep_remove_dist_sleeves.py`

Purpose:
- Ablation script for selected top target portfolios.
- Rebuilds the exact sleeves used in each target portfolio.
- Tests:
  - baseline as-is
  - removing each `dist_mr_*` sleeve one at a time
  - removing all `dist_mr_*` sleeves together

Why this exists:
- To test how much the distance-pairs sleeves are contributing to the strongest portfolio constructions.

Important behavior:
- Equal portfolio weighting.
- Portfolio-level costs only.

Main output:
- `portfolio_sweep_remove_dist_sleeves.csv`

## Suggested Workflow

Typical order:

1. Run the base signal sweeps that feed the shortlist in `doc/strategy_log.md`.
2. Run `signal_sweep_conditioning_best_strategies.py`.
3. Run `portfolio_sweep_conditioned_winner_pairs.py`.
4. If useful, run `portfolio_sweep_conditioned_portfolio_combos.py`.
5. Run `portfolio_sweep_target_10_extensions.py`.
6. Run `portfolio_sweep_remove_dist_sleeves.py` for ablation on the best target portfolios.

## High-Level Intent

The work today moved from:
- standalone signal discovery

to:
- conditioned standalone sleeves

to:
- 2-sleeve portfolio search

to:
- portfolio-of-portfolio combinations

to:
- hand-designed target portfolios plus controlled extension searches

to:
- sleeve ablation on the strongest target constructions.

## NOTES

Audit against the refreshed `results/2026-05-18/*.csv` sweep outputs is now clean:

- Every `TOP_STRATEGIES` entry in [signal_sweep_conditioning_best_strategies.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_conditioning_best_strategies.py) matches an exact `name` label in the dated result CSVs.
- The `event` shortlist was updated from the older `resid_gap__...` naming to the current `resid_gap_reversion__...` labels used by `results/2026-05-18/signal_sweep_event.csv`.
- The two stale residual mean-reversion `trend_plus_vol` entries were replaced with the highest-Sharpe unused current rows from `results/2026-05-18/signal_sweep_resid-mean-reversion.csv`:
  - `etf_factor_resid_mr_5d__r10__vol_20_100`
  - `etf_factor_resid_mr_5d__r21__vol_20_60`


factor_model_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75
mr_5d__r10__trend_50_200__cond__breadth_weak_40
dist_mr_k3_z20__r21__cond__vol_contraction_10_60
dist_mr_k3_z60__r5__cond__vol_expansion_10_60
dist_mr_k3_z60__r10__cond__none
dist_mr_k3_z10__r10__cond__panic_10d_minus5
monoton_120d__r21__crash_10_5pct
resid_gap_reversion__r10__breadth_40