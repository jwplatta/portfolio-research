# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research workspace for developing and evaluating trading ideas. Organized around `experiments/` (backtests using the `qstudy` Study pipeline), `notebooks/` (exploratory Jupyter work), and `data/` (options/market data CSVs).

The `qstudy` package is installed from a sibling directory (`../qstudy`) as `trade-lab` via uv editable source — `import qstudy as qs` is the canonical import.

## Skills

Use the `quant-studies` skill for designing, running, and iterating on cross-sectional equity backtests and related study workflows. Use `skillex` for skill management (`skillex list`, `skillex pull <skill-name> --agent codex`). Local skills under `skills/` cover Python project setup, testing, and code quality.

## Commands

```bash
# Install dependencies
uv sync

# Run a single study version
uv run python experiments/<experiment-name>/v0.py

# Run all versions in an experiment and write results.json / results.csv
uv run qstudy run <experiment-name>

# Create a new iteration of an experiment
uv run qstudy iterate <experiment-name> <version-name>

# Show results for an experiment
uv run qstudy show-results <experiment-name>

# Lint
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy .

# Tests
uv run pytest
uv run pytest tests/test_foo.py::test_bar   # single test
```

## Experiment Structure

Each experiment lives in `experiments/<experiment-name>/` and follows one of two patterns:

**Newer pattern** (managed by `qstudy` CLI):
- `shared.py` — cached data loaders, constants (`START_DATE`, `END_DATE`, `COST_BPS`, etc.), and reusable signal/position/scaler functions
- `v0.py` — baseline study; each file defines `run_study() -> dict` and calls `.run()` then returns `study.metrics_dict()`
- `v<N>_<description>.py` — each iteration imports from `shared.py` and modifies one thing vs baseline
- `run.py` — executes all `v*.py` and writes `results.json` + `results.csv`
- `iteration_index.json` — append-only metadata created by `qstudy iterate`
- `log.md` — running notes on what each version changed and its result

**Older pattern** (manual):
- `common.py` — shared helpers and `build_study()` factory
- `v<N>.py` — minimal, often a single call to `build_study(...)` with parameters
- `run_all.py` — loads each version and writes `results.csv`
- `LOG.md` — notes

## qstudy Study Pipeline

Studies are built as a fluent chain on `Study(universe, benchmark, factors, name)`:

```python
Study(...)
  .residualize_returns()          # factor-residualize returns
  .base_signal(signal_fn)         # signal_fn(**cache) -> DataFrame
  .transform_signal(fn)           # optional signal transforms
  .add_vol_filter(...)            # filter out high-vol names
  .add_volume_zscore_filter(...)  # filter out low-volume names
  .add_momentum_context_filter(...)
  .add_filter(custom_fn)          # signal_fn(signal, **cache) -> DataFrame
  .add_tradeable_constraint(qs.liquidity(...))
  .build_positions(position_fn)   # position_fn(signal, **cache) -> DataFrame
  .scale_risk(scaler_fn)          # scaler_fn(positions, **cache) -> DataFrame
  .with_transaction_costs(cost_bps=10.0)
  .rebalance(every=1)
  .run()                          # returns Study with metrics
```

The `**cache` dict passed to signal/filter/scaler functions contains: `returns`, `residual_returns`, `benchmark`, `factors`, `_tradeable_mask`, `_liquidity_mask`.

Data is loaded with `qs.download(tickers, start_date, end_date)` and cached by `qstudy` under `.qstudy-data/`. The `.qstudy.toml` points `studies_dir = "experiments"` and `data_dir = ".qstudy-data"`.

Key constants available from `qstudy.constants`: `SP500` (ticker list), `SECTOR_ETFS`.

## Key Metrics

Study results expose via `study.metrics_dict()`: `sharpe`, `ann_return`, `ann_vol`, `max_drawdown`, `max_drawdown_duration`, `avg_daily_turnover`, `benchmark_corr`, `information_ratio`.

## Code Style

- Line length: 100 (ruff)
- Python 3.10+
- Signal, filter, and scaler functions are closures that return a named inner function; always set `fn.__name__` for traceability in results
- Use `@functools.cache` on data loader functions in `shared.py`
