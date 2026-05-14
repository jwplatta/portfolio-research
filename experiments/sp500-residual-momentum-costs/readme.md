# Sp500 Residual Momentum Costs

Created with `qstudy create sp500-residual-momentum-costs`.

Purpose:
- pull the residual momentum `v29` study out of `sp500-market-neutral-xsect-momentum`
- make that extracted study the new `v0` baseline
- keep the experiment in the CLI layout with reusable config and loaders in `shared.py`
- leave `COST_BPS` in `shared.py` at `0.0` so costed variants can branch from the unchanged baseline

Files:
- `v0.py`: extracted baseline study entrypoint with `run_study() -> dict`
- `shared.py`: shared dates, universe/benchmark loaders, baseline constants, and signal helper
- `run.py`: execute top-level `v*.py` files and write `results.json` and `results.csv`
- `iteration_index.json`: append-only metadata for CLI-created study iterations
- `log.md`: experiment notes

Workflow:
1. Adjust reusable settings in `shared.py`.
2. Keep version-specific study logic in `v0.py` or future `v*.py` files.
3. Run `uv run qstudy iterate sp500-residual-momentum-costs <version-name>` to create the next iteration.
4. Run `uv run qstudy run sp500-residual-momentum-costs`.
5. Inspect `results.json` or use `uv run qstudy show-results sp500-residual-momentum-costs`.
