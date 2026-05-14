# Sp500 Industry Relative Med Horizon Momentum Costs

Created with `qstudy create sp500-industry-relative-med-horizon-momentum-costs`.

Files:
- `v0.py`: baseline study entrypoint with `run_study() -> dict`
- `shared.py`: shared universe, benchmark, and signal helpers
- `run.py`: execute all top-level `v*.py` files and write `results.json` and `results.csv`
- `iteration_index.json`: append-only metadata for CLI-created study iterations
- `log.md`: experiment notes

Workflow:
1. Edit `shared.py` and `v0.py`.
2. Run `uv run qstudy iterate sp500-industry-relative-med-horizon-momentum-costs <version-name>` to create the next version file.
3. Run `python run.py` inside this directory.
4. Inspect `results.json` or use `qstudy show-results sp500-industry-relative-med-horizon-momentum-costs`.
