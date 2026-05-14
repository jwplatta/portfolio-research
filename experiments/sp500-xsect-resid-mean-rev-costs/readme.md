# Sp500 Xsect Resid Mean Rev Simple

Created with `qstudy create sp500-xsect-resid-mean-rev-simple`.

Purpose:
- Rebuild the old `sp500-xsect-resid-mean-rev` study from a much thinner baseline.
- Keep reusable data loading in `shared.py`.
- Keep the full baseline study definition and version-specific logic in `v0.py`.

Files:
- `v0.py`: baseline study entrypoint with `run_study() -> dict`
- `shared.py`: shared universe, benchmark, factor, and date loaders
- `run.py`: execute all top-level `v*.py` files and write `results.json` and `results.csv`
- `iteration_index.json`: append-only metadata for CLI-created study iterations
- `log.md`: experiment notes

Workflow:
1. Edit `v0.py` for baseline-specific logic and `shared.py` only for reusable data/config.
2. Run `uv run qstudy iterate sp500-xsect-resid-mean-rev-simple <version-name>` to create the next version file.
3. Run `uv run qstudy run sp500-xsect-resid-mean-rev-simple`.
4. Inspect `results.json` or use `qstudy show-results sp500-xsect-resid-mean-rev-simple`.
