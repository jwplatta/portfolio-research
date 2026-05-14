from __future__ import annotations

from pathlib import Path

from qstudy.experiments import run_experiment


def main() -> int:
    rows = run_experiment(Path(__file__).resolve().parent)
    print(f"Ran {len(rows)} study version(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
