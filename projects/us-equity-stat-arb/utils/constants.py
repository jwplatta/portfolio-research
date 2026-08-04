"""Shared constants for us-equity-stat-arb scripts, utils, and signal sweeps."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

# us-equity-stat-arb/ — one level up from utils/
PROJECT_ROOT = Path(__file__).parent.parent

# Shared output directory: us-equity-stat-arb/out/
OUT_ROOT = PROJECT_ROOT / "out"

# ---------------------------------------------------------------------------
# Training / evaluation window
# ---------------------------------------------------------------------------

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
WARMUP_YEARS = 1  # years of data loaded before TRAIN_START for indicator warmup

# ---------------------------------------------------------------------------
# Universe / data
# ---------------------------------------------------------------------------

BENCHMARK_TICKER = "SPY"
FACTOR_TICKERS = [
    "SPY", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLRE", "XLB",
]

# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

COST_BPS = 10.0   # one-way transaction cost in basis points
N_LONG = 20       # number of long positions
N_SHORT = 20      # number of short positions
REBALANCE_PERIODS = [1, 5, 10, 21]  # daily, weekly, bi-weekly, monthly
