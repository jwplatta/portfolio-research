"""
SPXW options flow metric for today.

For each sample timestamp across all SPXW chains collected today:
  flow = Σ( new_volume * delta * distance_weight )
  distance_weight = 1 / |strike - spot|

SPX price comes from underlying_price in the chain; falls back to 1-min candle data.

Output: spxw_flow_<date>.csv  with columns: timestamp, flow, spx_price
"""

import csv
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

TODAY = date.today()
OPTIONS_DIR = Path.home() / ".tickrake/data/options/schwab" / str(TODAY.year) / f"{TODAY.month:02d}" / f"{TODAY.day:02d}"
CANDLES_FILE = Path.home() / ".tickrake/data/history/ibkr-paper/SPX_1min.csv"
OUTPUT_FILE = Path(f"spxw_flow_{TODAY}.csv")

# --------------------------------------------------------------------------- #
# 1. Discover today's SPXW snapshot files grouped by sample timestamp
# --------------------------------------------------------------------------- #

# filename pattern: SPXW_exp<exp>_<date>_<HH-MM-SS>.csv
FILE_RE = re.compile(r"SPXW_exp(?P<exp>\d{4}-\d{2}-\d{2})_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})\.csv")

files_by_ts: dict[str, list[Path]] = defaultdict(list)

for path in sorted(OPTIONS_DIR.glob("SPXW_*.csv")):
    m = FILE_RE.match(path.name)
    if m and m["exp"] == str(TODAY):  # 0DTE only
        ts_key = f"{m['date']}T{m['time'].replace('-', ':')}Z"  # UTC
        files_by_ts[ts_key].append(path)

print(f"Found {len(files_by_ts)} sample timestamps across {sum(len(v) for v in files_by_ts.values())} SPXW snapshot files.")

# --------------------------------------------------------------------------- #
# 2. Load 1-min SPX candles as fallback price source
# --------------------------------------------------------------------------- #

spx_candles: dict[str, float] = {}  # datetime_utc -> close price

if CANDLES_FILE.exists():
    with open(CANDLES_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            spx_candles[row["datetime"]] = float(row["close"])

def get_spx_price_from_candles(ts_utc: str) -> float | None:
    """Find the most recent 1-min candle at or before ts_utc."""
    dt = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
    best_price = None
    best_dt = None
    for candle_ts, price in spx_candles.items():
        try:
            cdt = datetime.fromisoformat(candle_ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if cdt <= dt:
            if best_dt is None or cdt > best_dt:
                best_dt = cdt
                best_price = price
    return best_price

# --------------------------------------------------------------------------- #
# 3. Track prior-snapshot volume per contract to compute new_volume
# --------------------------------------------------------------------------- #

prior_volume: dict[str, int] = {}  # symbol -> last seen total_volume

# --------------------------------------------------------------------------- #
# 4. Process each timestamp in chronological order
# --------------------------------------------------------------------------- #

results: list[dict] = []

for ts_key in sorted(files_by_ts.keys()):
    snapshot_files = files_by_ts[ts_key]

    rows_all: list[dict] = []
    for path in snapshot_files:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_all.append(row)

    if not rows_all:
        continue

    # Determine SPX spot price: prefer underlying_price from chain
    spot: float | None = None
    for row in rows_all:
        raw = row.get("underlying_price", "").strip()
        if raw:
            try:
                spot = float(raw)
                break
            except ValueError:
                pass

    if spot is None:
        spot = get_spx_price_from_candles(ts_key)

    if spot is None or spot == 0:
        print(f"  [skip] {ts_key}: no SPX price available")
        continue

    # Compute flow
    flow = 0.0
    for row in rows_all:
        symbol = row.get("symbol", "").strip()

        # delta
        raw_delta = row.get("delta", "").strip()
        if not raw_delta:
            continue
        try:
            delta = float(raw_delta)
        except ValueError:
            continue

        # total_volume -> new_volume
        raw_vol = row.get("total_volume", "").strip()
        if not raw_vol:
            continue
        try:
            total_vol = int(float(raw_vol))
        except ValueError:
            continue

        prev_vol = prior_volume.get(symbol, 0)
        new_volume = max(0, total_vol - prev_vol)
        prior_volume[symbol] = total_vol

        if new_volume == 0:
            continue

        # strike
        raw_strike = row.get("strike", "").strip()
        if not raw_strike:
            continue
        try:
            strike = float(raw_strike)
        except ValueError:
            continue

        distance = abs(strike - spot)
        if distance < 0.01:
            distance = 0.01  # avoid div-by-zero for ATM

        distance_weight = 1.0 / distance
        flow += new_volume * delta * distance_weight

    results.append({"timestamp": ts_key, "flow": round(flow, 6), "spx_price": round(spot, 2)})

# --------------------------------------------------------------------------- #
# 5. Write CSV
# --------------------------------------------------------------------------- #

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["timestamp", "flow", "spx_price"])
    writer.writeheader()
    writer.writerows(results)

print(f"Wrote {len(results)} rows to {OUTPUT_FILE}")
