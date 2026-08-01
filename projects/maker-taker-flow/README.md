# maker-taker-flow

Labels SPXW option chain snapshots as taker or maker flow using the Lee-Ready algorithm, then plots net dollar-weighted flow over time.

## Data

Reads from `~/.tickrake/data/options/schwab/<year>/<month>/<day>/SPXW_samples_<date>.parquet`. Each parquet file contains per-minute snapshots of the full SPXW chain for that sample date.

## Lee-Ready Rules

1. `last > midpoint` → **taker** (trade lifted the ask)
2. `last < midpoint` → **maker** (trade hit the bid)
3. `last == midpoint` → tick test: uptick = taker, downtick = maker, zero-tick = carry forward

## Usage

```bash
# Full chain for a given sample date and expiration — shows chart + summary
uv run python maker-taker-flow/query_chain.py \
    --sample-date 2026-07-17 \
    --expiration 2026-07-17

# Filter to puts only
uv run python maker-taker-flow/query_chain.py \
    --sample-date 2026-07-17 \
    --expiration 2026-07-20 \
    --contract-type PUT

# Write labeled rows to CSV instead of printing sample rows
uv run python maker-taker-flow/query_chain.py \
    --sample-date 2026-07-17 \
    --expiration 2026-07-17 \
    --output results/spxw_lr_labels.csv

# Skip the chart
uv run python maker-taker-flow/query_chain.py \
    --sample-date 2026-07-17 \
    --expiration 2026-07-17 \
    --no-plot
```

## Chart

Each bar represents one sample minute. Orange = net taker flow, blue = net maker flow. Bar height is net dollar volume (`last × last_size`) at that timestamp across all strikes and contract types matching the query.
