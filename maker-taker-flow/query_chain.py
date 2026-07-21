"""
Query SPXW option chain samples for a specific expiration and sample date,
then label each quote using the Lee-Ready algorithm:

  - trade price > midpoint  → taker (customer buy / maker sell)
  - trade price < midpoint  → maker (customer sell / maker buy)
  - trade price == midpoint → tick test tiebreaker (uptick → taker, downtick → maker)

Usage:
    uv run python maker-taker-flow/query_chain.py \
        --sample-date 2026-07-17 \
        --expiration 2026-07-17

    uv run python maker-taker-flow/query_chain.py \
        --sample-date 2026-07-17 \
        --expiration 2026-07-20 \
        --contract-type CALL
"""

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

DATA_ROOT = Path.home() / ".tickrake/data/options/schwab"


def parquet_path(sample_date: str) -> Path:
    dt = pd.Timestamp(sample_date)
    return DATA_ROOT / f"{dt.year}/{dt.month:02d}/{dt.day:02d}/SPXW_samples_{sample_date}.parquet"


def load_chain(sample_date: str, expiration: str, contract_type: str | None = None) -> pd.DataFrame:
    path = parquet_path(sample_date)
    if not path.exists():
        raise FileNotFoundError(f"No SPXW data found at {path}")

    filters = [f"expiration_date = '{expiration}'"]
    if contract_type:
        filters.append(f"contract_type = '{contract_type.upper()}'")
    where = " AND ".join(filters)

    query = f"""
        SELECT
            contract_type,
            symbol,
            strike,
            expiration_date,
            bid,
            bid_size,
            ask,
            ask_size,
            last,
            last_size,
            mark,
            total_volume,
            open_interest,
            delta,
            gamma,
            theta,
            vega,
            volatility,
            underlying_price,
            sampled_at
        FROM read_parquet('{path}')
        WHERE {where}
          AND last > 0          -- only rows with a trade price
          AND bid > 0
          AND ask > 0
        ORDER BY strike, contract_type, sampled_at
    """
    return duckdb.query(query).df()


def apply_lee_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each row as taker (+1) or maker (-1) using Lee-Ready:
      1. Compute midpoint = (bid + ask) / 2
      2. last > mid  → taker (+1)
         last < mid  → maker (-1)
         last == mid → tick test:
             last > prev_last → taker (+1)
             last < prev_last → maker (-1)
             last == prev_last → carry forward prior label (zero-tick rule)
    """
    df = df.copy()
    df["midpoint"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]

    # Quote test
    df["lr_label"] = 0
    df.loc[df["last"] > df["midpoint"], "lr_label"] = 1
    df.loc[df["last"] < df["midpoint"], "lr_label"] = -1

    # Tick test for at-mid trades — apply per (symbol, strike, contract_type) series
    at_mid_mask = df["last"] == df["midpoint"]
    if at_mid_mask.any():
        group_cols = ["symbol", "contract_type", "strike"]
        df["_prev_last"] = df.groupby(group_cols)["last"].shift(1)

        tick = pd.Series(0, index=df.index)
        tick[df["last"] > df["_prev_last"]] = 1
        tick[df["last"] < df["_prev_last"]] = -1

        # Zero-tick: carry forward the last non-zero tick direction per group
        tick_for_fill = tick.copy()
        tick_for_fill[tick_for_fill == 0] = pd.NA
        # forward-fill within each group
        df["_tick_direction"] = (
            df.groupby(group_cols)["last"]
            .transform(lambda s: s.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else pd.NA)))
            .ffill()
            .fillna(0)
            .astype(int)
        )

        df.loc[at_mid_mask, "lr_label"] = df.loc[at_mid_mask, "_tick_direction"]
        df = df.drop(columns=["_prev_last", "_tick_direction"])

    df["flow"] = df["lr_label"].map({1: "taker", -1: "maker", 0: "unknown"})
    return df


def summarize(df: pd.DataFrame) -> None:
    total = len(df)
    counts = df["flow"].value_counts()
    print(f"\nChain: {len(df['strike'].unique())} strikes, {total} quote snapshots with trades\n")
    print(counts.to_string())
    print()

    # Dollar-weighted flow
    df["dollar_volume"] = df["last"] * df["last_size"]
    flow_dv = df.groupby("flow")["dollar_volume"].sum()
    print("Dollar-weighted flow:")
    print(flow_dv.to_string())


def plot_flow(df: pd.DataFrame, sample_date: str, expiration: str) -> None:
    df = df.copy()
    df["dollar_volume"] = df["last"] * df["last_size"]

    # Normalize sampled_at to minute precision for grouping
    df["sample_minute"] = df["sampled_at"].dt.floor("min").dt.tz_convert("America/Chicago")

    agg = (
        df.groupby(["sample_minute", "flow"])["dollar_volume"]
        .sum()
        .unstack(fill_value=0)
    )
    # Ensure both columns exist
    for col in ("taker", "maker", "unknown"):
        if col not in agg.columns:
            agg[col] = 0.0

    # Net flow per bar: positive = net taker, negative = net maker
    net = agg.get("taker", 0) - agg.get("maker", 0)
    colors = ["#E07B39" if v >= 0 else "#4C72B0" for v in net]  # orange=taker, blue=maker

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(agg.index, net, color=colors, width=pd.Timedelta(seconds=40), align="center")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Sample time (CT)", fontsize=11)
    ax.set_ylabel("Net dollar-weighted flow ($)", fontsize=11)
    ax.set_title(
        f"SPXW Lee-Ready flow  |  sample date {sample_date}  |  expiration {expiration}\n"
        f"Orange = net taker  ·  Blue = net maker",
        fontsize=12,
    )

    fig.autofmt_xdate(rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query SPXW chain and label flow with Lee-Ready")
    parser.add_argument("--sample-date", default="2026-07-17", help="Date of the chain snapshot (YYYY-MM-DD)")
    parser.add_argument("--expiration", default="2026-07-17", help="Option expiration date (YYYY-MM-DD)")
    parser.add_argument("--contract-type", choices=["CALL", "PUT"], default=None, help="Filter to CALL or PUT only")
    parser.add_argument("--output", default=None, help="Optional path to write labeled CSV")
    parser.add_argument("--no-plot", action="store_true", help="Skip the matplotlib chart")
    args = parser.parse_args()

    print(f"Loading SPXW chain  sample_date={args.sample_date}  expiration={args.expiration}")
    df = load_chain(args.sample_date, args.expiration, args.contract_type)
    print(f"  Loaded {len(df):,} rows")

    df = apply_lee_ready(df)
    summarize(df)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote {len(df):,} rows to {args.output}")
    else:
        print("\nSample rows:")
        cols = ["contract_type", "strike", "bid", "ask", "last", "midpoint", "flow", "sampled_at"]
        print(df[cols].head(20).to_string(index=False))

    if not args.no_plot:
        plot_flow(df, args.sample_date, args.expiration)


if __name__ == "__main__":
    main()
