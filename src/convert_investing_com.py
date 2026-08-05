"""
Convert investing.com manually-downloaded CSV files to tickrake format.

investing.com export format:
    "Date","Price","Open","High","Low","Vol.","Change %"
    "11/02/2018","44.44","44.44","44.47","44.41","23.22M","0.07%"

Tickrake format:
    datetime,open,high,low,close,volume
    2018-11-02T06:00:00Z,44.44,44.47,44.41,44.44,23220000

Usage:
    uv run python scripts/convert_investing_com.py <file> <ticker>
    uv run python scripts/convert_investing_com.py "data/investing.com/CA Stock Price History.csv" CA
"""

import argparse
import re
from pathlib import Path

import pandas as pd

OUT_DIR = Path.home() / ".tickrake/data/history/tickrake"


def parse_volume(v: str) -> int:
    """Convert '23.22M' or '1.5B' or '23,220,000' to int."""
    if not isinstance(v, str):
        return 0
    v = v.strip().replace(",", "")
    if not v or v == "-":
        return 0
    m = re.match(r"([\d.]+)([MBK]?)$", v, re.IGNORECASE)
    if not m:
        return 0
    num, suffix = float(m.group(1)), m.group(2).upper()
    multiplier = {"M": 1_000_000, "B": 1_000_000_000, "K": 1_000}.get(suffix, 1)
    return int(num * multiplier)


def convert(src: Path, ticker: str) -> Path:
    df = pd.read_csv(src, thousands=",")

    # Normalize column names
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Date: MM/DD/YYYY -> datetime
    df["datetime"] = pd.to_datetime(df["Date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%dT06:00:00Z")

    # Price = close
    out = pd.DataFrame({
        "datetime": df["datetime"],
        "open":     pd.to_numeric(df["Open"], errors="coerce"),
        "high":     pd.to_numeric(df["High"], errors="coerce"),
        "low":      pd.to_numeric(df["Low"],  errors="coerce"),
        "close":    pd.to_numeric(df["Price"], errors="coerce"),
        "volume":   df["Vol."].apply(parse_volume),
    })

    out = out.dropna(subset=["open", "close"]).sort_values("datetime").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace(".", "_").replace("-", "_")
    out_path = OUT_DIR / f"{safe_ticker}_day.csv"

    # Merge with existing file if present
    if out_path.exists():
        existing = pd.read_csv(out_path)
        out = pd.concat([existing, out]).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    out.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to investing.com CSV export")
    parser.add_argument("ticker", help="Ticker symbol (e.g. CA, BLL)")
    args = parser.parse_args()

    src = Path(args.file)
    if not src.exists():
        print(f"File not found: {src}")
        raise SystemExit(1)

    out_path = convert(src, args.ticker)
    df = pd.read_csv(out_path)
    print(f"Written {len(df)} rows to {out_path}")
    print(f"Date range: {df['datetime'].min()[:10]} to {df['datetime'].max()[:10]}")


if __name__ == "__main__":
    main()
