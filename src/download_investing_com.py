"""
Download historical OHLCV data from investing.com for delisted S&P 500 tickers.

Strategy: Load each ticker's historical-data page with Playwright (establishes session/cookies),
then call the internal investing.com financialdata API via page.evaluate() to fetch the full
date range. Saves one CSV per ticker to data/investing.com/.

Usage:
    uv run python scripts/download_investing_com.py
    uv run python scripts/download_investing_com.py --ticker AGN
    uv run python scripts/download_investing_com.py --skip-existing
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

OUT_DIR = Path("data/investing.com")
CSV_PATH = Path("data/research/delisted_tickers_2015.csv")

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"


def load_tickers(only: str | None = None) -> list[dict]:
    tickers = []
    with open(CSV_PATH) as f:
        next(f)  # skip header
        for line in f:
            line = line.rstrip("\n")
            # Split on first two commas only — notes/URL may contain commas
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            ticker, status, notes = parts[0].strip(), parts[1].strip(), parts[2].strip()

            if status not in ("yes", "partial"):
                continue
            if only is not None and ticker != only:
                continue

            # Extract URL from notes field
            match = re.search(r"https://www\.investing\.com/\S+", notes)
            url = match.group(0).rstrip(".,;") if match else ""
            if not url:
                continue

            tickers.append({"ticker": ticker, "url": url, "status": status})
    return tickers


def get_pair_id(page) -> int | None:
    """Extract the canonical instrument id from the page's MobX store JSON.

    investing.com injects a large state blob into the page HTML containing
    equityStore.instrument.base.id which is the correct pair id for the
    financialdata API. There are many other pair_ids on the page (for ads,
    related instruments, etc.) so we must target this specific path.
    """
    html = page.content()

    # Primary: identifiers.instrument_id embedded in page state
    m = re.search(r'"instrument_id"\s*:\s*"(\d+)"', html)
    if m:
        return int(m.group(1))

    # Fallback: instrumentId in the price store
    m = re.search(r'"instrumentId"\s*:\s*"(\d+)"', html)
    if m:
        return int(m.group(1))

    # Last resort: equityStore base id
    m = re.search(r'"equityStore".*?"base"\s*:\s*\{"id"\s*:\s*"(\d+)"', html, re.DOTALL)
    if m:
        return int(m.group(1))

    return None


def fetch_history(page, pair_id: int, start: str, end: str) -> list[dict]:
    """Use the browser's session to call the investing.com history API."""
    js = f"""
        async () => {{
            const r = await fetch(
                'https://api.investing.com/api/financialdata/historical/{pair_id}' +
                '?start-date={start}&end-date={end}&time-frame=Daily&add-missing-rows=false',
                {{
                    headers: {{
                        'Accept': 'application/json',
                        'domain-id': 'www',
                    }}
                }}
            );
            if (!r.ok) return null;
            return await r.json();
        }}
    """
    result = page.evaluate(js)
    if result and isinstance(result, dict) and "data" in result:
        return result["data"]
    return []


def rows_to_csv(rows: list[dict], out_path: Path):
    if not rows:
        return
    fieldnames = ["date", "open", "high", "low", "close", "volume"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "date": row.get("rowDateTimestamp", "")[:10],
                "open": row.get("last_open", ""),
                "high": row.get("last_max", ""),
                "low": row.get("last_min", ""),
                "close": row.get("last_close", ""),
                "volume": row.get("volumeRaw", ""),
            })


def new_stealth_page(browser):
    """Create a fresh browser context + page with stealth applied."""
    context = browser.new_context(
        viewport={"width": 1280, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    page = context.new_page()
    Stealth().apply_stealth_sync(page)
    return context, page


def load_page(browser, url: str, ticker: str):
    """Load URL in a fresh context. Retry with longer wait on Cloudflare."""
    for attempt in range(3):
        ctx, page = new_stealth_page(browser)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)
        except Exception as e:
            print(f"[{ticker}] Load error (attempt {attempt+1}): {e}")
            page.close()
            ctx.close()
            time.sleep(5)
            continue

        if "Just a moment" not in page.title():
            return ctx, page

        print(f"[{ticker}] Cloudflare (attempt {attempt+1}) — waiting before retry")
        page.close()
        ctx.close()
        time.sleep(10 * (attempt + 1))

    print(f"[{ticker}] Blocked after 3 attempts, skipping")
    return None, None


def download_ticker(browser, ticker: str, url: str, out_dir: Path) -> bool:
    out_path = out_dir / f"{ticker}.csv"
    print(f"[{ticker}] Loading {url}")

    page = load_page(browser, url, ticker)
    if page is None:
        return False

    # Dismiss cookie/consent banners
    for selector in ["#onetrust-accept-btn-handler", "button:has-text('Accept All')", "button:has-text('I Accept')"]:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=1500):
                btn.click()
                time.sleep(0.5)
                break
        except Exception:
            pass

    pair_id = get_pair_id(page)
    if not pair_id:
        print(f"[{ticker}] Could not find pair_id")
        page.close()
        return False

    print(f"[{ticker}] pair_id={pair_id}, fetching {START_DATE} to {END_DATE}")
    rows = fetch_history(page, pair_id, START_DATE, END_DATE)
    page.close()

    if not rows:
        print(f"[{ticker}] No data returned")
        return False

    rows_to_csv(rows, out_path)
    print(f"[{ticker}] Saved {len(rows)} rows to {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="Download a single ticker only")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already-downloaded tickers")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_tickers(only=args.ticker)
    print(f"Tickers to process: {len(tickers)}")

    results: dict[str, list[str]] = {"ok": [], "failed": []}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        for item in tickers:
            ticker = item["ticker"]
            out_path = OUT_DIR / f"{ticker}.csv"

            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                print(f"[{ticker}] Skipping (exists)")
                continue

            try:
                ok = download_ticker(context, ticker, item["url"], OUT_DIR)
            except Exception as e:
                print(f"[{ticker}] Unexpected error: {e}")
                ok = False
            results["ok" if ok else "failed"].append(ticker)
            time.sleep(2)

        browser.close()

    print("\n=== RESULTS ===")
    print(f"Downloaded ({len(results['ok'])}): {results['ok']}")
    print(f"Failed    ({len(results['failed'])}): {results['failed']}")

    # Write a summary
    summary_path = OUT_DIR / "_download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
