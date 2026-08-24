"""Fetch KRX (KOSPI) daily closing prices and store them in SQLite.

Uses FinanceDataReader (not pykrx - pykrx's session handshake no longer
matches KRX's current site and always returns "LOGOUT"). Loops over the
KOSPI ticker list and pulls each ticker's full history in one call.

Usage
-----
    python collect.py --start 2023-01-01 --end 2026-08-24
    python collect.py                      # resumes from the last stored date
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import FinanceDataReader as fdr
import pandas as pd

from store import get_connection, last_date, upsert_series, upsert_tickers

DEFAULT_START = "2023-01-01"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD; default resumes from last stored date")
    parser.add_argument("--end", default=date.today().isoformat(), help="YYYY-MM-DD (default: today)")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N tickers (for testing)")
    args = parser.parse_args()

    conn = get_connection()

    if args.start:
        start = args.start
    else:
        last = last_date(conn)
        start = (pd.Timestamp(last) + timedelta(days=1)).date().isoformat() if last else DEFAULT_START

    if start > args.end:
        print("Already up to date.")
        return

    listing = fdr.StockListing("KOSPI")
    tickers = listing[["Code", "Name"]].dropna()
    if args.limit:
        tickers = tickers.head(args.limit)

    upsert_tickers(conn, dict(zip(tickers["Code"], tickers["Name"])))
    print(f"Fetching {len(tickers)} KOSPI tickers, {start} .. {args.end}")

    ok, failed = 0, 0
    for i, (code, name) in enumerate(zip(tickers["Code"], tickers["Name"]), start=1):
        try:
            df = fdr.DataReader(code, start, args.end)
        except Exception as e:
            failed += 1
            continue

        if df is None or df.empty or "Close" not in df.columns:
            failed += 1
            continue

        close_by_date = {d.strftime("%Y-%m-%d"): c for d, c in df["Close"].items() if pd.notna(c)}
        upsert_series(conn, code, close_by_date)
        ok += 1

        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} tickers processed ({ok} ok, {failed} failed)")

    print(f"Done. {ok} tickers stored, {failed} failed.")


if __name__ == "__main__":
    main()
