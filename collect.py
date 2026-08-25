"""Fetch KRX (KOSPI) daily closing prices via the official KRX Open API.

One request per trading day covers the whole KOSPI universe (유가증권
일별매매정보), which is far fewer round-trips than looping per ticker, and
won't silently break when KRX changes their own website - this hits the
versioned Open API instead of scraping the site (which is why the earlier
pykrx-based version stopped working).

Requires an approved KRX_AUTH_KEY (see .env.example), issued at
https://openapi.krx.co.kr (registration + per-service approval required).

Usage
-----
    python collect.py --start 2023-01-01 --end 2026-08-24
    python collect.py                      # resumes from the last stored date
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

from store import get_connection, last_date, upsert_day, upsert_tickers

API_URL = "https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"
DEFAULT_START = "2023-01-01"


def _num(raw: str | None) -> float | None:
    """KRX returns numbers as comma-grouped strings, and empty strings for
    fields that don't apply. None (unknown) is deliberately distinct from
    0.0 (known to be zero) - see upsert_day's caller."""
    if raw is None:
        return None
    cleaned = str(raw).replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def fetch_day(auth_key: str, ymd: str, retries: int = 4, backoff: float = 3.0) -> list[dict]:
    """One trading day's KOSPI rows, retrying transient network failures.

    A full-history backfill is ~3,000 sequential requests over roughly an
    hour; a single DNS hiccup or dropped connection partway through would
    otherwise abort the whole run and lose the remaining days. Retries cover
    transport errors and 5xx only - a 4xx (bad key, bad date) is a real
    error and re-sending it just wastes time, so it propagates immediately.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                API_URL, params={"AUTH_KEY": auth_key, "basDd": ymd}, timeout=20
            )
            if resp.status_code >= 500:
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json().get("OutBlock_1", [])
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code < 500:
                raise
            last_error = e
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                ValueError) as e:
            last_error = e
        if attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"fetch_day({ymd}) failed after {retries} attempts") from last_error


def main() -> None:
    load_dotenv()
    auth_key = os.environ.get("KRX_AUTH_KEY")
    if not auth_key:
        raise SystemExit("KRX_AUTH_KEY not set. Copy .env.example to .env and fill it in.")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD; default resumes from last stored date")
    parser.add_argument("--end", default=date.today().isoformat(), help="YYYY-MM-DD (default: today)")
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

    days = [d.strftime("%Y%m%d") for d in pd.bdate_range(start, args.end)]
    print(f"Fetching {len(days)} candidate trading days ({start} .. {args.end})")

    fetched = 0
    for i, ymd in enumerate(days, start=1):
        rows = fetch_day(auth_key, ymd)
        if not rows:
            continue  # weekend already excluded by bdate_range; this is a holiday

        close_by_ticker, name_by_ticker, extras_by_ticker = {}, {}, {}
        for r in rows:
            price = r.get("TDD_CLSPRC", "").replace(",", "")
            if not price:
                continue
            close_by_ticker[r["ISU_CD"]] = float(price)
            name_by_ticker[r["ISU_CD"]] = r["ISU_NM"]
            # ACC_TRDVAL (거래대금) is the ADV input costs.py sizes slippage
            # against; the rest are free in the same response. A blank or
            # unparseable field becomes None, not 0 - a halted stock traded
            # no value, but a *missing* field is unknown, and 0 would make
            # costs.py price it as infinitely illiquid.
            extras_by_ticker[r["ISU_CD"]] = {
                "trdval": _num(r.get("ACC_TRDVAL")),
                "volume": _num(r.get("ACC_TRDVOL")),
                "mktcap": _num(r.get("MKTCAP")),
                "list_shrs": _num(r.get("LIST_SHRS")),
            }

        iso_date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
        upsert_day(conn, iso_date, close_by_ticker, extras_by_ticker)
        upsert_tickers(conn, name_by_ticker)
        fetched += 1

        if i % 20 == 0:
            print(f"  {i}/{len(days)} days checked, {fetched} trading days stored so far")

    print(f"Done. Stored {fetched} trading days.")


if __name__ == "__main__":
    main()
