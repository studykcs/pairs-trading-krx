"""SQLite storage for KRX daily closing prices.

Long format (date, ticker, close) so tickers can be added later without a
schema change and re-fetching a date range safely upserts.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "prices.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    close REAL NOT NULL,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS tickers (
    ticker TEXT PRIMARY KEY,
    name TEXT NOT NULL
);
"""


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def last_date(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
    return row[0]


def upsert_day(conn: sqlite3.Connection, date: str, close_by_ticker: dict[str, float]) -> None:
    conn.executemany(
        "INSERT INTO prices (date, ticker, close) VALUES (?, ?, ?) "
        "ON CONFLICT(date, ticker) DO UPDATE SET close = excluded.close",
        [(date, ticker, close) for ticker, close in close_by_ticker.items()],
    )
    conn.commit()


def upsert_series(conn: sqlite3.Connection, ticker: str, close_by_date: dict[str, float]) -> None:
    """Upsert one ticker's whole history in a single transaction."""
    conn.executemany(
        "INSERT INTO prices (date, ticker, close) VALUES (?, ?, ?) "
        "ON CONFLICT(date, ticker) DO UPDATE SET close = excluded.close",
        [(d, ticker, close) for d, close in close_by_date.items()],
    )
    conn.commit()


def upsert_tickers(conn: sqlite3.Connection, name_by_ticker: dict[str, str]) -> None:
    conn.executemany(
        "INSERT INTO tickers (ticker, name) VALUES (?, ?) "
        "ON CONFLICT(ticker) DO UPDATE SET name = excluded.name",
        list(name_by_ticker.items()),
    )
    conn.commit()


def ticker_names(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT ticker, name FROM tickers").fetchall())


def load_prices(
    conn: sqlite3.Connection, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Wide DataFrame: date index, one column per ticker, closing prices."""
    query = "SELECT date, ticker, close FROM prices"
    clauses, params = [], []
    if start:
        clauses.append("date >= ?")
        params.append(start)
    if end:
        clauses.append("date <= ?")
        params.append(end)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    long = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    if long.empty:
        return pd.DataFrame()

    wide = long.pivot(index="date", columns="ticker", values="close").sort_index()
    wide.index.name = "Date"
    return wide
