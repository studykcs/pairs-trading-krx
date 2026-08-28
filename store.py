"""SQLite storage for KRX daily closing prices.

Long format (date, ticker, close) so tickers can be added later without a
schema change and re-fetching a date range safely upserts.

`trdval` (거래대금, daily traded value in KRW) is what costs.py needs to
size market-impact slippage against ADV - traded *value* rather than share
volume, because a 100,000-share order means something completely different
in a 4,000원 stock than in a 400,000원 one. `mktcap`/`list_shrs` come from
the same KRX response and are stored because they're free at collection
time and a market-cap filter otherwise has nothing to filter on.

These three are nullable: rows collected before they were added stay NULL
rather than being back-filled with a fabricated number, so a backtest over
an un-backfilled window fails loudly in costs.py instead of silently
pricing slippage off a zero ADV.
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

# Added after the initial (date, ticker, close) schema shipped; applied
# idempotently on every connect so an existing prices.db upgrades in place.
_MIGRATIONS = [
    ("prices", "trdval", "ALTER TABLE prices ADD COLUMN trdval REAL"),
    ("prices", "volume", "ALTER TABLE prices ADD COLUMN volume REAL"),
    ("prices", "mktcap", "ALTER TABLE prices ADD COLUMN mktcap REAL"),
    ("prices", "list_shrs", "ALTER TABLE prices ADD COLUMN list_shrs REAL"),
]


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    # A long collect.py backfill commits once per trading day for tens of
    # minutes; any concurrent reader (a backtest, a progress query) would
    # otherwise fail the writer outright with "database is locked" and throw
    # away the run. WAL lets readers work alongside the writer, and
    # busy_timeout makes the remaining brief contentions wait instead of raise.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    for table, column, ddl in _MIGRATIONS:
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(ddl)
    conn.commit()
    return conn


def last_date(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
    return row[0]


def upsert_day(
    conn: sqlite3.Connection,
    date: str,
    close_by_ticker: dict[str, float],
    extras_by_ticker: dict[str, dict[str, float | None]] | None = None,
) -> None:
    """Upsert one trading day. `extras_by_ticker` optionally carries
    trdval/volume/mktcap/list_shrs per ticker; a ticker missing from it (or
    a missing key) leaves those columns untouched via COALESCE, so a
    close-only re-fetch never wipes liquidity data already collected."""
    extras_by_ticker = extras_by_ticker or {}
    rows = []
    for ticker, close in close_by_ticker.items():
        e = extras_by_ticker.get(ticker, {})
        rows.append((date, ticker, close, e.get("trdval"), e.get("volume"),
                     e.get("mktcap"), e.get("list_shrs")))
    conn.executemany(
        "INSERT INTO prices (date, ticker, close, trdval, volume, mktcap, list_shrs) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(date, ticker) DO UPDATE SET close = excluded.close, "
        "trdval = COALESCE(excluded.trdval, prices.trdval), "
        "volume = COALESCE(excluded.volume, prices.volume), "
        "mktcap = COALESCE(excluded.mktcap, prices.mktcap), "
        "list_shrs = COALESCE(excluded.list_shrs, prices.list_shrs)",
        rows,
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


def period_label(index: pd.DatetimeIndex) -> str:
    """'{start} .. {end} ({n:,} 거래일)' - printed at the top of every backtest
    script's output so a result is traceable to the period actually used
    (post --start/--end, post per-pair dropna), not assumed to be "the whole
    history" by default."""
    if len(index) == 0:
        return "no data"
    return f"{index.min().date()} .. {index.max().date()} ({len(index):,} 거래일)"


def warn_thin_warmup(n_days: int, warmup_days: int) -> None:
    """Warns when a requested date range is thin relative to the warmup a
    walk-forward / regime-filtered strategy needs before it produces any
    signal at all (beta re-estimation window + z-score window + GMM/HMM
    refit window - `warmup_days` is normally just the GMM/HMM window, since
    with the defaults (beta 120 + z 60 + gmm/hmm 250) it's the largest of the
    three and therefore the binding constraint).

    The 2x-warmup threshold is a documented rule of thumb, not a validated
    cutoff - below it, a large share of the requested window has no
    tradeable signal in it at all, so total-return/Sharpe over that window
    reflect mostly warmup, not strategy behavior.
    """
    if n_days < warmup_days * 2:
        pct = warmup_days / n_days * 100 if n_days else 100.0
        print(f"WARNING: 워밍업({warmup_days}거래일)이 표본의 {pct:.0f}%를 차지합니다 - "
              f"결과 해석에 주의하세요.\n")


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


def load_field(
    conn: sqlite3.Connection,
    field: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Same wide shape as load_prices(), for any one numeric price column
    (trdval / volume / mktcap / list_shrs). NULLs stay NaN - see the module
    docstring on why they are not back-filled."""
    allowed = {"close", "trdval", "volume", "mktcap", "list_shrs"}
    if field not in allowed:
        raise ValueError(f"field must be one of {sorted(allowed)}, got {field!r}")

    query = f"SELECT date, ticker, {field} FROM prices"
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

    wide = long.pivot(index="date", columns="ticker", values=field).sort_index()
    wide.index.name = "Date"
    return wide
