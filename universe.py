"""Economically-motivated pair candidate universes.

The earlier approach (screen every ticker with full-history data, in ticker-
code order, up to some cap) has no economic content - ticker codes are
assigned administratively, not by sector or ownership structure. Testing 250
tickers in code order against Samsung Electronics is mostly testing unrelated
companies, which is exactly why FDR correction wiped out every "hit": there
was no real signal in the pool to survive correction.

Each function here instead encodes an actual reason two tickers might share a
long-run equilibrium:

- preferred_pairs: common/preferred shares of the same company are claims on
  the same cash flows, so they're structurally the strongest cointegration
  candidates available.
- holdco_pairs: a holding company's value is largely its stake in a core
  subsidiary, so the NAV discount is a mean-reverting spread by construction.
- sector_peers: same industry, so shared macro/sector exposure gives a
  plausible (if weaker) common stochastic trend.

All three only return tickers that actually have rows in the local prices
table - a candidate with no price history can't be tested, and silently
dropping it would make it impossible to later answer "why isn't this stock in
the universe?".
"""

from __future__ import annotations

import re
import sqlite3

from store import ticker_names

# Korean preferred-share suffixes: 우 / 우B / 2우B / 3우B / 4우B ...
# (the leading digit only ever runs 2-4 in practice, but we don't rely on
# that - any single digit followed by 우 and an optional B is treated as a
# suffix candidate, and matching is what actually confirms it, not the regex)
_PREFERRED_SUFFIX = re.compile(r"[0-9]?우B?$")


def _tickers_with_data(conn: sqlite3.Connection, tickers: list[str]) -> set[str]:
    if not tickers:
        return set()
    placeholders = ",".join("?" for _ in tickers)
    rows = conn.execute(
        f"SELECT DISTINCT ticker FROM prices WHERE ticker IN ({placeholders})",
        tickers,
    ).fetchall()
    return {r[0] for r in rows}


def _filter_pairs_with_data(
    conn: sqlite3.Connection, pairs: list[tuple[str, str]], label: str
) -> list[tuple[str, str]]:
    all_tickers = sorted({t for pair in pairs for t in pair})
    have_data = _tickers_with_data(conn, all_tickers)
    kept = [(a, b) for a, b in pairs if a in have_data and b in have_data]
    dropped = len(pairs) - len(kept)
    if dropped:
        print(f"{label}: dropped {dropped} pair(s) with no stored price data (of {len(pairs)} candidates)")
    return kept


def _filter_flat_with_data(conn: sqlite3.Connection, tickers: list[str], label: str) -> list[str]:
    have_data = _tickers_with_data(conn, tickers)
    kept = [t for t in tickers if t in have_data]
    dropped = len(tickers) - len(kept)
    if dropped:
        print(f"{label}: dropped {dropped} ticker(s) with no stored price data (of {len(tickers)} candidates)")
    return kept


def preferred_pairs(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Common/preferred share pairs, matched by name rather than ticker code.

    A preferred ticker's code conventionally ends in 5 (vs 0 for common), but
    that's a convention with exceptions, not a guarantee - so instead this
    strips the Korean preferred-share suffix (우 / 우B / 2우B / ...) off each
    name and checks whether the remainder matches an actual common-stock name
    in the tickers table. That match is the real confirmation; the regex only
    proposes candidates.
    """
    names = ticker_names(conn)  # ticker -> name
    ticker_by_name: dict[str, str] = {}
    for ticker, name in names.items():
        ticker_by_name.setdefault(name, ticker)

    pairs: list[tuple[str, str]] = []
    for ticker, name in names.items():
        m = _PREFERRED_SUFFIX.search(name)
        if not m:
            continue
        base = name[: m.start()]
        if not base or base == name:
            continue
        common_ticker = ticker_by_name.get(base)
        if common_ticker is None or common_ticker == ticker:
            continue
        pairs.append((common_ticker, ticker))

    pairs = _filter_pairs_with_data(conn, pairs, label="preferred_pairs")

    print(f"preferred_pairs: {len(pairs)} common/preferred pairs found")
    for common, pref in pairs[:5]:
        print(f"  {names.get(common, common)} ({common}) <-> {names.get(pref, pref)} ({pref})")

    return pairs


# Each pair is a (holding company, core subsidiary) ticker pair. The
# rationale in the comment is what makes it a candidate, not the tickers
# themselves - a holdco's share price tracks NAV (largely its subsidiary
# stakes) plus a discount that tends to mean-revert, which is the spread
# these pairs are meant to test.
_HOLDCO_PAIRS: list[tuple[str, str]] = [
    ("034730", "000660"),  # SK - SK하이닉스: SK그룹 NAV의 절대 비중을 차지하는 핵심 자회사
    ("034730", "017670"),  # SK - SK텔레콤: 그룹 통신 자회사, 안정적 배당수익의 핵심 축
    ("003550", "051910"),  # LG - LG화학: 배터리소재/화학 핵심 자회사, LG 지분가치의 큰 축
    ("003550", "066570"),  # LG - LG전자: 그룹 전자 핵심 자회사
    ("028260", "005930"),  # 삼성물산 - 삼성전자: 삼성물산 순자산가치의 절대 비중이 삼성전자 지분
    ("001040", "097950"),  # CJ - CJ제일제당: 식품/바이오 핵심 자회사
    ("000880", "009830"),  # 한화 - 한화솔루션: 화학/에너지 핵심 자회사
    ("078930", "006360"),  # GS - GS건설: 건설 핵심 자회사
]


def holdco_pairs(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Hand-picked holding company / core subsidiary pairs (see _HOLDCO_PAIRS)."""
    pairs = _filter_pairs_with_data(conn, _HOLDCO_PAIRS, label="holdco_pairs")
    print(f"holdco_pairs: {len(pairs)} / {len(_HOLDCO_PAIRS)} holding-subsidiary pairs have stored price data")
    return pairs


def industry_universe(keyword: str, min_marcap: float | None = None) -> list[str]:
    """KOSPI tickers whose KRX-DESC Industry text matches `keyword` (regex, OR-able with |).

    KRX's own industry classification is coarse (e.g. Samsung Electronics is
    filed under "통신 및 방송 장비 제조업", not "반도체 제조업"), so this is
    meant to be given a few related industry strings at once, not one exact
    match - see cointegration's --industry-keyword help text for an example.

    `min_marcap` (KRW) drops names below that market cap. This is meant to
    cut out micro-caps whose price series are erratic enough that
    Engle-Granger can flag "cointegration" against a large-cap target just
    from sharing a long-run uptrend, not from a real short-run relationship -
    it's a sanity filter, not a guarantee the survivors are stable.
    """
    import FinanceDataReader as fdr

    desc = fdr.StockListing("KRX-DESC")
    kospi = desc[desc["Market"] == "KOSPI"]
    matched = kospi[kospi["Industry"].str.contains(keyword, na=False, regex=True)]
    codes = matched["Code"].tolist()

    if min_marcap is not None:
        listing = fdr.StockListing("KOSPI")[["Code", "Marcap"]]
        marcap_by_code = dict(zip(listing["Code"], listing["Marcap"]))
        codes = [c for c in codes if marcap_by_code.get(c, 0) >= min_marcap]

    return codes


def sector_peers(conn: sqlite3.Connection, keyword: str, min_marcap: float | None = None) -> list[str]:
    """industry_universe(), restricted to tickers with stored price data."""
    codes = industry_universe(keyword, min_marcap=min_marcap)
    filtered = _filter_flat_with_data(conn, codes, label="sector_peers")
    print(f"sector_peers: {len(filtered)} / {len(codes)} KOSPI tickers matching '{keyword}' have stored price data")
    return filtered
