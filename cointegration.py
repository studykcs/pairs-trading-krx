"""Screen the KOSPI universe for pairs cointegrated with a target stock.

Runs an Engle-Granger cointegration test (statsmodels' coint) between the
target and every other ticker with enough overlapping history, then applies
Benjamini-Hochberg FDR correction across all p-values. Testing hundreds of
pairs at once means some will look "significant" by chance; FDR keeps the
expected share of false positives among the flagged pairs bounded, which a
raw p < 0.05 cutoff does not.

Usage
-----
    python cointegration.py --target 005930          # Samsung Electronics
    python cointegration.py --target 005930 --alpha 0.05 --min-obs 500
"""

from __future__ import annotations

import argparse

import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

from store import get_connection, load_prices, ticker_names


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


def screen_pairs(
    prices: pd.DataFrame, target: str, min_obs: int = 500, candidates: list[str] | None = None
) -> pd.DataFrame:
    """Engle-Granger test of `target` against every candidate column.

    `candidates` restricts the search universe (e.g. to one sector); defaults
    to every other column in `prices`. Returns one row per candidate with
    the raw p-value; FDR correction is applied separately in main().
    """
    if target not in prices.columns:
        raise SystemExit(f"{target} not found in stored prices. Run collect.py first.")

    pool = candidates if candidates is not None else list(prices.columns)
    rows = []
    for ticker in pool:
        if ticker == target or ticker not in prices.columns:
            continue
        pair = prices[[target, ticker]].dropna()
        if len(pair) < min_obs:
            continue
        _, pvalue, _ = coint(pair[target], pair[ticker])
        rows.append({"ticker": ticker, "n_obs": len(pair), "pvalue": pvalue})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="005930", help="Ticker to pair against (default: Samsung Electronics)")
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR-adjusted significance level")
    parser.add_argument("--min-obs", type=int, default=500, help="Minimum overlapping trading days required")
    parser.add_argument("--out", default="pairs.csv", help="Where to write the significant pairs")
    parser.add_argument(
        "--industry-keyword",
        default=None,
        help='Restrict candidates to KOSPI tickers whose Industry matches this regex, '
             'e.g. "반도체|전자부품|통신 및 방송 장비". Default: whole KOSPI universe.',
    )
    parser.add_argument(
        "--min-marcap", type=float, default=None,
        help="Drop candidates below this market cap in KRW, e.g. 500000000000 for 5000억. "
             "Only applies together with --industry-keyword.",
    )
    args = parser.parse_args()

    conn = get_connection()
    prices = load_prices(conn)
    if prices.empty:
        raise SystemExit("No data in prices.db. Run collect.py first.")

    names = ticker_names(conn)
    target_name = names.get(args.target, args.target)

    candidates = None
    if args.industry_keyword:
        candidates = industry_universe(args.industry_keyword, min_marcap=args.min_marcap)
        cap_note = f", marcap >= {args.min_marcap:,.0f}" if args.min_marcap else ""
        print(f"Industry filter matched {len(candidates)} KOSPI tickers{cap_note}")

    n_candidates = len(candidates) if candidates is not None else prices.shape[1] - 1
    print(f"Testing {target_name} ({args.target}) against {n_candidates} tickers, {len(prices)} days available")

    results = screen_pairs(prices, args.target, min_obs=args.min_obs, candidates=candidates)
    if results.empty:
        raise SystemExit("No candidate pairs had enough overlapping history.")

    rejected, adj_pvalues, _, _ = multipletests(results["pvalue"], alpha=args.alpha, method="fdr_bh")
    results["adj_pvalue"] = adj_pvalues
    results["significant"] = rejected
    results["name"] = results["ticker"].map(names)
    results = results.sort_values("adj_pvalue")

    significant = results[results["significant"]]
    print(f"\n{len(significant)} / {len(results)} pairs significant after FDR correction (alpha={args.alpha})\n")
    print(significant[["ticker", "name", "n_obs", "pvalue", "adj_pvalue"]].to_string(index=False))

    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nFull results (all {len(results)} candidates) written to {args.out}")


if __name__ == "__main__":
    main()
