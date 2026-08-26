"""Screen every pairwise combination within a hand-picked basket of stocks.

Unlike cointegration.py (one target vs many candidates), this tests all
C(n,2) combinations within a small basket and FDR-corrects across all of
them at once - meant for "check these specific well-known names against
each other" rather than a broad one-vs-universe scan.

Usage
-----
    python screen_basket.py --tickers 005930,051900,035420,068270
    python screen_basket.py --tickers 005930,051900,035420,068270,000270,005380 --min-obs 2000
"""

from __future__ import annotations

import argparse
from itertools import combinations

import pandas as pd

from cointegration import apply_significance_filters, screen_explicit_pairs
from store import get_connection, load_prices, ticker_names
from universe import holdco_pairs, preferred_pairs, sector_peers


def screen_basket(prices: pd.DataFrame, tickers: list[str], min_obs: int = 500) -> pd.DataFrame:
    return screen_explicit_pairs(prices, list(combinations(tickers, 2)), min_obs=min_obs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker codes (overridden by --universe)")
    parser.add_argument(
        "--universe", choices=["preferred", "holdco", "sector"], default=None,
        help="Use an economically-motivated basket from universe.py instead of --tickers. "
             "'preferred' and 'holdco' test their own fixed pairs directly (all C(n,2) "
             "combinations are NOT taken - the pairing itself is the hypothesis); 'sector' "
             "builds a ticker basket via sector_peers() and tests all pairwise combinations, "
             "and requires --industry-keyword.",
    )
    parser.add_argument(
        "--industry-keyword", default=None,
        help='Only with --universe sector: regex matched against KOSPI Industry text, '
             'e.g. "반도체|전자부품".',
    )
    parser.add_argument(
        "--min-marcap", type=float, default=None,
        help="Only with --universe sector: drop candidates below this market cap in KRW.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-obs", type=int, default=500)
    parser.add_argument(
        "--min-halflife", type=float, default=5,
        help="Drop (downgrade) pairs whose OU half-life is below this many days - see "
             "cointegration.apply_significance_filters() docstring for why 5/60 are "
             "assumptions, not validated thresholds.",
    )
    parser.add_argument(
        "--max-halflife", type=float, default=60,
        help="Drop (downgrade) pairs whose OU half-life is above this many days.",
    )
    parser.add_argument("--out", default="pairs_basket.csv")
    args = parser.parse_args()

    if not args.tickers and not args.universe:
        raise SystemExit("Provide --tickers or --universe {preferred,holdco,sector}.")

    conn = get_connection()
    prices = load_prices(conn)
    names = ticker_names(conn)

    if args.universe in ("preferred", "holdco"):
        pairs = preferred_pairs(conn) if args.universe == "preferred" else holdco_pairs(conn)
        if not pairs:
            raise SystemExit(f"No {args.universe} pairs with stored price data.")
        print(f"Screening {len(pairs)} {args.universe} pairs directly (no combinations)")
        results = screen_explicit_pairs(prices, pairs, min_obs=args.min_obs)
    elif args.universe == "sector":
        if not args.industry_keyword:
            raise SystemExit("--universe sector requires --industry-keyword.")
        tickers = sector_peers(conn, args.industry_keyword, min_marcap=args.min_marcap)
        print(f"Screening {len(tickers)} tickers -> {len(list(combinations(tickers, 2)))} pairwise combinations")
        results = screen_basket(prices, tickers, min_obs=args.min_obs)
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]
        print(f"Screening {len(tickers)} tickers -> {len(list(combinations(tickers, 2)))} pairwise combinations")
        results = screen_basket(prices, tickers, min_obs=args.min_obs)

    if results.empty:
        raise SystemExit("No pair had enough overlapping history.")

    results = apply_significance_filters(results, args.alpha, args.min_halflife, args.max_halflife)
    results["name_a"] = results["a"].map(names)
    results["name_b"] = results["b"].map(names)
    results = results.sort_values("adj_pvalue")

    significant = results[results["significant"]]
    print(f"\n{len(significant)} / {len(results)} pairs significant after FDR + half-life filters (alpha={args.alpha})\n")
    cols = ["name_a", "name_b", "n_obs", "pvalue", "adj_pvalue", "direction", "half_life"]
    print(results[cols].to_string(index=False))

    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
