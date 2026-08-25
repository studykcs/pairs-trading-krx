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
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

from store import get_connection, load_prices, ticker_names


def screen_basket(prices: pd.DataFrame, tickers: list[str], min_obs: int = 500) -> pd.DataFrame:
    rows = []
    for a, b in combinations(tickers, 2):
        if a not in prices.columns or b not in prices.columns:
            continue
        pair = prices[[a, b]].dropna()
        if len(pair) < min_obs:
            continue
        _, pvalue, _ = coint(pair[a], pair[b])
        rows.append({"a": a, "b": b, "n_obs": len(pair), "pvalue": pvalue})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker codes")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-obs", type=int, default=500)
    parser.add_argument("--out", default="pairs_basket.csv")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    conn = get_connection()
    prices = load_prices(conn)
    names = ticker_names(conn)

    print(f"Screening {len(tickers)} tickers -> {len(list(combinations(tickers, 2)))} pairwise combinations")
    results = screen_basket(prices, tickers, min_obs=args.min_obs)
    if results.empty:
        raise SystemExit("No pair had enough overlapping history.")

    rejected, adj_pvalues, _, _ = multipletests(results["pvalue"], alpha=args.alpha, method="fdr_bh")
    results["adj_pvalue"] = adj_pvalues
    results["significant"] = rejected
    results["name_a"] = results["a"].map(names)
    results["name_b"] = results["b"].map(names)
    results = results.sort_values("adj_pvalue")

    significant = results[results["significant"]]
    print(f"\n{len(significant)} / {len(results)} pairs significant after FDR correction (alpha={args.alpha})\n")
    cols = ["name_a", "name_b", "n_obs", "pvalue", "adj_pvalue"]
    print(results[cols].to_string(index=False))

    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
