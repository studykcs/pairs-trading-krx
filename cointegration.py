"""Screen the KOSPI universe for pairs cointegrated with a target stock.

Runs an Engle-Granger cointegration test (statsmodels' coint) between the
target and every other ticker with enough overlapping history, then applies
Benjamini-Hochberg FDR correction across all p-values. Testing hundreds of
pairs at once means some will look "significant" by chance; FDR keeps the
expected share of false positives among the flagged pairs bounded, which a
raw p < 0.05 cutoff does not.

Each pair is tested in both directions. EG's first stage regresses one series
on the other to get a residual, then runs ADF on that residual - regressing A
on B gives a different residual (and p-value) than regressing B on A, so
coint(a, b) and coint(b, a) are not interchangeable. We keep whichever
direction is more significant and carry that choice forward (see `direction`
in the results) since the half-life regression, and later the hedge-ratio
regression, both need to use the same direction that produced the winning
residual. Because both directions are genuinely tested, FDR correction is
applied across all 2n p-values, not the n minimums - see
apply_significance_filters().

Pairs that survive FDR are further filtered on OU half-life: a cointegrated
pair that mean-reverts over 200 days is not tradeable at any reasonable
capital turnover, so half-life outside [--min-halflife, --max-halflife] (or
non-mean-reverting entirely) downgrades a pair to not-significant without
dropping it from the output - see apply_significance_filters() for why the
default range is what it is.

Usage
-----
    python cointegration.py --target 005930          # Samsung Electronics
    python cointegration.py --target 005930 --alpha 0.05 --min-obs 500
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

from store import get_connection, load_prices, ticker_names
from universe import holdco_pairs, preferred_pairs, sector_peers


def _eg_residual(dep: pd.Series, indep: pd.Series) -> np.ndarray:
    """OLS residual from regressing `dep` on `indep` with a constant.

    This reproduces coint()'s internal first-stage regression (it uses
    trend='c' by default) since statsmodels doesn't expose that residual -
    we need it ourselves to run the half-life regression on the same
    spread that produced the winning direction's p-value.
    """
    x = sm.add_constant(indep.values)
    fitted = sm.OLS(dep.values, x).fit().predict(x)
    return dep.values - fitted


def _ou_half_life(spread: np.ndarray) -> float:
    """OU half-life from Δspread_t = λ * spread_{t-1} + ε, fit by OLS through
    the origin (no intercept - the spec is a pure mean-reversion term, and
    the EG residual is already ~zero-mean by construction of the first-stage
    regression).

    half_life = -ln(2) / λ only makes sense when λ < 0 (the spread actually
    pulls back toward zero). λ >= 0 means this residual isn't mean-reverting
    on this sample despite passing the EG test - EG tests stationarity, not
    reversion speed, so this can happen - and half_life is NaN in that case
    rather than a nonsense negative or infinite number.
    """
    lag = spread[:-1]
    diff = spread[1:] - spread[:-1]
    lam = float(np.dot(lag, diff) / np.dot(lag, lag))
    if lam >= 0:
        return float("nan")
    return -np.log(2) / lam


def _test_pair_bidirectional(prices: pd.DataFrame, x: str, y: str, min_obs: int) -> dict | None:
    """Engle-Granger both directions, keeping the more significant one.

    coint(x, y) regresses x on y in its first stage; coint(y, x) regresses y
    on x. Different residual, different ADF p-value - so we run both and
    keep whichever is smaller. `direction` records which ticker was treated
    as the dependent variable, since the half-life regression below (and the
    hedge-ratio regression downstream) must use that same direction rather
    than an arbitrary one.
    """
    pair = prices[[x, y]].dropna()
    if len(pair) < min_obs:
        return None

    _, p_fwd, _ = coint(pair[x], pair[y])  # x dependent, y independent
    _, p_rev, _ = coint(pair[y], pair[x])  # y dependent, x independent

    used_fwd = p_fwd <= p_rev
    if used_fwd:
        direction, pvalue, dep, indep = x, p_fwd, pair[x], pair[y]
    else:
        direction, pvalue, dep, indep = y, p_rev, pair[y], pair[x]

    half_life = _ou_half_life(_eg_residual(dep, indep))

    return {
        "n_obs": len(pair),
        "pvalue_fwd": p_fwd,
        "pvalue_rev": p_rev,
        "pvalue": pvalue,
        "used_fwd": used_fwd,
        "direction": direction,
        "half_life": half_life,
    }


def screen_pairs(
    prices: pd.DataFrame, target: str, min_obs: int = 500, candidates: list[str] | None = None
) -> pd.DataFrame:
    """Bidirectional Engle-Granger test of `target` against every candidate column.

    `candidates` restricts the search universe (e.g. to one sector); defaults
    to every other column in `prices`. Returns one row per candidate with
    both directions' p-values, the winning one, its direction, and the OU
    half-life of the winning residual. FDR correction and half-life gating
    are applied separately in apply_significance_filters().
    """
    if target not in prices.columns:
        raise SystemExit(f"{target} not found in stored prices. Run collect.py first.")

    pool = candidates if candidates is not None else list(prices.columns)
    rows = []
    for ticker in pool:
        if ticker == target or ticker not in prices.columns:
            continue
        result = _test_pair_bidirectional(prices, target, ticker, min_obs)
        if result is None:
            continue
        rows.append({"ticker": ticker, **result})

    return pd.DataFrame(rows)


def screen_explicit_pairs(prices: pd.DataFrame, pairs: list[tuple[str, str]], min_obs: int = 500) -> pd.DataFrame:
    """Bidirectional Engle-Granger test on a fixed list of (a, b) pairs,
    rather than one target against many candidates. Used for --universe
    preferred/holdco, where the pairing itself (not the target) is the
    economic hypothesis."""
    rows = []
    for a, b in pairs:
        if a not in prices.columns or b not in prices.columns:
            continue
        result = _test_pair_bidirectional(prices, a, b, min_obs)
        if result is None:
            continue
        rows.append({"a": a, "b": b, **result})
    return pd.DataFrame(rows)


def apply_significance_filters(
    results: pd.DataFrame, alpha: float, min_halflife: float, max_halflife: float
) -> pd.DataFrame:
    """FDR-correct across both EG directions, then downgrade by OU half-life.

    FDR: each pair contributed two hypothesis tests (pvalue_fwd, pvalue_rev),
    even though only the smaller of the two is reported as `pvalue`. If we
    FDR-corrected only the n post-min p-values, we'd be correcting as though
    n tests were run when 2n actually were, understating the correction and
    inflating the false-discovery rate. So instead we correct the full 2n
    p-value vector (both directions, all pairs) and then, for each pair, take
    whichever of the two corrected values corresponds to the direction that
    was actually used.

    Half-life: --min-halflife/--max-halflife default to 5/60 days. This is
    not an empirically-derived range - it's an assumption chosen to be
    consistent with strategy.py's z-window default of 60 days (a spread that
    takes longer than the whole z-score lookback to revert isn't something
    that lookback can trade against; one that reverts in under ~5 days
    barely holds a position). Treat it as a default worth overriding per
    pair, not a validated threshold.

    Pairs that fail either filter are downgraded (`significant=False`,
    `reject_reason` set) but kept in the output, so a rejection is traceable
    instead of silently vanishing from the universe.
    """
    n = len(results)
    results = results.reset_index(drop=True)

    all_pvalues = pd.concat([results["pvalue_fwd"], results["pvalue_rev"]], ignore_index=True)
    _, adj_all, _, _ = multipletests(all_pvalues, alpha=alpha, method="fdr_bh")
    adj_fwd, adj_rev = adj_all[:n], adj_all[n:]
    results["adj_pvalue"] = np.where(results["used_fwd"].values, adj_fwd, adj_rev)

    reject_reason = pd.Series([None] * n, index=results.index, dtype=object)

    fdr_fail = results["adj_pvalue"] > alpha
    reject_reason[fdr_fail] = "fdr"

    not_mean_reverting = ~fdr_fail & results["half_life"].isna()
    reject_reason[not_mean_reverting] = "not_mean_reverting"

    remaining = ~fdr_fail & ~not_mean_reverting
    too_fast = remaining & (results["half_life"] < min_halflife)
    reject_reason[too_fast] = "halflife_too_fast"

    too_slow = remaining & (results["half_life"] > max_halflife)
    reject_reason[too_slow] = "halflife_too_slow"

    results["reject_reason"] = reject_reason
    results["significant"] = results["reject_reason"].isna()

    n_fdr_pass = int((~fdr_fail).sum())
    n_halflife_dropped = int((not_mean_reverting | too_fast | too_slow).sum())
    print(
        f"Half-life filter [{min_halflife}, {max_halflife}] days: dropped {n_halflife_dropped} "
        f"of {n_fdr_pass} FDR-significant pair(s) "
        f"({int(not_mean_reverting.sum())} not mean-reverting, "
        f"{int(too_fast.sum())} too fast, {int(too_slow.sum())} too slow)"
    )

    return results.drop(columns=["used_fwd"])


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
    parser.add_argument(
        "--tickers", default=None,
        help="Comma-separated explicit candidate list (overrides --industry-keyword), "
             'e.g. "055550,086790,316140" for a hand-picked universe.',
    )
    parser.add_argument(
        "--universe", choices=["preferred", "holdco", "sector"], default=None,
        help="Use an economically-motivated candidate universe from universe.py instead of "
             "--tickers/--industry-keyword. 'preferred' and 'holdco' test their own fixed pairs "
             "directly (--target is ignored); 'sector' feeds sector_peers() as --target's "
             "candidates and requires --industry-keyword.",
    )
    parser.add_argument(
        "--min-halflife", type=float, default=5,
        help="Drop (downgrade) pairs whose OU half-life is below this many days - see "
             "apply_significance_filters() docstring for why 5/60 are assumptions, not "
             "validated thresholds.",
    )
    parser.add_argument(
        "--max-halflife", type=float, default=60,
        help="Drop (downgrade) pairs whose OU half-life is above this many days.",
    )
    args = parser.parse_args()

    conn = get_connection()
    prices = load_prices(conn)
    if prices.empty:
        raise SystemExit("No data in prices.db. Run collect.py first.")

    names = ticker_names(conn)

    if args.universe in ("preferred", "holdco"):
        pairs = preferred_pairs(conn) if args.universe == "preferred" else holdco_pairs(conn)
        if not pairs:
            raise SystemExit(f"No {args.universe} pairs with stored price data.")

        results = screen_explicit_pairs(prices, pairs, min_obs=args.min_obs)
        if results.empty:
            raise SystemExit("No candidate pairs had enough overlapping history.")

        results = apply_significance_filters(results, args.alpha, args.min_halflife, args.max_halflife)
        results["name_a"] = results["a"].map(names)
        results["name_b"] = results["b"].map(names)
        results = results.sort_values("adj_pvalue")

        significant = results[results["significant"]]
        print(f"\n{len(significant)} / {len(results)} pairs significant after FDR + half-life filters (alpha={args.alpha})\n")
        print(significant[["name_a", "name_b", "n_obs", "pvalue", "adj_pvalue", "direction", "half_life"]].to_string(index=False))

        results.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\nFull results (all {len(results)} candidates) written to {args.out}")
        return

    target_name = names.get(args.target, args.target)

    candidates = None
    if args.tickers:
        candidates = [t.strip() for t in args.tickers.split(",")]
        print(f"Explicit ticker list: {len(candidates)} candidates")
    elif args.universe == "sector":
        if not args.industry_keyword:
            raise SystemExit("--universe sector requires --industry-keyword.")
        candidates = sector_peers(conn, args.industry_keyword, min_marcap=args.min_marcap)
    elif args.industry_keyword:
        candidates = sector_peers(conn, args.industry_keyword, min_marcap=args.min_marcap)

    n_candidates = len(candidates) if candidates is not None else prices.shape[1] - 1
    print(f"Testing {target_name} ({args.target}) against {n_candidates} tickers, {len(prices)} days available")

    results = screen_pairs(prices, args.target, min_obs=args.min_obs, candidates=candidates)
    if results.empty:
        raise SystemExit("No candidate pairs had enough overlapping history.")

    results = apply_significance_filters(results, args.alpha, args.min_halflife, args.max_halflife)
    results["name"] = results["ticker"].map(names)
    results = results.sort_values("adj_pvalue")

    significant = results[results["significant"]]
    print(f"\n{len(significant)} / {len(results)} pairs significant after FDR + half-life filters (alpha={args.alpha})\n")
    print(significant[["ticker", "name", "n_obs", "pvalue", "adj_pvalue", "direction", "half_life"]].to_string(index=False))

    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nFull results (all {len(results)} candidates) written to {args.out}")


if __name__ == "__main__":
    main()
