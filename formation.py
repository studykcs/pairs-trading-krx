"""Rolling formation/trading pair selection (Gatev, Goetzmann & Rouwenhorst 2006).

This is the fix for the repo's biggest structural flaw (see CLAUDE.md item 1):
cointegration.py picks pairs on the *entire* sample, then backtest.py trades
those same pairs over that same entire sample. Beta is walk-forward, but the
decision "trade this pair at all" is not - in 2015 there was no way to know a
pair would turn out cointegrated over 2015-2026, because 2016-2026 hadn't
happened yet.

The Gatev fix separates the two decisions in time:

    formation window (--formation-days, default 252): run the bidirectional
        Engle-Granger test + half-life filter (cointegration.py) on ONLY this
        window's prices, FDR-corrected within the window. Pairs that survive
        get an initial hedge ratio and a frozen z-score reference (mean/std
        of the formation-period spread).

    trading window (--trading-days, default 126): trade ONLY the pairs
        selected in the immediately preceding formation window, using the
        beta and z-reference frozen at formation - no re-selection, no
        re-estimation. Position is forced flat at the end of the window,
        because the pair's "selected" status expires with the window.

    roll forward by exactly --trading-days: the next window's formation
    period starts where this window's trading period starts. Trading windows
    therefore tile the full sample with no gaps and no overlap; formation
    windows overlap each other (by formation_days - trading_days, with the
    defaults), which is fine because formation is only ever used for
    selection, never for computing a return that gets reported.

Everything statistical is reused, not reimplemented: universe construction
comes from universe.py, the bidirectional EG test + half-life filter come
from cointegration.py's screen_pairs/screen_explicit_pairs/
apply_significance_filters, the beta estimator and z-to-position mapping come
from backtest.py, and per-day transaction costs come from costs.py. This
module's only new logic is the window-rolling loop, the frozen-parameter
trading simulation, and forced end-of-window liquidation.

Multiple pairs selected in the same window are combined equal-weight: each
gets capital/n_pairs, so a pair's simple daily return times its 1/n weight,
summed across pairs, is exactly the window's portfolio return relative to
total capital (this is exact, not an approximation, because the linear cost
components scale with capital and market impact is computed on the actual
smaller per-pair order size rather than being scaled after the fact).

Expect this to look much worse than backtest.py's full-sample numbers - see
--compare. That is the point, not a bug: Gatev-style out-of-sample selection
throwing away most of the apparent edge is exactly what "the pair selection
itself was look-ahead" means. Do not narrow the half-life range or shrink
--formation-days to make a window's numbers look better; report the defaults
as they come out, including windows that select zero pairs and trade nothing
- "tradeable pairs are rare out-of-sample" is a legitimate finding here, not
a failed run.

Usage
-----
    python formation.py --universe holdco
    python formation.py --universe preferred --formation-days 252 --trading-days 126
    python formation.py --universe sector --industry-keyword "반도체|전자부품" --target 005930
    python formation.py --universe holdco --compare
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest import _positions_from_zscore, hedge_ratio, run_backtest_walkforward
from cointegration import apply_significance_filters, screen_explicit_pairs, screen_pairs
from costs import CostModel, pair_cost_series
from store import get_connection, load_field, load_prices, ticker_names
from universe import holdco_pairs, preferred_pairs, sector_peers

# >= leg_cost_series' default vol_window (60) / adv_window (20), so the first
# days of every trading window get a real trailing ADV/vol estimate instead
# of NaN -> zero impact. Hardcoded rather than threaded through from costs.py
# because pair_cost_series doesn't expose those window lengths as parameters.
BUFFER_DAYS = 60


@dataclass
class WindowResult:
    window: int
    formation_start: str
    formation_end: str
    trading_start: str
    trading_end: str
    n_candidates: int
    pairs: list[dict] = field(default_factory=list)
    window_return: float = 0.0
    n_trades: int = 0


def _iter_windows(n_days: int, formation_days: int, trading_days: int):
    """Yields (k, formation_start, formation_end, trading_start, trading_end)
    positional (iloc) indices. trading_start of window k+1 equals
    trading_end of window k, since each window starts trading_days after the
    previous one - trading periods tile the sample with no gaps or overlap;
    formation periods are allowed to overlap (they're only ever used for
    selection, and their overlap is exactly what lets the roll advance by
    trading_days instead of formation_days)."""
    k = 0
    while True:
        formation_start = k * trading_days
        formation_end = formation_start + formation_days
        trading_start = formation_end
        trading_end = trading_start + trading_days
        if trading_end > n_days:
            return
        yield k, formation_start, formation_end, trading_start, trading_end
        k += 1


def _select_pairs_in_window(
    formation_prices: pd.DataFrame,
    universe_kind: str,
    target: str | None,
    candidates_or_pairs,
    min_obs: int,
    alpha: float,
    min_halflife: float,
    max_halflife: float,
) -> pd.DataFrame:
    """B-2's bidirectional EG test + half-life filter, run on ONLY this
    formation window's prices and FDR-corrected within the window - each
    window is its own independent multiple-testing universe, not part of a
    global correction. Delegates entirely to cointegration.py; this function
    just normalizes screen_pairs' (target-vs-many) and screen_explicit_pairs'
    (fixed pair list) different column shapes into a common dep/indep pair
    identity so the rest of this module doesn't need to know which mode
    produced a row.
    """
    if universe_kind == "sector":
        results = screen_pairs(formation_prices, target, min_obs=min_obs, candidates=candidates_or_pairs)
        if not results.empty:
            results["x"] = target
            results["y"] = results["ticker"]
    else:
        results = screen_explicit_pairs(formation_prices, candidates_or_pairs, min_obs=min_obs)
        if not results.empty:
            results["x"] = results["a"]
            results["y"] = results["b"]

    if results.empty:
        return results

    results = apply_significance_filters(results, alpha, min_halflife, max_halflife)
    results["indep"] = np.where(results["direction"] == results["x"], results["y"], results["x"])
    results = results.rename(columns={"direction": "dep"})
    return results


def _initial_hedge_ratio(formation_prices: pd.DataFrame, dep: str, indep: str) -> tuple[float, float, float] | None:
    """Beta from backtest.py's hedge_ratio(), estimated once on the formation
    window and then frozen for the whole trading window (Gatev - no
    re-estimation during trading). mu/sigma are the formation-period
    spread's mean/std, used as the trading window's fixed z-score reference
    instead of a rolling window that would need trading-period data to fill.
    """
    pair = formation_prices[[dep, indep]].dropna()
    log_dep, log_indep = np.log(pair[dep]), np.log(pair[indep])
    result = hedge_ratio(log_dep, log_indep)
    if result is None:
        return None
    beta, _alpha = result
    spread = log_dep - beta * log_indep
    sigma = float(spread.std())
    if not sigma or pd.isna(sigma):
        return None
    return beta, float(spread.mean()), sigma


def _run_trading_window(
    prices: pd.DataFrame,
    trdval: pd.DataFrame | None,
    dep: str,
    indep: str,
    beta: float,
    mu: float,
    sigma: float,
    trading_start: int,
    trading_end: int,
    entry: float,
    exit_z: float,
    capital: float,
    cost_model: CostModel,
    allow_missing_adv: bool,
) -> tuple[pd.Series, int]:
    """Trades one pair through one trading window with beta/mu/sigma frozen
    at formation. Pulls in BUFFER_DAYS of history before the window so
    costs.py's rolling ADV/vol windows are populated on day 1 of trading
    (fed zero position over the buffer, so no phantom trading happens there)
    then reindexes back down to just the trading window before returning.
    Position is forced to zero on the window's last day: the pair's selected
    status ends with the window, so whatever is open must be liquidated
    (and pay the liquidation cost) rather than carried into a window where
    it was never re-validated.

    Returns a LOG-return series (matching backtest.py's spread_ret
    convention) over the trading window, and the number of position changes
    (trades) including the forced exit.
    """
    ext_start = max(0, trading_start - BUFFER_DAYS)
    ext_index = prices.index[ext_start:trading_end]
    trading_index = prices.index[trading_start:trading_end]

    log_dep = np.log(prices[dep].reindex(ext_index))
    log_indep = np.log(prices[indep].reindex(ext_index))
    spread = log_dep - beta * log_indep
    z = (spread - mu) / sigma

    position_trading = _positions_from_zscore(z.loc[trading_index], entry, exit_z)
    position = pd.Series(0.0, index=ext_index)
    position.loc[trading_index] = position_trading
    position.iloc[-1] = 0.0  # forced liquidation at the window boundary

    beta_series = pd.Series(beta, index=ext_index)
    target_ret = prices[dep].reindex(ext_index).pct_change(fill_method=None)
    pair_ret = prices[indep].reindex(ext_index).pct_change(fill_method=None)
    target_adv = trdval[dep].reindex(ext_index) if trdval is not None and dep in trdval.columns else None
    pair_adv = trdval[indep].reindex(ext_index) if trdval is not None and indep in trdval.columns else None

    cost = pair_cost_series(
        position=position, beta=beta_series, target_ret=target_ret, pair_ret=pair_ret,
        target_adv=target_adv, pair_adv=pair_adv, capital=capital, model=cost_model,
        allow_missing_adv=allow_missing_adv,
    )

    spread_ret = log_dep.diff() - beta * log_indep.diff()
    gross_ret = (position * spread_ret).fillna(0)
    net_ret = (gross_ret - cost["total"]).loc[trading_index]

    n_trades = int((position.diff().fillna(position.iloc[0]) != 0).sum())
    return net_ret, n_trades


def run_formation_backtest(
    prices: pd.DataFrame,
    trdval: pd.DataFrame | None,
    universe_kind: str,
    candidates_or_pairs,
    target: str | None,
    formation_days: int,
    trading_days: int,
    min_obs: int,
    alpha: float,
    min_halflife: float,
    max_halflife: float,
    entry: float,
    exit_z: float,
    capital: float,
    cost_model: CostModel,
    allow_missing_adv: bool,
    verbose: bool = True,
) -> tuple[list[WindowResult], pd.Series]:
    """Rolls formation/trading windows across the whole sample. Returns the
    per-window record list and a single stitched SIMPLE-return series
    (windows are contiguous and non-overlapping by construction, so this is
    a plain concatenation, not a re-weighting)."""
    windows: list[WindowResult] = []
    all_returns: list[pd.Series] = []

    for k, f_start, f_end, t_start, t_end in _iter_windows(len(prices), formation_days, trading_days):
        formation_slice = prices.iloc[f_start:f_end]
        trading_index = prices.index[t_start:t_end]

        selected = _select_pairs_in_window(
            formation_slice, universe_kind, target, candidates_or_pairs,
            min_obs, alpha, min_halflife, max_halflife,
        )
        sig = selected[selected["significant"]] if not selected.empty else selected

        pair_infos, pair_log_returns, n_trades_window = [], [], 0
        n_pairs = len(sig)
        capital_per_pair = capital / n_pairs if n_pairs else 0.0

        for _, row in sig.iterrows():
            dep, indep = row["dep"], row["indep"]
            hr = _initial_hedge_ratio(formation_slice, dep, indep)
            if hr is None:
                continue  # flat leg or zero-variance spread over formation - can't trade this pair
            beta, mu, sigma = hr

            net_ret, n_trades = _run_trading_window(
                prices, trdval, dep, indep, beta, mu, sigma, t_start, t_end,
                entry, exit_z, capital_per_pair, cost_model, allow_missing_adv,
            )
            pair_log_returns.append(net_ret)
            n_trades_window += n_trades
            pair_infos.append({
                "dep": dep, "indep": indep, "pvalue": float(row["pvalue"]),
                "adj_pvalue": float(row["adj_pvalue"]), "half_life": float(row["half_life"]),
                "beta": beta,
            })

        if pair_log_returns:
            # Convert each pair's log return to a simple return before
            # averaging: with equal capital/n_pairs per leg, the mean of
            # simple returns IS the portfolio's simple return relative to
            # total capital exactly (see module docstring) - averaging log
            # returns instead would only be a small-return approximation.
            simple_rets = [np.exp(r) - 1 for r in pair_log_returns]
            window_ret_series = pd.concat(simple_rets, axis=1).mean(axis=1)
        else:
            window_ret_series = pd.Series(0.0, index=trading_index)

        window_total_return = float((1 + window_ret_series).prod() - 1)

        result = WindowResult(
            window=k,
            formation_start=str(prices.index[f_start].date()),
            formation_end=str(prices.index[f_end - 1].date()),
            trading_start=str(prices.index[t_start].date()),
            trading_end=str(prices.index[t_end - 1].date()),
            n_candidates=len(selected) if not selected.empty else 0,
            pairs=pair_infos,
            window_return=window_total_return,
            n_trades=n_trades_window,
        )
        windows.append(result)
        all_returns.append(window_ret_series)

        if verbose:
            if n_pairs == 0:
                print(f"[window {k:2d}] formation {result.formation_start}~{result.formation_end} | "
                      f"trading {result.trading_start}~{result.trading_end} | "
                      f"{result.n_candidates} tested, 0 selected -> no trading this window")
            else:
                print(f"[window {k:2d}] formation {result.formation_start}~{result.formation_end} | "
                      f"trading {result.trading_start}~{result.trading_end} | "
                      f"{result.n_candidates} tested, {n_pairs} selected | "
                      f"return {window_total_return*100:+.2f}% | trades {result.n_trades}")

    full_returns = pd.concat(all_returns) if all_returns else pd.Series(dtype=float)
    return windows, full_returns


def run_full_sample_baseline(
    prices: pd.DataFrame,
    universe_kind: str,
    candidates_or_pairs,
    target: str | None,
    min_obs: int,
    alpha: float,
    min_halflife: float,
    max_halflife: float,
    beta_window: int,
    reestimate_every: int,
    z_window: int,
    entry: float,
    exit_z: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduces this repo's ORIGINAL pipeline exactly, for --compare:
    select pairs once on the FULL sample (single global FDR + half-life
    filter, min_obs=500 to match cointegration.py's own CLI default) then
    backtest each selected pair over that same full sample with backtest.py's
    walk-forward beta - unmodified, imported directly, no cost model (the
    same frictionless assumption backtest.py already makes).

    This isolates the look-ahead effect from pair selection, but NOT from
    cost treatment - formation.py's rolling result includes costs.py,
    this baseline does not, so part of any gap is "costs got added", not
    just "look-ahead got removed". Reported as-is; see README.
    """
    selected = _select_pairs_in_window(
        prices, universe_kind, target, candidates_or_pairs,
        min_obs=500, alpha=alpha, min_halflife=min_halflife, max_halflife=max_halflife,
    )
    if selected.empty:
        return selected, pd.Series(dtype=float)

    sig = selected[selected["significant"]]
    pair_simple_rets = []
    for _, row in sig.iterrows():
        dep, indep = row["dep"], row["indep"]
        pair_prices = prices[[dep, indep]].dropna()
        df, _ = run_backtest_walkforward(
            pair_prices[dep], pair_prices[indep], beta_window, reestimate_every, z_window, entry, exit_z,
        )
        pair_simple_rets.append(np.exp(df["strategy_ret"]) - 1)

    baseline_returns = (
        pd.concat(pair_simple_rets, axis=1).mean(axis=1).sort_index()
        if pair_simple_rets else pd.Series(dtype=float)
    )
    return selected, baseline_returns


def _aggregate_stats(returns: pd.Series) -> dict:
    """Stats over a stitched SIMPLE-return series. Not backtest.py's
    _summarize(): that function assumes one log-return series driven by one
    position/beta series, which doesn't fit a multi-pair, multi-window
    portfolio built by averaging simple returns across heterogeneous pairs."""
    if returns.empty:
        return {"total_return": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan"), "n_days": 0}
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else float("nan")
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "n_days": len(returns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", choices=["preferred", "holdco", "sector"], required=True,
                         help="Candidate universe from universe.py - see B-1. No plain whole-KOSPI mode here on purpose.")
    parser.add_argument("--target", default="005930", help="Only used with --universe sector")
    parser.add_argument("--industry-keyword", default=None, help="Required with --universe sector")
    parser.add_argument("--min-marcap", type=float, default=None, help="Only with --universe sector")
    parser.add_argument("--formation-days", type=int, default=252)
    parser.add_argument("--trading-days", type=int, default=126)
    parser.add_argument(
        "--min-obs", type=int, default=None,
        help="Minimum overlapping days required WITHIN a formation window. Default: "
             "90%% of --formation-days (tolerates some missing/halted days without "
             "requiring perfectly complete coverage).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR level, applied independently per formation window")
    parser.add_argument("--min-halflife", type=float, default=5)
    parser.add_argument("--max-halflife", type=float, default=60)
    parser.add_argument("--entry", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--capital", type=float, default=100_000_000, help="Total capital in KRW, split equally across a window's selected pairs")

    g = parser.add_argument_group("cost model (costs.py)")
    g.add_argument("--commission-bps", type=float, default=CostModel.commission_bps)
    g.add_argument("--tax-bps", type=float, default=CostModel.tax_bps)
    g.add_argument("--half-spread-bps", type=float, default=CostModel.half_spread_bps)
    g.add_argument("--borrow-bps", type=float, default=CostModel.borrow_fee_annual_bps)
    g.add_argument("--impact-coef", type=float, default=CostModel.impact_coef)
    g.add_argument("--allow-missing-adv", action="store_true")

    c = parser.add_argument_group("--compare (old full-sample pipeline, for side-by-side)")
    c.add_argument("--compare", action="store_true",
                   help="Also run the original full-sample selection + backtest.py pipeline")
    c.add_argument("--beta-window", type=int, default=120, help="[--compare] backtest.py walk-forward beta window")
    c.add_argument("--reestimate-every", type=int, default=20, help="[--compare] backtest.py beta re-estimation interval")
    c.add_argument("--z-window", type=int, default=60, help="[--compare] backtest.py rolling z-score window")

    parser.add_argument("--windows-out", default="formation_windows.csv", help="Per-window summary CSV")
    parser.add_argument("--pairs-out", default="formation_pairs.csv", help="Selected pairs across all windows, CSV")
    args = parser.parse_args()

    if args.universe == "sector" and not args.industry_keyword:
        raise SystemExit("--universe sector requires --industry-keyword.")

    conn = get_connection()
    prices = load_prices(conn)
    if prices.empty:
        raise SystemExit("No data in prices.db. Run collect.py first.")
    trdval = load_field(conn, "trdval")
    names = ticker_names(conn)

    if args.universe == "preferred":
        candidates_or_pairs = preferred_pairs(conn)
    elif args.universe == "holdco":
        candidates_or_pairs = holdco_pairs(conn)
    else:
        candidates_or_pairs = sector_peers(conn, args.industry_keyword, min_marcap=args.min_marcap)

    min_obs = args.min_obs if args.min_obs is not None else int(args.formation_days * 0.9)
    cost_model = CostModel(
        commission_bps=args.commission_bps, tax_bps=args.tax_bps,
        half_spread_bps=args.half_spread_bps, borrow_fee_annual_bps=args.borrow_bps,
        impact_coef=args.impact_coef,
    )

    n_days = len(prices)
    n_windows_expected = max(0, (n_days - args.formation_days) // args.trading_days)
    print(f"{n_days} trading days available -> up to {n_windows_expected} rolling formation/trading "
          f"windows (formation={args.formation_days}d, trading={args.trading_days}d, min_obs={min_obs})\n")

    windows, full_returns = run_formation_backtest(
        prices, trdval, args.universe, candidates_or_pairs, args.target,
        args.formation_days, args.trading_days, min_obs, args.alpha,
        args.min_halflife, args.max_halflife, args.entry, args.exit_z,
        args.capital, cost_model, args.allow_missing_adv,
    )

    n_zero_pair_windows = sum(1 for w in windows if not w.pairs)
    n_total_pairs_selected = sum(len(w.pairs) for w in windows)
    stats = _aggregate_stats(full_returns)

    print(f"\n{len(windows)} windows rolled | {n_zero_pair_windows} selected zero pairs "
          f"({n_zero_pair_windows / len(windows) * 100:.0f}%) | "
          f"{n_total_pairs_selected} pair-window selections total\n")
    print("Rolling formation/trading result (out-of-sample selection, costs.py applied):")
    print(f"  total return: {stats['total_return']*100:+.2f}%")
    print(f"  Sharpe (annualized): {stats['sharpe']:.2f}")
    print(f"  max drawdown: {stats['max_drawdown']*100:.2f}%")
    print(f"  trading days: {stats['n_days']}")
    print(f"  total trades: {sum(w.n_trades for w in windows)}")

    windows_df = pd.DataFrame([{
        "window": w.window, "formation_start": w.formation_start, "formation_end": w.formation_end,
        "trading_start": w.trading_start, "trading_end": w.trading_end,
        "n_candidates": w.n_candidates, "n_selected": len(w.pairs),
        "window_return": w.window_return, "n_trades": w.n_trades,
    } for w in windows])
    windows_df.to_csv(args.windows_out, index=False, encoding="utf-8-sig")
    print(f"\nPer-window summary written to {args.windows_out}")

    pairs_rows = []
    for w in windows:
        for p in w.pairs:
            pairs_rows.append({
                "window": w.window, "trading_start": w.trading_start, "trading_end": w.trading_end,
                "dep": p["dep"], "dep_name": names.get(p["dep"], p["dep"]),
                "indep": p["indep"], "indep_name": names.get(p["indep"], p["indep"]),
                "pvalue": p["pvalue"], "adj_pvalue": p["adj_pvalue"],
                "half_life": p["half_life"], "beta": p["beta"],
            })
    pairs_df = pd.DataFrame(pairs_rows)
    pairs_df.to_csv(args.pairs_out, index=False, encoding="utf-8-sig")
    print(f"Selected pairs (all windows) written to {args.pairs_out}")

    if args.compare:
        print("\n--- --compare: original full-sample selection + backtest.py (frictionless) ---")
        baseline_selected, baseline_returns = run_full_sample_baseline(
            prices, args.universe, candidates_or_pairs, args.target,
            min_obs=500, alpha=args.alpha, min_halflife=args.min_halflife, max_halflife=args.max_halflife,
            beta_window=args.beta_window, reestimate_every=args.reestimate_every, z_window=args.z_window,
            entry=args.entry, exit_z=args.exit_z,
        )
        n_baseline_selected = int(baseline_selected["significant"].sum()) if not baseline_selected.empty else 0
        baseline_stats = _aggregate_stats(baseline_returns)
        print(f"Full-sample selection: {n_baseline_selected} / {len(baseline_selected)} pairs significant "
              f"(single global FDR + half-life filter, min_obs=500)")
        print(f"  total return: {baseline_stats['total_return']*100:+.2f}%")
        print(f"  Sharpe (annualized): {baseline_stats['sharpe']:.2f}")
        print(f"  max drawdown: {baseline_stats['max_drawdown']*100:.2f}%")
        print(f"  trading days: {baseline_stats['n_days']}")
        print("\nNote: this baseline has NO cost model (backtest.py has none); the rolling result above")
        print("does. Part of the gap between the two is look-ahead removal, part is cost realism - see README.")


if __name__ == "__main__":
    main()
