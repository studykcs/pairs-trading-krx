"""Backtest a z-score pairs-trading strategy on one cointegrated pair.

Two hedge-ratio methods:

  static (old default): OLS on log prices over the *whole* sample, once.
  Simple, but has look-ahead bias - beta used for a 2023 trade is informed
  by 2026 data that didn't exist yet. Useful only as an upper-bound sanity
  check.

  walkforward (default now): beta is re-estimated every `--reestimate-every`
  trading days, using only the trailing `--beta-window` days *before* that
  point, and held constant until the next re-estimation. Each day only ever
  trades on a beta computed from data that would have actually been
  available at the time - no look-ahead.

Strategy: rolling z-score of the spread (log_target - beta * log_pair).
Enter when |z| crosses `--entry`, flatten when it falls back inside
`--exit-z`. Position is additionally lagged one day (trade on the next
bar's return, not the bar that generated the signal).

Usage
-----
    python backtest.py --target 000660 --pair 005930
    python backtest.py --target 000660 --pair 005930 --method static
    python backtest.py --target 000660 --pair 005930 --beta-window 120 --reestimate-every 20
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from store import get_connection, load_prices, ticker_names


def hedge_ratio(log_target: pd.Series, log_pair: pd.Series) -> tuple[float, float] | None:
    """OLS log_target ~ alpha + beta * log_pair. Returns (beta, alpha), or None.

    Returns None if log_pair is (near-)constant over the window - a frozen
    or halted stock - which makes the design matrix singular and leaves
    statsmodels with no slope coefficient to return.
    """
    X = sm.add_constant(log_pair)
    if X.shape[1] < 2:
        return None
    model = sm.OLS(log_target, X).fit()
    if len(model.params) < 2:
        return None
    return float(model.params.iloc[1]), float(model.params.iloc[0])


def walk_forward_beta(
    log_t: pd.Series, log_p: pd.Series, beta_window: int = 120, reestimate_every: int = 20
) -> pd.Series:
    """Piecewise-constant beta, re-estimated periodically from trailing data only.

    beta.iloc[i] uses only log_t/log_p.iloc[i-beta_window:i] - strictly
    before i - so it's information a trader would actually have had on day i.
    NaN until enough history has accumulated (no trading before that).
    """
    betas = pd.Series(index=log_t.index, dtype=float)
    last_beta = float("nan")
    for i in range(len(log_t)):
        if i >= beta_window and i % reestimate_every == 0:
            result = hedge_ratio(log_t.iloc[i - beta_window : i], log_p.iloc[i - beta_window : i])
            if result is not None:
                last_beta, _ = result
            # else: pair leg was flat (halted/frozen) over this window - keep the last valid beta
        betas.iloc[i] = last_beta
    return betas


def _positions_from_zscore(z: pd.Series, entry: float, exit_z: float) -> pd.Series:
    pos, positions = 0, []
    for zi in z:
        if pd.isna(zi):
            positions.append(0)
            continue
        if pos == 0:
            if zi > entry:
                pos = -1  # spread too high: short target, long pair
            elif zi < -entry:
                pos = 1  # spread too low: long target, short pair
        elif abs(zi) < exit_z:
            pos = 0
        positions.append(pos)
    return pd.Series(positions, index=z.index).shift(1).fillna(0)  # trade next bar, not signal bar


def _summarize(position: pd.Series, strategy_ret: pd.Series, beta_series: pd.Series) -> tuple[pd.DataFrame, dict]:
    # strategy_ret is built from log-price differences (spread_ret = log_t.diff() - beta*log_p.diff()),
    # i.e. it's already a log return, not a simple one. Compounding it with (1+r).cumprod() silently
    # treats it as simple returns - fine for small moves, but distorts equity/drawdown once |r| gets
    # large (exactly the regime this backtest is meant to stress-test). exp(cumsum()) compounds log
    # returns correctly.
    equity = np.exp(strategy_ret.cumsum())
    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    n_trades = int((position.diff().fillna(0) != 0).sum())
    ann_factor = 252
    sharpe = (
        strategy_ret.mean() / strategy_ret.std() * np.sqrt(ann_factor)
        if strategy_ret.std() > 0
        else float("nan")
    )

    stats = {
        "beta_mean": float(beta_series.mean()),
        "beta_last": float(beta_series.iloc[-1]) if len(beta_series) else float("nan"),
        "total_return": equity.iloc[-1] - 1 if len(equity) else float("nan"),
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "n_trades": n_trades,
        "n_days": len(strategy_ret),
    }
    df = pd.DataFrame({
        "beta": beta_series, "z": None, "position": position,
        "strategy_ret": strategy_ret, "equity": equity, "drawdown": drawdown,
    })
    return df, stats


def run_backtest_static(
    target: pd.Series, pair: pd.Series, window: int = 60, entry: float = 2.0, exit_z: float = 0.5
) -> tuple[pd.DataFrame, dict]:
    log_t, log_p = np.log(target), np.log(pair)
    beta, _ = hedge_ratio(log_t, log_p)
    beta_series = pd.Series(beta, index=log_t.index)
    spread = log_t - beta * log_p

    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    position = _positions_from_zscore(z, entry, exit_z)
    spread_ret = log_t.diff() - beta * log_p.diff()
    strategy_ret = (position * spread_ret).fillna(0)

    df, stats = _summarize(position, strategy_ret, beta_series)
    df["z"] = z
    return df, stats


def run_backtest_walkforward(
    target: pd.Series, pair: pd.Series, beta_window: int = 120, reestimate_every: int = 20,
    z_window: int = 60, entry: float = 2.0, exit_z: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    log_t, log_p = np.log(target), np.log(pair)
    beta_series = walk_forward_beta(log_t, log_p, beta_window, reestimate_every)
    spread = log_t - beta_series * log_p

    z = (spread - spread.rolling(z_window).mean()) / spread.rolling(z_window).std()
    position = _positions_from_zscore(z, entry, exit_z)
    spread_ret = log_t.diff() - beta_series * log_p.diff()  # beta[i] only ever used data before i
    strategy_ret = (position * spread_ret).fillna(0)

    df, stats = _summarize(position, strategy_ret, beta_series)
    df["z"] = z
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="005930")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--method", choices=["static", "walkforward"], default="walkforward")
    parser.add_argument("--window", type=int, default=60, help="[static] rolling window for the spread z-score")
    parser.add_argument("--beta-window", type=int, default=120, help="[walkforward] trailing days used per beta estimate")
    parser.add_argument("--reestimate-every", type=int, default=20, help="[walkforward] days between beta re-estimates")
    parser.add_argument("--z-window", type=int, default=60, help="[walkforward] rolling window for the spread z-score")
    parser.add_argument("--entry", type=float, default=2.0, help="|z| to enter a position")
    parser.add_argument("--exit-z", type=float, default=0.5, help="|z| to flatten a position")
    parser.add_argument("--out", default=None, help="Optional CSV path for the day-by-day series")
    args = parser.parse_args()

    conn = get_connection()
    prices = load_prices(conn)
    if args.target not in prices.columns or args.pair not in prices.columns:
        raise SystemExit("Target or pair ticker not found in stored prices.")

    names = ticker_names(conn)
    both = prices[[args.target, args.pair]].dropna()

    if args.method == "static":
        df, stats = run_backtest_static(both[args.target], both[args.pair], args.window, args.entry, args.exit_z)
    else:
        df, stats = run_backtest_walkforward(
            both[args.target], both[args.pair], args.beta_window, args.reestimate_every,
            args.z_window, args.entry, args.exit_z,
        )

    t_name = names.get(args.target, args.target)
    p_name = names.get(args.pair, args.pair)
    print(f"{t_name} ({args.target}) vs {p_name} ({args.pair}) - {stats['n_days']} days - method={args.method}")
    print(f"  beta (mean / last):  {stats['beta_mean']:.4f} / {stats['beta_last']:.4f}")
    print(f"  total return:        {stats['total_return'] * 100:+.2f}%")
    print(f"  Sharpe (annualized): {stats['sharpe']:.2f}")
    print(f"  max drawdown:        {stats['max_drawdown'] * 100:.2f}%")
    print(f"  trades:              {stats['n_trades']}")

    if args.out:
        df.to_csv(args.out, encoding="utf-8-sig")
        print(f"\nDay-by-day series written to {args.out}")


if __name__ == "__main__":
    main()
