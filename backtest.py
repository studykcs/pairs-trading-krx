"""Backtest a z-score pairs-trading strategy on one cointegrated pair.

The hedge ratio (beta) is estimated once via OLS on log prices over the
whole sample - a simple choice that has look-ahead bias (later data informs
early trades). A walk-forward re-estimation would remove that but adds
complexity; treat these results as an upper-bound sanity check, not a
production backtest.

Strategy: rolling z-score of the spread (log_target - beta * log_pair).
Enter when |z| crosses `--entry`, flatten when it falls back inside
`--exit-z`. Position is lagged one day (trade on the next bar's return,
not the bar that generated the signal).

Usage
-----
    python backtest.py --target 005930 --pair 000660
    python backtest.py --target 005930 --pair 000660 --window 60 --entry 2.0 --exit-z 0.5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from store import get_connection, load_prices, ticker_names


def hedge_ratio(log_target: pd.Series, log_pair: pd.Series) -> tuple[float, float]:
    """OLS log_target ~ alpha + beta * log_pair. Returns (beta, alpha)."""
    X = sm.add_constant(log_pair)
    model = sm.OLS(log_target, X).fit()
    return float(model.params.iloc[1]), float(model.params.iloc[0])


def run_backtest(
    target: pd.Series, pair: pd.Series, window: int = 60, entry: float = 2.0, exit_z: float = 0.5
) -> tuple[pd.DataFrame, dict]:
    log_t = np.log(target)
    log_p = np.log(pair)
    beta, alpha = hedge_ratio(log_t, log_p)
    spread = log_t - beta * log_p

    roll_mean = spread.rolling(window).mean()
    roll_std = spread.rolling(window).std()
    z = (spread - roll_mean) / roll_std

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

    position = pd.Series(positions, index=z.index).shift(1).fillna(0)  # trade next bar, not signal bar
    spread_ret = log_t.diff() - beta * log_p.diff()
    strategy_ret = (position * spread_ret).fillna(0)

    equity = (1 + strategy_ret).cumprod()
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
        "beta": beta,
        "alpha": alpha,
        "total_return": equity.iloc[-1] - 1 if len(equity) else float("nan"),
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "n_trades": n_trades,
        "n_days": len(strategy_ret),
    }

    df = pd.DataFrame({
        "z": z, "position": position, "strategy_ret": strategy_ret,
        "equity": equity, "drawdown": drawdown,
    })
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="005930")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--window", type=int, default=60, help="Rolling window for the spread z-score")
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

    df, stats = run_backtest(both[args.target], both[args.pair], args.window, args.entry, args.exit_z)

    t_name = names.get(args.target, args.target)
    p_name = names.get(args.pair, args.pair)
    print(f"{t_name} ({args.target}) vs {p_name} ({args.pair}) - {stats['n_days']} days")
    print(f"  hedge ratio (beta): {stats['beta']:.4f}")
    print(f"  total return:       {stats['total_return'] * 100:+.2f}%")
    print(f"  Sharpe (annualized): {stats['sharpe']:.2f}")
    print(f"  max drawdown:       {stats['max_drawdown'] * 100:.2f}%")
    print(f"  trades:             {stats['n_trades']}")

    if args.out:
        df.to_csv(args.out, encoding="utf-8-sig")
        print(f"\nDay-by-day series written to {args.out}")


if __name__ == "__main__":
    main()
