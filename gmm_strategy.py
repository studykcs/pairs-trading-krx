"""GMM regime-filtered pairs trading strategy.

Layers a regime filter on top of the walk-forward z-score strategy in
backtest.py. At each re-estimation point, fits a 2-component Gaussian
Mixture on the trailing window's spread behavior - the rolling z-score and
the rolling volatility of the spread's daily change - using only data
available up to that point (no look-ahead, same discipline as beta's
walk-forward re-estimation). One component ends up as the "calm" regime
(small, mean-reverting moves); the other is "turbulent" - the kind of
runaway divergence that produced the -98% drawdown in the plain walk-forward
SK Hynix / HD Hyundai Energy Solutions backtest.

New entries are only taken while the current day classifies as calm. An
open position is force-flattened the day the regime flips to turbulent,
rather than waiting for the z-score to (maybe never) revert on its own.

This is a filter, not a replacement: cointegration screening still decides
*which* pair to trade; GMM decides *when* the pair's current behavior still
looks like the historical relationship the test found, versus a breakdown
of it. The price for that is discovery, not more precision - it can only
reduce time-in-market, not fix a fundamentally bad pair.

Usage
-----
    python gmm_strategy.py --target 000660 --pair 322000
    python gmm_strategy.py --target 000660 --pair 322000 --gmm-window 250
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from backtest import _summarize, hedge_ratio, run_backtest_walkforward, walk_forward_beta
from store import get_connection, load_prices, period_label, ticker_names, warn_thin_warmup


def regime_labels(
    z: pd.Series, spread_vol: pd.Series, gmm_window: int = 250, reestimate_every: int = 20,
) -> pd.Series:
    """Walk-forward regime classification: 0 = calm, 1 = turbulent.

    Refits only every `reestimate_every` days on the trailing `gmm_window`
    days of [z, spread_vol] - strictly before the current day - then labels
    each day with the most recently fitted model. NaN until enough history
    (`gmm_window` days) has accumulated.
    """
    features = pd.DataFrame({"z": z, "vol": spread_vol})
    labels = pd.Series(index=z.index, dtype="float64")
    model, calm_id = None, None

    for i in range(len(z)):
        if i >= gmm_window and i % reestimate_every == 0:
            train = features.iloc[i - gmm_window : i].dropna()
            if len(train) >= gmm_window // 2:
                model = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(train.values)
                # "calm" = whichever component has the lower mean spread volatility (feature index 1).
                # Sorting on z's mean (feature index 0) instead is unstable: both components' z-means
                # can sit close to 0 depending on the window's sign balance, so argmin(|mean_z|) can
                # flip which component is "calm" between refits (EM label switching) even when the
                # underlying calm/turbulent split hasn't changed. Volatility is unsigned and separates
                # the two regimes far more consistently across refits.
                calm_id = int(np.argmin(model.means_[:, 1]))
        if model is not None:
            row = features.iloc[i]
            if row.notna().all():
                pred = model.predict(row.values.reshape(1, -1))[0]
                labels.iloc[i] = 0 if pred == calm_id else 1

    return labels


def run_gmm_backtest(
    target: pd.Series, pair: pd.Series, beta_window: int = 120, reestimate_every: int = 20,
    z_window: int = 60, gmm_window: int = 250, entry: float = 2.0, exit_z: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    log_t, log_p = np.log(target), np.log(pair)
    beta_series = walk_forward_beta(log_t, log_p, beta_window, reestimate_every)
    spread = log_t - beta_series * log_p

    z = (spread - spread.rolling(z_window).mean()) / spread.rolling(z_window).std()
    spread_vol = spread.diff().rolling(z_window).std()
    regime = regime_labels(z, spread_vol, gmm_window, reestimate_every)

    pos, positions = 0, []
    for i in range(len(z)):
        zi, regime_i = z.iloc[i], regime.iloc[i]
        if pos != 0 and regime_i == 1:
            pos = 0  # force-flatten the moment the regime turns turbulent
        elif pos == 0 and regime_i == 0 and pd.notna(zi):
            if zi > entry:
                pos = -1
            elif zi < -entry:
                pos = 1
        elif pos != 0 and pd.notna(zi) and abs(zi) < exit_z:
            pos = 0
        positions.append(pos)

    position = pd.Series(positions, index=z.index).shift(1).fillna(0)
    spread_ret = log_t.diff() - beta_series * log_p.diff()
    strategy_ret = (position * spread_ret).fillna(0)

    df, stats = _summarize(position, strategy_ret, beta_series)
    df["z"], df["regime"] = z, regime
    stats["pct_calm_days"] = float((regime == 0).mean()) if regime.notna().any() else float("nan")
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="005930")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--beta-window", type=int, default=120)
    parser.add_argument("--reestimate-every", type=int, default=20)
    parser.add_argument("--z-window", type=int, default=60)
    parser.add_argument("--gmm-window", type=int, default=250, help="Trailing days used to fit each GMM refit")
    parser.add_argument("--entry", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD, default: full stored history")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD, default: full stored history")
    parser.add_argument("--out", default=None, help="Optional CSV path for the day-by-day series")
    args = parser.parse_args()

    conn = get_connection()
    prices = load_prices(conn, start=args.start, end=args.end)
    if args.target not in prices.columns or args.pair not in prices.columns:
        raise SystemExit("Target or pair ticker not found in stored prices.")

    names = ticker_names(conn)
    both = prices[[args.target, args.pair]].dropna()
    print(f"기간: {period_label(both.index)}")
    warn_thin_warmup(len(both), args.gmm_window)
    t_name, p_name = names.get(args.target, args.target), names.get(args.pair, args.pair)

    base_df, base_stats = run_backtest_walkforward(
        both[args.target], both[args.pair], args.beta_window, args.reestimate_every, args.z_window, args.entry, args.exit_z
    )
    gmm_df, gmm_stats = run_gmm_backtest(
        both[args.target], both[args.pair], args.beta_window, args.reestimate_every,
        args.z_window, args.gmm_window, args.entry, args.exit_z,
    )

    print(f"{t_name} ({args.target}) vs {p_name} ({args.pair}) - {base_stats['n_days']} days\n")
    print(f"{'':22s}{'walk-forward':>16s}{'+ GMM filter':>16s}")
    print(f"{'total return':22s}{base_stats['total_return']*100:>15.2f}%{gmm_stats['total_return']*100:>15.2f}%")
    print(f"{'Sharpe (annualized)':22s}{base_stats['sharpe']:>16.2f}{gmm_stats['sharpe']:>16.2f}")
    print(f"{'max drawdown':22s}{base_stats['max_drawdown']*100:>15.2f}%{gmm_stats['max_drawdown']*100:>15.2f}%")
    print(f"{'trades':22s}{base_stats['n_trades']:>16d}{gmm_stats['n_trades']:>16d}")
    print(f"{'% days classified calm':22s}{'':>16s}{gmm_stats['pct_calm_days']*100:>15.1f}%")

    if args.out:
        gmm_df.to_csv(args.out, encoding="utf-8-sig")
        print(f"\nDay-by-day GMM series written to {args.out}")


if __name__ == "__main__":
    main()
