"""HMM regime-filtered pairs trading - the persistence-aware version of
gmm_strategy.py.

Same idea as the GMM filter: classify each day's spread behavior as calm or
turbulent from [z-score, spread volatility], only open positions on calm
days, and force-flatten when the regime turns. The difference is what the
model knows about *time*.

A Gaussian mixture treats every day as an independent draw, so it has no
notion that regimes persist. On overlapping regimes that produces constant
label flicker - in a controlled test against a known 2-regime process with
43 true regime switches, the mixture produced 481 label switches (79.6%
accuracy) while the HMM produced 39-79 (93-96% accuracy). Each spurious
flip is a forced flatten and re-entry, which under the cost model in
costs.py is not free.

An HMM's transition matrix prices that persistence directly: P(stay) is
estimated from the data rather than assumed away.

Look-ahead discipline
---------------------
Two separate guards, both necessary:

  1. **Walk-forward fitting** - parameters are re-estimated only every
     `reestimate_every` days on the trailing `hmm_window` days strictly
     *before* the current day, exactly as gmm_strategy.py and
     walk_forward_beta do.

  2. **Filtered, not smoothed, decoding** - the day's label comes from
     `filter_proba` (forward pass only, conditions on the past alone). This
     is the trap in most HMM backtests: the natural library call
     (`predict()`, Viterbi over the whole sequence) labels each day using
     the entire series including the future, which quietly turns the regime
     filter into an oracle. See hmm.py's docstrings.

`--min-calm-proba` is the one genuinely new control the HMM affords: rather
than taking the argmax state, require P(calm) to clear a threshold before
entering. Higher = fewer, more confident entries.

Usage
-----
    python hmm_strategy.py --target 000660 --pair 322000
    python hmm_strategy.py --target 000660 --pair 322000 --min-calm-proba 0.8
    python hmm_strategy.py --target 000660 --pair 322000 --compare
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest import _summarize, walk_forward_beta
from hmm import GaussianHMM
from store import get_connection, load_prices, period_label, ticker_names, warn_thin_warmup


def regime_proba(
    z: pd.Series,
    spread_vol: pd.Series,
    hmm_window: int = 250,
    reestimate_every: int = 20,
    n_states: int = 2,
) -> pd.Series:
    """Walk-forward P(calm) per day, using causal (filtered) decoding.

    NaN until `hmm_window` days of history exist. "Calm" is identified as
    the state with the lowest mean spread volatility (feature index 1) -
    unsigned and stable across refits, unlike sorting on the z-mean, which
    suffers the same EM label-switching problem documented in
    gmm_strategy.py.
    """
    features = pd.DataFrame({"z": z, "vol": spread_vol})
    proba = pd.Series(index=z.index, dtype="float64")
    model: GaussianHMM | None = None
    calm_id: int | None = None

    for i in range(len(z)):
        if i >= hmm_window and i % reestimate_every == 0:
            train = features.iloc[i - hmm_window : i].dropna()
            if len(train) >= hmm_window // 2:
                try:
                    candidate = GaussianHMM(
                        n_states=n_states, n_init=4, random_state=0
                    ).fit(train.values)
                except RuntimeError:
                    candidate = None  # keep the previous model rather than going blind
                if candidate is not None:
                    model = candidate
                    calm_id = int(np.argmin(model.means[:, 1]))

        if model is None:
            continue

        # Filter over the trailing window ending at *today*. Re-running the
        # forward pass on the window (rather than carrying one alpha vector
        # forward) keeps this correct across refits, where the parameters -
        # and therefore the meaning of the state indices - change.
        hist = features.iloc[max(0, i - hmm_window + 1) : i + 1].dropna()
        if len(hist) < 2 or features.iloc[i].isna().any():
            continue
        try:
            f = model.filter_proba(hist.values)
        except (ValueError, FloatingPointError):
            continue
        proba.iloc[i] = float(f[-1, calm_id])

    return proba


def run_hmm_strategy(
    target: pd.Series,
    pair: pd.Series,
    beta_window: int = 120,
    reestimate_every: int = 20,
    z_window: int = 60,
    hmm_window: int = 250,
    entry: float = 2.0,
    exit_z: float = 0.5,
    min_calm_proba: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    log_t, log_p = np.log(target), np.log(pair)
    beta_series = walk_forward_beta(log_t, log_p, beta_window, reestimate_every)
    spread = log_t - beta_series * log_p

    z = (spread - spread.rolling(z_window).mean()) / spread.rolling(z_window).std()
    spread_vol = spread.diff().rolling(z_window).std()
    p_calm = regime_proba(z, spread_vol, hmm_window, reestimate_every)
    regime = (p_calm < min_calm_proba).astype("float64").where(p_calm.notna())

    spread_ret = log_t.diff() - beta_series * log_p.diff()

    pos, positions = 0, []
    for i in range(len(z)):
        zi, regime_i = z.iloc[i], regime.iloc[i]
        if pos != 0 and regime_i == 1:
            pos = 0
        elif pos == 0 and regime_i == 0 and pd.notna(zi):
            if zi > entry:
                pos = -1
            elif zi < -entry:
                pos = 1
        elif pos != 0 and pd.notna(zi) and abs(zi) < exit_z:
            pos = 0
        positions.append(pos)

    position = pd.Series(positions, index=z.index).shift(1).fillna(0)
    strategy_ret = (position * spread_ret).fillna(0)

    df, stats = _summarize(position, strategy_ret, beta_series)
    df["z"], df["regime"], df["p_calm"] = z, regime, p_calm
    stats["pct_calm_days"] = float((regime == 0).mean()) if regime.notna().any() else float("nan")
    stats["n_regime_switches"] = count_regime_switches(regime)
    return df, stats


def count_regime_switches(regime: pd.Series) -> int:
    """Label changes between consecutive *observed* days.

    Counting `(regime.diff() != 0).sum()` instead would be wrong twice
    over: NaN != 0 is True, so every leading warm-up day and every gap
    counts as a switch, inflating the number by hundreds on a series that
    is NaN for its first `hmm_window` days.
    """
    observed = regime.dropna()
    if len(observed) < 2:
        return 0
    return int((observed.values[1:] != observed.values[:-1]).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="005930")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--beta-window", type=int, default=120)
    parser.add_argument("--reestimate-every", type=int, default=20)
    parser.add_argument("--z-window", type=int, default=60)
    parser.add_argument("--hmm-window", type=int, default=250, help="Trailing days used for each HMM refit")
    parser.add_argument("--entry", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument(
        "--min-calm-proba", type=float, default=0.5,
        help="Require P(calm) above this to hold/enter; higher = fewer, more confident entries",
    )
    parser.add_argument("--compare", action="store_true", help="Also run the GMM filter side by side")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD, default: full stored history")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD, default: full stored history")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    conn = get_connection()
    prices = load_prices(conn, start=args.start, end=args.end)
    if args.target not in prices.columns or args.pair not in prices.columns:
        raise SystemExit("Target or pair ticker not found in stored prices.")

    names = ticker_names(conn)
    both = prices[[args.target, args.pair]].dropna()
    print(f"기간: {period_label(both.index)}")
    warn_thin_warmup(len(both), args.hmm_window)
    t_name = names.get(args.target, args.target)
    p_name = names.get(args.pair, args.pair)

    df, stats = run_hmm_strategy(
        both[args.target], both[args.pair], args.beta_window, args.reestimate_every,
        args.z_window, args.hmm_window, args.entry, args.exit_z, args.min_calm_proba,
    )

    print(f"{t_name} ({args.target}) vs {p_name} ({args.pair}) - {stats['n_days']} days")
    print(f"  HMM filter, min P(calm) = {args.min_calm_proba}\n")

    if not args.compare:
        print(f"  total return:        {stats['total_return']*100:+.2f}%")
        print(f"  Sharpe (annualized): {stats['sharpe']:.2f}")
        print(f"  max drawdown:        {stats['max_drawdown']*100:.2f}%")
        print(f"  trades:              {stats['n_trades']}")
        print(f"  % days calm:         {stats['pct_calm_days']*100:.1f}%")
        print(f"  regime switches:     {stats['n_regime_switches']}")
    else:
        from gmm_strategy import run_gmm_backtest

        gdf, gstats = run_gmm_backtest(
            both[args.target], both[args.pair], args.beta_window, args.reestimate_every,
            args.z_window, args.hmm_window, args.entry, args.exit_z,
        )
        g_switches = count_regime_switches(gdf["regime"])
        print(f"{'':24s}{'HMM':>14s}{'GMM':>14s}")
        print(f"  {'total return':22s}{stats['total_return']*100:>13.2f}%{gstats['total_return']*100:>13.2f}%")
        print(f"  {'Sharpe':22s}{stats['sharpe']:>14.2f}{gstats['sharpe']:>14.2f}")
        print(f"  {'max drawdown':22s}{stats['max_drawdown']*100:>13.2f}%{gstats['max_drawdown']*100:>13.2f}%")
        print(f"  {'trades':22s}{stats['n_trades']:>14d}{gstats['n_trades']:>14d}")
        print(f"  {'% days calm':22s}{stats['pct_calm_days']*100:>13.1f}%{gstats['pct_calm_days']*100:>13.1f}%")
        print(f"  {'regime switches':22s}{stats['n_regime_switches']:>14d}{g_switches:>14d}")

    if args.out:
        df.to_csv(args.out, encoding="utf-8-sig")
        print(f"\nDay-by-day series written to {args.out}")


if __name__ == "__main__":
    main()
