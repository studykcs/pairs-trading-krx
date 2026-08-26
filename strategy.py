"""The full pairs-trading spread strategy: walk-forward beta + GMM regime
filter + transaction costs + a hard stop-loss.

backtest.py and gmm_strategy.py answered "does the idea work at all" under
frictionless, unlimited-risk assumptions. This module adds the two things a
real implementation can't skip:

  - transaction costs: a round-trip cost in bps, charged on every position
    change (entry, exit, forced flatten, or stop-out), proportional to the
    size of the position change. Korean equities also carry a sell-side
    securities transaction tax (~0.15% on KOSPI as of recent years) on top
    of brokerage commission - `--cost-bps` should reflect both, not just
    commission.

  - hard stop-loss: force-flattens a position if its cumulative return
    since entry drops below `--stop-loss`, independent of the z-score or
    GMM regime signal. This is a backstop for exactly the kind of runaway
    divergence GMM mostly, but not entirely, screens out (see the SK Hynix
    / HD Hyundai Energy Solutions results: GMM cut the loss roughly in
    half over the full 1,658-day sample, but didn't prevent one).

Usage
-----
    python strategy.py --target 000660 --pair 322000
    python strategy.py --target 000660 --pair 322000 --cost-bps 20 --stop-loss 0.10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest import _summarize, walk_forward_beta
from costs import CostModel, pair_cost_series
from gmm_strategy import regime_labels
from store import get_connection, load_field, load_prices, ticker_names


def run_strategy(
    target: pd.Series, pair: pd.Series, beta_window: int = 120, reestimate_every: int = 20,
    z_window: int = 60, gmm_window: int = 250, entry: float = 2.0, exit_z: float = 0.5,
    cost_bps: float = 15.0, stop_loss: float = 0.15, use_gmm: bool = True,
    cost_model: CostModel | None = None, capital: float = 100_000_000,
    target_adv: pd.Series | None = None, pair_adv: pd.Series | None = None,
    allow_missing_adv: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """When `cost_model` is given, costs come from costs.py (commission,
    sell-side tax, half-spread, ADV-scaled impact, and daily borrow on the
    short leg) and `cost_bps` is ignored. Without it, the original flat
    `cost_bps` per unit of turnover is used - kept so the two can be
    compared directly rather than the old number just disappearing."""
    log_t, log_p = np.log(target), np.log(pair)
    beta_series = walk_forward_beta(log_t, log_p, beta_window, reestimate_every)
    spread = log_t - beta_series * log_p

    z = (spread - spread.rolling(z_window).mean()) / spread.rolling(z_window).std()
    if use_gmm:
        spread_vol = spread.diff().rolling(z_window).std()
        regime = regime_labels(z, spread_vol, gmm_window, reestimate_every)
    else:
        # No regime filter: every day with a valid z-score counts as "calm".
        regime = pd.Series(0, index=z.index).where(z.notna(), other=float("nan"))
    spread_ret = log_t.diff() - beta_series * log_p.diff()

    pos, entry_cum, positions, stop_outs = 0, 0.0, [], 0
    for i in range(len(z)):
        zi, regime_i, ret_i = z.iloc[i], regime.iloc[i], spread_ret.iloc[i]

        if pos != 0:
            # `pos` is still the value decided on the PREVIOUS iteration - today's
            # entry/exit branch below hasn't run yet - so this is positions[i-1] *
            # spread_ret[i], the same product `gross_ret` computes further down via
            # position.shift(1) * spread_ret. entry_cum is therefore already the
            # running sum of realized (shift-consistent) daily P&L, not a preview of
            # today's not-yet-decided position. Do not lag ret_i again here - that
            # would pair today's position with yesterday's return instead of today's.
            entry_cum += pos * ret_i  # cumulative log-return of the open trade so far
            if entry_cum < -abs(stop_loss):
                pos = 0
                stop_outs += 1
            elif regime_i == 1:
                pos = 0  # GMM regime flip
            elif pd.notna(zi) and abs(zi) < exit_z:
                pos = 0  # normal signal exit

        if pos == 0 and regime_i == 0 and pd.notna(zi):
            if zi > entry:
                pos, entry_cum = -1, 0.0
            elif zi < -entry:
                pos, entry_cum = 1, 0.0

        positions.append(pos)

    position = pd.Series(positions, index=z.index).shift(1).fillna(0)
    gross_ret = (position * spread_ret).fillna(0)

    breakdown = None
    if cost_model is None:
        turnover = position.diff().abs().fillna(0)  # 0->1 or 1->-1 etc; each unit = one leg's worth of trading
        cost = turnover * (cost_bps / 10000)
    else:
        breakdown = pair_cost_series(
            position=position, beta=beta_series,
            target_ret=target.pct_change(fill_method=None),
            pair_ret=pair.pct_change(fill_method=None),
            target_adv=target_adv, pair_adv=pair_adv,
            capital=capital, model=cost_model,
            allow_missing_adv=allow_missing_adv,
        )
        cost = breakdown["total"]

    net_ret = gross_ret - cost

    df, stats = _summarize(position, net_ret, beta_series)
    df["z"], df["regime"], df["gross_ret"], df["cost"] = z, regime, gross_ret, cost
    if breakdown is not None:
        for component in ("commission", "tax", "half_spread", "impact", "borrow"):
            df[f"cost_{component}"] = breakdown[component]
            stats[f"cost_{component}"] = float(breakdown[component].sum())
    stats["gross_total_return"] = float(np.exp(gross_ret.cumsum()).iloc[-1] - 1)  # match _summarize's log-return compounding
    stats["total_cost"] = float(cost.sum())
    stats["stop_outs"] = stop_outs
    stats["pct_calm_days"] = float((regime == 0).mean()) if regime.notna().any() else float("nan")
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="005930")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--beta-window", type=int, default=120)
    parser.add_argument("--reestimate-every", type=int, default=20)
    parser.add_argument("--z-window", type=int, default=60)
    parser.add_argument("--gmm-window", type=int, default=250)
    parser.add_argument("--entry", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--cost-bps", type=float, default=15.0, help="Round-trip cost per leg, in bps (commission + tax)")
    parser.add_argument("--stop-loss", type=float, default=0.15, help="Force-flatten if a trade's cumulative log-loss exceeds this")
    parser.add_argument("--no-gmm", action="store_true", help="Skip the GMM regime filter - useful for pairs stable enough not to need it")
    parser.add_argument("--out", default=None)

    g = parser.add_argument_group("realistic cost model (costs.py)")
    g.add_argument("--realistic-costs", action="store_true",
                   help="Use costs.py instead of the flat --cost-bps: sell-side tax, "
                        "borrow fee on the short leg, and ADV-scaled market impact")
    g.add_argument("--capital", type=float, default=100_000_000,
                   help="Position size in KRW - impact scales with order size / ADV, so this matters (default: 1억)")
    g.add_argument("--commission-bps", type=float, default=CostModel.commission_bps)
    g.add_argument("--tax-bps", type=float, default=CostModel.tax_bps, help="Sell side only")
    g.add_argument("--half-spread-bps", type=float, default=CostModel.half_spread_bps)
    g.add_argument("--borrow-bps", type=float, default=CostModel.borrow_fee_annual_bps,
                   help="Annualized 대차수수료 on the short leg")
    g.add_argument("--impact-coef", type=float, default=CostModel.impact_coef,
                   help="Square-root impact coefficient (assumption, not calibrated - see costs.py)")
    g.add_argument("--allow-missing-adv", action="store_true",
                   help="Price impact at zero where ADV is missing instead of failing (understates cost)")
    args = parser.parse_args()

    conn = get_connection()
    prices = load_prices(conn)
    if args.target not in prices.columns or args.pair not in prices.columns:
        raise SystemExit("Target or pair ticker not found in stored prices.")

    names = ticker_names(conn)
    both = prices[[args.target, args.pair]].dropna()
    t_name, p_name = names.get(args.target, args.target), names.get(args.pair, args.pair)

    cost_model, target_adv, pair_adv = None, None, None
    if args.realistic_costs:
        cost_model = CostModel(
            commission_bps=args.commission_bps, tax_bps=args.tax_bps,
            half_spread_bps=args.half_spread_bps,
            borrow_fee_annual_bps=args.borrow_bps, impact_coef=args.impact_coef,
        )
        trdval = load_field(conn, "trdval")
        target_adv = trdval[args.target] if args.target in trdval.columns else None
        pair_adv = trdval[args.pair] if args.pair in trdval.columns else None

    df, stats = run_strategy(
        both[args.target], both[args.pair], args.beta_window, args.reestimate_every, args.z_window,
        args.gmm_window, args.entry, args.exit_z, args.cost_bps, args.stop_loss, use_gmm=not args.no_gmm,
        cost_model=cost_model, capital=args.capital,
        target_adv=target_adv, pair_adv=pair_adv, allow_missing_adv=args.allow_missing_adv,
    )

    print(f"{t_name} ({args.target}) vs {p_name} ({args.pair}) - {stats['n_days']} days")
    if cost_model is None:
        print(f"  cost assumption:     {args.cost_bps:.1f} bps/leg (flat), stop-loss {args.stop_loss*100:.0f}%, gmm={'off' if args.no_gmm else 'on'}\n")
    else:
        print(f"  cost model:          realistic (costs.py), capital {args.capital:,.0f} KRW")
        print(f"                       commission {cost_model.commission_bps}bp, tax {cost_model.tax_bps}bp (sell), "
              f"half-spread {cost_model.half_spread_bps}bp,")
        print(f"                       borrow {cost_model.borrow_fee_annual_bps}bp/yr (short leg), impact coef {cost_model.impact_coef}")
        print(f"  stop-loss {args.stop_loss*100:.0f}%, gmm={'off' if args.no_gmm else 'on'}\n")
    print(f"  total return (net):  {stats['total_return']*100:+.2f}%")
    print(f"  total return (gross):{stats['gross_total_return']*100:+.2f}%")
    print(f"  total cost drag:     {stats['total_cost']*100:.2f}%")
    if cost_model is not None:
        for label, key in [("commission", "cost_commission"), ("tax (sell)", "cost_tax"),
                           ("half-spread", "cost_half_spread"), ("impact (ADV)", "cost_impact"),
                           ("borrow (short)", "cost_borrow")]:
            print(f"    {label:<18s}{stats[key]*100:>6.2f}%")
    print(f"  Sharpe (annualized): {stats['sharpe']:.2f}")
    print(f"  max drawdown:        {stats['max_drawdown']*100:.2f}%")
    print(f"  trades:              {stats['n_trades']}")
    print(f"  stop-loss exits:     {stats['stop_outs']}")
    print(f"  % days classified calm: {stats['pct_calm_days']*100:.1f}%")

    if args.out:
        df.to_csv(args.out, encoding="utf-8-sig")
        print(f"\nDay-by-day series written to {args.out}")


if __name__ == "__main__":
    main()
