"""Flat vs realistic transaction-cost comparison, on one shared signal path.

The numbers in README's "what the flat model was hiding" table used to come
from a one-off script run once and never kept. That means the conditions
behind them (pair, capital, date range, code version) were not reproducible.
This script is that comparison, made rerunnable: it runs `strategy.py`'s
signal generation exactly once and prices the resulting position path under
both cost models, so the two columns differ only in the cost model, never in
the trades taken.

Usage
-----
    python compare_costs.py
    python compare_costs.py --target 000660 --pair 322000 --capital 100000000
    python compare_costs.py --start 2020-01-01 --end 2024-12-31 --out costs.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import fields

import numpy as np
import pandas as pd

from costs import CostModel
from store import get_connection, load_field, load_prices, ticker_names
from strategy import run_strategy

COMPONENTS = ["commission", "tax", "half_spread", "impact", "borrow"]
IMPACT_SWEEP_COEFS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return commit + (" (dirty working tree)" if dirty else "")
    except Exception:
        return "unknown (not a git repo or git unavailable)"


def _holding_stats(position: pd.Series) -> tuple[int, float]:
    """Number of held (non-flat) episodes and their mean length in days."""
    lengths, run_len = [], 0
    for p in position.to_numpy():
        if p != 0:
            run_len += 1
        elif run_len > 0:
            lengths.append(run_len)
            run_len = 0
    if run_len > 0:
        lengths.append(run_len)
    return len(lengths), (float(np.mean(lengths)) if lengths else float("nan"))


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="000660")
    parser.add_argument("--pair", default="322000")
    parser.add_argument("--capital", type=float, default=100_000_000)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD, default: earliest overlapping day")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD, default: latest overlapping day")
    parser.add_argument("--cost-bps", type=float, default=15.0, help="Flat model rate, per-leg bps of turnover")
    parser.add_argument("--out", default=None, help="Optional CSV path for the comparison table")
    parser.add_argument(
        "--impact-sweep", action="store_true",
        help="Also show total cost and x-flat multiple across impact_coef in "
             f"{IMPACT_SWEEP_COEFS} - impact_coef is a literature assumption, not "
             "calibrated (no execution data exists to fit it against), so this shows "
             "how much the 'flat model understates cost by Nx' conclusion moves with it",
    )
    args = parser.parse_args()

    conn = get_connection()
    prices = load_prices(conn)
    if args.target not in prices.columns or args.pair not in prices.columns:
        raise SystemExit("Target or pair ticker not found in stored prices.")

    names = ticker_names(conn)
    both = prices[[args.target, args.pair]].dropna()
    if args.start:
        both = both.loc[both.index >= args.start]
    if args.end:
        both = both.loc[both.index <= args.end]
    if both.empty:
        raise SystemExit("No overlapping price data for the given tickers/date range.")

    trdval = load_field(conn, "trdval")
    target_adv = trdval[args.target] if args.target in trdval.columns else None
    pair_adv = trdval[args.pair] if args.pair in trdval.columns else None

    model = CostModel()

    df_flat, stats_flat = run_strategy(
        both[args.target], both[args.pair], cost_bps=args.cost_bps, capital=args.capital,
    )
    df_real, stats_real = run_strategy(
        both[args.target], both[args.pair], capital=args.capital,
        cost_model=model, target_adv=target_adv, pair_adv=pair_adv,
    )

    if not df_flat["position"].equals(df_real["position"]):
        raise RuntimeError(
            "flat and realistic runs produced different position paths - the cost "
            "model must not influence signal generation; this comparison is invalid"
        )

    n_trades, avg_hold = _holding_stats(df_flat["position"])
    turnover_sum = df_flat["position"].diff().abs().fillna(0).sum()

    t_name, p_name = names.get(args.target, args.target), names.get(args.pair, args.pair)

    print("=" * 72)
    print("COST MODEL COMPARISON - flat (strategy.py --cost-bps) vs realistic (costs.py)")
    print("=" * 72)
    print(f"pair:             {t_name} ({args.target}) vs {p_name} ({args.pair})")
    print(f"period:           {both.index.min().date()} .. {both.index.max().date()}  ({len(both)} trading days)")
    print(f"capital:          {args.capital:,.0f} KRW")
    print(f"flat cost_bps:    {args.cost_bps}")
    print("realistic CostModel:")
    for f in fields(CostModel):
        print(f"  {f.name:<24s}{getattr(model, f.name)}")
    print(f"git commit:       {_git_commit()}")
    print()

    print(f"{'component':<20s}{'flat':>20s}{'realistic':>20s}")
    for comp in COMPONENTS:
        print(f"{comp:<20s}{'N/A (분해 불가)':>20s}{_pct(stats_real[f'cost_{comp}']):>20s}")
    print(f"{'total cost drag':<20s}{_pct(stats_flat['total_cost']):>20s}{_pct(stats_real['total_cost']):>20s}")
    print(
        f"\n  flat total cost = turnover_sum({turnover_sum:.4f}) x cost_bps({args.cost_bps}bp) "
        f"= {_pct(stats_flat['total_cost'])}  <- no per-component breakdown exists in this model"
    )
    print()

    print(f"{'':<20s}{'flat':>20s}{'realistic':>20s}")
    print(f"{'gross return':<20s}{_pct(stats_flat['gross_total_return']):>20s}{_pct(stats_real['gross_total_return']):>20s}")
    print(f"{'net return':<20s}{_pct(stats_flat['total_return']):>20s}{_pct(stats_real['total_return']):>20s}")
    print()

    print(f"trades (shared position path, both models): {n_trades}")
    print(f"avg holding days per trade:                 {avg_hold:.1f}")

    if args.impact_sweep:
        print()
        print("=" * 72)
        print("IMPACT_COEF SENSITIVITY - not calibrated, no execution data to fit it against")
        print("=" * 72)
        print(f"{'impact_coef':<24s}{'total cost':>14s}{'x flat':>10s}{'impact only':>16s}")
        for coef in IMPACT_SWEEP_COEFS:
            _, stats_sweep = run_strategy(
                both[args.target], both[args.pair], capital=args.capital,
                cost_model=CostModel(impact_coef=coef), target_adv=target_adv, pair_adv=pair_adv,
            )
            multiple = stats_sweep["total_cost"] / stats_flat["total_cost"]
            if coef == 0.0:
                label = f"{coef:.1f} (lower bound)"
            elif coef == model.impact_coef:
                label = f"{coef:.1f} (default)"
            else:
                label = f"{coef:.1f}"
            print(f"{label:<24s}{_pct(stats_sweep['total_cost']):>14s}{multiple:>9.2f}x{_pct(stats_sweep['cost_impact']):>16s}")
        print(
            "\n  0.0 (lower bound) = 수수료+세금+스프레드+대차만, 시장충격 가정 없음 - "
            "공시된 요율로만 계산되므로 가장 방어하기 쉬운 하한선.\n"
            "  결론 '몇 배 과소평가'는 위 impact_coef 선택에 따라 달라진다. "
            "어느 계수가 '맞다'는 주장은 하지 않는다 - 캘리브레이션할 체결 데이터가 없기 때문이다."
        )

    if args.out:
        out_df = pd.DataFrame(
            {
                "component": COMPONENTS + ["total"],
                "flat": [None] * len(COMPONENTS) + [stats_flat["total_cost"]],
                "realistic": [stats_real[f"cost_{c}"] for c in COMPONENTS] + [stats_real["total_cost"]],
            }
        )
        out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\nComparison table written to {args.out}")


if __name__ == "__main__":
    main()
