"""Realistic transaction-cost engine for the KRX pairs strategy.

strategy.py originally charged a single flat `--cost-bps` on every unit of
turnover. That is fine as a first pass but it hides three things that decide
whether a spread strategy is actually implementable:

  1. **Tax is sell-side only, and asymmetric.** Korean equities carry
     증권거래세 + 농어촌특별세 on *sales*, not purchases. A flat round-trip
     bps figure charges the same on the buy leg, which both overstates entry
     cost and understates how much worse a high-turnover strategy gets.

  2. **The short leg is not free to hold.** A pairs trade is short one leg
     for its whole life, and 대차수수료 (stock borrow fee) accrues daily on
     that leg's notional whether or not the position moves. A per-turnover
     cost model charges nothing for a position held 40 days, which is
     exactly where the borrow bill shows up.

  3. **Slippage scales with order size relative to liquidity.** Trading
     ₩1bn of a stock that turns over ₩500bn a day is nearly free; the same
     ₩1bn in a stock that turns over ₩2bn a day is not. A constant bps
     assumption is only right for one (unstated) order size.

Cost components
---------------
Per trade, per leg:
    commission (위탁수수료)   both sides, bps of traded notional
    tax (거래세+농특세)       SELL side only, bps of traded notional
    half-spread              both sides, bps - crossing the bid/ask
    market impact            both sides, square-root law (below)

Per day held:
    borrow fee (대차수수료)   SHORT leg only, annual rate accrued daily

Market impact uses the standard square-root law (Almgren-Chriss / Barra
family):

    impact_bps = impact_coef * daily_vol_bps * sqrt(order_value / ADV)

i.e. impact grows with the square root of participation rate, scaled by how
volatile the name already is. `impact_coef` is a free parameter - the
literature puts it near 0.5-1.0 for equities; it is NOT calibrated on this
dataset (there is no execution data here to calibrate against), so it is a
documented assumption, not a fitted value. Treat the level as indicative and
the *comparison* across turnover levels as the meaningful output.

ADV is trailing average daily traded *value* (거래대금, KRW), not share
volume - see store.py. If ADV data is missing for a window, cost_series()
raises rather than silently substituting a number, because a fabricated ADV
makes an illiquid pair look tradable.

Defaults are documented and dated below - they are typical retail/small-
institution KRX levels, not quotes from any specific broker, and should be
overridden with real numbers before any of this means anything for a live
account.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class CostModel:
    """All rates in basis points unless named otherwise.

    Defaults reflect KRX conditions as of 2025 for a small account:
      - commission_bps 1.5: online brokerage 위탁수수료, roughly 0.015%.
      - tax_bps 15.0: KOSPI sale, 증권거래세 0.0% + 농어촌특별세 0.15%
        (the 2025 schedule; KOSDAQ is 0.15% with no 농특세 split). Applied
        to sells only.
      - half_spread_bps 5.0: half of a ~10bp round-trip spread, typical for
        a liquid KOSPI large cap. Small caps are materially worse.
      - borrow_fee_annual_bps 300.0: 3%/yr 대차수수료 on the short leg -
        a mid-range general-collateral rate. Hard-to-borrow names run far
        higher and are exactly the names a spread signal tends to pick.
      - impact_coef 0.6: square-root-law coefficient, literature-typical,
        NOT calibrated here (see module docstring).
    """

    commission_bps: float = 1.5
    tax_bps: float = 15.0
    half_spread_bps: float = 5.0
    borrow_fee_annual_bps: float = 300.0
    impact_coef: float = 0.6
    trading_days: int = TRADING_DAYS

    def borrow_daily_rate(self) -> float:
        return (self.borrow_fee_annual_bps / 10_000) / self.trading_days


def _impact_bps(
    order_value: pd.Series, adv: pd.Series, daily_vol: pd.Series, coef: float
) -> pd.Series:
    """Square-root market impact, in bps. Zero where nothing is traded.

    daily_vol is a *fractional* daily return stdev (0.02 = 2%); it is
    converted to bps here so the result is on the same scale as the other
    cost components.
    """
    participation = order_value.divide(adv).replace([np.inf, -np.inf], np.nan)
    participation = participation.clip(lower=0).fillna(0.0)
    vol_bps = (daily_vol.fillna(0.0) * 10_000).clip(lower=0)
    return coef * vol_bps * np.sqrt(participation)


def leg_cost_series(
    exposure: pd.Series,
    price_ret: pd.Series,
    adv: pd.Series,
    capital: float,
    model: CostModel,
    vol_window: int = 60,
    adv_window: int = 20,
    allow_missing_adv: bool = False,
) -> pd.DataFrame:
    """Per-day cost of running ONE leg, as a fraction of `capital`.

    `exposure` is the signed leg position in units of capital: +0.8 means
    long 0.8 x capital of this name, -1.0 means short a full unit. Its
    day-over-day change is the traded amount; its sign on a held day decides
    whether borrow fee accrues.

    Returns a DataFrame with one column per cost component (all positive
    numbers = money lost) plus `total`.
    """
    exposure = exposure.fillna(0.0)
    trade = exposure.diff().fillna(exposure.iloc[0] if len(exposure) else 0.0)
    traded_value = trade.abs() * capital

    if adv is None or adv.dropna().empty:
        if not allow_missing_adv:
            raise ValueError(
                "No ADV (거래대금) data for this window - run collect.py to backfill "
                "prices.trdval, or pass allow_missing_adv=True to price impact at zero "
                "and accept that the result understates cost."
            )
        adv_aligned = pd.Series(np.nan, index=exposure.index)
    else:
        adv_aligned = adv.reindex(exposure.index)

    # Trailing ADV, not same-day: on the day you trade you do not yet know
    # that day's full traded value, and using it would leak look-ahead
    # liquidity into the cost estimate.
    adv_trailing = adv_aligned.shift(1).rolling(adv_window, min_periods=5).mean()
    # Trailing vol, not same-day: impact is estimated at order-placement time,
    # before today's return is known, so today's realized vol cannot feed it.
    daily_vol = price_ret.shift(1).rolling(vol_window, min_periods=10).std()

    commission = traded_value / capital * (model.commission_bps / 10_000)
    half_spread = traded_value / capital * (model.half_spread_bps / 10_000)

    # Tax hits sales only. A sale is any trade that decreases signed
    # exposure: closing a long, or opening/increasing a short.
    sold_value = (-trade).clip(lower=0) * capital
    tax = sold_value / capital * (model.tax_bps / 10_000)

    impact_bps = _impact_bps(traded_value, adv_trailing, daily_vol, model.impact_coef)
    impact = traded_value / capital * (impact_bps / 10_000)

    # Borrow accrues on the short exposure carried into the day, so shift(1):
    # you pay for what you were already short overnight, not for what you
    # short at today's close.
    short_exposure = (-exposure).clip(lower=0).shift(1).fillna(0.0)
    borrow = short_exposure * model.borrow_daily_rate()

    out = pd.DataFrame(
        {
            "commission": commission,
            "tax": tax,
            "half_spread": half_spread,
            "impact": impact,
            "borrow": borrow,
        },
        index=exposure.index,
    ).fillna(0.0)
    out["total"] = out.sum(axis=1)
    return out


def pair_cost_series(
    position: pd.Series,
    beta: pd.Series,
    target_ret: pd.Series,
    pair_ret: pd.Series,
    target_adv: pd.Series,
    pair_adv: pd.Series,
    capital: float,
    model: CostModel,
    allow_missing_adv: bool = False,
) -> pd.DataFrame:
    """Total cost of the two-leg pair trade, as a fraction of capital/day.

    `position` is +1 (long target / short pair), -1 (the reverse), or 0.
    The pair leg is beta-weighted, matching the spread definition in
    backtest.py (spread = log_target - beta * log_pair), so its exposure is
    -beta * position.
    """
    beta = beta.reindex(position.index).ffill().fillna(0.0)
    target_exposure = position.astype(float)
    pair_exposure = -beta * position

    t_costs = leg_cost_series(
        target_exposure, target_ret, target_adv, capital, model,
        allow_missing_adv=allow_missing_adv,
    )
    p_costs = leg_cost_series(
        pair_exposure, pair_ret, pair_adv, capital, model,
        allow_missing_adv=allow_missing_adv,
    )
    return t_costs.add(p_costs, fill_value=0.0)
