# pairs-trading-krx

KOSPI pairs-trading research pipeline: collect KRX daily data → screen for
cointegrated pairs with multiple-testing correction → filter trading days by
market regime (Gaussian mixture **or** a from-scratch hidden Markov model) →
backtest the spread strategy under a realistic transaction-cost model
(sell-side tax, stock borrow fee, ADV-scaled market impact).

**The headline result of this repo is negative, and it is reported as such.**
See [Results](#results). Nothing here was tuned until it looked good.

## Where each piece lives

| Component | File |
|---|---|
| KRX Open API collection (price, 거래대금, 시가총액, 상장주식수) | [`collect.py`](collect.py) |
| SQLite storage, schema migrations, long→wide loaders | [`store.py`](store.py) |
| Engle-Granger cointegration, 1×N with FDR correction | [`cointegration.py`](cointegration.py) |
| All-pairs C(n,2) basket screen | [`screen_basket.py`](screen_basket.py) |
| Static / walk-forward-beta z-score backtest | [`backtest.py`](backtest.py) |
| **GMM** regime filter | [`gmm_strategy.py`](gmm_strategy.py) |
| **HMM** (Baum-Welch + scaled forward-backward, from scratch) | [`hmm.py`](hmm.py) |
| **HMM** regime-filtered strategy | [`hmm_strategy.py`](hmm_strategy.py) |
| **Realistic transaction-cost engine** (대차수수료 · ADV 슬리피지) | [`costs.py`](costs.py) |
| Full strategy: regime filter + costs + stop-loss | [`strategy.py`](strategy.py) |

## Setup

```
pip install pandas numpy statsmodels scikit-learn requests python-dotenv
```

The HMM is implemented in numpy in [`hmm.py`](hmm.py) — no `hmmlearn`, which
ships no wheel for recent Pythons and needs a C++ toolchain to build.

A KRX Open API key (free, registration + per-service approval at
<https://openapi.krx.co.kr>) goes in `.env`:

```
KRX_AUTH_KEY=your_key
```

```
python collect.py --start 2015-01-02          # one request per trading day, whole KOSPI
python cointegration.py --target 005930 --tickers "000660,005380,..."
python strategy.py --target 000660 --pair 322000 --realistic-costs
```

## Regime filter: GMM vs HMM

Both filters classify each day as **calm** or **turbulent** from
`[z-score, spread volatility]`, only open positions on calm days, and
force-flatten when the regime turns. They differ in whether the model knows
that regimes *persist*.

A Gaussian mixture treats every day as an independent draw. An HMM estimates
a transition matrix, so staying in a regime is explicitly likelier than
switching.

**Validation on a synthetic 2-regime process with known parameters**
(deliberately *overlapping* regimes — the easy well-separated case does not
discriminate between the two methods, both score ~100%):

| | accuracy | regime switches (true: 43) |
|---|---|---|
| HMM, filtered (causal) | 0.933 | 79 |
| HMM, smoothed | 0.956 | 39 |
| GMM (no persistence) | 0.796 | **481** |

The HMM also recovers the transition matrix closely (fitted
`[[0.977, 0.023], [0.043, 0.957]]` vs true `[[0.98, 0.02], [0.05, 0.95]]`).

### Look-ahead discipline

Two separate guards, both required:

1. **Walk-forward fitting** — parameters re-estimated only every
   `--reestimate-every` days on the trailing window strictly *before* the
   current day.
2. **Filtered, not smoothed, decoding** — the day's label comes from the
   forward pass only (`filter_proba`). This is the trap in most HMM
   backtests: the natural library call (`predict()` = Viterbi over the whole
   sequence) labels every day using the entire series *including the future*,
   quietly turning the regime filter into an oracle. `hmm.py` exposes both and
   documents which is which; only the causal one feeds trading decisions.

## Transaction costs (`costs.py`)

The original model charged one flat `--cost-bps` on every unit of turnover.
That hides three things that decide whether a spread strategy is actually
implementable:

| Component | Applies to | Note |
|---|---|---|
| 위탁수수료 commission | both sides | bps of traded notional |
| 증권거래세 + 농특세 | **sell side only** | 15bp KOSPI (2025 schedule) — a flat round-trip figure wrongly charges the buy too |
| half-spread | both sides | crossing the bid/ask |
| **market impact** | both sides | **square-root law**, scales with order size / ADV |
| **대차수수료 borrow fee** | **short leg, daily** | accrues while held — a per-turnover model charges *nothing* for a 40-day hold |

Impact uses the standard square-root law (Almgren-Chriss / Barra family):

```
impact_bps = impact_coef × daily_vol_bps × sqrt(order_value / ADV)
```

ADV is trailing average daily traded **value** (거래대금, KRW), lagged one day
so the day's own volume cannot leak into its own cost estimate. If ADV is
missing for a window, `costs.py` **raises rather than substituting a number** —
a fabricated ADV makes an illiquid pair look tradable.

**Verified scaling** (005930/000660, identical position path, varying capital):

| capital | commission | tax | half-spread | **impact** | **borrow** | total |
|---|---|---|---|---|---|---|
| 1억 | 0.120% | 0.600% | 0.400% | **0.182%** | 2.381% | 3.68% |
| 100억 | 0.120% | 0.600% | 0.400% | **1.816%** | 2.381% | 5.32% |
| 2,000억 | 0.120% | 0.600% | 0.400% | **8.120%** | 2.381% | 11.62% |

100× the capital gives 10× the impact (√100) and a further 20× gives 4.47×
(√20) — the square-root law holds exactly. Bps-proportional components are
size-invariant, and borrow depends on holding period, not order size, as it
should.

**What the flat model was hiding** — same pair, same signal, same 1,658 days
(000660/322000, 1억 capital), only the cost model differs:

| | flat `--cost-bps 15` | `--realistic-costs` |
|---|---|---|
| total cost drag | 4.50% | **10.60%** |
| ├ commission | — | 0.73% |
| ├ tax (sell only) | — | 3.67% |
| ├ half-spread | — | 2.45% |
| ├ impact (ADV) | — | 0.39% |
| └ **borrow (short leg)** | **0.00%** | **3.36%** |
| net total return | −64.14% | −66.26% |

The flat model understates cost by **2.4×** here, and the single largest
omission is the borrow fee — a cost that is invisible to any per-turnover
model because it accrues with *time held*, not with trading.

`impact_coef` (default 0.6) is a **documented assumption, not a fitted
value** — there is no execution data here to calibrate against. Treat the
*level* as indicative and the *comparison across turnover* as the meaningful
output. Same for the default rates: typical retail/small-institution KRX
levels, not quotes from any specific broker.

## Results

### Cointegration screening finds almost nothing

Samsung Electronics (005930) against 250 KOSPI names with full 2015–2026
history: **0 of 250 pairs significant after FDR correction** (α=0.05). The
strongest candidate reached raw p=0.0002 but adjusted p=0.050. Bank-sector
screening (신한지주 vs 하나·우리·KB·메리츠·기업): **0 of 5**.

### Both regime filters lose money on the pairs tested

SK Hynix (000660) / HD Hyundai Energy Solutions (322000), 1,658 days:

| | HMM | GMM |
|---|---|---|
| total return | **−52.03%** | **−31.18%** |
| Sharpe | −0.54 | −0.19 |
| max drawdown | −55.50% | −56.88% |
| trades | 20 | 26 |
| % days calm | 39.6% | 43.7% |
| regime switches | **21** | **33** |

The HMM does what it was built to do — 36% fewer regime switches than the
GMM, matching the synthetic result — and still **performs worse on this
pair**. Fewer whipsaws is not the same as a profitable signal, and neither
filter can rescue a pair that is not genuinely cointegrated. Reported as
found; not re-tuned to produce a better-looking table.

## Known limitations

These are real and not yet fixed:

1. **Pair-selection look-ahead.** `cointegration.py` selects pairs on the
   full sample, then `backtest.py` backtests on that same sample. Beta is
   walk-forward, but the decision of *which pair to trade* still uses future
   information. This is the biggest flaw in the repo and needs a rolling
   formation/trading split to fix properly.
2. **Multiple testing is corrected for p-values but not for performance.**
   FDR is applied when screening cointegration; the reported Sharpe of the
   best pair out of hundreds gets no equivalent haircut.
3. **`impact_coef` is uncalibrated** (above).
4. **No borrow availability modeling.** The borrow fee is charged, but the
   model assumes the short leg is always borrowable. Hard-to-borrow names —
   exactly the ones a divergence signal tends to select — may not be
   shortable at any price.
5. **Survivorship.** The ticker universe is what KRX returns today; names
   delisted mid-sample are absent from screens over historical windows.
