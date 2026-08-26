# pairs-trading-krx

KOSPI pairs-trading research pipeline: collect KRX daily data → screen for
cointegrated pairs (bidirectional Engle-Granger + half-life filter, multiple-
testing corrected) on economically-motivated universes → select and trade
pairs on a rolling, look-ahead-free formation/trading split (Gatev et al.
2006) → filter trading days by market regime (Gaussian mixture **or** a
from-scratch hidden Markov model) → backtest the spread strategy under a
realistic transaction-cost model (sell-side tax, stock borrow fee, ADV-scaled
market impact).

**The headline result of this repo is negative, and it is reported as such.**
See [Results](#results). Nothing here was tuned until it looked good.

## Where each piece lives

| Component | File |
|---|---|
| KRX Open API collection (price, 거래대금, 시가총액, 상장주식수) | [`collect.py`](collect.py) |
| SQLite storage, schema migrations, long→wide loaders | [`store.py`](store.py) |
| Economically-motivated pair universes (common/preferred, holdco/sub, sector) | [`universe.py`](universe.py) |
| Bidirectional Engle-Granger cointegration + OU half-life filter, 1×N with FDR correction | [`cointegration.py`](cointegration.py) |
| All-pairs C(n,2) basket screen | [`screen_basket.py`](screen_basket.py) |
| Static / walk-forward-beta z-score backtest (full-sample pair selection — kept, biased, for comparison) | [`backtest.py`](backtest.py) |
| **Rolling formation/trading split** (Gatev et al. 2006) — look-ahead-free pair selection | [`formation.py`](formation.py) |
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
python cointegration.py --universe holdco     # economically-motivated candidates, not ticker-code order
python strategy.py --target 000660 --pair 322000 --realistic-costs
python formation.py --universe holdco --compare   # look-ahead-free pair selection vs the old pipeline
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
a fabricated ADV makes an illiquid pair look tradable. (`--allow-missing-adv`
overrides this and prices impact at zero; it understates cost and exists only
for comparison runs.)

**ADV coverage**: 거래대금 is backfilled for all 2,856 trading days
(2015-01-02 – 2026-08-21). 1,207 of 2.63M rows (0.05%, 6 tickers) remain NULL —
names that sat on KOSDAQ during 2023–2024 while `collect.py`'s endpoint
(`stk_bydd_trd`) covers 유가증권/KOSPI only. Those rows stay NULL rather than
being imputed, so a backtest touching them fails loudly.

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
(000660/322000, 1억 capital), only the cost model differs.[^1] Reproducible via
[`compare_costs.py`](compare_costs.py) (defaults match this table exactly):

```
python compare_costs.py
```

| | flat `--cost-bps 15` | `--realistic-costs` |
|---|---|---|
| total cost drag | 4.50% | **16.16%** |
| ├ commission | N/A (분해 불가) | 0.73% |
| ├ tax (sell only) | N/A (분해 불가) | 3.67% |
| ├ half-spread | N/A (분해 불가) | 2.45% |
| ├ **impact (ADV)** | N/A (분해 불가) | **5.95%** |
| └ **borrow (short leg)** | N/A (분해 불가) | **3.36%** |
| net total return | −64.14% | −68.08% |

The flat model has no per-component concept at all — its 4.50% is
`turnover_sum × cost_bps`, a single number, not a breakdown that happens to
be zero on the other four rows. `compare_costs.py` prints it that way
(`N/A`, not `0.00%`) deliberately.

[^1]: These figures (total 16.16%, impact 5.95%, net −68.08%) differ slightly
from an earlier version of this table (16.18%, 5.97%, −68.09%) after fixing
a look-ahead bug in `costs.py`'s market-impact estimate: `daily_vol` was
computed from same-day returns instead of being lagged one day like `ADV`
already was, so today's own volatility was leaking into today's impact
cost. Fixed by adding `shift(1)` to `daily_vol` in `leg_cost_series`. The
square-root scaling law (see above) is unaffected by this fix; only the
absolute impact level shifted, and only slightly, because daily volatility
is highly autocorrelated day-to-day for this pair.

**How much of that depends on `impact_coef`?** It's a literature-typical
square-root-law coefficient with no execution data behind it to calibrate
against, and the "flat model understates cost by Nx" headline moves with it.
`compare_costs.py --impact-sweep` sweeps it explicitly instead of hiding that
dependence behind one default:

```
python compare_costs.py --impact-sweep
```

| impact_coef | total cost | × flat | impact only |
|---|---|---|---|
| 0.0 (lower bound) | 10.21% | **2.27×** | 0.00% |
| 0.2 | 12.19% | 2.71× | 1.98% |
| 0.4 | 14.18% | 3.15× | 3.97% |
| 0.6 (default) | 16.16% | 3.59× | 5.95% |
| 0.8 | 18.14% | 4.03× | 7.93% |
| 1.0 | 20.13% | 4.47× | 9.91% |

The `0.0` row is not an assumption — it's commission + sell-side tax +
half-spread + borrow only, all four taken straight from disclosed rates, with
market impact switched off entirely. That makes **2.27×** the floor of this
claim: the most defensible number, since it needs no impact model at all.
No claim is made here that any particular `impact_coef` is "correct" — the
point is showing how far the conclusion moves under the assumption, not
picking a value.

Under the default assumption (`impact_coef=0.6`), the flat model understates
cost by **3.6×** here, and the two largest
omissions are precisely the two a per-turnover model structurally cannot
see: impact, which depends on order size against the name's liquidity
(322000 is far thinner than the 005930-class names the 15bp figure implicitly
assumes), and borrow, which accrues with *time held* rather than with
trading.

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

### Rolling formation/trading collapses the apparent edge (`formation.py`)

Every result above — and everything `backtest.py`/`strategy.py` produce —
selects its pair on the *same* sample it then backtests. Beta is
walk-forward, but the decision "trade this pair at all" still uses
information that did not exist at the start of the sample.
[`formation.py`](formation.py) implements the standard fix (Gatev, Goetzmann
& Rouwenhorst 2006): pick pairs on a rolling **formation** window (bidirectional
Engle-Granger + FDR + half-life filter, corrected independently *within* each
window), then trade only those pairs, unchanged, through the following
**trading** window, forced flat at the boundary. No pair is ever selected and
traded on the same data.

**common/preferred universe, defaults (formation=252d, trading=126d, α=0.05,
half-life ∈[5,60]d, costs.py's default cost model), 2015–2026, 20 rolling
windows, 134 candidate pairs:**

| | rolling formation/trading | old pipeline (`--compare`: full-sample select + `backtest.py`, no costs) |
|---|---|---|
| pairs significant | 61 pair-window picks across 16/20 windows | 31 / 126 (single global FDR) |
| total return | **+3.58%** | **+1,944.14%** |
| Sharpe | 0.10 | 0.51 |
| max drawdown | −34.05% | −12.74% |
| total trades | 144 | — |

**The old pipeline's +1,944% is not a real achievable return — it is what
pair-selection look-ahead looks like when it is large.** It comes from
picking, with the full benefit of hindsight, the 31 out of 126 common/preferred
pairs that happened to diverge and revert profitably somewhere in 2015-2026,
backtesting each frictionlessly over that *same* full history, and
equal-weighting them — which lets diversification across many independently
"lucky" curve-fit bets compound their gains while diversifying away much of
each one's individual volatility drag. Every ingredient (which pairs, that
they have no transaction costs, that they're evaluated on the sample that
picked them) is information a 2015 trader did not have. Once pair selection
is pushed out-of-sample and realistic costs are added, the same universe
returns **+3.58%** over the same eleven years — Sharpe 0.10, indistinguishable
from noise, on 144 trades. This is the headline number for this repo: the
size of the pair-selection look-ahead bias its own original pipeline was
carrying, on the most literally "cointegrated by construction" universe
available (common and preferred shares of the same company).

Reproduce: `python formation.py --universe preferred --compare`

**holdco universe** (holding company / core subsidiary, 8 hand-picked pairs)
tells the same story from the opposite direction — here almost nothing
survives out-of-sample at all:

| | rolling formation/trading | old pipeline (`--compare`) |
|---|---|---|
| pairs ever selected | 6 pair-window picks across 3/20 windows | 1 (LG / LG전자) |
| total return | **−25.12%** | −12.41% |
| Sharpe | −0.28 | 0.08 |
| max drawdown | −35.67% | −42.84% |
| windows with zero pairs selected | **17 / 20 (85%)** | n/a (single global selection) |

With only ~252 formation days and per-window FDR correction, 17 of 20
windows find nothing that survives both the cointegration test and the
half-life filter, even in a universe chosen for having a real economic
reason to co-move. **Out-of-sample, tradeable cointegrated pairs are rare**
— reported as a finding, not a failed run.

Reproduce: `python formation.py --universe holdco --compare`

In both universes the gap between the two columns is not *purely*
look-ahead: the old pipeline also has no cost model (`backtest.py` never
did), so part of the difference is costs.py being applied on the rolling
side and not the baseline. `formation.py --compare` prints this caveat every
run rather than letting the comparison imply more than it shows. Defaults
were not tuned after the fact for either universe — these are the first and
only runs of `formation.py` against each.

## Known limitations

These are real and not yet fixed:

1. **Multiple testing is corrected for p-values but not for performance.**
   FDR is applied when screening cointegration; the reported Sharpe of the
   best pair out of hundreds gets no equivalent haircut. `formation.py`'s
   independent per-window FDR correction narrows this (each window is its
   own smaller multiple-testing universe) but does not fix it end-to-end —
   the *portfolio-level* Sharpe of "whichever pairs each window happened to
   pick" is still an unadjusted statistic.
2. **`impact_coef` is uncalibrated** (above).
3. **No borrow availability modeling.** The borrow fee is charged, but the
   model assumes the short leg is always borrowable. Hard-to-borrow names —
   exactly the ones a divergence signal tends to select — may not be
   shortable at any price.
4. **Survivorship.** The ticker universe is what KRX returns today; names
   delisted mid-sample are absent from screens over historical windows.
5. **`formation.py`'s multi-pair portfolio is equal-weight, not risk-weighted.**
   When a window selects more than one pair, capital is split evenly across
   them regardless of each pair's volatility or half-life. This is a
   documented simplification, not a claim that equal weight is optimal.
