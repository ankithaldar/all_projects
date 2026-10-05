# Indicator-Driven Strategies — Measured Results

**Every number on this page is a property of a price GENERATOR, not a fact
about Indian equities.** The panels are synthetic geometric random walks built
by `random.Random`, not NSE data. A Sharpe of 5.5 below means the strategy
extracted the drift that was *put into* the generator by arithmetic. It says
nothing about whether the rule would survive Indian delivery costs, because
there is no market in the panel to survive.

Read this page as a description of the code's behaviour and as a record of
what the five strategies did under two generators. Do not quote any figure from
it as evidence about a market.

- Source: `src/stock_rl/strategies.py`
- Tests: `tests/test_strategies.py` (115 tests)
- Engine: `stock_rl.portfolio.run_portfolio`, the only backtester, unmodified
- Baselines: the five in `stock_rl.baselines`, unmodified

---

## The headline, stated plainly

**On the flat (zero-drift) panel, four of the five indicator strategies LOSE to
equal weight. One — `atr_breakout` — beat it on that panel, by 0.107 Sharpe,
and that number does not survive the search-length gate.**

This is the repository's existing finding reproduced at the indicator level:
simple baselines beat the learned policies. Here, on a panel with no built-in
trend, four of five hand-built technical rules did worse than holding all
twelve names equally. That is the expected result, not a bug. Rink (2023)
measured technical rules on 41 national indices and concluded they "should be
treated as noise"; `docs/research/methodology/technical-indicators-nse.md`
records that the indicator thresholds behind this project's ranking have **no
published Indian evidence**, that standalone technical signals have been
measured **inverted**, and that Nifty-50 constituents are the *most* efficient
segment, so weak signals on large caps are the expected outcome rather than a
failure of the implementation.

`rsi_mean_reversion` was the worst strategy on the drifted panel and lost to
equal weight on **15 of 15 seeds** of it. RSI has no Indian equity evidence
(Rink tested 600 RSI rules; the family is almost never best) and the repository
chose to measure it anyway precisely so the negative result would be on record
rather than assumed.

---

## The five strategies and their DECLARED thresholds

Every threshold below is a **conventional published reading of the indicator**,
fixed before the panel was run. **None was adjusted after seeing a result.**
`docs/INDICATOR-STRATEGIES.md` and the module docstrings carry the same values,
so the claim is checkable.

| Strategy | Declared values | Why these values |
|---|---|---|
| `rsi_mean_reversion` | Wilder RSI(14), entry 30, top 10 | 14 is Wilder's own default and is what every description of the indicator uses; 30 is the oversold line printed on it. Sized by *depth* below the line, so a reading exactly on the line gets zero — no measured edge is no position. |
| `ema_trend_following` | slow EMA 200, trend_slope over 20 bars, top 10 | 200 is the conventional long window; 20 is a conventional one-month direction check. Two conditions, not one: price above a long average says a name *has* risen, a positive recent slope says it is *still* rising. |
| `volatility_scaled_momentum` | 12-1 momentum (252 lookback, 21 skip), vol window 60, top 10 | Same momentum leg as `baselines.momentum_ranked`, so the comparison isolates one change: each name's share divided by its realised volatility. 252/21/60 are the conventional daily-bar horizons. |
| `atr_breakout` | ATR(14), 60-bar closing high, advance ≥ 1.0 ATR, top 10 | 14 is Wilder's default. 60 bars sits inside the 20-to-60-bar band `technical-indicators-nse.md` identifies as the only cost-viable MA horizon in Indian data, rather than the retail 52-week high. One ATR is the smallest advance distinguishable from a rounding tick. |
| `volatility_filtered_momentum` | 12-1 momentum, vol window 60, band [0.15, 0.45], top 10 | Band endpoints from published descriptions of Indian large-cap volatility: a Nifty-style index has spent most of two decades between roughly 12% and 30% annualised, so 0.15 excludes names calmer than any broad Indian index and 0.45 admits only crisis conditions. Han/Yang/Zhou (2013) find MA timing works on *high*-vol portfolios and fails on low-vol ones; Maheshwari & Dhankar (2017) measure momentum turning negative in crises. The band excludes both ends. |

The repository's own methodology file says to condition on volatility rather
than on a trend-or-range label, which is why strategy 5's gate is a volatility
band and not an ADX-style regime read.

### The 50/200 pairing has no Indian evidence, and this says so

`technical-indicators-nse.md` records that 50/200 is folklore in India, that the
only cost-aware peer-reviewed Indian study of moving-average rules (Mitra 2011)
found short crossovers losing 7–13% annualised *before costs*, and that the
200-bar EMA was tested paired with a **9- or 12-bar** EMA, never a 50-bar one.
`ema_trend_following` uses 200 because it is the declared conventional value,
**not** because it is the evidenced one. That distinction is in the function's
docstring too.

---

## The panels — described and labelled SYNTHETIC

Both panels: 12 symbols (above `1 / max_weight`, or four of five baselines
collapse into one uniform book), 1,260 daily bars = 5 years, one calendar day
between bars, IST-naive timestamps, prices `round(100 + 5·offset, 4)` scaled by
a per-bar multiplicative return. Engine spec identical for every row:
`rebalance_days=21`, `history=60`, `capital=10,000,000`, `max_weight=0.10`,
`costs=DELIVERY`.

| | Panel A — DRIFTED | Panel B — FLAT |
|---|---|---|
| Generator | six symbols at **+0.20%/bar**, six at **−0.15%/bar** | **zero drift**; all twelve are pure random walks |
| Per-bar noise | `gauss(0, 0.012)` on both | `gauss(0, 0.012)` on both |
| Seed | 20260101 | 20260101 |
| What it is | a panel with a **systematic trend built in** | a panel with **no cross-sectional trend to find** |
| Trading-window bars | 1,200 (4.76 years) | 1,200 (4.76 years) |

Panel A is the more flattering of the two by construction: any rule that buys
rising names is handed the effect it trades. **The gap between the two panels is
the actual finding**, and it is a measurement of the generator.

---

## Measured results — Panel A, DRIFTED (SYNTHETIC)

Sharpe is annualised over the 1,200-bar trading window. `held%` is the mean
fraction of symbol-slots carrying a non-zero weight.

| strategy | Sharpe | growth | maxDD | turnover | cost ₹ | held% |
|---|---|---|---|---|---|---|
| equal_weight | 1.389 | 1.413 | 0.049 | 3.89 | 76,655 | 100.0% |
| buy_and_hold | 1.389 | 1.413 | 0.049 | 3.89 | 76,655 | 100.0% |
| momentum_ranked | 2.472 | 1.937 | 0.046 | 7.51 | 171,647 | 86.5% |
| low_volatility | 1.368 | 1.407 | 0.051 | 5.44 | 105,155 | 100.0% |
| trend_filtered_momentum | 4.129 | 2.360 | 0.035 | 19.32 | 464,462 | 51.3% |
| **rsi_mean_reversion** | **−1.379** | 0.902 | 0.100 | 6.63 | 100,346 | 6.0% |
| **ema_trend_following** | **5.047** | 2.407 | 0.020 | 11.75 | 296,425 | 32.3% |
| **volatility_scaled_momentum** | **5.525** | 2.849 | 0.024 | 2.65 | 70,155 | 40.6% |
| **atr_breakout** | **2.097** | 1.207 | 0.013 | 9.47 | 168,760 | 7.6% |
| **volatility_filtered_momentum** | **5.411** | 2.793 | 0.024 | 2.84 | 73,964 | 40.4% |

Note that `trend_filtered_momentum`, an existing baseline, beats every indicator
strategy here except the two volatility-scaled momentum rules. The strongest
baseline is a **baseline**.

## Measured results — Panel B, FLAT, zero drift (SYNTHETIC)

| strategy | Sharpe | growth | maxDD | turnover | cost ₹ | held% |
|---|---|---|---|---|---|---|
| equal_weight | **0.067** | 1.010 | 0.085 | 3.41 | 60,287 | 100.0% |
| buy_and_hold | 0.067 | 1.010 | 0.085 | 3.41 | 60,287 | 100.0% |
| momentum_ranked | 0.054 | 1.007 | 0.079 | 7.57 | 127,974 | 86.5% |
| low_volatility | 0.057 | 1.008 | 0.075 | 5.13 | 87,186 | 100.0% |
| trend_filtered_momentum | 0.000 | 0.996 | 0.059 | 23.31 | 381,826 | 42.4% |
| **rsi_mean_reversion** | **−0.450** | 0.970 | 0.056 | 5.02 | 80,711 | 3.9% |
| **ema_trend_following** | **−0.953** | 0.859 | 0.143 | 20.86 | 314,936 | 30.1% |
| **volatility_scaled_momentum** | **−0.271** | 0.953 | 0.098 | 9.49 | 153,427 | 41.4% |
| **atr_breakout** | **0.174** | 1.010 | 0.025 | 5.22 | 84,632 | 4.1% |
| **volatility_filtered_momentum** | **−0.297** | 0.949 | 0.111 | 9.55 | 154,233 | 40.9% |

---

## Did any strategy beat equal weight?

**Panel A (drifted):** yes — four of five, by 0.71 to 4.14 Sharpe.
**Panel B (flat):** one of five, `atr_breakout`, by **0.107**.

Stated plainly and without softening either panel:

1. On the drifted panel the four wins are an artefact of the generator's built-in
   drift. `volatility_scaled_momentum`'s Sharpe of 5.525 is the drift, extracted.
   It is not evidence that volatility scaling works.
2. On the flat panel `atr_breakout` beat equal weight by 0.107 on a panel with
   **no** trend to find, across a 4.76-year window. That is the single most
   interesting number on this page and it is very small.
3. **The panel B win does not pass the gate.** `minimum_backtest_length(10, 0.174)`
   requires **81.52 years** of returns to defend a best-of-ten Sharpe of 0.174.
   The panel has 4.76. `stock_rl.strategies.measure` therefore **refuses** the
   whole Panel B report and no Sharpe from it should be quoted. That refusal is
   the module working as intended.
4. The **seed sweep** below is the honest robustness check, and it is the one
   that settles the question.

### Seed sweep — 15 independent seeds per panel (SYNTHETIC)

Each seed regenerates the whole panel. "beats EW" counts how many of the 15 had
a Sharpe above that seed's own `equal_weight`.

| strategy | Panel A mean | A min | A max | **A beats EW** | Panel B mean | B min | B max | **B beats EW** |
|---|---|---|---|---|---|---|---|---|
| equal_weight *(baseline)* | 0.860 | 0.548 | 1.127 | — | −0.411 | −0.760 | −0.100 | — |
| buy_and_hold *(baseline)* | 0.860 | 0.548 | 1.127 | — | −0.411 | −0.760 | −0.100 | — |
| momentum_ranked *(baseline)* | 1.874 | 1.385 | 2.259 | — | −0.514 | −0.822 | −0.146 | — |
| low_volatility *(baseline)* | 0.870 | 0.506 | 1.216 | — | −0.401 | −0.804 | −0.057 | — |
| trend_filtered_momentum *(baseline)* | 4.073 | 3.143 | 5.313 | — | −0.230 | −0.473 | 0.198 | — |
| **rsi_mean_reversion** | −1.660 | −2.091 | −1.304 | **0 / 15** | −0.142 | −0.748 | 0.281 | **14 / 15** |
| **ema_trend_following** | 4.493 | 3.649 | 5.319 | **15 / 15** | −0.510 | −0.790 | −0.180 | **4 / 15** |
| **volatility_scaled_momentum** | 5.279 | 4.643 | 6.177 | **15 / 15** | −0.461 | −0.864 | 0.052 | **7 / 15** |
| **atr_breakout** | 1.912 | 1.380 | 2.622 | **15 / 15** | −0.563 | −0.874 | −0.283 | **5 / 15** |
| **volatility_filtered_momentum** | 5.237 | 4.614 | 6.088 | **15 / 15** | −0.514 | −0.934 | −0.015 | **7 / 15** |

**This table is the finding.** Read the B column: with the drift removed, four
of the five strategies beat equal weight fewer than half the time
(4/15, 5/15, 7/15, 7/15), and `rsi_mean_reversion`'s apparent 14/15 is the
*opposite* of a win — it beats equal_weight by being less bad than a losing
market, since its mean Sharpe (−0.142) is far above the equal-weight mean
(−0.411). It is mean-reversion failing to capture a falling market, and it pays
for that in a 40% drawdown on the drifted panel (maxDD 0.100 vs equal weight's
0.049).

The 15/15 column on Panel A is not robustness. It is the generator's drift being
found reliably, which is exactly what 15/15 should look like when the signal is
present in the data by construction.

---

## NO-LOOK-AHEAD — verified, every strategy

`run_portfolio` shows the provider only `panel[:index]`, so the fill price is
structurally invisible; the property tested is that the provider does not reach
around its argument. Test `TestNoLookAhead` replaces **every bar after index 300
with a fabricated rising series at 4× the price** and asserts that all 12
pre-cut target books are bit-identical between the real and fabricated runs, and
that the post-cut books genuinely differ (so the assertion is not vacuous).

Bar 300 sits between rebalance decisions at bar 294 and bar 315, so 12 of the 27
snapshots are pre-cut.

| Strategy | snapshots | pre-cut books bit-identical | post-cut books changed | panel length preserved | prefix preserved |
|---|---|---|---|---|---|
| `atr_breakout` | 27 | **yes** (12/12) | **yes** | yes | yes |
| `ema_trend_following` | 27 | **yes** (12/12) | **yes** | yes | yes |
| `rsi_mean_reversion` | 27 | **yes** (12/12) | **yes** | yes | yes |
| `volatility_filtered_momentum` | 27 | **yes** (12/12) | **yes** | yes | yes |
| `volatility_scaled_momentum` | 27 | **yes** (12/12) | **yes** | yes | yes |

Two supporting controls run beside these: the provider is called twice on the
same argument and must return the identical mapping **and** leave the panel
object unmutated (a cached or mutating provider would answer differently the
second time, which is the other route by which a future leak reaches a past
decision); and a shorter visible history must give a **different** answer, so
the test above cannot be passing merely because the provider ignores its input.

A decision at the close of bar `t` fills at the open of bar `t+1` and is sized on
bar `t−1` close. `indicators.py` is causal throughout and has already had one
off-by-N warm-up bug, so each strategy additionally checks its own declared warm-up
explicitly: `_skip_momentum` drops the last `skip` closes before calling
`momentum`, rather than indexing backwards from the end, because that arithmetic
was inverted once in this repository and inversion silently buys the worst
performer while wearing the label of momentum.

---

## The search-length gate

`stock_rl.strategies.require_history` wraps `metrics.minimum_backtest_length` and
**raises** rather than blanking a field, so no number escapes for a panel that
cannot support it. It is applied to the **best** row, not per row: gating one row
and quoting another would be selecting a Sharpe after seeing it.

| Report | best row | best Sharpe | trials | years required | years available | verdict |
|---|---|---|---|---|---|---|
| Panel A, 10 configurations | `volatility_scaled_momentum` | 5.525 | 10 | 0.08 | 4.76 | reportable (and meaningless — see above) |
| Panel B, 10 configurations | `atr_breakout` | 0.174 | 10 | **81.52** | 4.76 | **REFUSED** |

Note that Panel A *passes* the gate and is still worthless. The gate measures
sample length against a claimed Sharpe; it does not and cannot detect that the
Sharpe came out of the generator. A gate that passes is necessary, not
sufficient.

---

## Parameter sweep — every row reported, best-of-N deflated

**The declared values above were NOT tuned.** A sweep was run afterwards
*specifically to show the cost of searching*, and **every row is printed**, on
both panels. The headline row of a sweep is a **best-of-N** and is passed through
`deflated_sharpe` accordingly.

### Panel A (DRIFTED, SYNTHETIC) — 28 configurations

| strategy | configuration | Sharpe |
|---|---|---|
| `volatility_filtered_momentum` | low 0.15, high 0.45 **(declared)** | 5.411 |
| | low 0.10, high 0.30 | 5.524 |
| | low 0.10, high 0.60 | 5.524 |
| | low 0.20, high 0.60 | 2.670 |
| | low 0.05, high 0.25 | 5.524 |
| | low 0.25, high 0.45 | **0.000** (never fires) |
| `volatility_scaled_momentum` | vol_window 20 | 5.492 |
| | vol_window 60 **(declared)** | 5.525 |
| | vol_window 120 | 5.518 |
| | vol_window 252 | 5.524 |
| `ema_trend_following` | slow 200, trend 20 **(declared)** | 5.047 |
| | slow 50, trend 10 | 4.079 |
| | slow 100, trend 20 | 5.102 |
| | slow 150, trend 30 | 4.991 |
| | slow 200, trend 50 | **5.646 (best)** |
| | slow 250, trend 20 | 4.989 |
| `atr_breakout` | lookback 60, strength 1.0 **(declared)** | 2.097 |
| | lookback 20, strength 1.0 | 2.122 |
| | lookback 120, strength 1.0 | 2.209 |
| | lookback 60, strength 0.5 | 2.376 |
| | lookback 60, strength 2.0 | 1.383 |
| | lookback 250, strength 1.0 | 2.289 |
| `rsi_mean_reversion` | entry 30 **(declared)** | −1.379 |
| | entry 20 | −1.213 |
| | entry 25 | −0.870 |
| | entry 35 | −2.637 |
| | entry 40 | −2.896 |
| | entry 45 | −2.907 |

Best: `ema_trend_following` slow 200 / trend 50, Sharpe **5.646**.
**The declared 200/20 pair scored 5.047 and was not replaced** — 5.646 is the
search's finding, not the strategy's configuration.

- As a **best-of-28**: `deflated_sharpe` = **0.156** (empirical trial dispersion)
  / 7.6e−89 (theoretical plug-in)
- As a **best-of-5** (declared defaults only): `deflated_sharpe` = **0.999997** /
  2.9e−11

The 0.156 is the number that matters. A 5.6 Sharpe surviving a 28-configuration
search with an honest empirical dispersion estimate has a **15.6% probability**
of being a real edge rather than the luckiest of 28 draws. On this repository's
own reading of `deflated_sharpe`, "deflation magnitude, not a 5% error rate" —
but 0.156 is not a probability anyone would act on.

### Panel B (FLAT, SYNTHETIC) — 28 configurations

| strategy | configuration | Sharpe |
|---|---|---|
| `volatility_filtered_momentum` | low 0.15, high 0.45 **(declared)** | −0.297 |
| | low 0.10, high 0.30 | −0.255 |
| | low 0.10, high 0.60 | −0.255 |
| | low 0.20, high 0.60 | −0.387 |
| | low 0.05, high 0.25 | −0.255 |
| | low 0.25, high 0.45 | 0.000 (never fires) |
| `volatility_scaled_momentum` | vol_window 20 | −0.275 |
| | vol_window 60 **(declared)** | −0.271 |
| | vol_window 120 | −0.266 |
| | vol_window 252 | −0.260 |
| `ema_trend_following` | slow 200, trend 20 **(declared)** | −0.953 |
| | slow 50, trend 10 | **0.339 (best)** |
| | slow 100, trend 20 | −0.783 |
| | slow 150, trend 30 | −0.606 |
| | slow 200, trend 50 | −0.316 |
| | slow 250, trend 20 | −0.858 |
| `atr_breakout` | lookback 60, strength 1.0 **(declared)** | 0.174 |
| | lookback 20, strength 1.0 | 0.051 |
| | lookback 120, strength 1.0 | 0.013 |
| | lookback 60, strength 0.5 | 0.310 |
| | lookback 60, strength 2.0 | −0.117 |
| | lookback 250, strength 1.0 | 0.336 |
| `rsi_mean_reversion` | entry 30 **(declared)** | −0.450 |
| | entry 20 | 0.000 (never fires) |
| | entry 25 | −0.687 |
| | entry 35 | −0.185 |
| | entry 40 | −0.025 |
| | entry 45 | −0.177 |

Best: `ema_trend_following` slow 50 / trend 10, Sharpe **0.339** — the *opposite*
configuration from the one the evidence favours, which is the sweep's own
warning sign.

- As a **best-of-28**: `deflated_sharpe` = **0.222**; `minimum_backtest_length`
  needs **36.46 years**, panel has 4.76
- As a **best-of-5**: `deflated_sharpe` = **0.445**; needs **12.40 years**

**Both panels' sweep results fail the length gate.** The searches produced a
best-of-28 Sharpe that 4.76 years cannot defend.

Two configurations in these sweeps produce `Sharpe = 0.000` because the signal
never fires (`volatility_filtered_momentum` at [0.25, 0.45] and
`rsi_mean_reversion` at entry 20 on the flat panel). That is the strategy
correctly refusing to hold anything, and it is reported rather than hidden: a
sweep whose grid contains dead regions is a real grid.

---

## What to do with this

**Keep:** `volatility_scaled_momentum`, as a candidate only. It is the most
defensible of the five because its risk adjustment rests on the volatility
effect — the second-best-evidenced anomaly in Indian data after momentum — rather
than on an untested oscillator, and dividing a position size by a volatility
estimate is a sizing convention that needs no directional claim. It is also the
cheapest to trade of the momentum rules on the drifted panel (turnover 2.65,
₹70,155). **Its measured advantage here is the generator's drift. Nothing on
this page supports shipping it.**

**Do not keep:** `rsi_mean_reversion` (no Indian evidence; 0/15 on the drifted
panel; the worst strategy measured here), `ema_trend_following`'s 50/200 pairing
(untested in India; measured **significantly negative** on the flat panel at
−0.953, the worst row in the whole table), `atr_breakout` (ATR has zero
evidence of directional predictive power anywhere — it is a sizing tool, and
`technical-indicators-nse.md` says so).

**Not a finding, and worth saying again:** none of these numbers is about
Indian equities. The panel that makes the strategies look good was built by
putting drift into it. The panel that removes the drift shows four of five
losing to holding twelve names equally, which is the repository's standing
finding reproduced one level down.

## Follow-ups (not done — deliberately out of scope for this file)

1. **Indicators from the concurrent modules.** Only the seven primitives that
   exist today were used. `indicators_trend.py`, `indicators_oscillators.py`,
   `indicators_volatility.py` and `indicators_volume.py` are in flight; a
   stochastic-oscillator mean-reversion rule, a Bollinger-band reversion rule,
   a Donchian-channel breakout replacing the hand-rolled 60-bar high, and an
   OBV/accumulation-distribution volume confirmation are all candidates once
   those modules land. **Do not import them until they exist**; a strategy that
   fails to import is worthless.
2. **Real vendor data.** Nothing here has touched an NSE CSV. Every claim above
   is about a generator.
3. **Mitra-style breakeven cost per rule.** `technical-indicators-nse.md` lists
   this as non-negotiable validation 1. The engine reports `total_cost` and
   `turnover`, so breakeven is computable without touching `run_portfolio`, but
   it is not computed on this page and no rule here has been shown to clear the
   ~0.3% one-way viability bar.
4. **Walk-forward.** Validation item 3 in the same file: select on 36 months,
   trade the next 36. Not done. The sweep tables above would not survive it
   unchanged.
5. **Overlapping positions.** `volatility_scaled_momentum` holds a single book
   at each rebalance; the 21-day rebalance interval rather than a drift trigger
   is the ceiling noted in `portfolio.py`'s own `# ponytail:` comment.
6. **The ATR strategy has no exit rule.** Noted in its `# ponytail:` comment:
   a failed breakout is held until the next monthly rebalance, which is the
   exposure Mitra's breakeven-cost table is most sensitive to. A stop in ATR
   multiples is a subtraction from the visible panel and costs no new data.
