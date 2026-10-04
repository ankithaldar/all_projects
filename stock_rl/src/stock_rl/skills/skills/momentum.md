---
name: momentum
version: 1
enabled: true
evidence_tier: supported
data_sources:
  - nse_sec_bhavdata_full_archive
  - nse_eod_bhavcopy
kill_criteria:
  - id: beat_equal_weight
    statement: Diebold-Mariano on realised equity curves, this skill minus equal weight
    threshold: p <= 0.10 with |t| > 3.0, holding in at least 4 of 5 walk-forward folds
  - id: breakeven_cost
    statement: Mitra-style one-way breakeven cost of the rank rule
    threshold: at least 30 bps, against a delivery round trip of about 22 bps
  - id: turnover_cap
    statement: monthly one-sided turnover of the rebalanced book
    threshold: at most half of book value, per Novy-Marx and Velikov (2016)
  - id: delisted_names_present
    statement: the backtest panel must include delisted Nifty members
    threshold: survivor-only results are void
---

## Purpose

Rank the universe on trailing total return, hold the leaders, and report
nothing as an improvement unless it beats an equal-weight book over the
same names and the same dates.

This is the control arm rather than a contribution. It is the family the
research calls strongest in Indian data, and it is the bar every other
skill file in this directory has to clear before anyone is allowed to say
a new signal helped.

## Data sources

- NSE security bhavcopy archive, `sec_bhavdata_full`, which includes
  delisted names. Survivor-only panels inflate results, and a momentum
  backtest on survivors is measuring the index, not the effect.
- NSE end-of-day bhavcopy, used only where the archive has a gap.

Prices only. No fundamentals, no news, no vendor feed. That is the
single strongest argument for this skill: it has the best Indian evidence
*and* the fewest ways to be wrong about data.

## What it claims

That a cross-sectional rank on the trailing twelve-to-one month return,
rebalanced monthly and capped for concentration, is the strongest
directional signal available on Nifty-50 names from price data alone, and
that it must be treated as a priced risk factor rather than as free alpha.

The second half of the claim matters as much as the first.

## What the evidence actually says

Momentum is the strongest anomaly family in Indian data and the most
replicated. Maheshwari and Dhankar (2017), on 470 BSE stocks from 1997 to
2013, find it profitable after controlling for size, value and
illiquidity, and asymmetric: it is driven by the **winners**, so the long
leg works without shorting, which is the only leg an Indian delivery book
can run. A 2017 study of the five BRICS markets reports that the Indian
market has the strongest momentum effect of the five. Singh and
colleagues, on BSE monthly data from 1996 to 2020, find time-series
momentum beats cross-sectional and is driven by the net-long leg.
Barik and Balakrishnan (2022) confirm idiosyncratic volatility
significantly affects it.

Three honest caveats, all of which are reasons to cap rather than to
drop:

- **It may be a priced factor, not a mispricing.** Kedia and colleagues
  (2024) find the winner-minus-loser factor's alpha goes to roughly zero
  under a four-factor Carhart model.
- **It inverts in crises.** Maheshwari and Dhankar (2017) find momentum
  returns turn negative during crisis periods. This is a risk to disclose,
  not a flaw in the effect.
- **The spectacular Sharpe ratios in this literature are artefacts.**
  Reported figures of 6 to 9 on winner-minus-loser long-short come from
  markets where shorting was banned for years and remains restricted, and
  they are not harvestable by a long-only delivery portfolio.

Technical rules in general die out of sample. Rink (2023) tested 6,406
rules with a stepwise Superior Predictive Ability test and found best-
in-sample rules underperform buy-and-hold out of sample once costs are
charged. India is not in his sample, and his own framing is that
profitability decays with competition, which argues for the slow,
low-turnover, long-horizon version of momentum rather than for abandoning
it.

## Kill criteria

Pre-registered above, and pre-registered is the operative word: these
numbers were written into the file before the experiment ran, so they
cannot be moved afterwards to rescue a result.

1. **beat_equal_weight.** A p-value at or below the conventional
   threshold with a t-statistic above 3, which is Harvey, Liu and Zhu's
   (2016) bar rather than the usual two, holding in most of the
   walk-forward folds. Harvey, Liu and Zhu catalogued more than 316
   claimed predictors and find 158 of 296 published significant factors
   are false discoveries, so the conventional threshold is not
   defensible here.
2. **breakeven_cost.** The delivery round trip is about 22 bps, so a
   rule with a breakeven below roughly 30 bps is dead on arrival before
   any other consideration.
3. **turnover_cap.** Monthly one-sided turnover above half of book value
   is fatal regardless of backtest return, per Novy-Marx and Velikov
   (2016).
4. **delisted_names_present.** A survivor-only panel is void, not
   optimistic.

## What this skill may not do

- May not be described as producing alpha without a CAPM or Carhart
  alpha and its p-value reported.
- May not be scaled by a volatility estimate from this same price series
  and called two independent bets.
- May not be re-tuned on the evaluation period. Rink's design selects on
  36 months and trades the next 36; if the out-of-sample result beats the
  in-sample one, the bug is in the harness.
