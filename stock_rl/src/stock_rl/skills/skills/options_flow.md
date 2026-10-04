---
name: options_flow
version: 1
enabled: true
evidence_tier: experimental
data_sources:
  - nse_fno_bhavcopy
  - nse_participant_wise_open_interest
  - nse_fii_dii_activity
kill_criteria:
  - id: beat_price_only
    statement: Diebold-Mariano on realised equity curves, context arm minus price arm
    threshold: p <= 0.10 with |t| > 3.0, holding in at least 4 of 5 walk-forward folds
  - id: turnover_cap
    statement: monthly one-sided turnover of the book
    threshold: at most half of book value
  - id: survives_expiry_contamination_check
    statement: the signal must survive removal of the day after an expiry
    threshold: result must hold on non-expiry sessions
---

## Purpose

Read what large participants actually did in the derivatives market, and
test whether that positioning adds predictive value beyond price on the
same names.

This is the one skill in this directory with a genuine
**informational-asymmetry** prior, and it is worth being precise about
why, because the reason is not "options are clever". It is that options
positioning is a record of trades by participants with capital at risk,
whereas a news sentiment score is a record of what the public already
knows. Ben-Rephael, Da and Israelsen (2017) make the same distinction for
attention: what matters is not how much coverage a name has but whether
the attention carries information.

It is experimental because a strong prior is not evidence. Nothing here
establishes that Indian options positioning predicts Indian equity
returns, and this file makes no claim that it does.

## Data sources

- NSE F&O bhavcopy: open interest, traded volume and implied volatility
  per strike, free daily.
- NSE participant-wise open interest: the client, foreign-institutional
  and proprietary split, which is where the "large participants did this"
  claim actually comes from.
- NSE FII/DII activity, for the participant classification only.

All free. History runs from roughly 2016 to 2020 depending on the file,
which is the binding constraint: context features have far less history
than price features, and a policy trained on eight years of price and
three of context is fitting a higher-dimensional state from a shorter
sample.

## What it claims

That the change in open interest, its skew across strikes and the
participant split carry information about the next few days of the
underlying that price history alone does not.

It does **not** claim any measured effect size, and there is none in this
file to quote.

## What the evidence actually says

The evidence is a mechanism argument, not a measurement.

- **Free and daily.** The bhavcopy carries open interest and implied
  volatility per strike with no licence cost, which is unusual for a
  differentiated signal and is the practical reason to try this first.
- **It is per-stock.** The research makes this the decisive point. The
  FII/DII aggregate is one India-wide number and therefore carries no
  cross-sectional information at all; options positioning is genuinely
  per-stock, genuinely differentiated, and genuinely hard to obtain.
- **Contamination is the main data risk.** Open interest is published
  once per settlement cycle, is contaminated by the previous expiry's
  expiry-day activity, and in Nifty is dominated by a handful of
  institutional strikes. That is why the third kill criterion exists.
- **No published Indian study tests options positioning as an equity
  return predictor.** Absence of evidence, reported as such.

**A correction to the design docs, recorded here so it is not
reintroduced.** The design puts put-call ratio and open interest into the
hedge environment's state vector. No paper in that literature uses them
there. Open interest appears only in old ANN-pricing work as a weak
feature, and the hedging papers' state inputs are the underlying level,
the Black-Scholes delta of the liability at current implied volatility,
time to expiry and hedge inventory. Adding put-call ratio and open
interest to a hedging state adds state dimension, measurement noise and
look-ahead risk, and earns none of them back.

This file therefore keeps those two quantities **out of the hedge state**
and proposes them as a cross-sectional equity context feature, which is
the use the asymmetry argument actually supports. That is a different
claim from the one the design doc made, and it needs its own evidence.

## Kill criteria

1. **beat_price_only.** The pre-registered comparison: price-only arm
   versus price plus this feature, Diebold-Mariano on realised equity
   curves, with a t-statistic above 3 in most folds. Harvey, Liu and Zhu
   (2016) require the stricter bar for exactly this situation.
2. **turnover_cap.** A daily positioning signal generates high turnover
   by construction, and Novy-Marx and Velikov (2016) find only anomalies
   with monthly one-sided turnover below their threshold survive costs.
3. **survives_expiry_contamination_check.** If the result depends on
   sessions adjacent to an expiry, it is measuring expiry mechanics, not
   information.

The overarching rule, from the research: if this arm does not clear the
price-only arm, **delete the numeric context work rather than promoting
it to a language-model stage.**

## What this skill may not do

- May not be re-added to the hedging state vector. See the correction
  above; it was measured, not assumed.
- May not be described as validated. It has a prior and a plan, not a
  result.
- May not run live. It is experimental, and the registry refuses to
  activate it outside experiment mode whatever flags are passed.
