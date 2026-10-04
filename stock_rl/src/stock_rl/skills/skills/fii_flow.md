---
name: fii_flow
version: 1
enabled: true
evidence_tier: rejected
data_sources:
  - nse_fii_dii_activity
  - nsdl_fpi_trends
kill_criteria:
  - id: cross_sectional_dispersion
    statement: the feature must vary across the 50-name cross-section
    threshold: zero variance across names is unpassable by construction
  - id: sign_stability
    statement: the flow-to-next-day-return relationship must hold out of sample
    threshold: p <= 0.10 with |t| > 3.0 in at least 4 of 5 walk-forward folds
---

## Purpose

None. This file exists to record a structural impossibility, so that the
idea is not proposed again in six months by somebody who has not read the
research.

## Data sources

- NSE FII/DII activity: aggregate daily buy, sell and net, compiled from
  NSDL PANs, free, with a historical archive.
- NSDL FPI trends: free daily sector-level FPI equity and debt flows,
  often finer-grained than the NSE aggregate.

Both are genuinely free and both are genuinely useful. The problem is what
they can be attached to.

## What it claims

Nothing. A skill with an evidence tier of `rejected` makes no claim. The
registry will refuse to activate it under any flag combination, and the
reason it returns says so.

## What the evidence actually says

The FII/DII aggregate is **one number for all of India**.

Handing it to a per-stock agent across a 50-name cross-section produces
fifty duplicate values. There is no cross-sectional dispersion, so there
is nothing to rank on, nothing to sort by, and no way for a
cross-sectional learner to use it as a per-name feature at all. It is a
regime indicator wearing the costume of a stock feature, and any measured
effect would be a bet on the index conditional on the index.

This is not a data problem and no better vendor fixes it. The finest free
source, NSDL sector-level FPI flow, is sector-level: it discriminates
across about a dozen sectors, which is a coarse cross-section, and it
still cannot discriminate across the several Nifty-50 names inside one
sector.

Where it belongs is exactly where the research puts it: **one global
state variable**, alongside the other India-wide context scalars, used to
condition the regime rather than to tilt a name.

## Why this is rejected

Rejected as a **per-stock feature**, for a structural reason rather than
an empirical one, which is why no experiment can rescue it:

1. Zero cross-sectional dispersion by construction. Fifty names, one
   number.
2. The finer free source is sector-level, not name-level, so it cannot be
   promoted to a per-name feature either.
3. The design doc's framing of a "FII Flow Agent" per ticker is
   structurally incoherent, and building it would produce a plausible
   looking result that is entirely a bet on market direction.

Rejected is not the same as useless. The same series, used once as a
single global state variable in the context arm, is a legitimate input.
That is a different feature with a different name, and it belongs in the
context arm of the minimal falsifiable experiment rather than in a
per-stock agent.

## Kill criteria

1. **cross_sectional_dispersion.** The feature must vary across the
   cross-section. It cannot. This criterion is unpassable by
   construction and that is the point of writing it down.
2. **sign_stability.** Recorded for completeness, and deliberately
   recorded as though it might one day be evaluated. It cannot be, for
   criterion one. A file with no pre-registered criteria can never be
   stopped, and this one is stopped before it starts.

## What this skill may not do

- May not run, ever, under any flag combination. That is enforced in
  `stock_rl.skills.registry`, not merely asserted here.
- May not be implemented as a per-stock feature in the state vector.
- May not be reintroduced as a per-name feature with a sector
  interaction, since sector interaction reproduces the same coarse
  cross-section.
