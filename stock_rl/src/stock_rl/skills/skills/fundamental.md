---
name: fundamental
version: 1
enabled: true
evidence_tier: rejected
data_sources:
  - nse_integrated_filing_results
  - nse_xbrl_documents
  - screener_in_screen
kill_criteria:
  - id: point_in_time_depth
    statement: the panel must carry ten or more years of as-filed fundamentals
    threshold: about seven quarters exist for free
  - id: licence_permits_persistent_store
    statement: the source's terms must permit copying into a persistent store
    threshold: screener terms prohibit modify or copy
  - id: factor_premium_still_present
    statement: the value spread must still exist post-GFC
    threshold: a Chow test must reject a structural break
---

## Purpose

None as a backtested feature. This file records why a free-data
fundamental backtest cannot be built honestly, and it separates that from
the separate, worthwhile job of building a point-in-time store for live
use.

## Data sources

- NSE integrated filing results, which carry a `broadcast_date` per
  filing. That timestamp is the point-in-time anchor, and it is the most
  valuable free artefact in this directory.
- The XBRL documents themselves, which are Ind AS and carry the
  financials and the availability metadata in one file, including
  board-approval date and restatement flags.
- Screener.in, listed here only so its exclusion is on the record.

## What it claims

Nothing. The tier is `rejected` and the registry refuses to activate it
under any flag combination.

## What the evidence actually says

Four independently disqualifying findings, which is why this is rejected
rather than experimental.

**1. Free Indian fundamental data is not point-in-time.** Screener.in
serves current values, meaning the last restatement for each period. A
current-value row can never be un-leaked, so any fundamental backtest
built on it is retroactively contaminated and you cannot tell by looking
at the results whether it is. This is not a precision problem to be
handled with a lag; the information is absent from the data.

**2. The licence forbids the thing that would be required.** Screener's
terms grant personal, non-commercial transitory viewing and prohibit
copying or modifying the materials. Building a persistent database from
it is the prohibited act, its robots rules disallow the machine-readable
paths, and the free tier allows roughly fifty anonymous page views a day
against a backfill that needs far more.

**3. History depth is about seven quarters.** The integrated filings API
returns a total count of 12 filings for the largest Nifty names with the
oldest period ending in March 2025, and the structured XBRL flags cover
2025 onward. Going back further means parsing roughly 14 years of PDF
financial results with material extraction risk. A point-in-time
fundamental panel cannot be constructed from free sources.

**4. Even a clean panel would not support the two factors most people
want.** The value premium decayed to almost nothing after the financial
crisis: the price-to-book long-short spread went from 2.36 percent
before the crisis to 0.12 percent after it, with a Chow test confirming a
structural break, and in value-weighted samples the cheapest decile beat
the dearest, so the anomaly reversed. Accruals have the **wrong sign**
in India. Sehgal and colleagues (2012) find accruals positively
associated with future returns, meaning Indian investors underprice
accruals and overprice cash flows, and Pincus, Rajgopal and
Venkatachalam report an India loading consistent with that reversal. The
US accrual factor, imported unchanged, would be a bet on the wrong sign.

**Survivorship bias is the same size as the effect.** Measured on the
universe these papers use, survivor-only backtests inflate annual returns
by 4.94 percentage points. So a fundamental backtest on a
survivor-only free panel is dominated by an artifact comparable to the
thing it is trying to measure.

## Why this is rejected

Rejected **as a backtestable feature**, on four independent grounds where
any one of them would be enough:

1. Restatement look-ahead is unfixable on the free sources.
2. The free source's terms prohibit the persistent store the backtest
   needs.
3. The history is roughly seven quarters of clean structured data.
4. The value premium has decayed to near zero and the accrual factor has
   the wrong sign, so there is little to measure even if the data were
   clean.

**This does not mean "never build fundamental data".** The opposite is
true, and it is the highest-value engineering decision available: build a
point-in-time fundamental layer **for live use**, backfilled from late
2024, with every row carrying its broadcast date as the availability
timestamp, joined on that date and never on the period-end date, with
restatements stored as versions rather than overwritten, and every filing
retained from day one so a 2030 backtest is possible. That is cheap now
and impossible to reconstruct later.

It is also not a signal this package can backtest, and that is the
distinction the tier records.

## Kill criteria

1. **point_in_time_depth.** Ten or more years of as-filed fundamentals
   are required. Roughly seven quarters exist for free, so this cannot be
   met without a licensed vendor.
2. **licence_permits_persistent_store.** Screener's terms prohibit the
   copy that the backtest requires.
3. **factor_premium_still_present.** The value premium is gone
   post-crisis and the accrual sign is reversed, so there is no premium
   to test.

## What this skill may not do

- May not run, ever, under any flag combination.
- May not be built on Screener or any current-values source, at any
  depth, for any horizon.
- May not import the US accrual factor. The Indian sign is reversed.
- May not be treated as blocking the point-in-time live store, which is
  the opposite of what this file recommends.
