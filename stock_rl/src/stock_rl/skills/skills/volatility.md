---
name: volatility
version: 1
enabled: true
evidence_tier: supported
data_sources:
  - nse_sec_bhavdata_full_archive
  - nse_eod_bhavcopy
kill_criteria:
  - id: sizing_only_no_direction
    statement: the feature may only scale position size
    threshold: any directional use is void
  - id: no_regime_label
    statement: conditioning must be on a volatility measure
    threshold: a trend or range label is void, per Han, Yang and Zhou (2013)
  - id: drawdown_reduction
    statement: the sized book must reduce maximum drawdown against equal weight
    threshold: no improvement is void, and no alpha is claimed either way
---

## Purpose

Scale position sizes by a volatility estimate. **Nothing else.**

This is the one skill in this directory that is supported as an input and
explicitly refused as a signal. It cannot say buy or sell, and a
directional use of it is a schema violation of this file's intent rather
than a tuning choice.

## Data sources

- NSE security bhavcopy archive, for a survivorship-free panel.
- NSE end-of-day bhavcopy as a gap filler.

Realised volatility from returns. No implied volatility is required for
the sizing use, which keeps the input free and available over the full
history.

## What it claims

That conditioning on volatility, rather than on a trend or range label,
is the one regime-conditional construction with published support, and
that it belongs in the sizing path.

## What the evidence actually says

Han, Yang and Zhou (2013), in *JFQA*, apply a trend-timing overlay to
portfolios **sorted on volatility**: moving-average timing works on
high-volatility portfolios and fails on low-volatility ones. That is a
filter on a sorted universe. It is not evidence for classifying the
market as trending or ranging, and it is the reason the second kill
criterion exists.

A design doc's alternative -- an average directional index used as a
trend/range classifier -- has no supporting evidence at all. No Indian
equity study was found for it, and treating it as a regime label is a
category error on top of an unsupported one.

Supporting facts that shape the sizing use rather than a signal:

- Volatility as a signal is a sizing tool. There is no evidence of
  directional predictive power, and the Indian VIX literature *forecasts*
  the VIX rather than using it to time equities, with the best reported
  directional accuracy close to a coin flip.
- Low volatility on its own is a supported directional factor, which is
  what `low_volatility` in this directory is for. Reusing that evidence
  to justify a directional volatility signal here would be double
  counting the same prior.
- Volatility spikes are a drawdown signal, not a return signal. A VIX
  in its top decile persisting for several sessions is the documented
  trigger for a tail overlay, which is a risk control.

## Kill criteria

1. **sizing_only_no_direction.** If this file's feature ever moves a
   sign, the skill is void and the change is a new skill with its own
   pre-registered criteria.
2. **no_regime_label.** Condition on a volatility number. A trend or
   range label is void per Han, Yang and Zhou (2013).
3. **drawdown_reduction.** The sized book must reduce maximum drawdown
   against equal weight. A null result is an acceptable outcome; an
   unexplained absence of effect is not.

## What this skill may not do

- May not claim alpha, in either direction. This file reports a risk
  input and nothing else, and no return figure is asserted anywhere in
  it.
- May not be used to justify a volatility-timing overlay on the index.
- May not be counted as a second independent signal alongside
  `low_volatility`; in Indian data the two share most of their prior.
