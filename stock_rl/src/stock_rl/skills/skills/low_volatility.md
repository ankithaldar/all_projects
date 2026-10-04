---
name: low_volatility
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
  - id: real_vol_sort_not_beta_sort
    statement: the book must be sorted on realised volatility, not on beta
    threshold: beta-sorted results are void, per the 2021 Indian evidence
  - id: turnover_cap
    statement: monthly one-sided turnover of the reweighted book
    threshold: at most half of book value
  - id: delisted_names_present
    statement: the backtest panel must include delisted Nifty members
    threshold: survivor-only results are void
---

## Purpose

Weight each name inversely to its realised volatility over a rolling
window, so the book carries less of each name's own risk than the
market does. Rank on the standard deviation of returns, **not** on beta.

This is the second-best evidenced family in Indian equities and it needs
nothing but price data, which makes it the cheapest useful thing in this
directory to maintain.

## Data sources

- NSE security bhavcopy archive for a survivorship-free panel.
- NSE end-of-day bhavcopy as a gap filler.

Price only. No earnings, no leverage ratio, no vendor screen.

## What it claims

That a volatility-sorted, inverse-volatility-weighted Nifty-50 book has
positive expected excess return in India over long horizons, that the
effect is not merely a small-cap or a value effect in disguise, and that
it **grows** rather than decays.

The non-decay claim is the one that distinguishes the Indian result from
the American one, and it is the reason this file is supported rather than
experimental.

## What the evidence actually says

Agarwalla, Jacob and Varma (IIMA working paper 2014-07-01) find betas
against betas earn significant positive returns in India and dominate
size, value and momentum. A study of 4,400 companies over 19 years
(SSRN 4398656) reports a strong convex return-volatility relationship
that does not diminish over time and survives factor controls.
Joshipura and Peswani (2019), on NSE data from 1997 to 2018, report a
low-volatility quintile alpha spread of 25.53 percent per year against
the high-volatility quintile, with the highest risk-adjusted returns in
**large** caps -- which is where this universe lives.

**Sort on realised volatility, not on beta.** A separate study reports a
low-volatility excess return of 10.62 percent on a standard-deviation
sort against 4.82 percent on a beta sort. That is the operational reason
for the second kill criterion.

One tension to disclose rather than bury. Peswani and Joshipura (2021)
conclude that leverage constraints predominantly explain the Indian
low-risk effect and therefore argue for measuring risk with beta. The two
Indian findings disagree. This file follows the standard-deviation sort
because it is the one with the larger measured effect and the lower data
requirement, and records the disagreement here so a reviewer does not
have to rediscover it.

Two further caveats:

- **In India low-volatility and quality are partly the same signal.**
  Low-volatility stocks have higher profitability, so building a
  low-volatility book and a quality book is two bets on overlapping
  information, not three independent ones.
- **The effect carries a growth tilt in India**, the opposite of the
  value tilt it has in the United States. Do not describe it as a value
  proxy here.

## Kill criteria

1. **beat_equal_weight.** p at or below the conventional threshold with
   a t-statistic above 3, the Harvey, Liu and Zhu (2016) bar, holding in
   most walk-forward folds.
2. **real_vol_sort_not_beta_sort.** A beta-sorted result is not a result
   for this skill. The two sorts are different portfolios and the
   2019-versus-2021 disagreement above means the distinction is not a
   detail.
3. **turnover_cap.** Inverse-volatility reweighting drifts; if monthly
   one-sided turnover exceeds half of book value the drift is costing
   more than the effect earns.
4. **delisted_names_present.** A survivor-only panel is void.

## What this skill may not do

- May not be run a second time with a different window and reported as a
  confirmation. One window, chosen now, or the criterion is void.
- May not be described as independent of a quality or momentum book in
  this package, because in Indian data it is partly the same signal.
- May not be given an expected return figure. None is claimed here and
  none should be quoted from this file.
