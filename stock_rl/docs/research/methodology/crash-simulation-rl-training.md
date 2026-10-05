# Simulating Crashes for RL Training

Should `stock_rl` build a crash simulator to train against, and if so
which of the six families of method is worth the money? The short
answer is **no simulator yet, but one cheap and one moderately cheap
intervention now**, and the reason is a defect in our own reward that no
simulator can repair from the outside.

Method note. `websearch` worked for this review. SEBI, NSE, BIS, the
Federal Reserve, `arxiv.org`, the Duke economics mirror and the Bank of
England working-paper series were all reachable, and most of this file's
Indian facts come from primary regulatory pages rather than from
secondary summaries. Two access failures are recorded honestly: DR4DRL
could not be fetched as a PDF on the first attempt (its abstract and
§4.2.3 were recovered through a text proxy later, so it is cited from
those and *not* from its results tables), and Springer / SAGE / Emerald /
ScienceDirect full texts were unreachable throughout. Everything cited
below is either a full text I read, a primary page, or explicitly
labelled abstract-only. Read
[corrections-to-design-docs.md](../corrections-to-design-docs.md) first:
several claims in the design docs about market structure are fabricated
and are not reused here.

## Executive summary

1. **We should not build a crash simulator yet, and the reason is
   internal.** `portfolio_env._reward` scores a running Sharpe, which is
   a *ratio*, hence scale-invariant. Measured over 200 seeds, halving
   every return changed the Sharpe term by exactly `+0.000000`. A policy
   that permanently halves its exposure is paid nothing for surviving.
   Simulating crashes does not fix this; it makes it worse, because a
   better crash simulator produces more crash bars that the reward does
   not want avoided.
2. **The de-risking penalty is worse than that.** On a Nifty-like series
   ending in the real 23-Mar-2020 bar of **−12.98%**, halving the book
   for the crash bar *raises* the Sharpe term by `+0.0336` and costs
   `0.2500` in turnover — a **7.4x net penalty** for halving your loss.
   Across a whole 301-bar episode the fully-invested policy scores
   `+8.5823` against `+8.3506` for the de-risking policy: **the crash
   avoider loses 0.23 reward units for taking half the loss.**
3. **Selective de-risking is a second, separate pathology.** On a
   negative-mean series, cutting to zero on only the 5 worst bars lifts
   the final running Sharpe by `+0.0597` against a turnover charge of
   `1.0000` — a **16.8x net penalty**. Both the do-nothing and the
   clever policy are penalised, and the penalty grows with how targeted
   the avoidance is.
4. **The cost term is irrelevant to this, and that is the actionable
   finding.** Real `DELIVERY` round-trip cost on a half-book is
   `0.001614` of capital. The turnover term charges `0.2500` for the same
   trade — **154.9x larger**. Any crisis-cost modelling added to
   `costs.py` moves the reward in the fourth decimal. Only the turnover
   weight moves it in the first. **Fix the reward scale before you fix
   the market.**
5. **The Indian crash geometry is known and thin.** Since the 2-Jul-2001
   circuit-breaker regime, the index-wide 10% trip has fired roughly a
   handful of times: 17-May-2004, 21-Jan-2008, 13-Mar-2020 and
   23-Mar-2020 are the documented downside cases. On 23-Mar-2020 Nifty
   fell **12.98%** and Sensex **13.15%** after a 45-minute halt, having
   tripped 10% at 09:58. The single largest Nifty drawdown on record is
   **−55.1%** peak-to-trough over 11 months (Dec 2007 to Nov 2008), with
   recovery taking 59 months. There is no dataset of Indian crashes. A
   simulator is not filling a gap here; it is manufacturing a
   distribution we cannot validate.
6. **Two regimes changed under us mid-review and both matter.** SEBI's
   Closing Auction Session went live **3 Aug 2026**, replacing the
   30-minute VWAP close for F&O stocks; SEBI's September 2026
   consultation paper proposes cutting CAS to 10 minutes and moving
   settlement to a blended VWAP. The 13-Aug-2026 Sensex CAS close of
   **78,079.96** came off a 15:15 reference of 77,829.60 — a 113.61-point
   recovery inside the auction window. Any backtest spanning Aug 2026
   mixes two close definitions, and **no post-CAS Indian evidence
   exists yet**.
7. **Survivorship bias compounds with the simulator rather than adding to
   it.** Jain (2026) measures **+0.8 to +3.3 pp/yr** of phantom return
   across just two universe vintages — a 4x spread on the same market.
   A block bootstrap that resamples today's surviving names cannot
   produce a crash in a name that was delisted into it. The bias is not
   additive with the simulator's error; it is multiplicative at exactly
   the tail we care about.
8. **Of the six method families, exactly one has Indian evidence, and it
   is the weakest of the six for our purpose.** Regime switching has
   Indian estimates (Ahmad & Kamaiah 2011: bear regime `mu = −0.19%/day`,
   persistence `p11 = 0.95`). Block bootstrap, FHS, EVT and stress
   testing have *some* Indian tail work but none on RL training. Domain
   randomisation and multi-agent simulation have **no Indian evidence
   at all** — the one DR paper we found is Dow Jones, DDPG, abstract-only
   on results.

## The six families, ranked by what they can actually do for us

| Family | Preserves | Indian evidence | Evidence for RL training | Verdict |
|---|---|---|---|---|
| Historical block bootstrap | Marginal + local dependence | None found | None found | **Baseline, cheap** |
| Filtered historical simulation + GARCH | Conditional vol scaling | Strong (Nifty GARCH-EVT) | None | **Viable, moderate** |
| EVT / POT tail replacement | Tail shape | Strong (Nifty, 2005 & 2026) | None | **Layer on FHS** |
| Scenario / stress (historical, hypothetical, reverse) | Severity × duration jointly | None found for policy training | None | **Reverse stress only** |
| Domain randomisation | Parameter robustness | **None** | Abstract-only, DJIA | **Defer** |
| Regime switching / HMM | Persistence of state | Moderate | None | **Cheapest first win** |

Ranked by cost-to-build and evidence-per-rupee, the order is: regime
switching, then block bootstrap, then FHS+EVT, then reverse stress, then
domain randomisation. Everything after FHS is unjustified today.

## Finding 1 — The reward cannot express crash avoidance, and we can
prove it numerically

This is the load-bearing section. Everything else in this file is
downstream of it.

### 1.1 The Sharpe term is scale-invariant

`portfolio_env._reward` (`src/stock_rl/rl/portfolio_env.py:454-471`)
computes

```
running_mean = fmean(self._returns)
running_vol  = stdev(self._returns)
sharpe       = 0.0 if running_vol <= 0.0 else running_mean / running_vol
```

Mean and standard deviation are both homogeneous of degree 1 in the
returns. The ratio is homogeneous of degree **0**. Scaling the entire
return series by `k` leaves the term exactly unchanged:

```
max |sharpe(x) - sharpe(0.5x)| over 200 draws: 0.000000
```

This is arithmetic, not a modelling choice. **The reward assigns exactly
the same value to a book that runs at half exposure forever as to a book
that runs fully invested.** A crash simulator that produces more −13%
bars cannot fix this, because the extra bars the agent avoids are worth
zero and the extra bars it does not avoid are worth the same as any
other.

### 1.2 And the avoidance is actively charged

`AllocationReward.turnover = 0.5` and the reward subtracts
`0.5 * traded_weight` every bar. Using the real crash bar — Nifty
**−12.98%** on 23-Mar-2020, corroborated by Reuters and the Indian
Express — and a calm run-up of 300 bars at `mu = 0.04%/day`,
`sd = 1.1%/day`:

```
sharpe term, last bar, w = 1.0 : +0.0942
sharpe term, last bar, w = 0.5 : +0.1278
  -> halving exposure changes the Sharpe term by +0.0336
  -> and costs a turnover charge of 0.2500
```

Halving your loss on the single worst day in Nifty's modern history is
worth `+0.0336` and costs `0.2500`. **Net: −0.216 per crash bar.** Over
a full 301-bar episode, 200 seeds:

```
A total reward (fully invested):        mean +8.5823
B total reward (half book into crash): mean +8.3506
delta from de-risking (whole book):     -0.2316
```

The policy that takes half the loss scores **0.23 lower**. This is not a
tie that could be broken by better features. It is a gradient pointing
the wrong way.

### 1.3 Targeted avoidance is penalised harder

A policy that only de-risks on the worst bars is smarter, and it is
punished more. On a Nifty-like series with negative drift
(`mu = −0.06%/day`, `sd = 1.2%/day`, 200 bars), cutting to zero exposure
on the 5 worst days:

```
always-invested   final running Sharpe: mean -0.0494  sd 0.0740
crash-avoider    final running Sharpe: mean +0.0102  sd 0.0777
effect of avoiding 5 worst bars: +0.0597
turnover charged: 1.0000
```

Avoiding the five worst days moves the Sharpe term by `+0.0597` and
costs `1.0000` in turnover. **Net: −0.94.** The smarter the avoidance,
the worse the reward. This is a distinct pathology from 1.2 and it needs
a distinct fix: it is not that the reward dislikes de-risking, it is
that the reward's *only* exposure-sensitive channel is a linear
transaction charge that is one to two orders of magnitude too large
relative to the signal it is supposed to mediate.

### 1.4 The cost term is 155x too small to matter

The repo's own `DELIVERY` model, on the repo's own default `Rs 1 crore`
capital:

```
round trip on Rs 1 crore: Rs 32,263.46 = 0.3226% of capital
round trip on half book (Rs 50 lakh): Rs 16,139.40 = 0.1614%
reward.cost = 1.0 charge on a half-book round trip: 0.001614
reward.turnover = 0.5 charge on a 0.5 weight move:   0.250000
ratio turnover-charge / cost-charge: 154.9x
```

So `costs.py` is not the lever. A perfect crisis cost model inside
`CostModel` — spreads tripling, impact spiking, STT and stamp duty
unchanged because they are statutory — moves the reward by about `0.0005`.
Meanwhile the abstract turnover weight moves it by `0.25`. **Before any
crash simulator, rescale the reward so the exposure-sensitive term is
comparable in magnitude to the return signal.** That is a two-line change
to `AllocationReward` and it is the prerequisite for everything below.

### 1.5 The `portfolio_env.py:29` docstring is not the reason

`portfolio_env.py:29` says Zernikov's default reward *"collapses to a
constant on any non-negative P&L"*. That claim describes **Zernikov's**
reward, not ours. Our `AllocationReward.drawdown` defaults to `0.0`
(line 99), so the drawdown channel is off and the literal collapse does
not apply here.

**Verdict on "is crash training moot?":** no, with low-to-moderate
confidence. The honest reading is **degraded, not impossible**. Training
is not moot — but the reward currently **actively mis-signals** the
exact decision crash training exists to teach. A crash simulator built
today would train against a reward that pays nothing for surviving and
charges for trying.

## Finding 2 — Indian crash episodes: what actually happened

Retrieved from SEBI and NSE primary pages plus contemporaneous
reporting. Every date below is a documented index-wide circuit-breaker
event, not an inferred one.

### 2.1 The mechanism, exactly

SEBI `SMDRPD/Policy/Cir-37/2001`, 28 June 2001, effective **2 July 2001**:

- Three stages, **10% / 15% / 20%**, either direction.
- Triggered by **whichever of BSE Sensex or NSE CNX Nifty breaches
  first**.
- 10% before 1pm: 1 hour halt. 10% from 1pm to 2:30pm: 30 minutes.
  10% after 2:30pm: no halt. 15% before 1pm: 2 hours. 15% from 1pm to
  2pm: 1 hour. 15% after 2pm: rest of day. 20%: rest of day.

Modified by `CIR/MRD/DP/25/2013` (3 Sept 2013): the limits are now
**recomputed daily off the previous close**, rather than quarterly in
absolute points, and a **15-minute pre-open call auction** is inserted
after each halt, reducing the halt itself by 15 minutes. NSE implemented
via circular **85/2013, 11 Oct 2013**.

The *post-2013* durations, which are the ones that matter for anything
recent:

| Trigger | Time | Halt | Pre-open auction |
|---|---|---|---|
| 10% | before 1:00 pm | 45 min | 15 min |
| 10% | 1:00pm–2:30pm | 15 min | 15 min |
| 10% | after 2:30pm | none | n/a |
| 15% | before 1:00 pm | 1h 45min | 15 min |
| 15% | 1:00pm–2:00pm | 45 min | 15 min |
| 15% | on/after 2:00pm | rest of day | n/a |
| 20% | any time | rest of day | n/a |

Two features of this table are load-bearing for a simulator and neither
is in our code. First, **the halt is time-of-day dependent**: a 10% move
at 09:20 costs 45 minutes, the same 10% move at 15:00 costs nothing.
Second, **a 10% trip does not stop the day** — on 23-Mar-2020 the market
halted, reopened, and then fell a further ~3% to its worst close ever.

### 2.2 The documented events

| Date | Index | Move | CB | Detail |
|---|---|---|---|---|
| 17 May 2004 | Sensex | ~11% intraday | tripped | Post-election; first use |
| 17 Oct 2007 | — | — | tripped | Participatory-note tightening |
| 21 Jan 2008 | Sensex | −1,408 pts to 17,605 | intraday halt | "Black Monday"; BSE technical snag 2:30pm |
| 18 May 2009 | Sensex | — | **two upper trips in one day** | UPA victory |
| 13 Mar 2020 | Nifty | **−10.07%** (−966 pts) | **10%, 09:20, 45-min halt** | First halt in 12 years; closed **+3.81%** |
| 23 Mar 2020 | Nifty | **−12.98%** (−1,135.20 pts to 7,610.25) | **10%, 09:58, 45-min halt** | Worst day in index history |

Sensex on 23-Mar-2020: **−13.15%** (−3,934.72 pts to 25,981.24).
Sector damage on the same day ran to **−18.40%** for banking and
**−16.1%** for capital goods. India VIX closed **71.56**, up 6.64%.

Two observations that matter more than the numbers:

**The 13-Mar-2020 day closed positive (+3.81%) after tripping the
breaker.** The halt, the pre-open auction, and the calm afternoon
together turned a −10.07% morning into a green day. A simulator that
models a CB day as "a −10% bar" would get this backwards.

**The halts are 45 minutes of no trading, not a −10% print.** Our bar
data has no representation of a halt. `grep` over `src/stock_rl/` finds
no halt or circuit-breaker logic anywhere under `rl/`. A policy that
decides to de-risk at 09:20 in a real 2020 would have discovered it
could not trade for 45 minutes.

### 2.3 Drawdown magnitudes, which are what actually set portfolio scale

| Episode | Peak→trough | Duration | Recovery |
|---|---|---|---|
| 2008 GFC (Dec 2007 → Nov 2008) | **−55.1%** | 11 months | **59 months** |
| COVID (Dec 2019 → Mar 2020) | **−29.3%** | 3 months | 8 months |
| 2015–16 (Feb 2015 → Feb 2016) | −21.5% | 12 months | 13 months |
| 2004 (Dec 2003 → May 2004) | −21.1% | 5 months | 6 months |

Intra-daily drawdowns were **deeper** than these month-end figures — the
2008 low was −59.9% from the peak on an intra-day basis. The
2008-to-2013 recovery is the single most important number in this table
and it is not reproduced by any of the generators below: a −55% move
followed by 59 months of grinding recovery is a *path*, and generators
that produce the magnitude without the duration will produce policies
that re-risk into the recovery.

## Finding 3 — Family 1: historical block bootstrap

**What it is.** Resample contiguous blocks of historical returns instead
of individual days, so that whatever dependence lives inside a block —
volatility clustering, momentum, cross-asset correlation on the same day
— is carried into the resample intact.

**Block length is the whole decision.** Politis & Romano (1994,
*JASA* 89(428)) introduce the stationary bootstrap: blocks of
*geometrically distributed* length, which keeps the resampled series
stationary where a fixed-length moving-block bootstrap does not. The
choice of the mean block length `b` is left open, and Politis & White
(2004, *Econometric Reviews* 23(1) 53–70) supply the standard automatic
selector: estimate the optimal expected block size for the stationary
bootstrap from the flat-top lag-window spectral estimator,

```
b*_SB = (2 G^2_DSB)^(1/3) N^(1/3)
```

with Patton's MATLAB implementation. There is also a **2009 correction**
to the paper (Patton, Politis & White, *Econometric Reviews* 28(4)
372–375), which matters if you implement the estimator from the 2004
paper alone.

**The honest caveat, and it is the important one.** `b*` is derived from
the **correlogram of the returns themselves**. Daily equity returns
have near-zero linear autocorrelation but strongly dependent absolute and
squared values. A selector reading the raw-return correlogram can
therefore select a short block *even when volatility clustering is
pronounced*. For a crash simulator that is the wrong thing to optimise.
Two honest options: run the selector on `|r|` or `r^2` instead of `r`,
or declare a block length in trading days and report it. **Do not present
a Politis-White `b*` computed on raw Nifty returns as evidence that
clustering is captured.**

**What it cannot do.** History bounds the tail by construction. A block
bootstrap over Nifty's daily history cannot produce a day worse than
23-Mar-2020's −12.98%, and cannot produce a 2008-style 11-month −55%
descent unless a contiguous stretch of it happens to be sampled. Every
tail number it reports is an interpolation within the observed range,
not an extrapolation.

**Indian evidence: none.** No Indian study found that applies block
bootstrap to Nifty crash simulation. The nearest Indian work bootstraps
*Tehran Stock Exchange* data for portfolio scenario optimisation
(PLOS One, 2026), which is a different market and a different objective.

**Verdict: build it, as the cheapest baseline, and be explicit about
its ceiling.** It is the only family that preserves real cross-sectional
rows — correlation, coskewness, tail dependence — with no distributional
assumption at all. Everything else either approximates that structure or
invents it. Its ceiling is the historical maximum loss.

## Finding 4 — Family 2: filtered historical simulation, GARCH, EVT

### 4.1 FHS

**What it is.** Two steps. **Devol**: divide each historical return by a
volatility estimate *for that day*. **Revol**: multiply the resulting
residuals by a volatility estimate *for the day you are forecasting*.
Hull & White (1989) and Barone-Adesi, Giannopoulos & Vosper (1999) are
the canonical references; Barone-Adesi et al. (2002, *EFM* 8(1) 31–58)
is the version with a full backtest.

**Why it beats plain historical simulation.** The Fed's own FEDS note on
the hidden dangers of historical simulation (FEDS 2001-27) puts it
precisely: historical simulation models *"are both under-responsive to
changes in conditional risk; and respond to changes in risk in an
asymmetric fashion: measured risk increases when the portfolio
experiences large losses, but not when it earns large gains."* FHS
fixes the responsiveness half. It also notes FHS *"requires additional
refinement to account for time-varying correlations; and to choose the
appropriate length of historical sample period"* and that **2 years of
daily data may not contain enough extreme outliers for 1% VaR at a
10-day horizon**. That last point is decisive for us: a 2-year Nifty
window contains at most one 23-Mar-2020.

**It is procyclical, and the Fed still uses it.** The BoE working paper
on FHS and its competitors (Gurrola-Perez & Murphy 2015) is explicit
that FHS's procyclicality is a known design property, not a defect.
Meanwhile the Fed's *2025 Supervisory Stress Test Methodology*
retains a **historical simulation model** for projecting
operational-risk losses. The lineage survived the crisis.

**Indian evidence: it exists and it disagrees with plain HS.** An IJETT
2018 study of five Nifty sectoral indices over 2007–2017 (2,478 days)
reports 99% VaR of **−9.94%** for FHS against **−12.41%** for
generalised EV and **−4.20%** for unfiltered historical simulation, with
the GJR-GARCH-EVT-t-copula hybrid best at **−11.73%**. Kupiec and
Christoffersen tests pass for all four. Two things follow. First, in
India the plain-HS baseline is **2.4x too small** relative to FHS at the
99% level — the conditioning is not optional. Second, the study itself
notes FHS and GEV *"overestimate the portfolio VaR"* relative to other
methods, which is the conservative direction and is what you want in a
training generator.

### 4.2 GARCH on Nifty: persistence is high and the estimates disagree

| Source | Sample | alpha + beta | Half-life (computed) |
|---|---|---|---|
| NYU Stern vollab, `VOL.NIFTY:IND-R` | daily, current | **0.996** | **173 days** |
| ijoqm 14(2) | 2006-01 to 2008-05, n=600 | 0.9648 | 19 days |
| IISTE RJFA 7(13) 2016 | 2001-04 to 2016-03 | 0.9786 | 32 days |
| preprints.org 202509.0997 | 2001–2025 | 0.9796 | 34 days |
| 2026 post-2015 study | 2015–2025, ~2,750 obs | **> 0.96, several indices > 0.98** | — |

Half-lives are computed as `ln(0.5)/ln(alpha+beta)`. The spread is
**19 to 173 days** — a 9x range across sources for the same index. One
2026 paper on only 249 observations reports **0.4097** (0.8-day half-life)
which I believe is an underpowered artefact and which I could not
re-verify; it is in the "could NOT verify" list below.

The direction is unanimous even where the number is not: **persistence is
high, volatility shocks decay over weeks not days, and negative shocks
hit harder than positive ones** (leverage asymmetry, found in every
Indian GARCH study retrieved, including a 2017 benchmark establishing a
**0.94–0.98** persistence band that the post-2015 estimates exceed).

**What this means for block length.** A 40–80-day volatility half-life is
the empirical justification for blocks of that order if you want a single
resampler to carry clustering. It is also a reason to prefer FHS, which
conditions on volatility explicitly rather than hoping a resampler
captures it.

### 4.3 EVT

**Indian evidence is genuinely good here**, which is unusual in this
file. A 2005 report on Nifty and money-market rates (2,475 daily
observations, 7 Nov 1994 to 21 Oct 2004) fits a GPD to the tails of the
**innovation** distribution via peaks-over-threshold:

- Left tail: threshold at **6%** of extreme negatives (148
  observations), `xi = 0.2152` (se 0.0942), `beta = 0.4826`.
- Right tail: threshold at **9%** of extreme positives (223
  observations), `xi = 0.1188` (se 0.0698).
- **The left tail is heavier than the right**, confirming negative skew
  in the tails, and the reciprocal `1/0.2152 = 4.65` means moments above
  the 4th are unbounded. GPD tail quantiles fit the empirical quantiles
  where the normal does not.

More recent: a 2026 study on Nifty 50 finds the same asymmetry via POT
and GPD, and the GARCH-EVT hybrids dominate plain GARCH on backtest.
Indian 99% VaR estimates cluster between **−9.94%** (FHS) and **−12.41%**
(GEV), with the best hybrid at **−11.73%**.

**The tail event frequency, computed from Indian data.** On a 3,750-bar
Nifty history, `P(1-day drop > 5%) ≈ 0.00534`, i.e. **~20 expected days
in the sample**. On a 2% threshold the POET/Darboux fit puts **4.6% of
days** in the tail, i.e. **~172 days**. So a POT threshold at 2% leaves
you a few hundred tail observations; at 1% it leaves you too few to fit
a GPD stably, which is exactly why the 2005 paper chose 6%.

**Verdict: this is the strongest of the six families for Indian data, and
it is still not an RL-training result.** FHS+EVT is the right *generator*.
It has no evidence that it improves a *policy*. And note that FHS
conditions on volatility, so it generates a −13% day from a calm start
only if current volatility is high — which means it cannot generate the
23-Mar-2020 surprise from a quiet February by construction. **FHS
answers "what does a crash look like given we are already in a crash?",
not "how do we get into one".**

## Finding 5 — Family 3: scenario and stress testing

### 5.1 The lineage is entirely capital-adequancy, not policy

The Basel Committee's principles (bcbs147 consultative Jan 2009, bcbs155
final May 2009, superseded and restated in 2018) are the origin of
modern stress-testing doctrine, and they are explicit about their
purpose. Stress tests should *"provide forward-looking assessments of
risk, complement information from models and historical data, be an
integral part of capital and liquidity planning, guide the setting of a
bank's risk tolerance, and facilitate the development of risk mitigation
or contingency plans across a range of stressed conditions."*

**Every one of those is a bank-adequacy objective.** The question they
answer is "does the firm survive", which is a solvency question with a
binary answer. RL policy training asks "which of many continuous
positions maximises expected risk-adjusted return", which is an
optimisation question with a continuous answer. **These are different
questions and the transfer is not free.**

The Basel text is also brutally clear about why historical scenarios
underperformed, and it is worth quoting because it applies to us:

- *"Historical scenarios ... were not able to capture risks in new
  products ... Furthermore, the severity levels and duration of stress
  indicated by previous episodes proved to be inadequate. The length of
  the stress period was viewed as unprecedented and so historically
  based stress tests underestimated the level of risk and interaction
  between risks."*
- On hypothetical scenarios: *"banks generally applied only moderate
  scenarios ... it was difficult for risk managers to obtain senior
  management buy-in for more severe scenarios. Scenarios that were
  considered extreme or innovative were often regarded as implausible."*

That second quotation describes exactly the failure mode we would hit:
the plausible-crash ceiling is set by whoever has to sign off on it.

### 5.2 The Federal Reserve's actual numbers, as a severity anchor

The Fed's supervisory severely adverse scenario is the most
quantitatively specified public stress scenario and it is a useful
calibration target even though we will not use it directly.

2024 scenario (Q1 2024 to Q1 2027): US unemployment +6.3pp from Q4 2023
to a **10%** peak in Q3 2025; real GDP **−8.5%**; **equity prices −55%**
Q4 2023 to Q4 2024; VIX quarterly max peaks at **70**; BBB spread +4.1pp
to **5.8pp**; house prices **−36%**; CRE **−40%**.

2025 scenario (Q1 2025 to Q1 2028): unemployment +5.9pp to **10%** peak;
GDP **−7.8%**; **equities −50%**; VIX peak **65**; BBB +3.9pp to **5pp**;
house **−33%**; CRE **−30%**.

Two facts from these documents are directly relevant and neither is
obvious:

1. **A −50% equity move over four quarters is the supervisory standard
   of severity.** Nifty's worst 11-month drawdown was −55.1%. So
   Indian severity is *in the same range as the Fed's calibrated severe
   scenario*, and a generator that tops out at −13% (one bad day) is
   calibrating to the wrong thing. **The correct target for an Indian
   crash generator is a multi-quarter drawdown, not a single bar.**
2. The scenarios are specified in **28 macroeconomic variables**, with
   equity and VIX among them, and there is a separate **global market
   shock** component applied to trading books at a fixed as-of date.
   This is a *macro-coherent* scenario, not a univariate return shock.
   Building only the equity leg produces incoherence: equity down 50%
   with VIX at 32 and spreads unchanged is not a state any market has
   been in.

### 5.3 Reverse stress testing is the one that fits

Reverse stress testing inverts the direction: given a target outcome,
find the smallest shock that produces it. Basel principle 9 asks firms to
determine *"what scenarios could challenge the viability of the bank"*,
and notes the areas that benefit most are *"new products and new markets
which have not experienced severe strains; and exposures where there are
no liquid two-way markets."*

Baes & Schaanning (2023, *Mathematical Finance* 33(2) 209–256) applied
this to the 2016 EBA stress test, deriving optimal bank responses and
then maximising contagion. They cluster 10,000 worst-case scenarios into
twelve families and report an **"Anna Karenina principle of stress
testing"**: *"Not all stressful scenarios are alike, but every stressful
scenario stresses the same banks."* Their conclusion is the one to keep:
*"the precise specification of a scenario is not of primal importance as
long as the most vulnerable banks are targeted and sufficiently
stressed."*

For us: **the specific crash path matters much less than whether it
attacks the exposure we actually hold.** A crash that leaves a
diversified long-only Nifty-50 book untouched teaches nothing; a −20%
drawdown concentrated in the sectors we hold teaches a great deal.

Grigat & Caccioli (2018, *Scientific Reports* 7:15616) show the same
shape on 44 European banks: as the largest eigenvalue of the leverage
matrix rises, worst-case shocks get *smaller* and more *concentrated*.
Albanese, Crépey & Iabichino (2020, updated 2023) do forward-looking
reverse stress testing by Monte Carlo, selecting the scenarios that
maximise **KVA scenario differentials** — and their most useful finding
for a simulator-builder is a warning: *"a pricing model should be
declared invalid if systematic recalibrations are required on a path
leading to identified stress conditions."* A crash simulator whose
parameters visibly need recalibrating in the crash region is broken in
exactly this sense, and the failure is invisible if you only look at the
return distribution.

**Verdict: adopt the reverse-stress *method* (target an exposure and
solve for the shock that hurts it), reject the Basel/CCAR *artefacts*
(the 28-variable macro scenario set) as an RL training curriculum.**
Transfer risk is high and the jurisdiction is wrong.

## Finding 6 — Family 4: domain randomisation and multi-agent simulation

**This is where the evidence is thinnest and I will be blunt.**

**Domain randomisation** comes from robotics: randomise simulator
parameters so the real world looks like just another draw. Tobin et al.
(2017, *IROS*, arXiv:1703.06907) established it — a real-world object
detector trained **only** on simulated images with procedurally
generated textures, accurate to **1.5 cm**, and pre-training on real
images turned out to be unnecessary. They also report the honest
negative: performance *"degrades significantly when fewer than 1,000
textures are used"*. The mechanism is regularisation, not learning.

Even in robotics the technique is contested. Tobin et al. (2018,
*CoRL*) randomised springs and cameras for sim-to-real grasping.
Mehta et al. (2019, arXiv:1904.04762) found that uniform domain
randomisation *"may lead to suboptimal, high-variance policies"* because
of the uniform sampling, and proposed learning the sampling strategy
instead. Their stated failure mode is ours: *"when using DR
unsuccessfully, policy transfer fails, but with no clear way to understand
the underlying cause. After a failed transfer, randomization ranges are
tweaked heuristically via trial-and-error."* Tiboni et al. (2024, ICLR,
DORAEMON) make the same point: DR *"heavily hinges on the choice of the
sampling distribution ... high variability is crucial to regularize the
agent's behavior but notoriously leads to overly conservative policies."*

**The one financial application is Dow Jones, DDPG, and I only have its
abstract and methodology section.** The DR4DRL study implements
parameter sets `theta ~ p(Theta)` per episode covering volatility scale,
return skew and a cost multiplier, and reports *"statistically
significant improvements in Sharpe ratios across multiple metrics
(maximum, quartile, and mean)."* **I could not open the PDF on the first
attempt and the results tables are not something I have verified.**
Treat the claim as unverified-strength.

There is also newer work in this direction that I have only seen at
abstract level: an ensemble-imitation-learning interactive simulator
reporting max drawdown cut from **−9.11%** to **−4.97%** on NASDAQ-100,
with a bear-market case where buy-and-hold lost **−23.2%** and the
simulator-trained agent **−11.6%**. Single market, no seeds reported, no
Indian test, no circuit breakers. **Not a basis for building anything.**

**Multi-agent / LOB simulation has a documented exploitation problem and
it is directly relevant to us.** ABIDES (Byrd, Hybinette & Balch 2020,
*SIGSIM*) is the standard platform. Vyetrenko et al. (2020, ICAIF, "Get
Real") built the realism metrics the field now uses. The critical finding
is from the LOBGAN follow-up work on conditional generators: the
simulators are **exploitable by an RL agent that optimises against them**,
because relative order placement lets the agent manipulate the
simulator's own state, and the model's parameters become shortcuts. The
authors note the general RL-relevant risk of training against a learned
market model, and that market-maker strategies find unrealistically
profitable behaviour in the simulator.

**That is a direct argument against building a market simulator to train
a crash-avoidance policy on.** A policy rewarded for surviving a crash
will search the simulator for artifacts that make surviving easy. Our
earlier synthetic-panel work already found a Sharpe of **5.525** from
drift alone on a generated panel and equal weight beating the agent in
**4 of 5** configurations — the simulator's artifacts dominate real
signal. A crash simulator raises the reward for artifact-hunting rather
than lowering it.

**Transfer risk: total. No Indian evidence exists for either family.**

**Verdict: do not build. If you ever do, it is for *evaluation* of
execution, not for *training* of allocation, and it goes in the harness
as a held-out test, never in the training panel.**

## Finding 7 — Family 5: regime switching and HMM

This has the best Indian evidence of any family and the lowest build cost.

Ahmad & Kamaiah (2011, MPRA 37174) fit a two-state MS-AR(2) to daily
Nifty and Sensex, 1997–2010 (3,303 / 3,319 observations):

- Bear regime: `mu = −0.19%/day`, `sigma^2 = 6.661`.
- Bull regime: `mu = +0.171%/day`, `sigma^2 = 1.122`.
- **Bear persistence `p11 = 0.95`**, mean duration **20 days**.
- **Bull persistence `p22 = 0.97`**, mean duration **39 days**.
- 50 distinct bear and bull regimes identified in the Nifty sample;
  average bull 45 days, bear 22 days.

The volatility ratio matters: **bear sigma^2 is ~5.9x bull sigma^2**.
That is the parameter a crash generator needs and it is directly
measured on Indian data.

Corroborating Indian regime work: a study using Markov regime switching
with the St. Louis Fed Financial Stress Index finds the probability of
the Indian bull regime *"approaches zero when the stress in the US
financial system crosses the level of two"* — Indian regimes are
externally driven. Post-COVID HMM work on Nifty (Sept 2017 – Sept 2023)
finds base-regime self-transition of **0.8469** pre-COVID rising to
**0.9083** post, i.e. **the regime became stickier after the crash**,
which is a stationarity assumption violation for any fixed HMM.

**How to use it, and how not to.** An HMM is a **state detector**, not a
crash generator. Using filtered regime probabilities as a feature is a
small, cheap, well-evidenced addition. Using them to *sample crash
episodes* inherits the post-COVID persistence shift and the external-
driver fragility. `grep` confirms **there is no India VIX in
`src/stock_rl/` at all** — the single best Indian regime indicator, which
touched **85.13** on 17-Nov-2008 and **83.61** on 24-Mar-2020 (intraday
**86.63**), is not in the codebase.

**Verdict: add India VIX and HMM regime probabilities as *features*
first. That is the cheapest first experiment in this file.** Do not use
an HMM as a crash generator.

## Finding 8 — Family 6: reward injection vs data-changing

These are two different interventions and conflating them is the most
common error in the crash-simulation literature.

**Reward injection** changes the *scoring* of an episode without
changing the *data*. Terminal risk penalties, CVaR terms, drawdown
penalties. The market path is untouched; the agent's ranking of
trajectories changes. Advantage: no new data, no simulation, immediate.
Disadvantage: the agent is being told what to think about a world it
cannot otherwise experience.

**Data-changing** replaces or augments the *price paths* the agent trains
on. Block bootstrap, FHS, diffusion augmentation. The reward is
untouched. Advantage: the agent actually experiences more crash paths.
Disadvantage: the realised transition dynamics no longer match the
recorded tape, which breaks the critic.

The second point is not speculation, it is a formalised failure. A
2026 preprint on scenario-conditioned RL for portfolio management
(arXiv:2602.24037) proves that coupling scenario-based *rewards* with
*tape-based* continuations produces a **hybrid Bellman operator** with
a distinct fixed point from the scenario-consistent objective — a
**reward-transition mismatch**. They report the mismatch gap falling from
**0.526** to **3e-4** and critic residual-AUC from **0.217** to
**0.102** once a counterfactual continuation is mixed into the bootstrap
target. Their own diagnosis is that in distribution shifts *"the policy
may generate state and action pairs that are not supported by economic
structures"*.

The same paper's headline result is a warning about crash training in
general: Sharpe **0.457 → 0.506** and max drawdown **0.381 → 0.168**
from adding scenario-conditioned training. A **65% drawdown reduction
from synthetic scenarios** is a large claimed effect, and it is on US
equities, on a preprint, with no cost model reported and no Indian
transfer. I would not build on it. But the *mechanism* — that
injecting synthetic rewards into tape-bootstrapped TD learning creates a
critic trained on the wrong fixed point — is a real and general warning
that applies directly to `portfolio_env`.

**The deep-hedging path in this repo already took the reward-injection
route, and `hedge_env` handles the sign problem properly:**
`cvar()` requires `alpha >= min_tail_confidence = 0.85`, and
`risk_penalty()` negates the CVaR so that *"a book that never hedges"*
cannot score best. That is exactly the class of bug that crash training
would reintroduce into `portfolio_env` if we went down the reward-injection
road — and `AllocationReward.drawdown` defaults to **0.0** precisely
because an asymmetric per-step penalty makes the doubling strategy
optimal.

**Verdict: if we intervene at all, intervene on the reward's *scale*, not
on its *shape*, and keep the data real.** Adding a crash-penalty term to
a scale-invariant Sharpe is treating a symptom of a magnitude bug with a
shape bug.

## Finding 9 — Indian crash costs versus `costs.py`

Our cost model is a **calm-market rate card**: `brokerage_pct = 0.0`,
`stt = 0.1%` per side, `exchange = 0.00307%`, `stamp_duty = 0.015%` buy,
`gst = 18%`, `dp_charge = Rs 15.34`, `slippage_bps = 5.0`. Every field is
a statutory rate or a fixed spread assumption. None of them responds to
market state.

**The Indian evidence says two things happen in a crash, and only one of
them is what people expect.**

*Spreads widen and price impact rises.* A working paper on 655 NSE
companies over 17 years (2005–2022) finds the **Corwin-Schultz spread
proxy and Amihud illiquidity both elevated** in the 2008 crisis and in
COVID. It finds liquidity deterioration was **more pronounced and longer
in 2008 than in COVID**, reversed within about **3 months** of the
2020 lockdown but persisting **over 6 months** for GFC.

*Volume does not simply collapse — it depends on the crisis.* The same
study finds **volume rose during COVID** (attributed to 14.2 million new
demat accounts in FY2021, a threefold jump) but **fell in 2008**. So the
common mental model, "spreads widen and volume collapses", is **half
right and specifically wrong for India's 2020 crash**. A simulator that
models a crash as thin volume plus wide spreads is calibrated to 2008,
not to 2020, and the two behave differently on recovery timing.

Indian intraday microstructure reinforces the point: a study of NSE
trade and quote data finds **U-shaped volume and spread patterns** and,
distinctively for an order-driven market, *"a contradictory feature of
concurrent high trading volume and wide spreads."* And a four-dimensional
liquidity study of the Nifty 500 (2009–2019) concludes the Indian market
has *"consistent depth, strong breadth, and immediacy but lower
tightness"*, with depth and tightness driven by each other's **lagged**
values. In other words **Indian liquidity state is autocorrelated**, which
is an argument for FHS-style conditioning and against a constant spread.

**The magnitude problem is worse than the modelling problem.** As
Finding 1.4 established, `costs.py` barely moves the reward. `DELIVERY`
charges **0.3226%** round trip on Rs 1 crore; the turnover weight charges
**0.2500** for the same half-book move. **So the highest-value cost work
in this repo is not a crisis cost model — it is turning the turnover
weight down by ~100x so that the real cost term can do its job.** Fix
that first, then a crisis cost model becomes worth having.

**What we should not do.** Do not inflate `slippage_bps` by a crisis
multiplier pulled from a US paper. There is no Indian estimate for the
multiplier, the Corwin-Schultz proxy is a *daily* high-low estimator
whose crisis behaviour is itself confounded by the very volatility it is
meant to measure, and `slippage_bps` is applied to **both sides**
regardless of order direction or size, so it is not even a
spread-plus-impact model.

## Finding 10 — Survivorship bias compounds with the simulator

The repo's own measured figure, from `indicator-mean-reversion.md`: the
survivorship bias on point-in-time Indian universes is **+0.8 to +3.3
pp/yr** (Jain 2026), a **4x range** driven entirely by universe vintage
and survivor-filter choice, with terminal wealth inflated **+7% to +38%**
and **24%** of a 2015 universe unresolvable on the free data source. Jain
also documents that *"exits are not uniformly losers: merger and buyout
exits mean survivor filters sometimes remove winners"* — so the sign is
not even reliably positive per name.

`nsepit` (NegativeZone) takes the right approach for our purposes: it
**measures the bias per strategy** rather than quoting a constant, and it
notes that the number moves several-fold across strategy families and is
worst for exactly the families that concentrate in names heading for
trouble. NSE's licence does not permit redistributing survivorship-free
data, so no vendor hands you a clean panel.

**The compounding argument, which is the part that matters.** Suppose our
panel is survivor-only. A block bootstrap resamples from that panel. Then:

1. The historical maximum loss is **understated**, because the worst
   outcomes in Indian equities were concentrated in names that were
   delisted, merged or wound down — exactly the names a survivor filter
   removes.
2. The **cross-sectional dispersion** in a simulated crash is
   understated, because the names that would have driven dispersion are
   absent.
3. The **timing** is wrong, because delisting is itself a crisis event
   and it has been scheduled out of the sample.

So the simulator is least accurate precisely at the tail it is supposed
to model, and the error is **multiplicative, not additive**, in the
crash region. A `+3.3 pp/yr` bias on the calm panel does not correct to a
`+3.3 pp/yr` bias on the crash panel.

**Concretely for this file:** a crash simulator must not be built on any
panel whose membership is not point-in-time. `NSE bhavcopy` archive
including delisted names is the only free Indian source that cannot have
that hole. Until the panel is PIT, **any crash verdict from this repo is
void on the same grounds the `low_volatility` skill already declares
`delisted_names_present` a kill criterion.**

## Finding 11 — The 3 August 2026 close change, and what it does to a
crash simulator

**This is new and it is not in the sibling docs' quantitative treatment.**

SEBI circular `HO/47/11/11(3)2025-MRD-POD2/I/2765/2026`, dated **16 Jan
2026**, following consultations on **5 Dec 2024** and **22 Aug 2025**,
introduced a Closing Auction Session effective **3 Aug 2026**:

- A 20-minute CAS, 15:15 to 15:35, for F&O stocks, with four phases:
  reference price from 15:15, order entry 15:20 and 15:25, matching
  15:30, random close in the final two minutes.
- **±3% price band** around a reference price set by the VWAP of the last
  15 minutes of continuous trading.
- Continuous trading for CAS stocks **ends at 15:15**; non-CAS stocks
  trade to 15:30. Derivatives run to 15:40, cash post-close 15:50–16:00.
- Pre-open auction aligned to the same structure from **7 Sept 2026**.
- Settlement price for stock and index derivatives now based on the CAS
  closing price.

A SEBI order in the August 2026 archive documents the first contested
case: on **13 Aug 2026**, a Sensex weekly-expiry day, the 15:15 reference
price was **77,829.60** and the CAS closing price came out at
**78,079.96** — a **113.61-point recovery** inside the auction window,
with three separate spikes in the indicative equilibrium price
identified, including one in a 28-second interval (15:25:49 to
15:26:17).

**SEBI's own September 2026 consultation paper** proposes further
changes: a **blended VWAP** settlement (last 30 min of CTS plus the 10
min CAS), or CTS-only; a reduction of the CAS transition period from 5
minutes to 1, making the effective CAS window **10 minutes**; and either
a post-3:30pm CAS with derivatives to 3:45, or the current 3:15pm start
with derivatives to 3:30.

**Three implications for crash simulation:**

1. **The last 20 minutes of the day are now a different microstructure.**
   Our `calendar.py` already notes CAS under NSE/FAOP/74467 of 29 May
   2026 and correctly refuses to fold it into the normal session. But the
   bar data itself carries no CAS phase information, so any crash day
   after 3 Aug 2026 has a close that was formed in an auction.
2. **CAS can *create* a price move that no intraday path justifies.** A
   113.61-point Sensex move inside an auction with a ±3% band is a
   closing-price artifact, not a crash. A simulator that learns "the last
   bar is where the big moves are" from post-CAS data would learn an
   artifact.
3. **There are ~2 months of post-CAS Indian evidence as of this writing
   and no peer-reviewed study of it.** A crash simulator calibrated on
   post-CAS closes is calibrated on a sample too short to characterise
   even one crisis. Prefer pre-CAS data for calibration and treat the
   post-CAS window as out-of-sample.

The sibling file `indicator-mean-reversion.md` records that Tradetron
measured 25 post-CAS auction sessions against 81 pre-CAS controls. That
is the right shape of study and it is the right number of sessions to
have. **It is not enough to calibrate a tail.**

## Confidence levels

| Claim | Confidence | Why |
|---|---|---|
| Our Sharpe term is scale-invariant, so permanent de-risking is worth 0 | **Very high** | Arithmetic identity, verified numerically over 200 seeds |
| Our turnover term penalises crash avoidance by 7.4x (whole-book) and 16.8x (targeted) | **High** | Own computation on a stated toy; sign and magnitude robust, absolute value depends on chosen mu/sd |
| Real cost is 155x weaker than the turnover charge at the same trade | **High** | Computed from `costs.py` and `AllocationReward` directly |
| `portfolio_env.py:29`'s "collapses to a constant" does not describe our default | **High** | `drawdown = 0.0` at line 99 |
| Indian CB trips are as numerous as the four documented downside cases | **Moderate** | Primary circulars plus contemporaneous reporting; an independent automated count says **1 of 1,853** Nifty sessions since 2019 tripped a 10% CB |
| FHS materially outperforms plain HS on Indian data | **Moderate–high** | IJETT 2018, 2,478 days, five sectoral indices, Kupiec/Christoffersen pass |
| Nifty GARCH persistence is high (0.96–0.996, half-life 19–173 days) | **High** for direction, **low** for any single number | Five sources spanning 2x in half-life |
| Indian tail is left-skewed and heavy (xi = 0.2152 left vs 0.1188 right) | **Moderate–high** | 2005 report on 2,475 obs; corroborated 2026 |
| Indian crash costs rise (spread, Amihud) and persist longer in 2008 than 2020 | **Moderate** | One 655-name working paper; not peer-reviewed as retrieved |
| Indian crash volume *falls* | **Low — contradicted** | Same source finds volume **rose** in COVID; do not model volume collapse as universal |
| Basel/CCAR severity is a valid target for policy training | **Low** | The lineage is capital-adequancy; no study transfers it to policy learning |
| Reverse stress testing transfers to policy training | **Moderate on method, low on numbers** | Anna Karenina result is about banks, and it is a macroprudential finding |
| Domain randomisation improves financial RL policies | **Very low** | One DJIA DDPG study, abstract-level; zero Indian evidence |
| LOB/multi-agent simulators are exploitable by an RL agent | **Moderate–high** | Documented for LOBGAN specifically; generalises as a mechanism |
| HMM regime detection has Indian support | **Moderate–high** | Ahmad & Kamaiah 2011 with persistence and duration parameters |
| Post-COVID regime persistence rose (0.847 → 0.908) | **Low–moderate** | Single study, Nifty 2017–2023, abstract-level |
| Survivorship bias is +0.8 to +3.3 pp/yr and variance-dependent | **Moderate–high** | Jain 2026, working paper, 2x2 design, PIT universes |
| Survivorship bias is **multiplicative** with simulator error at the tail | **Moderate** | Mechanism argument, not measured; plausible and untested |
| Post-CAS Indian evidence is too short to calibrate a tail | **High** | ~2 months as of writing; no peer-reviewed study found |

## Could NOT verify

- **DR4DRL (`gausslighter.com/DR4DRL.pdf`) full results.** The direct
  PDF fetch returned "Unsupported fetched file content type:
  application/pdf"; the `.html` variant 404s. I recovered the abstract
  and §4.2.3 (methodology) through a text proxy, so the *"statistically
  significant improvements in Sharpe ratios"* claim is from the **author
  abstract**, not from results tables I have read. The stated
  significance tests, effect sizes, seed count and data split are all
  unknown to me.
- **The 2026 GARCH paper reporting alpha+beta = 0.4097 on Nifty.** Seen
  in an earlier pass of this review on only 249 observations. I could not
  re-open it. I believe it is underpowered rather than a genuine
  contradiction, but **I have not verified that belief.** If it is real,
  the persistence table above has a genuine outlier.
- **Springer / SAGE / Emerald / ScienceDirect full texts:** 403 or bot
  challenges on every attempt. Everything from those publishers in this
  file is abstract-level. This affects the Siddique et al. regime/HMM
  work, the Bhattacharya & Bhattacharya illiquidity paper, and the
  *Discover Artificial Intelligence* efficiency-clustering paper.
- **The independent "1 of 1,853 sessions tripped a CB" count.** This comes
  from a vendor blog that ran the calculation against a broker API; the
  number is plausible and consistent with the documented events, but I
  did not reproduce it against NSE data myself. Treat as corroborating,
  not as a source.
- **Whether Sensex tripped a 10% index CB on 21-Jan-2008 or only hit the
  threshold intraday.** The Wikipedia account says BSE "stopped trading
  for a while at 2:30 pm due to a technical snag", which reads as an
  operational failure rather than a CB halt. The 22-Jan-2008 entry is
  clearer (one-hour BSE suspension after a 10% intraday breach). I have
  not found an NSE or BSE notice confirming the January 2008 mechanism.
- **The Tradetron post-CAS measurement (25 vs 81 sessions)** is recorded
  from the sibling file, not re-derived here. I did not open the
  Tradetron source.
- **NSE/FAOP/74467 of 29 May 2026**, the exchange circular cited in our
  `calendar.py`. I verified the SEBI circular and its effective date but
  not the NSE exchange circular number or its content.
- **Kritzman & Li (2010) turbulence and the absorption ratio applied to
  India.** The methodology is clear and the March 2020 absorption spike
  to above the 90th percentile is documented by an institutional user.
  **I did not find an India-specific turbulence study.** Do not assume
  Indian turbulence thresholds.
- **Any Indian study applying domain randomisation or a multi-agent
  simulator to NSE.** Searched; found none. The absence is itself the
  finding, but a null search is weaker evidence than a positive one.

## Three-arm A/B test, with pre-registered kill criteria

The existing harness (`experiment/harness.py`) already has the right
shape: `ArmConfig` guarantees every arm is measured identically, `Arm` is
a three-way enum, and `KillCriteria` in `context/fuse.py` is
pre-registered. I am **not proposing a new framework.** The three arms
below map onto the existing `Arm` enum's role: A is the control that
already exists, B and C are the two interventions.

The question is: **does training against crash exposure improve a policy
evaluated on real Nifty?**

| Arm | Training data | Reward | Panel kind |
|---|---|---|---|
| **A — control** | Real point-in-time Nifty panels | Current `AllocationReward` | `observed` |
| **B — crash-exposed** | Same panels + the four real Indian crash windows (2020-03-12 to 2020-03-25, 2008-01-21, 2008-10-24, 2024-06-04) oversampled by resampling with replacement | **Current** `AllocationReward` | `observed` + `crash-resampled` |
| **C — reward-fixed** | Identical to B | `AllocationReward` with the turnover weight rescaled per Finding 1.4 | `observed` + `crash-resampled` |

**Arm C exists because of the finding, not because it is interesting in
isolation.** If the reward cannot express crash avoidance, then arm B
should *fail* and arm C should *also* fail, and the failure mode will
tell us whether the reward or the data was the binding constraint. If C
passes where B fails, the turnover weight was the bug. **If both B and C
fail, do not build a simulator — the reward cannot use one.**

Pre-registered criteria, extending `KillCriteria` with two
crash-specific fields and reusing the existing five:

```yaml
kill_criteria:
  # Inherited from KillCriteria, unchanged.
  - id: alpha_vs_control
    threshold: p <= 0.10 on CAPM alpha, arm minus arm A
  - id: sharpe_vs_control
    threshold: p <= 0.10 on the Sharpe comparison, arm minus arm A
  - id: abs_t
    threshold: |t| > 3.0   # Harvey, Liu & Zhu: 158 of 296 published factors were false discoveries
  - id: folds_passed
    threshold: at least 4 of 5 walk-forward folds
  - id: turnover_cap
    threshold: monthly one-sided turnover <= 50%   # Novy-Marx & Velikov
  # New, crash-specific. Both are absolute refusals.
  - id: trained_only_on_real
    statement: the evaluation folds must contain no crash-resampled bars at all
    threshold: any resampled bar in an evaluation fold voids the verdict
  - id: pit_universe
    statement: the panel must include delisted Nifty members, point-in-time
    threshold: survivor-only results are void   # same rule as skills/low_volatility.md
  - id: no_cas_spanning_verdict
    statement: the evaluation window must not mix pre- and post-3-Aug-2026 close definitions
    threshold: a spanning window reports INCONCLUSIVE, never KEEP
  - id: crash_window_gain
    statement: on the four real Indian crash windows only, arm minus arm A
    threshold: >= 0.50 reduction in max drawdown, held in at least 3 of the 4 windows
```

**The `trained_only_on_real` criterion is the one that makes the rest
honest.** It is the analogue of `panel_kind='synthetic'` in the existing
harness, and it exists because the most likely failure mode of this whole
exercise is producing a good Sharpe on a panel we built ourselves. **A
crash-simulation verdict is not evidence unless it was measured on bars
the simulator never touched.**

`panel_kind` should be set to `'observed+crash-resampled'` for arms B and
C and `'observed'` for A, and the report must carry it.

**`crash_window_gain` is deliberately separate from the Sharpe and alpha
criteria.** Those two can be passed by a policy that is simply less
exposed all the time — which is exactly the kind of null result that
looks like a win. The crash-window drawdown test is the only one here
that measures the thing we actually want, and it is measured on real
bars.

## Conclusion: what to build, and what not to

### Shortlist, in build order

1. **Rescale `AllocationReward.turnover`.** Not crash simulation. A
   one-line change with a 155x justification from the repo's own cost
   model. **Prerequisite for everything else.**
2. **Add India VIX and HMM regime probabilities as features.** The best
   Indian evidence in the whole file (bear `p11 = 0.95`, duration 20
   days, bear `sigma^2` 5.9x bull) at the lowest build cost of anything
   on this list. `grep` confirms no VIX in `src/stock_rl/` at all.
3. **Point-in-time panel including delisted names, from NSE bhavcopy.**
   Not an intervention; a precondition. Without it every other result is
   void under the `pit_universe` criterion.
4. **Block bootstrap with a declared block length**, as the cheapest
   generator that preserves cross-sectional structure. Declared, not
   Politis-White-selected on raw returns, for the reason in Finding 3.
5. **FHS + EVT overlay on top.** Best Indian evidence of any generator
   family. Note it cannot generate a crash onset from a calm start.
6. **Reverse stress testing by hand**, in the Baes-Schaanning sense:
   pick our actual exposures, solve for the shock that hurts them. Not a
   pipeline, a design discipline.
7. **Crash-window oversampling as arm B of the three-arm test.** Only
   after 1–3, because before 3 it is measuring survivorship.

**Not building: domain randomisation, multi-agent / LOB simulation,
generative diffusion models, or a full CCAR-style 28-variable macro
scenario engine.** The evidence is absent for the first two, the
mechanism is actively adverse for both (the agent optimises against the
simulator's artifacts, and our own synthetic panel already produced a
drift-only Sharpe of 5.525 with equal weight winning 4 of 5), and the
third transfers a capital-adequacy artefact to a policy problem.

### The cheapest first experiment

Not a simulator. **The cheapest experiment is arm C of the three-arm
test with arms B and C only, on a panel short enough to run in an
afternoon, with the turnover weight rescaled.** It costs one
`AllocationReward` constructor change and one harness invocation, and it
is decisive: if a policy trained with real 2020 crash bars cannot beat
the control on real 2020 crash bars *after* the reward is fixed, then
every generator on this list is premature, because the pipeline it would
feed cannot consume the output.

### The named ceiling

**Ceiling: crash-window gain must reach ≥0.50 max-drawdown reduction in
at least 3 of the 4 real Indian crash windows, evaluated on bars the
simulator never produced.** That number is a **declared constant, not a
validated threshold.** It is chosen because it is large enough that a
half-hearted risk-parity overlay could not accidentally clear it, and
because 2008-10-24 and 2020-03-23 are the only two days in the sample
where the index CB tripped, so a policy that merely de-risks on a 10%
signal would be measured against real, exchange-enforced evidence.

### Upgrade path, in order of what unlocks what

- If the ceiling is **missed on arms B and C alike** → stop. The reward
  cannot use crash data. Do not build a generator; fix the objective,
  which points back at `hedge_env`'s terminal coherent risk measure
  (`cvar` / `semi_rmse`) rather than the running Sharpe.
- If **C passes and B fails** → the turnover weight was the binding
  constraint, and the reward fix is the finding. Ship that; do not build
  the generator.
- If **B and C both pass** → the cheapest generator that adds information
  over real crash bars is FHS+EVT (best Indian evidence), and the block
  bootstrap is the fallback if cross-sectional structure matters more
  than conditional volatility.
- If **both pass but `crash_window_gain` fails** → the improvement is
  generic de-risking, not crash learning. Declare the simulator
  unnecessary and stop.

### What would change this conclusion

A paper that (a) trains a policy on crash-augmented data, (b) evaluates
it **only on real out-of-sample Indian bars spanning at least one index
circuit-breaker trip**, (c) uses a point-in-time universe with delisted
names, (d) compares against a control trained on identical real data,
and (e) reports seeds, dispersion and a Deflated Sharpe against the
control. **Nothing like this exists in the Indian literature**, and the
two international papers claiming large crash-robustness gains
(arXiv:2602.24037 and arXiv:2510.07099) are preprints on US equities
with no cost model, no seeds reported and no independent replication.

The cheapest thing that would move this file furthest is not more
simulator research. **It is a PIT Indian panel and a fixed reward.**