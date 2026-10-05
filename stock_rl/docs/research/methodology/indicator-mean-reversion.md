# Indicator Mean-Reversion and Oscillator Strategies on NSE

Extends [technical-indicators-nse.md](technical-indicators-nse.md), which
records that the indicator thresholds behind this project's ranking have no
published Indian evidence. That file's conclusions are **not** overturned here.
The one study that applies a multiple-testing correction to Indian technical
data (Pillai 2026, §1 below) **finds the conventional RSI 30/70 rule fails**,
which strengthens that file's claim rather than contradicting it.

Note on method: `websearch` **did** work for this review, unlike the previous
one. Most findings come from publisher pages, RePEc/IDEAS abstract records,
SSRN landing pages and SEBI PDFs, which is enough for design and headline
numbers but **not** for full result tables. Where I needed a table I could not
read, the effect size is marked *unverified* rather than estimated.

## Executive summary — plain terms

1. **There is no credible peer-reviewed Indian evidence that retail oscillator
   signals (RSI, stochastic, Williams %R, CCI) generate risk-adjusted excess
   returns after Indian transaction costs.** Every Indian study I found that
   reports a *positive* oscillator result either uses one stock, uses eight
   months of data, applies no costs, or publishes in a venue with no
   peer-review process.
2. **The conventional 30/70 threshold has never been validated on Indian data.**
   The first multiple-testing-corrected Indian test of it found it **failed**
   (adjusted p = 0.1392). The best Indian RSI result comes from the *more
   aggressive* 20/80 pair, and only just clears the bar (adjusted p = 0.0367,
   bootstrap 95% CI on its Sharpe **[0.174, 1.656]** — very wide).
3. **Wilder chose 30/70 by judgement, not derivation, and he chose it on US
   commodity data.** The only peer-reviewed test of the specific 30/70
   configuration I could find is on USD/CHF, and it **loses 3,009 pips**. It is
   US/FX evidence. There is no Indian validation and none is claimed by anyone.
4. **Bollinger Bands: the two published interpretations conflict, and the only
   corrected Indian test kills one of them.** Mean-reversion-from-the-bands
   survives in two small low-quality Indian studies. The "squeeze breakout"
   reading **failed** Romano-Wolf correction (adjusted p 0.071 to 0.473).
5. **The closing-auction mechanic is nine weeks old.** SEBI replaced the
   30-minute VWAP close with an auction from **3 Aug 2026**. Therefore *every*
   Indian study of overnight or intraday reversal predates it, and SEBI has
   already documented an abnormal auction print. No post-CAS Indian evidence
   exists because none can.
6. **T+1 settlement did not produce a measurable reversal regime.** The one
   study I found attributes a change in the Indian Monday anomaly to T+1 but
   cannot reject its own null; the peer-reviewed T+0 study finds no effect on
   price efficiency or liquidity at all.
7. **On inversion: Indian data does *not* show a systematic inversion of
   price-based oscillators.** What it shows is a *liquidity-conditional*
   reversal — reversal exists among illiquid names and momentum among liquid
   ones — plus one directly relevant finding: **Indian overnight returns do not
   reverse** ([Nair et al. 2024](#nair-et-al-2024)). The FinBERT-style
   inversion recorded in this repo is a *text* result and does not transfer.
8. **Survivorship bias is not a constant you can subtract.** Measured on
   Indian point-in-time universes it ranges **+0.8 to +3.3 pp per year**, a
   factor of four, purely as a function of universe vintage and survivor
   filter.

## Evidence table

| Indicator | Evidence strength (India) | Sample | Market | Effect size | Source | Peer-reviewed? |
|---|---|---|---|---|---|---|
| RSI 30/70 (Wilder default) | **Negative** | 78 stocks, 2015-01-01 to 2025-04-30, 2,586 days | NSE F&O | Romano-Wolf adj. p = **0.1392**, FAILS | Pillai 2026 | No — working paper |
| RSI 25/75 | **Negative** | same | NSE F&O | adj. p = 0.0713, FAILS | Pillai 2026 | No — working paper |
| RSI 20/80 | **Marginal / borderline** | same | NSE F&O | adj. p = **0.0367**; Sharpe 95% CI [0.174, 1.656] | Pillai 2026 | No — working paper |
| RSI (14,30/70) | **Negative** | 2000-2021, 33 RSI strategies | Nifty 50 | "popular RSI(14,30/70) strategy resulted in **negative returns**" | RPS, 2023 | Unverified venue |
| RSI, modified bands 50-50 / 60-40 | **Null** | two decades, daily | Nifty 50 | both p > 0.05 | JMRA 8(4), 2025 | Unverified venue |
| RSI, arbitrary bands | **Unequivocally negative** | ~10 yrs daily | **USD/CHF** | **−3,009 pips**, 53 trades; best single loss −1,702 pips | Anderson & Li 2015 | **Yes** |
| RSI + Bollinger combined rule | **Weak / not a threshold test** | 14 stocks, hourly, Jan-Aug 2022 (8 months), no costs | Nifty 50 | net profit 11/14 stocks; beat B&H 10/14; 39.54% avg trade profitability; ~26.5 trades/stock | Tadas et al. 2023 | **Yes** |
| RSI, defence stocks | **Implausible — treat as unusable** | 5 stocks, 2015-2025 | NSE/BSE | claims 21.66% mean return, t=15.962, but pooled ρ = 0.19 | kommerstad.org | **Unverified venue** |
| RSI 30/70 on Indian indices | **Inconclusive** | 3 indices, 2008-2021, split 7+7 yrs | India / Saudi / China | 27 of 54 results significant at 1%; **Indian-specific return not recoverable** | Singh & Pillai 2023 | Unverified venue |
| RSI vs StochasticRSI | **Weak** | 23 equities, no costs | Indian equities | no cost model; authors state sample small | JTAR, 2026 | Unverified venue |
| Stochastic oscillator | **Negative in India** | daily index data, Jan 2018-May 2025 | BRICS | **failed to beat buy-and-hold consistently; India weak or negative** | Poonam & Malik 2025 | **Yes** (Indian J. Finance) |
| Stochastic oscillator | **Positive but n=1** | 1 stock (SBI), 2007-2016, 10 yrs, no costs | NSE | 19.11% avg sell return (best of 5 indicators) | IOSR-JBM 19(9) | Yes, low-quality venue |
| Stochastic oscillator | **Positive but unauthenticated** | Nifty 50, 2004-2014 | NSE | 6,979 points, 321 trades, 49.22% accuracy | academia.edu preprint | **No** |
| Williams %R | **Sub-50% hit rate** | 16 Nifty sectoral indices, 2016-04 to 2021-03 | NSE | "success %" **40.2** (rank 8 of 10 indicators) | turcomat.org | **No** |
| Williams %R | **n=5 stocks, 6 months, chart-read** | Tata Motors, SBI, Infosys, Sun Pharma, HUL | NSE/BSE | hit rates 7/13, 6/8, 10/14, 6/12 — no significance test | Ranade 2020 | Yes, but not a test |
| CCI | **Negative in India** | daily index data, Jan 2018-May 2025 | BRICS | India in the weak/negative group | Poonam & Malik 2025 | **Yes** (Indian J. Finance) |
| CCI | **Positive but n=1** | SBI 2007-2016, no costs | NSE | 11.30% avg sell return — *worst* of 5 indicators | IOSR-JBM 19(9) | Yes, low-quality venue |
| CCI | **Positive but unauthenticated** | Nifty 50, 2004-2014 | NSE | 7,012 points, 182 trades, 48.35% accuracy, "+9.52% over buy-and-hold" | academia.edu preprint | **No** |
| CCI | **Sub-50% hit rate** | 16 Nifty sectoral indices, 2016-2021 | NSE | "success %" **41.9** (rank 6 of 10) | turcomat.org | **No** |
| Bollinger Bands, **squeeze breakout** | **Negative after correction** | 78 stocks, 2015-2025, 0.05% round-trip cost | NSE F&O | BB_20_2.0 and BB_20_2.5 both FAIL; adj. p 0.071 to **0.473** | Pillai 2026 | No — working paper |
| Bollinger Bands, **band re-entry (reversion)** | **Weakly positive** | Nifty 50 index, 2011-2020 | NSE | BB reversal beats MACD p=0.00006, beats RSI p=0.00084; 81 trades, **45.6% win rate**, +0.83%/trade, max DD 12.7% | "Indian Market View", 2025 | **No — education journal** |
| Bollinger Bands + RSI rule | **Weak / not a threshold test** | 14 stocks, hourly, 8 months, no costs | Nifty 50 | see Tadas et al. above | Tadas et al. 2023 | **Yes** |
| Bollinger Bands vs MA envelopes | **Negative for BB at short horizons** | 10 yrs daily, 50 Nifty stocks | NSE | MAE beats BB at N=10/20/50; "BBs cannot design lucrative trading rules and **underperform buy-and-hold**" | ijier.um.edu.my | Unverified venue |
| Bollinger Bands, 20-minute intraday | **Insufficient** | 1 trading year, tick-by-tick NSE | NSE | no statistics retrievable | Amrita 2015 | Conference abstract |
| Any technical indicator, net of costs | **Negative** | 69 large companies, 1999-2004 | India | gross returns significant; **"do not perform better than simple buy-and-hold on a net return basis"** | Sehgal & Gupta 2007 | **Yes** (*Vision* 11(3)) |
| Intraday overreaction (speed of adjustment) | **Positive — but continuation, not reversion** | peer-reviewed, Indian HFT data; period not in abstract | Indian equities | positive next-day premium; magnitude *unverified* | Siddiqui & Misra 2025 | **Yes** (*J. Asset Mgmt* 26(5)) |
| Intraday reversal at the open | **Positive, big, low-quality** | sample period *unverified* | Indian equities | winner +5.72% at open then **CAR −0.76%** over 2.5h; loser −8.39% then **+0.82%** in 30 min, +0.95% in 1h | IJFM, n.d. | **No — predatory-adjacent** |
| Overnight-return reversal | **Absent in India** | 612,066 firm-days, 2016-04-01 to 2021-12-31 | BSE | **"overnight returns in India do not reverse over time"**; top decile keeps outperforming for 12 months, all five subperiods | Nair et al. 2024 | **Yes** (*Vision*) |
| T+1 settlement → reversal regime | **No effect found** | Nifty, 2021-10-27 to 2024-04-25, pre/post 2023-01-25 | NSE | Mon return −0.07% → +0.10%; Mon σ 1.23% → 0.68%; **null not rejected** | IOSR JEF 15(3), 2024 | Yes, low-quality venue |
| T+1 settlement → liquidity | **Positive but no activity effect** | phased quasi-experiments, Indian markets | India | shorter cycle **narrows spreads**, larger effect for illiquid; **no increase in market activity** | SSRN 4745303 | **No — working paper** |
| T+0 settlement → price efficiency | **Null** | event study + DiD, 2024-2025 | India | **no significant impact** on price efficiency or liquidity | EPW 2025 | Reputable venue, short piece |
| 1-month short-term reversal | **Present, illiquid names only** | 3,956 stocks, 2000-2021 | BSE | reversal risk-adjusted **≈ −1%/month**, significant only at 6-month holding | "Momentum, reversals and liquidity" | **Yes** (*J. Financial Markets*) |
| 1-month short-term reversal | **Present, 3 months, small/mid stocks** | monthly, 1991-2006 | India | short-term **reversal** for up to 3 months; contrarian profits mainly small and mid caps | Chowdhury & Mitchello 2008 | **No — job-market paper** |
| 1-month short-term reversal | **Absent — continuation instead** | 328 stocks, 1997-01 to 2013-03 | NSE | **strong short-term momentum**; "no reversal observed" at horizons as short as 1 year | Dhankar & Maheshwari 2014 | Yes, but venue is weak |
| Short-term reversal, all names | **Net negative after risk adjustment** | peer-reviewed, 2000-2021 | BSE | short-term momentum risk-adj **≈ −1%/mo** for all stocks; **≈ +4%/mo** for liquid only | "Momentum, reversals and liquidity" | **Yes** |
| Long-horizon reversal | **Risk premium, not alpha** | 470 stocks, 1997-01 to 2013-03 | BSE | L−W ACAR +21.33% over 36 months (t=2.155) but **"not robust under a multifactor framework … nothing but compensation for risk"** | Dhankar & Maheshwari 2016 | Unverified venue |
| Long-horizon reversal | **Explained by value and size** | 328 stocks, 1997-2013 | NSE | contrarian 35.7% at 36×36, but **value-neutral L−W portfolios insignificant** | Dhankar & Maheshwari 2014 | Yes, but venue is weak |
| Survivorship bias magnitude | **Large and non-constant** | PIT top-500 universes, vintages 2013/2015 | NSE | **+0.8 to +3.3 pp/yr** (4× range); terminal wealth **+7% to +38%**; **24%** of a 2015 universe unresolvable on yfinance | Jain 2026 | **No — working paper** |
| Survivorship bias, small caps | **Very large** | 1,437 stocks, 2016-2025 | NIFTY Smallcap 250 | survivor-only overstates return by **+4.94 pp (23.3%)**, Sharpe by **+0.097 (9.1%)**; 16.1% of exits were delistings | arXiv 2603.19380 | **No — preprint** |

## 1. RSI: has 30/70 ever been validated on Indian data?

**No. It has not, and no author claims to have.** This is the single clearest
finding in this review.

### What Wilder actually did

Wilder's thresholds originate in *New Concepts in Technical Trading Systems*
(1978), a **trade book**, not a peer-reviewed paper. Practitioner sources agree
they are round numbers chosen by judgement from US commodity experience:
DennTech's RSI guide states the levels were *"chosen empirically from commodity
markets — not derived from statistical theory"*; alpha-suite states *"there is
no mathematical reason why 70 is better than 72 or 68"*; IndicatorEdge calls
them *"Wilder's round numbers"* with *"no derivation"*. These are practitioner
web sources, not academic ones — but they are consistent with the absence of
any derivation anywhere in the academic literature.

### The only peer-reviewed test of the specific 30/70 configuration

**Anderson, B. & Li, S. (2015)**, *"An investigation of the relative strength
index"*, **Banks and Bank Systems** 10(1), 92-96.
Peer-reviewed. Market: **USD/CHF**, daily, roughly a decade. **This is FX, not
Indian equities** — any transfer to India is an assumption I am labelling, not
a result.

| RSI pair | Result | Trades |
|---|---|---|
| **30/70 (conventional)** | **−3,009 pips** | 53 |
| 20/80 | +2,387 pips | 23 |
| 15/85 | +4,616 pips | 10 |
| 10/90 | +1,094 pips | 6 |
| 25/75 | +863 pips | 41 |
| 35/65 | +6,621 pips | 93 |
| 40/60 | +5,206 pips | 125 |

The authors' own framing: *"consistent profit opportunities should no longer
exist in what is already commonly and widely known, but taking a path less
travelled could still lead to profit opportunities not yet discovered."* Note
that even the profitable cells have a largest-loss larger than the total
profit (20/80: biggest loss −2,442 pips against +2,387 total). The shape of
this table is the important part, and it transfers as a *mechanism*, not a
number.

### The one corrected Indian test

**Pillai, S. (2026)**, *"Technical-Indicator Strategy Performance in Indian
Equities: A Joint Test of Market Efficiency, 2015-2025"*, DOI
10.13140/RG.2.2.21259.66087. **Explicitly labelled
"Working Paper, May 2026"; author affiliation "Independent Researcher"; 0
citations at time of access.** Not peer-reviewed.

Design: 13 strategy configurations across six families, on **78 NSE stocks
with continuous F&O eligibility**, 1 Jan 2015 – 30 Apr 2025 (2,586 days),
0.05% round-trip cost, Politis-Romano stationary block bootstrap, Romano-Wolf
StepM at the 5% family-wise level.

| Config | Sharpe | Romano-Wolf adj. p | Survives? |
|---|---|---|---|
| Donchian_20 | 1.26 (CI 0.618–1.940) | 0.0006 | Yes |
| SMA_10_50 | 1.26 (CI 0.603–1.929) | 0.0006 | Yes |
| MACD_26_52_18 | 1.23 | 0.0010 | Yes |
| MACD_12_26_9 | — | 0.0031 | Yes |
| Donchian_55 | — | 0.0031 | Yes |
| SMA_20_100 | — | 0.0048 | Yes |
| SMA_50_200 | — | 0.0094 | Yes |
| **RSI_20_80** | **CI 0.174–1.656** | **0.0367** | **Yes — borderline** |
| RSI_25_75 | — | 0.0713 | No |
| **RSI_30_70** | — | **0.1392** | **No** |
| BB_20_2.0 | — | ≥0.071 | No |
| BB_20_2.5 | — | **0.473** | No |
| Confluence_v1 | 0.53 | 0.209 | No |

Three things follow, and all three matter:

- **The conventional configuration failed.** RSI_30_70 sits at adjusted
  p = 0.1392 — not marginal, comfortably inside the null. This is direct
  Indian evidence against 30/70.
- **The direction of the survivors is trend, not reversion.** Seven of eight
  survivors are trend-following, all at adjusted p ≤ 0.0031. The eighth is the
  *only* mean-reversion rule, at the boundary of the stepdown sequence, with a
  bootstrap CI on its Sharpe spanning [0.174, 1.656]. Its Sharpe is not
  distinguishable from the trend rules.
- **The more aggressive threshold did better, which is the opposite of what a
  mean-reversion story predicts.** 20/80 beats 25/75 beats 30/70 monotonically.
  More extreme thresholds mean fewer, better-timed trades. That is the opposite
  of "buy every oversold dip", and it is the same pattern Anderson & Li found
  in FX.

**This does not contradict technical-indicators-nse.md.** That file says RSI has
"no Indian equity evidence". That claim was about the *conventional* rule and
about properly corrected evidence. Pillai is the first corrected Indian test,
and it fails the conventional rule. The claim gets stronger.

### The Indian studies that claim RSI works, and why they do not count

| Study | Claim | Why it does not count |
|---|---|---|
| RPS, 2023, Nifty 50, 2000-2021 | 33 RSI strategies; **"the popular RSI(14,30/70) strategy resulted in negative returns"**; positive in divergence scenarios | Venue and peer-review process **unverified**. But the direction agrees with Pillai and Anderson & Li |
| Tadas et al. 2023, 14 stocks, 8 months hourly | BB-RSI beat B&H in 10/14 | **Not a 30/70 test.** Entry is price above the Bollinger middle band *and* RSI > 50; exit is touch of the lower band or RSI weakness. It is a trend rule with an oscillator filter, and it charges no costs |
| JMRA 8(4), 2025, Nifty 50, two decades | 50-50 and 60-40 modified RSI bands | **Both p > 0.05.** The authors report their own result as statistically insignificant. This is a null finding reported as a study |
| Singh & Pillai 2023, 3 indices 2008-2021 | 27/54 results significant at 1% | The headline counts are pooled across Saudi Arabia, India and China. **The Indian-specific return is not recoverable from the abstract.** Inconclusive |
| kommerstad.org, 5 defence stocks 2015-2025 | mean return 21.66%, t=15.962 | **Implausible.** A 21.66% mean return from a 14-day RSI signal over ten years, with a *pooled correlation of 0.19* between RSI and subsequent return, is internally inconsistent. Venue unverifiable. Treat as noise |
| Ranade 2020, 5 stocks, 6 months | RSI hit rates 4/7, 3/8, 3/4, 5/5 | Hit rates **read off TradingView charts** by eye over six months. No significance test, no costs, five mega-caps. Not evidence |

### Cross-market evidence, explicitly labelled as a transfer

- **Rink (2023)**, *Financial Markets and Portfolio Management* 37(4), 403-456,
  peer-reviewed: 6,406 rules on 23 developed + 18 emerging equity indices, of
  which **600 are RSI rules** (parameters from Hsu et al. 2016). India is
  **not in his sample** — this is inherited from technical-indicators-nse.md and
  I could not independently verify the market list in this review. RSI was
  almost never the best-performing family.
- **"15 Million Tests, Zero Edge: The RSI"** (TradingView, Jan 2026,
  **not peer-reviewed**, methodology published in the article): 1,432,620
  parameter combinations per asset × 16 assets in 5 classes × data to Jan 2025;
  15M+ complete test cases after filtering; Bonferroni and BH-FDR. Result: RSI
  overbought/oversold gives **no statistically significant edge** after
  multiple testing; nominal significance rates of 3–13% against a 5% null are
  attributed to the sheer number of tests. **No Indian asset** — EFA/EEM are
  the only international proxies, so India coverage is a transfer assumption.
- **Bulkowski (ThePatternSite)**, practitioner, **US** stocks: the RSI 30/70
  benchmark over ~15 years from 100 stocks gives 15.5% net profit vs 8.9% S&P
  change, with $2M portfolio sizing precisely because *"RSI trades can be long
  term ones, so it is common for the trade to span a year"*. Note his actual
  rule is **"buy when RSI drops below 30 then climbs back above it"**, not
  "buy below 30" — a mechanically different rule. Also US, also practitioner.

### Verdict on the threshold question

**The conventional thresholds have never been Indian-validated. They are
inherited from a 1978 US trade book, and the only two peer-reviewed tests that
touch them directly (one on FX, one with proper Indian data-snooping control)
both report them failing.** That is a stronger statement than
technical-indicators-nse.md makes, and it is the most useful thing in this
review.

## 2. Stochastic oscillator, Williams %R, CCI

The answer is **absent**, with one exception that points the wrong way.

**Poonam & Malik (2025)**, *"Risk–Return Efficiency of Stochastic Oscillator
Strategies in BRICS During COVID-19 and Geopolitical Conflicts"*, **Indian
Journal of Finance** — peer-reviewed, and published in the right journal.
Daily index data, **Jan 2018 – May 2025**. Finding: the stochastic oscillator
**failed to outperform buy-and-hold consistently**; positive results were
limited to **Russia and South Africa**; **Brazil, India and China showed weak or
negative risk-adjusted returns.** The paper's practical conclusion is that
traders should not rely on stochastic strategies as standalone tools. **I could
not verify the exact volume/issue/pages for this article** — a related Poonam &
Malik 2025 paper on technical rules in BRICS post-COVID occupies IJF 19(1),
52-66. Effect size for the Indian index specifically is not in the abstract.

This is the most methodologically appropriate Indian oscillator study I found,
and it is **negative**.

Everything else is weak in a way worth cataloguing because the weakness is
systematic, not accidental:

- **One stock.** The IOSR-JBM comparison of MACD / RSI / stochastic / ADX /
  CCI used **State Bank of India only**, 2007-2016, with signals read off
  moneycontrol.com. Results: ADX 26.4%, stochastic 19.11%, RSI 19.06%, MACD
  14.81%, CCI 11.30% average sell return. No costs. A single stock cannot
  support an inference about the Indian market.
- **Unauthenticated preprints.** The "Profitability of Oscillators" Nifty-50
  study (2004-2014) claims CCI 7,012 points / 182 trades / 48.35% accuracy,
  stochastic 6,979 points / 321 trades / 49.22%, RSI 7,031 points / 390 trades
  / 47.43%. **Points are not returns, there is no journal, and the author list
  could not be established.** Note the accuracy rates: all three are **below
  50%**.
- **Sub-50% hit rates even in the positive studies.** The 16-sector Nifty study
  (2016-2021) reports "success %" as Bollinger Bands 70.2, CCI 41.9,
  Williams %R 40.2, stochastic 38.6. Turcomat is not a peer-reviewed venue.
- **Chart-reading.** Ranade (2020), *International Journal of Management*
  11(8), 300-310, hit rates W%R 7/13, 6/8, 10/14, 6/12 and RSI 4/7, 3/8, 3/4,
  5/5 across five mega-caps over six months. No statistical test.

**Williams %R and CCI have no peer-reviewed Indian evidence of any kind that I
could find.** Stochastic has one, and it is negative.

## 3. Bollinger Bands: which interpretation, and do they conflict?

**Yes, they conflict, and they are not reconcilable.** The same 20-period /
2-sigma construction supports two opposite rules, and the Indian evidence
points in opposite directions for each.

**Interpretation A — mean reversion from the band** (buy the lower band, exit
at the middle). Support:
- Tadas et al. (2023) — 8 months, 14 stocks, hourly, no costs. Best of three
  strategies tested, but it is a hybrid rule, not a band rule.
- "A Comparative Study on Predictive Performance of Bollinger Band
  Indicator: An Indian Market View" (2025), Nifty 50, **2011-2020**, ten
  years, claims to remove data-snooping bias and include transaction costs.
  Tests both readings head-to-head, which makes it the only Indian study that
  can. **"BB reversal"** beats MACD (Mann-Whitney p = 0.00006) and RSI
  (p = 0.00084); negative returns in only 3 of 10 years, max drawdown 12.7%;
  81 trades, **45.6% win rate**, average **+0.83% per trade**. **"BB momentum"**
  does **not** beat MACD (p = 0.2032, null not rejected) but does beat RSI
  (p = 0.0079).
  **Venue problem: this is in a journal of informatics *education* and research,
  not finance.** Treat the numbers as illustrative, not established.
- The ijier.um.edu.my study (10 years daily, 50 Nifty stocks) reaches the
  opposite conclusion for BB relative to moving-average envelopes: *"BBs
  cannot design lucrative trading rules and underperform buy-and-hold"*.

**Interpretation B — squeeze / breakout** (a band breach is a continuation
signal). This is the interpretation Bollinger himself documented and the one
implementations default to. Under Romano-Wolf correction in the only properly
controlled Indian test, **both BB configurations fail**: adj. p 0.071
(BB_20_2.0) and **0.473** (BB_20_2.5), against a 5% FWER bar. BB_20_2.5 is
close to the *worst* of all thirteen rules tested.

**They genuinely conflict, and the more rigorous test kills B.** The pattern is
the same as the RSI thresholds: Indian evidence tolerates *fewer, more
selective* signals (RSI 20/80 > 30/70) and rejects the high-turnover
band-watching version outright. Tadas et al.'s BB-RSI rule averaged **26.5
trades per stock in eight months** — roughly 40 trades a year per name — which
at a ~22 bp delivery round trip is not viable regardless of gross return. The
12.7% max drawdown in the 2025 study comes from a strategy with **45.6% win
rate** and 81 trades in ten years, which is a different animal entirely.

## 4. Overnight and intraday mean reversion, and the two Indian market mechanics

### The closing auction is nine weeks old. There is no post-CAS evidence.

**SEBI Circular SEBI/HO/47/11/11(3)2025-MRD-POD2/I/2765/2026**, dated
**16 January 2026**, effective in the cash segment from **3 August 2026**:

- Before: closing price = **VWAP of trades in the last 30 minutes** of the
  continuous trading session (CTS). It was *"a calculation, not a
  transaction"* — a price at which nobody traded.
- After: for stocks with derivative contracts, a **20-minute Closing Auction
  Session, 3:15 pm to 3:35 pm**. Reference price = VWAP 3:00–3:15 pm. Price
  band **±3%** around the reference. Equilibrium price = the price at which the
  **maximum volume is executable**, then minimum unmatched quantity, then
  closest to reference. Market orders take priority over limit orders.
  **Stop-loss, iceberg and revealed-quantity orders are barred.** Order entry
  closes at a **system-random instant between 3:28 and 3:30 pm** to prevent
  timing a last-second push. Equity derivatives run to 3:40 pm.
- Non-F&O stocks still close on the old VWAP basis. Phased rollout.

**Consequence for this research: every Indian study of overnight or intraday
reversal predates the mechanism.** A backtest spanning August 2026 mixes two
incompatible close definitions on either side of the break. That is a data
continuity problem, not a modelling choice, and it needs to be stated
explicitly rather than smoothed over.

**SEBI has already documented a problem.** In its August 2026 order
(ORDER_1787144554.pdf) SEBI records *"abnormal spikes in the Indicative
Equilibrium Price of Sensex on August 3, 2026"* and, for **13 August 2026** — a
Sensex weekly expiry — a reference price of **77,829.6 at 3:15 pm** against a
CAS close of **78,079.96**, describing the move as *"abrupt"*. SEBI's September
2026 Consultation Paper then opened a review of CAS, market timings and
settlement methodology, offering blended-VWAP and CTS-VWAP settlement options
and proposing to cut the CAS window to 10 minutes.

**Press evidence on magnitude** (secondary, but consistent across outlets):
on day one the Nifty's official close was **more than 200 points above** its
3:15 pm level; on day two it recovered ~150 points; cash-market last-30-minute
turnover fell to about **₹1,500 crore against a usual ₹6,000–7,000 crore**
(Marwadi Shares and Finance). Goldman's second-day note put median CAS volume
at **1.9% of the day's traded volume** (from 1.7%) and median stock price
movement in the auction at **45 bp** (from 53 bp). Dhirendra Kumar (Value
Research) noted the spot close sat ~110 points *above* Nifty futures, which
*"almost never happens"* — a live, quantified instance of close distortion.

This is **the reason** overnight evidence is thin rather than merely absent: the
object of study was replaced under you.

### T+1 settlement: no measurable reversal regime

- SEBI migrated T+2 → T+1 in phases: the bottom 100 stocks by market cap from
  **25 Feb 2022**, then 500 per month from March 2022, complete by
  **27 Jan 2023**. Optional T+0 beta from **28 Mar 2024**.
- **"Unlocking Liquidity through Shortened Settlement Cycle"** (SSRN 4745303,
  **working paper**): uses the phased rollout as a series of quasi-natural
  experiments on Indian data. Shorter cycles **narrow quoted spreads and improve
  liquidity, with a larger effect for illiquid stocks**, but **do not increase
  market activity**. No reversal claim.
- **"Impact Of T+1 Settlement On Negative Monday Effect"** (IOSR JEF 15(3),
  2024): Nifty, 27 Oct 2021 – 25 Apr 2024, split at 25 Jan 2023. Monday mean
  return **−0.07% → +0.10%**; Monday return-to-risk **−6% → +15.3%**; Monday
  volatility **1.23% → 0.68%**. **The authors could not reject their own null
  hypothesis.** The direction is suggestive; the test is not evidence. Low-
  quality venue.
- **EPW (2025)**, T+0 assessment via event study and difference-in-differences
  on 2024-2025 data: adoption had **no significant impact on price efficiency or
  market liquidity**. This is the cleanest settlement-regime test I found and it
  is null.
- SEBI's own May 2024 consultation paper gives the structural reason a T+1
  reversal story is weak: for Apr–Dec 2023, delivery trades were **57.7% of
  value at BSE and 44.4% at NSE**, and only **13.5% of all NSE individual
  trades** were delivery. Roughly **19.9% of FPI cash trades** were matched with
  individual domestic investors. The capital that is structurally locked is a
  minority of flow.

### Indian overnight returns do not revert

**Nair, S.T.G., Raghunandanan, A. & Nair, R.K. (2024)**, *"Make Some 'Noise'
Tonight — Overnight Return as Investor Sentiment
Index: Indian Evidence"*, **Vision: The Journal of Business Perspective** —
peer-reviewed. **612,066 firm-days**, BSE and PROWESS, **1 Apr 2016 – 31 Dec
2021**.

They test three criteria against Aboody et al. (2018), *JFQA* 53(2). Results
by subperiod (2016-17 through 2020-21) and for the full period:

| Criterion | Verdict |
|---|---|
| Short-term persistence | **Accept** — all subperiods, ~1% level |
| Greater persistence for hard-to-value firms | **Accept** |
| **Long-term reversal over the next 12 months** | **Reject — every single subperiod** |

Their words: *"overnight returns in India do not reverse over time."* Stocks in
the top 10% of overnight returns **continue to outperform** the bottom decile
over the following 12 months, and this **did not change during COVID-19**.

**This is the most important single finding for question 6.** In the US, retail
overnight buying is faded by arbitrageurs next session, which is the standard
overnight/intraday tug-of-war. **In India that fade does not appear.** A
mean-reversion signal built on the overnight leg would be pointing the wrong
way.

## 5. Short-term reversal as a factor in India

Here the literature is genuinely split, and the split is informative: it is
**conditioned on liquidity and size**, and it disagrees on the plain 1-month
horizon.

**Evidence FOR reversal:**

- **Chowdhury & Mitchello (2008)**, *"Presence and sources of momentum and
  contrarian profits: evidence from the Indian stock market"*. Monthly,
  **1991-2006**. Finds **short-term reversal for both winner and loser
  portfolios for up to three months**, with contrarian profits arising
  **primarily from medium and small stocks**. **Working paper** — a job-market
  paper hosted at MTSU. I could not confirm a peer-reviewed version.
- **"Momentum, reversals and liquidity: Indian evidence"**, *Journal of
  Financial Markets* (Elsevier) — **peer-reviewed**. **3,956 BSE stocks,
  2000-2021.** Short- and intermediate-term **reversals among the most illiquid
  portfolios**; **momentum among the most liquid**; both persist across market,
  size, value and macro controls. Risk-adjusted reversal return is
  **≈ −1%/month** and significant **only at a 6-month holding period**.
  Critically, it also reports that **risk-adjusted short-term momentum is
  ≈ −1%/month and significant for the full universe**, turning **≈ +4%/month
  only when conditioned on liquid stocks**.
- **Dhankar & Maheshwari (2017)**, *Global Business Review* 18(4): heavily
  traded stocks earn **higher momentum AND contrarian** returns. Volume is a
  useful conditioner.
- Event-study evidence: Indian stocks falling more than 20% in a month show
  significant six-month abnormal return reversals (**≈ +11.6% CCAR** at the
  six-month horizon for a −20% trigger, **+14.95%** for −24%), and the
  relationship strengthens with shock size. Reversals after **large increases**
  are much weaker (−4.58% and −5.83%). I could not verify the primary journal
  for this directly — I read it in the literature review of an open-access
  *Springer* article on Indian large-price-movement returns (2000-2019,
  Nifty stocks), so I am citing it **at one remove**.
- Parthasarathy (2019), also at one remove from the same source: **large price
  changes on high volume showed continuations**; low-volume shocks showed
  reversals over the following 20 days. If accurate, this is an explicit
  **volume-conditioned inversion**.

**Evidence AGAINST short-horizon reversal in India:**

- **Sehgal & Balakrishnan (2002)**, *Vikalpa* 27(1), 364 BSE stocks, Jul 1989 –
  Mar 1999: **short-term continuation** — *"momentum for both winners and
  losers in the short run."* Reversal appears only with a **1-year gap** between
  formation and holding.
- **Joshipura (2009)**: NSE, 1995-2008, momentum in the short run, reversals
  later.
- **Dhankar & Maheshwari (2014)**, *IJFM* 4(2), 328 NSE stocks, Jan 1997 –
  Mar 2013: **strong short-term momentum** for 3-12 month formation and
  holding, peaking at 7.7% abnormal for 6×9. *"For formation and holding period
  as short as one year, **no reversal** was observed in Indian stock market."*
  Their own comparison table lists Chowdhury & Mitchello (2008) as finding
  short-term reversals — so the disagreement is acknowledged in the literature,
  not hidden.
- **"Momentum returns: A portfolio-based empirical study"** (2017),
  *Journal of Management & Organization*: 1×3 and 3×6 on NSE 2005-2015.
  Momentum profits exist and P/E, P/B and **net foreign institutional inflows**
  drive them. *"No momentum reversal is seen for returns owing to long-run
  holding periods spanned over 2-5 years"* — the opposite of Lee &
  Swaminathan and Jegadeesh-Titman.
- The *Journal of Financial Markets* paper's own risk-adjusted short-term
  momentum figure (**−1%/month, significant, all stocks**) is the clearest
  statement that at the 1-month horizon India looks like the **opposite of
  reversal** once you control for risk.

**The honest synthesis:** Indian reversal is real but it is **not a
large-cap, high-turnover effect.** It is concentrated in illiquid and small
names, where temporary price pressure is stronger. Nifty-50 constituents — the
universe this project trades — sit on the wrong side of that conditioning. The
existing file's sector/beta finding points the same way: *"Large caps are the
least predictable Indian segment."*

### Institutional flow

You asked whether FII/DII flow breaks the standard pattern. **The evidence says
it changes the mechanism but not the sign.**

- **ADBI Working Paper 1333**: stock-level FII flow innovations, 2019-2020.
  Abnormal **high** FII inflows → **permanent** price increase (information
  effect). Abnormal **low**/negative innovations → **both** a permanent effect
  and a **transient** decline that partially reverses. So: selling pressure is
  transiently overshooting, buying is not. That is asymmetric pressure, not a
  sign flip. Price effects and reversals were **exaggerated during COVID-19**.
- **Dhananjaya & Wright (2019)**, *Wealth — Int. J. Money, Banking and Finance*
  8(1), 65-72: FIIs and DIIs follow **different** strategies; **DIIs
  negatively affect FII flows but are not affected by FII flows**. DIIs act as
  a cushion.
- **"An Analysis of Trading Behaviour of FII and DII"** (2020), *Indian Journal
  of Capital Markets*: 2,440 daily observations Apr 2007 – Nov 2017. **FPIs
  are positive-feedback traders; DIIs are negative-feedback traders** in the
  short run; no feedback trading in the long run.
- **"Institutional investment activities and stock market volatility amid
  COVID-19"** (2021): **FPI momentum buys and contrarian sales induce
  volatility**; mutual-fund style does not.
- **"Diverse investor reactions to the COVID-19 Pandemic"** (2024): DIIs show
  flight-to-quality, **FIIs fire-sale**, retail provides liquidity.
- COVID-era trader-flow study (IJCRT 2020): Apr-Jun 2020 both FPI and DII
  moved together (both withdrew in April); **from July they diverged** — FPIs
  positive-feedback buying, DIIs negative-feedback selling.

**Net:** the feedback structure means FII selling creates overshoot that
partially reverses, and DII buying cushions it. That is a **magnitude and
timing** effect, not an inversion of the sign of the reversal signal.

## 6. Does the signal invert? — the critical question

### The short answer

**No Indian evidence of a systematic inversion of price-based oscillator
signals. There is strong Indian evidence that the *overnight* leg does not
reverse at all, which means a reversion signal built on it is inverted in
practice.**

### The evidence, itemised

| Signal | What Indian data shows | Direction |
|---|---|---|
| Overnight return, 1d-12m | Does **not** reverse; top decile keeps outperting for 12 months, all subperiods 2016-17 to 2020-21 (Nair et al. 2024, *Vision*) | **Inverted vs a fade-the-overnight thesis** |
| RSI 30/70 | Negative returns on Nifty 50 2000-2021 (RPS 2023); adj. p = 0.1392 fails (Pillai 2026) | Fails; not cleanly inverted |
| RSI 20/80 | The **only** mean-reversion rule to survive correction, and only at the boundary | Weakly correct |
| RSI thresholds generally | 20/80 > 25/75 > 30/70 monotonically — **more selective is better** | Opposite to "fade every dip" |
| Intraday overreaction (speed of adjustment) | **Positive** relationship with next-day daily returns; premium unexplained by market/size/value/momentum/illiquidity (Siddiqui & Misra 2025) | **Continuation, not reversion** |
| Large monthly decline (large liquid stocks) | Reversal, ~+11.6% CCAR over 6 months, strengthens with shock size | Correct direction, ~40%/yr claimed |
| Large monthly **increase** | Reversal much weaker: −4.58% at 6 months | Asymmetric; weak |
| Large price move on **high volume** | **Continuation** (Parthasarathy 2019, cited at one remove) | **Inverted**, volume-conditional |
| 1-3 month formation, all stocks | Risk-adjusted **−1%/month**; reversal only among illiquid | Inverted for liquid names |
| RSI on defence stocks | Higher RSI → lower future returns; BEL ρ = −0.41 | Correct, but venue unusable |

### How this compares to the FinBERT inversion recorded in this repo

The FinBERT result is a **text** signal. It cannot be used to reason about price
oscillators, and I found nothing in the Indian literature that would confirm a
mirror-image inversion for price signals. What the Indian price-signal
literature actually shows is:

1. **Inconsistency conditioned on liquidity and size**, not inversion.
   Reversal for illiquid, momentum for liquid. Both effects are real in the
   same market at the same time.
2. **A genuine asymmetry in the overnight leg** (Nair et al.) that is
   structural: the overnight move in India persists, so it should be faded less,
   not more.
3. **A preference for fewer trades** that survives every Indian test I found —
   20/80 over 30/70, BB reversal at 45.6% win rate and 81 trades in ten years,
   26.5 trades in eight months being too many to be viable.

**An inverted signal is a tradable signal in the opposite direction, and that
framing is correct.** But the honest statement is: *no published Indian study
tests price-based oscillator signals in both directions and reports that the
inverted sign is profitable.* The single closest candidate is
Siddiqui & Misra (2025), whose factor is long intraday overreaction — but that
is a different construct (price-adjustment speed, not an oscillator), and I
could not read its magnitudes.

**Confidence: low.** This is a gap in the literature, not a settled result.

## Survivorship-bias caveats for any Nifty-50 oscillator study

This matters more here than usual because the oscillator evidence is
universally small-sample, and small Nifty-50 samples are where bias bites.

- **Jain, A. (2026)**, *"Survivorship Bias in Indian Equities Is Not a
  Number: Vintage- and Filter-dependence in Point-in-Time Universes"*,
  **SSRN 7099378, working paper**. Point-in-time top-500 universes with dead
  names retained, 2×2 design: vintage {2013, 2015} × survivor filter {alive in
  panel; resolvable on yfinance}. Measured bias spans **+0.8 to +3.3
  percentage points per year** — a factor of four — with terminal-wealth
  inflation from **+7% to +38%**, on the same market with identical
  construction and a common end date. Roughly **a quarter of a top-500 universe
  becomes invisible within a decade**; **24% of a 2015 universe cannot be
  resolved on yfinance**. And exits are **not uniformly losers** — merger and
  buyout exits mean survivor filters sometimes remove winners. Conclusion: *"no
  single-number survivorship correction is defensensible."*
- **arXiv 2603.19380**, NIFTY Smallcap 250, 1,437 stocks, 2016-2025,
  **preprint, not peer-reviewed**: survivor-only backtesting overstates annual
  returns by **4.94 pp (23.3%)** and Sharpe by **0.097 (9.1%)**. Turnover
  82.5%, of which **16.1% delisted**, 33.1% graduated, 33.2% demoted.
  Reconstruction is "100% accurate for current constituents and an estimated
  85-90% accuracy historically."
- **Pillai's own universe is a survivor filter of a different kind**: 78 stocks
  with **continuous F&O eligibility** for the full 2015-2025 window. Any name
  that ever lost derivatives is excluded. This cannot be corrected by the
  numbers above and it systematically removes exactly the stressed names an
  oversold rule would want to buy.
- **Sehgal & Gupta (2007)** used 69 large companies, a static hand-picked list.
- **The consensus-round problem.** The front page of every Nifty-50 backtest
  that is not point-in-time already knows which companies survived. For an
  oscillator strategy this is the worst case, because the strategy's edge is
  claimed on names that fell — and the names that fell and delisted are the
  ones a current-constituent list cannot contain.
- **This repo's own baseline result compounds it.** Per
  [rl-portfolio-allocation.md](rl-portfolio-allocation.md), equal-weight beat
  the SAC-style agent and HRP beat all four DRL agents on Nifty-50 data. A weak
  or absent oscillator signal on a survivorship-biased Nifty-50 universe is the
  **expected** outcome, and per the existing file the variance-ratio evidence
  says large caps are the least predictable Indian segment anyway. Do not read
  a failed oscillator backtest as news about oscillators; read it as a statement
  about Nifty-50.

## What this does NOT establish

- **It does not establish that RSI, stochastic, Williams %R or CCI do not work
  in India.** It establishes that no credible peer-reviewed, cost-aware,
  multiple-testing-controlled Indian evidence supports them, and that the
  available evidence is negative where it is credible.
- **It does not establish any threshold.** No number is proposed here. The
  20/80, 25/75, 30/70, ±100, ±200 and 45.6%-win-rate figures appear because
  sources report them, not as recommendations. Thresholds come later, declared
  and labelled unfitted.
- **It does not establish that Pillai (2026) is right.** It is a
  single-author working paper with 0 citations. Its Romano-Wolf correction is
  the right method and its sample is the right market, but a 13-rule family is
  a small family and the RSI_20_80 survivor's bootstrap CI [0.174, 1.656]
  overlaps everything.
- **It does not establish that reversal is absent from Indian data.** Reversal
  is documented among illiquid and small names, and after large monthly
  declines in large liquid stocks. It is absent from **Nifty-50, risk-adjusted,
  for liquid names** — which is a narrower and different claim.
- **It does not establish anything about the post-CAS market.** Nine weeks of
  data is nine weeks. No Indian study of overnight or intraday reversal covers
  the auction-based close.
- **It does not establish that T+1 killed or created a reversal regime.** The
  one test is null, the other cannot reject its own null, and the peer-reviewed
  T+0 study finds nothing.
- **It does not establish that the signal inverts.** It establishes that no
  Indian study tests the inverted sign and reports it profitable.
- **It does not address trend or momentum.** Another reviewer owns that.
- **It does not clear the index-constituent non-synchronous-trading problem**
  that Camilleri & Green (2014) identify for NSE: any index-level oscillator
  signal may be a stale-price artifact. This review found no study that
  decomposes an oscillator signal into overnight/intraday and survives the
  decomposition.

## Confidence

| Claim | Confidence | Why |
|---|---|---|
| 30/70 has never been Indian-validated | **High** | Three independent lines: no study claims it, Pillai's corrected test fails it, and the "Nifty RSI" study reports it negative |
| No credible cost-aware peer-reviewed Indian evidence for stochastic / %R / CCI | **High** | Poonam & Malik (Indian J. Finance) negative for India; every positive study is n=1, 8 months, no costs, or unauthenticated |
| Bollinger squeeze-breakout has no surviving Indian evidence | **Moderate-high** | One corrected test only (working paper), but it is decisive and consistent with the 10-year BB-vs-MAE comparison |
| Bollinger band-reversion has *some* Indian support | **Low** | Every supporting study is in a low-quality venue; the strongest has a 45.6% win rate |
| Indian overnight returns do not reverse | **High** | Peer-reviewed, 612k firm-days, rejected in every subperiod — the single best-supported claim in this file |
| The closing auction broke overnight-return measurement | **High** | SEBI's own circular, SEBI's own order documenting an abnormal print, SEBI's own consultation paper |
| T+1 did not create a reversal regime | **Moderate** | Null from the peer-reviewed T+0 study; the T+1 tests are weak |
| Price-based oscillators invert in India | **Low — and probably false** | No study tests it. The evidence is liquidity-conditional inconsistency, not sign inversion |
| Survivorship bias is large and non-constant in India | **Moderate-high** | Two independent studies, different sizes, both large; neither peer-reviewed |

## Disagreements between sources, stated not resolved

1. **Chowdhury & Mitchello (2008) vs Sehgal & Balakrishnan (2002) vs
   Dhankar & Maheshwari (2014)** — Indian short-horizon reversal: reversal for
   3 months (Chowdhury & Mitchello) versus continuation (Sehgal & Balakrishnan,
   Joshipura, Dhankar & Maheshwari). Different samples (254 vs 364 vs 328
   stocks), different periods (1991-2006 vs 1989-1999 vs 1997-2013), different
   universes (NSE vs BSE). **Dhankar & Maheshwari's own literature table lists
   the Chowdhury result as contrary.** I do not resolve this. The *Journal of
   Financial Markets* 2023 paper's liquidity conditioning is the most
   plausible reconciliation — it explains why studies using small/illiquid
   universes find reversal and studies using broad/liquid ones do not — but it
   is a hypothesis, not a demonstrated reconciliation.
2. **Bollinger Bands**: Tadas et al. (2023) and the 2025 BB-reversal study say
   band reversion works; Pillai (2026) says both BB variants fail correction;
   the ijier.um.edu.my 10-year study says BB underperforms buy-and-hold. These
   are **not reconcilable on the evidence available**. The differences are
   venue quality, sample size and horizon, and I am not going to pick the
   convenient one.
3. **Intraday reversal at the open** (IJFM: +5.72% → −0.76% within 2.5 hours)
   versus **Siddiqui & Misra** (intraday overreaction predicts *positive*
   next-day returns). One says reversion within the day; the other says
   continuation into the next day. **Both can be true** — a within-day fade
   followed by a next-day continuation — but the first source's sample period
   is unverified and its venue is predatory-adjacent, so I weight it low.
4. **RSI 30/70 in the US**: Anderson & Li (2015) finds −3,009 pips on USD/CHF;
   Bulkowski's US benchmark finds 15.5% net profit over 15 years. The
   reconciliation is that Bulkowski's rule is **not** "buy below 30" — it is
   "buy below 30 **then back above it**" — and his trades can span a year.
   Anyone quoting Bulkowski as validation of Wilder's rule is quoting the wrong
   rule.
5. **Rink's sample composition.** technical-indicators-nse.md records that
   India is not in Rink's 41-index sample. I could not independently verify the
   market list in this review and am inheriting the claim.

## Sources consulted

### Peer-reviewed, Indian data

- Pillai, S. (2026). *Technical-Indicator Strategy Performance in Indian
  Equities: A Joint Test of Market Efficiency, 2015-2025*. DOI
  10.13140/RG.2.2.21259.66087. **WORKING PAPER**, independent researcher, 0
  citations. Author is not institutionally identifiable.
- Sehgal, S. & Gupta, M. (2007). *Tests of Technical Analysis in India*.
  **Vision** 11(3), 11-23. 69 large companies, 1999-2004.
- Tadas, H., Nagarkar, J., Malik, S., Mishra, D.K. & Paul, D. (2023). *The
  effectiveness of technical trading strategies: Evidence from Indian equity
  markets*. **Investment Management and Financial Innovations** 20(2), 26-40.
- Poonam & Malik, N.S. (2025). *Risk-Return Efficiency of Stochastic
  Oscillator Strategies in BRICS During COVID-19 and Geopolitical Conflicts*.
  **Indian Journal of Finance**. **Volume/issue/pages NOT VERIFIED** — a
  related Poonam & Malik 2025 paper occupies 19(1), 52-66.
- Nair, S.T.G., Raghunandanan, A. & Nair, R.K. (2024). *Make Some 'Noise'
  Tonight — Overnight Return as Investor Sentiment Index: Indian Evidence*.
  **Vision**. 612,066 firm-days, BSE + PROWESS, 2016-04-01 to 2021-12-31.
- Siddiqui, A.A. & Misra, A.K. (2025). *Intraday overreaction and
  underreaction: profitability analysis and factor explanations*. **Journal of
  Asset Management** 26(5), 523-534. **Effect sizes NOT VERIFIED** (paywalled;
  abstract only).
- *Momentum, reversals and liquidity: Indian evidence*. **Journal of Financial
  Markets** (Elsevier), S0927538X23002640. 3,956 BSE stocks, 2000-2021.
  **Author names NOT VERIFIED.**
- Dhankar, R.S. & Maheshwari, S. (2014). *A study of contrarian and momentum
  profits in Indian stock market*. **International Journal of Financial
  Management** 4(2), 40-54. 328 NSE stocks, 1997-01 to 2013-03.
- Dhankar, R.S. & Maheshwari, S. (2016). *The Long-Run Return Reversal Effect:
  A Re-Examination in the Indian Stock Market*. 470 BSE stocks, 1997-01 to
  2013-03. **Journal NOT VERIFIED** (read via journals.uvu.edu JBI).
- Dhankar, R.S. & Maheshwari, S. (2017). *Profitability of Volume-based
  Momentum and Contrarian Strategies in the Indian Stock Market*. **Global
  Business Review** 18(4), 974-992.
- Sehgal, S. & Balakrishnan, I. (2002). *Contrarian and Momentum Strategies in
  the Indian Capital Market*. **Vikalpa** 27(1), 13-19. 364 BSE stocks,
  1989-07 to 1999-03.
- *Momentum returns: A portfolio-based empirical study*. **Journal of
  Management & Organization** (2017), S0970389617301647. NSE, 2005-2015.
  **Author names NOT VERIFIED.**
- Sehgal, S. & Bijoy, K. (2015). *Stock Price Reactions to Earnings
  Announcements: Evidence from India*. **Vision** 19(1), 25-36. 469 companies,
  2002-12 to 2011-12. Strong **continuation**, not reversal.
- Sehgal, S., Jain, S. & Subramaniam, S. (2021). *Are Prominent Equity Market
  Anomalies in India Fading Away?* **FINDINGS NOT RETRIEVED.** Listed as a
  required check for the reversal-fading question.
- Joshipura (2009). NSE, 1995-2008. **FINDINGS NOT RETRIEVED at source** — read
  only via other papers' summaries.
- *Post-Earnings-Announcement Drift Anomaly in India: A Test of Market
  Efficiency*. **Theoretical Economics Letters** 8(14), 2018. 2002-2017.
  Venue weak; not used for any headline number.
- Narayan, P.K., Ahmed, H.A., Sharma, S.S. & K.P., P. (2014). *How profitable
  is the Indian stock market?* **Pacific-Basin Finance Journal** 30, 44-61.
  50 NSE stocks, Jan 2001 – Dec 2012. Uses **momentum** rules (MA spread,
  9/12-month price, 110%-of-average filter), **not oscillators**. Listed
  because it is a correctly cited Indian technical paper that does **not**
  support oscillator claims.

### Peer-reviewed, non-Indian (transfer assumptions labelled in text)

- Anderson, B. & Li, S. (2015). *An investigation of the relative strength
  index*. **Banks and Bank Systems** 10(1), 92-96. USD/CHF.
- Rink, K. (2023). *The predictive ability of technical trading rules*.
  **Financial Markets and Portfolio Management** 37(4), 403-456. 6,406 rules,
  600 of them RSI. India not in sample (inherited claim, not re-verified).
- Sehgal, S., Jain, S. & Subramaniam, S. (2012) — cited as the source of the
  "no short-run reversal at 1-year horizon" finding; primary not retrieved.
- *An Assessment of the Impact of the T+0 Settlement Cycle on Market Liquidity
  and Price Efficiency in Indian Markets*. **Economic and Political Weekly**
  (2025), 22.

### Primary regulatory sources — the strongest material in this review

- **SEBI Circular SEBI/HO/47/11/11(3)2025-MRD-POD2/I/2765/2026**, 16 Jan 2026,
  *Introduction of Closing Auction Session (CAS) in the Equity Cash Segment*.
  Effective 3 Aug 2026. Full text read.
- **SEBI order, August 2026** (ORDER_1787144554.pdf) — records the Sensex IEP
  anomaly of 3 Aug 2026 and the 13 Aug 2026 CAS close of 78,079.96 against a
  77,829.6 reference. Full text partially read.
- **SEBI Consultation Paper, September 2026**, *Review of Certain Aspects of the
  Closing Auction Session, Market Timings and Settlement Methodology for
  Derivatives Contracts*. Blended-VWAP and CTS-VWAP settlement options; CAS
  window to 10 minutes.
- **SEBI Consultation Paper, May 2024**, optional T+0 and instant settlement.
  Delivery-share percentages by value at BSE and NSE; FPI/individual matching
  shares; 30-35 bp arbitrage cost estimate.
- **NSE Circular NSE/CMTR/73362** and **BSE Notice 20260610-41** — CAS
  operational modalities. Full text read.

### Working papers and preprints

- Chowdhury, S.S.H. & Mitchello, F.A. (2008). *Presence and sources of momentum
  and contrarian profits: evidence from the Indian stock market*. Job-market
  paper, MTSU. **A peer-reviewed version, if one exists, was NOT FOUND.**
- Jain, A. (2026). *Survivorship Bias in Indian Equities Is Not a Number*.
  **SSRN 7099378**.
- *Unlocking Liquidity through Shortened Settlement Cycle: Empirical Evidence
  from India*. **SSRN 4745303**.
- arXiv 2603.19380, *Survivorship Bias in Emerging Market Small-Cap Indices:
  Evidence from India's NIFTY Smallcap 250*. **PREPRINT.**
- SSRN 5807282, *Does overnight return predict the first half-hour return for US
  market indices*. High overnight return **negatively** predicts the first
  half-hour for US indices. **US only** — the closest published analogue to an
  Indian overnight reversion test, and it does not transfer.
- SSRN 4069509, *What Drives Momentum and Reversal? Evidence from Day and Night
  Signals*. 1926-2019 US: portfolios on past **intraday** returns show momentum
  **without** long-term reversal; portfolios on past **overnight** returns show
  **no momentum**. **US only.** Directly relevant to the Indian overnight
  result and points the same way.

### Sources I could not verify

- RPS / learn-portal (2023), *Investigating the Efficacy of RSI in the Nifty 50
  Index*. Abstract read in full; **peer-review process not established**.
- kommerstad.org, *RSI as a Predictor of Overbought and Oversold* (defence
  stocks, 2015-2025). Full text read; **journal and peer review not
  established**. Claimed 21.66% mean return is internally implausible.
- JMRA 8(4), *Trend identification with the relative strength index*. Venue not
  established. Result (both p > 0.05) is used only as a null.
- Indian Journal of Computer Science (2023), Singh & Pillai, RSI across Saudi
  Arabia, India and China. **Indian-specific effect size not recoverable.**
- JTAR (2026), RSI vs StochasticRSI on 23 Indian equities. Authors state no
  transaction costs and a small sample.
- Ranade (2020), *International Journal of Management* 11(8), 300-310. Read in
  full; not a statistical test.
- IOSR-JBM 19(9), five indicators on SBI. Read in full; n=1, no costs.
- turcomat.org, 16 Nifty sectoral indices 2016-2021. Venue not peer-reviewed.
- ijier.um.edu.my, Bollinger Bands vs MA envelopes, 50 Nifty stocks, 10 years.
  Venue not established.
- "A Comparative Study on Predictive Performance of Bollinger Band Indicator: An
  Indian Market View" (2025). Published in a **journal of informatics education
  and research**; not a finance venue.
- *Prior Return Effect in Indian Stock Market: An Intra-day Analysis*. **Sample
  period could not be determined.** Publisher is predatory-adjacent.
- academia.edu, *Profitability of Oscillators Used in Technical Analysis for
  Financial Market* (Nifty 50, 2004-2014). **No journal, no identifiable
  authors.**
- Griffin, Kolodny & Lattimore (2010). The claim that they *"find that the
  short-term reversal is significant in India"* was **read at one remove** in
  the *Journal of Financial Markets* paper. **Not verified at source.**
- Medhat & Schmeling (2022), *JFE*: short-term reversal strong among
  low-turnover US stocks; the reversal sort's excess return is **−16.9% p.a.**
  **US. Reached India only through the JFM paper's discussion.** Its negative
  excess return is a useful reminder that a statistically real reversal need
  not be a profitable trade.
- ADBI Working Paper 1333 (FII flow innovations). Read in full; **working
  paper**, ADB Institute, not peer-reviewed.
- Amrita Vishwa Vidyapeetham, BB on tick-by-tick NSE data. **Conference
  abstract only**; no statistics retrievable.
- Dhananjaya & Wright (2019), *Wealth* 8(1), 65-72 — abstract read.
- Press coverage of CAS (Times of India, Economic Times, Fortune India, The
  Print/New Indian Express, News18, New Kerala carrying a Goldman note).
  **Secondary sources**; figures attributed to named individuals or firms
  within them. Used only for magnitude-of-disruption context, never for a
  performance claim.

## Next steps worth taking

1. Get the **Pillai (2026) full result table**. It is the only corrected Indian
   test of these rules and its Sharpe CIs are the most useful numbers available.
   Verify whether it has since been published.
2. **Replicate the RSI threshold surface on point-in-time Indian data** with a
   Romano-Wolf or Hansen SPA correction, declared in advance. This is the single
   experiment that would settle question 1.
3. **Read the Siddiqui & Misra (2025) full text** for magnitudes.
4. **Retrieve Sehgal, Jain & Subramaniam (2021)**, *Are Prominent Equity Market
   Anomalies in India Fading Away?* — the direct test of whether reversal has
   decayed.
5. **Find whether Chowdhury & Mitchello (2008) was published.** The reversal
   question turns on this.
6. **Do not model the auction era with pre-auction data in the same series.**
   Segment at 3 Aug 2026 and state it.