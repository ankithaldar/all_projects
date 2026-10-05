# Indicator-Based Trend and Momentum Strategies on NSE

Extends [technical-indicators-nse.md](technical-indicators-nse.md), which
established that the *thresholds* behind this project's indicator ranking have
no published Indian evidence. This file asks a narrower question: **does the
trend/momentum family itself have Indian evidence, and how strong is it?**

Method note: `websearch` **did** work for this review, unlike the previous
one. Sources are OpenAlex-adjacent aggregators (RePEc/IDEAS, Semantic
Scholar, Exa), publisher pages, and direct PDF reads where reachable. SSRN
full texts returned HTTP 403 throughout and are cited from abstracts only.

## Executive summary

1. **Momentum is the best-evidenced indicator family in India, and it is
   genuinely Indian — not a transfer from the US.** Replicated across BSE
   and NSE, cross-sectional and time-series forms, by more than ten
   independent groups including IIM Ahmedabad's factor library, IIMB, FMS
   Delhi and IIM Trichy, spanning 1989 to 2025.
2. **But the evidence is almost entirely gross of costs and gross of the
   long-only constraint.** The one cost-aware, cost-aware-family paper
   (Mitra 2011) finds MA profits do not survive Indian costs. Every
   momentum study with a large reported effect size either uses long-short
   (not deliverable in India) or omits costs entirely.
3. **50/200 now has Indian data — but only from two non-peer-reviewed
   sources, and both are weak.** A Zenodo/JOIREM paper finds 50/200 on
   Nifty 50 *underperforms* buy-and-hold (4.01% vs 9.90% CAGR). A 2026
   arXiv-free ResearchGate working paper finds 50/200 survives a
   Romano–Wolf multiple-testing correction on 78 stocks. They contradict
   each other. **Confidence in 50/200 specifically remains very low.**
4. **Trend-following with volatility scaling has essentially no Indian
   equity evidence.** The two Indian papers on "risk-managed momentum" use
   a *market-state screen*, not inverse-volatility sizing. Volatility
   scaling of Indian equities is a **transfer assumption**, unverified.
5. **Breakout/Donchian has one Indian peer-reviewed negative result**
   (Trading Range Breakout test on BSE-200, 2000–2010: null accepted in
   100% of tests) and one non-peer-reviewed positive result. Net: weak.
6. **An inversion *does* exist in Indian price indicators, but only at
   high frequency and in the most recent window.** Devulapally &
   Tripurana (arXiv 2023) find daily-horizon momentum *reversed* on NSE
   500, 2014–2021. A 2025 low-tier paper finds losers beat winners on
   NSE 500, Jul 2024–Jun 2025 (t = 5.03). Neither is a clean inversion
   of the 12-1 signal.
7. **Correction to the existing research.** Rink (2023) **does** include
   India — BSE Sensex, 03/04/1979–31/05/2016. The previous file states
   "India is not in his sample." That is wrong. See below.

## The evidence table

| Strategy | Evidence strength | Sample | Market | Reported effect size | Source | Peer-reviewed? |
|---|---|---|---|---|---|---|
| 12-1 / 6-12 cross-sectional momentum | **Strong (gross)** | 1997–2013 | NSE, 328 stocks | WML abnormal return up to 7.7% at 6×9 formation/holding; peak at ~6-month formation, decays beyond | Dhankar & Maheshwari 2014, *IJFM* 4(2) | Yes |
| 12-1 cross-sectional momentum, long-only | **Strong (gross)** | 1990–2026 | NSE Nifty-100 | +10.70%/yr over Nifty-100; mean turnover 32.1%/month | Raju & Chandrasekaran 2019, SSRN 3510433 | No (working paper) |
| Cross-sectional momentum, liquidity-conditioned | **Moderate–strong** | 2000–2021 | BSE, 3,956 stocks | Significant intermediate and long-term momentum; **strongest in the most liquid portfolio**, persistent 12 months | Chui, Ranganathan, Rohit & Veeraraghavan 2023, *PBFJ* 82:102193 | Yes |
| Cross-sectional momentum, cost-of-broker argument | **Moderate** | 2022–2023 | NSE 500 | Momentum anomaly significant under all four factor models; does not reverse under VIX stress | Kalyanaraman & et al., *J. Asset Mgmt* (SAGE), online 2020 | Yes |
| Time-series (absolute) momentum 12-month, skip 1 | **Moderate (gross, long-short)** | Jan 1996 – Dec 2020 | BSE, 441 stocks | 12×1 excess return **1.26%/month**; 12×3 1.357%/month; 4 of 5 horizons significant; survives CAPM and FF3 | Singh, Walia, Bekiros, Gupta, Kumar & Mishra 2022, *JEFAS* 27(54):328–343 | Yes |
| Time-series momentum, crash behaviour | **Moderate** | same | BSE | **−56% cumulative in Apr–May 2009**; 3 of 5 worst payoffs in market *recovery*, not the crash | Singh et al. 2022, *JEFAS* | Yes |
| 50/200 SMA crossover, Nifty 50 index | **Negative** | Jan 2010–Jun 2022, 3,077 days | Nifty 50 | **4.01% vs 9.90% CAGR**; vol 12.54% vs 22.52%; **lower Sharpe than buy-and-hold** | Lakshan A. 2026, *JOIREM* 4(7) / Zenodo 21308597 | No (Zenodo DOI, low-tier journal) |
| 50/200 SMA crossover, stock level | **Contested positive** | Jan 2015–Apr 2025 | NSE, 78 F&O-eligible stocks | Survives Romano–Wolf StepM: adjusted **p = 0.0094**; nominal trend-follower alphas +4.93 to +7.12 pp over Nifty 50 | Pillai 2026, ResearchGate WP | No (working paper, 0 citations) |
| Dual MA system (many pairs) | **Mixed** | BRICS, 4,428 strategies | India (755 stocks) | India among highest average returns, but *"few combinations of moving averages were able to outperform buy and hold"* | Souza et al. 2018, *Financial Innovation* 4(1) | Yes |
| MA rules, cost-aware | **Negative on cost** | Dec 2000–Nov 2010 | Nifty, Nifty Junior + 2 more | Gross positive; breakeven one-way cost 0.15%–2.27% by lag; 6d/12d and 6d/40d EMA pairs **lost 13.1% and 7.7% per year before costs** | Mitra 2011, *IJBM* 6(7) (OA companion to *Quant. Finance* 11(2)) | Yes |
| MA rules, beta-conditioned | **Moderate** | 27 yrs, NSE 500 | NSE 500 | Mid-beta alpha 7–14% gross; **6–11% net of costs** at 20d/50d | Bhama 2025, *IMFI* 22(4):260–275 | Yes |
| Trading-range / Donchian breakout | **Negative** | Apr 2000–Mar 2010 | BSE 200, 200 stocks | Null (weak-form efficiency) accepted in **all** Buy / Sell / Buy-Sell tests and all 7 sub-periods | Sapate 2014, *AIMS* 14(4) | Yes (weak journal) |
| Donchian 20 breakout | **Positive, uncorrected-strong** | 2015–2025 | NSE, 78 stocks | Sharpe **1.26**, Romano–Wolf adj. **p = 0.0006** — best in the paper's universe | Pillai 2026 WP | No |
| Volatility-scaled trend following | **None found** | — | — | — | — | — |
| Inverse-vol weighting inside a momentum book | **Weak (working paper)** | 2005–2021 | NSE 200/500/750 | IVW portfolios show **highest Sharpe across all universe sizes**; differences vs market-cap weighted "often statistically significant" | Raju 2023, SSRN 4607933 | No (working paper) |
| Momentum + low-vol combined, long-only | **Moderate** | Jan 2004–Dec 2018 | NSE top-500 | Low-risk + high-momentum: **+~5 pp CAGR over pure low-risk**, no volatility increase | Joshipura & Joshipura 2020, *IMFI* 17(2):128–145 | Yes |
| Momentum as an official index (live AUM) | **Real-world** | Apr 2005 – Feb 2026 | NSE | Nifty200 Momentum 30 **19.19% vs Nifty 200 15.08%** since inception; gap 3–6 pp; beat parent in **15 of 20** calendar years | NSE Indices whitepaper, Apr 2026 | No (index provider; marketing) |
| Trend following, index level, 80 algorithms | **Negative except in crashes** | 2005–2012 | Nifty | Worked in *sharply declining* markets; "far less so or not at all" in rising, mixed, or trendless markets. **No transaction costs assumed** | Slivka, Keswani & Li 2014, *Indian J. Finance* 8(1) | Yes |
| Momentum 50/200 EMA on Nifty-50 names | **Negative** | Jan 2006–Dec 2017 | NSE Nifty 50, 31 names | Aggregate return of EMA 5-20, 5-50 and **5-200** momentum buys/sells "insignificant in most cases" | (Academia.edu upload, authorship unverified) | **Unverified** |
| Reversal (contrarian) instead of momentum, recent | **Positive — the inversion** | Jun 2024–Jun 2025 | NSE 500 | Losers **1.40%/month vs winners 0.17%/month**; paired t = 5.03, p = 0.0002 | "Behavioural Biases and Market Efficiency", *IJES* 11(19s) 2025 | No (very low tier) |
| Momentum INVERTED at daily horizon | **Inversion found** | 2014–2021 | NSE 500 | Best daily-horizon portfolio is **contrarian**, 16× initial investment; *"the loser group shows stronger momentum continuation, while the winner groups show a strong reversal"* | Devulapally & Tripurana 2023, arXiv:2302.13245 | No (arXiv preprint) |
| Trend following with vol scaling, Indian futures | **Weak (working paper)** | sample not verified | Indian equity futures | Momentum factor average annual return **21.9%**; optimal long horizon 1 year, short 1 month | Srivastava, Chakravorty & Singhal 2019, SSRN 3345280 | No (working paper) |

## Question 1 — Time-series momentum, 12-1 or 6-12, skipping the last month

**The strongest single Indian data point is Singh et al. (2022), *JEFAS*.**
441 BSE stocks, monthly adjusted closes, January 1996 to December 2020,
long-short absolute-momentum portfolios built on Moskowitz–Ooi–Pedersen
construction with a **one-period gap** between lookback and holding
(they cite Lehmann 1990 for it — the skip is explicit). Reported:

- 12-month formation, 1-month holding: **1.26% excess return per month**
- 12×3: **1.357%/month**, the best of five holding periods
- 4 of 5 tested horizons significant; significant after CAPM and after FF3
- Newey-West t-stats used for inference

That is a **gross, long-short** number. Two things make it non-deliverable:
the short leg does not exist in Indian delivery, and the paper **explicitly
does not model transaction costs** — it lists this as a limitation.

### Comparison with US/developed evidence

Direct numbers, from the sources themselves:

- Moskowitz, Ooi & Pedersen (2012), 58 futures, 1985–2009: the founding
  result; gross, no Indian analogue per contract.
- Hurst, Ooi & Pedersen, "Trends Everywhere" (AQR, SSRN 3386035):
  12-month TSMOM **gross Sharpe 1.17** on traditional assets; **1.60** for a
  diversified book. The 7 emerging-market equity index futures include
  **SGX CNX Nifty** — the only Indian instrument in the canonical TSMOM
  evidence. **The alpha there is index-futures-level, not Indian
  single-stock, and is gross.**
- Singh et al. 2022 report *returns*, not Sharpe ratios, in the abstract.
  Their best risk-adjusted figure is a CVaR improvement, not a Sharpe.
- Comparison of 21.9%/yr (Srivastava et al., Indian futures) against 12.5%/yr
  for 12-month TSMOM in the AQR paper suggests the Indian *futures* number
  is not an outlier, but the two are not measured the same way and the
  comparison should not be leaned on.

**Honest read:** Indian time-series momentum is *reported* to be roughly
comparable to developed-market magnitudes, but the comparison is not
clean, and every Indian number is gross.

### Is it decayed in recent data?

**Genuinely contested. Do not pick a side.**

Evidence for decay:

- **Sharma, Subramaniam & Sehgal (2021), *Global Business Review* 22(1)**,
  NSE 500, **July 2005 – June 2016**: the momentum anomaly **is explained by
  risk models** — "contrary to prior evidence." Their own framing: the
  Indian market "seems to be informationally more efficient... in line with
  recent financial market reforms." This is the single most direct
  Indian statement that momentum's abnormal return has gone.
- **Rink (2023), *FMAPM* 37(4)**, includes **India, BSE Sensex,
  03/04/1979–31/05/2016** (see correction below): predictability
  "diminishes drastically over time in all markets"; in the final
  sub-period 2009–2016 **"almost all emerging market indices are
  unpredictable."** Only 4 of 18 emerging markets retain any significant
  rule at ≥20 bp single-trip cost.
- **A 2025 low-tier reversal study** on NSE 500, Jul 2024–Jun 2025,
  finds losers beating winners at t = 5.03.
- **Raju's "Shades of Momentum"** (SSRN 4977717, working paper, abstract
  only) finds momentum effects **weaken significantly as the holding period
  lengthens** — the opposite of the "persist with long holds" claim.
- **Live funds corroborate the recent softness independently of any
  backtest.** Per *Mint* (7 Oct 2025): Motilal Oswal Nifty 200 Momentum
  30, UTI Nifty200 Momentum 30 and Edelweiss Nifty Midcap150 Momentum 50
  all returned **−7% to −15% over the preceding year**, while value funds
  returned +3% to +5% and the Nifty 50 returned ~+1.5%. That is
  realised, point-in-time, non-backtested data and it is the highest-
  quality recent evidence available.

Evidence against decay:

- **NSE Indices' own official momentum indices**, Apr 2005 – Feb 2026:
  Nifty200 Momentum 30 19.19% vs Nifty 200 15.08% since inception; beat the
  parent in **15 of the last 20 calendar years**. Caveats: it is a
  volatility-adjusted *ranking* score with a 6-month component, not a pure
  12-1 signal; the provider has a commercial interest; no transaction
  costs.
- **Chui et al. (2023), *PBFJ***, sample to 2021, finds momentum
  *persists for the next 12 months* and is **strongest in the most liquid
  names** — i.e. in exactly the Nifty-50-class universe this project cares
  about. This directly contradicts the illiquidity story below.
- **Kedia & Satpathy (2024)**, 232 Nifty-500 firms, **July 2015 – June
  2024**, i.e. almost entirely post-2016: WML Sharpe **6.21 (3×3) to 8.78
  (6×12)**, monotonic in horizon. Journal is *Educational
  Administration: Theory and Practice* — a pedagogy journal, not finance,
  and a WML Sharpe of 6–9 is implausible as a cost-aware number. Treat as
  unverified-strength.
- **BacktestIndia** (Desai, non-peer-reviewed, Dec 2006–Jun 2025, claims
  survivorship-mitigated with 1,700+ stocks incl. delisted): **14.01% net
  CAGR vs Nifty 50's 10.42%**, +3.59%/yr alpha *after* Indian LTCG/STCG
  and 0.11%/trade costs. Sharpe 0.35 net. But see the liquidity split
  below — this number does not survive its own decomposition.

## Question 2 — Moving-average crossovers, 50/200 and dual-MA systems

### The two studies that actually test 50/200 on Indian data

Both are new relative to
[technical-indicators-nse.md](technical-indicators-nse.md), and **they
disagree**.

**Lakshan (2026)** — Nifty 50 index, 3,077 daily observations, Jan 2010 –
Jun 2022. 11 Golden Crosses and 11 Death Crosses. The mechanical 50/200
rule **underperformed buy-and-hold substantially: 4.01% vs 9.90% CAGR**,
with materially lower realised volatility (12.54% vs 22.52%). The paper's
own conclusion is that the lower risk *does not* compensate: **lower Sharpe
than holding the index.** It also notes the signal's unavoidable lag.
Peer review: **no** — Zenodo DOI, *Journal of International Research for
Engineering & Management*, ISSN 3107-6696, 0 downloads at time of access.

**Pillai (2026)** — 78 NSE F&O-eligible stocks, Jan 2015 – Apr 2025,
13 configurations from six families, **Romano–Wolf StepM stepdown** at 5%
FWER, 0.05% round-trip cost. **SMA_50_200 survives with adjusted p = 0.0094.**
Donchian_20 is the paper's best (Sharpe 1.26, adj. p = 0.0006) and
SMA_10_50 ties it (Sharpe 1.26, 12.14%/yr, +7.12 pp over Nifty 50).
Notably the **multi-indicator confluence rule — a 200-day SMA trend filter
plus 12/26 EMA plus a 14-period ATR filter — FAILS** (Sharpe 0.53, adj. p =
0.209) at the highest turnover in the universe (12.36 round trips/stock/yr).
Peer review: **no** — independent researcher, 0 citations, ResearchGate
self-post, May 2026.

Why they disagree: one is index-level with ~22 events over 12 years (a
sample of 22), the other is a stock-level average over 2,586 days. Neither
is the design the other ran. **Do not treat this as a resolved question.**

### Does the parameter sensitivity imply a fitted effect?

**Where we can measure it, the sensitivity is severe enough to be
diagnostic.**

- **Mitra (2011)**'s breakeven-cost table, already reproduced in
  [technical-indicators-nse.md](technical-indicators-nse.md), moves from
  0.15% one-way for SMA(3) to 1.24% for SMA(60) — a **8× swing driven
  entirely by lag length** — and the two fastest EMA pairs are *negative*.
  That is not a smooth surface. It is a landscape with holes.
- **Bhama (2025)**: only 20-day and 50-day windows retain net positive
  alpha after costs (6–11%); 5- and 10-day are wiped out; the lowest-beta
  portfolio has **negative alpha in all 5-day and 10-day lags**.
- **Souza et al. (2018)**: 4,428 combinations. India had among the highest
  average returns of the five BRICS, but only **13.83% (SMA/SMA),
  14.12% (EMA/EMA), 13.47% (SMA/EMA) of individual stocks beat
  buy-and-hold.** And raising brokerage from 0% to 2% to 5% *"shifted
  significantly the range of the short-term MAs that were better."* A
  result that moves when you change the cost assumption is a fit to the
  cost assumption.
- **Slivka, Keswani & Li (2014)** ran **80 trend-following algorithms** on
  Nifty over 2005–2012 with a Levich–Thomas bootstrap and found the family
  works **only in sharply declining markets**. 80 rules, one conclusion:
  the effect is a function of which regime you sample, not of the rule.

**This is the strongest argument in the file.** Five independent Indian
studies, four different designs, all report that the trend result is
*conditional on lag, on cost assumption, or on regime.* That is the
signature of a fitted effect, not a structural one. It also matches the
multiple-testing failures reported elsewhere in this repo — Souza's 4,428
and Sobreiro's 1,581×3 rules were never subjected to a Reality Check or
SPA test.

## Question 3 — Trend following with volatility scaling

**There is no Indian equity study of volatility-scaled trend following that
I could find.** Stating that plainly rather than substituting US evidence.

What exists in India is adjacent and is frequently confused with it:

- **Singh, Walia, Panda & Gupta (2022), *FIIB Business Review* 11(3)**,
  450 BSE stocks. "Risk-managed momentum" **doubles the adjusted Sharpe
  ratio** of relative momentum and improves downside risk. The mechanism is
  a **market-state screen** (Cooper–Gutierrez–Hameed style: compare lagged
  1-period vs 2-year market return to decide normal/bullish/bearish, then
  go long-only or short-only). It is **not** inverse-volatility sizing.
- **Singh, Walia, Bekiros, Gupta, Kumar & Mishra (2022), *JEFAS* 27(54)**,
  the time-series version of the same idea. Their rule lifts the 12×1
  return from **1.26%/month to 3.214%/month** and roughly **triples the
  adjusted Sharpe**; CVaR improves from **−22.610% to −14.505%** for the
  12×1 case. Robust to a 36-month signal window, three sub-periods
  (1996–2003, 2004–2011, 2012–2020), and an NSE-listed replication
  (2005–2020). This is the **best-documented risk-management result in
  Indian momentum** — and it is a *regime* overlay, not a *sizing* overlay.
  Their own caveat: profits come substantially from the **long** leg.
- **Joshipura & Joshipura (2020), *IMFI* 17(2)**: momentum as a *filter on
  a volatility-sorted book* adds ~5 pp CAGR. This is volatility sorting,
  not volatility scaling.
- **Raju (2023), SSRN 4607933**, weighting schemes in an Indian momentum
  book: **inverse-volatility weighting gives the highest Sharpe across all
  universe sizes and holding counts**, with differences from market-cap
  weighting "often statistically significant." Closest thing to the right
  question. Working paper, abstract/full text only.

**Transfer assumptions** (label these if you use them):
Moreira & Muir (2017) and Barroso & Santa-Clara (2015) are the origin of
the claim. They are US-only, and Cederburg, O'Doherty, Wang & Yan (2020),
*JFE* 138(1):95–117, **finds no systematic benefit** across 103 equity
strategies — volatility-managed versions beat unscaled in 53 cases and
lose in 50, with only 8 statistically significant. Their own point is that
Moreira & Muir's spanning-regression alphas are **not implementable in real
time**. Any Indian volatility-scaling claim inherits that unresolved
dispute. Say so out loud.

## Question 4 — Breakout / Donchian / ATR-based systems

- **Trading Range Breakout test, BSE 200, 200 stocks, Apr 2000 – Mar 2010**
  (Sapate, *AIMS* 14(4), 2014): Buy, Sell and Buy–Sell TRB rules tested
  over the full 10 years and seven sub-periods. **The null of weak-form
  efficiency was accepted in every single test.** Buy accepted at 93.33%,
  Buy–Sell at 85.64%. The paper's conclusion: TRB rules "cannot produce
  economically significant returns relative to the buy and hold strategy."
  This is the only peer-reviewed Indian Donchian test I found, and it is
  negative.
- **Pillai (2026)** finds Donchian_20 and Donchian_55 both survive
  Romano–Wolf on 78 stocks, 2015–2025. Non-peer-reviewed, and it flatly
  contradicts Sapate — plausibly because Sapate is 2000–2010 and Pillai is
  2015–2025, i.e. **the difference is a decade of market maturation, not
  a rule.**
- **ATR as a directional signal: zero Indian evidence found.** Consistent
  with [technical-indicators-nse.md](technical-indicators-nse.md). The only
  Indian ATR study located is a 15-minute intraday Super Trend backtest
  across 5 Nifty-50 names, 2019–2025 — not a production basis, and its own
  literature review cites works I could not independently verify.
- Notably, Pillai's **ATR-filtered confluence rule fails** (adj. p = 0.209)
  while the plain crossovers pass. That is weak evidence *against* ATR as a
  signal filter, from the one source that tested it under a correction.

## Question 5 — Transaction costs

Delivery round trip is **22.25 bps**, 90% of it STT
(see [india-transaction-costs.md](../compliance/india-transaction-costs.md)).
One-way is ~11–13 bps. Now against each strategy:

**12-1 momentum, monthly rebalance, top 30 of 200.** Raju &
Chandrasekaran report **mean turnover 32.1%/month**. At 32% one-way
turnover × 12 months × ~12 bps one-way ≈ **0.46%/yr of cost drag**; with
5 bps/side slippage add roughly 0.4 pp, so **~0.8–0.9%/yr**. Against a
reported gross alpha of **+10.70%/yr**, that is not fatal — it survives.
This is the cost arithmetic that actually works in India, and it works
because the turnover is low.

**The same strategy semi-annually.** BacktestIndia reports **1.22%/yr of
tax drag alone** at 6-month rebalancing, because most positions exit
inside 12 months and are taxed at STCG 20% rather than LTCG 12.5%.
**Tax, not STT, is the binding constraint on Indian momentum.** No
peer-reviewed Indian momentum paper I found models capital-gains tax at
all. This is a real gap in the literature.

**MA crossovers.** This is where costs kill the family, and it is measured,
not speculated. From Mitra (2011), already tabulated in
[technical-indicators-nse.md](technical-indicators-nse.md):

| Rule | Breakeven one-way | Viable at ~22 bps delivery? |
|---|---|---|
| SMA(3), 86 trades/yr | 0.15% | No |
| SMA(20), 20 trades/yr | 0.72% | Marginal |
| SMA(60), 14 trades/yr | 1.24% | Yes |
| EMA 6d/12d, 5.8 trades/yr | **negative** | **Lost 13.1%/yr gross** |

The crossover family lives or dies on turnover, and only the slowest
crossovers clear Indian costs. Anything faster is dead on arrival.

**Rink (2023)**, the one study that does the cost sweep properly:
only **4 of 18 emerging markets** retain a significant rule at ≥20 bp
single-trip. Pillai (2026) used 0.05% round-trip — **4× lighter than the
real Indian delivery stack of ~22 bps plus slippage**. His surviving rules
would be under much more pressure at 22 bps. That is the most important
caveat on the single most favourable Indian study in this file.

**The Bottom line on costs:** monthly-rebalanced 12-1 momentum survives
Indian costs with room to spare. **Daily and short-lag MA crossovers do
not.** Everything in between is a parameter choice, which is exactly the
problem.

## Correction to the existing research: Rink (2023) includes India

[technical-indicators-nse.md](technical-indicators-nse.md) states, twice,
that Rink's sample **excludes** India, and builds on that: "India is not in
his sample."

**It is in the sample.** Rink (2023), *Financial Markets and Portfolio
Management* 37(4):403–456, Table 3 Panel B lists:

> IND | India | BSE Sensex | 03/04/1979 – 31/05/2016

of 18 emerging markets. The classification is IMF 2016.

Verification status: read from two independent full-text extractions of the
paper (a RePEc-hosted copy and a ResearchGate copy), not from the typeset
Springer PDF, which was behind a bot challenge. **Confident but not
first-hand from the publisher's PDF.** The BSE Sensex rather than the Nifty
is a real difference — BSE Sensex is the older, less institutionally
traded index.

Why this matters: the previous file's "where it is weaker" caveat about
Rink not covering India is void, and India's inclusion makes the
"emerging markets unpredictable in 2009–2016" finding apply directly.

## Does this NOT establish

**Not established:**

1. **That 50/200 works, or that it does not.** Two non-peer-reviewed
   sources disagree. No peer-reviewed Indian study of the 50/200 Golden/
   Death Cross on Nifty-50 data exists. Confidence: **very low**.
2. **That Indian momentum is comparable to US momentum.** No Indian study
   benchmarks an Indian strategy against a US one on a common footing. Every
   cross-country comparison in this file is mine and is a transfer
   assumption.
3. **That any momentum effect survives Indian costs and Indian tax in a
   long-only delivery book with peer-reviewed evidence.** The studies that
   claim a large net alpha either omit costs (Kedia & Satpathy), are
   working papers (Raju, BacktestIndia), or trade long-short (Singh et al.).
   **None of the peer-reviewed Indian momentum evidence is
   implementable-as-stated.**
4. **That volatility scaling helps Indian trend following.** No Indian
   study. Entirely inherited from the US, where the primary source's own
   main result has been substantially rebutted.
5. **That momentum has or has not decayed post-2015 in India.** Both
   directions have published support and the sources disagree sharply.
   Report both. Do not resolve it.
6. **That the momentum premium is behavioural.** Direct conflict inside
   the Indian literature: Sehgal, Pandey & Sen (2024/25) attribute it to
   investor overreaction; Kedia & Satpathy (2024) call it a *priced risk
   factor* because the alpha vanishes under Carhart; Maheshwari &
   Dhankar's crisis result implies a liquidity-timing effect; Chui et al.
   (2023) find it *strongest in liquid names*, which fits neither
   behavioural underreaction in illiquid stocks nor an illiquidity premium.
7. **Anything about Nifty-50 specifically as distinct from Nifty-500 or
   BSE-500.** Nearly all momentum evidence is on broad universes. The
   variance-ratio finding already in
   [technical-indicators-nse.md](technical-indicators-nse.md) says large
   caps are the *most* efficient segment. Weak Nifty-50 results are the
   expected outcome, not a bug.
8. **That the reported Sharpe ratios are achievable.** Two calibration
   facts from this repo apply directly. Grądzki (2026): selecting the
   best seed inflates Sharpe by ~44%. Boryszski et al.: SAC PPO-style
   walk-forward carries 17.5% PBO. Any Sharpe above ~1.0 in a published
   Indian momentum backtest should be assumed inflated until a trial count
   and seed dispersion are declared.

## The disagreement that matters most

**Chui et al. (2023), *PBFJ*, is the highest-quality study in this file**
(large sample — 3,956 BSE stocks to 2021; standard Lo–MacKinlay and
Jegadeesh–Titman decomposition; controls for market, size, value and
macroeconomic variables; peer-reviewed in a solid finance journal). It
finds momentum **strongest and most persistent in the most liquid
portfolio**.

**BacktestIndia** (unverified, non-peer-reviewed, but explicit about its
data including delisted names) reaches the *opposite* conclusion: splitting
Nifty200 Momentum 30 by scaled turnover, the **15 most liquid** momentum
stocks returned **8.51% net CAGR — below the Nifty 50's 10.41%** — while
the 15 least liquid returned **19.43%**. All the alpha in the illiquid half.

If Chui et al. are right, momentum is a real, liquid, exploitable effect in
the Nifty-50-class universe and this project should use it. If BacktestIndia
is right, Indian momentum is an illiquidity premium that a
Nifty-50-concentrated book cannot harvest at all — and would be actively
misleading. **A peer-reviewed replication of the Chui et al. liquidity
decomposition on NSE data is the single highest-value unfilled gap in this
file.** I did not find one.

## Confidence levels

| Claim | Confidence | Why |
|---|---|---|
| Indian momentum exists in cross-sectional form, gross | **High** | ≥10 independent groups, BSE + NSE, 1989–2025, several peer-reviewed |
| Indian momentum exists in time-series form, gross, long-short | **Moderate–high** | One strong peer-reviewed paper (JEFAS 2022), one positive working paper |
| Indian momentum has decayed post-2015 | **Low — genuinely contested** | Balanced published evidence both ways; don't resolve |
| 50/200 crossover has positive Indian evidence | **Very low** | Two non-peer-reviewed sources, they contradict each other |
| 50/200 crossover has *negative* Indian evidence | **Low–moderate** | One Zenodo source, index-level, n=22 events |
| Momentum survives Indian costs | **Low–moderate for 12-1 monthly; near-zero for fast crossovers** | Only Mitra measures breakevens; most momentum papers ignore costs |
| Momentum survives Indian capital-gains tax | **Unknown** | No peer-reviewed Indian momentum paper models it |
| Volatility scaling helps Indian trend following | **Unknown — no Indian study** | US evidence itself disputed (Cederburg et al. 2020) |
| Breakout/Donchian works on Indian equities | **Low** | One peer-reviewed negative, one non-peer-reviewed positive, different decades |
| Momentum is behavioural (underreaction/overreaction) | **Low** | Direct conflict across Indian sources; Chui et al. fits neither |
| Rink (2023) includes India | **High** | Table 3 Panel B read in two independent copies; not from publisher PDF |

## Every source consulted

**Read in full or near-full:**

1. Singh, Walia, Bekiros, Gupta, Kumar & Mishra (2022), *JEFAS* 27(54):
   328–343 — scielo.org.pe full text. Indian time-series momentum, the
   central positive result.
2. NSE Indices (Apr 2026), "The Science of Momentum" whitepaper — full PDF
   read. Official index performance since Apr 2005.
3. Singh, Walia, Panda & Gupta (2022), *FIIB Business Review* 11(3) —
   abstract + references via RePEc and SAGE.
4. Devulapally & Tripurana (2023), arXiv:2302.13245 — arXiv HTML full
   text. The daily-horizon inversion.
5. Bhama (2025), *IMFI* 22(4):260–275 — publisher page + full PDF body
   via exa. Beta-conditioned MA, cost-aware.
6. Joshipura & Joshipura (2020), *IMFI* 17(2):128–145 — publisher PDF
   body. Low-risk + momentum filter.
7. Lakshan (2026), Zenodo 21308597 / *JOIREM* 4(7) — record page, Zenodo
   and joirem.com. **The 50/200 Nifty test.** Full PDF not retrievable.
8. Pillai (2026), "Technical-Indicator Strategy Performance in Indian
   Equities" — ResearchGate abstract + extensive body text via search
   index. ResearchGate itself 403'd on direct fetch. **The 50/200 stock-level
   and Donchian test.**
9. Kedia & Satpathy (2024/25), *Educational Administration: Theory and
   Practice* 30(8) — kuey.net full PDF. Momentum 2015–2024.
10. Rink (2023), *FMAPM* 37(4) — full text read via two mirrors
    (econstor copy, RePEc/ResearchGate). Table 3, Table 5, methods.
11. Chui, Ranganathan, Rohit & Veeraraghavan (2023), *PBFJ* 82:102193 —
    abstract from publisher, ScienceDirect and Manipal repository.
12. Sharma, Subramaniam & Sehgal (2021), *GBR* 22(1):255–270 — full
    abstract via RePEc. The decay result.
13. Maheshwari & Dhankar (2017), *JAMR* 14(1) and (2017), *Vision* 21(1)
    — abstracts via RePEc, SAGE, Emerald.
14. Sehgal & Gupta (2007), *Vision* 11(3) — abstract via RePEc.
15. Slivka, Keswani & Li (2014), *Indian J. Finance* 8(1) — SSRN + ResearchGate
    abstract.
16. Sapate (2014), *AIMS* 14(4) — publisher PDF. TRB null result.
17. Mitra (2011), *IJBM* 6(7) / *Quant. Finance* 11(2) — abstract via RePEc
    and OpenAlex. Tables as already reproduced in
    [technical-indicators-nse.md](technical-indicators-nse.md).
18. Souza, Ramos, Pena, Sobreiro & Kimura (2018), *Financial Innovation*
    4(1) — extensive body text via exa. 4,428 strategies.
19. Butt, Kolari & Sadaqat (2021), *PBFJ* 65:101486 — abstract via RePEc
    and SSRN. **Emering-market momentum is *lower* than developed.**
20. Moreira & Muir (2017), *JF* 72(4) — full text via NBER/Yale/NYU PDFs.
21. Cederburg, O'Doherty, Wang & Yan (2020), *JFE* 138(1) — full text via
    Lehigh PDF.
22. Hurst, Ooi & Pedersen (2017), "Trends Everywhere" — full text via SSRN
    mirror.
23. Sehgal, Pandey & Sen (2024/25), *IJEM* 20(10):4288–4307 — abstract
    via Crossref/Semantic Scholar/Booksci.

**Abstract-only, or metadata only — flag where it matters:**

24. Raju & Chandrasekaran (2019), SSRN 3510433 — abstract only (SSRN 403).
25. Raju (2023), SSRN 4607933 (weighting schemes) — abstract + substantial
    body text via Scribd.
26. Raju (2023), "Shades of Momentum", SSRN 4977717 — abstract only.
27. Raju (2023), "Timing the Tide", SSRN 4687044 — abstract only.
28. Raju (2024), SSRN 5026391, "Long-Term Asset Returns in India" —
    metadata only, not used for any claim.
29. Srivastava, Chakravorty & Singhal (2019), SSRN 3345280 — abstract +
    summary bullets via Academia.edu.
30. Mohapatra & Misra (2020), *IIMB Management Review* 32(1):75–84 —
    extensive body text via publisher + exa. Open access.
31. Barik & Balakrishnan (2022), *IIMB Management Review* 34(1) — abstract
    via exa.
32. Dhankar & Maheshwari (2014), *IJFM* 4(2) — extensive body text via
    Academia.edu. 7.7% at 6×9.
33. Maheshwari & Dhankar (2015), *Seasonality in Momentum Profits* — abstract
    via publishingindia.com and academia.edu.
34. Misra & Mohapatra (2015), *Margin* 9(2) — abstract via RePEc.
35. Ansari & Khan (2012), *Managerial Finance* 38(2) — abstract via citation
    context only.
36. Sehgal & Balakrishnan (2002), *Vikalpa* 27(1) — abstract via SAGE and
    citation context. The original Indian momentum paper.
37. Sehgal & Jain (2015), *IJEM* 10(4):801–819 — abstract via Emerald.
38. Sehgal & Jain (2011), *JAMR* 8(1) — abstract via Emerald.
39. Sehgal & Dhankar (2017), *Global Business Review* 18(4) — abstract.
40. Balakrishnan & Barik (2021), *Futures Business & Economics* 7(1) —
    full PDF via Springer Open. "6-6 strongest."
41. Saeed, Chowdhury & Michello (2014), Indian momentum/contrarian — abstract
    via exa and citation context.
42. Maheshwari & Dhankar (2015), *J. Business Inquiry* — full PDF via
    journals.uvu.edu. Long-run reversal.
43. Saraf & Kayal (2022), MSE Working Paper 215 / *IIMB Management Review*
    35(2) 2023 — full PDF via mse.ac.in. Volatility anomaly, not scaling.
44. Kayal (2023), MSE Working Paper 242 — full PDF. Volatility-managed
    portfolios **on US factors**, hosted by an Indian institution. **This is
    not Indian evidence** despite the host. Flagged.
45. Peswani & Joshipura (2019), *IMFI* 16(3):62–75 — publisher PDF body.
46. Tadas, Nagarkar, Malik, Mishra & Paul (2023), *IMFI* 20(2):26–40 —
    publisher PDF body. 8 months of hourly data on 14 Nifty-50 names.
47. Kantaria & Tanna (2022), *International Management Review* 18 —
    PDF body via americanscholarspress.us. 1,296 MA combos, 13 names.
    "a common approach does not outperform a buy-and-hold."
48. BacktestIndia (Desai) momentum backtest, Dec 2025/Mar 2026 — full page
    read. Non-peer-reviewed; author-declared not-SEBI-registered.
49. Kiran P K (Aug 2026), dual-momentum blog — full page read. Personal
    project, PIT-corrected. Its survivorship numbers are the useful part.
50. Capitalmind / Anoop Vijaykumar (Dec 2019), "Does Momentum Investing work
    in India?" — full page read. PIT and delisted-aware; no costs.
51. *Mint*, 7 Oct 2025, on momentum funds' negative trailing year — via
    search extract. Newspaper, not peer-reviewed, but realised data.
52. "Behavioural Biases and Market Efficiency: Testing Overreaction and
    Reversal Effects in NSE 500 Stocks", *IJES* 11(19s) 2025 — full PDF
    via theaspd.com. **Very low-tier journal.** Used only for the
    reversal-inversion finding, labelled as such.

### Could NOT verify

- **SSRN full texts: HTTP 403 on every attempt** (papers.ssrn.com
  abstract pages and Delivery.cfm PDFs, both browser and text proxy).
  Every SSRN item above is cited from its abstract or from a mirror. The
  affected items are #24–29. If any of those claims matter, they need
  the PDF.
- **Springer, SAGE, ScienceDirect, Emerald, Taylor & Francis full texts:**
  bot challenges or 403 on all direct fetches. Abstract-level only
  throughout for those publishers.
- **Pillai (2026) full PDF:** ResearchGate 403. Reconstruction is from
  the search index's body-text extraction. Multiple independent extracts
  agreed on all figures quoted, which is why I trust them, but I have not
  read the PDF.
- **Lakshan (2026) full PDF:** Zenodo binary and publisher PDF both
  refused. Only the record-page abstract. Effect sizes are from the
  abstract, not from the tables.
- **Rink (2023) Table 5, panel B, India's specific rule count.** I
  confirmed India is in the sample and read the headline EM findings, but
  **I did not isolate how many of the 6,406 rules beat buy-and-hold on the
  BSE Sensex specifically.** That number exists in the paper and I could not
  reach it. Do not assume it.
- **Joshipura & Joshipura (2020) turnover table.** Referenced as Table 6
  with one-way annual churn, but the churn *values* were not in any
  extractable text. **The churn figure for the low-risk + high-momentum
  strategy is unknown.** This matters — it is the input to the cost
  arithmetic in Question 5.
- **Srivastava, Chakravorty & Singhal (2019) sample period.** The
  "21.9% average annual return" is attributed to "the study period"
  without dates in any extract. **Unverified.** Do not quote that number
  with a date range.
- **The 2025 NSE-500 reversal study's authorship.** *IJES*, ISSN 2229-7359,
  Vol 11 No 19s 2025. No named authors recoverable. I would not cite it in
  anything load-bearing.
- **"Tests of Technical Analysis in India" sample period.** Sehgal &
   Gupta (2007), *Vision* 11(3). Paywalled; abstract only. Noted because
  it is one of only two Indian studies that test MA rules head-to-head
  against buy-and-hold and it finds technical indicators **do not** beat
  buy-and-hold net, and calls EMH confirmed. Strong qualitative prior;
  no sample period, so unusable for a table row.
- **Alhashel & Almudhaf (2021), *EMFT* 57(13)** — Gulf region, not India.
  Excluded from all tables. Retrieved only to confirm it is not Indian.
- **Kedia & Satpathy's WML Sharpe ratios of 6.21–8.78.** These are as
  published, but they are implausible as cost-aware numbers from a
  pedagogy journal. Recorded as stated; **treat as unverified-strength.**
- **No Indian study found that specifically tests a 12-1 or 6-12 month
  formation on Nifty-50 constituents specifically with realistic costs.**
  That gap is real and is the reason Question 5 cannot be answered
  cleanly.

## What would settle the open questions

1. A peer-reviewed replication of **Chui et al. (2023)'s** liquidity
   decomposition on NSE data. Resolves the central disagreement in this
   file.
2. **Pillai's 13 rules re-run at 22 bps delivery round trip plus 5 bps/side
   slippage** instead of 0.05%. His adjusted p-values would not all
   survive, and by how much is the answer.
3. **Any Indian momentum paper that models STCG/LTCG.** None exists. Until
   one does, "momentum works in India" is a pre-tax claim.
4. A peer-reviewed Indian **50/200** test on point-in-time Nifty-50
   constituents. Currently the rule has two contradictory non-peer-reviewed
   sources and nothing else.