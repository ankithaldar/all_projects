# Fundamentals for NSE Equities

The instinct "don't ship a fundamental backtest on free data" is correct
— and stronger than framed. The problem is not only that free Indian
fundamental data is point-in-time deficient. **It is**, but a free XBRL
pipeline exists that most Indian quant work has not found. The catch is
history depth: **~7 quarters of clean structured XBRL, not 26 years.**

## Data availability

### Screener.in — do not build on it

ToS (Mittal Analytics, Aug 2018): *"personal, non-commercial transitory
viewing only... under this license you may not: **modify or copy the
materials**"*.

- Building a persistent database from it **is** the prohibited act.
- robots.txt disallows the machine-readable paths: `/*?q=`, `/*?sort=`,
  `/*?limit=`, `/*?page=`, `/company/source/quarter/*`.
- ~50 anonymous page views/day. A 50-stock × 15-year backfill is impossible.
- No official API; premium CSV export is Rs 4,999/year.
- Values are **current** (last restatement) — unusable for backtesting
  regardless of licence.

### NSE official APIs — unevenly documented, worth knowing

| Endpoint | Status | Content |
|---|---|---|
| `/api/corporate-announcements?index=equities&symbol=X` | 200 | **Full filing history to listing.** RELIANCE: 3,355 records from 13-Dec-2004 |
| `/api/integrated-filing-results?...&period_ended=all` | 200 | **PIT anchor + XBRL URL** |
| `/api/corporate-share-holdings-master` | 200 | Quarterly shareholding, **22 quarters to Sep-2021**, with `broadcastDate`, `revisedStatus`, `revisionRemark` |
| `archives.nseindia.com/.../sec_bhavdata_full_*.csv` | 200 | **Daily bhavcopy incl. delisted.** ~4,300 files since 2009 |
| `/api/quote-equity` | **403** | Akamai blocks non-browser sessions |
| `api.bseindia.com` announcements | **403** | Akamai; needs cookie tricks |

**Rate limits:** none published. NSE sits behind Akamai; `nsepython`,
`jugaad-data` and `niftyterminal` all say the same: no official API, no
key, reuse browser cookies, sessions break intermittently. **Assume it
breaks and build recovery logic.**

**Legal:** personal research fine. **Never redistribute, never ship NSE
data as a product.**

### Vendors

| Source | Cost | PIT? |
|---|---|---|
| **CMIE Prowess** | Rs 250,000/yr (ProwessDX ~USD 4,500) | **Yes** — filing dates tracked. The standard Indian academic source |
| **Capitaline** | Rs 250,000/yr per user | Yes |
| **EODHD** | Fundamentals $59.99/mo | **Unclear** — vendor feeds typically store latest-restated value per period unless they expose a publication timestamp. Verify before trusting |

### Free APIs

| Source | India | Verdict |
|---|---|---|
| **Yahoo `query1.finance.yahoo.com/v8/finance/chart/SYMBOL.NS`** | Excellent, verified live, no crumb needed | Best free price source, but **delisted/renamed names are erased** (~24% of a top-500 2015 vintage) and not licensed for redistribution |
| Yahoo `quoteSummary` | **401** Invalid Crumb | Fundamentals blocked |
| Alpha Vantage | Poor India coverage | Unusable for a 50-stock panel |
| Twelve Data | **India absent** from EOD exchange list | Unusable |
| **NSE bhavcopy archive** | Complete, incl. delisted | **Best survivorship-free price source** |

## The free PIT pipeline nobody uses

```
GET /api/integrated-filing-results?index=equities&symbol=RELIANCE
    &issuer=Reliance%20Industries%20Limited&period_ended=all
    &type=Integrated Filing-%20Financials
```

Returns per filing: **`broadcast_Date`** (exact public-availability
timestamp — this is your `available_from`), `qe_Date`, `xbrl` URL,
`type_Sub` (Original/Revised), `revised_Date`, `audited`, `consolidated`.

The XBRL itself (verified, 46 KB, 142 facts) is Ind AS and carries **both
the financials and the PIT metadata in one document**: board-approval
date, results type, audited flag, plus the full statements.

**This is a complete, correctly-timestamped, restatement-aware
fundamental store. Free. No vendor.**

Libraries: `nse-xbrl` handles the four-context structure and ~100 line
items; `india-xbrl-filings` does resumable bulk download with checksums.

### The binding constraint: history depth

Measured:

- `integrated-filing-results`: **totalCount = 12** for RELIANCE, HDFCBANK
  *and* TCS. Oldest `qe_Date` = 31-MAR-2025.
- `hasXbrl` flags only 2025–2026.
- BSE legacy XBRL paths: **404, dead.**
- `corporate-announcements` carries quarterly "Financial Result Updates"
  back to ~2010 **with PDFs only** — ~14 years of PDF parsing is a real
  project with material extraction risk.

**Honest read: ~7–8 quarters of clean structured PIT fundamentals, or ~14
years of PDF scraping with error risk. You cannot construct a 2005–2025
PIT fundamental panel from free sources.**

## Survivorship bias is the same magnitude as the effects

Measured on the very Nifty-500 universe these papers use:

- Value **4.4%/yr VW**, Size **3.53%**, Momentum **3.97%**
- A 2026 study reconstructing Nifty Smallcap 250 (1,437 stocks, 2016–2025)
  from bhavcopy: survivor-only backtests inflate annual returns by
  **+4.94pp (+23.3% relative)** and Sharpe by +0.097.

Prior Indian Fama-French studies (Bahl 2006; Sehgal & Jhanwar 2007;
Tripathi 2008; Aziz & Ansari 2014; Upadhyay 2017) *"largely ignored
survivorship bias and may be misleading."*

> **Caveat:** these are working papers and self-published audits, not
> peer-reviewed. The **direction** is robust; the **magnitude** is ±.

> **Treat with suspicion:** several prominent "Indian look-ahead bias"
> blog posts claim specific inflation figures ("1.5–3.5% CAGR for
> quarterly-rebased strategies"). The mechanism is real; the numbers are
> vendor marketing with no methodology. Do not cite them.

## Factor evidence in India

### Momentum — strongest, most robust

| Study | Finding |
|---|---|
| Kedia et al. (*Kuey*, 2024) | 232 Nifty-500 firms: WML Sharpe 6.21 (3×3) to 8.78 (6×12); **alpha → ~0 under Carhart 4-factor.** A priced risk factor |
| Ansari & Khan (2012), *Managerial Finance* | 470 BSE stocks: profitable **after** size/value/illiquidity controls; driven by **winners**, not losers |
| Time-series vs cross-sectional (2021) | Both significant, **TS > CS**; TS does not decay at long horizons |
| *How smart is a momentum strategy?* (2023) | **6-month lagged return + quarterly rebalance** = best risk-adjusted |

**Caution:** Sharpe ratios of 6–9 on WML long-short are **implausible** —
they are short legs in a market where shorting was banned 2001–2007 and
remains heavily restricted. All such results are largely non-implementable.
**Long-only momentum is the real trade.**

### Value — historically strong, **decayed to zero post-GFC**

The key negative result: *"Changing Nature of the Value Premium"*
(2018): **P/B long-short spread = 2.36% pre-GFC → 0.12% post-GFC.**
Chow test confirms a structural break. In value-weighted samples
**P10 beat P1** — the anomaly *reversed*.

Industry-level value (2022): significant in 15/17 industries over
1999–2020 but **declined 51–76%** pre→post-2008-09.

*Are anomalies fading?* (*IBR*, 2019): **value and momentum are fully
explained by FF risk models**, contrary to prior Indian evidence.

A 2017 study found HML 9.08% annualised but with **7 drawdowns >20%, worst
−53% lasting 7.1 years.**

**If you build a value signal for a live system, you are betting on
mean reversion of a factor whose premium has been ~zero for a decade and
whose historical premium came with multi-year −50% drawdowns.**

### Quality / profitability — strongest fundamental evidence in India

**Jacob, Pradeep & Varma** (IIM-A WP2022-11-01 / SSRN 4284686): **QMJ
four-factor alpha 0.92%/month (~11–13%/yr annualized) over 26 years.**
Long-only QMJ alpha **0.69%/month** — roughly 2× the US estimate, and
significant against Harvey-Liu-Zhu thresholds. Drivers: **profitability
and payout** (tunnelling hypothesis). **Low churn**, lower risk, shorter
drawdowns, works long-only large-cap.

Capitalmind (2024): Rs 100 → Rs 1,933 over 17 years vs Rs 810 Nifty TRI.
**GPTA (profits/assets) is monotonic; margins are NOT — margins fail the
key validation criterion.** Median 5-yr 17.5% vs 13.1%.

Gross profitability + momentum (2025): combined FF alpha **1.08%/month
(t=3.90)**, 1.78× standalone profitability.

### Accruals — **the sign is wrong in India. Do not use the US version.**

**Sehgal et al.** (*IMFI*, 2012): *"**Accruals are found to be positively
associated with future returns.**"* Indian investors **underprice
accruals, overprice cash flows** — the accrual anomaly **reverses**. The
cash-flow anomaly also reverses.

Pincus, Rajgopal & Venkatachalam (TAR, 20 countries): India shows
β_ACC **0.66** — high overweighting, consistent with Sehgal.

Sharma (2025, ProwessIQ 2004–2023): Accruals (Sloan) mean annual return
0.13%, Sharpe 0.07; Net Operating Assets **−4.15%, Sharpe −1.01**.

**Verdict: skip accruals entirely.** No evidence it pays; best Indian
evidence says the effect vanishes under a multi-factor model.

### Low volatility — present, large, and needs only price data

**Agarwalla, Jacob & Varma** (IIMA WP2014-07-01): BAB earns significant
positive returns in India, **dominating size, value, momentum**.

*Low-Risk Anomaly: Evidence from India* (SSRN 4398656): 4,400 companies,
19 years. **Strong convex return-volatility relationship. Does not diminish
over time. Survives factor controls.**

Joshipura & Peswani (*IMFI* 2019): NSE 1997–2018, **low-vol quintile
annual alpha spread 25.53% vs high-vol**; highest risk-adjusted returns in
**LARGE caps** (LV large-cap Sharpe 0.319).

LV effect stronger than beta effect (SCIRP): LV excess return **10.62%**
(std-dev sort) vs 4.82% (beta sort). **LV has a GROWTH tilt in India —
opposite to the US value tilt.**

Peswani & Joshipura (2021): leverage constraints predominantly explain the
Indian low-risk effect — **risk should be measured by beta.**

**This is the best-evidenced anomaly in Indian equities after momentum, and
it requires only price data.**

### Size — contaminated, and irrelevant for Nifty-50

The size premium (SMB 3.4%/month, positive) and the survivorship artifact
(3.53%/yr) are **the same size**. Nifty-50 is large-cap only, so size is
near-constant. **Drop it.**

## Reporting lag — the actual regulation

**Regulation 33(3) of SEBI (LODR) Regulations, 2015.** From SEBI's Master
Circular for LODR compliance, para 9:

> *"the quarterly (audited / unaudited) and the annual (audited) financial
> results have to be submitted **within a period of 45 days and 60 days,
> respectively**, from the end of the quarter / financial year."*

NSE compliance calendar confirms: Jun-30 → 14-Aug; Sep-30 → 14-Nov;
Dec-31 → 14-Feb; Mar-31 → 30-May. Indian fiscal year ends **31 March**.

**Lateness is itself disclosable** — a late filer must state reasons to
the exchanges within one working day of the due date. So lateness is an
observable public event, which is exactly why using actual `broadcast_Date`
is correct and `period_end + 45` as a proxy is not.

| Use | Minimum lag | Conservative |
|---|---|---|
| Quarterly fundamentals | 45 days | 60 days |
| Annual fundamentals | 60 days | 90 days |
| **With actual `broadcast_Date`** | **0 days — use the timestamp** | — |

Note also: **interim results are limited-reviewed, not audited**, and
Indian firms show **measurably higher earnings management in interim
periods (29.6%) than annual (17.2%)**. Interim numbers deserve less
weight — a reason to prefer TTM-combined signals over single-quarter.

## Costs and turnover — a point in favour of fundamentals

Round trip ~0.22–0.27% delivery. At one-way 11–13 bps:

| Annual turnover (one-way) | Drag |
|---|---|
| 50% | 0.12% |
| 100% | 0.25% |
| 200% | 0.50% |
| 400% | 1.00% |
| 800% | 2.00% |

For a **quarterly-rebalanced** Nifty-50 fundamental strategy with a
12-month horizon, expected turnover is **25–40%/yr → 0.06–0.10% annual
drag. Utterly negligible.**

**This is the point the US "factors die after costs" literature gets
wrong for our case** — that critique is calibrated on high-turnover US
factor sorts. Costs are decisive only for daily/intraday or monthly
signals.

SEBI's FY26 study: transaction costs were **35% of gross losses for
losers vs 21% of gross profits for winners.**

**The real risks at low turnover are different:** concentration (a 50-stock
screen can end up holding 10 names), exit liquidity in a small-sample
tail, and parameter-choice data mining. With 50 names, **effective sample
size is the binding constraint** — cross-sectional dispersion is estimated
from ~50 observations, so any z-scoring or rank-transform choice is a
large lever and you will overfit it.

## Recommendation

**Do NOT build a backtestable fundamental signal** on Screener or any free
multi-year Indian panel. Three independently disqualifying reasons:

1. **Licence.** Screener's ToS prohibits copying into a persistent store.
   No free, sanctioned, storable Indian dataset with 10+ years of history.
2. **Restatement look-ahead is unfixable.** Current-value rows can never
   be un-leaked. Any fundamental backtest on them is retroactively
   contaminated and you cannot tell by looking at the results.
3. **Survivorship bias equals the effect size.**

**DO build a PIT-clean fundamental layer for live use**, because
`broadcast_Date` + XBRL + restatement flags are free:

- Backfill from **Dec 2024** onward (~Nifty-200 names).
- Every row carries `available_from = broadcast_Date`. **Join on that,
  never on `qe_Date`.**
- Keep `type_Sub`/`revised_Date` — **store restatements as versions, do
  not overwrite.**
- **Store all financial filings ever, from day one.** That is what makes a
  2030 backtest possible and it costs nothing today. Highest-value
  engineering decision available.
- Use `corporate-share-holdings-master` — promoter holding drift is a
  genuine PIT signal you get free.

### Signals worth building

| Rank | Signal | Evidence | Data | Recommendation |
|---|---|---|---|---|
| 1 | **Momentum (12-1 or 6m, quarterly)** | Very strong, multiple Indian replications, long-only | **Price only** | **Build first.** Best Sharpe, lowest turnover, zero look-ahead risk |
| 2 | **Low volatility** | Very strong; 4,400 stocks/19yr; LV excess 10.6% vs HV; survives controls | **Price only** | **Build.** Use std-dev not beta. Note growth tilt |
| 3 | **Quality / GPTA** | Strongest fundamental evidence; QMJ alpha 0.92%/mo over 26yr, low churn | XBRL (recent only) | **Build for live.** Use **profits-to-assets, not margins** — margins fail monotonicity |
| 4 | Sector momentum | Plausible; leadership demonstrably rotates | Price only | Optional. Evidence in India is thin — mostly descriptive |
| 5 | Value | Decayed to ~0, occasionally reversed | XBRL recent | **Do not build standalone.** Mild tilt or spread monitor at most |
| 6 | Size | Effect ≈ survivorship artifact | — | **Drop** |
| 7 | Accruals | **Sign reversed vs US**; zero-to-negative | — | **Do not build** |
| — | NTM earnings yield | Analyst overestimation bias 2.5%/yr, **21% for high-uncertainty names**; free India consensus **not available** | Paid feed | **Cannot build** |

**Convergence:** momentum, low-vol and quality are your three. Two need
only price. The third has the best fundamental evidence in India and the
lowest turnover. **Note that in India low-vol and quality are partially
the same signal** (low-vol stocks have higher profitability), so you are
building two bets, not three. Be deliberate about that.

## What actually decides this project

**With 50 names, effective sample size is the binding constraint, not data
quality.** Six of the Indian studies report Sharpe 6–9 on long-short
momentum — artefacts of restricted shorting. A real quality signal at
0.69%/month long-only is a **modest, plausible** number.

**Optimise for the plausible number.** Keep the parameter count low, hold
out a genuine forward period, apply multiple-testing corrections.
**A 50-stock cross-section will manufacture a 15% Sharpe out of noise if
you let it.**