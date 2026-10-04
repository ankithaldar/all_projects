# SEBI Algorithmic Trading Compliance

Verified against SEBI's own circular index and the Master Circulars,
October 2026.

## The structural fact that reframes everything

**SEBI imposes almost no direct obligations on the algo trader.
Obligations fall on the stock broker / trading member.**

- 2012 circular para 5: *"Stock exchange shall ensure that the stock
  broker shall provide the facility of algorithmic trading only upon the
  prior permission of the stock exchange."*
- Feb 2025 circular para 5(II)(a): *"The facility of algo trading shall be
  provided by the broker only after obtaining requisite permission of the
  stock exchange for **each algo**."*

There is **no lighter track for proprietary algos**. A broker's own
in-house algo goes through the same exchange approval as a client algo.

## Instrument stack

| Date | Number | Subject |
|---|---|---|
| 30 Mar 2012 | **CIR/MRD/DP/09/2012** | Broad Guidelines on Algorithmic Trading — **the foundation** |
| 21 May 2013 | CIR/MRD/DP/16/2013 | System audit every **6 months**; doubles OTR charges |
| 9 Apr 2018 | SEBI/HO/MRD/DP/CIR/P/2018/62 | **Unique identifier per algo**; co-location; OTR ±0.75% LTP |
| 24 Jun 2020 | .../2020/107 | OTR slabs to 2000; 15-min cooling-off on 3rd instance in 30 days |
| 24 Nov 2020 | .../2020/234 | Mock sessions; monthly participation |
| 2 Sep 2022 | .../2022/117 | **No past or expected return/performance claims** for algo services |
| 4 Feb 2025 | .../2025/0000013 | Safer retail participation: white/black box, static IP, OAuth+2FA, no open APIs |
| 30 Sep 2025 | .../2025/132 | Glide path — mandatory for all brokers from 1 Apr 2026 |
| 4 Feb 2026 | HO/47/11/16(2)2025.../2026 | OTR revision; equity options ±40%/Rs 20 exemption; eff. 6 Apr 2026 |

Codified in the **Master Circular for Stock Exchanges and Clearing
Corporations, SEBI/HO/MRD-PoD2/CIR/P/2024/00181** (30 Dec 2024), Ch. 2
para 7, and the **Master Circular for Stock Brokers,
SEBI/HO/MIRSD/MIRSD-PoD/P/CIR/2025/90** (Jun 2025), Section 59.

> **Source-quality warning.** The CSE and BSE copies of the Feb 2025
> circular render the effective date as "August 01, 2026". SEBI's own PDF
> says "August 01, 2025". Exchange-hosted reproductions are not reliable.

## Hard requirements worth encoding as assertions

| # | Assertion | Source | Type |
|---|---|---|---|
| A1 | Any order from automated execution logic **is** algo trading — no de-minimis, no human-in-the-loop exemption | 2012 para 3 | Hard legal |
| A2 | Trading own account is still an algo; needs exchange permission via the broker | 2012 para 5, 8 | Hard legal |
| A3 | No algo in production without a **unique exchange-allotted Algo ID** | 2018 paras 15–16 | Hard legal |
| A4 | Any **modification** to an approved algo requires **fresh approval before go-live** | 2012 para 8(v); NSE INVG/69255 §9 | Hard legal |
| A5 | Logic changes = **a new algo**, not a version bump | NSE INVG/69255 §9.1 | Exchange rule |
| A6 | System audit **every 6 months**; serious deficiencies ⇒ trading software **suspended** | CIR/MRD/DP/16/2013 | Hard legal |
| A8 | Black-box provider must be **SEBI-registered Research Analyst** with a per-algo research report | Feb 2025 para 5(V) | Hard legal |
| A9 | Black-box confined to **family**: self, spouse, dependent children, dependent parents | Feb 2025 para 5(I)(c) | Hard legal |
| A10 | **No reference to past or expected return/performance** of an algo on any platform | 2022/117 para 4.1 | Hard legal |

## Pre-trade risk controls (2012 para 6) — the minimum set

```
HARD  price within exchange price band
HARD  qty <= exchange max order qty for security
HARD  value <= exchange 'value per order'
HARD  cumulative open order value per client <= broker limit
HARD  net position <= exchange position limit
HARD  automated-execution check: account for ALL executed, unexecuted
      and unconfirmed orders before releasing the next one
```

Source: 2012 circular para 6(i)–(v). This is the requirement that makes
co-locating the risk and execution modules in **one process** a safety
property rather than a style choice — see
[project-structure.md](../architecture/project-structure.md).

Other hard constraints:

- **No algo market orders in equity** (NSE/MSD/67753 §8.1.1.12);
  **no IOC or market orders in commodity** (NCDEX).
- **No cross trades**, including crossing a client's own orders via API
  (NSE CMTR/21793; INVG/69255 §10.1).
- **Servers located in India**, no interlink to systems or IDs outside
  India (2012 para 4(iii)). **This rules out US-region LLM APIs.**
- **Kill switch per Algo ID**, and it must **automatically** trigger a
  halt on pre-defined conditions — not a UI button.
- **Static IP whitelisting**: max one primary + one secondary, changeable
  at most once per calendar week. **OAuth only, 2FA mandatory.**
- **OTR** (order-to-trade ratio): ±0.75% of LTP exemption band; slabs to
  2000; **3rd instance of OTR ≥2000 in rolling 30 days ⇒ no orders for
  the first 15 minutes next day**; repeated high OTR >10 times in 30 days
  ⇒ proprietary trading suspended for the first trading hour.
- Base minimum capital with algo: **Rs 50 lakh**.

## Retention — the definitive answer

| Figure | What it governs | Status |
|---|---|---|
| **8 years** | Books of account and records, SEBI (Stock Brokers) Regulations **2026** Reg. 16, effective 7 Jan 2026 | **Current law** |
| 5 years | Reg. 18 of the 1992 Regulations | **Repealed 7 Jan 2026** |
| **5 years** | Exchange audit trail for TM API orders (BSE Annexure 2 §10.3) | Exchange floor |
| 3 years | **NYSE Rule 105** | Wrong jurisdiction |

**SEBI's algo circulars specify no retention period at all.** 2012 para
8(iv) says only *"maintain logs of all trading activities to facilitate
audit trail."* The 8-year figure is the general broker-records obligation
that necessarily captures algo logs.

```python
# Pick per record class; do not hardcode one number.
RETENTION_YEARS = {
  'broker_books_of_account': 8,   # SEBI (Stock Brokers) Regs 2026, Reg 16
  'algo_audit_trail': 8,          # satisfies both Reg 16 and the 5yr floor
  'exchange_minimum': 5,          # BSE/NSE operational standards floor
}
```

Preserve originals **indefinitely** once taken by an enforcement agency
(SEBI circular, 4 Aug 2005).

## Algo ID structure — exchange-implemented, not SEBI-specified

SEBI says only "a unique identifier provided by the Stock Exchange".

**NSE** (non-NEAT NNF field, 15 digits): digits 1–12 sentinel or
PIN+Branch+UserID; **13th digit is the algo flag** (0 = algo, 1 =
non-algo, 2 = algo via SOR, 3 = non-algo via SOR, 7 = basket, 8 = batch);
separate **ALGO ID** field per registered strategy. Unregistered client
algo (≤10 OPS): Algo ID = `"99999"`, 13th digit = `"0"`.

Pre-trade validation (NSE/FAOP/69296): 13th digit in {0,2,4} with Algo ID
= 0 ⇒ **reject**; 13th digit in {1,3,5,6,7,8} with Algo ID ≠ 0 ⇒
**reject**.

**Encode `algo_id` as an exchange-namespaced opaque string, not an
integer.** Validate membership per venue. A single algo needs separate
registration per exchange.

## Market abuse and MNPI

Governing law: SEBI (Prohibition of Insider Trading) Regulations **2015**
(last amended 12 Mar 2025) + PFUT Regulations 2003 + SEBI Act 1992.

**The useful definition** — Regulation 2(1)(m), "generally available
information": *"information that is accessible to the public on a
non-discriminatory basis and shall include research and analysis based
thereon."* The Note in the explanatory memorandum:

> *"organisations that provide research reports may add value ... so long
> as the research is based on generally available information, the findings
> of such research would not become UPSI."*

| # | Assertion | Source |
|---|---|---|
| M1 | **No signal from generally-available information is UPSI** | Reg 2(1)(m) + Note |
| M2 | Any ingestion path reaching non-public info **instantly creates insider status**, no intent needed | Reg 2(1)(l) |
| M3 | Reg 4(1): possession of UPSI ⇒ trades **presumed motivated** by it. Reasons for trading are not relevant | Reg 4(1) Explanation |
| M4 | If a vendor item carries an embargo flag, block the symbol until dissemination lag is satisfied | Reg 3(3) two-trading-day analogue |
| M8 | **Chinese wall** is an affirmative *defence*, not a permission — you must prove it worked | Reg 3B; Schedule B cl. 4 |
| M9 | **Trading plans** (Reg 5) are the only general safe harbour: pre-approved, publicly disclosed, **irrevocable, no deviation** | Reg 5 |
| M10 | Contra-trades within **6 months** of a pre-cleared trade ⇒ **profits liable to disgorgement** | Schedule B cl. 10 |
| M13 | Trades above **Rs 10 lakh** per quarter disclosed within **2 trading days** | Reg 7 |
| M14 | Spoofing/layering prosecuted under PFUTP Regs 3 & 4. SEBI's 28 Apr 2025 ex-parte order impounded **Rs 3.22 crore** across 173 scrips, 621 spoofing instances | PFUTP; SEBI order |

**Wall-crossing:** India has **no codified market-sounding regime**.
Reg 3(3) is a narrow legitimate-purpose carve-out requiring a company board
to opine and an NDA, and requiring dissemination at least two trading days
before the transaction — a **company-side** mechanism, not self-serve.

**Because there is no self-serve safe harbour, the compliant design is the
conservative one:** trade only on information that was already generally
available when your licensed vendor delivered it, and hard-block any item
flagged non-general.

## What I could NOT verify — do not encode as fact

1. Any April 2021 SEBI algo circular. **Does not exist.**
2. Any 2010 SEBI algo guidance. **Does not exist.**
3. A SEBI-stated retention period for algo records. SEBI is silent; 8
   years comes from the broker-records regulation.
4. "Static vs dynamic algo" classification. Not in any instrument found.
5. A SEBI algo-wise-turnover monthly return by algo traders. **No such
   filing exists.** Monthly algo turnover is an exchange-to-SEBI
   obligation via the Monthly Development Report.
6. The numeric OPS threshold *in the SEBI circular*. Feb 2025 leaves it to
   the Broker's Industry Standards Forum footnote. **10 OPS** is
   exchange-set — encode as venue config, not a SEBI constant.
7. NSE 15-digit NNF digits 4, 5, 6, 9. Verified 0, 1, 2, 3, 7, 8 from
   exchange documents; the rest are partially documented. Do not hardcode
   a full map without the current NNF protocol CD.
8. Whether the 10 OPS threshold also governs whether third-party-vendor
   API trading is classified as algo. Genuinely ambiguous.

## Sources

**Tier 1, primary:** `sebi.gov.in` algo index
(`?doListingAll=yes&search=Algorithmic`), the 2012, 2013, 2018, 2020,
2022 and 2025 circulars listed above, the 2026 Stock Brokers Regulations,
and the PIT Regulations with explanatory Notes.

**Tier 2, exchange:** NSE INVG/67858, INVG/69255, INVG/70309, FAOP/69296,
SURV/45016, SURV/48818, CMTR/21793, NNF API protocol CD; BSE Notice
20250723-41 Annexure 2.

**Tier 4, treat with suspicion:** broker explainers and vendor blogs. Useful
for orientation, unreliable for citation, and several contain concrete
errors including the CSE/BSE "August 2026" typo and the invented
static/dynamic dichotomy.