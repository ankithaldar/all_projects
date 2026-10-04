# Scraping, Licensing and Data Compliance (India)

Verified against primary sources: NSE Terms of Use, IT Act 2000
(consolidated), the Copyright Act, the DPDP Act 2023, Delhi HC judgments,
and NSE's current authorised-vendor list.

## The myth register — delete these from code comments

| Claim as stated | Verdict | Correct framing |
|---|---|---|
| "Violation = IP block + legal notice under IT Act 2000" | **Embellishment** | Contractual. NSE ToU cl.9 expressly bans automated collection; cl.6/21 reserve blocking and legal action. IT Act s.43 gives **civil compensation only**. s.66 criminal only if "dishonest or fraudulent" in the IPC s.24/25 sense |
| "InvestingNote" is an authorised NSE vendor | **Fabricated** | Use NSE's current 26-vendor real-time PDF. Note entitlements are **per segment**, not boolean |
| "check robots.txt" as a legal requirement | **Convention** | Voluntary standard, **zero Indian case law**. Matters only where a ToU incorporates it, and a ToU *is* enforceable |
| "max 1 req/sec" as a compliance requirement | **Engineering** | No Indian legal source found for any number. Justified by IT Act s.43(e)/(f) disruption risk |
| "store headline+summary per s.52 fair dealing for research" | **Misreading** | s.52 is exhaustive (*Super Cassettes*, 2011); "private use" historically excluded commercial use. Correct basis is facts-not-expression plus no republication |
| "no personal data under DPDP 2023" | Right, **wrong reason** | DPDP has no sensitive-data tier. PAN/Aadhaar are restricted by Income-tax Act s.360 + PDPR 2010, and Aadhaar Act s.4(4)/s.29. DPDP is simply **out of scope** if no identifiable individuals are processed |
| "DPDP doesn't apply to publicly available data" | **Myth** | That carve-out was in the 2019 **Bill**. The enacted 2023 Act's s.17 has no such exemption |
| "Indian Express v. Zeal" | **Does not exist** | Real case is *OLX v. Padawan*, Delhi HC 2016 |
| "no insider tips" as a scraping control | **Category error** | Insider trading is a conduct-of-trading regime, not a data-acquisition one |

## NSE's actual prohibition is contractual, not statutory

**NSE Terms of Use:**

> **Clause 9:** *"User is prohibited to conduct any systematic or
> automated data collection activities (including scraping, data mining,
> data extraction and data harvesting) on or in relation to our Website /
> Mobile Application."*

> **Clause 8:** content shall not be "... **stored (either in hardcopy or
> in an electronic retrieval system)** ... without prior written
> permission of NSE."

Clause 8 also carves out: *"Unless the information or Content is available
for download, not to aggregate, copy or duplicate in any manner."*

NSE Data & Analytics Ltd has an identical clause at cl.18 of its Data
Portal ToU. NSE Clearing Ltd likewise. This is house style, not a one-off.

The separate **Data Usage and Data Sharing Policy V2.0** (eff. 01.08.2019)
requires restrictions be set out in the Relevant Agreement (6.1), forbids
reverse engineering (6.3), confers no ownership (6.4), forbids
redistribution (6.5), and reserves **audit and spot-check rights** (10.3,
10.4).

**No reported Indian case exists in which NSE sued a data scraper.** NSE's
documented enforcement is cease-and-desist against apps mimicking
real-time trading (YourStory, Oct 2022) and trademark/bankruptcy
proceedings against impersonators.

## The IT Act exposure, read properly

- **s.43** chapeau: *"If any person **without permission of the owner**
  ... (b) downloads, copies or extracts any data, computer data base or
  information from such computer..."* → *"liable to pay damages by way of
  compensation."* **Civil compensation only.**
- **s.66** — *"If any person, **dishonestly or fraudulently**, does any act
  referred to in section 43, he shall be punishable with imprisonment...
  or with fine which may extend to five lakh rupees or with both."*
  Explanation: **"dishonestly" and "fraudulently" carry their IPC s.24
  and s.25 meanings** — deception to cause loss.

**A scraper fetching pages the server willingly serves is not deceiving
anyone into granting access.** The criminal hook is not the exposure for
ordinary scraping.

**No s.43(3) good-faith proviso exists any more** — removed by the IT
Amendment Act 2008. Do not write it in a comment.

## The Tier 1 rule is overstated — split it

SEBI circular, Feb 2022, *"Approach to securities market data access and
terms of usage of data provided by data sources in Indian securities
markets"*:

> *"As far as the data provided by various data sources in Indian
> securities markets pursuant to regulatory mandates for reporting and
> disclosure in public domain are concerned, such data should be made
> available to users, 'free of charge' both for 'viewing' the data as
> also for download ... as well as their **usage for the value addition
> purposes**."*

> ⚠️ sebi.gov.in was unreachable; the above is a verbatim quote published
> by XBRL International with the circular URL. Verify before relying.

There is also a SEBI circular dated 20 Dec 2024 requiring exchanges to
split market data into two baskets — basket 1 (anonymised trading
statistics, shareholding patterns, investor grievances, **historical
prices**, volatility) shareable free up to 2 GB/year per researcher;
basket 2 (KYC, client-wise holdings, trading logs) restricted.
⚠️ Known only via MoneyControl secondary reporting — **do not quote
without reading the circular.**

**Correct three-tier structure:**

| Tier | Content | Basis |
|---|---|---|
| **1a. Licensed** | Real-time, depth, corporate actions | Contractual; must pay; redistribution controlled |
| **1b. SEBI-mandated public disclosure** | Filings, bhavcopy, corporate announcements, shareholding | Regulator says free to download and use for value-add |
| **2. Public web** | News RSS, GDELT, RBI DBIE | ToU + copyright + courtesy |
| **3. Prohibited** | Login/CAPTCHA/paywall bypass, insider tips, personal data | Contract + statute |

A blanket "scraping nseindia.com is prohibited" conflates 1a with 1b.

## The authorised vendor list

TrueData and Global Datafeeds are both correct. Current list (26
real-time vendors, from NSE's own PDF):

Activ Financial, Bloomberg, Cogencis, FactSet, Fidessa, FIS Global,
**Global Financial Datafeeds LLP**, Imagine Software, Instinet, ICE Data
Services, Liquidnet, Market Pulse Technologies, Morningstar, Quantsapp,
Reliable Software Systems, Refinitiv, SIX Financial Information, Spider
Software, Trading Technologies, Transaction Network Services, Ticker Data
Ltd, **Trading View Inc.**, **TrueData Financial Information Pvt Ltd**,
Virtu ITG Holdings, Xignite Inc, Vtrender Charts.

Canonical URL: `nseindia.com/market-data/real-time-data-subscription` →
"List of Authorized Realtime Data Vendors (.pdf)".

Two corrections:
1. This list is **real-time only**, and entitlements are **per segment**
   (CM, F&O, CD, Debt, Index). An older archived version carried a
   separate 20-entry "1 Min Snapshot Data Vendors" list.
2. **TradingView Inc. is on the list.** Treating TradingView as
   "free/public" is wrong at the exchange level.

The list changes — the current version dropped Accelpix, Citi, Deutsche
Bank, Two Sigma, Proseon, Investment Technology Group, and added Virtu ITG
and Vtrender. **Do not hardcode vendor names.**

## robots.txt has no legal force in India

- **No Indian court has decided the issue.** Searched specifically; nothing.
- The European Parliament's 2025 study: *"There is no law stating that
  /robots.txt must be obeyed, nor does it constitute a binding contract
  between site owner and user."*
- *Ziff Davis v. OpenAI*, 2025 WL 3635559 (S.D.N.Y. 15 Dec 2025) called it
  a "keep off the grass sign" — not controlling access. Not Indian law.
- Strongest pro-robots.txt material is **academic argument, not holding**:
  Chang & He, *"The Liabilities of Robots.txt"*, *Computer Law & Security
  Review* (2025), arXiv:2503.06035, argues it can serve as notice
  sufficient for tortious liability **in common-law jurisdictions**.
- The real point: **the ToU creates the obligation; robots.txt only
  supplies the parameter.**

**Recommended comment wording:**

> `robots.txt is a voluntary technical convention with no legal force in
> India (no Indian case law); respected as risk management and because it
> is frequently incorporated by reference into a site's Terms of Use,
> which IS an enforceable contract under the IT Act ss.4 and 10A.`

## Copyright — s.52 does not say what the design claims

**s.52(1)(a)** is genuine fair dealing, for purpose of: (i) private or
personal use including research; (ii) criticism or review; (iii) reporting
of current events. The **Explanation** adds: *"The storing of any work in
any electronic medium for the purposes mentioned in this clause ... shall
not constitute infringement."*

Three problems citing it for "headline + summary + link":

1. **s.52 is a closed list.** *Super Cassettes Industries v. Chintamani
   Rao*, 2011 SCC OnLine Del 4712 — exhaustive; courts cannot devise new
   exceptions for new technology.
2. **"Private or personal use" historically excluded commercial use.**
   *Super Cassettes v. Hamar Television*; *Tips Industries v. Wynk Music*.
3. **"Headline + summary + link" is not a purpose s.52 recognises.** The
   Explanation's storage protection is available only once you've
   established one of the listed purposes.

**But this changed on 24 July 2026.** *ANI Media Pvt. Ltd. v. OpenAI
Opco LLC*, CS(COMM) 1028/2024, Delhi HC (Justice Amit Bansal J.) held at
the interim stage that storing and using ANI's news to train ChatGPT was
fair dealing under **s.52(1)(a)(i)**. A profit-making entity is not
automatically barred; "research" is not confined to human researchers;
three fairness factors adopted (use limited to training; no economic
competition; public interest).

**Four hard caveats:** it is an **interim** ruling on a preliminary
injunction, expressly with "no bearing on the final outcome"; it left open
whether OpenAI permanently stores training data to memorise works; it
relied on no paywall being broken and an opt-out being available; and it
concerns AI training, not a news-signal pipeline. **Extending it is
inference, not holding.**

**The correct justification, which is simpler and stronger:**

> Copyright does not protect facts, ideas, figures or news events — only
> the expression (*R.G. Anand v. Deluxe Films*, AIR 1978 SC 1613). The
> exposure from storing full text is the **reproduction** right (s.14);
> parsing does not avoid it. The exposure from republishing is
> **communication to the public** plus **market substitution** — precisely
> the factor the ANI court treated as decisive. So "headline + summary +
> link, never full text, never republish" is correct risk reduction —
> justified as avoiding reproduction of protected expression and market
> substitution, **not** as permitted by s.52.

**No express TDM exception exists in Indian law.** Commerce Ministry has
constituted an 8-expert panel to assess whether the Copyright Act fits AI.
Compare Japan Art. 30-4, Singapore s.244, EU CDSM Arts. 3–4. India has
none. That is the actual grey zone.

## DPDP Act 2023 — phased commencement

Assent 11 Aug 2023. Commencement by notification **G.S.R. 843(E),
13 Nov 2025**:

| Date | What starts |
|---|---|
| 13 Nov 2025 | Definitions; Data Protection Board; rule-making powers |
| 13 Nov 2026 | s.6(9) Consent Managers; Rule 4 registration |
| **13 May 2027** | **ss.3–5, 6(1)–(8)&(10), 7–17; penalties ss.28–34** |

**As of Oct 2026 the substantive obligations and penalties are not yet in
force.** Penalties when they are: s.8(5) safeguards Rs 250 crore; s.8(6)
breach notification Rs 200 crore; children's data Rs 200 crore; Significant
Data Fiduciary Rs 150 crore; any other breach Rs 50 crore.

**No special "sensitive personal data" tier exists.** The 2019 Bill's SPDI
concept was dropped; the enacted Act treats all personal data alike.

**If you store no personal data, DPDP does not apply.** s.2(i) defines a
Data Fiduciary by processing **personal data**; s.2(t) defines personal
data as data about an **identifiable** individual. No identifiable
individuals ⇒ not a Data Fiduciary ⇒ no duties, no penalties, no breach
notification. That is a scope conclusion, not an exemption.

**Three traps:**

1. **HTTP logs defeat this.** A retained client IP can be personal data
   and make you a Data Fiduciary. Scrub or truncate IPs.
2. **No publicly-available-data exemption** in the enacted Act.
   If you scrape an article byline or a promoter's name, DPDP applies in
   full from 13 May 2027.
3. **Any user account, waitlist, email or auth log** makes you a Data
   Fiduciary. Gate on "do we handle personal data? yes/no" and force the
   full track on "yes".

**No GDPR Article 6(1)(f) equivalent.** DPDP s.7 is a closed list of nine
"certain legitimate uses" with no balancing test and no catch-all. And
s.7(a)'s "voluntary provision" limb likely excludes scraped data, because
nobody volunteered it to you. Do not rely on it.

## The real Indian scraping case, and how it was decided

**OLX BV & Ors. v. Padawan Ltd. & Ors.**, CS(COMM) 232/2016, Delhi HC
(Justice Rajiv Sahai Endlaw), 15.12.2016.

- OLX sued to restrain scraping of listings, photographs and content,
  pleading breach of ToU, trespass to chattel, copyright, trademark.
- Padawan had copied listings and **re-posted them on its own site**.
- Interim order (31.03.2016) restrained copyright violation and trademark use.
- Final decree: **permanent injunction passed ex parte** on pleadings,
  after the defendant shut down and stopped resisting. No reasoned
  analysis, no evidence, defendant unrepresented.

**What it does NOT establish:** that scraping is unlawful per se in India.
There is no holding on private-use scraping. The law firm's gloss — *"no
infringement if used for private use"* — is **editorial commentary, not the
court's words.**

## What to actually change in the design

1. **Split Tier 1** into licensed feed (pay) vs. SEBI-mandated public
   disclosure (free). Verify both SEBI circulars from primary source first.
2. **Re-label every Tier 2 control by its real basis.** Each is
   engineering or contract, not statute. An honest risk register is
   defensible in code review in a way fake citations are not.
3. **Delete "InvestingNote" and "Indian Express v. Zeal."** Both inventions.
4. **Rewrite the copyright justification** around facts-not-expression
   plus no republication plus no market substitution.
5. **Add an explicit "do we touch personal data?" gate**, and make IP-log
   scrubbing part of Tier 2.
6. **Get counsel on one narrow question:** whether NSE's ToU negates
   "permission" for IT Act s.43 in the Indian view. That single question
   decides whether scraping nseindia.com is a civil-compensation risk or a
   contract risk. Nobody has answered it in a reported Indian case.

**Bottom line:** no Indian statute regulates web scraping as such.
Everything in Tier 2 rests on contract + copyright + tort, and the case
law is thin and mostly unreasoned. The compliance layer is risk management
dressed in the vocabulary of law — which is fine and worth doing, but the
comments should say so, because the invented citations are what will cost
credibility the day a publisher's counsel reads them.