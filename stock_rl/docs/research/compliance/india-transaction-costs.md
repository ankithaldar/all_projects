# India Transaction Costs — Verified Rates

All rates verified against **primary sources**, October 2026.

## Sources

| Source | Covers |
|---|---|
| NSE investor levies page (`nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies`), page-stamped **updated 17 Apr 2026** | STT, stamp duty, SEBI turnover fee, GST rate |
| **NSE/FA/73061**, 27 Feb 2026, effective 1 Mar 2026 | Exchange transaction charge + IPFT reclassification |
| **SEBI (Stock Brokers) Regulations 2026**, notified 7 Jan 2026, Reg. 41 | SEBI turnover fee schedule |
| Zerodha, Angel One, Sharekhan, Upstox, Flattrade published schedules | Brokerage and DP charges |

## Verification result

| Field | Ours | Correct | Status |
|---|---|---|---|
| `brokerage_pct` delivery | 0.0 | 0.0 (retail individual, discount broker) | OK, with caveat |
| `brokerage_pct` **intraday** | **0.0** | **0.03% or Rs 20/order** | **WRONG — fixed** |
| `stt_buy` / `stt_sell` delivery | 0.001 | 0.001 | OK |
| `stt_buy` intraday | 0.0 | 0.0 | OK |
| `stt_sell` intraday | 0.00025 | 0.00025 | OK |
| `exchange_pct` | **0.0000297** | **0.0000307** | **STALE — fixed** |
| `sebi_pct` | 0.000001 | 0.000001 | OK |
| `stamp_duty_buy` delivery | 0.00015 | 0.00015 | OK |
| `stamp_duty_buy` intraday | 0.00003 | 0.00003 | OK |
| `gst_pct` on fees, not STT | — | Correct | OK |
| `dp_charge` | 15.34 | 15.34, but **per day per scrip** | semantics corrected in docs |
| `slippage_bps` | 5.0 | Unverifiable | our assumption |

## The three fixes

**1. Intraday brokerage is not free.** Discount brokers charge 0.03% per
executed order capped at Rs 20 (Zerodha, Dhan); Groww 0.05%, Angel One
0.1%, Upstox flat Rs 20. At zero this understated a real intraday round
trip by **~60%**, because brokerage then dominates the 3.55 bps of
statutory levies. The cap binds below **Rs 66,667 per order**.

**2. Exchange transaction charge is 0.00307%, not 0.00297%.** From
NSE/FA/73061, effective 1 Mar 2026:

| Period | "Transaction charge" | IPFT contribution | Total client cost |
|---|---|---|---|
| Oct 2024 – Feb 2026 | Rs 297/crore | Rs 10/crore | **Rs 307/crore** |
| Since 1 Mar 2026 | Rs 306.99/crore | Rs 0.01/crore | **Rs 307/crore** |

The client always paid 0.00307%. The old figure is the pre-reclassification
line item. SEBI's "True to Label" circular
(`SEBI/HO/MRD/TPD-1/P/CIR/2024/92`, 1 Jul 2024) requires charges recovered
from end clients to equal what the member pays the MII, so the whole Rs
307 passes through. BSE cash is higher at 0.00375%.

**3. DP charges are per day per scrip, not per order.** Zerodha: "If you
sell shares of one company once or multiple times in a day, you pay DP
charges only once." Charged twice if sold in both the normal and auction
markets. Rs 15.34 is Zerodha-specific and **already GST-inclusive**;
range across brokers is Rs 15–25 incl. GST. Not a central tariff —
NSDL levies its DPs, who set their own fees.

## Confirmed: GST is not charged on STT

NSE's own levies page, Stock brokers Services category, **18.00%**, payable
by brokers and collected from clients. GST applies to brokerage + SEBI
turnover fee + exchange transaction charges + IPFT. **Not** to STT or
stamp duty — both are themselves taxes.

GST 2.0 (56th Council, 3–4 Sep 2025, effective 22 Sep 2025) collapsed the
structure to 5%/18% slabs but **did not change the broking rate**. NSE's
page as of Apr 2026 still shows 18.00%.

## Correction: the Oct 2024 STT increase did not touch equity cash

NSE's table shows both old and new columns. The 1 Oct 2024 revision
(Finance (No. 2) Act 2024) was **F&O only**: futures 0.0125%→0.02%,
options premium 0.0625%→0.1%. Delivery has been 0.1% both sides since
1 Jul 2020; intraday 0.025% sell-only since 1 Oct 2020.

**A newer change:** Budget 2026-27 (Finance Act 2026) raised F&O STT
effective 1 Apr 2026 — futures 0.02%→0.05%, options premium 0.10%→0.15%,
exercised 0.125%→0.15%. Equity delivery and intraday untouched.

**Base:** STT is levied on the **VWAP of the taxable transaction, per side
separately**. Not the limit price, not your fill-weighted average.
(Zerodha's intraday worked example uses a combined buy+sell average — a
broker presentation quirk, not the statutory rule.)

## All-in round trip

### Delivery — **22.25 bps** (levies only)

| Component | bps |
|---|---|
| STT (0.1% buy + 0.1% sell) | **20.000** |
| Stamp duty (0.015%, buy only) | 1.500 |
| Exchange txn chg (0.00307% x 2) | 0.614 |
| SEBI (0.0001% x 2) | 0.020 |
| GST 18% on (exchange + SEBI) x 2 | 0.114 |
| **Total** | **22.248** |

**STT is 90% of delivery cost.** Nothing else you tune matters.

DP as a function of ticket size: 1.53 bps at Rs 1L, 0.31 at Rs 5L, 0.015
at Rs 1 crore. With 5 bps/side slippage at a Rs 10L ticket: **~32.3 bps
all-in**.

### Intraday — **3.55 bps** levies / **9.55 bps** with brokerage

| Component | bps |
|---|---|
| STT (0.025%, sell only) | 2.500 |
| Stamp duty (0.003%, buy only) | 0.300 |
| Exchange txn chg x 2 | 0.614 |
| SEBI x 2 | 0.020 |
| GST x 2 | 0.114 |
| **Levies** | **3.548** |
| Brokerage 0.03% x 2 | +6.000 |
| **Total** | **9.548** |

With 5 bps/side slippage: **~19.5 bps**. Brokerage is **+63% of the true
intraday cost** — bigger than STT. This is why zeroing intraday brokerage
was the single most damaging rate bug.

## Turnover is what decides feasibility

One-way cost is roughly 11–13 bps; round trip 22–27 bps.

| Annual turnover (one-way) | Annual cost drag |
|---|---|
| 50% | 0.12% |
| 100% | 0.25% |
| 200% | 0.50% |
| 400% | 1.00% |
| 800% | 2.00% |

For a quarterly-rebalanced Nifty-50 fundamental strategy, expected turnover
is 25–40%/yr one-way → **0.06–0.10% annual drag, negligible**. Costs are
decisive only for daily/intraday or monthly signals. This is a point in
favour of low-turnover fundamentals, and it does **not** transfer from the
US high-turnover factor literature where "anomalies die after costs" is
calibrated.

## Pending changes: none affect equity cash

GST on broking, STT on equity cash, stamp duty, exchange charges and the
SEBI turnover fee all have **no announced or pending change**.

Live but irrelevant to an equity cash model: SEBI's CAS consultation
(comments closed ~3 Oct 2026), SEBI (Mutual Funds) Regulations 2026
(eff. 1 Apr 2026), ETF base-price circular (timeline extended 28 Aug 2026),
and the **NSE Closing Auction Session** (`NSE/FAOP/74467`, 29 May 2026) —
worth knowing if you trade near the 15:30 close, since CAS prints carry
different microstructure and slippage characteristics.

## Known ceilings in our model

1. **DP charged per order, not per day per scrip.** On daily bars the two
   coincide. An intraday caller is over-charged when selling one symbol
   twice in a session.
2. **STT modelled additively per fill, not on the side's session VWAP.**
   Slicing one large order into pieces slightly overstates STT.
3. **Delivery brokerage of 0.0 assumes a retail individual** at Zerodha or
   Groww. Zerodha charges delivery 0.1% or Rs 20 to HUF, trust, company,
   partnership and LLP accounts. Angel One and Upstox charge all clients.
4. **Slippage of 5 bps is entirely our assumption.** No source verifies or
   refutes it. At 5 bps per side, delivery is ~32 bps, not 22.

Upgrade path for 1 and 2: a dated rate card plus a per-session
accumulator. `CostModel` is already the whole interface, so neither
requires touching the arithmetic.

## Uncertainties, stated plainly

1. Rs 15.34 DP is a **broker** source, not a regulator one. NSDL confirms
   no central investor DP tariff exists. Range Rs 15–25 incl. GST.
2. The 0.0030699 / 0.000000001 split is verified from the NSE/FA/73061 PDF
   directly. Whether every broker's contract note prints "0.00307%" is the
   broker's formatting choice.
3. SEBI True-to-Label circular text was not retrievable directly from
   sebi.gov.in; its number and date are confirmed and its substance is
   quoted verbatim inside two NSE circulars read in full.
4. The **Closing Auction Session** may change execution costs near the
   close. Untested here.