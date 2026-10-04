# Signal Attribution and Audit Trail

Answers two questions: what attribution method is right for a trading
signal, and what India actually requires you to record.

**Method note:** `www.sebi.gov.in` was DNS-blocked from the research
sandbox. Exchange mirrors were used and two operative SEBI circulars were
verified in full text. Secondary sources are marked as such.

## The headline compliance correction

**There is NO Indian requirement to explain an individual BUY/HOLD/SELL
decision.** Stating that plainly because the design doc implies otherwise.

### Retention — a genuine conflict in the sources

Two independent research passes disagreed. Both agree the SEBI algo
circular is **silent**:

> **CIR/MRD/DP/09/2012**, para 8(iv): *"The stock broker shall maintain
> logs of all trading activities to facilitate audit trail. The stock
> broker shall maintain record of control parameters, orders, trades and
> data points emanating from trades executed through algorithm
> trading."*

No number. Where the figures come from:

| Figure | Source | Applicability |
|---|---|---|
| **5 years** | **NSE Detailed Operational Modalities** (retail algo, Jul 2025) **para 10.3**: *"The audit trail data should be available for at least 5 years."* Corroborated twice in the same document — the system-auditor checklist requires a Backup Policy and an Audit Trail Policy, both *"minimum 5 years"* | Exchange rule, binding on trading members and API algo providers |
| **8 years** | SEBI (Stock Brokers) Regulations 2026, Reg. 16 — books of account and records | General broker-records obligation; the algo trail is captured within it |
| 7 years | **No source.** Delete it | Wrong |
| 3 years | Companies Act 2013 s.129(1) for an **unlisted** company | Wrong regime |

**Defensible position for a proprietary desk: retain 5 years as the
regulatory floor, 8 as the conservative choice, and cite NSE para 10.3
rather than "as per SEBI".** The two figures are not contradictory —
one is the exchange's operational floor for algo trails, the other the
broader statutory books-of-account duty.

### What IS required

| Requirement | Source | Granularity |
|---|---|---|
| Algo ID tagging for audit trail | 2012 circular 6(vi); Feb 2025 circular II(b) | **Per order** |
| Logs of control parameters, orders, trades, data points | 2012 circular 8(iv) | Per order/trade |
| Strategy details on demand for inquiry/surveillance/investigation | 2012 circular **4(vi)** | Per algo |
| 5-level category disclosure + Strategy writeup + RMS writeup | NSE modalities 8.2.1.1, 8.2.5 | Per algo, at registration |
| Black-box algo: SEBI Research Analyst registration + research report | Feb 2025 circular V(a)(ii) | Per algo |
| Notify exchange of any modification | 2012 circular 8(v) | Per change |
| 16 pre-trade RMS checks | NSE 11.1 | Per order |
| Monthly mock/simulation participation | NSE 12.1 | Monthly |
| No self-crossing of orders | NSE 10.1 | Continuous |
| **Re-registration on any change to decision logic** | **NSE 9.1, 9.9** | Per change |

**The real compliance driver is re-registration, not explanation.**
NSE para 9.9: *"No modification shall be allowed for registered Blackbox
algos. Algo Provider shall be required to apply for fresh registration in
case of any change in logic governing the algo's."* This system is Black
Box. **Any change to decision logic — including a timeframe rule or an
indicator threshold — requires fresh Exchange registration.**

That is why machine-checkable per-decision reasons are worth building:
they are the evidence that the deployed code is the registered code.

### The NNF ID and algo tag structure

15-digit NNF ID. First 12 digits = platform:

| Platform | 12-digit NNF | Algo allowed? |
|---|---|---|
| CTCL (dealer terminal) | first 6 = PIN | Yes |
| IBT | `111111111111` | **No** |
| STWT | `333333333333` | **No** |
| DMA | `222222222222` | Yes |
| **Client Direct API** | `444444444444` | **Yes, 13th digit must always be algo** |

13th digit: `0` algo, `1` non-algo, `2` algo via SOR, `3` non-algo via
SOR, `4` inter-exchange algo, `5` RMS square-off, `6` after-market, `7`
basket, `8` batch upload.

Separate **Algo ID** field. Registered algos get a unique Exchange-allotted
ID. Unregistered client algos below the OPS threshold send the literal
**`99999`**.

**Does the algo self-report details? Yes — but at registration, statically,
not per order.** Registration requires a five-level disclosure: front end;
developer (vendor/TM in-house/tech-savvy retail); user (dealer/client);
logical grouping (white/black box); strategy (white: Execution/TWAP/VWAP;
black: Arbitrage/Alpha-seeking/HFT/Scalping/Others).

### The 16 pre-trade RMS checks (NSE 11.1)

Price vs exchange bands; quantity vs per-order limit; order value; trade
price protection; **market price protection — no market orders for algo
orders**; cumulative open order value (client level, no "Unlimited");
**automated execution check** (account for executed/unexecuted/unconfirmed
before releasing more, with pre-defined auto-stoppage for loop/runaway);
net position vs margin; RBI FII-restricted securities; MWPL violation;
position limit; trading limit; exposure limit; turnover limit;
security-wise value limit; efficient price discovery (commodity).

**Correction to the design doc:** VaR checks are **not** in this list.
"Fat-finger check" is what checks 1–5 collectively are. Post-trade
surveillance is the **exchange's** obligation, not yours.

### Self-trade / wash trades

**NSE para 10.1:** *"Trading members providing API / Algo Provider facility
for routing client orders shall not be allowed to cross trades of their
clients with each other. All orders must be offered to the market for
matching."*

On your own account, self-trading falls under SEBI PFUTP Regulations 2003
Reg 4(1)(a); front-running of client orders under the PIT Regulations
2015.

> **Caveat:** only study-notes summaries of PFUTP were retrievable, not
> the gazetted text. Verify Reg 4 sub-clauses and the PIT front-running
> definitions against the primary text before relying on them. The NSE
> 10.1 quote above **is** verified.

## Urgent flag

Run through Zerodha/Upstox API on your own account, this is **Client
Direct API** — NNF `444444444444`, 13th digit **always** algo. If the
algo exceeds the threshold (**currently 10 orders per second**) it **must
be registered with the Exchange through the broker**; unregistered
below-threshold algos must send `99999` in the Algo ID field.

The design doc's *"Zero SEBI risk"* framing is **not a defensible
position**. Get the broker's compliance desk to confirm current
registration status. Note NSE/INVG/73992 (30 Apr 2026) supersedes parts of
the registration process.

## Part A — Attribution methods

| Method | What it gives | Verdict |
|---|---|---|
| **Permutation importance** (Breiman 2001) | Global ranking by degrading performance | **Use this.** Model-level, stable, cheap, handles correlated features via joint contribution |
| **SHAP / Kernel SHAP** (Lundberg & Lee 2017) | Per-prediction additive attribution | See below — problem is real |
| **Integrated Gradients** (Sundararajan et al., ICML 2017) | Path-integrated gradients | Reasonable, but path/baseline assumption is awkward over sequential market data |
| **LIME** (Ribeiro et al., KDD 2016) | Local surrogate | **Poor fit for time-series finance.** Neighbourhoods are synthetic and not temporally valid; known unstable |
| **SHAP-Lorenz** | Stability of explanation under ordering | Worth knowing — measures whether your explanation is stable at all |

**For a genuinely rule-based indicator signal** (`RSI<30 AND price>200DMA`),
post-hoc attribution is **the wrong tool and arguably harmful**. The rule
*is* the explanation. See the self-explaining models literature
(Alvarez-Melos & Parikh 2018) where the model is constrained to be
intrinsically interpretable. Bolting LIME onto an already-legible decision
procedure introduces noise into the audit trail.

**For an RL policy**, attribution is genuinely hard and post-hoc SHAP on a
180-dim state is the weakest thing in the stack. Two stronger options:

1. Make the *decision procedure* auditable even if the *value function* is
   not: log the state vector, action, reward, and the small set of
   policy-derived quantities that gated the action.
2. The design doc's `reasoning_trace` field (bull/bear debate transcript)
   is closer to the right artefact than SHAP is.

### SHAP on correlated financial features — the criticism is unresolved

**Aas, Jullum & Løland (2021)**, *Artificial Intelligence*
(arXiv:1903.10464):

> *"Like several other existing methods, this approach assumes that the
> features are independent."*

> *"In observational studies and machine learning problems, it is very
> rare that the features are statistically independent, meaning that the
> Shapley value methods suffer from inclusion of predictions based on
> **unrealistic data instances when features are correlated**."*

The mechanism: standard SHAP computes `E[f(x_S)]` by marginalising over
the complement. With correlated features that evaluates the model on data
**that cannot occur** — e.g. RSI of 5 alongside a 200-DMA trend reading
that never co-existed. The attribution is an artefact of the counterfactual
distribution, not of the model. The fix is the **conditional expectation**
estimator, implemented in the R package `shapr`.

Also: *"On the failings of Shapley values for explainability"*,
*Information Systems Frontiers* — argues existing definitions "necessarily
yield **misleading** information about the relative importance of
features."

**State of the criticism: not rebutted.** SHAP remains the default because
of its clean axiomatic story, and the conditional estimator is not what
most libraries compute by default.

Financial time-series features are **doubly** problematic: correlated
across indicators *and* serially dependent. Standard marginal SHAP is
aware of neither.

If you use it, disclose that you are computing the estimator known to be
problematic, or use the conditional estimator.

### No standard rationale schema exists

There is **no published standard** for representing a rule-based trading
decision rationale in an audit schema. Genuine gap.

**FIX** gives a parameter-tagging mechanism that could be reused:
`StrategyParametersGrp` — tag 957 `NoStrategyParameters`, 958
`StrategyParameterName`, 959 `StrategyParameterType`, 960
`StrategyParameterValue`. Appears in `NewOrderSingle`, `ExecutionReport`,
and **`AlgoCertificateReport` (MsgType EJ)**. FIX has **no** field for a
per-decision reason or a timeframe choice.

**FCA/ESMA** (MiFID II Art. 17, RTS 6) require effective systems, risk
controls, thresholds, notification to the competent authority, and
annual review — but define **no "sufficient explanation" standard**. The
FCA's **Multi-firm Review of Algorithmic Trading Controls** (21 Aug 2025,
10 PTFs) found **no new rules**; its findings are the closest thing to
guidance: out-of-date documentation, unclear ownership of trade controls,
variable technical knowledge in compliance, **inconsistent testing
records**, under-resourced surveillance. Emphasis on documentation
completeness and testing evidence — **not per-trade reasons**.

**IOSCO never defines "a sufficient explanation."** PD788 identifies
explainability as the **top model risk** among members, ahead of bias,
robustness, hallucination and conflicts of interest. It warns that complex
AI *"require significant computational resources and can therefore have
associated latency — this may make them inappropriate in many algorithmic
trading contexts."* That applies directly to the design's 15 s agent path.
IOSCO PD428 and PD472 are the documents that actually shaped UK/Indian
algo rules.

**EU AI Act: algorithmic trading is NOT high-risk.** Read in full, Annex
III's eight categories are biometrics, critical infrastructure, education,
employment, essential services, law enforcement, migration,
justice/elections. **Algorithmic trading appears nowhere.** Credit
scoring (5(b)) excludes fraud detection; life/health insurance (5(c)) is
not trading. A directional Nifty signal generator is not high-risk.

On reach, Art. 2(1)(c) applies where output **is used in the Union**.
Trading NSE cash for your own account produces no such output →
**out of scope**. Art. 2(10) also excludes purely personal activity.

> **Timetable — do not quote the old dates.** Regulation (EU) 2026/1744
> (Digital Omnibus on AI), in force 27 July 2026, moved Annex III
> high-risk obligations from 2 Aug 2026 to **2 Dec 2027**, and Annex I to
> 2 Aug 2028. *(Reg number corroborated by several secondary sources; OJ
> text not retrieved directly.)*

**Build no AI-Act artefacts.** It would be inventing a requirement.

## Resolved trichotomy

| Claim | Verdict |
|---|---|
| Algo ID tagging on orders | **Hard legal requirement** |
| Audit trail ≥5 years | **Hard requirement**, sourced to NSE 10.3, not a SEBI circular |
| Logs of control parameters, orders, trades, data points | **Hard legal requirement** |
| Strategy disclosure on demand | **Hard legal requirement** (2012 circular 4(vi)) |
| Strategy + RMS writeup, 5-level category at registration | **Hard legal requirement** |
| Black-box algo: RA registration + research report | **Hard legal requirement** |
| **Re-registration on any logic change** | **Hard legal requirement**, strict for black box |
| 16 pre-trade RMS checks; monthly mock session | **Hard legal requirement** |
| **Per-decision reason code / attribution** | **Our own engineering choice. No rule requires it** |
| **Per-decision timeframe rationale** | **Our own engineering choice** |
| Machine-readable rationale schema | **Ours to design.** No standard exists |
| SHAP for attribution | Industry practice — and the **wrong** practice on correlated financial features |
| "7 years as per SEBI" | **Wrong. Delete it** |

## The defensible framing

Build per-decision rationale because it is genuinely valuable — and
justify it honestly:

> **Strategy-identity evidence**, required by NSE modalities 9.1/9.9
> (re-registration on logic change) and 8.2.5 (Strategy writeup),
> extended to per-decision granularity as the mechanism proving deployed
> behaviour matches registered behaviour. Per-decision rationale is **our
> internal control**; it is not itself a SEBI requirement.

Schema to emit: decision, rule/condition IDs, indicator values at decision
time, timeframe selected, timeframe selection criterion, policy version
hash, feature-vector hash, and the rule tree that fired. **The policy
version hash is what actually discharges the re-registration obligation.**