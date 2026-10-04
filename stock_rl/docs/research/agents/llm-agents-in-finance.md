# LLM Agents in Finance — Evidence Review

**Bibliography correction:** the paper commonly cited as "Deng et al. 2024"
is mis-attributed. *"Can Large Language Models Trade?"* is **Lopez-Lira
(2025), arXiv 2504.10789** — single author. Deng et al. is a different
group (FinRL-DeepSeek, AI-Trader). Both appear below where relevant.

## Verdict

**Mostly no.** Where LLM context has been tested properly with a fair
baseline, it does not help or the effect is unmeasurable. And the
multi-agent orchestration is **measurably harmful**.

## The definitive test: FINSABER

Li, Kim, Cucuringu, Ma — **KDD 2026 Oral**, arXiv:2505.07078, 7 revisions
through Jun 2026. Re-ran published LLM-agent results under bias-corrected
conditions, 2004–2024, including delisted S&P constituents:

| Setup | Buy & Hold Sharpe | FinMem | FinAgent |
|---|---|---|---|
| Random 5 (91 symbols) | **0.315** | −0.253 | 0.094 |
| Momentum Factor (84) | **0.384** | 0.025 | 0.104 |
| Volatility Effect (63) | **0.703** | −0.228 | 0.241 |
| FinCon Selection (80) | **0.389** | −0.292 | −0.076 |

Paired t-tests, **Buy & Hold vs FinAgent: p = 7.7e-4, 1.17e-2, 5.9e-4,
3.8e-3** — buy-and-hold beats both agents **significantly in every setup**.
CAPM decomposition: **neither agent produces statistically significant
alpha (all p > 0.34)**; FinMem's alpha is negative in every scenario. They
note the LLMs were not even given the benefit of the doubt on
contamination: *"our reported LLM performances do not adjust for potential
data leakage… but they still fail to outperform traditional strategies."*

**Why the originals looked good** — Table 1 of FINSABER is the kill shot:

| Method | Eval period | Eval symbols |
|---|---|---|
| **TradingAgents** | **3 months** | **3** |
| FinMem | 6 months | 5 |
| FinAgent | 6 months | 6 |
| FinCon | 8 months | 8 |

**StockBench** (arXiv:2510.02209, contamination-free, multi-month) agrees:
*"most models struggle to outperform the simple buy-and-hold baseline."*

**FINSABER's prescription targets this design directly:** *"prioritise
trend detection and regime-aware risk controls over mere scaling of
framework complexity."*

## The one genuinely positive result, and its built-in expiry

**Lopez-Lira & Tang**, *"Can ChatGPT Forecast Stock Price Movements?"*
(2023/2025). Strongest positive in the literature:

- GPT-4 long-short on news headlines, Oct 2021–May 2024: **Sharpe
  2.97–3.8**, ~34 bps mean daily return, 93.3% hit rate on initial reaction
- **FinBERT on the same task: Sharpe −0.33** (negative)
- Removing small caps: Sharpe 3.8 → **2.99**. Most alpha is small-cap
- Long leg Sharpe 0.90 vs short leg 2.12 — alpha is mostly the short side,
  which is expensive to borrow (Stambaugh-Yu-Yuan 2012: **78% of
  sentiment-anomaly profits are the short leg**)

Three caveats, fatal here:

1. **~190% daily portfolio turnover.** At 20 bps round trip the strategy is
   unprofitable. The paper says so.
2. **The signal decayed as it spread**: Sharpe 6.54 (late 2021) → 3.68
   (2022) → 2.33 (2023) → **1.22 (first 5 months of 2024)**. The authors
   explicitly model and observe this arbitrage-away.
3. US large/mid-cap with US news flow. **Nothing establishes transfer to
   Nifty-50.**

An independent replication (Johnsen & Shasharina, Feb 2026, 12 semis, 30
days) found the FinBERT signal **inverted** — FinBERT-positive days averaged
**−0.37%** next-day, spread **−0.03%**. Their LLM variant got r = 0.214
with a +1.26%/−1.26% spread that **vanished entirely at lag +2**. One-day
horizon only.

## Multi-agent decomposition: the direct evidence

**"Multi-Agent Debate for Explainable Trading"** (Huang et al., Stanford,
arXiv:2609.29701) built a **LangGraph propose→critique→revise→judge loop
with macro/value/risk/technical/sentiment roles — structurally identical to
the design doc's LangGraph bull/bear/judge.** 210 controlled runs, 35
quarterly scenarios, 2021Q4–2025Q3, US equities:

| Finding | Number |
|---|---|
| Reasoning quality vs Sharpe | **r = +0.07, p = 0.29** |
| Reasoning quality vs return | **r = +0.03, p = 0.70** |
| Within-scenario Δreasoning vs ΔSharpe | r = −0.02, p = 0.87 |
| **Debate beat naive equal-weight** | **33% of the time (binomial p < 0.001)** |
| Enriched prompts raised reasoning score 0.72 → 0.84 (+17.7%, d ≈ 2.0) | **no return improvement** |
| **4-agent, 5-round debate return** | **−0.27%** (while achieving the *highest* reasoning quality, ρ̄ ≈ 0.83–0.85) |

Three agents with basic prompts: **−3.64%**. Equal-weight: −3.28%.
**Debate loses.**

**Mechanism — sycophantic convergence.** Cash allocations converge from
10–50% at proposal to 55–61% by round 5; Jensen-Shannon divergence between
agent portfolios collapses after the *first* revision. **Three different
LLM providers produced byte-identical portfolios** on the same scenario.

The authors' own explanation is the part to internalise:

> *"The value of multi-agent debate in financial decision-making arises
> less from improving individual reasoning quality than from **preserving
> independent informational signals** across agents."*

That is an argument for **ensembling decorrelated feature extractors** — a
linear model with independent inputs — **not** a LangGraph debate graph.
Their Future Work: *"distill our strongest configurations… into
**single-agent prompts** to capture multi-agent benefits at lower latency
and cost."* The authors conclude the decomposition is not the point.

Their only statistically significant gain came from a **Jensen-Shannon
anti-convergence guard** (Sharpe +0.14, p = 0.028) — and adding a third
risk agent killed it (p = 0.53).

Economic conclusion: *"Under even weak-form market efficiency, this
information is largely incorporated into prices shortly after it becomes
available. In this setting, improved reasoning about the same information
cannot easily generate alpha."*

Supporting survey: *"Toward Reliable Evaluation of LLM-Based Financial
Multi-Agent Systems"* (arXiv:2603.27539) lists five failure modes that
*"can reverse the sign of reported returns."* Note its own caveat that the
"Coordination Primacy Hypothesis" is *"a falsifiable research hypothesis…
rather than an empirically validated conclusion"*, and that the Tier-2
ablation evidence it cites comes from **the same papers FINSABER showed to
be broken**.

**Auditability benchmark (2026):** an evidence-ledger survey of 77 studies
found only **19** satisfied action-output plus closed-loop evaluation. Of
those: **2/19 report time-consistent splits**, **1/19 report explicit cost
models**, **0/19 reach top reproducibility tier**, **15/19 at the lowest**.
**The field does not currently know whether multi-agent decomposition
helps.**

## The orchestration is unnecessary

Evidence hierarchy, least to most complexity:

1. **Raw numeric features** — GDELT event counts, FII/DII net, India VIX,
   PCR, RBI rate surprise. Free, milliseconds, no LLM. **Test first.**
2. **FinBERT-class local classifier** — free to run. Caveat: Johnsen &
   Shasharina found its standalone trading signal **inverted** with a zero
   spread. Use as a **triage filter**, not a signal generator.
3. **A single LLM call per day** producing one scalar per ticker.
   Hybrid (FinBERT triage → judge) matched pure-LLM quality at **one-fifth
   the cost** ($0.10 vs $0.50/day for ~200 headlines).
4. **7–10 LangGraph agents with debate and a judge.** ~100x the cost.
   Stanford says it loses to equal-weight.

**Lopez-Lira & Tang's entire positive result is one prompt, one call, one
scalar per headline — no agents, no debate, no judge.**

**And if context is fused into an RL state vector, the architecture
question partly dissolves.** The RL policy is already an ensemble learner
over the state. Feeding it 10 scalar context features is strictly more
sample-efficient than asking 10 LLMs each to emit a sentiment adjective
and hoping the judge picks well.

## Cost and latency: "<15s" is not realistic

Verified list pricing: gpt-5-nano $0.05/$0.40, gpt-5-mini $0.25/$2.00,
gpt-4.1-mini $0.40/$1.60, gpt-5 $1.25/$10.00, gpt-5.6-sol $4/$20,
gpt-6-astra $10/$50 (input/output per 1M).

**7–10 agents x 50 stocks = 350–500 calls, plus bull/bear/judge ≈ +150 →
500–650 calls per inference.** At 4K input / 1K output:

| Tier | Per inference | 3 rounds/day |
|---|---|---|
| nano | ~$0.40 | **~$1.20** |
| gpt-5-mini | ~$2.00 | ~$6 |
| gpt-5 | ~$10 | ~$30 |
| gpt-6-astra | ~$55 | ~$165 |

**Reasoning tokens are the trap.** o-series/GPT-5-class models emit hidden
reasoning tokens billed as output. At 10K reasoning tokens x 600 calls =
6M output tokens → **$120/day on gpt-5, $300/day on gpt-6-astra.** And
LiveTradeBench found those models *underperform* anyway.

**Measured anchor:** the Stanford group reports *"each scenario requires
7–13 minutes and roughly **$5–$8 in API fees**"* for extended debate with
4 agents, 5 rounds, ~11 tickers, gpt-5-mini. Scaling to 7–10 agents x 50
tickers is **35–90x the token volume → $175–$720 per portfolio decision.**

**Latency is structurally impossible.** Bull-vs-bear with a judge is
**inherently serial**: bull proposes → bear critiques → both revise → judge
synthesises = **4 dependent hops minimum**. Even at 3 s/hop with perfect
parallelism that is 12 s of pure dependency, leaving zero headroom for
7–10 agents' context assembly. Realistic frontier-model figures with
reasoning: **60–180 s per inference.**

The "<15s" claim is achievable only with nano/mini, one agent, one round,
and **no debate at all** — i.e. a system that is not the design.

## Live benchmarks: reasoning harder ≠ trading better

**LiveTradeBench** (Yu, Li, You; UIUC) — 21 LLMs, 50 real trading days, live
streaming, no backtest leakage:

- US stocks: **every model made money** (+1.78% to +6.25%) — but **no
  buy-and-hold baseline is reported.** In a mega-cap bull run, "the agent
  made money" is not evidence.
- **Spearman correlation between LMArena score and stock return: 0.054.**
  In Polymarket: **−0.38.** General model capability is *negatively*
  related to trading skill.
- **Reasoning models did worse.** DeepSeek-R1, Qwen3-Thinking, GPT-o3 all
  showed elevated volatility from over-adjustment.
- Cross-market Sharpe correlation ≈ 0. GPT-4.1: best US performer
  (+6.25%), **−33.69%** on Polymarket.

**"Can LLMs Trade?"** (Lopez-Lira 2025) — the caveats are severe and
unstated in the abstract: it is an **ABIDES-style simulation**, not real
markets — no real price process, no order book, no transaction costs.
Agents are **literally given the fundamental value** (example: computes
`valuation: 28.0` from a perpetuity formula, compares to a market price of
$114.95). Prices start at a price/fundamental ratio of **3.57 vs a
historical average of 3.83** — agents were dropped into a pre-loaded
disequilibrium and watched to close it. **It cannot support a live design.**

## Narrative-is-noise, quantitatively

The strongest version of this argument is empirical:

- **Harvey, Liu & Zhu (2016, RFS):** catalogued **316+** claimed
  predictors; conventional t > 2.0 is inadequate given collective data
  mining. They require **t > 3.0**. Of 296 published significant factors,
  **158 are false discoveries**.
- **Hou, Xue & Zhang (2020, RFS):** of ~450 documented anomalies, **65%
  fail** even t = 2.0 once micro-cap influence is removed.
- **McLean & Pontiff (2016, JF):** published predictors earn **58% less**
  after publication (**72%** for post-2005 data).
- **Chen & Velikov (2022, JFQA):** across **204 anomalies**, after realistic
  costs with cost mitigation, **alpha decays 93%**, leaving the
  90th-percentile anomaly producing **6 basis points per month**.
- **Novy-Marx & Velikov (2016, RFS):** only anomalies with monthly
  one-sided turnover below **50%** survive costs. **A daily context-driven
  signal generates high turnover by construction.**
- **Muravyev, Pearson & Pollet (2025, JF):** long-short anomaly return drops
  from +0.14%/month to **−0.01%** after stock-borrowing fees.
- **Stambaugh, Yu & Yuan (2012, JFE):** **78% of sentiment-anomaly profits
  come from the short leg.**

**Ben-Rephael, Da & Israelsen (2017, RFS)**: what matters is not attention
volume but whether attention **carries information**. Johnsen &
Shasharina found exactly that: NVDA had the second-most coverage in their
dataset and sentiment had essentially zero predictive power (r = −0.03).

**This matters enormously here: Nifty-50 constituents are precisely the
large, well-covered, institutional stocks where FINSABER found LLM agents
have no significant alpha.** The coverage-dilution problem is worse than
NVDA's.

## India-specific data availability

| Source | Status | Notes |
|---|---|---|
| **NSE FII/DII** | **Free** | Aggregate daily buy/sell/net. Compiled from NSDL PANs. Historical archive available |
| **NSDL FPI trends** | **Free** | `fpi.nsdl.co.in` — daily sector-level FPI equity/debt flows. Often finer-grained than the NSE aggregate |
| **NSE F&O bhavcopy** | **Free** | Daily full derivative file (OI, volume, IV per strike). Historical from ~2016/2020 |
| **NSE participant-wise OI** | **Free** | Daily CSV, client/far/proprietary split |
| **NSE live/real-time** | **Paid** | NSE Data & Analytics leased line. This is the real licence cost |
| **GDELT** | **Free** | 100% free. Updates every 15 min, archives to 1979, 100+ languages |
| **RBI DBIE** | **Free** | Publication-schedule based — **must respect release lag to avoid look-ahead** |

**The real constraint is historical depth, not licence fees.** NSE bhavcopy
goes back ~6–10 years. Context features therefore have far less history
than price features. **An RL policy trained on 8 years of price history and
3 years of context history is learning a higher-dimensional state space
from the shorter sample. This alone can explain why context "doesn't
help."**

**FII/DII aggregate flow is a single India-wide number.** It is not a
per-stock feature — it is a regime indicator with ~50 duplicate values
across the cross-section. Handing it to a "FII Flow Agent" per stock is
structurally incoherent. It belongs in **one global state variable**.

**Options flow is the one genuinely per-stock, genuinely differentiated,
genuinely hard-to-get signal** — free daily from bhavcopy. That has the
best prior of adding information, because it is not public-information
sentiment: it is positioning.

## Verdict

**Do not build the multi-agent LLM system.** Three independent findings:

1. **The signal may not exist.** Buy-and-hold beats LLM agents
   significantly across 63–91 unbiased symbols (p ≤ 0.006). No agent
   generates significant alpha (all p > 0.34).
2. **The orchestration is actively harmful.** Debate loses to equal-weight
   67% of the time (Stanford, n=210). The 4-agent/5-round configuration
   earns −0.27% while achieving the *highest* reasoning quality. More
   agents → sycophantic convergence → byte-identical portfolios.
3. **Reasoning tokens are the anti-pattern.** Reasoning-specialised models
   underperform in live trading across two independent 2025 benchmarks.
   The proposal builds a system whose main cost line item buys the
   capability that empirically hurts.

## The minimal falsifiable experiment

**This is the whole thing. Do it before writing a single LangGraph node.**

**Hypothesis:** non-price context adds predictive value beyond
price/technical features for Nifty-50 constituents.

**Setup** — Universe: current Nifty-50 constituents, daily, 2015–2026
(~2,750 sessions). Backbone: **the existing engine, unchanged.** ≥5 seeds.
Walk-forward: 5 expanding-window folds. **Time-consistent splits only**
(2/19 of the field's primary studies even do this).

**Three arms:**

| Arm | State vector |
|---|---|
| **A (control)** | Price/technical only |
| **B (numeric context)** | A + ~20 scalars: GDELT event counts (company + India macro, 1d/3d/7d), NSDL sector FPI flow, NSE FII/DII 5d cumulative net *(one global var, not per-stock)*, India VIX, RBI repo-rate surprise, PCR from bhavcopy, IV rank, 20d sector-relative correlation |
| **C (B + LLM scalar)** | B + one scalar per ticker from **one** LLM call per day: *"is this news material and good or bad for \<ticker\> over 1–3 days? answer with a single number in [-1,1] or 0 if immaterial"* |

**Statistics — be strict or you will fool yourself:**

- Diebold-Mariano on realized equity curves: A vs B, B vs C
- Paired t-test on per-stock Sharpe across the 50 names
- Report **CAPM alpha with a p-value**, not just Sharpe
- Require **|t| > 3.0** (Harvey/Liu/Zhu), not 2.0
- Must hold in **≥4 of 5** folds

**Pre-registered kill criteria — decide these before seeing results:**

- If **B − A** has p > 0.1 on Sharpe or alpha → **delete all numeric
  context work.** Do not proceed to LLM.
- If **C − B** has p > 0.1 → **never build LLM calls.** Use B or A.
- If daily turnover exceeds 50% monthly one-sided → abandon regardless of
  backtest return (Novy-Marx & Velikov).
- If it survives → *then and only then* consider **one** LLM summariser
  call. Never 7–10 agents.

**Cost:** Arm C at gpt-5-nano for 50 tickers x 2,750 days ≈ 137,500 calls
≈ **$25–40 total.** FinBERT local = $0. Arms A/B = no LLM cost.
**Timeline:** ~10–14 days including data plumbing.

**Two free facts worth getting regardless of outcome:**

1. **Options positioning from free bhavcopy** is the only context signal
   with a genuine informational-asymmetry prior. Not public sentiment;
   what large participants actually did.
2. **Nothing beats 1/N.** DeMiguel (2009) showed 1/N beats almost every
   optimized portfolio strategy across 7 datasets. Stanford found debate
   agents lose to 1/N two-thirds of the time. **Your benchmark must be 1/N
   over Nifty-50 and Nifty-50 buy-and-hold — not a trained RL baseline.**

## If you do eventually build agents

The only defensible architecture from the evidence is **independent
feature extractors averaged with equal weights**, plus an explicit
anti-convergence guard (the Stanford JSD intervention, Sharpe +0.14,
p = 0.028). **An LLM judge is strictly worse than a plain mean** — Stanford
compared both and mean-vote baselines were competitive. And use **3 agents,
not 10**.

## Compliance note

SEBI 2012 circular para 4(iii) requires algo servers **located in India**
with no interlink outside India. A **US-region LLM API** ships Indian
market data and live trading signals to a foreign datacenter. Run
inference on the same box (Ollama/vLLM) or in an Indian region — which
also removes per-token cost.