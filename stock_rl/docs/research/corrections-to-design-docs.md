# Corrections to the design docs

Claims in `../NIFTY-RL-Technical-Doc.md` and `../PORTFOLIO-HEDGING-RL.md`
that research found to be **fabricated, incorrect, or measured to lose
money**. Do not encode these; several were bugs in our own code.

## Fabricated — delete entirely

| Claim | Finding |
|---|---|
| "InvestingNote" is an NSE authorised data vendor | **Does not exist.** Appears on no NSE list. TrueData and Global Datafeeds are real; use NSE's current 26-vendor PDF |
| "Indian Express v. Zeal" scraping case | **Does not exist.** The real Indian scraping case is *OLX BV v. Padawan Ltd.*, CS(COMM) 232/2016, Delhi HC, 15.12.2016 — an *ex parte* injunction on copyright and republication, with no holding that scraping is unlawful per se |
| "SEBI algorithmic trading circular, April 2021" | **Does not exist.** SEBI's own index shows nothing algo-related in April 2021. The document everyone cites is a **Consultation Paper dated 9 Dec 2021**, filed under "Reports", not "Circulars" |
| "amending the 2010 guidance" | **No SEBI 2010 algo circular exists.** The foundation is **30 March 2012, CIR/MRD/DP/09/2012** |
| "SEBI distinguishes static and dynamic algorithmic trading" | **Hallucination-grade.** Not in any SEBI or NSE instrument. The real classifications are approved vs non-approved, white-box vs black-box, and Execution / Arbitrage / Alpha-seeking / HFT / Others |

## Incorrect — correct these

| Claim | Finding |
|---|---|
| "Violation = IP block + legal notice under IT Act 2000" | **Embellishment.** IT Act s.43 gives **civil compensation only**. The criminal hook, s.66, requires acting "dishonestly or fraudulently" in the IPC s.24/25 sense. A server willingly serving pages is not deception. NSE's ToU clause 9 bans automated collection contractually; that is the real exposure |
| "Audit log immutable for 7 years as per SEBI" | **Wrong number.** SEBI (Stock Brokers) Regulations **2026**, Reg. 16, notified 7 Jan 2026, requires **8 years**. The exchange-level floor for TM API audit trails is 5 years (BSE Annexure 2 s.10.3). The "3 years" figure in circulation is **NYSE Rule 105** — wrong jurisdiction |
| "max 1 req/sec per domain" as a compliance requirement | **No legal source found**, in India or elsewhere. It is engineering risk management against IT Act s.43(e)/(f) disruption provisions. Do not label it a legal requirement |
| "check robots.txt" | **Not legally binding.** Voluntary technical convention with **zero Indian case law**. It matters only where a site's Terms of Use incorporates it, and a ToU *is* enforceable |
| "store headline+summary per Indian Copyright Act s.52 fair dealing for research" | **Misreading.** s.52 is a closed list (*Super Cassettes Industries v. Chintamani Rao*, 2011), and "private or personal use" historically excluded commercial use. The correct basis is that copyright does not protect facts, only expression — so avoid reproduction and market substitution. *ANI Media v. OpenAI* (Delhi HC, 24.07.2026) helps at interim stage only |
| "DPDP Act 2023: don't store PAN/Aadhaar" | Right instinct, **wrong statute.** DPDP has no sensitive-data tier. PAN is restricted by the Income-tax Act s.360 and the Income-tax (Disclosure of PAN) Rules, 2010; Aadhaar by the Aadhaar Act, 2016 s.4(4) and s.29. Separately, DPDP is **out of scope entirely if no identifiable individuals are processed** |
| "DPDP doesn't apply to publicly available data" | **Myth.** That carve-out was in the 2019 Bill. The enacted 2023 Act's s.17 has no such exemption. Beware your own HTTP logs: a retained client IP can make you a Data Fiduciary |
| "Algo orders must file monthly returns / algo-wise turnover to SEBI" | **No such filing exists.** Monthly algo-turnover reporting is an exchange-to-SEBI obligation via the Monthly Development Report. SEBI imposes no periodic filing duty on algo traders |

## Technically impossible as specified

| Claim | Why it cannot be built as written |
|---|---|
| `action = {"nifty_future_hedge": Box(...), "put_strike": Discrete([...]), "put_expiry": Discrete([...])}` with **SAC** | SAC's squashed-Gaussian actor is **continuous-only**. A mixed Box+Discrete space needs separate heads with autoregressive conditioning, or the discrete choice must be removed from the policy entirely. See [deep-hedging.md](methodology/deep-hedging.md) for the three correct options |
| "Use cvxpy layer in policy to enforce sector limit" | A QP projection is **not differentiable**. You cannot backprop through cvxpy, so the policy gradient is silently wrong. Use a projection layer, a Lagrangian, or CPO (Achiam et al. 2017) |
| "Online fine-tune: last 3 months with small LR" | You cannot run online RL against a market — there is no interaction loop that is not market impact plus real capital at risk. This must mean offline replay |
| "50 stock specialists (PPO + LSTM)" | Fifty single-stock models cannot learn cross-sectional structure, share no data efficiently, and then the portfolio layer relearns the same 50 stocks. One cross-sectional policy is strictly less code |
| "Trigger agents when RL entropy > 0.8" | Policy entropy is **uncalibrated** and roughly constant across states. Use ensemble disagreement or critic value standard deviation |
| KEDA `kafka` trigger plus `cron desiredReplicas: "1"` in one ScaledObject | During market hours cron pins it at 1, so the Kafka trigger can never fire. Outside hours there is no consumer, so lag grows unbounded |
| Kill switch as a React button | SEBI: the kill switch "is expected to **automatically** trigger a halt on trading activity based on pre-defined conditions". A UI element is not a compliance control |
| Cross-attention "Contextual Augmenter" | Load-bearing and entirely unspecified. Nothing trains it; the RL reward does not supervise attention weights. This is the claimed "core innovation" and it is a blank |

## Measured to lose money

| Claim | Evidence |
|---|---|
| 7-10 LLM reasoning agents with LangGraph bull/bear debate improve decisions | Stanford, 210 runs: reasoning quality vs Sharpe **r = +0.07, p = 0.29**. Debate lost to equal-weight **33% of the time (p < 0.001)**. The 4-agent, 5-round configuration returned **−0.27%** while achieving the *highest* reasoning scores |
| LLM agents beat buy-and-hold | FINSABER, KDD 2026 Oral: buy-and-hold beat both agents significantly across 63-91 symbols, **p ≤ 0.006**. No agent produced significant alpha (all p > 0.34). The original papers evaluated on **3 stocks over 3 months** |
| `<100ms inference p99` is the right target | Zerodha Kite: **80-150 ms** typical order round trip, 450 ms observed, seconds at open. Ticks are **1-2 s stale**. Your compute is ~2% of the budget. The design's own 15 s agent budget is 150x the inference target |
| "Scale to zero saves ~15k INR/month GPU" | EKS cannot scale to zero (CoreDNS and the autoscaler must stay). **$625/month** of always-on overhead (control plane + MSK + NAT + nodes) to enable a mechanism worth $180. **3.5x net negative** |
| That architecture costs ~$180/month | **~$1,680/month** floor as written; **~$8,500** with 4xA100 nightly. The claim is **9-50x off** |
| Local stack runs on a 16 GB laptop | kind + Kafka + LocalStack + Neo4j + TimescaleDB + Qdrant + WireMock + Chaos Mesh + 11 pods needs **12-20 GB**, with a **15-25 minute** cold boot |
| 50/200 SMA crossover | **Untested in India.** No published study tests it on NSE/Nifty data. Longest horizons ever tested: 60-day SMA, 199-day EMA |
| ADX, OBV, VWAP, MACD, Bollinger, RSI, 52-week-high as signals | **No Indian equity evidence** for any of them. RSI's only serious cross-market test finds it is not a robust best-rule family. MACD worked in 2 of 5 developed markets (Hsu & Ng 2014) |
| "Buy after drawdowns" | India's documented short-horizon effect is **momentum, not reversal**. The reversal evidence is long-horizon and only appears with a 1-year formation-to-holding gap |
| 6 data stores for <2 GB of data | Nifty-50 10 years of 1-min OHLCV is **~150 MB**. Postgres and TimescaleDB are the *same database*. Redis holds a kill-switch boolean. Neo4j traverses a ~300-edge static graph |
| Deep hedging RL beats delta hedging | **Linear regressions on BS Greeks beat BS delta by 15-20% MSHE, and neural networks add zero** (Ruf & Wang, *JBES* 40(4), 2022). The distilled learned policy is a moneyness- and vol-scaled delta haircut with average expression complexity 9.6 — and **the closed-form formula beat the neural agent** |

## Claims that survive

Worth keeping from the original docs:

- **Purge and embargo.** Correct and necessary. Implemented in `src/stock_rl/splits.py`.
- **Deflated Sharpe / multiple-testing awareness.** Correct instinct, wrong formula in our first implementation. Now verified against the published example.
- **Legally licensed data, not scraped NSE.** Correct. NSE ToU clause 9 prohibits automated collection, and redistribution rights are not ours.
- **Model the real cost of Indian transaction costs.** Correct, and the reason three rate bugs were caught early.
- **SEBI algo ID tagging and a decision audit trail.** Correct and required, though retention is 8 years and the kill switch must be automatic.
- **SEBI prohibits algo market orders in equity, and IOC orders in commodity.** Correct (NSE/MSD/67753 s.8.1.1.12; NCDEX).

## One compliance finding the docs missed entirely

**SEBI requires algo servers located in India with no interlink to systems outside India** (2012 circular para 4(iii)). The docs propose calling LLM APIs. If those run in a US region, Indian market data and live trading signals leave the country. Run inference on the same box (Ollama/vLLM) or in an Indian region.