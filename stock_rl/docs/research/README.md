# Research Index

Findings that shaped this codebase. Each file states what was verified,
against which primary source, and what could **not** be verified.

Read these before changing anything the research contradicts. Several
original design assumptions in `../NIFTY-RL-Technical-Doc.md` and
`../PORTFOLIO-HEDGING-RL.md` were found to be fabricated, incorrect, or
measured to lose money. Those are flagged in
[corrections-to-design-docs.md](corrections-to-design-docs.md).

## Compliance

| File | Subject | Headline |
|---|---|---|
| [sebi-algo-trading.md](compliance/sebi-algo-trading.md) | SEBI algo registration, tagging, retention | There is no April 2021 algo circular. Retention is 8 years, not 3. "Static vs dynamic algo" is not a real classification |
| [india-transaction-costs.md](compliance/india-transaction-costs.md) | STT, GST, stamp duty, exchange charges, DP | Delivery round trip is 22.25 bps, 90% of it STT. Found and fixed 3 bugs in our own cost model |
| [scraping-and-data-licensing.md](compliance/scraping-and-data-licensing.md) | robots.txt, NSE ToU, copyright, DPDP | "InvestingNote" and "Indian Express v. Zeal" are fabricated. robots.txt has no legal force in India |

## Methodology

| File | Subject | Headline |
|---|---|---|
| [deflated-sharpe.md](methodology/deflated-sharpe.md) | Bailey & Lopez de Prado DSR, multiple testing | Our shipped DSR was wrong in three places and returned 0.0 for good strategies. Now reproduces the paper's published values |
| [rl-portfolio-allocation.md](methodology/rl-portfolio-allocation.md) | PPO/SAC vs HRP, EW, MVO on Nifty-50 | **HRP beats all four DRL agents on average.** SAC was the *worst* DRL agent and had the worst PBO |
| [deep-hedging.md](methodology/deep-hedging.md) | RL hedging of options/futures | Linear regressions beat BS delta by 15-20%; NNs add zero. The learned policy distills to a delta haircut |
| [signal-attribution-and-audit-trail.md](compliance/signal-attribution-and-audit-trail.md) | Per-decision rationale, SHAP, what India requires | **No Indian rule requires per-decision explanation.** The real driver is algo re-registration on logic change |
| [technical-indicators-nse.md](methodology/technical-indicators-nse.md) | What actually has evidence on NSE | The 50/200 crossover is untested folklore in India. 12/20 EMA is the best cost-adjusted rule. ADX, OBV, VWAP, MACD, Bollinger, RSI have no Indian evidence |
| [fundamentals-nse.md](methodology/fundamentals-nse.md) | Indian fundamental data, point-in-time, factors | Only ~7 quarters of clean PIT fundamentals exist for free. Value has decayed to zero post-GFC. Accruals have the *wrong sign* in India |

## Architecture

| File | Subject | Headline |
|---|---|---|
| [project-structure.md](architecture/project-structure.md) | Monolith vs microservices, real costs | The design's $180/month is $1,680 minimum, $8,500 as written. You cannot scale EKS to zero. 1 VM is correct for 1-3 people |

## Agents

| File | Subject | Headline |
|---|---|---|
| [llm-agents-in-finance.md](agents/llm-agents-in-finance.md) | Multi-agent LLM trading | Buy-and-hold beats LLM agents significantly (p≤0.006). Debate loses to equal-weight 67% of the time. More agents → worse |

## How this research changed the code

| Finding | Code consequence |
|---|---|
| Exchange charge was 0.00297%, actually 0.00307% | `costs.py` corrected |
| Intraday brokerage was 0.0, actually 0.03% capped | `costs.py` corrected |
| DSR denominator used Euler-Mascheroni instead of skew/kurtosis | `metrics.py` rewritten, verified against paper |
| `sqrt(T-1)` counted years not observations | `metrics.py` rewritten |
| Trial-Sharpe dispersion was the strategy's own volatility | `metrics.py` rewritten |
| Technical trading is noise out-of-sample (Rink 2023) | No indicator is trusted without breakeven-cost evidence |
| Undefined is not 0.0 | `deflated_sharpe` returns `None`, not `0.0` |
| 1/N beats most optimized portfolios (DeMiguel 2009) | Baselines, not a trained model, are the benchmark |