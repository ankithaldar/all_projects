# RL for Portfolio Allocation vs. Baselines

**Verdict: do not build a pure RL portfolio allocator as the primary
strategy.** Not one properly-controlled study (multi-seed, realistic
costs, HRP/MVO-min-var/EW baselines, walk-forward) shows RL winning on
average. The strongest Nifty-50-specific paper finds **HRP beats all four
DRL agents on average**.

The only defensible RL claim is narrow: **RL is a drawdown/risk
controller, not an alpha engine.** Build that, only as an overlay on a
robust optimizer.

## Headline: the Nifty-50 study (Sahu & Bhandari, 2026)

*Discover Artificial Intelligence*, DOI 10.1007/s44163-026-01869-x,
open access. 45 Nifty-50 stocks, 2010–2026. PPO/SAC/A2C/DDPG vs
Equal-Weight, Buy&Hold, MVO-MaxSharpe, MVO-MinVariance, HRP,
Black-Litterman. Same data, 0.1% one-way cost.

| Model | Sharpe | Sortino | Calmar | Ann.Ret | Ann.Vol | Avg MaxDD |
|---|---|---|---|---|---|---|
| **HRP** | **1.395** | **2.173** | **2.170** | 27.4% | **13.6%** | **12.6%** |
| MVO-MinVariance | 1.362 | 2.140 | 1.912 | 26.0% | 13.2% | 13.7% |
| DDPG (best DRL) | 1.288 | 1.779 | 1.749 | 27.0% | 15.2% | 16.0% |
| A2C | 1.208 | 1.608 | 1.757 | 25.4% | 15.3% | 15.1% |
| PPO | 1.126 | 1.524 | 1.772 | 23.1% | 14.2% | 13.1% |
| Equal-weight | 1.079 | 1.474 | 1.566 | 22.6% | 14.6% | 14.8% |
| Buy & Hold | 1.065 | 1.415 | 1.527 | 22.4% | 14.5% | 14.8% |
| SAC | 1.062 | 1.467 | 1.549 | 23.3% | 16.3% | 16.6% |
| Black-Litterman | 0.811 | 1.228 | 1.344 | 30.0% | 35.0% | 25.7% |
| MVO-MaxSharpe | 0.774 | 1.119 | 1.173 | 26.8% | 27.2% | 22.9% |

By regime cluster:

- **Cluster 0 (9 most-efficient blue-chips):** DDPG 1.752 > HRP 1.591.
  The *only* DRL agent beating the best classical benchmark, by +0.16
  Sharpe. Mean of all four DRL agents (1.26) is *below* the mean of the
  six classical benchmarks.
- **Cluster 1 (27 firms):** **classical wins.** MVO-MinVar 1.536 and
  Sortino 2.75, beating every DRL agent.
- **Cluster 2 (9 least-efficient):** **classical wins decisively.** HRP
  1.256, max drawdown −12.0%. Best DRL agent 0.895 — *below* the
  benchmark mean of 0.90.

Their own words: *"the primary strength of DRL lies in reducing risk and
drawdowns rather than generating higher returns"* and *"none of the DRL
models are best in every one of the three market efficiency regimes."*
They explicitly disclaim the alpha interpretation, and caveat that
260 OOS days leaves rankings *"statistically inconclusive."*

**Equal weight (1.079) beat SAC (1.062) and Buy & Hold (1.065).**

## Dissecting the most-cited "DRL beats MVO" claim

**Sood, Papasotiriou, Vaiciulis & Balch (2026), arXiv:2602.17098**
(JPMorgan AI Research, FinPlan'23 ICAPS). S&P 500 sectors + cash,
2012–2021, 10 walk-forward backtests, 5 seeds.

| | DRL (PPO) | MVO (Ledoit-Wolf Sharpe-max) |
|---|---|---|
| Annual return | 12.11% | 6.53% |
| Sharpe | **1.166** | 0.678 |
| Max DD | −33.0% | −33.0% |

**What they omit:** no equal-weight, no buy-and-hold. Over exactly
2012–2021 the S&P delivered **~16.6% CAGR at ~15% vol — arithmetic Sharpe
roughly 1.1.** So the DRL Sharpe of 1.166 is **statistically
indistinguishable from holding the index**, at 4.5–5 pp/yr *lower*
return. Their "win" is against a misconfigured MVO (6.5%/yr is terrible,
a sign of unconstrained Sharpe-max blowing up on noisy means), not
against reality. They state in Future Work: *"we would like to model
transaction costs and slippage."*

## DeMiguel (2009) — the reference result you must internalise

*Optimal Versus Naive Diversification*, *Review of Financial Studies*:

> *"**none is consistently better than the 1/N rule** in terms of Sharpe
> ratio, certainty-equivalent return, or turnover, which indicates that,
> out of sample, the gain from optimal diversification is more than
> offset by estimation error."*

And the number that should govern the design: the estimation window needed
for sample-MVO to beat 1/N is **~3,000 months for 25 assets and ~6,000
months for 50 assets** — **250 to 500 years**. You do not have that.

Reinforced by **Michaud (1989)**, "The Markowitz Optimization Enigma":
*"error maximization."* Watch MVO-MaxSharpe in Cluster 2 above: 41.9%
annualised return, 48.2% vol, **−36.8% max drawdown**. That is the
error-maximization signature.

## PBO: the standard walk-forward procedure is itself overfitted

**Boryszski, Cetnarowski et al. (2022/2023), arXiv:2209.05559**, CSCV
PBO over a **2,700-combination** hyperparameter grid:

| Agent | PBO | Verdict at α = 10% |
|---|---|---|
| **SAC** | **21.3%** | **OVERFITTED** |
| **PPO (walk-forward, the standard method)** | **17.5%** | **OVERFITTED** |
| PPO (their method) | 8.0% | passes |
| TD3 | 9.6% | passes |

Two punchlines: the industry-standard walk-forward + PPO procedure carries
~17% PBO, and **SAC — the flagship continuous-control algorithm — had the
worst PBO of the three**.

## Seed variance — the single most useful calibration number

**Grądzki (2026)**, "Unstable Gains: Multiplicity-Aware Evaluation of
Financial Deep Reinforcement Learning", *J. Finance & Data Science* 12,
100205. Fixed architecture/data/hyperparameters, only stochastic training
varied:

- 10k steps: Sharpe **0.604 mean, SD 0.193**, range 0.233–0.855
- 100k steps: **0.512 mean, SD 0.143**, range 0.251–0.749

> *"**selecting the best-performing seed instead of reporting the mean
> inflates the reported Sharpe by 44%**."*

Since virtually every RL portfolio paper reports a selected run,
**headline Sharpes are inflated by roughly 1.3–1.5x** versus production
reality.

**Compare against the effect size:** the RL-vs-HRP gap in the Nifty-50
study was **+0.16 Sharpe in one of three regimes, negative in the other
two.** The seed noise on RL Sharpe is **±0.19**. **The claimed effect is
smaller than the noise in the estimate.**

## Transaction costs invert the rankings

**Riera Abbade (2026), arXiv:2603.29086** (FinRL-Meta extension). Five
DRL algorithms, NASDAQ-100, fixed 10 bps baseline vs. an Almgren-Chriss
square-root impact model:

> *"most open-source backtesting environments assume negligible or fixed
> transaction costs, **causing agents to learn trading behaviors that fail
> under realistic execution**"*

1. *"the cost model materially changes both absolute performance **AND THE
   RELATIVE RANKING OF ALGORITHMS** across all three environments"*
2. AC model: daily costs $200k → $8k, turnover 19% → 1%
3. Hyperparameter tuning *"essential for constraining pathological
   trading, with costs dropping up to 82%"*
4. DDPG OOS Sharpe jumps −2.1 → 0.3 under AC; SAC's drops −0.5 → −1.2

**Point 1 is decisive: the RL leaderboard is an artifact of the cost
model you chose, not a property of the algorithms.**

Your Indian stack is **~22 bps round trip**. At 15% vol that is:

| Rebalance frequency | One-way turnover | Annual drag |
|---|---|---|
| Monthly | 20% | ~0.44% |
| Monthly | 50% | ~1.1% |
| **Daily** | **100% monthly** | **~2.7%** |

A Sharpe of 1.15 at 15% vol generates ~17%/yr. Give back 2.7% and you
lose ~16% of return and ~0.18 Sharpe units. **That single realistic
assumption erases most of the published RL "alpha."**

FinRL's own 2024 industry critique (Ndikum & Ndikum, arXiv:2403.07916):
*"transaction costs... play a crucial role. The incorporation of such
constraints can dramatically alter the performance of back-tested
simulations."*

## SAC specifically has a named failure mode in markets

**"Financial Entropy Trap" (arXiv:2606.10448, 2026):**

> *"The financial market is a typical **low signal-to-noise ratio**
> setting, which often **destabilizes off-policy maximum-entropy methods
> like Soft Actor-Critic (SAC)**. Specifically, noisy state
> representations may produce unreliable Q-value estimates, and
> **bootstrapping amplifies these errors**"*

SAC's entropy bonus + twin critics + off-policy bootstrapping are fine in
high-SNR control (MuJoCo) and pathological in low-SNR markets where
Q-values are mostly noise. Corroborated by the Nifty-50 study (SAC worst
DRL agent, Sharpe 1.062), the PBO study (SAC worst PBO, 21.3%), and the
MACE paper (SAC OOS Sharpe deteriorating −0.5 → −1.2 under realistic
costs).

Empirically the ranking in the Nifty-50 study is **A2C > PPO > DDPG >>
SAC**.

## Simplex projection: the three options and their real status

**(a) Softmax on logits** — what essentially every RL portfolio paper
does. Differentiable, but:
- logits are identified only up to a constant shift, so the softmax
  Jacobian has a rank-1 null direction
- it induces an **entropy prior**, biasing toward under-diversified
  allocations unless the logit scale is tuned. On 50 names a
  randomly-initialised net emits near-uniform weights, so the gradient
  signal distinguishing skill from architectural prior is weak
- no leverage or weight bounds for free
- convergence of softmax-parameterised policy gradient under simplex
  constraints is **still not fully understood even with exact policy
  evaluation** (arXiv:2404.03372)

**(b) Hard Euclidean projection** (`x = max(v-θ,0); x /= sum(x)`) — a
piecewise-linear map whose Jacobian is identity a.e. on the interior and
zero on the boundary. Biased, high-variance gradient; no
constraint-satisfaction guarantee on the *pre*-projection action. If a
paper reports this without straight-through or Gumbel-Softmax, be
suspicious.

**(c) Differentiable convex optimisation layers** — the principled option:

| Layer | Idea | Note |
|---|---|---|
| **OptNet** (Amos & Kolter, 2017) | differentiate through a convex program via its dual | breaks with non-unique solutions |
| **cvxpylayers** (Agrawal et al., 2019) | backward pass of a disciplined convex program's cone program | production-ready, but fails on degenerate active sets |
| Ambrogioni et al. (2019) | abstract linear operator | needed when multiple active constraints — **exactly the degenerate case in portfolios** |
| **DFWLayer** (arXiv:2308.10806, 2023) | Frank-Wolfe, *"without projections"* and no Hessian | **most directly relevant for simplex constraints** |

**Note:** there is essentially **no finance literature using these inside an
RL policy.** The field is still on softmax. That itself is evidence about
maturity.

**This is why the design's "cvxpy layer in policy" is wrong:** a raw cvxpy
solve is not differentiable. Use `cvxpylayers`/`DFWLayer` if the
constraint must be in the gradient, or keep hard feasibility out of the
policy entirely.

## Is CPO recommended? Honestly: no

**CPO = Constrained Policy Optimization** (Achiam, Held, Tamar &
Abbeel, ICML 2017, arXiv:1705.10528): *"the first general-purpose policy
search algorithm for constrained reinforcement learning with guarantees
for near-constraint satisfaction at each iteration."*

But not for portfolios:

1. **Essentially unused in finance.** No paper applies CPO to portfolio
   allocation. Its demonstrated domain is MuJoCo locomotion with physical
   safety constraints.
2. **Its requirements are awkward.** It needs a differentiable estimate of
   expected cost and a cost value function, and historical replay gives
   you the first only through the same backtest you are trying not to
   overfit — and not at all for market impact. **You get a safety
   guarantee on a simulator.**
3. In the safe-RL benchmark literature, penalty/Lagrangian variants
   (TRPO-Penalty, PPO-Lagrangian) have generally **matched or beaten
   CPO**. (Directionally reliable; specific numbers not re-verified.)

**Recommendation:** put hard feasibility **outside** the policy —
project onto the simplex as post-processing in the environment so the
executed trade is always feasible, and treat the pre-projection action as
unconstrained. Model soft constraints in the reward. Constrain **turnover**
with a hard band in the environment, because that is the constraint the
22 bps stack makes non-negotiable.

## What to build instead

**A constrained robust optimizer with volatility targeting and a momentum
overlay. No RL in the allocation path.** If you want ML, spend it on
covariance *estimation*, not on the decision.

1. **Core allocator: HRP on shrunk covariance.** It won every metric in
   the Nifty-50 study and was the most consistent across regimes. Its
   advantage is structural: quasi-diagonalization + recursive bisection
   never invert a noisy covariance matrix and never produce the extreme
   error-maximizing weights that kill MVO-MaxSharpe (López de Prado 2016,
   *JPM* 42(4):59–69). Use **Ledoit-Wolf shrinkage** — raw sample
   covariance is a non-starter. `skfolio` (scikit-learn native, ships
   Palomar's CUP 2025 textbook) or Riskfolio-Lib.
2. **Guardrail baseline: equal weight with volatility scaling.** DeMiguel's
   1/N is the benchmark you must clear. Inverse-vol on EW is five lines
   and historically close to MVO-min-var without the estimation error.
   **This must be the control arm and the kill criterion.**
3. **Only justified active overlay: vol targeting + slow trend signal,
   with a hard turnover band.** This is what the successful "RL wins" are
   doing once you strip the costume — Benhamou's RL agent collapsed onto
   a single hedge (tactical timing); Sahu & Bhandari attribute the DRL
   edge to *"dynamic allocation effects: the ability to adjust portfolio
   weights in response to volatility signals"* — explicitly not alpha.
   Scale to a vol target with a slow estimator and a k-of-n rule; optional
   12-month momentum on the equity/cash split only, monthly, with a
   2-month confirmation lag.
4. **Costs: real 22 bps round trip plus square-root impact** for anything
   above ~Rs 2–5 crore per rebalance. At that size impact dominates STT.
5. **Validation:** walk-forward with a burn-in year, refit monthly; **≥30
   seeds** with mean *and* dispersion reported; Deflated Sharpe and PSR vs
   the EW-vol baseline with declared trial count; minimum-backtest-length
   check; **point-in-time Nifty-50 membership** — 45 current names is a
   survivorship-biased universe.

## If you want RL in the product

Put it in the **highest-SNR, most sequential sub-problem: execution and
rebalancing timing**, not asset weights. Inventory affects future fills;
impact is path-dependent; mistakes do not compound across a 15-year
holding period. Measure it by **implementation shortfall vs. a
predeclared arrival price**, not portfolio return.

## What would change this conclusion

A paper with ≥30 seeds, realistic 22 bps + impact costs, walk-forward with
burn-in, evaluated against **HRP + vol-scaled EW + MVO-min-var** on
**Nifty-50 with point-in-time membership**, reporting mean Sharpe with
dispersion and a DSR. Or a ≥2-year live track record with a pre-registered
spec. **Nothing like this exists today.**

The honest summary: **the algorithms are fine; the problem is that the
evaluation protocol reliably manufactures an alpha that isn't there.**

## Caveats

`arXiv:2512.10913` (systematic review of 167 studies) reports a
"plateau" — but its meta-analysis runs on a **synthetic dataset**, so its
quantitative claims are unreliable; only the qualitative direction is
corroborated elsewhere. **Mahayana et al. (2022)** found PPO on BTC/USDT
minute data **fails to beat buy-and-hold**. **Durall (2022)**: *"none of
them can even beat the equal weight strategy when having poor runs"* and
DRL models are *"unstable, i.e., weight initialization-dependent."*