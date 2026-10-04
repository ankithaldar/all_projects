# Deep Hedging with RL — Evidence Review

The deep hedging literature has a large credible-in-the-abstract body of
work showing RL hedges beat delta hedging *in simulation*, and a smaller
body showing the gain largely evaporates or **inverts** when you
(a) benchmark against anything better than plain delta, (b) use an
asymmetric reward, (c) evaluate on ordinary variance, or (d) leave the
training regime.

**Distilled economic content of the learned policy: a moneyness- and
vol-scaled delta haircut — one line, an afternoon to implement.**

**Bibliography correction:** the *Quantitative Finance* foundation is
**Buehler, Gonon, Teichmann & Wood, "Deep hedging," Quant. Finance
19(8):1271–1291 (2019/2020)** — JPMorgan. Jelnek & Venzke (2019,
*Journal of Derivatives*) is a different, smaller, supervised-learning-
adjacent paper on smile/skew-adjusted hedging. Not the foundation.

## Baseline claims and their magnitude

Standard claim: RL reduces hedging-error risk (MSE / semi-MSE / CVaR)
relative to delta and delta-gamma hedging, under costs and discrete
rebalancing.

Strongest recent pro-RL configuration (François, Gauthier, Godin & Pérez
Mendoza, arXiv:2504.06208, 2025), ATM straddle, T=63 d, 100k OOS paths,
**zero transaction costs**:

- Adding a **second hedging instrument** (longer-dated ATM call) helps
  **every** strategy including classical ones — delta-gamma cuts
  terminal-error std by **≥42%** vs the best single-instrument strategy.
  **Most of the headline "deep hedging beats delta" gain is instrument
  selection, not learning.**
- RL with two instruments beats delta-gamma by ≥60% on std, ≥92% on
  CVaR95%, ≥73% on semi-MSE.
- With costs (κ₁=0.05% underlying, κ₂∈{0.5,1,1.5,2}% option — realistic;
  Chaudhury 2019 reports ~0.95% average index call cost), RL achieves
  lower risk at similar or lower cost via **learned no-trade regions**.

Mikkilä & Kanniainen (*Quant. Finance* 23(1):111–122, 2023) is the
canonical empirical paper (real SPX options, OptionsDX): a DRL agent
trained on a calibrated stochastic-vol simulator beats the BS delta hedge
on both cost and risk. **Zernikov (arXiv:2605.21696, 2026)** replicates
walk-forward 2015–2023 with log downside-variance ratios vs daily-updated
BS of **−0.16 to −0.80**, negative and mostly significant in 7 of 9 years.

A −0.4 log ratio is roughly a **33% reduction in downside variance**. Real
but not transformative.

## Critical evaluation — four credible attacks

### (a) The gain is a speculative overlay, not a hedge

**Horikawa & Nakagawa** (*Finance Research Letters* 105101, 2024) prove
that in a complete market admitting statistical arbitrage, the difference
between the deep hedging position and the delta position **is** a
statistical arbitrage — zero-investment, positive mean, negative risk
measure. The agent is taking a levered directional bet that the risk
measure fails to price.

**François et al.** (*FRL* 73:106590, 2025) extend this to GARCH dynamics.
ATM call, T=63 d, 100k paths, initial price 3.16:

| CVaR α | ρ(ξ_DH) − ρ(ξ_Δ) | E[V_T^δ⁻(0)] | Spearman ρ(DH, Δ) | R² |
|---|---|---|---|---|
| 1% | −1.368 | **+1.372** | −0.270 | **0.003** |
| 10% | −1.325 | **+1.371** | −0.272 | 0.003 |
| 20% | −1.256 | **+1.371** | −0.273 | 0.003 |
| 85% | −0.129 | −0.039 | 0.939 | 0.773 |
| 95% | −0.121 | −0.369 | 0.969 | 0.808 |

At **α ≤ 20%** the difference strategy earns **+1.37 unconditionally** —
roughly **43% of the option's initial price, every path** — with R² 0.003
against delta. **The agent has abandoned hedging.** At α ≥ 85% it is a
genuine delta modification (R² 0.77–0.82).

**The risk measure is the whole ballgame. A soft CVaR at low α, or any
asymmetric reward, does not produce a hedge.**

### (b) Neural networks don't beat regressions

**Ruf & Wang**, "Hedging With Linear Regressions and Neural Networks,"
*JBES* 40(4):1442–1454 (2022). Real S&P 500 EOD and Euro Stoxx 50 tick
data, rolling 180-day windows, strict chronological splits:

> *"**In none of the datasets do ANNs outperform the linear regression
> models.** We conclude that the option sensitivities suffice to capture
> the nonlinearities in the data that are relevant for the hedging task."*

- Linear regressions on BS Delta/Vega/Vanna/Gamma improve MSHE over BS
  delta by **15–20%** on real data.
- Calls hedged at **0.9·δ_BS, puts at 1.1·δ_BS** — no historical data
  required — captures **−12.6% to −19.0%** relative MSHE reduction.
- Mechanism: the leverage effect (negative spot–IV correlation), i.e. the
  Hull & White (2017) minimum-variance delta correction.
- **Net Sharpe effect of the 15–20% MSHE improvement: ×1.10 to ×1.11.**

This is the closest thing to a replication study of the whole premise, and
it is a clean negative for anything more complex than a 3-coefficient
regression.

### (c) Only one of eight algorithms beats the benchmark

**Neagu, Godin & Kosseim** (arXiv:2504.05521, 2025) compared MCPG, PPO,
four DQL variants, two DDPG variants on GJR-GARCH(1,1):

> *"**MCPG is the only algorithm to outperform the Black–Scholes delta
> hedge baseline with the allotted computational budget.**"*

DDPG — the canonical deep hedging algorithm — did not clear the bar.

### (d) Honest walk-forward work can't claim dominance

**Lucius et al.** (arXiv:2512.12420, Dec 2025), leak-free environment with
cost-aware rewards and position limits:

> *"the learned policy improves risk-adjusted performance versus no-hedge,
> momentum, and volatility-targeting baselines (higher **point-estimate**
> Sharpe); only the GAE policy's test-sample Sharpe is statistically
> distinguishable from zero, although confidence intervals overlap with a
> long-SPY benchmark **so we stop short of claiming formal dominance**."*

**Hu, Chen, Yi & Sun** (arXiv:2603.06587, Feb 2026), SPY + XOP: RL wins on
**shortfall frequency** (best in 6/8 slices) and **transaction cost**
(consistently lowest turnover), but on **ES5% the winner is BS, JD or SV in
5 of 8 slices**. Rankings depend on whether you look at the middle or the
tail. Conclusion: friction-aware RL buys **cheaper and less frequent
trading**, not systematically better replication.

**Assembling:** the reproducible edge is **transaction-cost management and
turnover reduction**, plus a **vol-adjusted delta haircut** which is
closed-form. It is not a qualitatively new hedging capability.

## Volatility surface: what is actually needed

| Paper | Data | State inputs |
|---|---|---|
| Mikkilä & Kanniainen 2023 | OptionsDX SPX, mid-quotes | 4: forward moneyness, τ, inventory h, implied vol |
| François et al. 2025 | **OptionMetrics (commercial)**, 1996–2020 | **14** inputs incl. 5 IV surface factor coefficients |
| Zernikov 2026 | OptionsDX SPX | same 4 as Mikkilä + moneyness |
| Lucius et al. 2025 | SPX/SPY IV term structure, skew, realized vol, rates | term structure + skew + vol + rate context |

**Nobody in this literature uses put-call ratio or open interest as a
state variable.** OI appears only in old ANN-pricing literature
(Ghaziri 2000; Liang 2009) as a weak feature.

> **Design correction: drop PCR/OI from the hedging state.** It adds state
> dimension, look-ahead/measurement noise, and no cited paper shows it
> earning its keep. OI is also a poor real-time proxy: published once per
> settlement cycle, contaminated by the previous expiry's expiry-day
> activity, and in Nifty dominated by a handful of institutional strikes.

**Minimum viable data set, backed by Ruf & Wang — four numbers:**
1. Underlying level
2. BS delta of the liability at current IV
3. Time to expiry
4. Current hedge inventory

Greeks (vega/vanna/gamma) are cheap derivatives of the same IV and add
~5–10%. **Nothing supports needing a fitted 2D surface.**

**NSE reality:** NSE sells real-time feeds and markets "Analytical
Products" for Greeks, noting participants "require maintaining robust and
expensive infrastructure." Vendors resell chains with IV/Greeks/OI,
generally L1 for liquid strikes. Historical full-chain snapshots with OI
to meaningful depth are commercial and expensive. **You can get a reliable
ATM/near-money surface; deep wings and OI history are the expensive part —
and by Ruf & Wang you do not need them.**

### Two methodological traps from Ruf & Wang §6

- **Never drop "wrong-way" observations** where the index rises and the
  option falls. That is a normal consequence of bid-ask spread and the
  leverage effect, not a data error. **Removing them leaks information.**
- **Dropping missing quotes is biased**, because missingness is caused by
  illiquidity, and liquidity correlates with the vol surface being
  modelled. Their robustness check: removing zero-volume end-of-period
  samples cut the dataset 22% and **inflated put MSHE by >10%**.
  **Model missingness, don't filter on it.**

## Instrument choice for a Nifty-50 equity book

**Honest disclosure: the Indian-specific published evidence is weak.** No
credible study of RL-based Nifty hedging exists.

**Nifty futures are the default and should be the benchmark.** Min-variance
optimal hedge ratios for Nifty spot vs futures cluster around **0.7–1.0**;
one 2022–2024 study reports MVHR 0.87 cutting portfolio variance **63.4%**
for **12.8%** annualised return, most effective in bear markets. Futures
give no premium outlay, no IV exposure, no theta, deep liquidity, tight
spreads, exact sizing.

**Protective puts buy convexity, not variance reduction.** The consistent
finding across NSE and international evidence: covered calls deliver higher
returns and protective puts lower risk, and **protective puts frequently
*increase* the probability of monthly losses** while lowering magnitude
(Foltice 2022; *Risks* 14(6):126). That is exactly the asymmetry you do
*not* want if the objective is drawdown control — a tail hedge widens your
left tail less than it truncates your right tail.

**The genuine argument for options:** a futures hedge has **no convexity
floor** — a 40% Nifty drawdown becomes a 40% loss on your short notional,
and you eat it fully. A put spread (long 1-month ~5% OTM put, short ~3%
OTM, re-rolled quarterly) caps the loss beyond strike at roughly the
width, costing ~1.5–3% of notional per quarter depending on IV. A **collar**
is roughly cost-neutral but gives away upside — and if your edge is beta,
selling the upside is the expensive part.

**Recommendation:** futures as the workhorse (continuous/partial short),
with a **discrete, rule-based, VIX-conditional put-spread overlay**. Maintain
~0.5–0.7 short-futures-equivalent net; when India VIX is in its top decile
*and has been for >5 days*, add a 1–2 quarter put spread to ~95–98%
coverage on a notional slice. Defensible, cheap, auditable, needs no RL.

## Objective design: minimizing drawdown is actively dangerous

The failure mode is called the **doubling strategy**. An asymmetric reward
(penalises losses but not gains) lets the agent profit by *increasing*
exposure after losses, because the expected "recovery" gain from a bigger
position can exceed the incremental penalty. This is exactly a martingale.

François et al. had to add penalties *"to ensure that the RL agent learns a
true hedging strategy rather than engaging in speculative behavior"*,
warning about *"doubling strategies, where agents **continuously increase
their exposure in an attempt to recover successive losses**. Such
strategies are undesirable as they deviate from sound risk management
principles."*

**Why a drawdown penalty is specifically dangerous — three problems:**

1. **Drawdown is non-time-separable.** A functional of the whole path, not
   additive over steps. Using it as a per-step reward requires a
   running-maximum state variable (partially observed, growing) or a proxy.
   **Every paper uses a proxy. And the proxies are asymmetric → the doubling
   pathology.**
2. **It under-penalizes gains asymmetrically** — precisely the CVaR-α≤20%
   condition above. Zernikov's default reward `r = 10(0.03 + PnL − κ|PnL|^α)`
   with κ=α=1 reduces to `r = 0.3 − 20·PnL⁻`: **completely flat for
   non-negative daily P&L.** The agent learned a **systematic underhedge**
   (mean delta 0.44–0.68 vs BS 0.49–0.70, underhedged in 57–96% of
   intervals in every year).
3. **It rewards "never being down" over "recovering well."** Under a
   drawdown-only reward the agent has no absolute capital-preservation
   incentive, so a short-gamma premium-selling policy looks excellent until
   it doesn't.

**Empirical confirmation:** in 2022 (S&P −19.9%), Zernikov's agent lost
**1.905 accumulated reward units vs BS at p<5%** while neither variance
metric was significantly worse. Decomposition: index-down/call-down
intervals (54.7% of all intervals) contributed **−5.47**; everything else
**+3.56**. On those days the mean spot component of call revaluation was
−0.520 while the mean IV component was only **+0.014** — **IV increases
offset just 7.9% of the negative spot revaluation, the lowest in the
sample.**

### What to use instead, ranked

1. **Coherent terminal risk measures on the hedging error**: CVaR at
   **high α (85–95%)**, semi-RMSE (semideviation only — Carbonneau & Godin
   2023 recommend it as the fix that *"does not provide any reward for
   gains"*), or MSE. **If you use CVaR, use α ≥ 85% or you are not
   hedging.**
2. **If you must use a drawdown notion, make it a coherent expectation**:
   expected maximum drawdown **E(MDD)** (ESWA 2018) or **CDaR**
   (Kearns et al., clean LP formulation). Use as **constraints**, not as
   the per-step reward.
3. **Asymmetric proxies are OK only with an anti-speculation constraint** —
   Zernikov's `+λ·Pr(max_t ξ_t > V₀)` Lagrangian term, λ tuned on
   validation.
4. **Never optimise "return subject to drawdown ≤ D" as a per-step
   reward.** That is the textbook formulation of the pathology.

**Report ordinary variance and CVaR as separate primary metrics even if you
don't train on them.** Zernikov's most useful methodological contribution:
accumulated reward, downside variance, ordinary variance and CVaR **rank the
same hedge differently**, and a paper reporting only the trained objective
is hiding the failure.

## Regime issues — the counterintuitive finding

Zernikov's walk-forward 2015–2023, log agent-to-BS variance ratios:

| Year | Reward Δ | CVaR5% | Log downside var | Log ordinary var |
|---|---|---|---|---|
| 2015 | 1.946*** | 0.236 | **−0.802*** | −0.177 |
| 2017 | 0.409** | 0.037 | −0.162** | **0.187** |
| 2019 | 1.205** | 0.102 | −0.557*** | −0.186 |
| **2020** | −0.498 | 0.045 | **−0.677*** | −0.009 |
| 2021 | 1.674*** | 0.082 | −0.451*** | 0.142 |
| **2022** | **−1.905*** | 0.047 | −0.062 | −0.117 |
| 2023 | 0.095 | 0.057 | −0.168** | **0.527**** |

**Three regimes, three failure modes:**

- **2020 (violent crash, IV spike):** downside variance improves
  *significantly*. **Deep hedging worked here.** The vol channel cushions
  the option leg exactly as the vol-scaling correction intends.
- **2022 (grinding bear, weak vol cushion):** the real failure. Reward
  −1.905 at p<5%, in **every** random seed. A *timing* failure under the
  asymmetric objective.
- **2017 and 2023 (low-vol):** ordinary variance **fails**. Mechanism: BS
  becomes an extraordinarily tight variance hedge when option P&L is
  spot-dominated. In 2023 the spot-only regression explained **85.8%** of
  option-price changes and BS left **0.049 of residual variance out of 5.713
  option + 5.254 hedge gross variance — only 0.45% uncancelled.** Nothing
  left to remove; the agent's residual long spot adds dispersion.

**Long-horizon regime-transfer stress test** (frozen policies, 45
policy×year pairs): downside variance favorable in **36/45**, significantly
favorable in 13, **zero significantly unfavorable**. Ordinary variance
favorable in only **10/45**, **significantly unfavorable in 17/45**.

> **The downside delta correction generalises across regimes. Ordinary
> variance does not, and will actively hurt you.** If your mandate is
> drawdown, the asymmetry is defensible; if your risk report shows total
> variance, you will get flagged.

**Counter-evidence:** Lütkebohmert, Schmidt & Sester, "Robust deep hedging"
(*Quant. Finance* 22(8):1465–1480, 2022) shows robust deep hedging beats
standard approaches *"in particular in highly volatile periods"*, including
COVID-19 data. But it addresses *simulator parameter* uncertainty, not
trained-policy out-of-sample regime transfer. Different problem.

**Backtest specifically on 2015–2016, 2022, and 2023-like episodes** — not
just 2020.

## Action space: SAC cannot do this

**Correct.** SAC's squashed-Gaussian actor is unimodal and continuous. It
cannot represent a discrete choice over strikes and expiries.

Moreover the literature increasingly argues **unimodality itself is wrong
for hedging**: *"Deep Diffusion Reinforcement Learning for Options
Hedging"* (*Applied Intelligence*, Springer, 2026) opens with *"optimal
hedging strategies may exhibit multimodal structures under non-convex
risk landscapes, whereas existing RL approaches typically rely on
unimodal Gaussian or deterministic policies"*, and notes terminal CVaR
adds further non-convexity. **A Gaussian head can only average across
modes, which in hedging means "half-short, half-long" — a nonsensical
hedge.**

### What the papers actually do: they dodge it

**The dominant approach: never select strike or expiry in the policy.** The
instrument set is fixed exogenously; the action is a continuous position
vector.

- Mikkilä & Kanniainen: single instrument, 1-D continuous
- François et al. 2025: risk-free + underlying + **one fixed** ATM call at
  T*=84d. Action = 2-D continuous `(shares, options)`. Strike and expiry
  **hardcoded**
- Murray, Wood, Buehler, Wiese & Pakkanen (ICAIF 2022): continuous actions
  with a **convex admissible action set** induced by a convex cost function
- Zernikov: underlying only; Greeks **deliberately excluded** from the state
  so the policy learns its own delta correction

Where discrete actions appear: **DQN** variants discretize position size
into a grid (Neagu et al.); **distributional RL** over a discretized action
set (Cao et al. 2023); a CityU thesis uses **MCPG on discrete actions**.
Zernikov uses `clip(·, 0, 1)` — a *projection*, not a selector.
**Hierarchical RL is essentially absent from deep hedging**, and the one
monthly-cadence Meta-Controller paper is for *portfolio* drawdown, not
option hedging.

### Three correct options

**Option A (strongly preferred): reparameterise into continuous
instrument definitions.** Replace "pick strike, expiry, size" with "pick a
**target coverage ratio** for each of 2–4 **predefined structures**". For
Nifty: `f` short futures (continuous); `s` 1-month 95%-strike put spread
(fixed ladder, re-rolled on a schedule); `c` upside financing (only if you
run a collar). Action = 3 continuous Box variables. **Pure SAC-compatible.**
Strike and expiry selection becomes a **contract design decision made by
you**, revisited quarterly — which is how it is done in practice. Also
removes the exploding action space: Nifty has ~10 expiries × ~40 strikes
= 400 discrete choices per step.

**Option B: autoregressive factorisation with a discrete prefix.**
`π(a|s) = π₁(a_strike, a_expiry | s) · π₂(n | s, a_strike, a_expiry)` —
discrete categorical heads (Gumbel-Softmax straight-through, or SAC-Discrete
per-head log-prob) first, then a squashed Gaussian continuous head
**conditioned on the sampled discrete output**, not just its distribution.
Discrete-then-continuous is correct since size is state-dependent on what
you picked.

**Option C: hard hierarchical two-stage.** High-level (monthly) chooses a
structure; low-level (daily) sizes within it. **Statistically hopeless to
learn properly** — 10 years of Nifty is ~120 monthly decisions. This is why
the literature doesn't do it.

## Constraints: projection first

Roughly in order of frequency:

1. **Bounded / state-dependent output layer (projection)** — most common,
   least fragile.
   - **No-Transaction Band Network** (Imaki et al. 2021) — clamp built into
     the *architecture*
   - François et al. 2025: `f(Z,t) = min(Z, (V_t + B)/S_t)` — a **dynamic
     upper bound on the final output layer**. Cleanest implementation of a
     bounded hedge ratio
   - Murray et al.: the formal version — a convex cost `c(s,a)` with
     `c(s,0)=0`, non-negative, convex, defines a **convex admissible action
     set**. With `c(s_t,a_t) = α|a_t||h_t|` this is just the box from
     proportional transaction costs. **The transaction cost *is* the
     constraint.**
2. **Lagrangian soft penalty** — François et al.'s
   `+λ·Pr(max_t ξ_t > V₀)`. Demonstrably kills doubling when tuned.
3. **Reward shaping** — `r = 10(0.03 + PnL − κ|PnL|^α)`. Cheap but soft; a
   big enough reward buys constraint violation.
4. **Learned no-trade region (hard threshold)** — François et al.:
   rebalance only if the deviation exceeds `l`, **learned jointly with the
   network** by MSGD. **The cleanest cost-constraint mechanism in the
   literature**, and learnable because it is scalar.
5. **CPO / PCPO: essentially absent from the hedging literature.** Its
   practical reason: constraints here are almost always simple boxes, and
   box projection is exact, cheap and gradient-friendly, whereas CPO
   requires inverting a Fisher Information Matrix and its approximation
   errors force recovery steps. **Use projection first.**

### For the specific constraints

**"Cannot buy puts when IV > 30%" → mask, not penalty.** It is a **hard
business rule**, and a Lagrangian lets the agent trade it off against
reward. Implement as a **state-dependent admissible action set** — exactly
Murray et al.'s `A(s)` framework:

- Mask disallowed instruments in any categorical head (logits to −∞ before
  softmax)
- For the continuous target-coverage form: clamp
  `coverage_upper(σ) = min(A_max, k/σ)` — cheap when IV is rich
- **Apply the mask identically in the critic target computation.** A
  discontinuity in the actor's policy map makes the critic's Q-values for
  masked actions wrong — a real source of training instability.
- If masking destabilises training: keep the mask but **normalise the
  reward by the feasible frontier** (compute a reference feasible policy's
  reward and subtract), putting the constraint in the baseline rather than
  the gradient.

**"Bounded hedge ratio" → tanh output × capacity × state-dependent cap.**
`n_t = N_max·tanh(g_θ(s_t))·m(s_t)`. **And bound the turnover too:**
`|n_t − n_{t−1}|` and `|n_t| ≤ κ·max_t|portfolio value|`. Zernikov's
smoothing analysis shows tiny policy perturbations (average delta MAE 0.006)
change realized reward materially — the action path is **not robust**, so a
rate limit is required.

## Is RL worth it for a small Nifty-50 book?

**No, not now.**

**The ceiling on the gain.** Ruf & Wang: linear regressions on BS Greeks beat
BS delta by 15–20% MSHE; neural networks add **zero**. Rule of thumb:
**0.9·δ_BS for calls, 1.1·δ_BS for puts.**

**What RL actually learns, in closed form.** Zernikov distilled the TD3
policies via symbolic regression:

```
2016: clip(Δ_BS + σ(e^m − 2.949), 0, 1) ≈ 2.718·σ(m − 1.085)
2017: m − 1.045
2019: m − 1.063
2020, 2023: add a σ or √σ scale
```

Average complexity **9.6**. Mean correction **−0.055**, below BS delta in
**94.2%** of states. Correlations of the correction: **+0.565 with
moneyness, −0.346 with IV, −0.062 with maturity** — maturity barely matters.
It is a **moneyness-centred delta haircut, stronger in high-vol regions**.

**And the distilled formula beat the neural agent**: reward +0.604 vs the
agent; downside variance −0.098; formula vs BS: CVaR favorable in **8 of 9
years, significant in 6**. Adding complexity bought nothing — best-
validation-fit selection raised complexity 9.6 → 16.6 while test MAE
improved only 0.021 → 0.020.

The mechanism is the textbook minimum-variance delta:

```
h^MV = Δ^BS + ν · Cov(dσ, dS) / Var(dS)
```

For equity-index options spot and IV are negatively correlated, so with
positive call vega the minimum-variance hedge sits **below** practitioner
delta. **That is Hull & White (2017), and it is closed-form.**

### What to actually build

**Tier 0 (weeks, not months):**
1. Estimate portfolio **beta to Nifty** with a rolling 60-day regression.
   That single number is the hedge ratio.
2. Base hedge: short Nifty futures at a **fixed fraction of beta** (start
   ~0.5–0.7). Re-estimate monthly.
3. **Vol-scaled delta adjustment**:
   `hedge = Δ_target · (1 − λ·max(0, IV_rank − 0.8))`, λ ≈ 0.2–0.3. This
   is the distilled correction. Costs nothing, is auditable, and every
   rupee of P&L is attributable to it.
4. **VIX-conditional tail overlay**: 1–2 quarter put spread, fixed
   schedule, 95–98% delta coverage, activated only when India VIX is
   persistently top-decile. Rule-based, no learning.
5. **Turnover limit** in the execution layer.

**Tier 1 (the honest upgrade):** a **Hull-White / Delta-Vega-Vanna
regression** on your own Nifty data, walk-forward, delta coefficient
freely estimated. Three coefficients, refit monthly, fully explainable.
Per Ruf & Wang this captures the entire measured gain.

**Tier 2 (only with a reason):** RL — restricted to **sizing within a
fixed instrument set** (Option A), trained on **high-α CVaR or semi-RMSE**,
with the λ·P(max_t ξ_t > V₀) Lagrangian from the start, hard rules masked
not penalised, and **2022- and 2023-like periods held out as mandatory
validation**. Treat the RL result as a genuine finding only if it beats
Tier 0/1 on **ordinary variance and CVaR in a 2022-like regime**, not just
on the trained objective.

**Design against the Tier-0/1 baseline, not against Black-Scholes delta.**
If you benchmark only against delta you will conclude RL is transformative
when a 10% delta haircut is most of it.

## State of the art, 2025–2026

The frontier has moved **away from transformers** for hedging, toward
(a) richer state representations — full joint IV surface dynamics — and
(b) better policy classes for multimodality.

| Paper | Contribution |
|---|---|
| **Deep Diffusion RL for Options Hedging**, *Appl. Intell.* 2026, DOI 10.1007/s10614-026-11415-7 | Identifies that unimodal actors are structurally wrong for non-convex hedging. **Read this first for the policy class** |
| **Reliable Itô Signatures**, arXiv:2608.18120 (Jul 2026) | Non-RL, model-free, interpretable. Theoretically bounded hedging error, strong sample efficiency, substantially lower compute than NN. **A serious competitor to RL hedging** |
| **Model-Free Deep Hedging with Transaction Costs**, arXiv:2505.22836 | **256 trajectories suffice** to beat both BS and Leland in GBM. Kills the 10⁵–10⁶ path objection |
| **François et al.**, arXiv:2504.06208 | Best-engineered pro-RL: JIVR simulator, RNN-FNN, learned no-trade regions, soft anti-speculation constraint, 1996–2020 |
| **Lucius et al.**, arXiv:2512.12420 | Best reproducible template, with honest non-claiming |
| **Zernikov**, arXiv:2605.21696 | **The single most valuable paper for your evaluation.** Walk-forward, two-stage bootstrap, three seeds, frozen-policy stress test, symbolic distillation |

**On LLMs: essentially nothing credible.** Adjacent work is peripheral —
LLMs as *detectors* of dealer gamma-exposure structure (IEEE Big Data 2025,
71.5% unbiased detection on 242 days) and sentiment into hedging
decisions. Notably, detection accuracy stayed flat while economic
profitability varied quarterly — **the LLM identified the structural
constraint, not a profitable signal.** A useful prior for LLM-in-trading
generally.

## Priority reading list

| If you want | Read |
|---|---|
| The critical case, in one paper | Zernikov, arXiv:2605.21696 (2026) |
| The "it's just a regression" case | Ruf & Wang, *JBES* 40(4) (2022) |
| The objective-pathology case | François et al., *FRL* 73:106590 (2025) |
| Its antecedent | Horikawa & Nakagawa, *FRL* 105101 (2024) |
| Best-engineered pro-RL | François et al., arXiv:2504.06208 |
| The policy-class problem | DDRL for Options Hedging, 2026 |
| A cheap non-RL alternative | Itô Signatures, arXiv:2608.18120 |
| Canonical empirical setup | Mikkilä & Kanniainen, *Quant. Finance* 23(1) (2023) |
| Anti-doubling machinery | Buehler et al., arXiv:2111.07844; Carbonneau & Godin, *Risks* 11(8):140 |