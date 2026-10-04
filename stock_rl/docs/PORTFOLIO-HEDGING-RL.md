# Portfolio Management + Hedging with RL - Extension

This extends NIFTY-RL Alpha Engine with two new RL layers on top of stock-level signals.

## 1. Architecture - Dual RL

[Portfolio and Hedging RL Architecture](/mnt/data/resource/image_20261002_235043.webp)

**Flow:**
Single Stock Signals (BUY/SELL from existing system) -> Portfolio RL Agent (weights across 50) -> Hedging RL Agent (adds Nifty Futures/Options hedge) -> Final Portfolio + Hedge Orders

Both RL agents share the same State Representation but have different Rewards and are trained jointly (multi-agent RL).

## 2. Why Classical Fails in NSE

- **Markowitz** assumes static correlation. In India, correlation goes to 1 during crash (Mar 2020, all 50 fell). RL learns dynamic correlation from Correlation Agent.
- **Black-Scholes delta hedging** assumes continuous rebalancing and zero costs. In NSE, STT + gap up + liquidity = big loss. RL learns cost-aware hedging.
- **Fixed hedge** (always 50% Nifty future short) loses money in bull market. RL learns when to hedge.

## 3. Portfolio Management RL Agent

**Goal:** Maximize risk-adjusted return with constraints: max 10% per stock, max 30% per sector, turnover < 5x.

### State `s_t` (250-dim)

**A. Holdings State (from Execution Service):**
- Current weights w_t (50-dim)
- Unrealized PnL, cash, last rebalance time
- Sector exposure: IT 25%, Financials 30%, etc.

**B. Market State (from Feature Store + Agents):**
- Stock-level signals: confidence 0.81 for RELIANCE BUY, etc. (50-dim)
- Correlation Matrix (50x50) -> embedding via GNN: Correlation Agent pulls rolling 30d correlation, regime shift detection (HMM says correlation regime high)
- Volatility State: India VIX 14.5, stock-wise realized vol. Volatility Agent pulls: VIX term structure, VIX futures
- FII/DII Flow: FII net sell 1200cr last 5d -> risk off. FII Flow Agent pulls NSDL data
- Macro: USD/INR 83.2 depreciating, US 10Y 4.5% up. Macro Agent pulls RBI, Fed

**C. Risk State:**
- Portfolio VaR 95% (1-day), CVaR, Beta to Nifty, Max Drawdown YTD

### Action `a_t`

Continuous action space: target weights w_{t+1} in simplex sum=1, 0<=w_i<=0.10
Uses **SAC (Soft Actor-Critic)** or **DDPG** for continuous control, not PPO discrete.

```python
action_space = Box(low=0, high=0.10, shape=(50,)) # then project to sum=1
```

Constraints enforced via action masking: if sector >30%, penalize.

### Reward `r_portfolio`

Differential Sharpe + penalty terms:

```
r_t = alpha * differential_sharpe - beta * turnover_cost - gamma * sector_concentration_penalty - delta * max_drawdown

turnover_cost = 0.0011 * ||w_{t+1} - w_t||_1
sector_penalty = max(0, sector_exposure - 0.30)^2
alpha=1.0, beta=0.5, gamma=2.0, delta=1.0
```

Trained on 2014-2024, walk-forward monthly.

### Agents Feeding Portfolio RL

| Agent | What It Pulls | Why It Matters for Portfolio |
|---|---|---|
| **Correlation Agent** | 30d/90d correlation matrix, regime shift (HMM), GDELT for cross-asset | If IT stocks correlation jumps from 0.4 to 0.9, reduce IT weight, diversify |
| **Volatility Agent** | India VIX, VIX futures contango, stock realized vol forecast | VIX >20 -> reduce gross exposure, increase cash to 20% |
| **FII Flow Agent** | NSDL daily FII net, client-wise FII vs DII, US bond yields | FII selling 5 days straight -> defensive: FMCG, Pharma overweight |
| **Macro Agent** | RBI repo, inflation, crude, USD/INR | Rate hike -> underweight Banks, overweight IT (benefits from weak INR) |
| **Sector Dependency Agent** | Supply chain graph | If crude up -> overweight ONGC, underweight Paints, Aviation |
| **Geopolitical Tail Risk Agent** | GDELT tail risk score 0-1, event clustering (Hawkes) | Tail risk >0.8 -> trigger Hedging Agent, reduce beta |

## 4. Hedging with RL Agent

**Goal:** Minimize drawdown and tail risk at minimal cost. Learns when and how much to hedge using Nifty derivatives.

### Instruments Allowed (NSE Legal, liquid)

- **Nifty 50 Futures** (monthly)
- **Nifty 50 Options**: ATM/OTM puts, put spreads, collar (long put + short call)
- **Stock Futures**: For single-stock hedge (e.g., long RELIANCE, short RELIANCE future if earnings risk)
- **Not allowed:** Exotic, weekly Bank Nifty if portfolio is Nifty 50 (basis risk)

### State `s_hedge` (same unified state + Greeks)

- Portfolio Greeks: Delta, Gamma, Vega, Theta of current holdings + existing hedges
- Volatility Surface: Nifty ATM IV 15%, skew (25 delta put IV - call IV), Options Flow Agent pulls: PCR, OI build-up at strikes
- Hedging Cost: Put premium decay, bid-ask spread from order book
- Tail Risk: Geopolitical Agent score, Macro Agent event calendar (RBI MPC in 2 days, US Fed tonight)

### Action `a_hedge`

Continuous: hedge ratios

```python
action = {
  "nifty_future_hedge": Box(-0.5, 0.5), # -0.5 = 50% portfolio short via future
  "nifty_put_hedge": Box(0, 0.3), # 0-30% portfolio notional in puts
  "put_strike": Discrete([0.95, 0.90, 0.85] * spot), # 95%, 90%, 85% OTM
  "put_expiry": Discrete([7, 30, 60] days)
}
```

Uses **SAC** with action masking: cannot buy puts if IV > 30% (too expensive) unless tail risk >0.9.

### Reward `r_hedge`

Minimize drawdown, penalize cost:

```
r_hedge = - lambda1 * max_drawdown - lambda2 * hedging_cost + lambda3 * (var_reduction)

hedging_cost = option_premium_decay + futures_mark_to_market + transaction_cost (STT 0.0625% on options sell)
var_reduction = VaR_unhedged - VaR_hedged
lambda1=2.0 (drawdown heavy penalty), lambda2=1.0, lambda3=1.5
```

**Combined Reward (Portfolio + Hedging jointly trained):**

```
R = alpha * Sharpe - beta * MaxDD - gamma * Turnover - delta * HedgingCost
alpha=1, beta=2, gamma=0.5, delta=1
Objective: Maximize risk-adjusted return while hedging tail risk and reducing drawdown
```

Joint training via **Multi-Agent RL (CTDE: Centralized Training Decentralized Execution)** - during training, both agents see each other's actions, during execution they act independently.

## 5. How Agents Get Right Data - Active Pull

Unlike static feature store, agents actively pull based on RL uncertainty.

**Example Workflow: Iran-Israel Escalation 14:30 IST**

1. Geopolitical Tail Risk Agent detects GDELT event severity 0.88, Crude +4%
2. Volatility Agent sees India VIX jumps 12->18, Nifty put IV skew widens
3. Correlation Agent detects correlation Nifty 50 goes 0.45->0.82 (risk-off)
4. State Representation updates: tail_risk=0.88, VIX=18, correlation_regime=high
5. Portfolio RL: reduces high beta (ADANIENT, TATASTEEL) from 8% to 3%, increases FMCG (HUL) 5%->9%, cash 5%->15%
6. Hedging RL: action = buy 90% OTM 30-day Nifty puts for 10% notional, short 20% Nifty future. Reason: cheap hedge before IV explodes
7. Next day: Nifty falls 2.5%, puts gain 180%, futures hedge gains, portfolio drawdown only 0.4% vs unhedged 1.8%

**Data Pull Logic:**

```python
if portfolio_rl.uncertainty > 0.8 or tail_risk > 0.7:
    tasks = [
        volatility_agent.pull_iv_surface(),
        options_flow_agent.pull_pcr_oi(),
        correlation_agent.pull_rolling_corr(30),
        geopolitical_agent.pull_gdelt_last_6h(),
        fii_flow_agent.pull_nsdl()
    ]
    context = await asyncio.gather(tasks)
    state = augmenter.fuse(price_state, context)
```

## 6. Microservices for Portfolio & Hedging

Add two new services to existing 8:

| Service | When Spins Up | Compute | Key Logic |
|---|---|---|---|
| **Portfolio RL Service** | Every 15 min during market + on rebalance trigger (turnover>threshold) | 1 vCPU, 1 GPU for SAC inference <50ms | SAC policy -> target weights, project to constraints via cvxpy |
| **Hedging RL Service** | When tail_risk>0.6 or VIX>16 or portfolio beta>1.2 or event calendar (RBI MPC in 2d) | 1 vCPU, ONNX | SAC -> hedge ratios, strike/expiry, cost-aware |

**KEDA Scaling:**
```yaml
# Hedging service scales from 0
minReplicaCount: 0
triggers:
- type: redis
  metadata: { address: redis:6379, listName: tail_risk_queue, listLength: "1" } # tail_risk >0.6 pushes to queue
- type: cron
  metadata: { timezone: Asia/Kolkata, start: 0 9 * * 1-5, end: 0 16 * * 1-5, desiredReplicas: "1" } # market hours
```

**Backend Flow:**
`Feature Store + Reasoning Agents -> State Representation Service (new) -> Portfolio RL Service -> Risk Service (checks sector 30% limit) -> Hedging RL Service -> Combined Orders -> Execution Service (2 orders: cash + derivative)`

**Frontend Addition:**
- **Portfolio Dashboard:** Current vs target weights treemap, sector exposure, VaR dial, beta, Sharpe YTD
- **Hedging Panel:** Current hedges (e.g., Long 10x Nifty 22500 PE 30d), Greeks, P&L of hedge, cost spent this month, tail risk meter
- **What-If:** Slider "Nifty -5% tomorrow" -> shows portfolio with and without hedge: -4.2% vs -0.8%

## 7. Training - How to Train Both

**Phase 1: Portfolio RL Offline (2014-2023)**
- Data: 1-day OHLCV + corporate actions + FII flows
- Env: Gym `PortfolioEnv` - action weights, reward Sharpe
- Algo: SAC with action projection to simplex, batch size 1024, replay buffer 1M
- Constraint: Use cvxpy layer in policy to enforce sector limit

**Phase 2: Hedging RL Offline (with fixed portfolio)**
- Freeze portfolio weights = Nifty 50 equal weight
- Env: `HedgingEnv` - simulate buying puts/futures on historical crashes (2020, 2022, 2024 election)
- Reward: minimize drawdown
- Cost: Include real Nifty option historical bid-ask + STT

**Phase 3: Joint Fine-Tune (2023-2024)**
- Both agents train together, shared replay buffer
- Use CTDE: centralized critic sees both actions, decentralized actors
- Curriculum: easy (low vol) -> hard (high VIX, high tail risk)

**Local Testing for New Services:**
```bash
# Portfolio RL unit
pytest tests/portfolio/test_weights_sum_to_one.py
pytest tests/portfolio/test_sector_limit.py # sector >30% -> penalty

# Hedging RL
pytest tests/hedging/test_cost.py # put decay + STT cost included
pytest tests/hedging/test_greeks.py # delta hedge calculation

# E2E: Portfolio + Hedging
make e2e-portfolio
# Seed: portfolio 50% IT, tail risk 0.9
# Assert: portfolio reduces IT to 20%, hedging buys puts 10% notional
# Assert: combined VaR reduces 2.5% -> 0.9%
```

## 8. Risk & Legal for Hedging in India

- **SEBI:** Hedging via Nifty futures/options is allowed for portfolio, but must be declared as hedge, not speculative. Maintain hedge ratio < portfolio value.
- **STT:** 0.0625% on option premium sell, 0.0125% on futures sell. Include in cost.
- **Margin:** Hedged position margin benefit (NSE): long stock + short future = 60% margin benefit. RL should learn this - holding hedge reduces margin cost.
- **F&O Ban:** Cannot hedge with stock that is in ban. Risk service checks.
- **Audit:** Every hedge order tagged with `hedge_for: portfolio_id, reason: tail_risk 0.88 Iran`

## 9. Combined Reward Formula Implementation

```python
def combined_reward(portfolio_pnl, hedge_pnl, turnover, hedging_cost, var_unhedged, var_hedged, max_dd):
    sharpe = calc_differential_sharpe(portfolio_pnl + hedge_pnl)
    dd_penalty = max_dd ** 2
    var_reduction = var_unhedged - var_hedged
    cost = turnover * 0.0011 + hedging_cost
    return 1.0*sharpe - 2.0*dd_penalty - 0.5*cost + 1.5*var_reduction
```

**Objective:** Maximize risk-adjusted returns while hedging tail risk and reducing drawdown via dual RL agents.

---

## 10. What You Get

Before: `RELIANCE BUY 0.81`
After:
- Portfolio: `Target: RELIANCE 8% (was 6%), IT sector 18% (was 28%), Cash 12%`
- Hedge: `Buy 10x Nifty 22500 PE 30d (90% strike) for 10% notional, Short 20% Nifty Future`
- Reasoning: `Tail risk 0.88 Iran-Israel, VIX 18->25 expected, Correlation 0.82, FII outflow 1200cr, Options flow PCR 1.4`

This is production-grade portfolio management, not signal generator.
