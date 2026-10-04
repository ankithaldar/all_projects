# Project Structure and Microservices

**Recommendation: one VM, modular monolith, at most three processes.**
The design doc proposes ~11 deployables and ~25 infrastructure components
to serve **1,184 lines of dependency-free Python with 946 lines of tests**.

## The framing problem

The research read the actual repository, not just the doc:

| | |
|---|---|
| Source | `bars.py`, `metrics.py`, `splits.py`, `engine.py`, `costs.py`, `indicators.py` |
| Tests | 90% coverage floor enforced in CI |
| Runtime deps | **zero** — pure stdlib, by explicit design decision with a documented upgrade path |
| Design doc | 489 lines proposing 11 services + 6 datastores + 14 tools |

The doc proposes to dismantle a correct monolith-first discipline.

## What the evidence says

**Martin Fowler, MonolithFirst:**

> *"Almost all the successful microservice stories have started with a
> monolith that got too big and was broken up. Almost all the cases
> where I've heard of a system that was built as a microservice system
> from scratch, it has ended up in serious trouble."*

**MicroservicePremium** — the actual guideline:

> *"Don't even consider microservices unless you have a system that's too
> complex to manage as a monolith. **The majority of software systems
> should be built as a single monolithic application.** Do pay attention to
> good modularity within that monolith, but don't try to separate it into
> separate services."*

The stated fulcrum is **system complexity**, driven by: large teams,
multi-tenancy, independent business-function evolution, scaling. This
project has **one** of those four, and it is the weakest.

**On the counter-argument.** Stefan Tilkov's *Don't Start with a Monolith*
is the strongest dissent: microservices enforce modularity that discipline
alone fails to enforce. But note where he concedes:

> *"If it's just you and one of your co-workers building something over
> the course of a few weeks, it's entirely possible that you don't."*

That is a direct description of this project.

**Small-team evidence:** "When you get down to the other extreme end of
the scale, where you have less than five developers working on an entire
system, you'll find that there are huge advantages to sticking to a
monolith" (Simple Thread). HN consensus, repeated: **"Microservices takes
at least 25% of your engineering time just maintaining the platform."**
Even the microservices defender in that thread called 25% "a bit high."

**Verdict:** microservices solve a coordination problem that appears at a
team size you have not reached.

## What premature decomposition actually costs

**a) Distributed tracing becomes mandatory.** Once risk and execution are
separate services, a bad order may pass risk on Monday and execute on
Tuesday after a partition. The only way to reconstruct what happened is a
distributed trace. **Loki + Jaeger + OpenTelemetry in the design are not
nice-to-haves — they are the tax on the decision.** They never appear in
monolith designs.

**b) Test strategy collapses into "observability."** *"Integration testing
a distributed setup is a nearly-impossible problem, so we pretty much gave
up on that and replaced it with another one — Observability"* (Renegade
Otter). Fast deterministic zero-dependency unit tests get replaced by five
test layers, Pact contracts, Testcontainers, and a Locust suite that
measures the mock harness rather than the system.

**c) Eventual consistency becomes a correctness problem.** Four Kafka
topics means the order lifecycle becomes
`PLACED → QUEUED → VALIDATED → EXECUTED` with possible duplicates and
gaps. Reconciling that correctly is a project. In a monolith it is a
function call.

**d) Failure modes multiply faster than features.** 11 services means
~11×11 = **121 pairwise failure modes**, versus ~0 in one process.

**e) The DRY inversion.** Microservices duplicate boilerplate by default.
`costs.py` — the NSE-specific STT/GST/stamp-duty/DP model — is domain
logic that must exist in **exactly one place**. Split it and you get three
implementations of Indian transaction costs that silently drift. **That is
how a backtest stops matching live fills.**

**f) Deployment atomicity is lost.** One commit changing the state vector
must change feature builder + policy + backtest together. Across three
services that is a coordinated release during which a schema mismatch
means orders sized on a 179-dim state hit a 180-dim policy. In a monorepo,
`make check` prevents shipping.

## Scale-to-zero is economically inverted

The claim: "saves ~15k INR/month GPU" (~$180/mo). The design keeps **API
Gateway + Kafka always-on** (its own words).

| Always-on component | Monthly, ap-south-1 |
|---|---|
| EKS control plane | $73 |
| MSK 3x kafka.m7g.large | $319 |
| NAT Gateway x2 AZ | $86 |
| 2x m6i.large node group | $147 |
| **Pure overhead** | **$625** |

**$625/month of always-on overhead to enable a mechanism worth $180 — 3.5x
net negative** before a line of KEDA YAML is written.

Deeper problems:

- **You cannot scale an EKS cluster to zero.** AWS's own guidance: *"we
  cannot scale all of our nodegroups to 0 as we do need to guarantee a
  minimal stack of core components to be constantly up and running. Such a
  stack would include, at the very least: the cluster autoscaler, the
  CoreDNS."* Karpenter's controller also needs a node.
- **Pod scale-to-zero ≠ cost saving.** *"KEDA scales pods to zero, but if
  scaling to zero leaves a node empty, you only capture the saving when
  the cluster autoscaler actually removes that node. Scale-to-zero without
  node-level scale-down is half a win."* One team celebrated pod counts
  dropping before realizing *"the nodes were still there."*
- **GPU nodes are the worst possible subject.** Managed node groups with
  GPU accelerators historically cannot reliably go 0→1.
- **Cold starts are incompatible with trading.** Realistic KEDA HTTP cold
  start: **10–30 seconds**. For a market opening at 09:15 with 1–2 s stale
  ticks, a 30 s cold start is a strategy-level bug, not an optimisation.
- **The $180 is internally inconsistent.** A g5.xlarge is $1.208/hr
  on-demand ($882/mo). Perfect scale-to-zero at 6.25 h x 22 days = 137.5 h
  = $166/mo, saving $716. So the claim is 4x too low *and* misattributed —
  $180 is roughly the cost of 6 hours of GPU.

**What actually works:** schedule an EC2 GPU instance up for the training
window, then terminate it. An EventBridge/Lambda cron or a `cron` job —
~20 lines, no KEDA, no Kafka, no Karpenter, no node consolidation tuning,
no cold starts. It captures **100%** of the theoretical saving.

## Six datastores for <2 GB of data

| Asset | Size | Verdict |
|---|---|---|
| OHLCV, 50 symbols x 10y x 1-min | ~7M rows, **~150 MB** | Fits in RAM. **No TimescaleDB** |
| Orders/fills/audit | ~50–500 rows/day → ~15k/month | **SQLite**, WAL mode |
| News embeddings | ~50–500k x 768 ≈ 150 MB–1.5 GB | `sqlite-vec` exact KNN. **No Qdrant server** |
| Feature cache | 50 symbols x 180 floats | **A SQLite table** |
| Kill-switch flag | 1 boolean | **A SQLite row** |
| "Knowledge graph" | ~300–2000 edges, static | **A CSV/YAML file** |
| Audit trail | append-only, 5y retention | **S3 with Object Lock** — correct use of S3 |

**Redis is the most indefensible.** Its only listed jobs are caching
features for 50 symbols (a 50-row SQLite read) and holding the kill-switch
boolean. ~$130/mo and a new failure mode for two dictionary lookups.

**Neo4j is second.** A dependency graph for one asset class is a static
edge list; ~300 edges need no Cypher, no JVM, no container. Moreover
`metrics.py` and `engine.py` are pure-stdlib by design — a Cypher
dependency breaks the isolation that makes backtests reproducible.

**Postgres *and* TimescaleDB listed separately is the tell: TimescaleDB
*is* a Postgres extension.** Two of the six are the same database.

**Replacement: two file-based stores.** SQLite (WAL) for state, Parquet +
DuckDB for history. ~6 fewer datastores to back up, patch, monitor and
reconcile.

## Realistic costing

**Scenario A — as written, minimum viable (scale-to-zero works, no A100):**

| Item | $/mo |
|---|---|
| EKS control plane | 73 |
| MSK 3x kafka.m7g.large + storage | 339 |
| NAT Gateway x2 | 86 |
| ALB + 4x public IPv4 | 31 |
| General node group 2x m6i.large | 147 |
| Memory node group r6i.large | 95 |
| g5.xlarge, 137.5 h | 166 |
| RDS Postgres db.r6g.large | 146 |
| RDS Timescale (separate!) | 146 |
| ElastiCache Redis r6g.large | 130 |
| Neo4j Aura Professional | 65 |
| Qdrant Cloud Standard | 100 |
| S3 | 10 |
| CloudWatch | 80 |
| Grafana Cloud Pro | 40 |
| W&B team | 30 |
| **Total** | **~$1,680** |

**Scenario C — literally as written** (4xA100 nightly 12 h ≈ $6,785,
plus g5.xlarge left on-demand): **~$8,500–9,000/month**.

**Is $180 plausible? No — 9x off at the floor, 50x as written.**

Major drivers, ranked: GPU training 0–81% depending on config; MSK
$319–484; **database sprawl $487 for six stores**; EKS control plane $73
(unavoidable once you choose EKS — and **$438/mo in extended support** if
a k8s version ages; Kubernetes 1.34 reached EOL 27 Oct 2026); networking
$117–150 (NAT gateways are the classic surprise, and microservices
multiply inter-AZ chatter at $0.01/GB each way).

**Plus the cost that doesn't appear on the bill:** a competent engineer to
run EKS + KEDA + Karpenter + Istio + MSK + 6 datastores + 14 tools, and
repeat every version bump, node failure and broker incident. At 0.25 FTE
that is **$3,000–5,000/month — 20–30x the infrastructure itself.**

**A ~$180/month alternative that actually works:** 1x m6i.large ($74) + S3
($2) + a nightly 2 h g5.xlarge **spot** rental (~$28) + domain ($1) =
**~$109/mo**, plus the managed news feed (Rs 2,000–5,000/mo per the design
doc, which exceeds the compute anyway).

## Latency is the wrong target

**Zerodha Kite's own documentation:** *"Kite Connect is not meant for
latency based strategies or HFT. We don't guarantee any timeline for
carrying out a request."*

Measured on the Kite forum: **80–150 ms typical**, 450 ms observed,
**3–8+ seconds at market open**. Zerodha staff: *"there are a lot of hops,
internet lines and lease lines... it might take few more milliseconds to
seconds on a very heavy traffic day."*

**Critically, your data is stale before you start:** *"Zerodha tick data
is delayed by 1 to 2 seconds therefore, as traders, we only know the price
of an instrument after 1 to 2 seconds."*

NSE publishes exchange-internal mean latency of **~130 microseconds**. The
exchange is 1000x faster than the path around it. **Your latency budget is
broker and internet, not your code.**

| Stage | Budget |
|---|---|
| Vendor tick latency (uncontrollable) | **1,000–2,000 ms** |
| Your compute (state assembly + forward pass) | 5–50 ms |
| Risk checks (pure Python) | <1 ms |
| Your VM → Zerodha (Kochi ↔ Mumbai) | 20–80 ms |
| Zerodha RMS/OMS validation | 80–150 ms |
| Exchange matching | ~0.13 ms |
| Ack back | 20–80 ms |
| **Total to ack** | **~150–400 ms typical; seconds at open** |

**Where the 100 ms target fails:**

1. **It measures ~2% of the budget.** Inference 100→20 ms changes
   end-to-end by 80 ms on a 250 ms path.
2. **The design's own agents blow the budget by 150x.** Goals say
   *"reasoning agents < 15s"*. 15 s is 150x the inference target, and the
   dataflow diagram puts them squarely in the critical path.
3. **The API contract contradicts the latency requirement.**
   `reasoning_trace: {bull, bear, winner}` in the response means an LLM
   verdict per signal. That is seconds, not milliseconds.
4. **Optimisation effort is misallocated.** At 6.25 h/day and 50 symbols
   there are 450,000 symbol-decisions per year. The entire annual compute
   budget is ~$50 of instances.

For reference, real latency money: **Zerodha quotes NSE colocation at
Rs 18 lakh/year (~$21,600)**. That is the regime where this architecture
is justified. This project is **~200x away from it**.

**Optimise for correctness, determinism and auditability — not latency.**

### Compliance finding the doc missed

SEBI 2012 circular para 4(iii) requires algo orders routed through broker
servers **located in India** with *"no interlink with any system or ID
located/linked outside India."* If the multi-agent orchestrator calls a
**US-region LLM API**, Indian market data and live trading signals leave
the country. On-prem inference (Ollama/vLLM on the same box) resolves it
and removes per-token cost.

## Observability: 8 systems → 1 decision log

The design lists **both Loki and ELK** — a duplicate.

With one process there are no cross-service traces to correlate. Loki,
Jaeger and OpenTelemetry exist because you paid the Microservice Premium
and now need to see across the resulting seams. In a monolith, a
`correlation_id` in log lines **is** the trace.

**Minimum viable, in priority order:**

1. **Append-only decision log → SQLite + S3 Object Lock.** Every signal,
   order, fill, rejection with timestamp/symbol/decision/reason/model
   version. Simultaneously your **audit trail** (free compliance) and
   primary debugging tool. **Not optional.**
2. **Structured JSON logs to stdout**, `correlation_id` threaded. $0.
3. **Four alerts:** broker API error rate > 0; order-ack p99 > 2 s (vs the
   80–150 ms baseline); kill-switch tripped; daily realized loss > limit.
   Everything else trains you to ignore alerts.
4. **Equity curve + positions queryable in SQL.** *"What did the model
   decide at 10:15:40?"* must be a query, not a dashboard reconstruction.
5. Optional, $0: Grafana Cloud free tier.

**MLflow vs W&B: pick exactly one.** They overlap ~80%. Or use **git**,
which you already have: `models/run-47/{weights,manifest.json,backtest.csv}`
committed answers *"what changed between run 47 and 48?"* with `git diff`.

**Evidently on live data is near-useless early.** You cannot detect
distribution drift without a baseline, and you will not have months of
live samples for a long time. Defer until ~200 trading days. Note
`metrics.py` already computes a **Deflated Sharpe Ratio**, which addresses
multiple-testing more directly than feature drift.

## Local dev on a 16 GB laptop is not realistic

| Component | RAM |
|---|---|
| kind control plane + node | 600 MB – 1 GB |
| **Kafka (KRaft)** | 512 MB – 1 GB |
| **LocalStack** | 800 MB – 2 GB |
| **Neo4j (JVM + page cache)** | 1.5 – 2 GB |
| TimescaleDB | 400 MB – 1 GB |
| KEDA + metrics-server | 150 MB |
| Qdrant | 200 – 500 MB |
| MinIO | 200 MB |
| **WireMock (JVM)** | 300 – 500 MB |
| **Chaos Mesh** | 800 MB – 1.5 GB |
| 11 service pods | 1 – 2 GB |
| Docker Desktop overhead | 2 – 4 GB |
| IDE + browser | 3 – 6 GB |
| **Total** | **~12–20 GB** |

**You have 16 GB.** This does not fit with headroom — you will OOM-kill
pods mid-test, which presents as **flaky tests**, the most demoralizing
failure mode available.

**Boot time:** kind 20–60 s, then 8 stateful images (multi-GB, several
JVM-based) 5–15 min cold, Helm + KEDA 2–5 min, Chaos Mesh 3–5 min.
**A 15–25 minute `make local-up`**, and every service added makes it
worse forever.

Structural problems: **two JVM stacks (Kafka, Neo4j, WireMock, LocalStack)
for a system that needs neither.** The local environment doesn't match
production — kind's CNI-over-Docker adds 1–5 ms plus jitter, so the E2E
expectation *"Latency p99: 87ms PASS"* **measures kind's CNI, not your
system.** And the design says "no GPU" locally while the sentiment engine
needs one: the one component you most want to validate is the one you
can't run.

**Instead:**

| Instead of | Use |
|---|---|
| kind + helm + KEDA + Tilt + Chaos Mesh | **`docker compose`, 3 services**: `api`, `web` (static build), `db` (SQLite on a volume) |
| WireMock | **A 60-line fake broker** — you need `POST /orders/regular` → `{order_id}`, not a JVM |
| localstack + minio | **Local SQLite + Parquet directory**, `rsync` to S3 nightly |
| Kafka | **Nothing.** An in-process queue |
| Testcontainers | **Plain pytest.** With SQLite there is nothing to containerize |
| Pact contracts | **Type hints + mypy + a shared schema.** You cannot have contracts between modules in one process |
| Chaos Mesh | **Inject the fault.** `monkeypatch` the broker to raise; set `kill_switch=1` |
| Locust | **A `timeit`-based assertion in pytest** for your real ~50 ms budget |

If you deploy to k8s later, use **k3d/k3s, not kind** — k3s replaces etcd
with SQLite. And deploying to a single VM with compose makes local compose
**faithful**, which is worth more than fidelity to a cluster you must
maintain.

## When this architecture WOULD be right

All six must hold:

1. **≥15–25 engineers across ≥5 teams**, each owning a service. Fowler's
   own survey spans "a team of 60 with 20 services" to **"a team of 4 with
   200 services" — the second end is the failure mode, and it is where
   this project sits.**
2. **Measured independent-scaling mismatch**, not hypothetical. The one
   real candidate: training bursts to many GPUs while inference needs
   2 vCPUs for 6 h/day. The answer is a **separate batch host**, not a
   microservice. The design's strongest instinct pointed at the wrong
   remedy.
3. **Evidence of release-cadence blocking** — actual data: teams shipping
   20x/day, blocked on each other's deploys. Here: 1–3 people, ~6
   commits/week. **This problem does not exist.**
4. **Compliance requiring physical isolation. SEBI does not require
   this — and partly requires the opposite.** What SEBI *does* require:
   algo registration + unique Algo ID, OAuth + 2FA, static-IP whitelisting,
   audit trail 5 years, RMS pre-trade checks, kill switch per Algo ID,
   **servers located in India**. **None of that is a network boundary.**
   NSE Annexure I para 14: *"all the strategies shall be run on the
   brokers servers."* **Regulation wants centralized control, and
   microservices erode it.**
5. **Multiple asset classes with genuinely different risk profiles.** Even
   then: separate processes on one box first. This project is explicitly
   one asset class.
6. **A real latency requirement** — colocation, single-digit ms. At 1–2 s
   stale ticks and 80–150 ms of broker validation, code latency is rounding
   error.

**Honest framing:** the design is a well-built blueprint for a 15–50
person multi-asset, multi-strategy, possibly-colocated platform, applied
to a 1–3 person single-asset research project. **The engineering taste in
it is fine. The sizing is wrong by an order of magnitude.**

## Recommended structure

```
stock_rl/
  pyproject.toml                 # uv + hatchling, zero runtime deps
  src/stock_rl/
    bars.py  costs.py  metrics.py  splits.py  engine.py  indicators.py
    compliance/                  # robots, rate limit, algo tagging
    decisions.py                 # BUY/HOLD/SELL + timeframe + reasons
    baselines.py                 # 1/N, buy&hold, momentum, low-vol
    skills/                      # markdown skill registry for agents
    execution/audit.py           # append-only decision log
  scripts/
    fetch_data.py                # vendor -> Parquet
    train_nightly.py             # cron + spot GPU + git commit
  tests/
  deploy/
    docker-compose.yml           # THE ONLY runtime definition
    systemd/stock-rl.service
```

**Three processes maximum:**

1. **`train`** — nightly cron. Batch. **Never in the trading path.**
2. **`api`** — inference + risk + execution in **one process**, systemd
   unit, `Restart=always`.
3. **`web`** — static build on Caddy. *Optional:* a CLI equity-curve PNG is
   enough to validate.

## The decisive argument: co-locate risk + execution

SEBI/NSE require an RMS to enforce **per order**: order-value check,
cumulative open-order-value check, and an automated-execution check
accounting for all executed/unexecuted/unconfirmed orders before releasing
more.

**If risk and execution are separate network services, a partition between
them must fail either open** — unguarded orders reach the exchange, a
compliance breach — **or closed**, no orders place at all.

That is a safety-critical coupling, and microservices **split exactly the
thing you must never split.** The design's E2E suite tests *"kill
sentiment-engine pod → degraded price-only signal still works"* but never
tests **kill the risk service**, which is the only one that matters. In one
process this failure mode is **impossible by construction.**

Also: the design's kill switch is a React button. SEBI: *"the kill switch
is an emergency function and the last level of defence... It is expected to
**automatically** trigger a halt on trading activity based on pre-defined
conditions."* The real control is a **server-side circuit breaker that
trips automatically** — ~30 lines in `risk/`.

## Deployment target: one VM

| Component | Spec | $/mo |
|---|---|---|
| `m6i.large` ap-south-1 (or `t3.large` to start) | 2 vCPU / 8 GB, Ubuntu + Docker Compose | 74 |
| Public IPv4 | 1 | 4 |
| S3 (Parquet, models, Object-Lock audit) | ~50 GB | 2 |
| Nightly training: `g5.xlarge` **spot**, 2 h x 22 nights | launched + terminated | 28 |
| Domain | — | 1 |
| **Total** | | **~$109/mo** |

Plus the data feed (Rs 2,000–5,000/mo) you pay regardless. **~$150–180/mo
including data — the claimed number, while actually running the strategy.**

Deploy to **Kochi/Mumbai ap-south-1** — required by SEBI anyway.

## Escalation triggers

Move off one VM **only** when one of these actually happens:

| Trigger | Move to |
|---|---|
| Single VM dies mid-session → miss market | 2 VMs + ALB, or ECS Fargate |
| >5 concurrent strategies | separate **processes**, same box |
| Multiple asset classes (F&O, currency) | separate processes, separate RMS limits |
| Training needs to burst independently | separate **batch host** (not a microservice) |
| **>8–10 engineers, ≥3 teams** | *then* reconsider services |

Path: **1 VM → 2 VMs + ALB → ECS Fargate → EKS.** Each step only when an
event forces it. Note **ECS is strictly cheaper than EKS** — no
control-plane fee, where EKS costs $73/mo minimum.

## What the design gets right

- **Look-ahead bias, purge/embargo, NSE cost modelling are genuinely the
  hard parts.** Most systems get them wrong and report fantasy Sharpes. The
  existing code is better than most.
- **Deflated Sharpe** addresses statistical overfitting directly — real
  and underappreciated.
- **SEBI-aware audit trail thinking** is correct and required.
- **License-aware data sourcing** (~Rs 2–5k/mo) is realistic and avoids
  the scraping trap.

**Keep the zero-dependency pure-stdlib core.** It makes backtests
reproducible, fast and dependency-free. `costs.py` encoding Indian charges
per side is domain logic that belongs in exactly one place. `splits.py`
purge+embargo is the difference between a real Sharpe and a fantasy one.
**Don't let the doc's 11 services fragment any of it.**

## Sequencing

1. **Test the research hypothesis first.** The claim is that LLM-fused
   context beats price-only. You can test that against the code you have:
   a CSV of headlines, CPU embeddings, one extra feature in the state.
   **If the Deflated Sharpe does not move, no architecture saves it.** The
   design spends ~10,000 words on infrastructure and zero on this question.
2. **Get one Python process placing paper orders**, logging every
   decision. Full E2E, one process, SQLite.
3. **Run live with `minQty` lots and real money.** Get the audit trail
   right — that is the SEBI obligation.
4. **Only then consider deployment.** You will discover the real
   requirements in steps 1–3, and they will not be the ones in the doc.

**Total design doc: 489 lines. Existing working code: ~1,200 lines with
~1,000 lines of tests. The gap is not "needs 11 microservices" — it's
"needs about 3,000 more lines of Python."**