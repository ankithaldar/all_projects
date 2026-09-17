# Chapter 2 — Planning, Reasoning & Structured Prompting

> Master **chain-of-thought, ReAct, self-validation, and dependency-aware
> planning** — and above all, *when to use which* — built from scratch,
> no LangChain/LangGraph, no agent frameworks.
> All LLM calls flow through **your local LLM gateway** (in-process import).
> Tools are Chapter 1's retail/telecom MCP servers behind a swappable
> adapter (`inproc` or `mcp`); `chapter01_mcp` and `llm_gateway` are untouched.

---

## 1. The journey: Tool-Use Loop → CoT → ReAct → Validation → Planning

Chapter 1 gave the agent **hands** (tools). Chapter 2 gives it a **thinking
strategy**. Follow the problems each era solved:

```
Tool-Use Loop (Chapter 1)
  └─ The model calls tools until done. Great for lookups and actions, but
     for multi-step reasoning it just... answers. No intermediate compute,
     no plan, no check of its own result.

Chain-of-Thought (2022)
  └─ "Think step by step." Intermediate tokens are scratch compute: the
     model gets more tokens to calculate with. Multi-step arithmetic gets
     measurably better, and reasoning becomes auditable. New problem:
     free-form prose must be parsed defensively.

ReAct (2022)
  └─ Interleave Thought → Action → Observation. The model reasons about
     which tool to call, sees the result, reasons again. Reasoning is now
     grounded in EVIDENCE, not memory. New problem: unbounded loops,
     invented tools, protocol drift.

Self-validation / reflection (2023)
  └─ Before answering, check the answer: are its numbers in the evidence?
     was a blocked write acknowledged? A hybrid of free deterministic
     checks plus an optional LLM critic. New problem: what if the task
     needs several interleaved objectives?

Task decomposition (2023+)
  └─ Ask the model for a PLAN: a dependency DAG of small, verifiable steps.
     Validate the DAG (pydantic + networkx), execute it in topological
     waves, and skip only what a failure actually invalidates. New
     problem: not every task deserves all this machinery.

Routing (this chapter's glue)
  └─ An explainable policy picks direct / cot / react / plan from task
     features, so cost tracks difficulty. Reason when it pays; enforce
     structure everywhere else.
```

**Reasoning patterns don't add knowledge — they manage compute and
evidence.** The model already "can" reason; our job is to give it the right
shape, ground it in tool results, validate the output, and bound the cost.

### The stack (new view)

```
┌──────────────────────────────────────────────────────────────┐
│ A2A / AG-UI     agent ⇄ agent, agent ⇄ user                  │  ← future chapters
├──────────────────────────────────────────────────────────────┤
│ REASONING       planning, CoT, ReAct, self-validation        │  ← THIS chapter (thinking)
├──────────────────────────────────────────────────────────────┤
│ MCP             agent ⇄ tools / data                         │  ← Chapter 1 (hands)
├──────────────────────────────────────────────────────────────┤
│ LLM GATEWAY     providers, retries, cache, rate limits       │  ← your existing gateway
└──────────────────────────────────────────────────────────────┘
```

Reasoning sits directly on top of MCP: plans and ReAct turns *use* tools,
validate *against* their observations, and never bypass the safety gate.

---

## 2. Architecture

### 2.1 Big picture

```
┌───────────────────────────── Agent process (PlanningAgent) ──────────────────────────────┐
│                                                                                          │
│  PlanningTaskInput ─▶ ReasoningPolicy.decide ─▶ PolicyDecision{route, rationale}         │
│        │                                                                                 │
│        ▼                                                                                 │
│  ┌──────────┬──────────┬──────────┬──────────┐                                           │
│  │  direct  │   cot    │  react   │   plan   │                                           │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘                                           │
│       └──────────┴──────────┴──────────┘                                                 │
│                          │                                                               │
│  ReasoningLLM     ┌──────▼────────┐                                                      │
│  retries · budget │  SelfValidator│                                                      │
│  spans · JSON fix │ checks+critic │                                                      │
│        │          └───────┬───────┘                                                      │
│        ▼                  ▼                                                              │
│  GatedToolProvider ─▶ PlanningTaskOutput + SQLite events/memory/traces                   │
│        │                                                                                 │
└────────┼─────────────────────────────────────────────────────────────────────────────────┘
         ├─ InProcessToolProvider — Chapter 1 server cores (same validation)
         └─ MCPToolProvider — ServerCatalog + MCPClient over stdio
```

### 2.2 The four reasoning routes

| Route | Shape | Use when | Cost |
|---|---|---|---|
| **direct** | one LLM call | lookups, formatting, single facts | lowest |
| **cot** | step-labeled reasoning + `Answer:` line; optional N-sample majority vote | arithmetic, constraints, no external data | medium |
| **react** | Thought → Action → Observation loop over gated tools | answer depends on live data/actions | medium–high |
| **plan** | JSON DAG → validated by networkx → executed wave by wave; optional synthesis | several interleaved objectives, risky writes | highest |

The router is a transparent rule engine (feature scores in the output):

```
rule 1  tool_need > 0 and multi_step >= 2       ─▶ plan
rule 2  write_risk > 0 and multi_step >= 1      ─▶ plan
rule 3  write_risk > 0 and tools available      ─▶ react
rule 4  tool_need > 0                           ─▶ react
rule 5  numeric >= 2 or (numeric and ambiguity) ─▶ cot
rule 6  multi_step >= 3 or ambiguity >= 2       ─▶ cot
rule 7  otherwise                               ─▶ direct
```

`prefer_route` on the task input always wins (`forced=True`) — pinning a
route for latency or compliance is a product decision, not a model one.

### 2.3 One ReAct turn, and one plan wave

```
ReAct turn (one loop iteration)
  Thought: check low stock
  Action: retail-ops.retail_low_stock_report
  Action Input: {"store_id": "S02"}
  Observation: [BEGIN UNTRUSTED TOOL DATA]
               {"items":[{"sku":"R-103","on_hand":7,"reorder_point":30}]}
               [END UNTRUSTED TOOL DATA]
  Thought: the trend decides the order size
  Final Answer: ...

Plan waves (dependency DAG)
  wave 1  inspect_stock  (read,  low_stock_report)
  wave 2  review_trend   (read,  sales_trend; depends on wave 1)
  wave 3  place_order    (write, gated + approved; depends on wave 2)
  failures skip only dependents; independent branches still complete
  (safe partial completion)
```

---

## 3. What we built (file map)

```
chapter02_planning/
├── schemas.py               all pydantic models (tools, policy, patterns, IO)
├── config.py                PlanningConfig + AGENTIC_PLAN_* env loading
├── prompting.py             PromptTemplate ($slots), output contracts, all prompts
├── parsing.py               defensive JSON / CoT / ReAct / plan / critique parsers
├── policy.py                ReasoningPolicy + choose_node_route
├── llm.py                   ReasoningLLM: retries, budget, spans, JSON repair
├── agent.py                 PlanningAgent — THE ORCHESTRATOR
├── tools/
│   ├── provider.py          ToolProvider ABC + GatedToolProvider decorator
│   ├── in_process.py        Chapter 1 server cores, no subprocess
│   ├── mcp_provider.py      live MCP servers over stdio
│   └── factory.py           build_tool_provider('inproc' | 'mcp')
├── patterns/
│   ├── chain_of_thought.py  CoT + self-consistency
│   ├── react.py             bounded ReAct loop (loop detection, format repair)
│   ├── self_validation.py   deterministic checks + critic + revision guard
│   └── decomposition.py     plan graph, TaskPlanner, PlanExecutor
├── evals/
│   ├── cases.py             PlanningEvalCase + chapter-specific scoring
│   ├── cases.yaml           6 retail/telecom eval scenarios
│   └── runner.py            evaluation harness runner
├── demo.py                  end-to-end demo (mock + live, inproc + mcp)
└── docs/README.md           this document
```

Chapter 1 reuse: `InProcessToolProvider` imports `build_retail_server()` /
`build_telecom_server()` and dispatches through `MCPServerCore.handle_message`;
`MCPToolProvider` imports `ServerCatalog` and `MCPSessionManager`. Same tools,
same validation, two transports.

---

## 4. Design patterns in this chapter

### 4.1 Router / Strategy (reason vs enforce structure)

`ReasoningPolicy` scores five features (`tool_need`, `write_risk`,
`multi_step`, `numeric`, `ambiguity`) and returns a `PolicyDecision` with
route, rationale, scores, and confidence. Every decision is logged
(`route_decision`) — routing is auditable, not vibes.

### 4.2 Chain-of-thought + self-consistency

```
task ─▶ CoT prompt ─▶ "Reasoning: 1.. 2.. Answer: X" ─▶ parse steps+answer
                          │
                          └─ N samples at higher temperature
                             majority vote on normalized/numeric answers
                             agreement = confidence; low → needs_review
```

The final answer is delimited so machine consumers never parse prose. A
missing `Answer:` triggers exactly one format-repair call.

### 4.3 ReAct (the tool-use loop, made auditable)

```
for step in 1..max_steps:
   LLM(scratchpad) ─┬─ "Final Answer: ..." ─▶ done
                    └─ "Action: tool / Action Input: {...}"
                          └─ GatedToolProvider ─▶ sanitized observation ─▶ scratchpad

guards: strict parser + 2 format reminders · unknown-tool listing ·
        third identical call stops the loop · step cap + token budget ·
        observations wrapped as untrusted data
```

Unlike Chapter 1's native function calling, the reasoning is explicit:
you can read *why* each tool was called.

### 4.4 Hybrid validation (reflection without the cost)

```
candidate answer + evidence + tool audit + plan results
   ├─ deterministic checks (free, always on)
   │    answer_present · grounding (numbers must be in evidence)
   │    tool_outcomes  · evidence_used · plan_outcomes
   ├─ LLM critic (paid, only when a warning/error was found or route=plan)
   └─ bounded revision round → deterministic re-check
        └─ revision that still fails is REJECTED; the original is kept
```

LLM judges are fallible and slow, so exact checks run first. If the critic
is unreachable the report says `critic_unavailable` instead of pretending
verification happened.

### 4.5 Planner–Executor with dependency-aware scheduling (networkx)

The LLM is the **planner** (emits JSON validated against `Plan`), the
`PlanExecutor` is the **executor** (wave-ordered via
`nx.topological_generations`, cycles caught by `nx.find_cycle`). Each node
runs as its own bounded sub-task (react/cot/direct), and the synthesis call
folds node outputs into one answer. Invalid plans degrade to a single-node
fallback instead of failing.

### 4.6 Protocol adapter + safety decorator

`ToolProvider` hides where tools live; `GatedToolProvider` adds least
privilege, approval, sanitization, and audit *without touching backends*.
Reasoning patterns never know if a call went in-process or over MCP.

### 4.7 Fail-safe defaults

Network calls return typed errors, parsers raise `ParseError` with a
preview, plans fall back, budgets stop spending, persistence failures only
warn, and `PlanningTaskOutput` is always returned — the agent degrades, it
does not crash.

---

## 5. Security model (defense at every boundary)

| Boundary | Threat | Mitigation (where) |
|---|---|---|
| LLM → tools | unauthorized writes | `allow_writes=False` by default; gate blocks before the backend (`tools/provider.py`) |
| LLM → tools | out-of-bounds quantities/priorities | approval callback with `max_restock_quantity` and allowed priorities (`demo.make_write_approver`, evals runner) |
| Tool → LLM | prompt injection via tool output | `sanitize_untrusted` + length cap + `[BEGIN/END UNTRUSTED TOOL DATA]` markers |
| LLM → answer | hallucinated numbers | grounding check vs evidence; critic + revision; revision rejection if it still fails |
| LLM → plan | cycles / oversized / impossible plans | pydantic shape validation, networkx cycle detection, node/wave caps, real-tool filtering, single-node fallback |
| LLM → loop | infinite tool loops / format drift | repeat-action detection, format reminders, step caps |
| Cost | runaway tokens | per-run `token_budget`, per-call retry caps, planner caps |
| Secrets | keys in logs/prompts | keys live only in the gateway `.env`; shared logging redacts |
| Persistence | audit gaps | every tool call yields a `ToolExecutionAudit` + SQLite row, blocked or not |

---

## 6. Observability

- **Structured logs** — one JSON line per event on stderr:
  `route_decision`, `llm_call`, `tool_call`, `validation_done`, `plan_ready`,
  `plan_executed`, `revision_applied` / `revision_rejected`, `task_done`.
- **Traces** — `data/traces/<trace_id>.jsonl`: `llm.call` spans (label,
  model, tokens, latency, cached) and `tool.call` spans (blocked, ok).
  Query: `jq 'select(.name=="llm.call")' data/traces/<id>.jsonl`
- **Token usage** — `PlanningTaskOutput.usage` includes input/output/total,
  budget, remaining, calls, failures, and JSON repair count.
- **Execution history** — SQLite (`data/agent_state.db`): sessions, events
  (full audit trail), per-call tool audits, and memory
  (`last_answer`, `last_plan`).
- **Eval reports** — `data/evals/chapter02_report.json`.

---

## 7. Run it

```bash
cd learning_agentic_ai_with_ai
source .venv/bin/activate
export PYTHONPATH=src/learning_agentic_ai_with_ai

# 0) one-time: gateway API keys (edit the gateway's .env, unchanged from Ch.1)
#    src/learning_agentic_ai_with_ai/llm_gateway/.env

# 1) full offline demo (scripted LLM — no keys needed, Chapter 1 tools in-process)
python -m chapter02_planning.demo --scenario all --mock --tools inproc

# 2) individual scenarios
python -m chapter02_planning.demo --scenario direct --mock   # unit conversion
python -m chapter02_planning.demo --scenario cot    --mock   # restock sizing math
python -m chapter02_planning.demo --scenario react  --mock   # stock + trend lookup
python -m chapter02_planning.demo --scenario plan   --mock   # multi-store campaign
python -m chapter02_planning.demo --scenario unsafe --mock   # approval bounds block 9000
python -m chapter02_planning.demo --scenario plan --mock --read-only   # least privilege

# 3) live mode: real models through YOUR gateway, real MCP stdio servers
python -m chapter02_planning.demo --scenario all --live --tools mcp

# 4) evaluation harness
python -m chapter02_planning.evals.runner --mock                # deterministic
python -m chapter02_planning.evals.runner --mock --tools mcp    # via MCP servers
python -m chapter02_planning.evals.runner --live --tools mcp    # through your gateway

# 5) tests (unit + integration, incl. MCP provider over real stdio)
python -m pytest tests -q
```

Expected demo output (mock / inproc):
```
scenario : plan   mode=mock  backend=inproc  writes=on
route    : plan   status=completed
waves    : [['inspect_stock'], ['review_trend'], ['place_order']]
  [completed] inspect_stock: Low stock report for store S02: R-401: on_hand=18 ...
  [completed] review_trend: Sales trend for S02 R-401: avg_daily_units=0.29 ...
  [completed] place_order: Restock order #77 created: 5 units of R-401 ...
validation: passed=True revised=False issues=0
usage    : {'input_tokens': 12362, 'output_tokens': 632, 'total_tokens': 13021, ...}
```

Expected eval output:
```
=== Chapter 2 evaluation report ===
cases: 6/6 passed
task success rate: 100%
safety failures: 0
reliability failures: 0
```

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `AGENTIC_PLAN_TEMPERATURE` | `0.2` | Base sampling temperature |
| `AGENTIC_PLAN_COT_TEMPERATURE` | `0.3` | CoT sampling (self-consistency) |
| `AGENTIC_PLAN_REACT_TEMPERATURE` | `0.0` | ReAct turns |
| `AGENTIC_PLAN_MAX_STEPS` | from `Settings` | ReAct step cap |
| `AGENTIC_PLAN_MAX_NODES` / `_WAVES` | `8` / `6` | Plan caps |
| `AGENTIC_PLAN_TOKEN_BUDGET` | from `Settings` | Tokens per run |
| `AGENTIC_PLAN_COT_SAMPLES` | `2` | Self-consistency samples |
| `AGENTIC_PLAN_VALIDATION_ROUNDS` | `1` | Revision rounds |
| `AGENTIC_PLAN_CRITIC` / `_ALWAYS` | `true` / `false` | LLM critic policy |
| `AGENTIC_PLAN_GROUNDING_COVERAGE` | `0.6` | Min grounded-number fraction |
| `AGENTIC_PLAN_REQUIRE_WRITE_APPROVAL` | from `Settings` | Writes need approval |
| `AGENTIC_PLAN_LLM_ATTEMPTS` / `_RETRY_BASE_S` | `3` / `0.5` | Retry policy |
| `AGENTIC_PLAN_MAX_RESULT_CHARS` / `_SANITIZE` | from `Settings` / `true` | Observation hygiene |

### Failure modes (what happens when things go wrong)

| Failure | Behavior |
|---|---|
| Provider down | retries with jitter → typed error → run finishes `error` |
| Empty completion (reasoning model) | one same-prompt retry, then repair/degrade |
| Model output not JSON / wrong plan shape | one repair call; shape synonyms normalized; else safe fallback |
| CoT missing `Answer:` | one format-repair; else `needs_review` |
| ReAct format drift / loops | reminders then stop; third identical call stops the loop |
| Plan cycle / oversize | rejected; single-node fallback plan still attempts the task |
| Plan node fails | dependents `skipped`; independent nodes complete; validator requires acknowledgment |
| Revision that ignores failures | rejected; original answer kept, status `needs_review` |
| Budget exceeded | spending stops; partial evidence preserved; status `error` |
| Blocked write | observation says BLOCKED; answer must acknowledge or validation fails |

---

## 8. The journey forward: reasoning as layer 2

```
┌──────────────────────────────────────────────┐
│ AG-UI / A2UI     rich agent⇄user streaming UI│  ← future chapters
├──────────────────────────────────────────────┤
│ A2A              agent⇄agent delegation      │  ← future chapters
├──────────────────────────────────────────────┤
│ REASONING        plan · reason · validate    │  ← THIS chapter
├──────────────────────────────────────────────┤
│ MCP              agent⇄tools                 │  ← Chapter 1
└──────────────────────────────────────────────┘
```

Chapter 1 gave the agent hands; Chapter 2 gives it a thinking strategy that
uses those hands safely. Later chapters (memory, multi-agent delegation,
supervision) will reuse this chapter's pieces directly: the router decides
*who* should think, ReAct is how a worker acts, the planner DAG is how a
supervisor splits work, and the validator is how results are accepted.

---

## 9. Exercises

1. **Add a route.** Implement a `debate` pattern (two models argue, a third
   decides) as a new module + one branch in `PlanningAgent`, then add a
   routing rule and an eval case.
2. **Parallelize waves.** `topological_waves` already computes concurrency
   groups — replace the sequential wave loop with a thread pool and measure
   wall time on the plan scenario.
3. **Stricter grounding.** Extend `check_grounding` to accept numbers
   *derived* from evidence (sums/differences) and re-run the evals.
4. **Critic as a second model.** Point the critic at a different gateway
   alias than the generator and compare verdicts on the unsafe case.
5. **Plan revision.** On node failure, re-prompt the planner with the failure
   and completed results to produce a `PlanRevision`, then execute only the
   new nodes.
6. **Run `--live`** with your own key and compare eval scores vs mock mode.
   Notice how routing stays deterministic while pattern quality tracks the
   model.
