# Chapter 3 — Cognitive Architecture & Adaptive Planning

> Give the agent an **architecture**: four typed layers (Perception →
> Memory → Decision → Action), strategy profiles (Conservative,
> Exploratory, Fallback), agent-written Python plans in a capability-based
> sandbox, and adaptive retry loops that learn from every run.
> Built from scratch, no LangChain/LangGraph, no agent frameworks.
> All LLM calls flow through **your local LLM gateway** (in-process import).
> Tools are Chapter 1's retail/telecom MCP servers; `llm_gateway` is
> untouched.

---

## 1. The journey: loop → reasoning → planning → cognition

Each chapter solved the previous chapter's missing organ:

```
Chapter 1 — hands (MCP tools)
  └─ Problem: the tool-use loop acts, but has no model of the task, no
     memory, and no way to recover from a bad plan.

Chapter 2 — thinking shapes (CoT, ReAct, validation, DAG plans)
  └─ Problem: patterns were selected per task, but there was no persistent
     architecture: no perception layer, no learning, and no principled way
     to change approach after failure.

Chapter 3 — cognition (THIS chapter)
  └─ Perception turns raw text into typed signals; Memory recalls past runs,
     facts, and strategy performance; Decision selects a strategy, plans,
     and INTROSPECTS the plan; Action executes with bounded retry and tool
     switching; Reflection writes the episode back to memory so the next
     run starts smarter.

  └─ Plus: the model can write its own solve() functions - executed in a
     sandbox with capability-based tool access, because generated code is
     untrusted input with side effects.
```

**Architecture beats prompting.** A fixed pipeline with typed seams lets
you test each layer, swap planners, bound every loop, and explain every
decision after the fact.

### The stack (new view)

```
┌──────────────────────────────────────────────────────────────┐
│ A2A / AG-UI     agent ⇄ agent, agent ⇄ user                  │  ← future chapters
├──────────────────────────────────────────────────────────────┤
│ COGNITION       perceive · remember · decide · act · reflect │  ← THIS chapter
├──────────────────────────────────────────────────────────────┤
│ REASONING       CoT · ReAct · validation · DAG plans         │  ← Chapter 2
├──────────────────────────────────────────────────────────────┤
│ MCP             agent ⇄ tools / data                         │  ← Chapter 1
├──────────────────────────────────────────────────────────────┤
│ LLM GATEWAY     providers, retries, cache, rate limits       │  ← your existing gateway
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture

### 2.1 The 4-layer pipeline

```
       task
        │
        ▼
┌───────────────┐                ┌───────────────┐  MemoryContext
│ PERCEPTION    │                │ MEMORY        │
│ signals       │ ── Percept ──▶ │ episodic      │
│ entities      │                │ semantic      │
│ capabilities  │                │ procedural    │
└───────────────┘                └───────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ DECISION            │
                              │ strategy select     │
                              │ plan (dag/code)     │
                              │ introspection       │
                              └─────────────────────┘
                                         │ Decision
                                         ▼
                              ┌─────────────────────┐
                              │ ACTION              │
                              │ DAG waves or        │
                              │ sandboxed code      │
                              │ retry · switch      │
                              └─────────────────────┘
                                         │ ActionResult
                                         ▼
                 reflection: episode + strategy stats + facts
                 (memory improves the NEXT run's planning)
```

Layer contracts (all pydantic, all in `schemas.py`):

| Layer | In | Out | Responsibility |
|---|---|---|---|
| Perception | task text | `Percept` | domain, risk, ambiguity, data need, entities, required capabilities - **no LLM call** |
| Memory | `Percept` | `MemoryContext` | relevant episodes, facts, strategy stats, recall notes, recommended strategy |
| Decision | `Percept` + `MemoryContext` + `StrategyProfile` | `Decision` | strategy selection, plan generation (DAG or code), plan introspection |
| Action | `Decision` | `ActionResult` | execution, retry, tool switching, synthesis, audits |

### 2.2 Strategy profiles (how boldly the agent acts)

| Profile | Attempts/step | Tool switch | Replans | Writes | Code | Used when |
|---|---|---|---|---|---|---|
| **conservative** | 1 | no | 0 | approved only | yes | writes, risk, default for simple tasks |
| **exploratory** | 3 | yes | 1 | approved only | yes | ambiguity, complexity, memory says so |
| **fallback** | 1 | no | 0 | **no** | **no** | primary strategies struggling; degraded answer |

Selection is ordered and explainable: **hint > write/risk > memory
recommendation > ambiguity/complexity > default**.

### 2.3 Two nested adaptive loops

```
INNER (ActionLayer / AdaptiveLoop) - inside one plan execution

    attempt ──ok──▶ done
      │
    fail ──▶ classify: blocked/invalid_arguments/code_error ─▶ stop
             unknown_tool            ─▶ try fallback tool
             transient/empty/timeout ─▶ retry up to the cap
      │
      └─ attempts exhausted + fallback_tool ─▶ TOOL SWITCH
      └─ still failing ─▶ failed (dependents skip, independents run)

OUTER (CognitiveAgent) - across plan executions

    plan fails ──▶ strategy allows escalation? ──▶ conservative → exploratory → fallback
                   │                                      │
                   └─ re-plan once per escalation, bounded by max_replans
                   └─ every escalation recorded as an AdaptationRecord
```

### 2.4 Agent-written Python plans

```
LLM writes:  def solve(context):
                 data = context["tools"].call("retail-ops.retail_low_stock_report", {})
                 qty = 30 * 4 + 10 - 20
                 return {"answer": f"Order {qty} units", "used_tools": [...]}

Static validation (ast): no imports, no classes, no dunder access, no
exec/eval/open/getattr, exactly one `solve(context)`.

                 ┌──────────────────────┐        ┌───────────────────────┐
                 │ RestrictedExecRunner │  or    │  SubprocessRunner     │
                 │ in-process           │        │ child process -I -S   │
                 │ allowlisted builtins │        │ CPU/fd limits         │
                 │ ToolCaller capability│        │ hard timeout          │
                 │ cooperative deadline │        │ tools UNAVAILABLE     │
                 └──────────────────────┘        └───────────────────────┘
```

`ToolCaller` enforces the tool allowlist, write policy, a call budget, and
a deadline - generated code never receives a raw function or module.

---

## 3. What we built (file map)

```
chapter03_cognition/
├── schemas.py          every layer contract, strategy, plan, adaptation, IO
├── config.py           CognitionConfig + AGENTIC_COG_* env loading
├── jsonio.py           defensive JSON/code extraction (self-contained)
├── prompts.py          PromptTemplate ($slots), output contracts, all prompts
├── perception.py       Layer 1: deterministic signal + entity extraction
├── memory.py           Layer 2: MemoryStore (SQLite) + MemoryLayer
├── strategies.py       profiles + explainable StrategySelector
├── planner.py          Layer 3: DagPlanner, CodePlanner, PlannerIntrospector
├── codegen.py          sandbox: AST validation, ToolCaller, 2 runners
├── adaptive.py         error classification + bounded AdaptiveLoop
├── action.py           Layer 4: DAG/code execution, retry/switch, synthesis
├── agent.py            CognitiveAgent: pipeline + outer escalation + reflection
├── tools.py            Chapter 1-backed ToolBox + FaultInjector
├── llm.py              gateway facade: retries, budget, spans, JSON repair
├── demo.py             end-to-end demo (mock/live, inproc/mcp, runners)
├── evals/
│   ├── cases.py        CognitionEvalCase + chapter-specific scoring
│   ├── cases.yaml      8 retail/telecom eval scenarios
│   └── runner.py       evaluation harness runner
└── docs/README.md      this document
```

Chapter 1 reuse: `tools.py` imports `build_retail_server()` /
`build_telecom_server()` and `MCPServerCore.handle_message` for the
in-process backend, and `ServerCatalog` + `MCPSessionManager` for MCP.
Chapter 3 does not import Chapter 2 (standalone decision layer) but both
chapters share `agentic_common` and the gateway.

---

## 4. Design patterns in this chapter

### 4.1 Layered architecture with typed seams

Each layer has one pydantic input and one pydantic output; nothing crosses
a boundary as a raw dict. You can unit test Perception without a gateway,
Memory without a planner, and the sandbox without a model.

### 4.2 Strategy pattern (operating policies)

`StrategyProfile` bundles the knobs that must move together (attempts,
switching, replans, writes, code, temperature, step cap). The selector
picks a profile from percept + memory; the executor reads the profile - it
never hardcodes policy.

### 4.3 Planner–executor + plan introspection

The planner emits a validated DAG or a validated code function; the
executor runs it in topological waves. Before execution,
`PlannerIntrospector` reviews the plan like a senior engineer: unknown
tools, write-before-read ordering, capability coverage, cycle/size errors,
risk level, and confidence.

### 4.4 Capability-based sandbox (policy enforcement point)

Generated code is untrusted. It gets a `ToolCaller` object, not the
toolbox: allowlist, write policy, call budget, deadline. Two runners behind
one `CodeRunner` interface trade isolation (subprocess) for capability
(tools).

### 4.5 Reflection / episodic learning

After every run, `_reflect` writes an `Episode` (strategy, status, tool
sequence, error classes, adaptations, tokens), updates per-strategy stats,
and stores a `last_answer` fact. The next run recalls similar episodes and
uses their outcomes to recommend a strategy - learning without embeddings
and without a training loop.

### 4.6 Bounded adaptation (retry with circuit-breaker semantics)

Errors are classified (`transient`, `unknown_tool`, `blocked`,
`invalid_arguments`, `timeout`, `code_error`, ...). Only retryable classes
retry; blocked/invalid are terminal; unknown tools switch tools; repeated
failure escalates strategy. Every adaptation is an `AdaptationRecord`, so
runs explain themselves.

### 4.7 Fail-safe defaults

Parsers raise typed errors, planners fall back to a one-step reason plan,
sandboxes refuse unsafe code before execution, budgets stop spending,
reflection failures only warn, and `CognitiveTaskOutput` is always
returned.

---

## 5. Security model (defense at every boundary)

| Boundary | Threat | Mitigation (where) |
|---|---|---|
| LLM → tools | unauthorized writes | `ToolBox` gate: `allow_writes` + approval callback (`tools.py`) |
| LLM → plan | impossible tools / write-before-read | normalization + `PlannerIntrospector` warnings (`planner.py`) |
| LLM → code | malicious generated Python | AST denylist + allowlisted builtins + `ToolCaller` (`codegen.py`) |
| Generated code → tools | capability escalation | allowlist, call budget, deadline, write policy in `ToolCaller` |
| Generated code → OS | filesystem/network/process abuse | subprocess runner: `-I -S`, scrubbed env, CPU/fd limits, hard timeout |
| Tool → LLM | prompt injection via tool output | `sanitize_untrusted` + `[BEGIN/END UNTRUSTED TOOL DATA]` wrappers |
| LLM → answer | invented numbers / unverified claims | fallback answers carry explicit caveats; writes acknowledged via blocked audits |
| Memory | poisoned or stale facts | facts are namespaced by domain/session; episodes record provenance |
| Cost | runaway loops/tokens | per-step attempts, replans, token budget, code call budget |
| Secrets | keys in logs/prompts | keys live only in the gateway `.env`; shared logging redacts |

---

## 6. Observability

- **Structured logs** - one JSON line per event on stderr:
  `percept_ready`, `memory_recall`, `strategy_selected`, `plan_ready`,
  `plan_introspected`, `step_attempt_failed`, `fault_injected`,
  `action_done`, `episode_saved`, `task_done`.
- **Traces** - `data/traces/<trace_id>.jsonl`: `llm.call` spans (label,
  model, tokens) and `tool.call` spans (blocked/ok/latency).
- **Token usage** - `CognitiveTaskOutput.usage`: input/output/total, budget,
  remaining, calls, failures, JSON repair count.
- **Execution history** - SQLite (`data/agent_state.db`): sessions, events,
  tool-call audits, last-answer memory.
- **Cognitive memory** - SQLite (`data/chapter03_memory.db`): episodes,
  facts, strategy stats - query it to see what the agent learned:
  `sqlite3 data/chapter03_memory.db 'select strategy, runs, successes from strategy_stats'`
- **Eval reports** - `data/evals/chapter03_report.json`.

---

## 7. Run it

```bash
cd learning_agentic_ai_with_ai
source .venv/bin/activate
export PYTHONPATH=src/learning_agentic_ai_with_ai

# 0) one-time: gateway API keys (edit the gateway's .env, unchanged from Ch.1)
#    src/learning_agentic_ai_with_ai/llm_gateway/.env

# 1) all scenarios offline (scripted LLM; Chapter 1 tools in-process)
python -m chapter03_cognition.demo --scenario all --mock --fresh-memory

# 2) individual scenarios
python -m chapter03_cognition.demo --scenario perception   --mock   # layer 1 output
python -m chapter03_cognition.demo --scenario dag_retail   --mock   # JSON DAG plan
python -m chapter03_cognition.demo --scenario code_restock --mock   # agent-written solve()
python -m chapter03_cognition.demo --scenario adaptive     --mock   # retry + tool switch
python -m chapter03_cognition.demo --scenario fallback     --mock   # degraded answer
python -m chapter03_cognition.demo --scenario unsafe       --mock   # write blocked

# 3) sandbox choice for code plans
python -m chapter03_cognition.demo --scenario code_restock --mock --runner restricted
python -m chapter03_cognition.demo --scenario code_restock --mock --runner subprocess

# 4) live mode: real models through YOUR gateway, real MCP stdio servers
python -m chapter03_cognition.demo --scenario all --live --tools mcp

# 5) evaluation harness
python -m chapter03_cognition.evals.runner --mock                # deterministic
python -m chapter03_cognition.evals.runner --mock --tools mcp    # via MCP servers
python -m chapter03_cognition.evals.runner --live --tools mcp    # through your gateway

# 6) tests (unit + integration, incl. sandbox and MCP provider)
python -m pytest tests -q
```

Expected demo output (mock, code plan):
```
scenario : code_restock   mode=mock  backend=inproc
percept  : domain=retail risk=0.00 ambiguity=0.00 data_need=0.00 multi_step=0.12
entities : {'quantities': ['30', '4', '20', '10']}
strategy : conservative  (single approved write, verify each step, no retries)
plan     : source=code steps=1 risk=low confidence=0.70
action   : status=completed
  [completed] run_code tool=code attempts=1 Order 110 units: 30 units/day x 4 days = 120 plus buffer 10 minus on hand 20.
answer   : Order 110 units: 30 units/day x 4 days = 120 plus buffer 10 minus on hand 20.
```

Expected eval output:
```
=== Chapter 3 evaluation report ===
cases: 8/8 passed
task success rate: 100%
safety failures: 0
reliability failures: 0
```

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `AGENTIC_COG_PLANNER_TEMPERATURE` | `0.0` | DAG planning |
| `AGENTIC_COG_CODE_TEMPERATURE` | `0.0` | Code generation |
| `AGENTIC_COG_REASON_TEMPERATURE` | `0.2` | Reason/synthesis steps |
| `AGENTIC_COG_MAX_TOKENS` / `_TOKEN_BUDGET` | from `Settings` | LLM caps |
| `AGENTIC_COG_MAX_STEPS` | from `Settings` | Plan step cap |
| `AGENTIC_COG_MAX_ATTEMPTS` / `_MAX_REPLANS` | `2` / `1` | Adaptive caps |
| `AGENTIC_COG_CODE_ENABLED` | `true` | Allow code plans at all |
| `AGENTIC_COG_CODE_RUNNER` | `restricted` | Default sandbox |
| `AGENTIC_COG_CODE_TIMEOUT_S` | `5.0` | Code wall-clock budget |
| `AGENTIC_COG_CODE_CALL_BUDGET` | `8` | Tool calls from one code plan |
| `AGENTIC_COG_REQUIRE_WRITE_APPROVAL` | from `Settings` | Writes need approval |
| `AGENTIC_COG_MEMORY_DB` | `data/chapter03_memory.db` | Memory location |
| `AGENTIC_COG_MEMORY_RECALL` | `5` | Episodes recalled per task |
| `AGENTIC_COG_LLM_ATTEMPTS` / `_RETRY_BASE_S` | `3` / `0.5` | Retry policy |
| `AGENTIC_COG_SANITIZE` | `true` | Observation hygiene |

### Failure modes (what happens when things go wrong)

| Failure | Behavior |
|---|---|
| Gateway down | retries with jitter → typed error → run finishes `error` |
| Empty completion | one same-prompt retry, then repair/degrade |
| Bad plan JSON / cycle / oversize | single-step fallback plan still attempts the task |
| Unknown tool in a plan | converted to a reasoning step + introspection warning |
| Transient tool failure | retry within strategy cap; `retry` adaptation recorded |
| Persistent tool failure | tool switch to `fallback_tool` if the strategy allows |
| Blocked write | terminal for the step; run status `blocked`; answer says so |
| Unsafe generated code | refused before execution; one regeneration attempt; then fallback plan |
| Code timeout | restricted runner reports `timeout`; subprocess runner is killed |
| Code tries to use tools in subprocess | explicit `tools are unavailable` error |
| Plan fails entirely | outer loop escalates conservative → exploratory → fallback |
| Budget exceeded | spending stops; partial evidence preserved; status `error` |
| Reflection/memory write fails | warning only; the answer is unaffected |

---

## 8. The journey forward: cognition as layer 3

```
┌──────────────────────────────────────────────┐
│ AG-UI / A2UI     rich agent⇄user streaming UI│  ← future chapters
├──────────────────────────────────────────────┤
│ A2A              agent⇄agent delegation      │  ← future chapters
├──────────────────────────────────────────────┤
│ COGNITION        perceive·remember·decide·act│  ← THIS chapter
├──────────────────────────────────────────────┤
│ REASONING        CoT · ReAct · validation    │  ← Chapter 2
├──────────────────────────────────────────────┤
│ MCP              agent⇄tools                 │  ← Chapter 1
└──────────────────────────────────────────────┘
```

Chapter 3's layers become the building blocks for multi-agent work: each
worker is a `CognitiveAgent` with its own memory and strategy profile; the
supervisor's Decision layer delegates to workers' Action layers; episodes
from all workers form shared experiential memory; and the sandbox is how
agents safely run code written by other agents.

---

## 9. Exercises

1. **Add a layer.** Insert a `Critique` layer between Decision and Action
   that reviews the plan and can send it back once (bounded re-plan).
2. **Learn thresholds.** Replace the hardcoded ambiguity threshold with a
   value tuned from `strategy_stats` (e.g., explore more when the success
   rate of the default strategy drops).
3. **Fact extraction.** During reflection, ask the model to extract durable
   facts from the answer and store them with tags; measure recall quality on
   repeated tasks.
4. **Parallel waves.** Execute independent DAG steps in threads and compare
   wall time on `perception` (three-step plan).
5. **Stronger sandbox.** Add an import allowlist and a JSON-RPC bridge so
   the subprocess runner can call tools through the parent gate.
6. **Run `--live --tools mcp`** and compare eval scores and strategy
   statistics against mock mode.
