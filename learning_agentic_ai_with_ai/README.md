# Learning Agentic AI — from scratch (no LangChain / LangGraph)

Course-style implementation of agentic AI systems **from first principles**.
All LLM traffic flows through the local `llm_gateway` (multi-provider router
with retries, caching, rate limiting, and an Ollama fallback) — no provider
SDKs are called directly. No agentic frameworks are used; only general-purpose
libraries (pydantic, httpx, SQLite, stdlib).

## Layout

```
src/learning_agentic_ai_with_ai/
├── llm_gateway/       YOUR LLM gateway (pre-existing; see its own code)
├── agentic_common/    shared foundation: settings, logging, tracing,
│                      SQLite persistence, gateway client, security, eval harness
├── chapter01_mcp/     Chapter 1 — MCP: the tool protocol
├── chapter02_planning/ Chapter 2 — planning, reasoning, structured prompting
└── chapter03_cognition/ Chapter 3 — cognitive architecture & adaptive planning
```

## Setup

```bash
cd learning_agentic_ai_with_ai
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
export PYTHONPATH=src/learning_agentic_ai_with_ai

# add provider keys to the gateway env once:
#   src/learning_agentic_ai_with_ai/llm_gateway/.env  (see .env.example there)
```

## Chapter 1 — MCP (see `src/learning_agentic_ai_with_ai/chapter01_mcp/docs/README.md`)

Quickstart:

```bash
export PYTHONPATH=src/learning_agentic_ai_with_ai
python -m chapter01_mcp.demo --scenario restock --mock   # offline demo
python -m chapter01_mcp.demo --scenario restock --live   # via your LLM gateway
python -m chapter01_mcp.evals.runner --mock              # evaluation harness
python -m pytest tests -q                                 # unit + integration
```

Documentation: [`chapter01_mcp/docs/README.md`](src/learning_agentic_ai_with_ai/chapter01_mcp/docs/README.md)

## Chapter 2 — Planning, Reasoning & Structured Prompting (see `src/learning_agentic_ai_with_ai/chapter02_planning/docs/README.md`)

Covers chain-of-thought, ReAct, structured prompting, self-validation,
dependency-aware task decomposition (networkx), and when to let the model
reason vs. enforce structure. Reuses the Chapter 1 retail/telecom tools via
an adapter (`--tools inproc|mcp`) and routes every LLM call through your
gateway.

Quickstart:

```bash
export PYTHONPATH=src/learning_agentic_ai_with_ai
python -m chapter02_planning.demo --scenario all --mock --tools inproc  # offline
python -m chapter02_planning.demo --scenario all --live --tools mcp    # gateway + MCP
python -m chapter02_planning.evals.runner --mock                       # evals
python -m pytest tests -q                                               # unit + integration
```

Documentation: [`chapter02_planning/docs/README.md`](src/learning_agentic_ai_with_ai/chapter02_planning/docs/README.md)

## Chapter 3 — Cognitive Architecture & Adaptive Planning (see `src/learning_agentic_ai_with_ai/chapter03_cognition/docs/README.md`)

The 4-layer cognitive pipeline (Perception → Memory → Decision → Action)
with typed pydantic contracts, strategy profiles (conservative,
exploratory, fallback), agent-written Python plans executed in a
capability-based sandbox (restricted or subprocess), bounded adaptive retry
and tool switching, and memory that learns from every run. Reuses the
Chapter 1 retail/telecom tools; every LLM call goes through your gateway.

Quickstart:

```bash
export PYTHONPATH=src/learning_agentic_ai_with_ai
python -m chapter03_cognition.demo --scenario all --mock --fresh-memory  # offline
python -m chapter03_cognition.demo --scenario all --live --tools mcp    # gateway + MCP
python -m chapter03_cognition.evals.runner --mock                       # evals
python -m pytest tests -q                                               # unit + integration
```

Documentation: [`chapter03_cognition/docs/README.md`](src/learning_agentic_ai_with_ai/chapter03_cognition/docs/README.md)
