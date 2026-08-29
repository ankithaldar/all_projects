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
└── chapter01_mcp/     Chapter 1 — MCP: the tool protocol
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
