# Chapter 1 — MCP: The Tool Protocol

> Master the **Model Context Protocol** for tool registration, discovery, and invocation
> across servers — built from scratch, no LangChain/LangGraph, no agent frameworks.
> All LLM calls flow through **your local LLM gateway** (in-process import).

---

## 1. The journey: Foundation LLMs → Function Calling → Agentic AI → MCP

To understand *why* MCP exists, follow the problems each era solved:

```
Foundation LLMs (2020+)
  └─ Problem: the model only knows its training data. It cannot read your
     database, call your APIs, or check today's data.
  └─ Fix: **Retrieval / prompt engineering** — paste information into the prompt.
     Still limited: the model can READ text but cannot ACT.

Function Calling (2023)
  └─ The provider adds: "model may reply with a structured request to run a
     function". The application executes it and returns the result.
     Now the model can act — but every action must be hand-wired into YOUR
     application. Tool #37 means editing app code. Every app re-implements
     the same tools (get_weather exists 10,000 times in the world's codebases).

Agentic AI (2023+)
  └─ Put a LOOP around function calling: model → tools → results → model …
     until done. Now tools are chosen dynamically by the model, not by the
     developer. But the tools are still welded to one app.

MCP (Nov 2024, Anthropic)
  └─ The "USB-C for AI tools". Instead of each app implementing each tool,
     tools live in STANDALONE SERVERS speaking one open protocol.
     Any MCP client (Claude Desktop, your agent, an IDE) can use any MCP
     server. Tool vendors write their integration ONCE.
```

**MCP solves tool distribution, not intelligence.** The LLM still decides
*which* tool to call; MCP standardizes how tools are *listed*, *described*,
and *called* across process and language boundaries.

### The protocol stack (new view)

```
┌────────────────────────────────────────────────────────────┐
│ A2UI / AG-UI        Agent ⇄ User interfaces (streaming UI) │  ← presentation
├────────────────────────────────────────────────────────────┤
│ A2A                 Agent ⇄ Agent (peer delegation)        │  ← future chapter
├────────────────────────────────────────────────────────────┤
│ MCP                 Agent ⇄ Tools/Data (THIS chapter)      │  ← layer 1
└────────────────────────────────────────────────────────────┘
```

MCP is the **foundation layer**: an agent needs tools before it can delegate
to other agents (A2A) or render rich UIs. Every later chapter builds on it.

---

## 2. Architecture

### 2.1 Big picture

```
┌────────────────────────────── Agent process ──────────────────────────────┐
│                                                                            │
│   OpsAgent (tool-use loop)                                                 │
│      │                ┌──────────────┐                                     │
│      │ every LLM hop  │ LLM Gateway  │  ← your existing gateway            │
│      ▼                │ (in-process) │    (providers, retries, cache,      │
│   ┌─────────┐         └──────────────┘     Ollama fallback, logs)          │
│   │ Planner │                                                             │
│   └────┬────┘                                                             │
│        │ tool_calls (OpenAI format)                                       │
│   ┌────▼─────────┐   policy gate (schema, limits, approval)                │
│   │ MCP Toolbox  │──────────────────────────────┐                         │
│   └──┬────────┬──┘                               │                        │
└──────┼────────┼──────────────────────────────────┼────────────────────────┘
       │stdio   │stdio                             │SSE (HTTP)
┌──────▼─────┐ ┌▼──────────┐              ┌────────▼─────────┐
│ retail-ops │ │telecom-ops│              │ any MCP server   │
│ subprocess │ │subprocess │              │ (another host)   │
└────────────┘ └───────────┘              └──────────────────┘
```

### 2.1 Server/Client roles

| Role | In this course | Responsibility |
|---|---|---|
| **MCP Server** | `chapter01_mcp/server/*` + `servers/*` | Registers tools; answers JSON-RPC; executes business logic |
| **MCP Client** | `chapter01_mcp/client/*` | Handshake, tool discovery, invocation over a transport |

The same code can act as *host* (agent), *client* (protocol session), and
*server* (tool provider) — keep the roles separate in your head.

### 2.2 Transports

| | stdio | SSE |
|---|---|---|
| Where the server runs | child subprocess | HTTP server (any host) |
| Requests | stdin lines | HTTP POST |
| Responses | stdout lines | HTTP body (sync) + SSE stream |
| Best for | local tools, CLIs | shared/remote servers |
| Security | OS process isolation | network security applies |

**Iron rule for stdio servers:** stdout carries *only* protocol lines. All
logs go to stderr. Mixing them corrupts the session — this is the #1 stdio
bug in practice.

### 2.3 Wire format (JSON-RPC 2.0)

```
Client → Server :  {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
Server → Client :  {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",...}}
Client → Server :  {"jsonrpc":"2.0","method":"notifications/initialized"}   ← no id!
Client → Server :  {"jsonrpc":"2.0","id":2,"method":"tools/list"}
Server → Client :  {"jsonrpc":"2.0","id":2,"result":{"tools":[...]}}
Client → Server :  {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"t","arguments":{...}}}
```

Key MCP semantic (implemented in `server_core.py`):
- **Tool-level failure** (bad usage, business error) → *normal result* with
  `isError: true`. The model can read the error and recover.
- **Protocol-level failure** (unknown method, malformed JSON) → JSON-RPC error.

---

## 3. What we built (file map)

```
chapter01_mcp/
├── jsonrpc.py              hand-rolled JSON-RPC 2.0 codec
├── mcp_protocol.py         MCP method names, pydantic→JSON-Schema bridge
├── server_core.py          transport-agnostic server (registry + dispatch)
├── server/stdio.py         stdio transport (newline-delimited JSON on pipes)
├── server/sse.py           SSE transport (stdlib HTTP + SSE stream)
├── servers/
│   ├── ops_db.py           SQLite mock data (retail + telecom, seeded)
│   ├── retail_server.py    3 tools (2 read + 1 write)
│   ├── telecom_server.py   3 tools (2 read + 1 write)
│   ├── retail_main.py      stdio entrypoint   (python -m ...retail_main)
│   ├── telecom_main.py     stdio entrypoint
│   └── retail_sdk_server.py SAME tools via official mcp SDK (FastMCP)
├── client/
│   ├── transports.py       StdioClientTransport + SseClientTransport
│   ├── mcp_client.py       session lifecycle, tools/list, tools/call, retries
│   ├── registry.py         server catalog + dynamic discovery (summary+hints)
│   └── policy.py           ToolPolicyEngine (schema, limits, approval)
├── agent/
│   ├── orchestrator.py     session manager + policy wiring
│   ├── tools_bridge.py     gateway tool format ⇄ MCP, execution + audit
│   └── ops_agent.py        THE TOOL-USE LOOP
├── evals/cases.yaml        evaluation cases
├── evals/runner.py         evaluation harness runner
├── demo.py                 end-to-end demo (mock + live modes)
└── typescript/             official TS SDK server (cross-language interop)
```

---

## 4. Design patterns in this chapter

### 4.1 Tool-Use Loop (the core agentic pattern)

```
loop (max N iterations):
  1. messages → LLM (through gateway, with tool schemas)
  2. LLM returns either:
       final answer              → stop, return
       tool_calls[]              → for each call:
           a. policy gate: schema-valid? limits? approved?
       3. execute via MCP (tools/call)
       4. append result as role="tool" message
  5. budget checks (iterations, tokens) → stop safely
```
Every iteration is ONE gateway completion. The loop is the agent; everything
else (memory, policy, discovery) is infrastructure around it.

### 4.2 Planner-Executor split
The LLM is the **planner** (decides *what* to do). MCP servers are the
**executor** (know *how*). Our `MCPServerCore` never calls an LLM; the loop
never touches business data. You can swap either side.

### 4.3 Protocol adapter (anti-corruption layer)
`tools_bridge.py` translates between three vocabularies:
- OpenAI-style tools (gateway) ⇄ `ToolDescriptor` (MCP) ⇄ pydantic models (server).
- Qualified names `server.tool` prevent collisions across servers.

### 4.4 Policy Enforcement Point (security pattern)
All authority checks live in ONE object (`ToolPolicyEngine`) sitting between
the LLM and the tools. The LLM is *untrusted input*: its tool arguments are
validated against the advertised JSON Schema **before** any execution; write
tools require the approval callback; results are sanitized before re-entering
the prompt (prompt-injection surface reduction).

### 4.5 Discovery: summary layer + hint filtering
You never dump every tool of every server into the prompt:
1. **Summary layer**: each server advertises a one-line `summary` + `hints`.
2. `pick_servers(task)` scores servers against the task text (hints ×3 + summary overlap).
3. `pick_tools` filters a picked server's tools when it has many.
This keeps prompts small and tool choice reliable — the same idea behind
RAG but for *tools* instead of documents.

---

## 5. Security model (defense at every boundary)

| Boundary | Threat | Mitigation (where) |
|---|---|---|
| LLM → args | malformed/injected arguments | JSON-Schema validation in server (`pydantic`) **and** client policy (`policy.py`) |
| LLM → tools | unauthorized writes | write-tool registry + approval callback with caps (`orchestrator.ops_policy_engine`) |
| Tool → LLM | prompt injection via tool output | `sanitize_untrusted` + length caps (`agentic_common/security.py`) |
| Secrets | keys in logs/traces | `redact_secrets` in logging; keys only in gateway `.env` |
| Subprocess | runaway/broken servers | timeouts, SIGTERM→SIGKILL teardown, stderr ring buffer for diagnostics |
| Protocol | malformed lines | PARSE_ERROR with `id: null`; server never crashes (fuzz-tested) |

---

## 6. Observability

- **Structured logs** — one JSON line per event on stderr (`agentic_common/logging.py`).
- **Traces** — `data/traces/<trace_id>.jsonl`: one span per LLM call / tool call
  with attributes, latency, status, token usage. Query with jq:
  `jq 'select(.name=="tool.call")' data/traces/<id>.jsonl`
- **Token usage** — every gateway response carries usage; the loop accumulates
  it and enforces `AGENTIC_TOKEN_BUDGET`. Tool results are also budgeted (chars/4).
- **Execution history** — SQLite (`data/agent_state.db`): sessions, events,
  tool-call audits (server, tool, args, ok, approved, latency, error).
- **Gateway's own DB** — `data/gateway.db` logs every provider call (tokens, cost, latency, status).

---

## 7. Run it

```bash
cd learning_agentic_ai_with_ai
source .venv/bin/activate
export PYTHONPATH=src/learning_agentic_ai_with_ai

# 0) one-time: gateway API keys (edit the gateway's .env)
#    src/learning_agentic_ai_with_ai/llm_gateway/.env

# 1) seed + inspect the mock ops database
python -c "from chapter01_mcp.servers.ops_db import seed_if_empty, demo_summary; seed_if_empty(); print(demo_summary())"

# 2) full demo (offline, scripted LLM — no keys needed)
python -m chapter01_mcp.demo --scenario restock --mock
python -m chapter01_mcp.demo --scenario telecom --mock
python -m chapter01_mcp.demo --scenario unsafe  --mock   # policy blocks the write

# 3) live mode: LLM calls go through YOUR gateway (real models, real tool calls)
python -m chapter01_mcp.demo --scenario restock --live

# 4) evaluation harness
python -m chapter01_mcp.evals.runner --mock     # deterministic
python -m chapter01_mcp.evals.runner --live     # through your gateway

# 5) talk to a server manually (see the wire protocol live)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | \
  PYTHONPATH=src/learning_agentic_ai_with_ai .venv/bin/python -m chapter01_mcp.servers.retail_main

# 6) tests (unit + integration, incl. Python↔TypeScript interop)
python -m pytest tests -q
```

Expected demo output:
```
scenario : restock   (mode=mock)
status   : completed   iterations=4
answer   : Restock order #1 created: 5 units of R-101 for store S01 ...
tools    :
  [OK  ] retail-ops.retail_low_stock_report ...
  [OK  ] retail-ops.retail_sales_trend ...
  [OK  ] retail-ops.retail_restock_order ...
usage    : {'input_tokens': 689, 'output_tokens': 45, 'total_tokens': 746}
```

---

## 8. The journey forward: MCP as layer 1

```
┌─────────────────────────────────────────────┐
│ AG-UI / A2UI    rich agent⇄user streaming UI│  ← future chapters
├─────────────────────────────────────────────┤
│ A2A             agent⇄agent delegation      │  ← future chapters
├──────────────────────────────────────────────┤
│ MCP             agent⇄tools   (THIS chapter) │
└──────────────────────────────────────────────┘
```
MCP is the **foundation layer**: it gives the agent hands (tools). Once tools
exist, agents can call *other agents* (A2A — an agent is just another MCP
client with its own tools), and surfaces (AG-UI) can stream that activity to
users. Every later pattern in this course (planner-executor, reflection,
supervisor-worker) sits *on top of* the tool-use loop you built here.

---

## 9. Exercises

1. Add a `retail_price_check` tool to the retail server (read-only) and a unit
   test for it. Notice: zero client changes — discovery picks it up.
2. Add a new server (`hr-ops`) with one tool; add hints to its descriptor;
   verify `pick_servers` routes tasks to it.
3. Change the approval callback to a CLI prompt (`input()`) and observe the
   unsafe scenario interactively.
4. Break the stdio server on purpose (print a log line to stdout) and watch
   the client fail — then fix it. You'll never forget the stdio rule.
5. Run `--live` with a real key and compare eval scores vs mock mode.
