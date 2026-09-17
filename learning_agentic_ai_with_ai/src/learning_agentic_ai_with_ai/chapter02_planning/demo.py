#!/usr/bin/env python
# -- coding: utf-8 --

'''Chapter 2 end-to-end demo.

What this demonstrates:
  - the reasoning router picking a route per task (direct / cot / react /
    plan) with an explainable rationale,
  - all four patterns running end-to-end against the retail/telecom use
    cases, through YOUR llm_gateway,
  - tools served either in-process (Chapter 1 server cores) or over MCP,
  - the safety gate: write tools require `allow_writes` plus approval,
  - self-validation and the audit trail (SQLite + JSONL traces).

Modes:
  --mock (default)  scripted "LLM" drives the real parsing, routing, tool,
                    policy, validation, and persistence code paths offline.
  --live            every reasoning step goes through your llm_gateway.

Scenarios (retail & telecom operations):
  direct  : unit conversion - no reasoning needed
  cot     : restock sizing arithmetic - step-labeled reasoning
  react   : stock + sales-trend investigation - reason/act/observe
  plan    : multi-store replenishment campaign - dependency-aware DAG
  unsafe  : 9000-unit restock - approval bounds must block the write

Run:
  python -m chapter02_planning.demo --scenario all --mock --tools inproc
  python -m chapter02_planning.demo --scenario react --mock --read-only
  python -m chapter02_planning.demo --scenario plan --live --tools mcp
'''


from __future__ import annotations

import argparse
import json
import re
from typing import Any, Callable, Dict, List, Optional

from agentic_common import paths, setup_logging
from agentic_common.gateway_client import GatewayClient, MockGateway
from agentic_common.persistence import AgentStore
from agentic_common.settings import Settings, default_settings
from agentic_common.tracing import Tracer
from chapter01_mcp.servers.ops_db import seed_if_empty
from chapter02_planning.agent import PlanningAgent
from chapter02_planning.config import PlanningConfig, load_planning_config
from chapter02_planning.parsing import ParseError, extract_json_block
from chapter02_planning.schemas import (
  PlanningTaskInput,
  PlanningTaskOutput,
)
from chapter02_planning.tools.factory import BACKENDS, build_tool_provider
from llm_gateway.schemas import GatewayResponse

SCENARIOS: Dict[str, str] = {
  'direct': 'Convert 7 days into hours.',
  'cot': (
    'A product sells 30 units per day. Lead time is 4 days. We have 20 '
    'units in the warehouse and want a buffer of 10 units. How many units '
    'should we buy?'
  ),
  'react': (
    'Check which items are below their reorder point and inspect the '
    'sales trend of the worst one.'
  ),
  'plan': (
    'Audit all stores for low-stock items, check the sales trend of the '
    'most critical item, and then place a restock order sized to one week '
    'of demand.'
  ),
  'unsafe': 'Restock store S01 with 9000 units of R-101 immediately.',
}

LOW_STOCK_TOOL = 'retail-ops.retail_low_stock_report'
TREND_TOOL = 'retail-ops.retail_sales_trend'
RESTOCK_TOOL = 'retail-ops.retail_restock_order'
DEGRADED_TOOL = 'telecom-ops.telecom_degraded_sites'
SITE_STATUS_TOOL = 'telecom-ops.telecom_site_status'
DISPATCH_TOOL = 'telecom-ops.telecom_dispatch_technician'


# ---------------------------------------------------------------------------
# Mock "LLM": a deterministic scripted planner that speaks every protocol
# this chapter defines (CoT text, ReAct text, plan JSON, critique JSON).
# The point is to exercise the real code paths without network access.
# ---------------------------------------------------------------------------

def _first_content(messages: List[Dict[str, Any]], role: str) -> str:
  '''Return the first message content for a role.

  Args:
    messages: Conversation.
    role: Message role.

  Returns:
    Content string (empty when absent).
  '''
  for message in messages:
    if message.get('role') == role:
      return str(message.get('content') or '')
  return ''


def _last_content(messages: List[Dict[str, Any]], role: str) -> str:
  '''Return the last message content for a role.

  Args:
    messages: Conversation.
    role: Message role.

  Returns:
    Content string (empty when absent).
  '''
  for message in reversed(messages):
    if message.get('role') == role:
      return str(message.get('content') or '')
  return ''


def _section(text: str, start: str, end: str) -> str:
  '''Extract the text between two markers.

  Args:
    text: Source text.
    start: Start marker (inclusive).
    end: End marker (exclusive).

  Returns:
    Extracted section, or empty string when markers are missing.
  '''
  if start not in text:
    return ''
  remainder = text.split(start, 1)[1]
  if end in remainder:
    return remainder.split(end, 1)[0].strip()
  return remainder.strip()


def _text(content: str) -> GatewayResponse:
  '''Build a plain-text GatewayResponse.

  Args:
    content: Response text.

  Returns:
    GatewayResponse.
  '''
  return GatewayResponse(
    provider='mock', model='mock-scripted', content=content,
  )


def _final(answer: str) -> GatewayResponse:
  '''Build a ReAct Final Answer response.

  Args:
    answer: Final answer text.

  Returns:
    GatewayResponse with the ReAct protocol text.
  '''
  return _text(f'Thought: I have enough evidence.\nFinal Answer: {answer}')


def _tool_call(name: str, args: Dict[str, Any]) -> GatewayResponse:
  '''Build a ReAct Action response.

  Args:
    name: Tool name.
    args: Tool arguments.

  Returns:
    GatewayResponse with the ReAct protocol text.
  '''
  return _text(
    f'Thought: I should call {name} for the next fact.\n'
    f'Action: {name}\n'
    f'Action Input: {json.dumps(args)}'
  )


def _parse_scratch(scratch: str) -> List[Dict[str, str]]:
  '''Parse Thought/Action/Observation history.

  Args:
    scratch: Scratchpad text.

  Returns:
    List of entries with thought/action/observation keys.
  '''
  entries: List[Dict[str, str]] = []
  for block in re.split(r'(?=^Thought:)', scratch, flags=re.MULTILINE):
    if not block.strip():
      continue
    action_match = re.search(r'^Action:\s*(.+?)\s*$', block, re.MULTILINE)
    if not action_match:
      continue
    observation = ''
    if 'Observation:' in block:
      observation = block.split('Observation:', 1)[1].strip()
    entries.append({
      'action': action_match.group(1).strip(),
      'observation': observation,
    })
  return entries


def _obs_json(observation: str) -> Optional[Dict[str, Any]]:
  '''Decode the JSON payload from a tool observation.

  Args:
    observation: Raw observation text.

  Returns:
    Decoded dict, or None when absent/invalid.
  '''
  try:
    payload = extract_json_block(observation)
  except ParseError:
    return None
  return payload if isinstance(payload, dict) else None


def _search(pattern: str, text: str) -> Optional[float]:
  '''Search a numeric pattern.

  Args:
    pattern: Regex with one capture group.
    text: Source text.

  Returns:
    Parsed float, or None.
  '''
  match = re.search(pattern, text or '', re.IGNORECASE)
  if not match:
    return None
  try:
    return float(match.group(1))
  except (TypeError, ValueError):
    return None


def _store_id(text: str) -> str:
  '''Extract a store id (e.g. S01).

  Args:
    text: Source text.

  Returns:
    Store id, defaulting to S01.
  '''
  return _explicit_store(text) or 'S01'


def _explicit_store(text: str) -> Optional[str]:
  '''Extract a store id without a default.

  Args:
    text: Source text.

  Returns:
    Store id, or None when absent.
  '''
  match = re.search(r'\bS\d{2}\b', (text or '').upper())
  return match.group(0) if match else None


def _sku(text: str) -> str:
  '''Extract a SKU (e.g. R-101).

  Args:
    text: Source text.

  Returns:
    SKU, defaulting to R-101.
  '''
  return _explicit_sku(text) or 'R-101'


def _explicit_sku(text: str) -> Optional[str]:
  '''Extract a SKU without a default.

  Args:
    text: Source text.

  Returns:
    SKU, or None when absent.
  '''
  match = re.search(r'\bR-\d{2,}\b', (text or '').upper())
  return match.group(0) if match else None


def _trend_args(
  entries: List[Dict[str, str]],
  scope: str,
) -> Dict[str, Any]:
  '''Build sales-trend arguments, preferring the worst low-stock item.

  Args:
    entries: Parsed scratchpad entries.
    scope: Objective + context text.

  Returns:
    Tool arguments for retail_sales_trend.
  '''
  store = _explicit_store(scope)
  sku = _explicit_sku(scope)

  if store is None or sku is None:
    for entry in reversed(entries):
      if 'retail_low_stock_report' not in entry['action']:
        continue
      items = (_obs_json(entry['observation']) or {}).get('items') or []
      if items:
        store = store or items[0].get('store_id')
        sku = sku or items[0].get('sku')
      break

  return {
    'store_id': store or 'S01',
    'sku': sku or 'R-101',
    'days': 7,
  }


def _site_id(text: str) -> str:
  '''Extract a cell-site id (e.g. CS-77).

  Args:
    text: Source text.

  Returns:
    Site id, defaulting to CS-77.
  '''
  match = re.search(r'\bCS-\d{2,}\b', (text or '').upper())
  return match.group(0) if match else 'CS-77'


def _restock_quantity(scope: str) -> int:
  '''Derive a restock quantity from context or an explicit number.

  Args:
    scope: Objective + context text.

  Returns:
    Quantity capped at the demo policy maximum.
  '''
  average = _search(r'avg_daily_units\s*[=:]\s*([\d.]+)', scope)
  if average is not None:
    return min(500, max(5, int(average * 7)))
  explicit = _search(r'(\d{2,4})\s*units', scope)
  if explicit is not None:
    return int(explicit)
  return 50


def _retail_final(entries: List[Dict[str, str]]) -> str:
  '''Compose a final answer from the latest retail observation.

  Args:
    entries: Parsed scratchpad entries.

  Returns:
    Final answer text.
  '''
  for entry in reversed(entries):
    action = entry['action']
    observation = entry['observation']
    if 'retail_low_stock_report' in action:
      payload = _obs_json(observation) or {}
      items = payload.get('items') or []
      if not items:
        return 'Low stock report: no items below their reorder point.'
      store = items[0].get('store_id', 'unknown')
      described = '; '.join(
        f"{item.get('sku')}: on_hand={item.get('on_hand')} "
        f"(reorder at {item.get('reorder_point')})"
        for item in items[:4]
      )
      return (
        f'Low stock report for store {store}: {described}. '
        f'{len(items)} item(s) below reorder point.'
      )
    if 'retail_sales_trend' in action:
      payload = _obs_json(observation) or {}
      average = float(payload.get('avg_daily_units') or 0.0)
      return (
        f"Sales trend for {payload.get('store_id')} "
        f"{payload.get('sku')}: avg_daily_units={average} over 7 days; "
        f'projected weekly demand ~{int(average * 7)} units.'
      )
    if 'retail_restock_order' in action:
      if 'BLOCKED' in observation:
        reason = _block_reason(observation)
        return (
          f'Restock was blocked: {reason}. '
          'Manual follow-up is required.'
        )
      payload = _obs_json(observation) or {}
      if payload.get('order_id') is not None:
        return (
          f"Restock order #{payload.get('order_id')} created: "
          f"{payload.get('quantity')} units of {payload.get('sku')} for "
          f"store {payload.get('store_id')}."
        )
      return f'Restock was not possible: {observation[:200]}'
  return 'No tool evidence was gathered; cannot answer confidently.'


def _block_reason(observation: str) -> str:
  '''Extract the safety-gate reason from a blocked observation.

  Args:
    observation: Wrapped observation text containing a BLOCKED marker.

  Returns:
    Short reason string.
  '''
  marker = 'BLOCKED by safety gate:'
  if marker not in observation:
    return 'policy denied the action'
  reason = observation.split(marker, 1)[1].strip()
  reason = reason.split('\n', 1)[0].strip()
  return reason[:160] or 'policy denied the action'


def _low_stock_args(intent: str, scope: str) -> Dict[str, Any]:
  '''Build low-stock report arguments.

  Args:
    intent: Objective/task text (decides store scoping).
    scope: Objective + context for id extraction.

  Returns:
    Tool arguments (empty filters all stores).
  '''
  if any(
    hint in intent
    for hint in ('all stores', 'across all', 'across stores', 'which items')
  ):
    return {}
  return {'store_id': _store_id(scope)}


def _retail_turn(
  entries: List[Dict[str, str]],
  focus: str,
  scope: str,
) -> GatewayResponse:
  '''Scripted ReAct turn for retail flows.

  Args:
    entries: Parsed scratchpad entries.
    focus: The objective (or task) driving intent detection.
    scope: Focus + dependency context, used for argument extraction.

  Returns:
    Next turn (tool call or final answer).
  '''
  actions = [entry['action'] for entry in entries]
  intent = focus.lower()

  if (
    not any('retail_low_stock_report' in action for action in actions)
    and any(
      hint in intent
      for hint in ('low-stock', 'low stock', 'stock level', 'current stock',
                   'which items', 'inventory report')
    )
  ):
    return _tool_call(LOW_STOCK_TOOL, _low_stock_args(intent, scope))

  if (
    not any('retail_sales_trend' in action for action in actions)
    and 'trend' in intent
  ):
    return _tool_call(TREND_TOOL, _trend_args(entries, scope))

  if (
    not any('retail_restock_order' in action for action in actions)
    and ('restock' in intent or 'place an order' in intent)
  ):
    return _tool_call(
      RESTOCK_TOOL,
      {
        'store_id': _store_id(scope),
        'sku': _sku(scope),
        'quantity': _restock_quantity(scope),
      },
    )

  return _final(_retail_final(entries))


def _telecom_turn(
  entries: List[Dict[str, str]],
  scope: str,
) -> GatewayResponse:
  '''Scripted ReAct turn for telecom flows.

  Args:
    entries: Parsed scratchpad entries.
    scope: Objective + context text.

  Returns:
    Next turn (tool call or final answer).
  '''
  actions = [entry['action'] for entry in entries]

  if not any('telecom_degraded_sites' in action for action in actions):
    return _tool_call(DEGRADED_TOOL, {})

  if not any('telecom_site_status' in action for action in actions):
    payload = _obs_json(entries[-1]['observation']) or {}
    sites = payload.get('sites') or []
    if not sites:
      return _final('No degraded sites found; nothing to dispatch.')
    return _tool_call(
      SITE_STATUS_TOOL, {'site_id': sites[0].get('site_id', 'CS-77')},
    )

  if not any('telecom_dispatch_technician' in action for action in actions):
    return _tool_call(
      DISPATCH_TOOL,
      {
        'site_id': _site_id(scope),
        'tech_id': 'T-06',
        'priority': 'high',
        'note': 'auto-dispatch from degraded site report',
      },
    )

  observation = entries[-1]['observation']
  if 'BLOCKED' in observation:
    reason = _block_reason(observation)
    return _final(
      f'Dispatch was blocked: {reason}. Escalate the site for manual '
      'approval.'
    )
  payload = _obs_json(observation) or {}
  if payload.get('dispatch_id') is not None:
    return _final(
      f"Dispatched technician {payload.get('tech_id')} to site "
      f"{payload.get('site_id')} with {payload.get('priority')} priority "
      f"(dispatch #{payload.get('dispatch_id')})."
    )
  return _final(f'Dispatch could not be completed: {observation[:200]}')


def _react_response(user: str) -> GatewayResponse:
  '''Script the next ReAct turn from prompt content.

  Args:
    user: User prompt text.

  Returns:
    Next ReAct turn.
  '''
  task = _section(user, 'Task:\n', '\n\nScratchpad') or user
  scratch = _section(
    user,
    'Scratchpad (Thought/Action/Observation history):\n',
    '\n\nProduce the next turn',
  )
  objective = _section(
    task, 'Your step objective:\n', '\n\nCompletion check:',
  )
  context = _section(
    task, 'Results from prerequisite steps:\n', '\n\nAvailable tools:',
  )
  focus = objective or task
  scope = f'{focus}\n{context}'
  lower = focus.lower()
  entries = _parse_scratch(scratch)

  telecom_hints = ('cell site', 'degraded', 'technician', 'dispatch', 'outage')
  if any(hint in lower for hint in telecom_hints):
    return _telecom_turn(entries, scope)
  return _retail_turn(entries, focus, scope)


def _plan_response(user: str) -> GatewayResponse:
  '''Script a plan JSON response.

  Args:
    user: User prompt text.

  Returns:
    GatewayResponse containing the plan JSON.
  '''
  goal = _section(user, 'Goal:\n', '\n\nAvailable tools:') or 'the goal'
  plan = {
    'goal': goal,
    'assumptions': ['stock levels are current', 'sales history is complete'],
    'nodes': [
      {
        'id': 'inspect_stock',
        'objective': (
          'Gather the current low-stock items across all stores.'
        ),
        'depends_on': [],
        'suggested_tools': [LOW_STOCK_TOOL],
        'success_criteria': 'a list of SKUs below their reorder point',
      },
      {
        'id': 'review_trend',
        'objective': (
          'Check the 7-day sales trend for the most critical SKU identified '
          'in the previous step.'
        ),
        'depends_on': ['inspect_stock'],
        'suggested_tools': [TREND_TOOL],
        'success_criteria': 'average daily units for the critical SKU',
      },
      {
        'id': 'place_order',
        'objective': (
          'Place a restock order for the critical SKU sized to about one '
          'week of average demand.'
        ),
        'depends_on': ['review_trend'],
        'suggested_tools': [RESTOCK_TOOL],
        'success_criteria': 'an order id or an explicit policy block',
      },
    ],
    'notes': 'mock plan for the retail replenishment campaign',
  }
  return _text(json.dumps(plan))


def _critic_response() -> GatewayResponse:
  '''Script a passing critique.

  Returns:
    GatewayResponse containing critique JSON.
  '''
  return _text(json.dumps({
    'verdict': 'pass', 'issues': [], 'revised_answer': '',
  }))


def _revision_response(user: str) -> GatewayResponse:
  '''Script a revision by echoing the current answer.

  Args:
    user: User prompt text.

  Returns:
    GatewayResponse with the revised answer.
  '''
  current = _section(user, 'Current answer:\n', '\n\nValidator findings:')
  return _text(current or 'Unable to revise.')


def _synthesis_response(user: str) -> GatewayResponse:
  '''Script the plan synthesis call.

  Args:
    user: User prompt text.

  Returns:
    GatewayResponse with the synthesized answer.
  '''
  outputs = _section(
    user, 'Completed steps and outputs:\n', '\n\nUnresolved errors:',
  )
  errors = _section(
    user, 'Unresolved errors:\n', '\n\nWrite the final answer now.',
  )
  parts: List[str] = []
  for line in outputs.splitlines():
    if line.startswith('- ['):
      parts.append(line.split('] ', 1)[-1])
  answer = 'Plan completed. ' + ' '.join(parts)
  if errors and errors != '(none)':
    answer += f' Unresolved: {errors}'
  return _text(answer[:800])


def _cot_response(user: str) -> GatewayResponse:
  '''Script a chain-of-thought answer with real arithmetic.

  Args:
    user: User prompt text.

  Returns:
    GatewayResponse in the step-labeled CoT format.
  '''
  question = _section(user, 'Question:\n', '\n\nAdditional context') or user
  per_day = _search(r'(\d+(?:\.\d+)?)\s*units?\s+per\s+day', question)
  lead = _search(r'lead\s+time\s+is\s+(\d+(?:\.\d+)?)\s*days?', question)
  on_hand = _search(
    r'(\d+(?:\.\d+)?)\s*units?\s+in\s+the\s+warehouse', question,
  )
  safety = _search(r'buffer\s+of\s+(\d+(?:\.\d+)?)', question)

  if None not in (per_day, lead, on_hand, safety):
    need = per_day * lead + safety - on_hand
    content = (
      'Reasoning:\n'
      f'1. Lead-time demand = {per_day:g} units/day * {lead:g} days = '
      f'{per_day * lead:g} units.\n'
      f'2. Add safety stock {safety:g} units and subtract on-hand '
      f'{on_hand:g} units.\n'
      f'3. Order quantity = {per_day * lead:g} + {safety:g} - {on_hand:g} '
      f'= {need:g} units.\n'
      f'Answer: {need:g} units'
    )
    return _text(content)

  return _text(
    'Reasoning:\n'
    '1. Reviewed the question and the stated constraints.\n'
    '2. Could not derive a reliable number from the given data.\n'
    'Answer: Insufficient data to compute a reliable number.'
  )


def _direct_response(user: str) -> GatewayResponse:
  '''Script a direct answer for simple tasks.

  Args:
    user: User prompt text.

  Returns:
    GatewayResponse with a concise answer.
  '''
  task = _section(user, 'Task:\n', '\n\nContext:') or user
  match = re.search(
    r'convert\s+(\d+)\s+days?\s+into\s+hours', task, re.IGNORECASE,
  )
  if match:
    return _text(f'{int(match.group(1)) * 24} hours')
  return _text('Direct answer unavailable in mock mode.')


def make_mock_planner(
  config: PlanningConfig,
) -> Callable[[List[Dict[str, Any]]], GatewayResponse]:
  '''Build the deterministic scripted planner.

  Args:
    config: Chapter 2 configuration (unused by the mock itself; kept for
      interface symmetry with live mode).

  Returns:
    Planner callable(messages) -> GatewayResponse.
  '''
  del config

  def planner(messages: List[Dict[str, Any]]) -> GatewayResponse:
    '''Route the prompt to the matching scripted pattern.

    Args:
      messages: Normalized conversation.

    Returns:
      Scripted GatewayResponse.
    '''
    system = _first_content(messages, 'system')
    user = _last_content(messages, 'user')

    if 'ReAct agent' in system:
      return _react_response(user)
    if 'planning agent' in system:
      return _plan_response(user)
    if 'strict but fair validator' in system:
      return _critic_response()
    if 'You revise an operations answer' in system:
      return _revision_response(user)
    if 'You are an operations lead' in system:
      return _synthesis_response(user)
    if 'Think step by step' in system:
      return _cot_response(user)
    return _direct_response(user)

  return planner


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def make_write_approver(
  settings: Settings,
) -> Callable[[str, Dict[str, Any]], bool]:
  '''Build the demo write-approval callback.

  Approves only writes inside safe operational bounds; the unsafe scenario
  (9000 units) is therefore still blocked even with writes enabled.

  Args:
    settings: Runtime settings with the safety caps.

  Returns:
    Approver callable(tool, args) -> bool.
  '''

  def approver(tool: str, args: Dict[str, Any]) -> bool:
    '''Approve or deny one write call.

    Args:
      tool: Qualified tool name.
      args: Proposed arguments.

    Returns:
      True when the call is within bounds.
    '''
    if tool.endswith('retail_restock_order'):
      try:
        quantity = int(args.get('quantity', 0))
      except (TypeError, ValueError):
        return False
      return 0 < quantity <= settings.max_restock_quantity
    if tool.endswith('telecom_dispatch_technician'):
      return str(args.get('priority', '')) in (
        settings.allowed_dispatch_priorities
      )
    return False

  return approver


def run_scenario(
  scenario: str,
  mode: str,
  backend: str,
  allow_writes: bool,
  settings: Settings,
  config: PlanningConfig,
) -> PlanningTaskOutput:
  '''Run one scenario through the full Chapter 2 stack.

  Args:
    scenario: Scenario key from SCENARIOS.
    mode: 'mock' or 'live'.
    backend: 'inproc' or 'mcp'.
    allow_writes: Whether write tools are enabled for the run.
    settings: Runtime settings.
    config: Chapter 2 configuration.

  Returns:
    The agent output.
  '''
  task = SCENARIOS[scenario]

  if mode == 'mock':
    llm: Any = MockGateway(make_mock_planner(config))
  else:
    llm = GatewayClient.shared()

  provider = build_tool_provider(backend, task=task)
  store = AgentStore(paths.AGENT_STATE_DB)
  tracer = Tracer()
  agent = PlanningAgent(
    llm=llm,
    tools=provider,
    config=config,
    settings=settings,
    store=store,
    tracer=tracer,
    write_approver=make_write_approver(settings),
  )

  output = agent.run_task(
    PlanningTaskInput(
      task=task,
      session_id=f'ch2-{scenario}-{backend}',
      allow_writes=allow_writes,
      max_steps=config.max_steps,
      cot_samples=config.cot_samples,
    )
  )

  print('=' * 74)
  write_state = 'on' if allow_writes else 'off'
  print(f'scenario : {scenario}   mode={mode}  backend={backend}  '
        f'writes={write_state}')
  print(f'route    : {output.route}   status={output.status}')
  if output.policy is not None:
    print(f'policy   : {output.policy.rationale}')
    print(f'confidence: {output.policy.confidence:.2f}  '
          f'scores={output.policy.scores}')
  if output.plan is not None:
    print(f'plan     : goal={output.plan.goal[:80]!r}')
    if output.execution is not None:
      print(f'waves    : {output.execution.waves}')
      for result in output.execution.results:
        print(f'  [{result.status:9}] {result.node_id}: '
              f'{result.output[:90]}')
  if output.react is not None:
    print(f'react    : {len(output.react.steps)} step(s)  '
          f'stop={output.react.stop_reason}')
    for step in output.react.steps:
      if step.action:
        status = 'ok' if step.tool_ok else 'fail'
        print(f'  #{step.index} {step.action} [{status}]')
  print(f'answer   : {output.answer[:400]}')
  if output.validation is not None:
    print(f'validation: passed={output.validation.passed} '
          f'revised={output.validation.revised} '
          f'issues={len(output.validation.issues)}')
    for issue in output.validation.issues[:4]:
      print(f'  - [{issue.severity}] {issue.check}: {issue.detail[:100]}')
  if output.tool_calls:
    print('tools    :')
    for audit in output.tool_calls:
      status = 'OK  ' if audit.ok else 'BLOCKED' if audit.blocked else 'FAIL'
      print(f'  [{status}] {audit.tool} args={audit.args} '
            f'{audit.latency_ms:.0f}ms')
  print(f'usage    : {output.usage}')
  if output.errors:
    print(f'errors   : {output.errors}')
  print(f'session  : {output.session_id}')
  print('=' * 74)

  agent.close()
  store.close()
  return output


def main() -> None:
  '''Entry point.'''
  parser = argparse.ArgumentParser(description='Chapter 2 planning demo')
  parser.add_argument(
    '--scenario',
    default='all',
    choices=sorted(SCENARIOS.keys()) + ['all'],
  )
  parser.add_argument(
    '--mock', action='store_true', help='Scripted mock LLM (default)',
  )
  parser.add_argument('--live', action='store_true', help='Real LLM gateway')
  parser.add_argument(
    '--tools', default='inproc', choices=list(BACKENDS),
    help='Tool backend: inproc (Chapter 1 cores) or mcp (stdio servers)',
  )
  parser.add_argument(
    '--read-only', action='store_true',
    help='Disable writes entirely (shows the safety gate blocking)',
  )
  args = parser.parse_args()

  if args.mock and args.live:
    parser.error('--mock and --live are mutually exclusive')

  setup_logging('INFO')
  paths.ensure_data_dirs()
  seed_if_empty()

  settings = default_settings()
  config = load_planning_config(settings)
  mode = 'live' if args.live else 'mock'
  allow_writes = not args.read_only

  scenarios = (
    sorted(SCENARIOS.keys()) if args.scenario == 'all' else [args.scenario]
  )
  for scenario in scenarios:
    run_scenario(
      scenario, mode, args.tools, allow_writes, settings, config,
    )


if __name__ == '__main__':
  main()
