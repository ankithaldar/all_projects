#!/usr/bin/env python
# -- coding: utf-8 --

'''Chapter 3 end-to-end demo.

Scenarios (retail & telecom operations):
  perception   : show the Perception layer's signals/entities, no LLM
  dag_retail   : two-step retail investigation through a JSON DAG plan
  code_restock : restock sizing computed by an agent-written solve()
                 function in the sandbox
  adaptive     : injected tool failures exercise retry + tool switching
  fallback     : degraded mode with no tools, memory-and-reason answer
  unsafe       : oversized write refused by the approval bounds

Modes:
  --mock (default)  scripted "LLM" drives the real layers offline
  --live            every decision goes through your llm_gateway

Tool backends:
  --tools inproc (Chapter 1 server cores) or mcp (stdio servers)

Run:
  python -m chapter03_cognition.demo --scenario all --mock
  python -m chapter03_cognition.demo --scenario adaptive --mock
  python -m chapter03_cognition.demo --scenario dag_retail --live --tools mcp
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
from chapter03_cognition.agent import CognitiveAgent
from chapter03_cognition.config import CognitionConfig, load_cognition_config
from chapter03_cognition.memory import MemoryStore
from chapter03_cognition.schemas import (
  CognitiveTaskInput,
  CognitiveTaskOutput,
  StrategyName,
)
from chapter03_cognition.tools import FaultInjector, ToolBox
from llm_gateway.schemas import GatewayResponse

LOW_STOCK_TOOL = 'retail-ops.retail_low_stock_report'
TREND_TOOL = 'retail-ops.retail_sales_trend'
RESTOCK_TOOL = 'retail-ops.retail_restock_order'
DEGRADED_TOOL = 'telecom-ops.telecom_degraded_sites'
SITE_STATUS_TOOL = 'telecom-ops.telecom_site_status'
DISPATCH_TOOL = 'telecom-ops.telecom_dispatch_technician'

SCENARIOS: Dict[str, str] = {
  'perception': (
    'Audit all stores for low-stock items, check the sales trend of the '
    'most critical item, and then place a restock order sized to one week '
    'of demand.'
  ),
  'dag_retail': (
    'Check which items are below their reorder point and inspect the '
    'sales trend of the worst one.'
  ),
  'code_restock': (
    'A product sells 30 units per day. Lead time is 4 days. We have 20 '
    'units on hand and want a buffer of 10 units. How many units should '
    'we order?'
  ),
  'adaptive': (
    'Investigate the sales trend for store S02 and SKU R-401 right now.'
  ),
  'fallback': (
    'Check which items are below their reorder point right now.'
  ),
  'unsafe': 'Restock store S01 with 9000 units of R-101 immediately.',
}

FAULT_SCENARIOS = ('adaptive',)


# ---------------------------------------------------------------------------
# Mock "LLM": scripted planner for every prompt family in the chapter.
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
  '''Extract text between two markers.

  Args:
    text: Source text.
    start: Start marker.
    end: End marker.

  Returns:
    Extracted section or empty string.
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


def _search(pattern: str, text: str) -> Optional[float]:
  '''Search a numeric pattern.

  Args:
    pattern: Regex with one capture group.
    text: Source text.

  Returns:
    Parsed float or None.
  '''
  match = re.search(pattern, text or '', re.IGNORECASE)
  if not match:
    return None
  try:
    return float(match.group(1))
  except (TypeError, ValueError):
    return None


def _store_id(text: str) -> str:
  '''Extract a store id (default S01).

  Args:
    text: Source text.

  Returns:
    Store id.
  '''
  match = re.search(r'\bS\d{2}\b', (text or '').upper())
  return match.group(0) if match else 'S01'


def _sku(text: str) -> str:
  '''Extract a SKU (default R-101).

  Args:
    text: Source text.

  Returns:
    SKU.
  '''
  match = re.search(r'\bR-\d{2,}\b', (text or '').upper())
  return match.group(0) if match else 'R-101'


def _quantity(text: str) -> int:
  '''Extract an explicit quantity (default 50).

  Args:
    text: Source text.

  Returns:
    Quantity.
  '''
  explicit = _search(r'(\d{2,5})\s*units', text)
  if explicit is not None:
    return int(explicit)
  return 50


def _mock_dag_response(user: str) -> GatewayResponse:
  '''Script a JSON DAG plan for the current goal.

  Args:
    user: DAG planning prompt.

  Returns:
    GatewayResponse containing plan JSON.
  '''
  goal = _section(user, 'Goal:\n', '\n\nPerception:') or 'the goal'
  tools_section = _section(user, 'Available tools:\n', '\n\nProduce at most')
  no_tools = '(no tools available)' in tools_section
  lower = goal.lower()

  if no_tools:
    plan = {
      'goal': goal,
      'steps': [
        {
          'id': 'answer_from_memory',
          'objective': goal,
          'kind': 'reason',
          'depends_on': [],
          'expected_output': 'caveated answer',
        }
      ],
      'rationale': 'no tools available; reason from memory',
      'assumptions': ['live data unavailable'],
    }
    return _text(json.dumps(plan))

  if ('restock' in lower or 'place' in lower) and any(
    marker in lower
    for marker in (
      'audit', 'which items', 'low-stock', 'low stock', 'all stores',
    )
  ):
    steps = [
      {
        'id': 'find_low_stock',
        'objective': 'List items below their reorder point.',
        'kind': 'tool',
        'tool': LOW_STOCK_TOOL,
        'arguments': {'store_id': _store_id(goal)},
        'depends_on': [],
        'expected_output': 'low-stock SKUs with quantities',
      },
      {
        'id': 'check_trend',
        'objective': 'Read the sales trend for the most critical SKU.',
        'kind': 'tool',
        'tool': TREND_TOOL,
        'arguments': {
          'store_id': _store_id(goal),
          'sku': _sku(goal),
          'days': 7,
        },
        'depends_on': ['find_low_stock'],
        'expected_output': 'average daily units',
      },
      {
        'id': 'place_order',
        'objective': 'Place the restock order sized to one week of demand.',
        'kind': 'write',
        'tool': RESTOCK_TOOL,
        'arguments': {
          'store_id': _store_id(goal),
          'sku': _sku(goal),
          'quantity': 50,
        },
        'depends_on': ['check_trend'],
        'expected_output': 'order id or an explicit policy block',
      },
    ]
  elif 'restock' in lower or 'place' in lower:
    steps = [
      {
        'id': 'place_order',
        'objective': 'Place the restock order for the critical item.',
        'kind': 'write',
        'tool': RESTOCK_TOOL,
        'arguments': {
          'store_id': _store_id(goal),
          'sku': _sku(goal),
          'quantity': _quantity(goal),
        },
        'depends_on': [],
        'expected_output': 'order id or an explicit policy block',
      }
    ]
  elif 'sales trend' in lower and 'which items' not in lower:
    steps = [
      {
        'id': 'check_trend',
        'objective': 'Read the sales trend for the named store and SKU.',
        'kind': 'tool',
        'tool': TREND_TOOL,
        'arguments': {
          'store_id': _store_id(goal),
          'sku': _sku(goal),
          'days': 7,
        },
        'fallback_tool': LOW_STOCK_TOOL,
        'fallback_arguments': {'store_id': _store_id(goal)},
        'depends_on': [],
        'expected_output': 'average daily units',
      }
    ]
  else:
    steps = [
      {
        'id': 'find_low_stock',
        'objective': 'List items below their reorder point.',
        'kind': 'tool',
        'tool': LOW_STOCK_TOOL,
        'arguments': {'store_id': _store_id(goal)},
        'depends_on': [],
        'expected_output': 'low-stock SKUs with quantities',
      },
      {
        'id': 'check_trend',
        'objective': 'Read the sales trend for the most critical SKU.',
        'kind': 'tool',
        'tool': TREND_TOOL,
        'arguments': {
          'store_id': _store_id(goal),
          'sku': _sku(goal),
          'days': 7,
        },
        'fallback_tool': LOW_STOCK_TOOL,
        'fallback_arguments': {'store_id': _store_id(goal)},
        'depends_on': ['find_low_stock'],
        'expected_output': 'average daily units',
      },
    ]

  plan = {
    'goal': goal,
    'steps': steps,
    'rationale': 'mock plan for the operations goal',
    'assumptions': ['data is current'],
  }
  return _text(json.dumps(plan))


def _mock_code_response(user: str) -> GatewayResponse:
  '''Script an agent-written solve() function.

  The generated function computes restock quantity from the numbers stated
  in the task, which exercises the sandbox with real arithmetic.

  Args:
    user: Code planning prompt.

  Returns:
    GatewayResponse containing Python code.
  '''
  goal = _section(user, 'Goal:\n', '\n\nKnown facts') or user
  per_day = _search(r'(\d+(?:\.\d+)?)\s*units?\s+per\s+day', goal)
  lead = _search(r'lead\s+time\s+is\s+(\d+(?:\.\d+)?)\s*days?', goal)
  on_hand = _search(
    r'(\d+(?:\.\d+)?)\s*units?\s+on\s+hand', goal,
  )
  buffer = _search(r'buffer\s+of\s+(\d+(?:\.\d+)?)', goal)

  if None not in (per_day, lead, on_hand, buffer):
    code = (
      'def solve(context):\n'
      f'    per_day = {per_day:g}\n'
      f'    lead_days = {lead:g}\n'
      f'    on_hand = {on_hand:g}\n'
      f'    buffer = {buffer:g}\n'
      '    lead_time_demand = per_day * lead_days\n'
      '    quantity = lead_time_demand + buffer - on_hand\n'
      '    return {\n'
      '        "answer": (\n'
      '            f"Order {quantity:g} units: "\n'
      '            f"{per_day:g} units/day x {lead_days:g} days = "\n'
      '            f"{lead_time_demand:g} plus buffer {buffer:g} minus "\n'
      '            f"on hand {on_hand:g}."\n'
      '        ),\n'
      '        "used_tools": [],\n'
      '        "notes": ["computed from stated numbers"],\n'
      '    }\n'
    )
  else:
    code = (
      'def solve(context):\n'
      '    facts = context.get("facts", {})\n'
      '    return {\n'
      '        "answer": (\n'
      '            "Insufficient numeric inputs to compute a quantity; "\n'
      '            "provide units/day, lead time, on-hand, and buffer."\n'
      '        ),\n'
      '        "used_tools": [],\n'
      '        "notes": ["missing inputs"],\n'
      '    }\n'
    )
  return _text(code)


def _mock_reason_response(user: str) -> GatewayResponse:
  '''Script a reasoning step answer.

  Args:
    user: Reason prompt.

  Returns:
    GatewayResponse with a short reasoned output.
  '''
  objective = _section(user, 'Your step objective:\n', '\n\nExpected output:')
  context = _section(
    user, 'Context from completed steps:\n', '\n\n',
  )
  if 'no prior evidence' in user.lower() or not context:
    return _text(
      'Live tool data was not available, so this answer is based on '
      'remembered context only: the task requires a fresh data check '
      'before any action is taken. (unverified)'
    )
  return _text(
    f'Completed: {objective[:160]}. Evidence: {context[:300]}'
  )


def _summarize_step(line: str) -> str:
  '''Turn one '- [id] payload' synthesis line into a short summary.

  Args:
    line: Line from the synthesis prompt.

  Returns:
    Human-readable summary.
  '''
  step_id, _, payload = line.partition('] ')
  step_id = step_id.replace('- [', '').strip()
  payload = payload.replace('[BEGIN UNTRUSTED TOOL DATA]', '')
  payload = payload.replace('[END UNTRUSTED TOOL DATA]', '').strip()
  try:
    data = json.loads(payload)
  except (TypeError, ValueError):
    return f'{step_id}: {payload[:120]}'

  if isinstance(data, dict):
    if data.get('order_id') is not None:
      return (
        f"order #{data.get('order_id')} created for "
        f"{data.get('quantity')} units of {data.get('sku')} at "
        f"{data.get('store_id')}"
      )
    if data.get('avg_daily_units') is not None:
      return (
        f"{data.get('store_id')} {data.get('sku')} average "
        f"{data.get('avg_daily_units')} units/day"
      )
    if 'items' in data:
      items = data.get('items') or []
      return f'{step_id}: {len(items)} item(s) below reorder point'
  return f'{step_id}: {payload[:120]}'


def _mock_synthesis_response(user: str) -> GatewayResponse:
  '''Script the synthesis call.

  Args:
    user: Synthesis prompt.

  Returns:
    GatewayResponse with the combined answer.
  '''
  steps = _section(user, 'Completed steps:\n', '\n\nUnresolved errors:')
  errors = _section(
    user, 'Unresolved errors:\n', '\n\nWrite the final answer now.',
  )
  parts: List[str] = []
  for line in steps.splitlines():
    if line.startswith('- ['):
      parts.append(_summarize_step(line))
  answer = 'Plan complete. ' + '; '.join(parts)
  if errors and errors != '(none)':
    answer += f' Unresolved: {errors}'
  return _text(answer[:800])


def _mock_fallback_response(user: str) -> GatewayResponse:
  '''Script a degraded-mode answer.

  Args:
    user: Fallback prompt.

  Returns:
    GatewayResponse with an explicit caveat.
  '''
  task = _section(user, 'Task:\n', '\n\nRemembered facts:') or user
  return _text(
    f'Degraded-mode answer for: {task[:120]}. No live tool data or code '
    'execution was available, so this cannot be verified. Re-run with '
    'tools enabled before acting.'
  )


def make_mock_planner(
  config: CognitionConfig,
) -> Callable[[List[Dict[str, Any]]], GatewayResponse]:
  '''Build the deterministic scripted planner.

  Args:
    config: Cognition config (kept for interface symmetry).

  Returns:
    Planner callable(messages) -> GatewayResponse.
  '''
  del config

  def planner(messages: List[Dict[str, Any]]) -> GatewayResponse:
    '''Route the prompt to the matching scripted family.

    Args:
      messages: Normalized conversation.

    Returns:
      Scripted GatewayResponse.
    '''
    system = _first_content(messages, 'system')
    user = _last_content(messages, 'user')
    if 'You are a planning agent' in system:
      return _mock_dag_response(user)
    if 'You write small, safe Python functions' in system:
      return _mock_code_response(user)
    if 'You execute one step' in system:
      return _mock_reason_response(user)
    if 'You are an operations lead' in system:
      return _mock_synthesis_response(user)
    if 'degraded mode' in system:
      return _mock_fallback_response(user)
    return _text('Mock response.')

  return planner


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def make_write_approver(
  settings: Settings,
) -> Callable[[str, Dict[str, Any]], bool]:
  '''Build the bounded write-approval callback.

  Args:
    settings: Runtime settings with safety caps.

  Returns:
    Approver callable(tool, args) -> bool.
  '''

  def approver(tool: str, args: Dict[str, Any]) -> bool:
    '''Approve writes inside safe operational bounds.

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


def build_toolbox(
  backend: str,
  task: str,
  settings: Settings,
  config: CognitionConfig,
  allow_writes: bool,
  inject_faults: bool,
  audit_sink: Any,
  tracer: Tracer,
) -> Any:
  '''Build the (optionally fault-injected) toolbox.

  Args:
    backend: 'inproc' or 'mcp'.
    task: Task text for MCP server selection.
    settings: Runtime settings.
    config: Cognition config.
    allow_writes: Whether writes are permitted.
    inject_faults: Whether to wrap with a fault injector.
    audit_sink: Callback receiving audit records.
    tracer: Tracer instance.

  Returns:
    ToolBox or FaultInjector.
  '''
  toolbox = ToolBox(
    backend=backend,
    task=task,
    allow_writes=allow_writes,
    require_approval=config.require_write_approval,
    approver=make_write_approver(settings),
    max_result_chars=config.max_result_chars,
    sanitize=config.sanitize_observations,
    audit_sink=audit_sink,
    tracer=tracer,
  )
  if inject_faults:
    toolbox = FaultInjector(
      toolbox,
      failures={TREND_TOOL: 3},
      error='injected transient failure',
    )
  return toolbox


def _display(text: str) -> str:
  '''Strip untrusted-data markers for console display.

  Args:
    text: Raw text.

  Returns:
    Cleaned text.
  '''
  return (
    text.replace('[BEGIN UNTRUSTED TOOL DATA]', '')
    .replace('[END UNTRUSTED TOOL DATA]', '')
    .strip()
  )


def print_output(
  scenario: str,
  mode: str,
  backend: str,
  output: CognitiveTaskOutput,
) -> None:
  '''Print a structured run summary.

  Args:
    scenario: Scenario key.
    mode: 'mock' or 'live'.
    backend: Tool backend.
    output: Agent output.
  '''
  print('=' * 74)
  print(f'scenario : {scenario}   mode={mode}  backend={backend}')
  if output.percept is not None:
    signals = output.percept.signals
    print(
      f'percept  : domain={signals.domain} risk={signals.risk:.2f} '
      f'ambiguity={signals.ambiguity:.2f} data_need={signals.data_need:.2f} '
      f'multi_step={signals.multi_step:.2f}'
    )
    if output.percept.entities:
      print(f'entities : {output.percept.entities}')
    if output.percept.required_capabilities:
      print(f'needed   : {output.percept.required_capabilities}')
  if output.memory is not None:
    print(
      f'memory   : recalled={len(output.memory.relevant_episodes)} '
      f'facts={len(output.memory.facts)} '
      f'recommended={output.memory.recommended_strategy}'
    )
  if output.decision is not None:
    decision = output.decision
    print(
      f'strategy : {decision.strategy}  '
      f'({decision.strategy_reason})'
    )
    insight = decision.insight
    print(
      f'plan     : source={decision.plan.source} '
      f'steps={insight.step_count} risk={insight.risk_level} '
      f'confidence={insight.confidence:.2f}'
    )
    for step in decision.plan.steps:
      tool = f' tool={step.tool}' if step.tool else ''
      print(f'  - {step.id} [{step.kind}]{tool}')
    for warning in insight.warnings:
      print(f'  ! {warning}')
  if output.action is not None:
    print(f'action   : status={output.action.status}')
    for step in output.action.steps:
      tool_label = step.tool_used or '-'
      print(
        f'  [{step.status:9}] {step.step_id} '
        f'tool={tool_label} '
        f'attempts={len(step.attempts)} '
        f'{_display(step.output)[:80]}'
      )
    for record in output.action.adaptations:
      print(
        f'  ~ {record.kind}: {record.detail} ({record.reason})'
      )
    for audit in output.action.tool_audits:
      status = 'OK  ' if audit.ok else 'BLOCKED' if audit.blocked else 'FAIL'
      print(f'  [{status}] {audit.tool} args={audit.args}')
  print(f'answer   : {_display(output.answer)[:400]}')
  print(f'usage    : {output.usage}')
  if output.errors:
    print(f'errors   : {output.errors}')
  print(f'session  : {output.session_id}')
  print('=' * 74)


def run_scenario(
  scenario: str,
  mode: str,
  backend: str,
  allow_writes: bool,
  strategy_hint: Optional[StrategyName],
  no_tools: bool,
  code_runner: Optional[str],
  plan_mode: str,
  settings: Settings,
  config: CognitionConfig,
) -> CognitiveTaskOutput:
  '''Run one scenario through the full cognitive stack.

  Args:
    scenario: Scenario key.
    mode: 'mock' or 'live'.
    backend: Tool backend.
    allow_writes: Whether writes are permitted.
    strategy_hint: Optional strategy override.
    no_tools: Whether to run without a toolbox.
    code_runner: Optional sandbox override.
    plan_mode: 'auto', 'dag', or 'code'.
    settings: Runtime settings.
    config: Cognition config.

  Returns:
    The agent output.
  '''
  task = SCENARIOS[scenario]
  if mode == 'mock':
    llm: Any = MockGateway(make_mock_planner(config))
  else:
    llm = GatewayClient.shared()

  store = AgentStore(paths.AGENT_STATE_DB)
  memory = MemoryStore(config.memory_path())
  tracer = Tracer()

  agent = CognitiveAgent(
    llm=llm,
    toolbox=None,
    config=config,
    settings=settings,
    store=store,
    memory=memory,
    tracer=tracer,
  )
  if not no_tools:
    toolbox = build_toolbox(
      backend=backend,
      task=task,
      settings=settings,
      config=config,
      allow_writes=allow_writes,
      inject_faults=scenario in FAULT_SCENARIOS,
      audit_sink=agent.audit_sink,
      tracer=tracer,
    )
    agent.attach_toolbox(toolbox)
  output = agent.run_task(
    CognitiveTaskInput(
      task=task,
      session_id=f'ch3-{scenario}-{backend}',
      strategy_hint=strategy_hint,
      allow_writes=allow_writes,
      allow_code=config.code_execution_enabled,
      code_runner=code_runner,
      plan_mode=plan_mode,
      max_steps=config.max_steps,
    )
  )
  print_output(scenario, mode, backend, output)
  agent.close()
  store.close()
  memory.close()
  return output


def main() -> None:
  '''Entry point.'''
  parser = argparse.ArgumentParser(description='Chapter 3 cognition demo')
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
    '--tools', default='inproc', choices=['inproc', 'mcp'],
    help='Tool backend: inproc (Chapter 1 cores) or mcp (stdio servers)',
  )
  parser.add_argument(
    '--strategy',
    default=None,
    choices=['conservative', 'exploratory', 'fallback'],
    help='Override strategy selection',
  )
  parser.add_argument(
    '--runner', default=None, choices=['restricted', 'subprocess'],
    help='Code sandbox: restricted (tools allowed) or subprocess (pure)',
  )
  parser.add_argument(
    '--plan-mode', default=None, choices=['auto', 'dag', 'code'],
    help='Force a planner shape (defaults per scenario)',
  )
  parser.add_argument(
    '--fresh-memory', action='store_true',
    help='Clear the cognitive memory DB before running',
  )
  parser.add_argument(
    '--no-tools', action='store_true',
    help='Run without tools (shows degraded planning)',
  )
  parser.add_argument(
    '--allow-writes', action='store_true',
    help='Permit writes (still bounded by the approval callback)',
  )
  args = parser.parse_args()

  if args.mock and args.live:
    parser.error('--mock and --live are mutually exclusive')

  setup_logging('INFO')
  paths.ensure_data_dirs()
  seed_if_empty()

  settings = default_settings()
  config = load_cognition_config(settings)
  if args.fresh_memory:
    memory_path = config.memory_path()
    for suffix in ('', '-wal', '-shm'):
      candidate = memory_path.with_name(memory_path.name + suffix)
      if candidate.exists():
        candidate.unlink()
  mode = 'live' if args.live else 'mock'
  scenarios = (
    sorted(SCENARIOS.keys()) if args.scenario == 'all' else [args.scenario]
  )

  for scenario in scenarios:
    allow_writes = args.allow_writes or scenario == 'unsafe'
    strategy = args.strategy
    if scenario == 'adaptive' and strategy is None:
      strategy = 'exploratory'
    if scenario == 'fallback' and strategy is None:
      strategy = 'fallback'
    plan_mode = args.plan_mode or (
      'code' if scenario == 'code_restock' else 'auto'
    )
    run_scenario(
      scenario=scenario,
      mode=mode,
      backend=args.tools,
      allow_writes=allow_writes,
      strategy_hint=strategy,
      no_tools=args.no_tools or scenario == 'fallback',
      code_runner=args.runner,
      plan_mode=plan_mode,
      settings=settings,
      config=config,
    )


if __name__ == '__main__':
  main()
