#!/usr/bin/env python
# -- coding: utf-8 --

'''Chapter 1 end-to-end demo.

Modes:
- --mock (default): a scripted "LLM" drives the same tool-use loop; fully
  offline and deterministic - it proves the *plumbing* (MCP, policy, audit).
- --live: every LLM call goes through YOUR llm_gateway (in-process import).

Scenarios (retail & telecom operations):
- restock  : read low stock -> read sales trend -> write restock order
- telecom  : read degraded sites -> read site status -> write dispatch
- unsafe   : restock 9000 units -> policy engine must BLOCK the write
- cross    : retail report only -> proves discovery picks the right server
'''


from __future__ import annotations

import argparse
import json
import re
from typing import Any, Callable, Dict, List

from agentic_common import paths, setup_logging
from agentic_common.gateway_client import GatewayClient, MockGateway
from agentic_common.persistence import AgentStore
from agentic_common.settings import Settings, default_settings
from agentic_common.tracing import Tracer
from chapter01_mcp.agent.ops_agent import OpsAgent
from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.schemas import AgentTaskInput
from chapter01_mcp.servers.ops_db import seed_if_empty
from llm_gateway.schemas import FunctionCall, GatewayResponse, ToolCall

logger = get_mcp_logger(__name__)


# ---------------------------------------------------------------------------
# Scripted mock "LLM": deterministic planner over the same message protocol.
# ---------------------------------------------------------------------------

def _since_user(messages: List[Dict[str, Any]]) -> List[tuple]:
  '''Collect (tool_name, payload_text) pairs after the first user message.

  Args:
    messages: Conversation.

  Returns:
    List of (tool_name, content) tuples.
  '''
  pairs: List[tuple] = []
  for message in messages:
    if message.get('role') == 'tool':
      pairs.append((message.get('name', ''), message.get('content', '')))
  return pairs


def _worst_low_stock(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
  '''Most critical low-stock row seen so far.

  Args:
    messages: Conversation.

  Returns:
    Dict with store_id/sku/on_hand/reorder_point (defaults when absent).
  '''
  for name, content in _since_user(messages):
    if 'low_stock' not in name:
      continue
    try:
      items = json.loads(content).get('items', [])
    except (json.JSONDecodeError, AttributeError):
      continue
    if items:
      return items[0]
  return {'store_id': 'S01', 'sku': 'R-101', 'on_hand': 0, 'reorder_point': 40}


def make_mock_planner(
  settings: Settings,
) -> Callable[[List[Dict[str, Any]]], Any]:
  '''Build the scripted planner.

  The planner inspects the conversation (task + any tool results so far) and
  emits the next tool call - or a final answer when it has enough evidence.

  Args:
    settings: Runtime settings.

  Returns:
    Planner callable(messages) -> GatewayResponse.
  '''

  def tool_call(name: str, args: Dict[str, Any]) -> GatewayResponse:
    '''Build a GatewayResponse requesting one tool call.

    Args:
      name: Qualified tool name.
      args: Arguments dict.

    Returns:
      GatewayResponse with one tool call.
    '''
    return GatewayResponse(
      provider='mock',
      model='mock-scripted',
      tool_calls=[
        ToolCall(
          id=f'call_{len(json.dumps(args))}',
          function=FunctionCall(name=name, arguments=json.dumps(args)),
        )
      ],
    )

  def final(answer: str) -> GatewayResponse:
    '''Build a final answer response.

    Args:
      answer: Answer text.

    Returns:
      GatewayResponse with content.
    '''
    return GatewayResponse(
      provider='mock', model='mock-scripted', content=answer,
    )

  def planner(messages: List[Dict[str, Any]]) -> GatewayResponse:
    '''Decide the next move given the full conversation.

    Args:
      messages: Conversation so far.

    Returns:
      Next GatewayResponse (tool calls or final answer).
    '''
    task_text = ''
    for message in messages:
      if message.get('role') == 'user':
        task_text = str(message.get('content', ''))
        break
    task = task_text.lower()

    results = _since_user(messages)
    tools_used = [name for name, _ in results]

    # ---------------- Telecom scenario ----------------
    if 'cell site' in task or 'telecom' in task or 'technician' in task:
      if 'telecom-ops.telecom_degraded_sites' not in tools_used:
        return tool_call('telecom-ops.telecom_degraded_sites', {})
      if 'telecom-ops.telecom_site_status' not in tools_used:
        try:
          degraded_text = dict(results)['telecom-ops.telecom_degraded_sites']
          payload = json.loads(degraded_text)
          worst = payload['sites'][0]['site_id']
        except (json.JSONDecodeError, KeyError, IndexError):
          return final('No degraded sites found; nothing to dispatch.')
        return tool_call(
          'telecom-ops.telecom_site_status', {'site_id': worst},
        )
      if not any('telecom_dispatch_technician' in name for name in tools_used):
        status_text = dict(results).get('telecom-ops.telecom_site_status', '{}')
        try:
          worst = json.loads(status_text).get('site_id', 'CS-77')
        except (json.JSONDecodeError, AttributeError):
          worst = 'CS-77'
        return tool_call(
          'telecom-ops.telecom_dispatch_technician',
          {'site_id': worst, 'tech_id': 'T-06', 'priority': 'high',
           'note': 'auto-dispatch from degraded site report'},
        )

      # Final answer after dispatch attempt (success or policy block).
      dispatch_lines = [
        content for name, content in results
        if 'telecom_dispatch_technician' in name
      ]
      last = dispatch_lines[-1] if dispatch_lines else '{}'
      try:
        d = json.loads(last)
      except json.JSONDecodeError:
        d = {}
      if d.get('dispatch_id') is not None:
        return final(
          f"Dispatched technician {d.get('tech_id')} to site "
          f"{d.get('site_id')} with {d.get('priority')} priority "
          f"(dispatch #{d.get('dispatch_id')})."
        )
      return final(
        f'Could not dispatch: {last}. Escalate manually '
        f'or pick another technician.'
      )

    # ---------------- Unsafe scenario ----------------
    if '9000' in task:
      if 'retail-ops.retail_low_stock_report' not in tools_used:
        return tool_call(
          'retail-ops.retail_low_stock_report', {'store_id': 'S01'}
        )
      if not any('retail_restock_order' in name for name in tools_used):
        return tool_call(
          'retail-ops.retail_restock_order',
          {'store_id': 'S01', 'sku': 'R-101', 'quantity': 9000},
        )
      return final(
        'Restock request blocked: 9000 units exceeds the policy cap. '
        'Manual approval required for quantities this large.'
      )

    # ---------------- Restock scenario ----------------
    if 'retail-ops.retail_low_stock_report' not in tools_used:
      # Pick the store id mentioned in the task, when any (e.g. S02).
      store_match = re.search(r'\bS\d{2}\b', task_text.upper())
      return tool_call(
        'retail-ops.retail_low_stock_report',
        {'store_id': store_match.group(0) if store_match else 'S01'},
      )

    # Report-only tasks end after the stock report.
    requests_order = any(
      phrase in task
      for phrase in ('place', 'restock order', 'replenish', '9000')
    )
    if not requests_order:
      stock_text = dict(results).get('retail-ops.retail_low_stock_report', '')
      try:
        payload = json.loads(stock_text)
        items = payload.get('items', [])
      except (json.JSONDecodeError, AttributeError):
        items = []
      if items:
        lines = [
          f"{i.get('sku')}: {i.get('on_hand')} on hand "
          f"(reorder at {i.get('reorder_point')})"
          for i in items
        ]
        store_id = items[0].get('store_id', 'unknown')
        return final(
          f'Low stock report for store {store_id}: '
          + '; '.join(lines)
          + f'. {len(items)} item(s) below reorder point.'
        )
      return final(
        'Low stock report: all items are above their reorder points.'
      )

    if not any('retail_sales_trend' in name for name in tools_used):
      worst = _worst_low_stock(messages)
      return tool_call(
        'retail-ops.retail_sales_trend',
        {
          'store_id': worst.get('store_id', 'S01'),
          'sku': worst.get('sku', 'R-101'),
          'days': 7,
        },
      )

    if not any('retail_restock_order' in name for name in tools_used):
      worst = _worst_low_stock(messages)
      trend_text = (
        dict(results).get('retail-ops.retail_sales_trend', '')
      )
      try:
        avg = float(json.loads(trend_text).get('avg_daily_units') or 1.0)
      except (json.JSONDecodeError, ValueError, AttributeError):
        avg = 1.0
      need = max(5, int(avg * 7) - int(worst.get('on_hand', 0)))
      return tool_call(
        'retail-ops.retail_restock_order',
        {
          'store_id': worst.get('store_id', 'S01'),
          'sku': worst.get('sku', 'R-101'),
          'quantity': min(need, settings.max_restock_quantity),
        },
      )

    restock_text = dict(results).get('retail-ops.retail_restock_order', '')
    try:
      order = json.loads(restock_text)
    except json.JSONDecodeError:
      order = {}
    worst = _worst_low_stock(messages)
    if order.get('order_id') is not None:
      return final(
        f"Restock order #{order.get('order_id')} created: "
        f"{order.get('quantity')} units "
        f"of {order.get('sku')} for store {order.get('store_id')} "
        f"(on hand was {worst.get('on_hand')})."
      )
    return final(f'Restock was not possible: {restock_text}')

  return planner


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, str] = {
  'restock': (
    'Our retail stores need attention: find the most critical low-stock item '
    'in store S01, check its sales trend, and place a restock order sized to '
    'roughly one week of demand.'
  ),
  'telecom': (
    'Check which cell sites are degraded right now, inspect the worst one, '
    'and dispatch technician T-06 there with high priority.'
  ),
  'unsafe': (
    'Restock store S01 with 9000 units of R-101 immediately.'
  ),
}


def build_llm(mode: str, settings: Settings):
  '''Build the LLM client for the demo.

  Args:
    mode: 'mock' or 'live'.
    settings: Runtime settings.

  Returns:
    GatewayClient or MockGateway.

  Raises:
    SystemExit: When live mode is unavailable.
  '''
  if mode == 'mock':
    return MockGateway(make_mock_planner(settings))

  return GatewayClient.shared()


def run_scenario(scenario: str, mode: str, settings: Settings) -> None:
  '''Run one demo scenario through the full agent stack.

  Args:
    scenario: Scenario key.
    mode: 'mock' or 'live'.
    settings: Runtime settings.
  '''
  task_text = SCENARIOS[scenario]
  llm = build_llm(mode, settings)
  store = AgentStore(paths.AGENT_STATE_DB)
  tracer = Tracer()

  agent = OpsAgent(llm=llm, settings=settings, store=store, tracer=tracer)
  output = agent.run_task(
    AgentTaskInput(task=task_text, session_id=f'demo-{scenario}'),
  )

  print('=' * 72)
  print(f'scenario : {scenario}   (mode={mode})')
  print(f'status   : {output.status}   iterations={output.iterations}')
  print(f'answer   : {output.answer[:400]}')
  print('tools    :')
  for call in output.tool_calls:
    status = 'OK  ' if call.ok else 'FAIL'
    print(f'  [{status}] {call.server}.{call.tool} args={call.args} '
          f'approved={call.approved} {call.latency_ms:.0f}ms')
  if output.errors:
    print(f'errors   : {output.errors}')
  print(f'usage    : {output.usage}')
  print(f'session  : {output.session_id}')
  print(f'persist  : {paths.AGENT_STATE_DB} + data/traces/<trace_id>.jsonl')
  print('=' * 72)


def main() -> None:
  '''Entry point.'''
  parser = argparse.ArgumentParser(description='Chapter 1 MCP demo')
  parser.add_argument(
    '--scenario',
    default='restock',
    choices=sorted(SCENARIOS.keys()),
  )
  parser.add_argument(
    '--mock', action='store_true', help='Scripted mock LLM (offline)'
  )
  parser.add_argument('--live', action='store_true', help='Real LLM gateway')
  args = parser.parse_args()

  if args.mock and args.live:
    parser.error('--mock and --live are mutually exclusive')

  setup_logging('INFO')
  paths.ensure_data_dirs()
  seed_if_empty()

  settings = default_settings()
  mode = 'live' if args.live else 'mock'
  run_scenario(args.scenario, mode, settings)


if __name__ == '__main__':
  main()
