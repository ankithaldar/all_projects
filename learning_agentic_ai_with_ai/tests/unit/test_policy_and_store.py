#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: policy engine, discovery, and persistence.'''


from __future__ import annotations

from pathlib import Path

from agentic_common.persistence import AgentStore
from chapter01_mcp.client.policy import ToolPolicy, ToolPolicyEngine
from chapter01_mcp.client.registry import (
  ServerCatalog,
  pick_servers,
)
from chapter01_mcp.schemas import ToolCallResult, ToolDescriptor

# pytest fixtures and tuple unpacking patterns are intentional here.
# pylint: disable=redefined-outer-name


class TestToolPolicyEngine:
  '''The policy gate blocks schema, limits, and approval violations.'''

  WRITE_TOOLS = {'srv.writer'}

  def _descriptor(self) -> ToolDescriptor:
    return ToolDescriptor(
      name='tool',
      description='d',
      inputSchema={
        'type': 'object',
        'required': ['qty'],
        'properties': {
          'qty': {'type': 'integer', 'minimum': 0, 'maximum': 1000},
        },
      },
    )

  def test_schema_violation_blocks(self) -> None:
    engine = ToolPolicyEngine()
    allowed, reason, _ = engine.check(
      'srv', 'tool', {'qty': 'many'}, self._descriptor(),
    )
    assert not allowed
    assert 'schema' in reason

  def test_argument_limits_block(self) -> None:
    engine = ToolPolicyEngine()
    engine.set_tool_policy(
      'srv.tool',
      ToolPolicy(argument_limits={'qty': (1, 500)}),
    )
    # Schema allows 1000; the *policy* limit (500) must still catch 600.
    allowed, reason, _ = engine.check(
      'srv', 'tool', {'qty': 600}, self._descriptor(),
    )
    assert not allowed
    assert 'outside allowed range' in reason

  def test_allowed_values_block(self) -> None:
    engine = ToolPolicyEngine()
    engine.set_tool_policy(
      'srv.tool',
      ToolPolicy(allowed_values={'qty': {1, 2, 3}}),
    )
    allowed, _, _ = engine.check(
      'srv', 'tool', {'qty': 7}, self._descriptor(),
    )
    assert not allowed

  def test_write_without_approval_blocked(self) -> None:
    engine = ToolPolicyEngine(
      write_tools={'srv.tool'},
      approval_callback=lambda args, name: False,
    )
    allowed, reason, approved = engine.check(
      'srv', 'tool', {'qty': 5}, self._descriptor(),
    )
    assert not allowed
    assert approved is False
    assert 'not approved' in reason

  def test_write_with_approval_passes(self) -> None:
    engine = ToolPolicyEngine(
      write_tools={'srv.tool'},
      approval_callback=lambda args, name: True,
    )
    allowed, reason, approved = engine.check(
      'srv', 'tool', {'qty': 5}, self._descriptor(),
    )
    assert allowed
    assert reason == 'ok'
    assert approved is True

  def test_read_tool_needs_no_approval(self) -> None:
    engine = ToolPolicyEngine(write_tools=set())
    allowed, _, approved = engine.check(
      'srv', 'tool', {'qty': 5}, self._descriptor(),
    )
    assert allowed
    assert approved is True

  def test_sanitize_result_truncates(self) -> None:
    '''Oversized results are truncated to the policy cap.'''
    engine = ToolPolicyEngine(max_result_chars=50)
    text = 'z' * 500
    result = ToolCallResult(content=[{'type': 'text', 'text': text}])
    text = engine.sanitize_result(result)
    assert len(text) <= 50


class TestDiscovery:
  '''Hint-based server selection routes tasks to the right server.'''

  def test_retail_task_picks_retail_server(self) -> None:
    catalog = ServerCatalog()
    task = 'check retail inventory and restock store S01'
    picked = pick_servers(task, catalog)
    assert picked
    assert picked[0].name == 'retail-ops'

  def test_telecom_task_picks_telecom_server(self) -> None:
    catalog = ServerCatalog()
    task = 'which cell sites are degraded? dispatch a technician'
    picked = pick_servers(task, catalog)
    assert picked
    assert picked[0].name == 'telecom-ops'

  def test_unknown_task_falls_back_to_all(self) -> None:
    catalog = ServerCatalog()
    picked = pick_servers('xyzzy qwerty', catalog)
    assert len(picked) >= 1


class TestAgentStore:
  '''SQLite persistence for events, memory, and tool audits.'''

  def test_session_events_and_memory(self, tmp_path: Path) -> None:
    store = AgentStore(tmp_path / 'store.db')
    try:
      session = store.ensure_session('s1', meta={'k': 'v'})
      assert session.session_id == 's1'

      store.log_event('s1', 'e1', {'x': 1})
      store.log_event('s1', 'e2', {'x': 2})
      history = store.history('s1')
      assert [h.event_type for h in history] == ['e1', 'e2']

      store.remember('s1', 'facts', {'a': 1})
      assert store.recall('s1', 'facts') == {'a': 1}
      assert store.recall('s1', 'missing') is None

      store.log_tool_call(
        session_id='s1', server='srv', tool='t1', args={'q': 1},
        result={'r': 2}, ok=True, latency_ms=1.5,
      )
      calls = store.tool_calls('s1')
      assert len(calls) == 1
      assert calls[0].tool == 't1'
      assert calls[0].ok is True
    finally:
      store.close()

  def test_tool_call_audit_failure(self, tmp_path: Path) -> None:
    store = AgentStore(tmp_path / 'store2.db')
    try:
      store.log_tool_call(
        session_id='s', server='srv', tool='t', ok=False,
        error='boom', approved=False,
      )
      record = store.tool_calls('s')[0]
      assert record.ok is False
      assert record.error == 'boom'
      assert record.approved is False
    finally:
      store.close()
