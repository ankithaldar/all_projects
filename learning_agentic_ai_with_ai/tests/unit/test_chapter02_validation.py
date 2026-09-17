#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: hybrid self-validation (deterministic checks + critic).'''


from __future__ import annotations

import json

from agentic_common.gateway_client import MockGateway
from chapter02_planning.config import PlanningConfig
from chapter02_planning.llm import ReasoningLLM
from chapter02_planning.patterns.self_validation import (
  SelfValidator,
  check_answer_present,
  check_evidence_used,
  check_grounding,
  check_plan_outcomes,
  check_tool_outcomes,
)
from chapter02_planning.schemas import NodeResult, ToolExecutionAudit


def make_validator(planner=None, config=None) -> SelfValidator:
  '''Build a validator over a scripted gateway.

  Args:
    planner: Optional scripted planner; defaults to a pass-through.
    config: Optional PlanningConfig.

  Returns:
    SelfValidator.
  '''
  from llm_gateway.schemas import GatewayResponse

  def default_planner(_messages):
    return GatewayResponse(provider='mock', model='m', content='ok')

  llm = ReasoningLLM(
    MockGateway(planner or default_planner), config or PlanningConfig(),
  )
  return SelfValidator(llm, config or PlanningConfig())


def blocked_audit() -> ToolExecutionAudit:
  '''Build an audit for a blocked write call.

  Returns:
    ToolExecutionAudit.
  '''
  return ToolExecutionAudit(
    tool='retail-ops.retail_restock_order',
    args={'quantity': 9000},
    ok=False,
    approved=False,
    blocked=True,
    error='quantity exceeds approval cap',
  )


class TestDeterministicChecks:
  '''Free checks catch the common failure modes.'''

  def test_answer_present_flags_empty(self) -> None:
    issues = check_answer_present('')
    assert issues and issues[0].severity == 'error'

  def test_answer_present_info_on_short(self) -> None:
    issues = check_answer_present('168 h')
    assert issues and issues[0].severity == 'info'

  def test_grounding_flags_ungrounded_numbers(self) -> None:
    issues = check_grounding(
      'Order 999 units after selling 1234 units.', 'on_hand=4',
    )
    assert issues
    assert issues[0].check == 'grounding'
    assert issues[0].severity == 'warning'

  def test_grounding_passes_when_traceable(self) -> None:
    issues = check_grounding(
      'Order 110 units; demand 120 and buffer 10 and on hand 20.',
      'per_day=30 lead=4 buffer=10 on_hand=20 demand=120',
    )
    assert issues == []

  def test_blocked_write_must_be_acknowledged(self) -> None:
    issues = check_tool_outcomes('Restock order created successfully.', [
      blocked_audit(),
    ])
    assert any(issue.severity == 'error' for issue in issues)

  def test_blocked_write_acknowledged_passes(self) -> None:
    issues = check_tool_outcomes(
      'The restock was blocked; manual approval is required.',
      [blocked_audit()],
    )
    assert issues == []

  def test_failed_tool_warning(self) -> None:
    failed = ToolExecutionAudit(
      tool='srv.tool', ok=False, error='timeout',
    )
    issues = check_tool_outcomes('All good, here is the plan.', [failed])
    assert issues and issues[0].severity == 'warning'

  def test_evidence_used_warning(self) -> None:
    issues = check_evidence_used('react', [], '')
    assert issues and issues[0].check == 'evidence_used'
    assert check_evidence_used('react', [], 'some evidence') == []
    assert check_evidence_used('cot', [], '') == []

  def test_failed_plan_node_must_be_acknowledged(self) -> None:
    failed = NodeResult(node_id='step3', status='failed', errors=['empty'])
    skipped = NodeResult(
      node_id='step4', status='skipped', skipped_reason='dependency failed',
    )
    issues = check_plan_outcomes('Order 5 units placed.', [failed, skipped])
    assert issues and issues[0].severity == 'error'
    assert issues[0].check == 'plan_outcomes'

  def test_acknowledged_plan_failure_passes(self) -> None:
    failed = NodeResult(node_id='step3', status='failed', errors=['empty'])
    issues = check_plan_outcomes(
      'Step 3 could not be completed; manual follow-up required.', [failed],
    )
    assert issues == []


class TestSelfValidator:
  '''Validator orchestration, critic usage, and revision.'''

  def test_clean_answer_passes_without_critic(self) -> None:
    validator = make_validator()
    report = validator.validate(
      task='Report status.',
      answer='Status is healthy with 4 sites checked.',
      evidence='4 sites healthy',
      route='react',
      tool_calls=[
        ToolExecutionAudit(tool='srv.status', ok=True),
      ],
    )
    assert report.passed
    assert report.critic_used is False
    assert report.errors == []

  def test_critic_runs_on_warning_and_can_pass(self) -> None:
    def planner(messages):
      from llm_gateway.schemas import GatewayResponse
      system = messages[0].get('content', '') if messages else ''
      if 'strict but fair validator' in system:
        return GatewayResponse(
          provider='mock', model='m',
          content=json.dumps({'verdict': 'pass', 'issues': []}),
        )
      return GatewayResponse(provider='mock', model='m', content='ok')

    validator = make_validator(planner)
    report = validator.validate(
      task='Restock.',
      answer='Order 999 units for 12 stores.',
      evidence='nothing useful',
      route='direct',
      tool_calls=[],
    )
    assert report.critic_used is True
    assert report.passed is True
    assert any(issue.check == 'grounding' for issue in report.issues)

  def test_error_triggers_revision(self) -> None:
    def planner(messages):
      from llm_gateway.schemas import GatewayResponse
      system = messages[0].get('content', '') if messages else ''
      if 'You revise an operations answer' in system:
        return GatewayResponse(
          provider='mock', model='m',
          content='The restock was blocked; manual approval is required.',
        )
      return GatewayResponse(provider='mock', model='m', content='ok')

    validator = make_validator(planner)
    report = validator.validate(
      task='Restock S01.',
      answer='Restock order created successfully.',
      evidence='',
      route='react',
      tool_calls=[blocked_audit()],
    )
    assert report.revised is True
    assert report.rounds == 1
    assert 'blocked' in report.final_answer.lower()
    assert report.passed is True

  def test_critic_failure_marks_unavailable(self) -> None:
    def planner(_messages):
      raise RuntimeError('provider down')

    validator = make_validator(planner)
    report = validator.validate(
      task='x',
      answer='Order 12 units for 3 stores.',
      evidence='nothing matches',
      route='direct',
      tool_calls=[],
    )
    assert report.critic_unavailable is True
    assert report.errors
    assert report.final_answer

  def test_no_revision_when_disabled(self) -> None:
    config = PlanningConfig(max_validation_rounds=0)
    validator = make_validator(config=config)
    report = validator.validate(
      task='Restock.',
      answer='All done, order placed.',
      evidence='',
      route='react',
      tool_calls=[blocked_audit()],
    )
    assert report.revised is False
    assert report.passed is False

  def test_garbage_revision_is_rejected(self) -> None:
    '''A revision that ignores a failed plan node must not replace it.'''
    def planner(messages):
      from llm_gateway.schemas import GatewayResponse
      system = messages[0].get('content', '') if messages else ''
      if 'You revise an operations answer' in system:
        return GatewayResponse(
          provider='mock', model='m', content='User Safety: safe',
        )
      return GatewayResponse(provider='mock', model='m', content='ok')

    config = PlanningConfig(critic_enabled=False)
    validator = make_validator(planner, config)
    failed = NodeResult(node_id='step3', status='failed', errors=['empty'])
    original = 'Order 5 units placed for store S02.'
    report = validator.validate(
      task='Restock.',
      answer=original,
      evidence='S02 R-103 order 5',
      route='plan',
      tool_calls=[],
      node_results=[failed],
    )
    assert report.revision_rejected is True
    assert report.revised is False
    assert report.final_answer == original
    assert report.passed is False
