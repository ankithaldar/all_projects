#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: the adaptive loop (retry, tool switching, bounds).'''


from __future__ import annotations

from typing import List

from chapter03_cognition.adaptive import (
  AdaptiveLoop,
  classify_code_error,
  classify_tool_error,
)
from chapter03_cognition.config import CognitionConfig
from chapter03_cognition.schemas import (
  AdaptationRecord,
  CodeRunResult,
  PlanStep,
  StepAttempt,
  ToolResult,
)
from chapter03_cognition.strategies import profile_for


def failing_attempt(
  error_class: str,
  error: str = 'boom',
) -> StepAttempt:
  '''Build a failed attempt.

  Args:
    error_class: Error classification.
    error: Error message.

  Returns:
    StepAttempt.
  '''
  return StepAttempt(
    attempt=1, tool='t', ok=False, error_class=error_class, error=error,
  )


def ok_attempt() -> StepAttempt:
  '''Build a successful attempt.

  Returns:
    StepAttempt.
  '''
  return StepAttempt(
    attempt=1, tool='t', ok=True, error_class='none',
    observation_preview='data',
  )


class TestClassification:
  '''Error classification drives adaptation decisions.'''

  def test_blocked_result(self) -> None:
    result = ToolResult(
      tool='srv.w', ok=False, text='BLOCKED by safety gate: nope',
      error='nope',
    )
    assert classify_tool_error(result) == 'blocked'

  def test_unknown_tool(self) -> None:
    result = ToolResult(
      tool='srv.x', ok=False, text='tool error: unknown tool srv.x',
      error='unknown tool srv.x',
    )
    assert classify_tool_error(result) == 'unknown_tool'

  def test_transient(self) -> None:
    result = ToolResult(
      tool='srv.t', ok=False,
      text='tool error: injected transient failure',
      error='injected transient failure',
    )
    assert classify_tool_error(result) == 'transient'

  def test_timeout_is_transient(self) -> None:
    result = ToolResult(
      tool='srv.t', ok=False, text='tool error: request timed out',
      error='request timed out',
    )
    assert classify_tool_error(result) == 'transient'

  def test_empty_output(self) -> None:
    result = ToolResult(tool='srv.t', ok=False, text='', error=None)
    assert classify_tool_error(result) == 'empty_output'

  def test_generic_tool_error(self) -> None:
    result = ToolResult(
      tool='srv.t', ok=False, text='tool error: bad sku', error='bad sku',
    )
    assert classify_tool_error(result) == 'tool_error'

  def test_code_error(self) -> None:
    result = CodeRunResult(ok=False, error_class='timeout')
    assert classify_code_error(result) == 'timeout'
    assert classify_code_error(CodeRunResult(ok=True)) == 'none'


class TestAdaptiveLoop:
  '''Bounded retry and tool switching inside one step.'''

  @staticmethod
  def step(
    tool: str = 'srv.primary',
    fallback: str = '',
  ) -> PlanStep:
    '''Build a step for the loop.

    Args:
      tool: Primary tool.
      fallback: Fallback tool.

    Returns:
      PlanStep.
    '''
    return PlanStep(
      id='s', objective='do it', tool=tool,
      fallback_tool=fallback, fallback_arguments={'alt': True},
    )

  def test_retry_then_success(self) -> None:
    attempts: List[StepAttempt] = []

    def attempt_fn(step, tool, args, attempt):
      attempts.append(failing_attempt('transient'))
      if len(attempts) < 2:
        return attempts[-1]
      return ok_attempt()

    adaptations: List[AdaptationRecord] = []
    result = AdaptiveLoop(
      profile_for('exploratory'), CognitionConfig(),
    ).run_step(self.step(), attempt_fn, adaptations)
    assert result.status == 'completed'
    assert any(record.kind == 'retry' for record in adaptations)

  def test_tool_switch_after_exhaustion(self) -> None:
    calls: List[str] = []

    def attempt_fn(step, tool, args, attempt):
      calls.append(tool)
      if tool == 'srv.primary':
        return failing_attempt('transient')
      return ok_attempt()

    adaptations: List[AdaptationRecord] = []
    result = AdaptiveLoop(
      profile_for('exploratory'), CognitionConfig(),
    ).run_step(
      self.step(fallback='srv.fallback'), attempt_fn, adaptations,
    )
    assert result.status == 'completed'
    assert result.tool_used == 'srv.fallback'
    assert calls.count('srv.primary') == 3
    assert any(record.kind == 'tool_switch' for record in adaptations)

  def test_conservative_does_not_retry(self) -> None:
    calls: List[str] = []

    def attempt_fn(step, tool, args, attempt):
      calls.append(tool)
      return failing_attempt('transient')

    adaptations: List[AdaptationRecord] = []
    result = AdaptiveLoop(
      profile_for('conservative'), CognitionConfig(),
    ).run_step(self.step(), attempt_fn, adaptations)
    assert result.status == 'failed'
    assert len(calls) == 1
    assert adaptations == []

  def test_blocked_is_terminal(self) -> None:
    calls: List[str] = []

    def attempt_fn(step, tool, args, attempt):
      calls.append(tool)
      return failing_attempt('blocked', 'write not approved')

    adaptations: List[AdaptationRecord] = []
    result = AdaptiveLoop(
      profile_for('exploratory'), CognitionConfig(),
    ).run_step(
      self.step(fallback='srv.fallback'), attempt_fn, adaptations,
    )
    assert result.status == 'failed'
    assert calls == ['srv.primary']
    assert any(
      attempt.error_class == 'blocked' for attempt in result.attempts
    )

  def test_unknown_tool_switches_without_retry(self) -> None:
    calls: List[str] = []

    def attempt_fn(step, tool, args, attempt):
      calls.append(tool)
      if tool == 'srv.primary':
        return failing_attempt('unknown_tool')
      return ok_attempt()

    adaptations: List[AdaptationRecord] = []
    result = AdaptiveLoop(
      profile_for('exploratory'), CognitionConfig(),
    ).run_step(
      self.step(fallback='srv.fallback'), attempt_fn, adaptations,
    )
    assert result.status == 'completed'
    assert calls == ['srv.primary', 'srv.fallback']

  def test_per_step_attempt_override(self) -> None:
    calls: List[str] = []

    def attempt_fn(step, tool, args, attempt):
      calls.append(tool)
      return failing_attempt('transient')

    step = self.step().model_copy(update={'max_attempts': 2})
    adaptations: List[AdaptationRecord] = []
    AdaptiveLoop(
      profile_for('exploratory'), CognitionConfig(),
    ).run_step(step, attempt_fn, adaptations)
    assert len(calls) == 2
