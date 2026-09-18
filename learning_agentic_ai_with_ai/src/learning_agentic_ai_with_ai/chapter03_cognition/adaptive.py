#!/usr/bin/env python
# -- coding: utf-8 --

'''Adaptive planning loop: bounded retry, tool switching, escalation.

The adaptive loop wraps a single plan step and decides *how* to react to
failure, using only the knobs the active strategy profile allows:

    attempt 1 ──ok──▶ done
        │
      fail ──▶ classify error
        │        ├─ blocked / invalid_arguments ──▶ stop (no retry helps)
        │        ├─ unknown_tool ──▶ try fallback tool (tool switching)
        │        └─ transient / empty_output ──▶ retry up to the cap
        │
        └─ attempts exhausted + fallback available ──▶ tool switch
        └─ still failing ──▶ failed (dependents skip; independents run)

Every adaptation is recorded as an `AdaptationRecord` so the run can explain
itself ("step trend: retried twice, then switched to low_stock_report").
Whole-plan replans and strategy escalation are decided one level up, in the
ActionLayer/agent, because they affect more than one step.
'''


from __future__ import annotations

from typing import Callable, Dict, List

from agentic_common.logging import get_logger, log_event
from chapter03_cognition.config import CognitionConfig
from chapter03_cognition.schemas import (
  AdaptationRecord,
  CodeRunResult,
  ErrorClass,
  PlanStep,
  StepAttempt,
  StepResult,
  StrategyProfile,
  ToolResult,
)

logger = get_logger(__name__)

AttemptFn = Callable[
  [PlanStep, str, Dict[str, object], int], StepAttempt,
]

_RETRYABLE: frozenset = frozenset({
  'transient', 'empty_output', 'timeout', 'gateway_error',
})
_TERMINAL: frozenset = frozenset({
  'blocked', 'invalid_arguments', 'unknown_tool', 'code_error',
})


def classify_tool_error(result: ToolResult) -> ErrorClass:
  '''Classify a failed tool result.

  Args:
    result: Failed ToolResult.

  Returns:
    ErrorClass value.
  '''
  if result.ok:
    return 'none'
  error = (result.error or result.text or '').lower()
  if result.text.startswith('BLOCKED by safety gate'):
    return 'blocked'
  if 'unknown tool' in error:
    return 'unknown_tool'
  if 'invalid arguments' in error or 'invalid params' in error:
    return 'invalid_arguments'
  if 'injected transient' in error:
    return 'transient'
  if any(
    marker in error
    for marker in ('timeout', 'timed out', 'temporarily', 'unavailable')
  ):
    return 'transient'
  if not (result.text or '').strip():
    return 'empty_output'
  return 'tool_error'


def classify_code_error(result: CodeRunResult) -> ErrorClass:
  '''Return the error class of a failed code run.

  Args:
    result: Code run result.

  Returns:
    ErrorClass value.
  '''
  if result.ok:
    return 'none'
  return result.error_class


class AdaptiveLoop:
  '''Bounded retry/tool-switch executor for one plan step.'''

  def __init__(
    self,
    profile: StrategyProfile,
    config: CognitionConfig,
  ) -> None:
    '''Initialize the loop.

    Args:
      profile: Active strategy profile.
      config: Chapter 3 configuration.
    '''
    self._profile = profile
    self._config = config

  def run_step(
    self,
    step: PlanStep,
    attempt_fn: AttemptFn,
    adaptations: List[AdaptationRecord],
  ) -> StepResult:
    '''Execute one step with bounded adaptation.

    Args:
      step: Step to execute.
      attempt_fn: Callable performing one attempt.
      adaptations: Mutable adaptation log (shared with the run).

    Returns:
      StepResult; never raises.
    '''
    max_attempts = min(
      step.max_attempts or self._profile.retry.max_attempts_per_step,
      self._config.max_attempts_per_step * 2,
    )
    candidates = [(step.tool, dict(step.arguments))]
    if (
      self._profile.retry.tool_switch
      and step.fallback_tool
      and step.fallback_tool != step.tool
    ):
      fallback_args = dict(step.fallback_arguments) or dict(step.arguments)
      candidates.append((step.fallback_tool, fallback_args))

    attempts: List[StepAttempt] = []
    for tool, args in candidates:
      switched = tool != step.tool
      for attempt_index in range(1, max_attempts + 1):
        attempt = attempt_fn(step, tool, args, attempt_index)
        if switched:
          attempt.switched_from = step.tool
        attempts.append(attempt)

        if attempt.ok and attempt.observation_preview:
          return self._completed(step, tool, attempts)

        error_class = attempt.error_class
        log_event(
          logger, 30, 'step_attempt_failed',
          step_id=step.id, tool=tool, attempt=attempt_index,
          error_class=error_class,
        )
        if error_class in _TERMINAL:
          if error_class in ('blocked', 'invalid_arguments', 'code_error'):
            return self._failed(step, attempts)
          break
        if error_class in _RETRYABLE and attempt_index < max_attempts:
          adaptations.append(
            AdaptationRecord(
              kind='retry',
              step_id=step.id,
              attempt=attempt_index,
              reason=error_class,
              detail=f'retrying {tool} after {error_class}',
            )
          )
          continue
        break

      if switched:
        continue
      if len(candidates) > 1:
        adaptations.append(
          AdaptationRecord(
            kind='tool_switch',
            step_id=step.id,
            attempt=max_attempts,
            reason=attempts[-1].error_class if attempts else 'unknown',
            detail=f'switching to {candidates[1][0]}',
          )
        )

    errors = [
      attempt.error for attempt in attempts if attempt.error
    ]
    log_event(
      logger, 30, 'step_failed',
      step_id=step.id, attempts=len(attempts), errors=len(errors),
    )
    return self._failed(step, attempts)

  @staticmethod
  def _failed(
    step: PlanStep,
    attempts: List[StepAttempt],
  ) -> StepResult:
    '''Build a failed StepResult.

    Args:
      step: Executed step.
      attempts: All attempts made.

    Returns:
      StepResult with status failed.
    '''
    errors = [
      attempt.error for attempt in attempts if attempt.error
    ]
    return StepResult(
      step_id=step.id,
      status='failed',
      attempts=attempts,
      errors=errors or ['step failed without an error message'],
    )

  @staticmethod
  def _completed(
    step: PlanStep,
    tool: str,
    attempts: List[StepAttempt],
  ) -> StepResult:
    '''Build a completed StepResult.

    Args:
      step: Executed step.
      tool: Tool that succeeded.
      attempts: All attempts made.

    Returns:
      StepResult with status completed.
    '''
    return StepResult(
      step_id=step.id,
      status='completed',
      output=attempts[-1].observation_preview,
      tool_used=tool,
      attempts=attempts,
    )
