#!/usr/bin/env python
# -- coding: utf-8 --

'''Chapter 3 eval cases and scoring extensions.

The shared harness scores task success, safety, and reliability from the
final outcome. Chapter 3 adds assertions about the *architecture*: which
strategy ran, which plan shape was used, how the agent adapted, whether
sandboxed code ran, whether writes were blocked, and whether memory learned
from earlier runs. These are the regression tests that protect the
cognitive pipeline.
'''


from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agentic_common.eval.harness import (
  CheckResult,
  EvalCase,
  RunOutcome,
  evaluate_case,
)


class CognitionEvalCase(EvalCase):
  '''Eval case with Chapter 3 specific expectations.

  Attributes:
    expect_strategy: Strategy profile that must be selected.
    expect_plan_source: Plan source ('dag', 'code', 'fallback').
    expect_status: Expected run status.
    expect_adaptation_kinds: Adaptation kinds that must appear.
    expect_blocked_tool: Qualified tool that must have a blocked audit.
    expect_code_runner: Sandbox used for code plans.
    expect_percept_domain: Detected domain.
    expect_capability: Capability the percept must declare.
    plan_mode: Planner override for the case.
    allow_writes: Whether writes are permitted.
    inject_faults: Whether to inject tool failures.
    repeat: Number of runs (later runs see earlier memory).
    memory_episodes_min: Minimum recalled episodes on the final run.
    code_runner: Sandbox override for the case.
  '''

  expect_strategy: Optional[str] = None
  expect_plan_source: Optional[str] = None
  expect_status: Optional[str] = None
  expect_adaptation_kinds: List[str] = []
  expect_blocked_tool: Optional[str] = None
  expect_code_runner: Optional[str] = None
  expect_percept_domain: Optional[str] = None
  expect_capability: Optional[str] = None
  plan_mode: str = 'auto'
  allow_writes: bool = False
  inject_faults: bool = False
  no_tools: bool = False
  repeat: int = 1
  memory_episodes_min: Optional[int] = None
  code_runner: Optional[str] = None


def extra_checks(
  case: CognitionEvalCase,
  outcome: RunOutcome,
) -> List[CheckResult]:
  '''Evaluate Chapter 3 specific assertions.

  Args:
    case: The eval case.
    outcome: Observed run outcome (`extra` carries architecture metadata).

  Returns:
    List of CheckResult for strategy, plan, adaptation, safety, and memory.
  '''
  checks: List[CheckResult] = []
  extra: Dict[str, Any] = outcome.extra or {}

  if case.expect_strategy is not None:
    observed = str(extra.get('strategy', 'unknown'))
    checks.append(
      CheckResult(
        check=f'strategy:{case.expect_strategy}',
        passed=observed == case.expect_strategy,
        detail=f'observed strategy: {observed}',
      )
    )

  if case.expect_plan_source is not None:
    observed = str(extra.get('plan_source', 'unknown'))
    checks.append(
      CheckResult(
        check=f'plan_source:{case.expect_plan_source}',
        passed=observed == case.expect_plan_source,
        detail=f'observed plan source: {observed}',
      )
    )

  if case.expect_status is not None:
    observed = str(extra.get('status', 'unknown'))
    checks.append(
      CheckResult(
        check=f'status:{case.expect_status}',
        passed=observed == case.expect_status,
        detail=f'observed status: {observed}',
      )
    )

  for kind in case.expect_adaptation_kinds:
    observed = [str(item) for item in extra.get('adaptations', [])]
    checks.append(
      CheckResult(
        check=f'adaptation:{kind}',
        passed=kind in observed,
        detail='observed adaptations: ' + (', '.join(observed) or 'none'),
      )
    )

  if case.expect_blocked_tool is not None:
    blocked = [str(name) for name in extra.get('blocked_tools', [])]
    checks.append(
      CheckResult(
        check=f'blocked:{case.expect_blocked_tool}',
        passed=case.expect_blocked_tool in blocked,
        detail='blocked tools: ' + (', '.join(blocked) or 'none'),
      )
    )

  if case.expect_code_runner is not None:
    observed = str(extra.get('code_runner', 'none'))
    checks.append(
      CheckResult(
        check=f'code_runner:{case.expect_code_runner}',
        passed=observed == case.expect_code_runner,
        detail=f'observed runner: {observed}',
      )
    )

  if case.expect_percept_domain is not None:
    observed = str(extra.get('percept_domain', 'unknown'))
    checks.append(
      CheckResult(
        check=f'percept_domain:{case.expect_percept_domain}',
        passed=observed == case.expect_percept_domain,
        detail=f'observed domain: {observed}',
      )
    )

  if case.expect_capability is not None:
    observed = [str(item) for item in extra.get('capabilities', [])]
    checks.append(
      CheckResult(
        check=f'capability:{case.expect_capability}',
        passed=case.expect_capability in observed,
        detail='observed capabilities: ' + (', '.join(observed) or 'none'),
      )
    )

  if case.memory_episodes_min is not None:
    recalled = int(extra.get('memory_episodes', 0))
    checks.append(
      CheckResult(
        check=f'memory_episodes>={case.memory_episodes_min}',
        passed=recalled >= case.memory_episodes_min,
        detail=f'recalled episodes: {recalled}',
      )
    )

  return checks


def evaluate_cognition_case(
  case: CognitionEvalCase,
  outcome: RunOutcome,
) -> Any:
  '''Score one Chapter 3 case (shared checks + chapter checks).

  Args:
    case: The eval case.
    outcome: Observed run outcome.

  Returns:
    CaseResult with combined checks and recomputed pass/fail.
  '''
  result = evaluate_case(case, outcome)
  result.checks.extend(extra_checks(case, outcome))
  result.passed = all(check.passed for check in result.checks)
  return result


def run_checks(
  cases: List[CognitionEvalCase],
  task_runner: Callable[[CognitionEvalCase], RunOutcome],
) -> List[Any]:
  '''Run a Chapter 3 suite.

  Args:
    cases: Cases to run.
    task_runner: Callable executing one case (receives the case so it can
      honor plan_mode, strategy, faults, and repeats).

  Returns:
    List of CaseResult.
  '''
  results = []
  for case in cases:
    try:
      outcome = task_runner(case)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      outcome = RunOutcome(errors=[f'crashed: {exc}'])
    results.append(evaluate_cognition_case(case, outcome))
  return results
