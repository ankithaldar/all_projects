#!/usr/bin/env python
# -- coding: utf-8 --

'''Chapter 2 eval cases and scoring extensions.

The shared harness scores task success, safety, and reliability from the
final outcome. Chapter 2 adds assertions about *how* the outcome was
produced: the chosen route, whether validation passed, how much of the plan
completed, and whether forbidden writes were blocked. These are the
regression tests that protect the router and the safety gate.
'''


from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agentic_common.eval.harness import (
  CheckResult,
  EvalCase,
  RunOutcome,
  evaluate_case,
)
from chapter02_planning.schemas import ReasoningRoute


class PlanningEvalCase(EvalCase):
  '''Eval case with Chapter 2 specific expectations.

  Attributes:
    expect_route: Route the policy must choose.
    expect_validation_passed: Expected validation verdict.
    plan_nodes_completed_min: Minimum completed plan nodes.
    expect_blocked_tool: Qualified tool that must have a blocked audit.
    min_tool_calls: Minimum number of tool calls.
    allow_writes: Whether the run may use write tools.
    cot_samples: Self-consistency samples for CoT cases.
  '''

  expect_route: Optional[ReasoningRoute] = None
  expect_validation_passed: Optional[bool] = None
  plan_nodes_completed_min: Optional[int] = None
  expect_blocked_tool: Optional[str] = None
  min_tool_calls: Optional[int] = None
  allow_writes: bool = False
  cot_samples: int = 1


def extra_checks(
  case: PlanningEvalCase,
  outcome: RunOutcome,
) -> List[CheckResult]:
  '''Evaluate Chapter 2 specific assertions.

  Args:
    case: The eval case.
    outcome: Observed run outcome (`extra` carries route/validation/plan).

  Returns:
    List of CheckResult for route, validation, plan, and safety shape.
  '''
  checks: List[CheckResult] = []
  extra: Dict[str, Any] = outcome.extra or {}

  if case.expect_route is not None:
    route = str(extra.get('route', 'unknown'))
    checks.append(
      CheckResult(
        check=f'route:{case.expect_route}',
        passed=route == case.expect_route,
        detail=f'observed route: {route}',
      )
    )

  if case.expect_validation_passed is not None:
    validation = extra.get('validation_passed')
    checks.append(
      CheckResult(
        check='validation:expected',
        passed=validation == case.expect_validation_passed,
        detail=f'validation_passed={validation}',
      )
    )

  if case.plan_nodes_completed_min is not None:
    completed = int(extra.get('plan_nodes_completed', 0))
    checks.append(
      CheckResult(
        check=f'plan_nodes>={case.plan_nodes_completed_min}',
        passed=completed >= case.plan_nodes_completed_min,
        detail=f'completed nodes: {completed}',
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

  if case.min_tool_calls is not None:
    observed = len(outcome.tool_calls)
    checks.append(
      CheckResult(
        check=f'tool_calls>={case.min_tool_calls}',
        passed=observed >= case.min_tool_calls,
        detail=f'tool calls: {observed}',
      )
    )

  return checks


def evaluate_planning_case(
  case: PlanningEvalCase,
  outcome: RunOutcome,
) -> Any:
  '''Score one Chapter 2 case (shared checks + chapter checks).

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
  cases: List[PlanningEvalCase],
  task_runner: Callable[[PlanningEvalCase], RunOutcome],
) -> List[Any]:
  '''Run a Chapter 2 suite.

  Args:
    cases: Cases to run.
    task_runner: Callable executing one case (receives the case so it can
      honor `allow_writes` and `cot_samples`).

  Returns:
    List of CaseResult.
  '''
  results = []
  for case in cases:
    try:
      outcome = task_runner(case)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      outcome = RunOutcome(errors=[f'crashed: {exc}'])
    results.append(evaluate_planning_case(case, outcome))
  return results
