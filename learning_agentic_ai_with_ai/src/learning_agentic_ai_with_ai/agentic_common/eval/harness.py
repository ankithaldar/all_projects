#!/usr/bin/env python
# -- coding: utf-8 --

'''Generic evaluation harness.

An *eval case* binds a natural-language task to machine-checkable assertions:
- task success: did the agent produce the expected decision/answer?
- safety: were forbidden tools called? did write tools get approval?
- reliability: did the run converge within the iteration/token budget?

The harness is agent-agnostic: it consumes a `task_runner` callable that
returns a `RunOutcome`, so every chapter plugs its own agent in.
'''


from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from agentic_common.logging import get_logger, log_event
from agentic_common.tracing import TokenUsage

logger = get_logger(__name__)


class CheckResult(BaseModel):
  '''Result of one assertion inside an eval case.'''

  model_config = ConfigDict(extra='ignore')

  check: str
  passed: bool
  detail: str = ''


class RunOutcome(BaseModel):
  '''What a single agent run produced (the harness only understands this).'''

  model_config = ConfigDict(extra='ignore')

  answer: str = ''
  tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
  iterations: int = 0
  usage: TokenUsage = Field(default_factory=TokenUsage)
  errors: List[str] = Field(default_factory=list)
  extra: Dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
  '''One evaluation scenario.

  Attributes:
    id: Unique case id.
    task: Natural language task given to the agent.
    expect_tool_called: Tool names that must appear in the run.
    expect_tool_not_called: Tool names that must NOT appear.
    expect_answer_contains: Substrings (case-insensitive) in the final answer.
    expect_answer_regex: Regexes the answer must match.
    max_iterations: Reliability bound for the loop.
    max_total_tokens: Reliability token budget.
    require_zero_errors: Whether errors list must be empty.
    metadata: Free-form context (difficulty, use case).
  '''

  model_config = ConfigDict(extra='ignore')

  id: str
  task: str
  expect_tool_called: List[str] = Field(default_factory=list)
  expect_tool_not_called: List[str] = Field(default_factory=list)
  expect_answer_contains: List[str] = Field(default_factory=list)
  expect_answer_regex: List[str] = Field(default_factory=list)
  max_iterations: int = 8
  max_tokens: int = 60000
  require_zero_errors: bool = True
  metadata: Dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
  '''Outcome of evaluating one case.'''

  model_config = ConfigDict(extra='ignore')

  case_id: str
  passed: bool
  checks: List[CheckResult] = Field(default_factory=list)
  answer: str = ''
  iterations: int = 0
  total_tokens: int = 0
  duration_ms: float = 0.0
  errors: List[str] = Field(default_factory=list)


class EvalReport(BaseModel):
  '''Aggregate report over a suite of cases.'''

  model_config = ConfigDict(extra='ignore')

  generated_at: str = Field(
    default_factory=lambda: datetime.now(timezone.utc).isoformat(),
  )
  total_cases: int = 0
  passed_cases: int = 0
  task_success_rate: float = 0.0
  safety_failures: int = 0
  reliability_failures: int = 0
  avg_iterations: float = 0.0
  avg_tokens: float = 0.0
  results: List[CaseResult] = Field(default_factory=list)

  def summary_lines(self) -> List[str]:
    '''Render a human-readable summary.

    Returns:
      Summary text lines.
    '''
    return [
      f'cases: {self.passed_cases}/{self.total_cases} passed',
      f'task success rate: {self.task_success_rate:.0%}',
      f'safety failures: {self.safety_failures}',
      f'reliability failures: {self.reliability_failures}',
      f'avg iterations: {self.avg_iterations:.1f}  avg tokens: {self.avg_tokens:.0f}',
    ]


def _tool_names(outcome: RunOutcome) -> List[str]:
  '''Extract tool names from an outcome.

  Args:
    outcome: The run outcome.

  Returns:
    List of tool names in call order.
  '''
  return [str(call.get('tool', '')) for call in outcome.tool_calls]


def evaluate_case(case: EvalCase, outcome: RunOutcome) -> CaseResult:
  '''Score one outcome against one case.

  Args:
    case: The eval case with expectations.
    outcome: The observed agent run.

  Returns:
    CaseResult with one CheckResult per expectation.
  '''
  checks: List[CheckResult] = []
  called = _tool_names(outcome)
  answer_lower = (outcome.answer or '').lower()

  for tool in case.expect_tool_called:
    checks.append(
      CheckResult(
        check=f'tool_called:{tool}',
        passed=tool in called,
        detail='observed tools: ' + (', '.join(called) or 'none'),
      )
    )

  for tool in case.expect_tool_not_called:
    checks.append(
      CheckResult(
        check=f'tool_not_called:{tool}',
        passed=tool not in called,
        detail='observed tools: ' + (', '.join(called) or 'none'),
      )
    )

  for snippet in case.expect_answer_contains:
    checks.append(
      CheckResult(
        check=f'answer_contains:{snippet[:40]}',
        passed=snippet.lower() in answer_lower,
        detail='answer: ' + answer_lower[:200],
      )
    )

  import re

  for pattern in case.expect_answer_regex:
    try:
      matched = re.search(pattern, outcome.answer or '') is not None
    except re.error as exc:
      matched = False
      checks.append(
        CheckResult(check=f'answer_regex:{pattern}', passed=False, detail=f'bad regex: {exc}')
      )
      continue
    checks.append(
      CheckResult(
        check=f'answer_regex:{pattern[:40]}',
        passed=matched,
        detail='answer: ' + answer_lower[:80],
      )
    )

  reliability_ok = (
    outcome.iterations <= case.max_iterations
    and outcome.usage.total_tokens <= case.max_tokens
  )
  checks.append(
    CheckResult(
      check='reliability:budget',
      passed=reliability_ok,
      detail=(
        f'iterations={outcome.iterations}/{case.max_iterations} '
        f'tokens={outcome.usage.total_tokens}/{case.max_tokens}'
      ),
    )
  )

  safety_ok = not outcome.errors
  if case.require_zero_errors:
    checks.append(
      CheckResult(
        check='safety:no_errors',
        passed=safety_ok,
        detail='; '.join(outcome.errors) or 'no errors',
      )
    )

  passed = all(check.passed for check in checks)
  return CaseResult(
    case_id=case.id,
    passed=passed,
    checks=checks,
    answer=outcome.answer or '',
    iterations=outcome.iterations,
    total_tokens=outcome.usage.total_tokens,
    errors=list(outcome.errors),
  )


TaskRunner = Callable[[str], RunOutcome]


def run_eval_suite(
  cases: List[EvalCase],
  task_runner: TaskRunner,
  report_path: Optional[Path] = None,
) -> EvalReport:
  '''Run a full eval suite and produce a report.

  Args:
    cases: Eval cases to run.
    task_runner: Callable executing one task through the agent.
    report_path: Optional path to write the JSON report.

  Returns:
    The aggregate EvalReport.
  '''
  results: List[CaseResult] = []

  for case in cases:
    started = time.perf_counter()
    log_event(logger, 20, 'eval_case_start', case_id=case.id, task=case.task)
    try:
      outcome = task_runner(case.task)
    except Exception as exc:
      log_event(logger, 40, 'eval_case_crashed', case_id=case.id, error=str(exc))
      outcome = RunOutcome(errors=[f'crashed: {exc}'])

    result = evaluate_case(case, outcome)
    result.duration_ms = (time.perf_counter() - started) * 1000.0
    results.append(result)
    log_event(
      logger,
      20,
      'eval_case_done',
      case_id=case.id,
      passed=result.passed,
      tokens=result.total_tokens,
      iterations=result.iterations,
    )

  total = len(results)
  passed = sum(1 for r in results if r.passed)
  safety_failures = sum(
    1 for r in results for c in r.checks if c.check.startswith('safety:') and not c.passed
  )
  reliability_failures = sum(
    1
    for r in results
    for c in r.checks
    if c.check.startswith('reliability:') and not c.passed
  )

  report = EvalReport(
    total_cases=total,
    passed_cases=passed,
    task_success_rate=(passed / total) if total else 0.0,
    safety_failures=safety_failures,
    reliability_failures=reliability_failures,
    avg_iterations=(sum(r.iterations for r in results) / total) if total else 0.0,
    avg_tokens=(sum(r.total_tokens for r in results) / total) if total else 0.0,
    results=results,
  )

  if report_path is not None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
      json.dumps(report.model_dump(mode='json'), indent=2, default=str),
      encoding='utf-8',
    )

  return report
