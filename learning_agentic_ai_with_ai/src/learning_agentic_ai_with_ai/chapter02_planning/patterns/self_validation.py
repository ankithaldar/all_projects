#!/usr/bin/env python
# -- coding: utf-8 --

'''Self-validation: agents that check their own work before answering.

Pattern: HYBRID VALIDATION (deterministic checks first, LLM critic second)

    candidate answer + evidence + tool audit
        │
        ├─ 1. deterministic checks (free, exact, always on)
        │      grounding / policy / failure acknowledgement / non-empty
        ├─ 2. LLM critic (costly, only when needed)
        │      structured Critique {verdict, issues, revised_answer}
        └─ 3. bounded revision round, then re-run deterministic checks
        │
        └─▶ ValidationReport {passed, issues, revised, final_answer}

Engineering rationale: LLM judges are useful but slower, costlier, and
themselves fallible. Deterministic checks - "every number in the answer
must appear in evidence", "a blocked write must be acknowledged", "a failed
tool must not be reported as success" - catch the majority of real incidents
for free. The critic only runs when those checks flag something or the route
is high-stakes (plan). If the critic is unavailable, the report says so
instead of pretending the answer was verified.
'''


from __future__ import annotations

from typing import List, Optional

from agentic_common.logging import get_logger, log_event
from agentic_common.tracing import TokenUsage
from chapter02_planning.config import PlanningConfig
from chapter02_planning.llm import ReasoningLLM, usage_delta
from chapter02_planning.parsing import extract_number_tokens
from chapter02_planning.prompting import (
  CRITIC_PROMPT,
  CRITIC_SYSTEM,
  REVISION_PROMPT,
  REVISION_SYSTEM,
  output_contract,
)
from chapter02_planning.schemas import (
  Critique,
  NodeResult,
  ReasoningRoute,
  ToolExecutionAudit,
  ValidationIssue,
  ValidationReport,
)

logger = get_logger(__name__)

_ACKNOWLEDGEMENT_WORDS = (
  'blocked', 'not approved', 'could not', 'couldn\'t', 'unable', 'failed',
  'manual', 'escalat', 'denied', 'disabled', 'not completed', 'incomplete',
  'skipped', 'partial', 'unresolved',
)

_MIN_GROUNDED_NUMBERS = 2


def check_answer_present(answer: str) -> List[ValidationIssue]:
  '''Check that the answer is non-empty and substantive.

  Args:
    answer: Candidate answer.

  Returns:
    List of issues (empty when fine).
  '''
  text = (answer or '').strip()
  if not text:
    return [
      ValidationIssue(
        check='answer_present',
        severity='error',
        detail='answer is empty',
        suggestion='produce a best-effort answer or state what is missing',
      )
    ]
  if len(text) < 8:
    return [
      ValidationIssue(
        check='answer_present',
        severity='info',
        detail=f'answer is suspiciously short: {text!r}',
      )
    ]
  return []


def check_grounding(
  answer: str,
  evidence: str,
  min_coverage: float = 0.6,
) -> List[ValidationIssue]:
  '''Check that numbers in the answer are traceable to evidence.

  This is the cheapest high-value hallucination check for operational
  answers: quantities, counts, and identifiers must come from tool output
  or the task statement, not from the model's imagination.

  Args:
    answer: Candidate answer.
    evidence: All trusted/sanitized evidence text available.
    min_coverage: Minimum fraction of answer numbers present in evidence.

  Returns:
    One warning issue when coverage is below the threshold.
  '''
  answer_numbers = extract_number_tokens(answer)
  if len(answer_numbers) < _MIN_GROUNDED_NUMBERS:
    return []

  evidence_numbers = set(extract_number_tokens(evidence))
  ungrounded = [
    token for token in answer_numbers if token not in evidence_numbers
  ]
  coverage = 1.0 - (len(ungrounded) / len(answer_numbers))
  if coverage >= min_coverage:
    return []

  ungrounded_text = ', '.join(ungrounded[:6])
  return [
    ValidationIssue(
      check='grounding',
      severity='warning',
      detail=(
        f'only {coverage:.0%} of answer numbers found in evidence; '
        f'ungrounded: {ungrounded_text}'
      ),
      suggestion=(
        'remove unsupported numbers or derive them explicitly from evidence'
      ),
    )
  ]


def check_tool_outcomes(
  answer: str,
  tool_calls: List[ToolExecutionAudit],
) -> List[ValidationIssue]:
  '''Check that failures and blocks are acknowledged in the answer.

  Args:
    answer: Candidate answer.
    tool_calls: Tool audit records from the run.

  Returns:
    List of issues for unacknowledged failures/blocks.
  '''
  issues: List[ValidationIssue] = []
  lowered = (answer or '').lower()
  acknowledged = any(word in lowered for word in _ACKNOWLEDGEMENT_WORDS)

  blocked = [call for call in tool_calls if call.blocked]
  failed = [
    call for call in tool_calls if not call.ok and not call.blocked
  ]

  if blocked and not acknowledged:
    issues.append(
      ValidationIssue(
        check='write_acknowledged',
        severity='error',
        detail=(
          'a write call was blocked by policy but the answer does not '
          'mention it: ' + ', '.join(call.tool for call in blocked[:3])
        ),
        suggestion='state that the action was blocked and what is needed',
      )
    )
  if failed and not acknowledged:
    issues.append(
      ValidationIssue(
        check='failures_acknowledged',
        severity='warning',
        detail=(
          'tool failures not mentioned: '
          + ', '.join(call.tool for call in failed[:3])
        ),
        suggestion='mention failed calls and adapt the recommendation',
      )
    )
  return issues


def check_evidence_used(
  route: ReasoningRoute,
  tool_calls: List[ToolExecutionAudit],
  evidence: str,
) -> List[ValidationIssue]:
  '''Check that data-dependent routes actually used evidence.

  Args:
    route: Reasoning route that produced the answer.
    tool_calls: Tool audit records.
    evidence: Evidence text available to the answer.

  Returns:
    List of issues.
  '''
  if route not in ('react', 'plan'):
    return []
  if tool_calls or (evidence or '').strip():
    return []
  return [
    ValidationIssue(
      check='evidence_used',
      severity='warning',
      detail='data-dependent route produced no tool evidence',
      suggestion='re-run with tools available or answer with a caveat',
    )
  ]


def check_plan_outcomes(
  answer: str,
  node_results: List[NodeResult],
) -> List[ValidationIssue]:
  '''Check that failed/skipped plan nodes are acknowledged in the answer.

  Tool audits only cover calls that were attempted. A plan node can also
  fail without any tool call (empty model output, skipped dependency), so
  the validator must look at execution results too - otherwise an answer
  can silently drop a requirement.

  Args:
    answer: Candidate answer.
    node_results: Plan execution node results.

  Returns:
    One error issue when unresolved nodes are not acknowledged.
  '''
  unresolved = [
    result for result in node_results
    if result.status in ('failed', 'skipped')
  ]
  if not unresolved:
    return []

  lowered = (answer or '').lower()
  acknowledged = any(word in lowered for word in _ACKNOWLEDGEMENT_WORDS)
  if acknowledged:
    return []

  names = ', '.join(result.node_id for result in unresolved[:4])
  return [
    ValidationIssue(
      check='plan_outcomes',
      severity='error',
      detail=(
        f'plan steps were not completed and the answer does not say so: '
        f'{names}'
      ),
      suggestion=(
        'state explicitly which steps failed or were skipped and what '
        'remains to be done'
      ),
    )
  ]


class SelfValidator:
  '''Deterministic checks plus an optional structured LLM critic.'''

  def __init__(
    self,
    llm: ReasoningLLM,
    config: PlanningConfig,
  ) -> None:
    '''Initialize the validator.

    Args:
      llm: Gateway-backed reasoning LLM wrapper.
      config: Chapter 2 configuration.
    '''
    self._llm = llm
    self._config = config

  def validate(
    self,
    task: str,
    answer: str,
    evidence: str,
    route: ReasoningRoute,
    tool_calls: Optional[List[ToolExecutionAudit]] = None,
    node_results: Optional[List[NodeResult]] = None,
  ) -> ValidationReport:
    '''Validate and, when needed, revise a candidate answer.

    Args:
      task: Original task.
      answer: Candidate answer to validate.
      evidence: Sanitized evidence text (observations, reasoning, outputs).
      route: Route that produced the answer.
      tool_calls: Tool audit records from the run.
      node_results: Plan execution results (plan route only).

    Returns:
      ValidationReport. Never raises: transport problems are reported via
      `critic_unavailable` and `errors`.
    '''
    audits = tool_calls or []
    nodes = node_results or []
    report = ValidationReport(final_answer=answer)

    before = TokenUsage(
      input_tokens=self._llm.budget.spent.input_tokens,
      output_tokens=self._llm.budget.spent.output_tokens,
      total_tokens=self._llm.budget.spent.total_tokens,
    )

    issues = self._deterministic_checks(
      task, answer, evidence, route, audits, nodes,
    )
    report.issues.extend(issues)
    report.checks_run.extend([
      'answer_present', 'grounding', 'tool_outcomes', 'evidence_used',
      'plan_outcomes',
    ])

    critic = None
    if self._should_critique(
      [
        issue for issue in issues
        if issue.severity in ('warning', 'error')
      ],
      route,
    ):
      critic = self._run_critic(task, evidence, answer, report)
      if critic is not None:
        report.critic_used = True
        report.issues.extend(critic.issues)

    actionable = [
      issue for issue in report.issues if issue.severity == 'error'
    ] or (critic is not None and critic.verdict == 'fail')

    if actionable and self._config.max_validation_rounds > 0:
      self._revise(task, evidence, answer, report, critic, nodes)

    if report.revised:
      effective = [
        issue for issue in report.issues
        if issue.check.startswith('post_revision:')
      ]
    else:
      effective = report.issues
    report.passed = not any(
      issue.severity == 'error' for issue in effective
    )
    if not report.final_answer:
      report.final_answer = answer

    after = self._llm.budget.spent
    report.usage = usage_delta(before, after)

    log_event(
      logger, 20, 'validation_done',
      route=route,
      passed=report.passed,
      issues=len(report.issues),
      critic_used=report.critic_used,
      revised=report.revised,
    )
    return report

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _deterministic_checks(
    self,
    task: str,
    answer: str,
    evidence: str,
    route: ReasoningRoute,
    audits: List[ToolExecutionAudit],
    nodes: Optional[List[NodeResult]] = None,
  ) -> List[ValidationIssue]:
    '''Run all free checks.

    Args:
      task: Original task.
      answer: Candidate answer.
      evidence: Evidence text.
      route: Reasoning route.
      audits: Tool audit records.
      nodes: Plan execution results.

    Returns:
      Aggregated issues.
    '''
    combined_evidence = f'{task}\n{evidence}\n' + ' '.join(
      call.result_preview for call in audits
    )
    issues: List[ValidationIssue] = []
    issues.extend(check_answer_present(answer))
    issues.extend(
      check_grounding(
        answer, combined_evidence, self._config.grounding_min_coverage,
      )
    )
    issues.extend(check_tool_outcomes(answer, audits))
    issues.extend(check_evidence_used(route, audits, evidence))
    if nodes:
      issues.extend(check_plan_outcomes(answer, nodes))
    return issues

  def _should_critique(
    self,
    issues: List[ValidationIssue],
    route: ReasoningRoute,
  ) -> bool:
    '''Decide whether the paid critic should run.

    Args:
      issues: Deterministic issues found.
      route: Reasoning route.

    Returns:
      True when the critic should run.
    '''
    if not self._config.critic_enabled:
      return False
    if self._config.critic_always:
      return True
    if issues:
      return True
    return route == 'plan'

  def _run_critic(
    self,
    task: str,
    evidence: str,
    answer: str,
    report: ValidationReport,
  ) -> Optional[Critique]:
    '''Run the structured LLM critic.

    Args:
      task: Original task.
      evidence: Evidence text.
      answer: Candidate answer.
      report: Report under construction (receives errors on failure).

    Returns:
      Critique, or None when the critic call/parse failed.
    '''
    prompt = CRITIC_PROMPT.render(
      task=task,
      evidence=evidence[:6000] or '(none)',
      answer=answer,
      contract=output_contract(
        Critique, 'Return your validation verdict as JSON.',
      ),
    )
    critic = self._llm.complete_json(
      prompt,
      Critique,
      system=CRITIC_SYSTEM,
      temperature=self._config.critic_temperature,
      label='critic',
    )
    if critic is None:
      report.critic_unavailable = True
      report.errors.append('critic unavailable; deterministic checks only')
      log_event(logger, 30, 'critic_unavailable',
                error=self._llm.last_error)
    return critic

  def _revise(
    self,
    task: str,
    evidence: str,
    answer: str,
    report: ValidationReport,
    critic: Optional[Critique],
    nodes: Optional[List[NodeResult]] = None,
  ) -> None:
    '''Run one bounded revision round and re-check deterministically.

    Args:
      task: Original task.
      evidence: Evidence text.
      answer: Original answer.
      report: Report under construction.
      critic: Optional critic output (may carry a suggested revision).
      nodes: Plan execution results (re-checked against the revision).
    '''
    issues_text = '\n'.join(
      f'- [{issue.severity}] {issue.check}: {issue.detail}'
      for issue in report.issues
    ) or '- no deterministic issues'

    if critic is not None and critic.revised_answer:
      revised = critic.revised_answer
    else:
      prompt = REVISION_PROMPT.render(
        task=task,
        evidence=evidence[:6000] or '(none)',
        answer=answer,
        issues=issues_text,
      )
      result = self._llm.complete(
        prompt,
        system=REVISION_SYSTEM,
        temperature=self._config.critic_temperature,
        label='revision',
      )
      if result is None:
        report.errors.append('revision call failed; keeping original answer')
        report.final_answer = answer
        return
      revised = result.content.strip()

    if not revised:
      report.final_answer = answer
      report.errors.append('revision was empty; keeping original answer')
      return

    report.rounds += 1

    recheck = self._deterministic_checks(
      task, revised, evidence, 'direct', [], nodes,
    )
    report.issues.extend(
      issue.model_copy(update={'check': f'post_revision:{issue.check}'})
      for issue in recheck
    )

    post_errors = [issue for issue in recheck if issue.severity == 'error']
    if post_errors:
      # A revision that still fails the checks is worse than the original:
      # keep the original answer and make the rejection visible.
      report.revision_rejected = True
      report.final_answer = answer
      report.errors.append(
        'revision rejected; it did not resolve: '
        + ', '.join(issue.check for issue in post_errors[:3])
      )
      log_event(
        logger, 30, 'revision_rejected',
        failed_checks=len(post_errors),
      )
      return

    report.revised = True
    report.final_answer = revised
    log_event(
      logger, 20, 'revision_applied',
      issues_before=len(issues_text.splitlines()),
    )
