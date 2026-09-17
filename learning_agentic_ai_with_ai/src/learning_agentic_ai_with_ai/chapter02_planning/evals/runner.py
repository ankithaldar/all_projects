#!/usr/bin/env python
# -- coding: utf-8 --

'''Evaluation runner for Chapter 2.

Measures, per case:
- TASK SUCCESS : answer shape + expected tools called (shared harness)
- ROUTING      : the policy chose the expected reasoning route
- VALIDATION   : the self-validator verdict matches expectations
- PLANNING     : the plan executed the expected number of nodes
- SAFETY       : forbidden writes were blocked by the gate/approval bounds
- RELIABILITY  : LLM-call and token budgets respected

Modes:
- mock (default): scripted planner; deterministic; no API keys needed.
- live          : same cases through your real LLM gateway.

Tool backends:
- inproc (default): Chapter 1 server cores in-process.
- mcp             : live MCP stdio servers.

Run:
  python -m chapter02_planning.evals.runner --mock
  python -m chapter02_planning.evals.runner --live --tools mcp
'''


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, List

import yaml

from agentic_common import paths, setup_logging
from agentic_common.eval.harness import EvalReport, RunOutcome
from agentic_common.gateway_client import GatewayClient, MockGateway
from agentic_common.logging import get_logger, log_event
from agentic_common.persistence import AgentStore
from agentic_common.settings import Settings, default_settings
from agentic_common.tracing import TokenUsage, Tracer
from chapter02_planning.agent import PlanningAgent
from chapter02_planning.config import PlanningConfig, load_planning_config
from chapter02_planning.evals.cases import (
  PlanningEvalCase,
  run_checks,
)
from chapter02_planning.schemas import PlanningTaskInput
from chapter02_planning.tools.factory import BACKENDS, build_tool_provider

logger = get_logger(__name__)


def load_cases(path: Any) -> List[PlanningEvalCase]:
  '''Load and validate Chapter 2 eval cases from YAML.

  Args:
    path: Path to cases.yaml.

  Returns:
    List of validated PlanningEvalCase.
  '''
  raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
  return [
    PlanningEvalCase.model_validate(case) for case in raw['cases']
  ]


def make_task_runner(
  settings: Settings,
  config: PlanningConfig,
  backend: str,
) -> Callable[[PlanningEvalCase], RunOutcome]:
  '''Build the per-case task runner.

  Args:
    settings: Runtime settings (mock_llm decides the LLM transport).
    config: Chapter 2 configuration.
    backend: Tool backend ('inproc' or 'mcp').

  Returns:
    Callable mapping one eval case to a RunOutcome.
  '''

  def run(case: PlanningEvalCase) -> RunOutcome:
    '''Execute one eval case through the full Chapter 2 stack.

    Args:
      case: The eval case.

    Returns:
      RunOutcome with chapter metadata in `extra`.
    '''
    llm: Any
    if settings.mock_llm:
      # Lazy import: live mode must not import the demo mock planner.
      # pylint: disable-next=import-outside-toplevel
      from chapter02_planning.demo import make_mock_planner

      llm = MockGateway(make_mock_planner(config))
    else:
      llm = GatewayClient.shared()

    provider = build_tool_provider(backend, task=case.task)
    store = AgentStore(paths.AGENT_STATE_DB)
    agent = PlanningAgent(
      llm=llm,
      tools=provider,
      config=config,
      settings=settings,
      store=store,
      tracer=Tracer(),
      write_approver=_make_approver(settings),
    )

    try:
      output = agent.run_task(
        PlanningTaskInput(
          task=case.task,
          session_id=f'eval-ch2-{case.id}',
          allow_writes=case.allow_writes,
          cot_samples=case.cot_samples,
          max_steps=config.max_steps,
        )
      )
    finally:
      agent.close()
      store.close()

    blocked = [audit.tool for audit in output.tool_calls if audit.blocked]
    completed_nodes = 0
    if output.execution is not None:
      completed_nodes = sum(
        1 for result in output.execution.results
        if result.status == 'completed'
      )

    log_event(
      logger, 20, 'eval_case_run',
      case_id=case.id, route=output.route, status=output.status,
      blocked=len(blocked),
    )

    return RunOutcome(
      answer=output.answer,
      tool_calls=[
        {
          'server': audit.tool.split('.', 1)[0],
          'tool': audit.tool,
          'args': audit.args,
          'ok': audit.ok,
          'approved': audit.approved,
          'blocked': audit.blocked,
        }
        for audit in output.tool_calls
      ],
      iterations=int(output.usage.get('calls', 0) or 0),
      usage=TokenUsage(
        input_tokens=int(output.usage.get('input_tokens', 0) or 0),
        output_tokens=int(output.usage.get('output_tokens', 0) or 0),
        total_tokens=int(output.usage.get('total_tokens', 0) or 0),
      ),
      errors=list(output.errors),
      extra={
        'route': output.route,
        'status': output.status,
        'validation_passed': (
          output.validation.passed
          if output.validation is not None else None
        ),
        'plan_nodes_completed': completed_nodes,
        'blocked_tools': blocked,
        'policy_rationale': (
          output.policy.rationale if output.policy is not None else ''
        ),
        'plan_waves': (
          output.execution.waves if output.execution is not None else []
        ),
      },
    )

  return run


def _make_approver(
  settings: Settings,
) -> Callable[[str, dict], bool]:
  '''Build the bounded write-approval callback for evals.

  Args:
    settings: Runtime settings with safety caps.

  Returns:
    Approver callable(tool, args) -> bool.
  '''

  def approver(tool: str, args: dict) -> bool:
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


def build_report(results: List[Any]) -> EvalReport:
  '''Aggregate case results into an EvalReport.

  Args:
    results: CaseResult list from `run_checks`.

  Returns:
    Aggregate report.
  '''
  total = len(results)
  passed = sum(1 for result in results if result.passed)
  safety_failures = sum(
    1
    for result in results
    for check in result.checks
    if (
      check.check.startswith(('safety:', 'blocked:', 'validation:'))
      and not check.passed
    )
  )
  reliability_failures = sum(
    1
    for result in results
    for check in result.checks
    if check.check.startswith('reliability:') and not check.passed
  )
  return EvalReport(
    total_cases=total,
    passed_cases=passed,
    task_success_rate=(passed / total) if total else 0.0,
    safety_failures=safety_failures,
    reliability_failures=reliability_failures,
    avg_iterations=(
      sum(result.iterations for result in results) / total
      if total else 0.0
    ),
    avg_tokens=(
      sum(result.total_tokens for result in results) / total
      if total else 0.0
    ),
    results=results,
  )


def main() -> None:
  '''Entry point.'''
  parser = argparse.ArgumentParser(
    description='Chapter 2 evaluation harness',
  )
  parser.add_argument(
    '--mock', action='store_true', help='Scripted mock LLM (default)',
  )
  parser.add_argument('--live', action='store_true', help='Real LLM gateway')
  parser.add_argument(
    '--tools', default='inproc', choices=list(BACKENDS),
    help='Tool backend: inproc or mcp',
  )
  args = parser.parse_args()

  setup_logging('WARNING')
  paths.ensure_data_dirs()

  # Lazy import: keeps the eval CLI import cheap and side-effect free.
  # pylint: disable-next=import-outside-toplevel
  from chapter01_mcp.servers.ops_db import seed_if_empty

  seed_if_empty()

  settings = default_settings().model_copy(
    update={'mock_llm': not args.live},
  )
  config = load_planning_config(settings)
  cases = load_cases(
    paths.SRC_DIR / 'chapter02_planning' / 'evals' / 'cases.yaml',
  )

  task_runner = make_task_runner(settings, config, args.tools)
  results = run_checks(cases, task_runner)
  report = build_report(results)

  report_path = paths.EVALS_DIR / 'chapter02_report.json'
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(
    json.dumps(report.model_dump(mode='json'), indent=2, default=str),
    encoding='utf-8',
  )

  print('\n=== Chapter 2 evaluation report ===')
  for line in report.summary_lines():
    print(line)
  mode_label = 'live' if args.live else 'mock'
  print(f'mode: {mode_label}  tools: {args.tools}')
  print(f'report: {report_path}\n')

  for result in report.results:
    status = 'PASS' if result.passed else 'FAIL'
    print(
      f'  [{status}] {result.case_id} '
      f'(calls={result.iterations}, tokens={result.total_tokens})'
    )
    if not result.passed:
      for check in result.checks:
        if not check.passed:
          print(f'        FAILED: {check.check} ({check.detail})')


if __name__ == '__main__':
  main()
