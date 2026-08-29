#!/usr/bin/env python
# -- coding: utf-8 --

'''Evaluation runner for Chapter 1.

Measures, per case:
- TASK SUCCESS : required tools called / forbidden tools avoided / answer shape
- SAFETY       : no runtime errors; policy blocks visible in the audit trail
- RELIABILITY  : iteration + token budgets respected

Modes:
- mock (default): scripted planner; deterministic; no API keys needed.
- live          : same cases through your real LLM gateway.

Run:
  .venv/bin/python -m chapter01_mcp.evals.runner --mock
  .venv/bin/python -m chapter01_mcp.evals.runner --live
'''


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, List

from agentic_common import paths, setup_logging
from agentic_common.eval.harness import (
  EvalCase,
  RunOutcome,
  run_eval_suite,
)
from agentic_common.gateway_client import GatewayClient
from agentic_common.logging import get_logger
from agentic_common.persistence import AgentStore
from agentic_common.settings import Settings, default_settings
from agentic_common.tracing import TokenUsage, Tracer

logger = get_logger(__name__)


def load_eval_cases(path: Any) -> List[EvalCase]:
  '''Load and validate eval cases from YAML.

  Args:
    path: Path to cases.yaml.

  Returns:
    List of validated EvalCase.
  '''
  import yaml

  raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
  return [EvalCase.model_validate(case) for case in raw['cases']]


def make_task_runner(settings: Settings) -> Callable[[str], RunOutcome]:
  '''Build the task runner for the harness.

  Args:
    settings: Runtime settings (mock_llm flag decides the LLM transport).

  Returns:
    Callable that maps one task string to a RunOutcome.
  '''

  def run(task: str) -> RunOutcome:
    '''Execute one eval task through the full agent stack.

    Args:
      task: Task text.

    Returns:
      RunOutcome.
    '''
    from chapter01_mcp.agent.ops_agent import OpsAgent
    from chapter01_mcp.schemas import AgentTaskInput

    llm: Any
    if settings.mock_llm:
      from agentic_common.gateway_client import MockGateway
      from chapter01_mcp.demo import make_mock_planner

      llm = MockGateway(make_mock_planner(settings))
    else:
      llm = GatewayClient.shared()

    agent = OpsAgent(
      llm=llm,
      settings=settings,
      store=AgentStore(paths.AGENT_STATE_DB),
      tracer=Tracer(),
    )
    output = agent.run_task(
      AgentTaskInput(
        task=task,
        session_id='eval-run',
        max_iterations=settings.agent_max_iterations,
      ),
    )

    return RunOutcome(
      answer=output.answer,
      tool_calls=[
        {
          'server': call.server,
          'tool': f'{call.server}.{call.tool}',
          'args': call.args,
          'ok': call.ok,
          'approved': call.approved,
        }
        for call in output.tool_calls
      ],
      iterations=output.iterations,
      usage=TokenUsage(
        input_tokens=int(output.usage.get('input_tokens', 0)),
        output_tokens=int(output.usage.get('output_tokens', 0)),
        total_tokens=int(output.usage.get('total_tokens', 0)),
      ),
      errors=list(output.errors),
      extra={'status': output.status},
    )

  return run


def main() -> None:
  '''Entry point.'''
  parser = argparse.ArgumentParser(description='Chapter 1 evaluation harness')
  parser.add_argument('--mock', action='store_true', help='Scripted mock LLM (default)')
  parser.add_argument('--live', action='store_true', help='Real LLM gateway')
  args = parser.parse_args()

  setup_logging('WARNING')
  paths.ensure_data_dirs()
  from chapter01_mcp.servers.ops_db import seed_if_empty

  seed_if_empty()

  settings = default_settings().model_copy(update={'mock_llm': not args.live})
  task_runner = make_task_runner(settings)
  cases = load_eval_cases(paths.CHAPTER1_DIR / 'evals' / 'cases.yaml')

  report = run_eval_suite(cases, task_runner)

  report_path = paths.EVALS_DIR / 'chapter01_report.json'
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(
    json.dumps(report.model_dump(mode='json'), indent=2, default=str),
    encoding='utf-8',
  )

  print('\n=== Chapter 1 evaluation report ===')
  for line in report.summary_lines():
    print(line)
  print(f'report: {report_path}')
  print()

  for result in report.results:
    print(
      f'  [{"PASS" if result.passed else "FAIL"}] {result.case_id} '
      f'(tokens={result.total_tokens}, iters={result.iterations})'
    )
    if not result.passed:
      for check in result.checks:
        if not check.passed:
          print(f'        FAILED: {check.check} ({check.detail})')


if __name__ == '__main__':
  main()
