#!/usr/bin/env python
# -- coding: utf-8 --

'''Evaluation runner for Chapter 3.

Measures, per case:
- TASK SUCCESS : answer shape and expected tools (shared harness)
- ARCHITECTURE : strategy, plan source, percept domain/capabilities
- ADAPTATION   : retry/tool-switch records on injected failures
- SANDBOX      : code plans ran in the expected runner
- SAFETY       : forbidden writes were blocked by approval bounds
- MEMORY       : repeated runs recall earlier episodes
- RELIABILITY  : LLM-call and token budgets respected

Modes:
- mock (default): scripted planner; deterministic; no API keys needed.
- live          : same cases through your real LLM gateway.

Tool backends:
- inproc (default): Chapter 1 server cores in-process.
- mcp             : live MCP stdio servers.

Run:
  python -m chapter03_cognition.evals.runner --mock
  python -m chapter03_cognition.evals.runner --live --tools mcp
'''


from __future__ import annotations

import argparse
import json
import tempfile
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
from chapter03_cognition.agent import CognitiveAgent
from chapter03_cognition.config import CognitionConfig, load_cognition_config
from chapter03_cognition.evals.cases import (
  CognitionEvalCase,
  run_checks,
)
from chapter03_cognition.memory import MemoryStore
from chapter03_cognition.schemas import CognitiveTaskInput
from chapter03_cognition.tools import FaultInjector, ToolBox

logger = get_logger(__name__)

TREND_TOOL = 'retail-ops.retail_sales_trend'


def load_cases(path: Any) -> List[CognitionEvalCase]:
  '''Load and validate Chapter 3 eval cases from YAML.

  Args:
    path: Path to cases.yaml.

  Returns:
    List of validated CognitionEvalCase.
  '''
  raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
  return [
    CognitionEvalCase.model_validate(case) for case in raw['cases']
  ]


def _make_approver(
  settings: Settings,
) -> Callable[[str, dict], bool]:
  '''Build the bounded write-approval callback.

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


def make_task_runner(
  settings: Settings,
  config: CognitionConfig,
  backend: str,
) -> Callable[[CognitionEvalCase], RunOutcome]:
  '''Build the per-case task runner.

  Args:
    settings: Runtime settings (mock_llm decides the LLM transport).
    config: Chapter 3 configuration.
    backend: Tool backend ('inproc' or 'mcp').

  Returns:
    Callable mapping one eval case to a RunOutcome.
  '''

  def run(case: CognitionEvalCase) -> RunOutcome:
    '''Execute one eval case through the cognitive pipeline.

    Args:
      case: The eval case.

    Returns:
      RunOutcome with architecture metadata in `extra`.
    '''
    llm: Any
    if settings.mock_llm:
      from chapter03_cognition.demo import (  # pylint: disable=import-outside-toplevel
        make_mock_planner,
      )

      llm = MockGateway(make_mock_planner(config))
    else:
      llm = GatewayClient.shared()

    memory_file = Path(tempfile.gettempdir()) / f'ch3_eval_{case.id}.db'
    for suffix in ('', '-wal', '-shm'):
      candidate = memory_file.with_name(memory_file.name + suffix)
      if candidate.exists():
        candidate.unlink()
    case_config = config.model_copy(
      update={'memory_db_path': str(memory_file)},
    )

    store = AgentStore(paths.AGENT_STATE_DB)
    memory = MemoryStore(memory_file)
    tracer = Tracer()
    agent = CognitiveAgent(
      llm=llm,
      toolbox=None,
      config=case_config,
      settings=settings,
      store=store,
      memory=memory,
      tracer=tracer,
    )

    if not case.no_tools:
      toolbox: Any = ToolBox(
        backend=backend,
        task=case.task,
        allow_writes=case.allow_writes,
        require_approval=case_config.require_write_approval,
        approver=_make_approver(settings),
        max_result_chars=case_config.max_result_chars,
        sanitize=case_config.sanitize_observations,
        audit_sink=agent.audit_sink,
        tracer=tracer,
      )
      if case.inject_faults:
        toolbox = FaultInjector(
          toolbox, failures={TREND_TOOL: 3},
          error='injected transient failure',
        )
      agent.attach_toolbox(toolbox)

    output = None
    try:
      for index in range(max(1, case.repeat)):
        output = agent.run_task(
          CognitiveTaskInput(
            task=case.task,
            session_id=f'eval-ch3-{case.id}-{index}',
            strategy_hint=(
              case.expect_strategy  # type: ignore[arg-type]
              if case.expect_strategy in (
                'conservative', 'exploratory', 'fallback',
              ) else None
            ),
            allow_writes=case.allow_writes,
            code_runner=case.code_runner,  # type: ignore[arg-type]
            plan_mode=case.plan_mode,  # type: ignore[arg-type]
            max_steps=case_config.max_steps,
          )
        )
    finally:
      agent.close()
      store.close()
      memory.close()

    assert output is not None
    blocked = [audit.tool for audit in agent._audits if audit.blocked]  # pylint: disable=protected-access
    completed_steps = 0
    adaptations: List[str] = []
    if output.action is not None:
      completed_steps = sum(
        1 for step in output.action.steps if step.status == 'completed'
      )
      adaptations = [record.kind for record in output.action.adaptations]

    log_event(
      logger, 20, 'eval_case_run',
      case_id=case.id,
      strategy=(
        output.decision.strategy if output.decision else 'unknown'
      ),
      status=output.status,
      adaptations=len(adaptations),
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
        for audit in agent._audits  # pylint: disable=protected-access
      ],
      iterations=int(output.usage.get('calls', 0) or 0),
      usage=TokenUsage(
        input_tokens=int(output.usage.get('input_tokens', 0) or 0),
        output_tokens=int(output.usage.get('output_tokens', 0) or 0),
        total_tokens=int(output.usage.get('total_tokens', 0) or 0),
      ),
      errors=list(output.errors),
      extra={
        'status': output.status,
        'strategy': (
          output.decision.strategy if output.decision else 'unknown'
        ),
        'plan_source': (
          output.decision.plan.source if output.decision else 'unknown'
        ),
        'adaptations': sorted(set(adaptations)),
        'blocked_tools': blocked,
        'code_runner': (
          output.action.code_runner if output.action else None
        ),
        'percept_domain': (
          output.percept.signals.domain if output.percept else 'unknown'
        ),
        'capabilities': (
          output.percept.required_capabilities if output.percept else []
        ),
        'memory_episodes': (
          len(output.memory.relevant_episodes)
          if output.memory else 0
        ),
        'steps_completed': completed_steps,
      },
    )

  return run


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
      check.check.startswith(('safety:', 'blocked:', 'status:'))
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
    description='Chapter 3 evaluation harness',
  )
  parser.add_argument(
    '--mock', action='store_true', help='Scripted mock LLM (default)',
  )
  parser.add_argument('--live', action='store_true', help='Real LLM gateway')
  parser.add_argument(
    '--tools', default='inproc', choices=['inproc', 'mcp'],
    help='Tool backend: inproc or mcp',
  )
  args = parser.parse_args()

  setup_logging('WARNING')
  paths.ensure_data_dirs()

  # Lazy import keeps the eval CLI import cheap.
  # pylint: disable-next=import-outside-toplevel
  from chapter01_mcp.servers.ops_db import seed_if_empty

  seed_if_empty()

  settings = default_settings().model_copy(
    update={'mock_llm': not args.live},
  )
  config = load_cognition_config(settings)
  cases = load_cases(
    paths.SRC_DIR / 'chapter03_cognition' / 'evals' / 'cases.yaml',
  )

  task_runner = make_task_runner(settings, config, args.tools)
  results = run_checks(cases, task_runner)
  report = build_report(results)

  report_path = paths.EVALS_DIR / 'chapter03_report.json'
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(
    json.dumps(report.model_dump(mode='json'), indent=2, default=str),
    encoding='utf-8',
  )

  print('\n=== Chapter 3 evaluation report ===')
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
