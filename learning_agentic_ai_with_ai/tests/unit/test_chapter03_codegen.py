#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: agent-written Python plans and the sandbox.'''


from __future__ import annotations

import time

import pytest

from chapter03_cognition.codegen import (
  Deadline,
  RestrictedExecRunner,
  SandboxBlocked,
  SandboxTimeout,
  SubprocessRunner,
  ToolCaller,
  build_code_runner,
  validate_code,
)
from chapter03_cognition.config import CognitionConfig
from chapter03_cognition.jsonio import extract_code_block
from chapter03_cognition.schemas import ToolResult

SAFE_CODE = (
  'def solve(context):\n'
  '    per_day = 30\n'
  '    quantity = per_day * 4 + 10 - 20\n'
  '    return {"answer": f"{quantity} units", "used_tools": []}\n'
)


class TestValidateCode:
  '''Static validation rejects dangerous constructs.'''

  def test_safe_code_passes(self) -> None:
    assert validate_code(SAFE_CODE) == []

  def test_import_rejected(self) -> None:
    code = 'import os\n' + SAFE_CODE
    assert any('import' in error for error in validate_code(code))

  def test_from_import_rejected(self) -> None:
    code = 'from os import system\n' + SAFE_CODE
    assert any('import' in error for error in validate_code(code))

  def test_dunder_attribute_rejected(self) -> None:
    code = (
      'def solve(context):\n'
      '    return {"answer": context.__class__.__name__}\n'
    )
    assert any('dunder' in error for error in validate_code(code))

  def test_forbidden_builtin_rejected(self) -> None:
    code = (
      'def solve(context):\n'
      '    return {"answer": eval("1+1")}\n'
    )
    assert any('forbidden' in error for error in validate_code(code))

  def test_class_rejected(self) -> None:
    code = 'class X:\n    pass\n' + SAFE_CODE
    assert any('class' in error for error in validate_code(code))

  def test_missing_entrypoint_rejected(self) -> None:
    assert any(
      'missing required function' in error
      for error in validate_code('x = 1\n')
    )

  def test_wrong_argument_count_rejected(self) -> None:
    code = 'def solve():\n    return {}\n'
    assert any(
      'exactly one argument' in error for error in validate_code(code)
    )

  def test_syntax_error_rejected(self) -> None:
    assert any(
      'syntax error' in error for error in validate_code('def solve(:')
    )

  def test_oversized_code_rejected(self) -> None:
    assert validate_code(SAFE_CODE, max_chars=10)


class TestExtractCode:
  '''Code extraction from model output.'''

  def test_fenced_code(self) -> None:
    text = f'Here you go:\n```python\n{SAFE_CODE}\n```'
    assert 'def solve' in extract_code_block(text)

  def test_prose_before_function(self) -> None:
    text = f'Sure, the function is:\n{SAFE_CODE}'
    assert extract_code_block(text).startswith('def solve')


class TestToolCaller:
  '''Capability object handed to generated code.'''

  @staticmethod
  def make_caller(
    invoker=None,
    allowed=None,
    writes=None,
    budget: int = 3,
    allow_writes: bool = False,
    seconds: float = 5.0,
  ) -> ToolCaller:
    '''Build a ToolCaller for tests.

    Args:
      invoker: Tool invoker callable.
      allowed: Allowed tool names.
      writes: Write tool names.
      budget: Call budget.
      allow_writes: Whether writes are allowed.
      seconds: Deadline seconds.

    Returns:
      ToolCaller.
    '''
    def default_invoker(_tool, _args):
      return ToolResult(tool='srv.read', ok=True, text='{"value": 7}')

    return ToolCaller(
      invoker=invoker or default_invoker,
      allowed_tools=set(allowed or {'srv.read'}),
      write_tools=set(writes or {'srv.write'}),
      deadline=Deadline(seconds),
      call_budget=budget,
      allow_writes=allow_writes,
    )

  def test_call_returns_decoded_data(self) -> None:
    caller = self.make_caller()
    result = caller.call('srv.read', {})
    assert result == {'ok': True, 'data': {'value': 7}, 'error': None}
    assert caller.calls_used == 1

  def test_unknown_tool_blocked(self) -> None:
    with pytest.raises(SandboxBlocked):
      self.make_caller().call('srv.other', {})

  def test_write_blocked(self) -> None:
    with pytest.raises(SandboxBlocked):
      self.make_caller().call('srv.write', {})

  def test_write_allowed_when_enabled(self) -> None:
    caller = self.make_caller(
      allowed={'srv.read', 'srv.write'}, allow_writes=True,
    )
    result = caller.call('srv.write', {})
    assert result['ok'] is True

  def test_budget_enforced(self) -> None:
    caller = self.make_caller(budget=1)
    caller.call('srv.read', {})
    with pytest.raises(SandboxBlocked):
      caller.call('srv.read', {})

  def test_deadline_enforced(self) -> None:
    caller = self.make_caller(seconds=0.01)
    time.sleep(0.2)
    with pytest.raises(SandboxTimeout):
      caller.call('srv.read', {})


class TestRestrictedRunner:
  '''In-process sandbox behavior.'''

  def test_safe_code_executes(self) -> None:
    result = RestrictedExecRunner().run(
      SAFE_CODE, 'solve', {}, timeout_seconds=2.0, call_budget=3,
    )
    assert result.ok is True
    assert result.output['answer'] == '110 units'

  def test_unsafe_code_refused_before_execution(self) -> None:
    result = RestrictedExecRunner().run(
      'import os\n' + SAFE_CODE, 'solve', {},
      timeout_seconds=2.0, call_budget=3,
    )
    assert result.ok is False
    assert result.error_class == 'code_error'

  def test_tool_access_through_caller(self) -> None:
    code = (
      'def solve(context):\n'
      '    data = context["tools"].call("srv.read", {"x": 1})\n'
      '    return {"answer": str(data["data"]["value"])}\n'
    )
    caller = TestToolCaller.make_caller()
    result = RestrictedExecRunner().run(
      code, 'solve', {'tools': caller},
      timeout_seconds=2.0, call_budget=3,
    )
    assert result.ok is True
    assert result.output['answer'] == '7'
    assert result.calls_used == 1

  def test_blocked_tool_surfaces_as_blocked(self) -> None:
    code = (
      'def solve(context):\n'
      '    return {"answer": context["tools"].call("srv.nope", {})}\n'
    )
    caller = TestToolCaller.make_caller()
    result = RestrictedExecRunner().run(
      code, 'solve', {'tools': caller},
      timeout_seconds=2.0, call_budget=3,
    )
    assert result.ok is False
    assert result.error_class == 'blocked'

  def test_runtime_error_is_code_error(self) -> None:
    code = (
      'def solve(context):\n'
      '    return {"answer": 1 / 0}\n'
    )
    result = RestrictedExecRunner().run(
      code, 'solve', {}, timeout_seconds=2.0, call_budget=3,
    )
    assert result.ok is False
    assert 'ZeroDivisionError' in (result.error or '')

  def test_non_dict_output_rejected(self) -> None:
    code = 'def solve(context):\n    return 42\n'
    result = RestrictedExecRunner().run(
      code, 'solve', {}, timeout_seconds=2.0, call_budget=3,
    )
    assert result.ok is False
    assert 'must return a dict' in (result.error or '')

  def test_print_is_captured(self) -> None:
    code = (
      'def solve(context):\n'
      '    print("debug line")\n'
      '    return {"answer": "ok"}\n'
    )
    result = RestrictedExecRunner().run(
      code, 'solve', {}, timeout_seconds=2.0, call_budget=3,
    )
    assert result.stdout == 'debug line'


class TestSubprocessRunner:
  '''OS-isolated pure-computation runner.'''

  def test_pure_compute(self) -> None:
    result = SubprocessRunner().run(
      SAFE_CODE, 'solve', {}, timeout_seconds=5.0, call_budget=3,
    )
    assert result.ok is True
    assert result.output['answer'] == '110 units'
    assert result.runner == 'subprocess'

  def test_tools_unavailable(self) -> None:
    code = (
      'def solve(context):\n'
      '    return {"answer": context["tools"].call("srv.read", {})}\n'
    )
    result = SubprocessRunner().run(
      code, 'solve', {}, timeout_seconds=5.0, call_budget=3,
    )
    assert result.ok is False
    assert 'tools are unavailable' in (result.error or '')

  def test_timeout_kills_infinite_loop(self) -> None:
    code = (
      'def solve(context):\n'
      '    while True:\n'
      '        pass\n'
    )
    result = SubprocessRunner().run(
      code, 'solve', {}, timeout_seconds=0.5, call_budget=3,
    )
    assert result.ok is False
    assert result.timed_out is True

  def test_validation_short_circuits(self) -> None:
    result = SubprocessRunner().run(
      'import os\n' + SAFE_CODE, 'solve', {},
      timeout_seconds=5.0, call_budget=3,
    )
    assert result.ok is False
    assert result.timed_out is False


class TestRunnerFactory:
  '''Runner factory behavior.'''

  def test_builds_both_runners(self) -> None:
    config = CognitionConfig()
    assert isinstance(
      build_code_runner('restricted', config), RestrictedExecRunner,
    )
    assert isinstance(
      build_code_runner('subprocess', config), SubprocessRunner,
    )

  def test_unknown_runner_raises(self) -> None:
    with pytest.raises(ValueError):
      build_code_runner('quantum', CognitionConfig())
