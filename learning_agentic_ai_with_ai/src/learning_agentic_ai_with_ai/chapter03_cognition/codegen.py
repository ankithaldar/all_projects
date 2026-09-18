#!/usr/bin/env python
# -- coding: utf-8 --

'''Agent-written Python plans: capability-based sandboxing.

The Decision layer may ask the model to write a `solve(context)` function
instead of a JSON DAG. Generated code is a powerful but dangerous tool, so
Chapter 3 treats it like any other untrusted capability:

  1. **Static validation** (`validate_code`): parse with `ast` and reject
     imports, classes, global/nonlocal, dunder attribute access, and a
     denylist of dangerous builtins (`exec`, `eval`, `open`, `getattr`,
     `__import__`, ...). The entrypoint must be a plain function taking one
     argument.
  2. **Capability-based tool access** (`ToolCaller`): generated code never
     receives a raw function. It receives an object whose `call()` enforces
     the allowed tool set, write policy, a call budget, and a deadline, and
     returns structured `{ok, data, error}` dicts.
  3. **Two runners behind one interface** (`CodeRunner`):
       - `RestrictedExecRunner` runs in-process with allowlisted builtins
         and a cooperative deadline (tool calls and post-run checks); fastest
         and the only runner that supports tool calls.
       - `SubprocessRunner` runs in an isolated child process (`-I -S`),
         with CPU/file-descriptor limits, a hard wall-clock timeout, and a
         scrubbed environment. Tools are *unavailable* there: use it for
         pure computation plans when OS isolation is required.

The interface lets you trade isolation for capability without changing the
rest of the pipeline.
'''


from __future__ import annotations

import ast
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

try:  # POSIX-only resource limits; absent on Windows.
  import resource
except ImportError:  # pragma: no cover - platform dependent
  resource = None  # type: ignore[assignment]

from agentic_common.logging import get_logger
from chapter03_cognition.config import CognitionConfig
from chapter03_cognition.schemas import (
  CodeRunnerName,
  CodeRunResult,
  ToolResult,
)

logger = get_logger(__name__)

ToolInvoker = Callable[[str, Dict[str, Any]], ToolResult]

FORBIDDEN_NAMES = frozenset({
  'exec', 'eval', 'compile', 'open', 'input', '__import__', 'globals',
  'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'breakpoint',
  'help', 'exit', 'quit', 'memoryview', 'object', 'type', 'super',
  'classmethod', 'staticmethod', 'property',
})

SAFE_BUILTINS: Dict[str, Any] = {
  'abs': abs,
  'all': all,
  'any': any,
  'bool': bool,
  'dict': dict,
  'divmod': divmod,
  'enumerate': enumerate,
  'filter': filter,
  'float': float,
  'int': int,
  'isinstance': isinstance,
  'len': len,
  'list': list,
  'map': map,
  'max': max,
  'min': min,
  'pow': pow,
  'range': range,
  'reversed': reversed,
  'round': round,
  'set': set,
  'sorted': sorted,
  'str': str,
  'sum': sum,
  'tuple': tuple,
  'zip': zip,
  'True': True,
  'False': False,
  'None': None,
}

_RESULT_SENTINEL = '__COGNITION_RESULT__'


class SandboxError(RuntimeError):
  '''Base class for sandbox refusals.'''


class SandboxTimeout(SandboxError):
  '''Raised when the code plan exceeds its wall-clock deadline.'''


class SandboxBlocked(SandboxError):
  '''Raised when code requests a capability it does not have.'''


class Deadline:
  '''Monotonic wall-clock deadline shared with the sandbox.'''

  def __init__(self, seconds: float) -> None:
    '''Initialize the deadline.

    Args:
      seconds: Seconds from now until expiry (minimum 0.1).
    '''
    self._end = time.monotonic() + max(0.1, float(seconds))

  @property
  def expired(self) -> bool:
    '''Whether the deadline has passed.

    Returns:
      True when expired.
    '''
    return time.monotonic() >= self._end

  @property
  def remaining(self) -> float:
    '''Seconds left.

    Returns:
      Non-negative remaining seconds.
    '''
    return max(0.0, self._end - time.monotonic())


class ToolCaller:
  '''Capability object handed to generated code as `context['tools']`.

  Generated code can only do what this object allows: call tools on the
  allowlist, within a call budget and deadline, and only writes the active
  strategy permits.
  '''

  def __init__(
    self,
    invoker: ToolInvoker,
    allowed_tools: Set[str],
    write_tools: Set[str],
    deadline: Deadline,
    call_budget: int,
    allow_writes: bool = False,
  ) -> None:
    '''Initialize the capability object.

    Args:
      invoker: Callable executing a gated tool call.
      allowed_tools: Qualified tool names the code may call.
      write_tools: Subset of tool names that mutate state.
      deadline: Shared deadline.
      call_budget: Maximum calls from this code plan.
      allow_writes: Whether write tools are permitted.
    '''
    self._invoker = invoker
    self._allowed = set(allowed_tools)
    self._write_tools = set(write_tools)
    self._deadline = deadline
    self._budget = max(0, int(call_budget))
    self._allow_writes = allow_writes
    self.calls_used = 0

  @property
  def deadline(self) -> Deadline:
    '''The shared wall-clock deadline.

    Returns:
      Deadline instance.
    '''
    return self._deadline

  def catalog(self) -> List[str]:
    '''List tools the code is allowed to call.

    Returns:
      Sorted tool names.
    '''
    return sorted(self._allowed)

  def call(
    self,
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    '''Call one tool with capability checks.

    Args:
      name: Qualified tool name.
      arguments: Tool arguments.

    Returns:
      Dict with `ok`, `data`, and `error` keys.

    Raises:
      SandboxTimeout: When the deadline has passed.
      SandboxBlocked: On unknown tools, disabled writes, or budget exhaustion.
    '''
    if self._deadline.expired:
      raise SandboxTimeout('code plan deadline exceeded before tool call')
    if self.calls_used >= self._budget:
      raise SandboxBlocked('tool call budget exhausted')
    if name not in self._allowed:
      raise SandboxBlocked(f'tool not available to code plan: {name}')
    if name in self._write_tools and not self._allow_writes:
      raise SandboxBlocked(f'writes disabled for code plan: {name}')

    self.calls_used += 1
    result = self._invoker(name, arguments or {})
    data = _decode_tool_text(result.text)
    return {
      'ok': bool(result.ok),
      'data': data,
      'error': result.error,
    }


def _decode_tool_text(text: str) -> Any:
  '''Best-effort decode of a tool result text as JSON.

  Args:
    text: Tool result text.

  Returns:
    Decoded value, or the raw string when not JSON.
  '''
  try:
    return json.loads(text)
  except (TypeError, ValueError):
    return text


def validate_code(
  code: str,
  entrypoint: str = 'solve',
  max_chars: int = 12000,
) -> List[str]:
  '''Statically validate generated Python before any execution.

  Args:
    code: Generated source.
    entrypoint: Required function name.
    max_chars: Maximum accepted source size.

  Returns:
    List of human-readable violations (empty when safe).
  '''
  errors: List[str] = []
  if not code or not code.strip():
    return ['code is empty']
  if len(code) > max_chars:
    return [f'code exceeds {max_chars} characters']

  try:
    tree = ast.parse(code, filename='<agent_plan>', mode='exec')
  except SyntaxError as exc:
    return [f'syntax error: {exc}']

  function_found = False
  for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
      errors.append('imports are not allowed')
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
      errors.append('global/nonlocal are not allowed')
    elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)):
      errors.append('classes and async functions are not allowed')
    elif isinstance(node, ast.Attribute):
      if node.attr.startswith('_'):
        errors.append(f'dunder attribute access is not allowed: {node.attr}')
    elif isinstance(node, ast.Name):
      if node.id in FORBIDDEN_NAMES:
        errors.append(f'forbidden name: {node.id}')
    elif isinstance(node, ast.FunctionDef):
      if node.name == entrypoint and not function_found:
        function_found = True
        if len(node.args.args) != 1:
          errors.append(f'{entrypoint}() must take exactly one argument')

  if not function_found:
    errors.append(f'missing required function: {entrypoint}()')

  return sorted(set(errors))


class CodeRunner(ABC):
  '''Abstract executor for agent-written Python plans.'''

  name: CodeRunnerName = 'restricted'

  @abstractmethod
  def run(
    self,
    code: str,
    entrypoint: str,
    context: Dict[str, Any],
    timeout_seconds: float,
    call_budget: int,
  ) -> CodeRunResult:
    '''Execute generated code and return its structured output.

    Args:
      code: Validated generated source.
      entrypoint: Function name to call.
      context: Dict passed to the function (`task`, `facts`, `tools`).
      timeout_seconds: Wall-clock budget.
      call_budget: Tool calls allowed (when tools are available).

    Returns:
      CodeRunResult; never raises.
    '''


class RestrictedExecRunner(CodeRunner):
  '''In-process runner with allowlisted builtins and a cooperative deadline.

  Timeouts are enforced at tool-call boundaries and after execution. A
  tight pure-Python CPU loop cannot be preempted in-thread; use the
  subprocess runner when hard preemption matters.
  '''

  name: CodeRunnerName = 'restricted'

  def run(
    self,
    code: str,
    entrypoint: str,
    context: Dict[str, Any],
    timeout_seconds: float,
    call_budget: int,
  ) -> CodeRunResult:
    '''Execute generated code in a restricted namespace.

    Args:
      code: Generated source.
      entrypoint: Function name to call.
      context: Dict passed to the function.
      timeout_seconds: Wall-clock budget.
      call_budget: Tool calls allowed.

    Returns:
      CodeRunResult.
    '''
    started = time.perf_counter()
    errors = validate_code(code, entrypoint)
    if errors:
      return CodeRunResult(
        ok=False, runner=self.name, error='; '.join(errors),
        error_class='code_error',
      )

    stdout: List[str] = []

    def _capture_print(*args: Any, **kwargs: Any) -> None:
      '''Capture generated-code prints without touching real stdout.

      Args:
        *args: Print arguments.
        **kwargs: Print keyword arguments (ignored).
      '''
      del kwargs
      if len(stdout) < 200:
        stdout.append(' '.join(str(arg) for arg in args)[:500])

    namespace: Dict[str, Any] = {
      '__builtins__': dict(SAFE_BUILTINS, print=_capture_print),
      '__name__': 'agent_plan',
      'json': json,
      'math': math,
      'statistics': statistics,
    }

    deadline = Deadline(timeout_seconds)
    tool_caller = context.get('tools')
    if isinstance(tool_caller, ToolCaller):
      deadline = tool_caller.deadline

    try:
      compiled = compile(code, '<agent_plan>', 'exec')
      exec(compiled, namespace)  # pylint: disable=exec-used
      function = namespace.get(entrypoint)
      if not callable(function):
        return CodeRunResult(
          ok=False, runner=self.name,
          error=f'{entrypoint} was not defined',
          error_class='code_error',
        )
      safe_context = {
        'task': context.get('task', ''),
        'facts': context.get('facts', {}),
        'tools': tool_caller,
      }
      output = function(safe_context)
    except SandboxTimeout as exc:
      return CodeRunResult(
        ok=False, runner=self.name, error=str(exc),
        error_class='timeout', timed_out=True,
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )
    except SandboxBlocked as exc:
      return CodeRunResult(
        ok=False, runner=self.name, error=str(exc),
        error_class='blocked',
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      return CodeRunResult(
        ok=False, runner=self.name, error=f'{type(exc).__name__}: {exc}',
        error_class='code_error', stdout='\n'.join(stdout),
        latency_ms=(time.perf_counter() - started) * 1000.0,
      )

    latency_ms = (time.perf_counter() - started) * 1000.0
    timed_out = deadline.expired
    if not isinstance(output, dict):
      return CodeRunResult(
        ok=False, runner=self.name,
        error=f'{entrypoint} must return a dict, got {type(output).__name__}',
        error_class='code_error', stdout='\n'.join(stdout),
        latency_ms=latency_ms,
      )

    return CodeRunResult(
      ok=not timed_out,
      runner=self.name,
      output=output,
      stdout='\n'.join(stdout),
      error='deadline exceeded after execution' if timed_out else None,
      error_class='timeout' if timed_out else 'none',
      timed_out=timed_out,
      latency_ms=latency_ms,
      calls_used=tool_caller.calls_used if isinstance(
        tool_caller, ToolCaller,
      ) else 0,
    )


def _limit_resources() -> None:
  '''Apply POSIX resource limits to the child process (best effort).'''
  try:
    cpu_seconds = 10
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
  except (ValueError, OSError):
    pass


_SUBPROCESS_HARNESS = '''
import json
import sys


class Tools:
  def call(self, name, arguments=None):
    raise RuntimeError(
      "tools are unavailable in the subprocess runner; "
      "use the restricted runner for tool-using code plans"
    )

  def catalog(self):
    return []


def main():
  payload = json.loads(sys.stdin.read() or "{}")
  namespace = {}
  exec(compile(payload["code"], "agent_plan.py", "exec"), namespace)
  function = namespace.get(payload["entrypoint"])
  if not callable(function):
    print("__COGNITION_RESULT__" + json.dumps(
      {"error": "entrypoint not defined"}))
    return
  context = {
    "task": payload.get("task", ""),
    "facts": payload.get("facts", {}),
    "tools": Tools(),
  }
  try:
    result = function(context)
  except Exception as exc:
    print("__COGNITION_RESULT__" + json.dumps(
      {"error": type(exc).__name__ + ": " + str(exc)}))
    return
  print("__COGNITION_RESULT__" + json.dumps(result, default=str))


main()
'''


class SubprocessRunner(CodeRunner):
  '''Isolated pure-computation runner (no tools, hard timeout).

  Runs generated code in a child process with `-I -S`, a scrubbed
  environment, CPU/file-descriptor limits, and a hard wall-clock timeout.
  Tool access is deliberately unavailable: use this runner when OS-level
  isolation matters more than tool capability.
  '''

  name: CodeRunnerName = 'subprocess'

  def run(
    self,
    code: str,
    entrypoint: str,
    context: Dict[str, Any],
    timeout_seconds: float,
    call_budget: int,
  ) -> CodeRunResult:
    '''Execute generated code in an isolated child process.

    Args:
      code: Generated source.
      entrypoint: Function name to call.
      context: Dict with `task` and `facts` (tools are ignored).
      timeout_seconds: Hard wall-clock budget.
      call_budget: Unused (tools unavailable).

    Returns:
      CodeRunResult.
    '''
    del call_budget
    started = time.perf_counter()
    errors = validate_code(code, entrypoint)
    if errors:
      return CodeRunResult(
        ok=False, runner=self.name, error='; '.join(errors),
        error_class='code_error',
      )

    payload = json.dumps({
      'code': code,
      'entrypoint': entrypoint,
      'task': context.get('task', ''),
      'facts': context.get('facts', {}),
    })
    environment = {
      'PATH': os.environ.get('PATH', ''),
      'PYTHONHASHSEED': '0',
      'LC_ALL': 'C',
    }

    with tempfile.TemporaryDirectory(prefix='cognition_code_') as workdir:
      harness = Path(workdir) / 'runner.py'
      harness.write_text(_SUBPROCESS_HARNESS, encoding='utf-8')
      kwargs: Dict[str, Any] = {}
      if resource is not None and hasattr(resource, 'RLIMIT_CPU'):
        kwargs['preexec_fn'] = _limit_resources
      try:
        completed = subprocess.run(
          [sys.executable, '-I', '-S', str(harness)],
          input=payload,
          capture_output=True,
          text=True,
          timeout=max(0.5, float(timeout_seconds)),
          cwd=workdir,
          env=environment,
          check=False,
          **kwargs,
        )
      except subprocess.TimeoutExpired:
        return CodeRunResult(
          ok=False, runner=self.name,
          error='subprocess code plan timed out',
          error_class='timeout', timed_out=True,
          latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    stdout = completed.stdout or ''
    stderr = (completed.stderr or '')[-1000:]
    marker_index = stdout.rfind(_RESULT_SENTINEL)
    if marker_index < 0:
      return CodeRunResult(
        ok=False, runner=self.name,
        error=(
          f'no result marker (exit={completed.returncode}): '
          f'{stderr or stdout[:300]}'
        ),
        error_class='code_error',
        stdout=stdout[-1000:], latency_ms=latency_ms,
      )

    raw = stdout[marker_index + len(_RESULT_SENTINEL):].strip()
    try:
      decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
      return CodeRunResult(
        ok=False, runner=self.name, error=f'bad result JSON: {exc}',
        error_class='code_error', latency_ms=latency_ms,
      )

    if isinstance(decoded, dict) and decoded.get('error'):
      return CodeRunResult(
        ok=False, runner=self.name, error=str(decoded['error']),
        error_class='code_error', stdout=stdout[-1000:],
        latency_ms=latency_ms,
      )
    if not isinstance(decoded, dict):
      return CodeRunResult(
        ok=False, runner=self.name, error='result was not a dict',
        error_class='code_error', latency_ms=latency_ms,
      )

    return CodeRunResult(
      ok=True, runner=self.name, output=decoded,
      stdout=stdout[-1000:], latency_ms=latency_ms,
    )


def build_code_runner(
  name: str,
  config: CognitionConfig,
) -> CodeRunner:
  '''Build a code runner by name.

  Args:
    name: `restricted` or `subprocess`.
    config: Cognition config (unused today; kept for symmetry).

  Returns:
    CodeRunner implementation.

  Raises:
    ValueError: When the runner name is unknown.
  '''
  del config
  if name == 'subprocess':
    return SubprocessRunner()
  if name == 'restricted':
    return RestrictedExecRunner()
  raise ValueError(f'unknown code runner {name!r}')
