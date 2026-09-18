#!/usr/bin/env python
# -- coding: utf-8 --

'''Gateway-backed LLM access for the cognitive pipeline (Chapter 3).

Every LLM call flows through the user's in-process `llm_gateway` (via
`agentic_common.gateway_client`) and this wrapper adds the behaviors the
cognitive layers need:

  - transient-failure retries with exponential backoff and jitter,
  - one same-prompt retry for empty completions (reasoning models),
  - a hard per-run token budget with a `BudgetTracker`,
  - tracing spans (`llm.call`) with label/provider/model/usage,
  - structured completion (`complete_json`) validated against a pydantic
    model, with exactly one validation-driven repair round.

Failures degrade to `None`; the last error is kept for observability, and
callers decide between retrying, switching strategy, or falling back.
'''


from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict

from agentic_common.logging import get_logger, log_event
from agentic_common.tracing import TokenUsage, Tracer
from chapter03_cognition.config import CognitionConfig
from chapter03_cognition.jsonio import JsonParseError, parse_into
from llm_gateway.schemas import ChatMessage, GatewayRequest

logger = get_logger(__name__)

ModelT = TypeVar('ModelT', bound=BaseModel)


def usage_delta(before: TokenUsage, after: TokenUsage) -> Dict[str, int]:
  '''Compute the usage difference between two snapshots.

  Args:
    before: Usage snapshot before a stage.
    after: Usage snapshot after a stage.

  Returns:
    Dict with input/output/total token deltas.
  '''
  return {
    'input_tokens': after.input_tokens - before.input_tokens,
    'output_tokens': after.output_tokens - before.output_tokens,
    'total_tokens': after.total_tokens - before.total_tokens,
  }


class LLMResult(BaseModel):
  '''Normalized result of one successful LLM call.'''

  model_config = ConfigDict(extra='ignore')

  content: str = ''
  provider: str = ''
  model: str = ''
  label: str = ''
  latency_ms: float = 0.0
  cached: bool = False
  usage: TokenUsage = TokenUsage()


class BudgetTracker:
  '''Per-run token budget accounting.'''

  def __init__(self, total_tokens: int) -> None:
    '''Initialize the tracker.

    Args:
      total_tokens: Maximum total tokens for the run.
    '''
    self._total = max(1, int(total_tokens))
    self.spent = TokenUsage()

  def spend(self, usage: TokenUsage) -> None:
    '''Add usage to the running total.

    Args:
      usage: Usage of one call.
    '''
    self.spent = self.spent.add(usage)

  @property
  def exceeded(self) -> bool:
    '''Whether the budget is exhausted.

    Returns:
      True when spent tokens exceed the budget.
    '''
    return self.spent.total_tokens > self._total

  @property
  def remaining(self) -> int:
    '''Remaining token allowance.

    Returns:
      Non-negative remaining tokens.
    '''
    return max(0, self._total - self.spent.total_tokens)

  def snapshot(self) -> Dict[str, int]:
    '''Serialize the budget state.

    Returns:
      Dict with spent, budget, and remaining tokens.
    '''
    return {
      'input_tokens': self.spent.input_tokens,
      'output_tokens': self.spent.output_tokens,
      'total_tokens': self.spent.total_tokens,
      'budget': self._total,
      'remaining': self.remaining,
    }


class ReasoningLLM:
  '''Retrying, budget-aware, traced wrapper around the LLM gateway.'''

  def __init__(
    self,
    llm: Any,
    config: CognitionConfig,
    tracer: Optional[Tracer] = None,
    trace_id: str = '',
    session_id: str = '',
  ) -> None:
    '''Initialize the wrapper.

    Args:
      llm: GatewayClient or MockGateway (same `complete()` interface).
      config: Chapter 3 configuration.
      tracer: Optional tracer for `llm.call` spans.
      trace_id: Trace id for span correlation.
      session_id: Session id forwarded to the gateway.
    '''
    self._llm = llm
    self._config = config
    self._tracer = tracer
    self._trace_id = trace_id
    self._session_id = session_id
    self.budget = BudgetTracker(config.token_budget)
    self.calls = 0
    self.failures = 0
    self.parse_repairs = 0
    self.last_error: Optional[str] = None

  def complete(
    self,
    prompt: str,
    system: str = '',
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    label: str = 'llm',
  ) -> Optional[LLMResult]:
    '''Run one completion with retries, tracing, and budget accounting.

    Args:
      prompt: User prompt text.
      system: Optional system prompt.
      temperature: Sampling temperature (defaults to config).
      max_tokens: Output cap (defaults to config).
      label: Short label identifying the caller.

    Returns:
      LLMResult on success, None when every attempt failed.
    '''
    messages: List[ChatMessage] = []
    if system:
      messages.append(ChatMessage(role='system', content=system))
    messages.append(ChatMessage(role='user', content=prompt))

    request = GatewayRequest(
      messages=messages,
      temperature=(
        self._config.reason_temperature
        if temperature is None else temperature
      ),
      max_tokens=(
        self._config.max_tokens if max_tokens is None else max_tokens
      ),
      session_id=self._session_id or None,
      metadata={
        'chapter': '03_cognition',
        'label': label,
        'trace_id': self._trace_id,
      },
    )

    attempts = max(1, self._config.llm_max_attempts)
    empty_retry_used = False
    for attempt in range(1, attempts + 1):
      started = time.perf_counter()
      try:
        response = self._llm.complete(request)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        self.last_error = str(exc)
        self.failures += 1
        log_event(
          logger, 30, 'llm_attempt_failed',
          label=label, attempt=attempt, attempts=attempts,
          error=str(exc),
        )
        if attempt < attempts:
          self._sleep_backoff(attempt)
        continue

      if (
        not (response.content or '').strip()
        and not response.tool_calls
        and not empty_retry_used
      ):
        empty_retry_used = True
        log_event(
          logger, 30, 'empty_completion_retry',
          label=label, attempt=attempt,
        )
        continue

      usage = TokenUsage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        total_tokens=response.usage.total_tokens,
      )
      self.budget.spend(usage)
      self.calls += 1
      latency_ms = (time.perf_counter() - started) * 1000.0

      result = LLMResult(
        content=response.content or '',
        provider=response.provider,
        model=response.model,
        label=label,
        latency_ms=latency_ms,
        cached=bool(getattr(response, 'cached', False)),
        usage=usage,
      )
      self._trace_call(result)
      log_event(
        logger, 20, 'llm_call',
        label=label,
        provider=result.provider,
        model=result.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_ms=round(latency_ms, 1),
        attempt=attempt,
      )
      return result

    return None

  def complete_json(
    self,
    prompt: str,
    model: Type[ModelT],
    system: str = '',
    temperature: Optional[float] = None,
    label: str = 'json',
    repair: bool = True,
  ) -> Optional[ModelT]:
    '''Run a completion whose output must match a pydantic model.

    Args:
      prompt: User prompt text.
      model: Pydantic model the JSON must validate against.
      system: Optional system prompt.
      temperature: Sampling temperature (defaults to config).
      label: Short label identifying the caller.
      repair: Whether to attempt one JSON repair round.

    Returns:
      Validated model instance, or None when parsing fails.
    '''
    result = self.complete(
      prompt, system=system, temperature=temperature, label=label,
    )
    if result is None:
      return None

    parsed, error = self._try_parse(result.content, model)
    if parsed is not None:
      return parsed

    if not repair:
      self.last_error = error
      log_event(logger, 30, 'json_parse_failed', label=label, error=error)
      return None

    self.parse_repairs += 1
    repair_prompt = (
      'Your previous output could not be parsed.\n'
      f'Parser error: {error}\n'
      f'Previous output preview: {result.content[:400]}\n\n'
      'Return exactly one valid JSON object matching the schema in the '
      'original instructions. Include every required top-level field and '
      'at least one item wherever a list is required. No prose, no '
      'markdown fences.'
    )
    repaired = self.complete(
      repair_prompt, system=system, temperature=0.0, label=f'{label}.repair',
    )
    if repaired is None:
      self.last_error = 'repair call failed'
      return None

    parsed, error = self._try_parse(repaired.content, model)
    if parsed is None:
      self.last_error = error
      log_event(logger, 30, 'json_repair_failed', label=label, error=error)
      return None

    log_event(logger, 20, 'json_repaired', label=label)
    return parsed

  def usage_snapshot(self) -> Dict[str, Any]:
    '''Summarize usage and call statistics.

    Returns:
      Dict with budget snapshot and call counters.
    '''
    snapshot = self.budget.snapshot()
    snapshot.update({
      'calls': self.calls,
      'failures': self.failures,
      'parse_repairs': self.parse_repairs,
    })
    return snapshot

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _try_parse(
    self,
    content: str,
    model: Type[ModelT],
  ) -> tuple[Optional[ModelT], str]:
    '''Attempt to parse content into the model.

    Args:
      content: Raw model output.
      model: Target pydantic model.

    Returns:
      (instance, '') on success or (None, error) on failure.
    '''
    try:
      instance = parse_into(content, model)
    except JsonParseError as exc:
      return None, str(exc)
    if not isinstance(instance, model):
      return None, f'parsed object was {type(instance).__name__}'
    return instance, ''

  def _sleep_backoff(self, attempt: int) -> None:
    '''Sleep with exponential backoff plus jitter.

    Args:
      attempt: 1-based attempt number that just failed.
    '''
    delay = self._config.retry_base_seconds * (2 ** (attempt - 1))
    delay += random.uniform(0.0, self._config.retry_base_seconds)
    time.sleep(min(delay, 8.0))

  def _trace_call(self, result: LLMResult) -> None:
    '''Emit one span for a successful LLM call.

    Args:
      result: The successful call result.
    '''
    if self._tracer is None or not self._trace_id:
      return
    with self._tracer.span(
      self._trace_id,
      'llm.call',
      label=result.label,
      model=result.model,
      provider=result.provider,
    ) as span:
      span.add_usage(result.usage)
      span.set_attr('latency_ms', round(result.latency_ms, 1))
      span.set_attr('cached', result.cached)
