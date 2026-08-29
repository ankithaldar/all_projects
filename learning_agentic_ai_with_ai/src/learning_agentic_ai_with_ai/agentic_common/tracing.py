#!/usr/bin/env python
# -- coding: utf-8 --

'''Lightweight tracing: spans with attributes, timings, and token usage.

A *trace* is one end-to-end agent run (e.g. answering one task). A *span* is
one step inside it (one LLM call, one tool call). Spans form a tree via
`parent_span_id` and are appended as JSON lines to
`data/traces/<trace_id>.jsonl` so any jq/duckdb/pandas session can analyze
them later.
'''


from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field

from agentic_common import paths
from agentic_common.logging import get_logger

logger = get_logger(__name__)


class TokenUsage(BaseModel):
  '''Token usage for one call or accumulated over a run.'''

  model_config = ConfigDict(extra='ignore')

  input_tokens: int = 0
  output_tokens: int = 0
  total_tokens: int = 0

  def add(self, other: 'TokenUsage') -> 'TokenUsage':
    '''Return the sum of this usage and another.

    Args:
      other: Usage to add.

    Returns:
      New accumulated TokenUsage.
    '''
    return TokenUsage(
      input_tokens=self.input_tokens + other.input_tokens,
      output_tokens=self.output_tokens + other.output_tokens,
      total_tokens=self.total_tokens + other.total_tokens,
    )


class SpanRecord(BaseModel):
  '''One traced unit of work.'''

  model_config = ConfigDict(extra='ignore')

  trace_id: str
  span_id: str
  parent_span_id: Optional[str] = None
  name: str
  started_at: str
  ended_at: Optional[str] = None
  duration_ms: float = 0.0
  status: str = 'ok'
  error: Optional[str] = None
  attributes: Dict[str, Any] = Field(default_factory=dict)
  usage: TokenUsage = Field(default_factory=TokenUsage)


class _ActiveSpan:
  '''Mutable bookkeeping for a span that is currently open.'''

  def __init__(
    self,
    record: SpanRecord,
    started_perf: float,
    tracer: 'Tracer',
  ) -> None:
    '''Initialize an active span.

    Args:
      record: The span record under construction.
      started_perf: `time.perf_counter()` at start.
      tracer: Owning tracer.
    '''
    self.record = record
    self.started_perf = started_perf
    self.tracer = tracer
    self.usage = TokenUsage()

  def set_attr(self, key: str, value: Any) -> None:
    '''Attach an attribute to the span.

    Args:
      key: Attribute name.
      value: JSON-serializable value.
    '''
    self.record.attributes[key] = value

  def add_usage(self, usage: TokenUsage) -> None:
    '''Accumulate token usage into the span.

    Args:
      usage: Usage from one LLM call.
    '''
    self.usage = self.usage.add(usage)


class Tracer:
  '''Writes span records to a JSONL trace file, thread-safe.'''

  def __init__(
    self,
    trace_dir: Optional[Path] = None,
    enabled: bool = True,
  ) -> None:
    '''Initialize the tracer.

    Args:
      trace_dir: Directory for trace files (default data/traces).
      enabled: When False, spans are cheap no-ops (still built).
    '''
    self._dir = trace_dir or paths.TRACES_DIR
    self._enabled = enabled
    self._lock = threading.Lock()

  def new_trace_id(self) -> str:
    '''Generate a fresh trace id.

    Returns:
      UUID4 hex string.
    '''
    return uuid.uuid4().hex

  @contextmanager
  def span(
    self,
    trace_id: str,
    name: str,
    parent_span_id: Optional[str] = None,
    **attrs: Any,
  ) -> Iterator[_ActiveSpan]:
    '''Context manager that opens, times, and closes a span.

    Args:
      trace_id: Owning trace id.
      name: Span name, e.g. 'llm.complete' or 'tool.call'.
      parent_span_id: Parent span id if nested.
      **attrs: Initial attributes.

    Yields:
      An _ActiveSpan you can mutate while the block runs.
    '''
    record = SpanRecord(
      trace_id=trace_id,
      span_id=uuid.uuid4().hex[:16],
      parent_span_id=parent_span_id,
      name=name,
      started_at=datetime.now(timezone.utc).isoformat(),
      attributes=dict(attrs),
    )
    active = _ActiveSpan(record, time.perf_counter(), self)

    try:
      yield active
    except Exception as exc:
      active.record.status = 'error'
      active.record.error = str(exc)
      raise
    finally:
      active.record.duration_ms = (
        time.perf_counter() - active.started_perf
      ) * 1000.0
      active.record.ended_at = datetime.now(timezone.utc).isoformat()
      active.record.usage = active.usage
      self._write(active.record)

  def _write(self, record: SpanRecord) -> None:
    '''Append one span record to the trace file.

    Args:
      record: Completed span record.
    '''
    if not self._enabled:
      return

    line = json.dumps(record.model_dump(mode='json'), default=str)
    try:
      with self._lock:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f'{record.trace_id}.jsonl'
        with path.open('a', encoding='utf-8') as handle:
          handle.write(line + '\n')
    except OSError as exc:
      logger.warning(
        'trace write failed',
        extra={'extra_fields': {'error': str(exc)}},
      )


class NullTracer(Tracer):
  '''Tracer that discards output - used in tests and offline runs.'''

  def __init__(self) -> None:
    '''Initialize a disabled tracer.'''
    super().__init__(enabled=False)

  def _write(self, record: SpanRecord) -> None:  # noqa: D102
    return
