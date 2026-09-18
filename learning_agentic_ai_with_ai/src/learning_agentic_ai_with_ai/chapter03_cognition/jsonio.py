#!/usr/bin/env python
# -- coding: utf-8 --

'''Defensive extraction helpers for model output (Chapter 3, self-contained).

Chapter 3 deliberately avoids importing Chapter 2, so this module provides
the small parsing surface the cognitive stack needs:

  - `extract_json`: pull the first JSON value out of free-form output using
    bounded, safe repairs (code fences, balanced spans, trailing commas,
    smart quotes, Python-literal fallback),
  - `parse_into`: validate that JSON against a pydantic model,
  - `extract_code_block`: pull generated Python out of fences or from the
    first `def` onward.

All functions are pure and raise `JsonParseError` with a short preview so
callers can re-prompt or degrade - never guess silently.
'''


from __future__ import annotations

import ast
import json
import re
from typing import Any, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar('ModelT', bound=BaseModel)

_FENCE_RE = re.compile(r'```(?:json|python|py)?\s*\n?(.*?)```', re.DOTALL)
_PY_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n?(.*?)```', re.DOTALL)
_SMART_QUOTES = {
  '\u201c': '"',
  '\u201d': '"',
  '\u2018': "'",
  '\u2019': "'",
}


class JsonParseError(ValueError):
  '''Raised when model output cannot be decoded into the expected shape.'''


def _preview(text: str, limit: int = 160) -> str:
  '''Collapse text into a short single-line preview.

  Args:
    text: Source text.
    limit: Maximum preview length.

  Returns:
    Collapsed preview string.
  '''
  collapsed = re.sub(r'\s+', ' ', (text or '').strip())
  if len(collapsed) <= limit:
    return collapsed
  return collapsed[: limit - 3] + '...'


def _repair(candidate: str) -> str:
  '''Apply safe textual repairs to a JSON-ish candidate.

  Args:
    candidate: Candidate JSON text.

  Returns:
    Repaired candidate.
  '''
  repaired = candidate.strip()
  for bad, good in _SMART_QUOTES.items():
    repaired = repaired.replace(bad, good)
  repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
  return repaired


def _balanced_from(text: str, start: int) -> Optional[str]:
  '''Extract a balanced brace/bracket span starting at an index.

  Args:
    text: Source text.
    start: Index of the opening brace/bracket.

  Returns:
    Balanced substring, or None when the span never closes.
  '''
  opener = text[start]
  closer = '}' if opener == '{' else ']'
  depth = 0
  in_string = False
  escaped = False

  for index in range(start, len(text)):
    char = text[index]
    if in_string:
      if escaped:
        escaped = False
      elif char == '\\':
        escaped = True
      elif char == '"':
        in_string = False
      continue

    if char == '"':
      in_string = True
    elif char == opener:
      depth += 1
    elif char == closer:
      depth -= 1
      if depth == 0:
        return text[start: index + 1]
  return None


def _balanced_spans(text: str, max_spans: int = 12) -> List[str]:
  '''Collect balanced brace/bracket spans in appearance order.

  Scanning multiple starts matters when untrusted-data wrappers like
  `[BEGIN TOOL DATA] {...}` create an early fake candidate.

  Args:
    text: Source text.
    max_spans: Maximum candidates to collect.

  Returns:
    List of balanced substrings.
  '''
  spans: List[str] = []
  index = 0
  while index < len(text) and len(spans) < max_spans:
    if text[index] in '{[':
      span = _balanced_from(text, index)
      if span is not None:
        spans.append(span)
        index += len(span)
        continue
    index += 1
  return spans


def extract_json(text: str) -> Any:
  '''Extract the first JSON value from free-form model output.

  Attempt order (first success wins): whole text, fenced blocks, balanced
  spans, repaired variants, `ast.literal_eval` fallback.

  Args:
    text: Raw model output.

  Returns:
    Decoded JSON value.

  Raises:
    JsonParseError: When no strategy yields a value.
  '''
  raw = (text or '').strip()
  if not raw:
    raise JsonParseError('empty output; expected JSON')

  candidates: List[str] = [raw]
  candidates.extend(match.group(1) for match in _FENCE_RE.finditer(raw))
  candidates.extend(_balanced_spans(raw))

  errors: List[str] = []
  for candidate in candidates:
    for attempt in (candidate, _repair(candidate)):
      try:
        return json.loads(attempt)
      except json.JSONDecodeError as exc:
        errors.append(str(exc))

  for candidate in candidates:
    try:
      return ast.literal_eval(_repair(candidate))
    except (ValueError, SyntaxError) as exc:
      errors.append(str(exc))

  last_error = errors[-1] if errors else 'unknown'
  raise JsonParseError(
    f'no JSON found (last error: {last_error}; preview: {_preview(raw)})'
  )


def parse_into(text: str, model: Type[ModelT]) -> ModelT:
  '''Parse model output into a pydantic model instance.

  Args:
    text: Raw model output.
    model: Target pydantic model class.

  Returns:
    Validated model instance.

  Raises:
    JsonParseError: On extraction or validation failure.
  '''
  payload = extract_json(text)
  try:
    return model.model_validate(payload)
  except ValidationError as exc:
    raise JsonParseError(
      f'JSON did not match {model.__name__}: '
      f'{exc.errors(include_url=False)[:3]}'
    ) from exc


def extract_code_block(text: str) -> str:
  '''Extract generated Python source from model output.

  Prefers fenced blocks; otherwise starts at the first `def` line so
  leading prose is discarded.

  Args:
    text: Raw model output.

  Returns:
    Python source text.

  Raises:
    JsonParseError: When no code-like content is found.
  '''
  raw = (text or '').strip()
  if not raw:
    raise JsonParseError('empty output; expected Python code')

  fences = [
    match.group(1) for match in _PY_FENCE_RE.finditer(raw)
  ]
  if fences:
    code = max(fences, key=len).strip()
    if code:
      return code

  def_index = raw.find('def ')
  if def_index >= 0:
    return raw[def_index:].strip()

  raise JsonParseError(f'no Python code found (preview: {_preview(raw)})')
