#!/usr/bin/env python
# -- coding: utf-8 --

'''Robust parsers for model output.

LLMs are not compilers: even with a strict output contract they emit code
fences, leading prose, trailing commas, smart quotes, or omit a closing
brace. Production agents therefore parse *defensively*:

  1. try the strict interpretation first (raw JSON, exact tags),
  2. apply bounded, safe repairs (fence stripping, balanced-brace slicing,
     trailing-comma removal, `ast.literal_eval` fallback),
  3. if everything fails, raise `ParseError` with a short preview so the
     caller can re-prompt or degrade - never guess silently.

All functions are pure and synchronous, which makes them cheap to unit test
without any LLM involved.
'''


from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chapter02_planning.schemas import Critique, Plan

_FENCE_RE = re.compile(
  r'```(?:json|JSON)?\s*\n?(.*?)```', re.DOTALL,
)

_THINK_BLOCK_RE = re.compile(
  r'<\s*(?:reasoning|thinking|thoughts)\s*>(.*?)'
  r'<\s*/\s*(?:reasoning|thinking|thoughts)\s*>',
  re.IGNORECASE | re.DOTALL,
)

_ANSWER_BLOCK_RE = re.compile(
  r'<\s*answer\s*>(.*?)(?:<\s*/\s*answer\s*>|$)',
  re.IGNORECASE | re.DOTALL,
)

_ANSWER_LINE_RE = re.compile(r'^\s*Answer\s*:\s*(.*)$', re.IGNORECASE)

_STEP_LINE_RE = re.compile(r'^\s*(?:step\s*)?(\d{1,2})[.)\-:]\s+(.*)$',
                           re.IGNORECASE)

_BULLET_LINE_RE = re.compile(r'^\s*[-*]\s+(.*)$')

_FINAL_ANSWER_RE = re.compile(r'Final Answer\s*:\s*(.*)$',
                              re.IGNORECASE | re.DOTALL)

_THOUGHT_RE = re.compile(
  r'Thought\s*:\s*(.*?)(?=\n\s*(?:Action|Final Answer|Observation)\s*:|$)',
  re.IGNORECASE | re.DOTALL,
)

_ACTION_RE = re.compile(
  r'Action\s*:\s*(.*?)(?=\n\s*(?:Action Input|Thought|Observation|'
  r'Final Answer)\s*:|$)',
  re.IGNORECASE | re.DOTALL,
)

_ACTION_INPUT_RE = re.compile(
  r'Action Input\s*:\s*(.*?)(?=\n\s*(?:Thought|Action|Observation|'
  r'Final Answer)\s*:|$)',
  re.IGNORECASE | re.DOTALL,
)

_KEY_VALUE_RE = re.compile(
  r'([A-Za-z_][A-Za-z0-9_-]*)\s*[:=]\s*'
  r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,;\n]+)',
)

_NUMBER_RE = re.compile(r'(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])')

_SMART_QUOTES = {
  '\u201c': '"',
  '\u201d': '"',
  '\u2018': "'",
  '\u2019': "'",
}


class ParseError(ValueError):
  '''Raised when model output cannot be parsed into the expected shape.'''


class CoTParsed(BaseModel):
  '''Parsed chain-of-thought output.'''

  model_config = ConfigDict(extra='ignore')

  steps: List[str] = Field(default_factory=list)
  answer: str = ''
  ok: bool = True
  error: Optional[str] = None


class ReActParsed(BaseModel):
  '''Parsed ReAct turn (either an action or a final answer).'''

  model_config = ConfigDict(extra='ignore')

  thought: str = ''
  action: str = ''
  action_input: Dict[str, Any] = Field(default_factory=dict)
  final_answer: str = ''
  ok: bool = True
  error: Optional[str] = None

  @property
  def is_final(self) -> bool:
    '''Whether this turn contains a final answer.

    Returns:
      True when `final_answer` is non-empty.
    '''
    return bool(self.final_answer.strip())


def _preview(text: str, limit: int = 160) -> str:
  '''Build a single-line preview of text for error messages.

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


def _repair_json_text(candidate: str) -> str:
  '''Apply safe textual repairs to a JSON-ish candidate.

  Args:
    candidate: Candidate JSON text.

  Returns:
    Repaired candidate text.
  '''
  repaired = candidate.strip()
  for bad, good in _SMART_QUOTES.items():
    repaired = repaired.replace(bad, good)
  repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
  return repaired


def _balanced_from(text: str, start: int) -> Optional[str]:
  '''Extract the balanced `{...}` or `[...]` span starting at an index.

  String state is tracked so braces inside strings do not confuse the scan.

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

  Scanning multiple starts matters in practice: untrusted-data wrappers like
  `[BEGIN TOOL DATA] {...}` produce an early fake candidate (`[BEGIN ...]`),
  and the real JSON appears later. The caller tries each candidate until one
  parses.

  Args:
    text: Source text.
    max_spans: Maximum number of candidate spans to collect.

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


def extract_json_block(text: str) -> Any:
  '''Extract the first JSON value from free-form model output.

  Attempt order (first success wins):
    1. the whole text as JSON,
    2. fenced code blocks,
    3. the first balanced brace/bracket span,
    4. the same span after safe repairs,
    5. `ast.literal_eval` on the repaired span (Python literals).

  Args:
    text: Raw model output.

  Returns:
    The decoded JSON value.

  Raises:
    ParseError: When no strategy yields a JSON value.
  '''
  raw = (text or '').strip()
  if not raw:
    raise ParseError('empty output; expected JSON')

  candidates: List[str] = [raw]
  candidates.extend(match.group(1) for match in _FENCE_RE.finditer(raw))
  candidates.extend(_balanced_spans(raw))

  errors: List[str] = []
  for candidate in candidates:
    repaired = _repair_json_text(candidate)
    for attempt in (candidate, repaired):
      try:
        return json.loads(attempt)
      except json.JSONDecodeError as exc:
        errors.append(str(exc))

  for candidate in candidates:
    repaired = _repair_json_text(candidate)
    try:
      return ast.literal_eval(repaired)
    except (ValueError, SyntaxError) as exc:
      errors.append(str(exc))

  last_error = errors[-1] if errors else 'unknown'
  raise ParseError(
    f'no JSON found (last error: {last_error}; '
    f'preview: {_preview(raw)})'
  )


def parse_json_into(text: str, model: type[BaseModel]) -> BaseModel:
  '''Parse model output into a pydantic model instance.

  Args:
    text: Raw model output.
    model: Target pydantic model class.

  Returns:
    Validated model instance.

  Raises:
    ParseError: On JSON extraction or pydantic validation failure.
  '''
  payload = extract_json_block(text)
  normalizer = getattr(model, 'from_llm_payload', None)
  if callable(normalizer):
    try:
      payload = normalizer(payload)
    except ValueError as exc:
      raise ParseError(f'{model.__name__} payload rejected: {exc}') from exc
  try:
    return model.model_validate(payload)
  except ValidationError as exc:
    raise ParseError(
      f'JSON did not match {model.__name__}: '
      f'{exc.errors(include_url=False)[:3]}'
    ) from exc


def parse_step_labeled(text: str) -> List[str]:
  '''Parse numbered/bulleted reasoning steps from text.

  Args:
    text: Reasoning text without the final answer.

  Returns:
    Ordered step strings (empty when nothing parseable is found).
  '''
  steps: List[str] = []
  for line in (text or '').splitlines():
    stripped = line.strip()
    if not stripped or _ANSWER_LINE_RE.match(stripped):
      continue
    match = _STEP_LINE_RE.match(stripped)
    if match:
      steps.append(match.group(2).strip())
      continue
    bullet = _BULLET_LINE_RE.match(stripped)
    if bullet:
      steps.append(bullet.group(1).strip())

  if steps:
    return steps

  paragraphs = [
    part.strip()
    for part in re.split(r'\n\s*\n', (text or '').strip())
    if part.strip() and not _ANSWER_LINE_RE.match(part.strip())
  ]
  return paragraphs


def parse_cot_output(text: str) -> CoTParsed:
  '''Parse chain-of-thought output into steps and a final answer.

  Supported shapes:
    1. `<reasoning>...</reasoning><answer>...</answer>` tags,
    2. `Reasoning:` ... `Answer: ...` step-labeled text,
    3. a trailing `Answer:` line with paragraphs above it.

  Args:
    text: Raw model output.

  Returns:
    CoTParsed with `ok=False` (and an error) when the answer marker is
    missing, so the caller can decide to re-prompt.
  '''
  raw = (text or '').strip()
  if not raw:
    return CoTParsed(ok=False, error='empty chain-of-thought output')

  think_match = _THINK_BLOCK_RE.search(raw)
  answer_match = _ANSWER_BLOCK_RE.search(raw)
  if think_match or answer_match:
    reasoning = think_match.group(1).strip() if think_match else raw
    answer = answer_match.group(1).strip() if answer_match else ''
    if not answer:
      return CoTParsed(
        steps=parse_step_labeled(reasoning), ok=False,
        error='reasoning tags found but no answer tags',
      )
    return CoTParsed(
      steps=parse_step_labeled(reasoning), answer=answer,
    )

  lines = raw.splitlines()
  answer_index: Optional[int] = None
  for index in range(len(lines) - 1, -1, -1):
    if _ANSWER_LINE_RE.match(lines[index]):
      answer_index = index
      break

  if answer_index is not None:
    match = _ANSWER_LINE_RE.match(lines[answer_index])
    answer = (match.group(1).strip() if match else '')
    reasoning = '\n'.join(lines[:answer_index])
    if not answer:
      return CoTParsed(
        steps=parse_step_labeled(reasoning), ok=False,
        error='answer marker present but empty',
      )
    return CoTParsed(steps=parse_step_labeled(reasoning), answer=answer)

  paragraphs = [p for p in raw.split('\n\n') if p.strip()]
  if len(paragraphs) >= 2:
    return CoTParsed(
      steps=parse_step_labeled('\n\n'.join(paragraphs[:-1])),
      answer=paragraphs[-1].strip(),
      ok=False,
      error='no "Answer:" marker; used last paragraph as answer',
    )

  return CoTParsed(
    steps=parse_step_labeled(raw), answer=raw,
    ok=False,
    error='no "Answer:" marker; used whole output as answer',
  )


def _parse_action_input(raw: str) -> Dict[str, Any]:
  '''Parse a ReAct `Action Input` value into a dict.

  Args:
    raw: Raw action input text.

  Returns:
    Argument dictionary.

  Raises:
    ParseError: When the input cannot be interpreted.
  '''
  candidate = (raw or '').strip().strip('`').strip()
  if not candidate or candidate.lower() in ('none', 'null', '{}', '[]'):
    return {}

  try:
    value = extract_json_block(candidate)
  except ParseError:
    value = None

  if isinstance(value, dict):
    return value
  if isinstance(value, list):
    return {'items': value}

  pairs = _KEY_VALUE_RE.findall(candidate)
  if pairs:
    parsed: Dict[str, Any] = {}
    for key, raw_value in pairs:
      parsed[key] = _coerce_scalar(raw_value.strip())
    return parsed

  raise ParseError(f'action input is not JSON or key=value: {_preview(raw)}')


def _coerce_scalar(raw: str) -> Any:
  '''Coerce a raw scalar string to bool/int/float when obvious.

  Args:
    raw: Raw scalar text (possibly quoted).

  Returns:
    Coerced Python value.
  '''
  value = raw.strip()
  if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
    return value[1:-1]
  lowered = value.lower()
  if lowered in ('true', 'false'):
    return lowered == 'true'
  if lowered in ('null', 'none'):
    return None
  try:
    return int(value)
  except ValueError:
    pass
  try:
    return float(value)
  except ValueError:
    return value


def parse_react_turn(text: str) -> ReActParsed:
  '''Parse one ReAct turn from model output.

  Args:
    text: Raw model output for a single turn.

  Returns:
    ReActParsed describing either an action or a final answer. `ok=False`
    means the turn violated the protocol and should be re-prompted.
  '''
  raw = (text or '').strip()
  if not raw:
    return ReActParsed(ok=False, error='empty ReAct turn')

  final_match = _FINAL_ANSWER_RE.search(raw)
  if final_match and final_match.group(1).strip():
    thought_match = _THOUGHT_RE.search(raw)
    return ReActParsed(
      thought=thought_match.group(1).strip() if thought_match else '',
      final_answer=final_match.group(1).strip(),
    )

  thought_match = _THOUGHT_RE.search(raw)
  action_match = _ACTION_RE.search(raw)
  input_match = _ACTION_INPUT_RE.search(raw)

  if action_match is None:
    return ReActParsed(
      thought=thought_match.group(1).strip() if thought_match else raw,
      ok=False,
      error='missing "Action:" or "Final Answer:"',
    )

  action = action_match.group(1).strip().strip('`').strip()
  if not action:
    return ReActParsed(ok=False, error='empty action name')

  try:
    action_input = _parse_action_input(
      input_match.group(1) if input_match else ''
    )
  except ParseError as exc:
    return ReActParsed(
      thought=thought_match.group(1).strip() if thought_match else '',
      action=action,
      ok=False,
      error=str(exc),
    )

  return ReActParsed(
    thought=thought_match.group(1).strip() if thought_match else '',
    action=action,
    action_input=action_input,
  )


def parse_plan_output(text: str) -> Plan:
  '''Parse and validate a plan from model output.

  Shape normalization (synonym fields, `plan`/`steps` wrappers, numeric ids)
  is owned by `Plan.from_llm_payload`, which `parse_json_into` invokes
  automatically for models that provide it.

  Args:
    text: Raw model output.

  Returns:
    Validated Plan.

  Raises:
    ParseError: When extraction or pydantic validation fails.
  '''
  return parse_json_into(text, Plan)  # type: ignore[return-value]


def parse_critique_output(text: str) -> Critique:
  '''Parse the validator's structured critique.

  Args:
    text: Raw model output.

  Returns:
    Validated Critique.

  Raises:
    ParseError: When extraction or validation fails.
  '''
  return parse_json_into(text, Critique)  # type: ignore[return-value]


def extract_number_tokens(text: str) -> List[str]:
  '''Extract numeric tokens from text (commas removed).

  Args:
    text: Source text.

  Returns:
    List of numeric token strings in appearance order.
  '''
  tokens: List[str] = []
  for match in _NUMBER_RE.finditer(text or ''):
    tokens.append(match.group(1).replace(',', ''))
  return tokens


def extract_numbers(text: str) -> set[float]:
  '''Extract numeric values from text.

  Args:
    text: Source text.

  Returns:
    Set of float values.
  '''
  values: set[float] = set()
  for token in extract_number_tokens(text):
    try:
      values.add(float(token))
    except ValueError:
      continue
  return values
