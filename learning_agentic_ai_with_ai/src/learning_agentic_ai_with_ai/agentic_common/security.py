#!/usr/bin/env python
# -- coding: utf-8 --

'''Security primitives shared by all chapters.

Threat model for this course:
1. Secrets (API keys) leaking into logs, traces, or LLM context.
2. Untrusted text (tool results, web pages, user input) carrying injection
   payloads or control characters.
3. LLM-driven tool calls exceeding authority: wrong tool, bad arguments,
   unsafe quantities, or write actions without approval.
'''


from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from agentic_common.logging import get_logger

logger = get_logger(__name__)


_SECRET_ENV_NAMES = (
  'OPENROUTER_API_KEYS',
  'GROQ_API_KEYS',
  'CEREBRAS_API_KEYS',
  'NVIDIA_NIM_API_KEYS',
  'GITHUB_API_KEYS',
)

_REDACTED = '***REDACTED***'
_TOKEN_PATTERN = re.compile(r'[A-Za-z0-9_\-]{12,}')


def _collect_secrets(extra: Optional[List[str]] = None) -> Set[str]:
  '''Gather secret values from known env vars.

  Args:
    extra: Additional literal strings to treat as secrets.

  Returns:
    Set of secret strings (len >= 8 only).
  '''
  secrets: Set[str] = set()

  for name in _SECRET_ENV_NAMES:
    raw = os.getenv(name, '')
    for part in raw.split(','):
      part = part.strip()
      if len(part) >= 8:
        secrets.add(part)

  for value in (extra or []):
    if len(value) >= 8:
      secrets.add(value)

  return secrets


def redact_secrets(
  text: str,
  extra_secrets: Optional[List[str]] = None,
) -> str:
  '''Replace known secret values with a redaction marker.

  Args:
    text: Text that may contain secrets.
    extra_secrets: Additional literal secrets to remove.

  Returns:
    Sanitized text.
  '''
  if not text:
    return text

  for secret in _collect_secrets(extra_secrets):
    if secret in text:
      text = text.replace(secret, _REDACTED)
  return text


def redact_generic_tokens(text: str, min_length: int = 20) -> str:
  '''Redact long high-entropy-looking tokens that are not common words.

  Conservative heuristic: strings of [A-Za-z0-9_-] with no spaces and length
  >= min_length are treated as potential credentials.

  Args:
    text: Input text.
    min_length: Minimum token length to redact.

  Returns:
    Text with generic long tokens redacted.
  '''
  def _should(match: re.Match[str]) -> str:
    token = match.group(0)
    looks_hexy = bool(re.fullmatch(r'[0-9a-fA-F]+', token))
    has_digits_and_letters = any(c.isdigit() for c in token) and any(
      c.isalpha() for c in token
    )
    if len(token) >= min_length and (looks_hexy or has_digits_and_letters):
      return _REDACTED
    return token

  return _TOKEN_PATTERN.sub(_should, text)


_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def sanitize_untrusted(
  text: str,
  max_chars: int = 8000,
) -> str:
  '''Make untrusted text safe to embed in prompts or logs.

  - Removes control characters (except tab/newline handled below).
  - Collapses newlines to spaces to keep single-line structures intact.
  - Caps length.

  Args:
    text: Raw untrusted text.
    max_chars: Maximum retained characters.

  Returns:
    Sanitized, length-capped text.
  '''
  cleaned = _CONTROL_CHARS.sub('', text or '')
  cleaned = cleaned.replace('\r', ' ').replace('\n', ' ')
  if len(cleaned) > max_chars:
    cleaned = cleaned[: max_chars - 3] + '...'
  return cleaned


def truncate_json(
  payload: Dict[str, Any],
  max_chars: int,
) -> Dict[str, Any]:
  '''Serialize a payload to JSON and hard-cap its size.

  Args:
    payload: JSON-serializable payload.
    max_chars: Maximum serialized length.

  Returns:
    Dict with 'truncated' flag when cut.
  '''
  serialized = json.dumps(payload, default=str)
  if len(serialized) <= max_chars:
    return {'data': payload, 'truncated': False}

  return {
    'data': serialized[: max_chars - 3] + '...',
    'truncated': True,
  }


# ---------------------------------------------------------------------------
# Minimal JSON-Schema validator (subset sufficient for MCP tool inputs)
# ---------------------------------------------------------------------------

def validate_against_json_schema(
  data: Any,
  schema: Dict[str, Any],
  path: str = '$',
) -> List[str]:
  '''Validate data against a JSON schema subset used by MCP tools.

  Supports: type, required, properties, enum, minimum, maximum,
  additionalProperties, items, and minLength/maxLength for strings.

  Args:
    data: Value to validate.
    schema: JSON schema (subset).
    path: Location path for error messages.

  Returns:
    List of human-readable error strings (empty when valid).
  '''
  errors: List[str] = []
  expected = schema.get('type')

  if expected is not None:
    type_ok = {
      'object': isinstance(data, dict),
      'array': isinstance(data, list),
      'string': isinstance(data, str),
      'integer': isinstance(data, int) and not isinstance(data, bool),
      'number': isinstance(data, (int, float)) and not isinstance(data, bool),
      'boolean': isinstance(data, bool),
      'null': data is None,
    }.get(expected, True)

    if not type_ok:
      return [f'{path}: expected type {expected}, got {type(data).__name__}']

  enum_values = schema.get('enum')
  if enum_values is not None and data not in enum_values:
    errors.append(f'{path}: value {data!r} not in enum {enum_values!r}')

  if expected == 'string':
    min_length = schema.get('minLength')
    max_length = schema.get('maxLength')
    if min_length is not None and len(data) < min_length:
      errors.append(f'{path}: shorter than minLength {min_length}')
    if max_length is not None and len(data) > max_length:
      errors.append(f'{path}: longer than maxLength {max_length}')

  if expected in ('integer', 'number'):
    minimum = schema.get('minimum')
    maximum = schema.get('maximum')
    if minimum is not None and data < minimum:
      errors.append(f'{path}: below minimum {minimum}')
    if maximum is not None and data > maximum:
      errors.append(f'{path}: above maximum {maximum}')

  if expected == 'object' and isinstance(data, dict):
    for req in schema.get('required', []):
      if req not in data:
        errors.append(f'{path}: missing required property {req!r}')

    properties = schema.get('properties', {})
    for key, value in data.items():
      if key in properties:
        errors.extend(
          validate_against_json_schema(value, properties[key], f'{path}.{key}')
        )
      elif schema.get('additionalProperties') is False:
        errors.append(f'{path}: unexpected property {key!r}')

  if expected == 'array' and isinstance(data, list):
    item_schema = schema.get('items')
    if item_schema:
      for index, item in enumerate(data):
        errors.extend(
          validate_against_json_schema(item, item_schema, f'{path}[{index}]')
        )

  return errors
