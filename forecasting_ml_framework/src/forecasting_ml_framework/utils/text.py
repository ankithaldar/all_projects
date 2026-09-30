#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text and parameter parsing helpers.

The two SQL-rendering helpers live here rather than in the dataset adapters
because the property they provide is a property of *text*, not of any one
back end: a name and a literal are rendered the same way whichever engine will
parse them, so the behaviour cannot diverge between the two.
"""

from __future__ import annotations

import datetime
import re
from decimal import Decimal
from typing import Any

from forecasting_ml_framework.exceptions import InvalidWriteModeError, MissingParameterError

#: Characters permitted in a SQL identifier, used by :func:`quote_identifier`.
#:
#: The check is a positive allow-list rather than a scan for dangerous
#: characters. The framework's own security argument is that runtime values come
#: from a trusted dispatcher rather than from an end user, so these helpers are
#: a second line of defence -- and a second line is only worth having if it fails
#: closed. An identifier outside the set is refused, not escaped, because a
#: silently-quoted name is harder to diagnose than a loud refusal.
_IDENTIFIER_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

#: The SQL string-literal delimiter, and the doubling that escapes it.
#:
#: Named rather than written inline because the character has to appear inside an
#: f-string that itself contains a quote, and a named constant keeps both the
#: literal and the escape legible at the point of use.
QUOTE_CHARACTER = "'"


def quote_identifier(name: str) -> str:
  """Render a SQL identifier, refusing anything outside the safe set.

  Args:
    name: The bare identifier.

  Returns:
    The identifier wrapped in backticks.

  Raises:
    InvalidWriteModeError: If the name is not a plain SQL identifier. A dataset
      name, a column name or a partition field carrying anything else -- a dot, a
      space, a quote, a semicolon -- is a configuration error that must be
      visible rather than interpolated.
  """
  if not _IDENTIFIER_PATTERN.match(str(name)):
    raise InvalidWriteModeError(
      f'{name!r} is not a plain SQL identifier. Identifiers must match '
      f'[A-Za-z_][A-Za-z0-9_]* so that a name can never change the meaning of a statement. '
      f'If a qualified name is intended, pass its parts separately.',
      name=str(name),
    )
  return f'`{name}`'


def quote_literal(value: Any) -> str:
  """Render a SQL literal from a Python value.

  A value is quoted, never escaped: the doubled-quote convention is implemented
  by *doubling* the quote character, which is the standard SQL literal form, and
  a backslash is rejected outright because its meaning is dialect-dependent. That
  makes a hostile value safe by construction rather than by a blacklist.

  Args:
    value: A string, date, number, boolean or ``None``.

  Returns:
    The rendered literal.

  Raises:
    InvalidWriteModeError: If the value is of a type that cannot be rendered
      unambiguously, or is a string containing a backslash.
  """
  if value is None:
    return 'NULL'
  if isinstance(value, bool):
    return 'TRUE' if value else 'FALSE'
  if isinstance(value, (int, float, Decimal)):
    return repr(value) if isinstance(value, float) else str(value)
  if isinstance(value, (datetime.datetime, datetime.date)):
    text = value.isoformat()
    if isinstance(value, datetime.datetime):
      text = text.replace('T', ' ')
    return f"'{text}'"
  if isinstance(value, str):
    if '\\' in value:
      raise InvalidWriteModeError(
        f'The value {value!r} contains a backslash, whose meaning is dialect-dependent. Refusing '
        f'to render it rather than guessing the escaping convention.',
        value=value,
      )
    escaped = value.replace(QUOTE_CHARACTER, QUOTE_CHARACTER * 2)
    return f"'{escaped}'"
  raise InvalidWriteModeError(
    f'Cannot render {type(value).__name__} as a SQL literal. Supported types are str, bool, int, '
    f'float, Decimal, date, datetime and None.',
    value=repr(value),
  )


def match_any(patterns: list[str] | None, value: str) -> bool:
  """Report whether any pattern matches a value.

  Patterns are regular expressions anchored at the *start* of the value, which
  is the behaviour the whole framework's feature-exclusion configuration depends
  on: ``'^tenure_.*'`` excludes all tenure-derived columns, while a bare
  ``'tenure'`` also matches ``'tenure_01'``.

  Args:
    patterns: A list of regular expressions, or ``None``.
    value: The candidate string.

  Returns:
    ``True`` if at least one pattern matches, ``False`` otherwise.
  """
  if not patterns:
    return False
  return any(re.match(pattern, value) for pattern in patterns)


def require(params: dict[str, Any], key: str, **context: object) -> Any:
  """Fetch a mandatory parameter or raise a descriptive error.

  Bare ``params['key']`` indexing was the framework's primary validation
  mechanism and its principal source of unhelpful tracebacks: the ``KeyError``
  named the key but gave no indication of which stage or model wanted it. This
  helper keeps the fail-fast behaviour while naming the context.

  Args:
    params: The parameter mapping.
    key: The mandatory key.
    **context: Diagnostic context attached to the raised error.

  Returns:
    The parameter value.

  Raises:
    MissingParameterError: If the key is absent.
  """
  if key not in params:
    available = ', '.join(sorted(str(name) for name in params)[:25])
    raise MissingParameterError(
      f'Required parameter {key!r} is missing. Supply it as a runtime parameter '
      f'(run --params={key}=<value>). Available keys include: {available}',
      key=key,
      **context,
    )
  return params[key]


def split_params(raw: str) -> dict[str, Any]:
  """Parse a ``key:value,key.subkey:value`` string into a nested dictionary.

  This is the framework's standalone command-line parameter parser. It is used
  by the bootstrap entry point, which must build a nested parameter structure
  *before* the orchestration framework starts, because the configuration
  rewriter needs to know which top-level keys were supplied at runtime.

  Args:
    raw: A comma-separated list of ``key`` or ``key.subkey`` assignments. A value
      may be quoted to protect a comma.

  Returns:
    A nested dictionary. Values are converted to ``int``, ``float`` or ``bool``
      when unambiguous, and left as strings otherwise.

  Raises:
    ValueError: If a segment does not contain a colon.
  """
  result: dict[str, Any] = {}
  for segment in _split_top_level(raw):
    segment = segment.strip()
    if not segment:
      continue
    if ':' not in segment:
      raise ValueError(f'Parameter segment {segment!r} must be of the form key:value or key.subkey:value')
    key, _, value = segment.partition(':')
    keys = [part.strip() for part in key.strip().split('.') if part.strip()]
    if not keys:
      raise ValueError(f'Parameter segment {segment!r} has an empty key')
    _assign_nested(result, keys, _try_convert(value.strip()))
  return result


def _split_top_level(raw: str) -> list[str]:
  """Split on commas that are not inside quotes or brackets.

  Unbalanced grouping is **refused** rather than tolerated. A closing bracket with
  no opener, an unterminated quote, or a leftover opener all meant the splitter
  kept consuming: ``'a:1],b:2'`` parsed as a single assignment whose value was
  ``'1],b:2'``, so the operator's second parameter was silently folded into the
  first. A run then executed with a parameter the operator never wrote, and the
  run log showed the mangled key rather than the typo that caused it.

  Args:
    raw: The raw parameter string.

  Returns:
    The top-level segments.

  Raises:
    ValueError: If a quote is unterminated or the brackets do not balance.
  """
  segments: list[str] = []
  depth = 0
  quote: str | None = None
  current: list[str] = []
  for char in raw:
    if quote:
      current.append(char)
      if char == quote:
        quote = None
      continue
    if char in ('"', "'"):
      quote = char
      current.append(char)
    elif char in '[{(':
      depth += 1
      current.append(char)
    elif char in ']})':
      depth -= 1
      if depth < 0:
        raise ValueError(
          f'Unbalanced {char!r} in the parameter string. A closing bracket with no opener means the '
          f'segments after it cannot be split, so the parameters would be silently merged: {raw!r}'
        )
      current.append(char)
    elif char == ',' and depth == 0:
      segments.append(''.join(current))
      current = []
    else:
      current.append(char)
  if quote is not None:
    raise ValueError(
      f'Unterminated {quote!r} in the parameter string. An unterminated quote hides every '
      f'comma after it, so the remaining parameters would be silently merged: {raw!r}'
    )
  if depth > 0:
    raise ValueError(
      f'Unclosed bracket in the parameter string: {depth} grouping level(s) remain open, so every '
      f'comma after them would be silently merged into one value: {raw!r}'
    )
  segments.append(''.join(current))
  return segments


def _assign_nested(target: dict[str, Any], keys: list[str], value: Any) -> None:
  """Assign a value into a nested dictionary, creating intermediate levels.

  Args:
    target: The dictionary to mutate.
    keys: The dotted key path.
    value: The value to assign.
  """
  cursor = target
  for part in keys[:-1]:
    nxt = cursor.get(part)
    if not isinstance(nxt, dict):
      nxt = {}
      cursor[part] = nxt
    cursor = nxt
  cursor[keys[-1]] = value


def _try_convert(value: str) -> Any:
  """Convert a raw string to a number or boolean where unambiguous.

  Args:
    value: The raw string.

  Returns:
    The converted value, or a string.
  """
  lowered = value.lower()
  if lowered in ('true', 'false'):
    return lowered == 'true'
  if lowered in ('none', 'null'):
    return None
  try:
    return int(value)
  except ValueError:
    pass
  try:
    return float(value)
  except ValueError:
    return value
