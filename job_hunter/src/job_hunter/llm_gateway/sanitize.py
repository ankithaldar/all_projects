#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Utilities for removing thinking/reasoning tokens from displayed output.'''


from __future__ import annotations

import re
from collections.abc import Callable

_THINK_TAG_GROUP = (
  r'(think|thinking|thoughts|reasoning|chain[_-]?of[_-]?thought)'
)

_MAX_INPUT_CHARS = 200_000

_THINK_TAGS_NORM = frozenset({
  'think',
  'thinking',
  'thoughts',
  'reasoning',
  'chainofthought',
})

_STREAM_START_RE = re.compile(
  r'<\s*'
  + _THINK_TAG_GROUP
  + r'\b[^>]*>',
  re.IGNORECASE,
)

_TAG_PREFIXES = (
  'think',
  'thinking',
  'thoughts',
  'reasoning',
  'chain_of_thought',
  'chain-of-thought',
  'chainofthought',
)


def _skip_spaces(text: str, index: int) -> int:
  '''Return index after any whitespace at position.'''
  while index < len(text) and text[index].isspace():
    index += 1
  return index


def _parse_angle_tag(
  text: str,
  lowered: str,
  index: int,
) -> tuple[str, str, int] | None:
  '''Parse a think-like angle tag at index, if any (linear, no regex).'''
  pos = _skip_spaces(text, index + 1)
  if pos >= len(text):
    return None
  is_close = False
  if text[pos] == '/':
    is_close = True
    pos = _skip_spaces(text, pos + 1)
  start = pos
  while pos < len(text) and (text[pos].isalnum() or text[pos] in '_-'):
    pos += 1
  if start == pos:
    return None
  norm = lowered[start:pos].replace('_', '').replace('-', '')
  if norm not in _THINK_TAGS_NORM:
    return None
  gt = text.find('>', pos)
  if gt == -1:
    return None
  return ('close' if is_close else 'open', norm, gt + 1)


def _parse_bracket_tag(
  text: str,
  lowered: str,
  index: int,
) -> tuple[str, str, int] | None:
  '''Parse a [think]/[/think] bracket tag at index, if any.'''
  pos = _skip_spaces(text, index + 1)
  if pos >= len(text):
    return None
  is_close = False
  if text[pos] == '/':
    is_close = True
    pos = _skip_spaces(text, pos + 1)
  if lowered[pos:pos + 5] != 'think':
    return None
  end = _skip_spaces(text, pos + 5)
  # 'think' must be a whole word inside brackets
  if end < len(text) and (text[end].isalnum() or text[end] in '_-'):
    return None
  if end >= len(text) or text[end] != ']':
    return None
  return ('close' if is_close else 'open', 'think', end + 1)


def _find_angle_close(
  text: str,
  lowered: str,
  start: int,
  tag: str,
) -> int:
  '''Find end index (after '>') of matching close tag, or -1.'''
  idx = start
  while True:
    lt = text.find('<', idx)
    if lt == -1:
      return -1
    parsed = _parse_angle_tag(text, lowered, lt)
    if parsed is not None and parsed[0] == 'close' and parsed[1] == tag:
      return parsed[2]
    idx = lt + 1


def _find_bracket_close(
  text: str,
  lowered: str,
  start: int,
) -> int:
  '''Find end index (after ']') of [/think], or -1.'''
  idx = start
  while True:
    br = text.find('[', idx)
    if br == -1:
      return -1
    parsed = _parse_bracket_tag(text, lowered, br)
    if parsed is not None and parsed[0] == 'close':
      return parsed[2]
    idx = br + 1


def sanitize_think_tokens(content: str) -> str:
  '''Remove thinking/reasoning tokens from content intended for display.

  Linear scan (str.find based) -- no backtracking regex, caps input so
  pathological input (e.g. thousands of unclosed ``<think>``) is O(n).

  Args:
    content: Raw model output.

  Returns:
    Sanitized content safe for display.
  '''
  if not content:
    return ''

  text = content[:_MAX_INPUT_CHARS]
  lowered = text.lower()
  out: list[str] = []
  index = 0
  length = len(text)
  bracket_no_close = False

  while index < length:
    ch = text[index]
    if ch == '<':
      parsed = _parse_angle_tag(text, lowered, index)
      if parsed is not None:
        kind, tag, end = parsed
        if kind == 'close':
          index = end
          continue
        close_end = _find_angle_close(text, lowered, end, tag)
        if close_end == -1:
          break  # unclosed: strip to end (previous contract)
        index = close_end
        continue
      out.append(ch)
      index += 1
      continue
    if ch == '[':
      parsed = _parse_bracket_tag(text, lowered, index)
      if parsed is not None and parsed[0] == 'open' and not bracket_no_close:
        close_end = _find_bracket_close(text, lowered, parsed[2])
        if close_end != -1:
          index = close_end
          continue
        bracket_no_close = True  # no close after here => none for later opens
      out.append(ch)
      index += 1
      continue
    out.append(ch)
    index += 1

  sanitized = ''.join(out)
  sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
  sanitized = sanitized.strip()

  return sanitized


def _could_be_think_start(suffix: str) -> bool:
  '''Check whether a suffix could become a thinking start tag.

  Args:
    suffix: Trailing text from a streaming buffer.

  Returns:
    True if the suffix could become a start tag.
  '''
  if not suffix.startswith('<'):
    return False

  body = suffix[1:].lstrip()
  if not body:
    return True

  lowered = body.lower()

  for tag in _TAG_PREFIXES:
    if tag.startswith(lowered):
      return True

    if lowered.startswith(tag):
      return True

  return False


def _could_be_think_end(suffix: str) -> bool:
  '''Check whether a suffix could become a thinking end tag.

  Args:
    suffix: Trailing text from a streaming buffer.

  Returns:
    True if the suffix could become an end tag.
  '''
  if not suffix.startswith('<'):
    return False

  body = suffix[1:].lstrip()
  if body and not body.startswith('/'):
    return False
  if body:
    body = body[1:].lstrip()
  if not body:
    return True

  lowered = body.lower()

  for tag in _TAG_PREFIXES:
    if tag.startswith(lowered):
      return True

    if lowered.startswith(tag):
      return True

  return False


def _possible_partial_cut(
  buffer: str,
  checker: Callable[[str], bool],
) -> int:
  '''Find the latest possible partial marker start index.

  Args:
    buffer: Current streaming buffer.
    checker: Function that checks whether a suffix could be a marker.

  Returns:
    Index where the possible partial marker begins, or -1.
  '''
  for index in range(len(buffer) - 1, -1, -1):
    if buffer[index] == '<' and checker(buffer[index:]):
      return index

  return -1


class StreamThinkSanitizer:
  '''Incrementally sanitizes streamed LLM output.'''

  def __init__(self) -> None:
    '''Initialize sanitizer state.'''
    self._buffer = ''
    self._inside = False
    self._end_pattern = None

  def feed(self, chunk: str) -> str:
    '''Feed one streamed chunk and return display-safe text.

    Args:
      chunk: New raw text chunk.

    Returns:
      Display-safe text for this chunk.
    '''
    if not chunk:
      return ''

    self._buffer += chunk
    output = ''

    while True:
      if not self._inside:
        match = _STREAM_START_RE.search(self._buffer)

        if match:
          output += sanitize_think_tokens(self._buffer[:match.start()])
          tag = match.group(1).lower()

          self._end_pattern = re.compile(
            r'<\s*/\s*' + re.escape(tag) + r'\s*>',
            re.IGNORECASE,
          )
          self._inside = True
          self._buffer = self._buffer[match.end():]
          continue

        cut = _possible_partial_cut(self._buffer, _could_be_think_start)

        if cut == -1:
          output += sanitize_think_tokens(self._buffer)
          self._buffer = ''
        else:
          output += sanitize_think_tokens(self._buffer[:cut])
          self._buffer = self._buffer[cut:]

        return output

      if self._end_pattern is not None:
        match = self._end_pattern.search(self._buffer)

        if match:
          self._inside = False
          self._end_pattern = None
          self._buffer = self._buffer[match.end():]
          continue

      cut = _possible_partial_cut(self._buffer, _could_be_think_end)

      if cut == -1:
        self._buffer = ''
      else:
        self._buffer = self._buffer[cut:]

      return output

  def flush(self) -> str:
    '''Flush remaining buffered text.

    Returns:
      Remaining display-safe text.
    '''
    if self._inside:
      self._buffer = ''
      self._inside = False
      self._end_pattern = None
      return ''

    remaining = self._buffer
    self._buffer = ''

    return sanitize_think_tokens(remaining)
