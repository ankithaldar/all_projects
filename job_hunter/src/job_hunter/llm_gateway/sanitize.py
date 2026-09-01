#!/usr/bin/env python
# -- coding: utf-8 --

'''Utilities for removing thinking/reasoning tokens from displayed output.'''


from __future__ import annotations

import re
from typing import Callable

_THINK_TAG_GROUP = (
  r'(think|thinking|thoughts|reasoning|chain[_-]?of[_-]?thought)'
)

_THINK_BLOCK_RE = re.compile(
  r'<\s*'
  + _THINK_TAG_GROUP
  + r'\b[^>]*>.*?<\s*/\s*\1\s*>',
  re.IGNORECASE | re.DOTALL,
)

_UNCLOSED_THINK_RE = re.compile(
  r'<\s*'
  + _THINK_TAG_GROUP
  + r'\b[^>]*>.*$',
  re.IGNORECASE | re.DOTALL,
)

_ORPHAN_THINK_MARKER_RE = re.compile(
  r'<\s*/?\s*'
  + _THINK_TAG_GROUP
  + r'\b[^>]*>',
  re.IGNORECASE,
)

_BRACKET_THINK_BLOCK_RE = re.compile(
  r'\[\s*think\s*\].*?\[\s*/\s*think\s*\]',
  re.IGNORECASE | re.DOTALL,
)

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


def sanitize_think_tokens(content: str) -> str:
  '''Remove thinking/reasoning tokens from content intended for display.

  Args:
    content: Raw model output.

  Returns:
    Sanitized content safe for display.
  '''
  if not content:
    return ''

  sanitized = content

  sanitized = _THINK_BLOCK_RE.sub('', sanitized)
  sanitized = _BRACKET_THINK_BLOCK_RE.sub('', sanitized)
  sanitized = _UNCLOSED_THINK_RE.sub('', sanitized)
  sanitized = _ORPHAN_THINK_MARKER_RE.sub('', sanitized)

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
  if not body.startswith('/'):
    return False

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
