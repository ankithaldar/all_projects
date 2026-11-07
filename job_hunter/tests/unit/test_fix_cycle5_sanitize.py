#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle 5 sanitize fix: trailing lone '<' split of </think>.'''

from __future__ import annotations

import pytest

from llm_gateway.sanitize import StreamThinkSanitizer


def _collect(chunks: list[str]) -> str:
  '''Feed every chunk then flush, returning concatenated output.

  Args:
    chunks: Raw streamed chunks.

  Returns:
    Concatenated display-safe output.
  '''
  sanitizer = StreamThinkSanitizer()
  out = ''
  for chunk in chunks:
    out += sanitizer.feed(chunk)
  out += sanitizer.flush()
  return out


def test_stream_lone_lt_splits_close_tag() -> None:
  '''Lone '<' at think chunk end buffers for split </think>.'''
  combined = _collect(['hello <think>secret g<', '/think>world'])
  assert 'secret' not in combined
  assert '/think>' not in combined
  assert 'hello' in combined
  assert 'world' in combined
  assert combined == 'helloworld'


def test_stream_lone_lt_at_final_chunk_inside_think_discarded() -> None:
  '''Trailing '<' inside think at flush emits nothing (no leak).'''
  combined = _collect(['hello <think>secret g<'])
  assert combined == 'hello'
  assert 'secret' not in combined
  assert '<' not in combined


@pytest.mark.parametrize(
  'first,second',
  [
    ('hello <think>secret</thi', 'nk>world'),
    ('hello <think>secret</th', 'ink>world'),
    ('hello <think>secret</', 'think>world'),
    ('hello <thi', 'nk>secret</think>world'),
  ],
)
def test_stream_partial_tag_splits_still_strip(
  first: str,
  second: str,
) -> None:
  '''Existing split-tag contract: partial prefixes stay hidden.'''
  combined = _collect([first, second])
  assert 'secret' not in combined
  assert 'hello' in combined
  assert 'world' in combined


def test_stream_trailing_lt_outside_think_emitted_on_flush() -> None:
  '''Trailing '<' outside think must still emit '<'.'''
  sanitizer = StreamThinkSanitizer()
  out = sanitizer.feed('a <')
  out += sanitizer.flush()
  assert out == 'a<'
