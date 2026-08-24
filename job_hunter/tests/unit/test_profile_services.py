#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for the skills taxonomy and structured-output repair loop.'''


from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel
from job_hunter.services.skills_taxonomy import SkillsTaxonomy


class FakeResponse:
  '''Mimics the gateway response surface minimally.'''

  def __init__(self, content: str) -> None:
    '''Wrap content text.

    Args:
      content: Response text.
    '''
    self.content = content
    self.tool_calls = []


class FakeGatewayClient:
  '''Returns queued contents in order; records session ids.'''

  def __init__(self, contents: list) -> None:
    '''Queue responses.

    Args:
      contents: Response strings in call order.
    '''
    self._contents = list(contents)
    self.session_ids: list = []

  async def acomplete_text(self, session_id: str, **kwargs):
    '''Pop the next canned response.

    Args:
      session_id: Correlation id.
      **kwargs: Ignored.

    Returns:
      Fake response object.
    '''
    self.session_ids.append(session_id)
    return FakeResponse(self._contents.pop(0))


def test_taxonomy_seed_and_resolve(tmp_path: Path) -> None:
  '''Aliases resolve case-insensitively and resolve_many dedupes.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  taxonomy = SkillsTaxonomy(tmp_path / 'app.db')
  from job_hunter.core.db import run_migrations
  run_migrations(tmp_path / 'app.db')
  taxonomy.load_seed({
    'skills': {
      'Python': {'category': 'language', 'aliases': ['python3']},
      'PyTorch': {'category': 'ml', 'aliases': ['torch']},
    },
  })
  assert taxonomy.resolve('PYTHON') == taxonomy.resolve('python3')
  assert taxonomy.resolve_many(['Python', 'py', 'torch', 'UnknownThing']) == [
    taxonomy.resolve('Python'),
    taxonomy.resolve('PyTorch'),
  ]


def test_structured_tolerates_prose_wrapped_json(tmp_path: Path) -> None:
  '''extract_json pulls JSON out of prose-wrapped answers.

  Args:
    tmp_path: Pytest temporary directory (unused).
  '''
  from job_hunter.llm.structured import extract_json
  value = extract_json('Here you go: {"name": "ok"} hope that helps')
  assert value == {'name': 'ok'}


def test_structured_repairs_after_validation_failure(tmp_path: Path) -> None:
  '''complete_structured replays with validator feedback until valid.

  Args:
    tmp_path: Pytest temporary directory (unused).
  '''
  from job_hunter.llm.structured import complete_structured

  class Out(BaseModel):
    name: str

  client = FakeGatewayClient([
    '{"nam": "wrong-field"}',
    json.dumps({'name': 'fixed'}),
  ])
  result = asyncio.run(complete_structured(
    client, Out, 'Extract name.', 'irrelevant', 'sess-1',
  ))
  assert result.name == 'fixed'
  assert len(client.session_ids) == 2
