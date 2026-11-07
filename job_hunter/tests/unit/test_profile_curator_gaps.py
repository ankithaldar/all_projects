#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for ProfileCurator extract and fallback helpers.'''


from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from job_hunter.core.config import AppSettings
from job_hunter.core.errors import StructuredOutputError
from job_hunter.services.profile_curator import ProfileCurator, json_loads


class _FakeResponse:
  '''Mimics the gateway response surface minimally.'''

  def __init__(self, content: str) -> None:
    '''Wrap content text.

    Args:
      content: Response text.
    '''
    self.content = content
    self.tool_calls = []


class _FakeClient:
  '''Fake gateway client returning canned contents in order.'''

  def __init__(self, contents: list) -> None:
    '''Queue canned responses.

    Args:
      contents: Response strings in call order.
    '''
    self._contents = list(contents)
    self.calls: list = []

  async def acomplete_text(self, session_id: str, **kwargs):
    '''Record the call and return the next canned response.

    Args:
      session_id: Correlation id.
      **kwargs: Forwarded gateway options.

    Returns:
      Fake response object.
    '''
    self.calls.append({'session_id': session_id, **kwargs})
    return _FakeResponse(self._contents.pop(0))


def _settings(tmp_path: Path, monkeypatch) -> AppSettings:
  '''Build settings backed by a tmp data directory.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest fixture.

  Returns:
    Application settings.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  return AppSettings(config_path)


def test_extract_parse_success(tmp_path: Path, monkeypatch) -> None:
  '''Extract parses gateway JSON into a validated extraction.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest fixture.
  '''
  payload = json.dumps({
    'skills': ['Python', 'PyTorch'],
    'summary': 'Staff data scientist.',
    'current_title': 'Staff Data Scientist',
    'experience_years': 7.5,
    'seniority_keywords': ['staff'],
  })
  client = _FakeClient([payload])
  curator = ProfileCurator(_settings(tmp_path, monkeypatch), client=client)
  result = asyncio.run(curator.extract('resume text about python'))
  assert result.skills == ['Python', 'PyTorch']
  assert result.current_title == 'Staff Data Scientist'
  assert result.experience_years == 7.5
  assert result.seniority_keywords == ['staff']
  assert len(client.calls) == 1


def test_extract_llm_failure_after_repairs(
  tmp_path: Path, monkeypatch
) -> None:
  '''Garbage gateway output raises after the repair attempts.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest fixture.
  '''
  client = _FakeClient(['not json at all'] * 3)
  curator = ProfileCurator(_settings(tmp_path, monkeypatch), client=client)
  with pytest.raises(StructuredOutputError):
    asyncio.run(curator.extract('resume text'))
  # One initial attempt plus the two structured repair attempts.
  assert len(client.calls) == 3


def test_json_loads_fallbacks() -> None:
  '''JSON column helper falls back to empty list on bad values.'''
  assert json_loads(['a']) == ['a']
  assert json_loads('["a", "b"]') == ['a', 'b']
  assert json_loads('not json') == []
  assert json_loads('{"a": 1}') == []
  assert json_loads(None) == []
  assert json_loads('') == []
