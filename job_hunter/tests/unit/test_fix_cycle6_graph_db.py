#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''RED tests for cycle-6 graph/db fixes (TDD, one per spec item).'''

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import job_hunter.core.db as db_mod
import job_hunter.llm.client as client_mod
import pytest
from job_hunter.core.config import AppSettings
from job_hunter.core.models import CandidateProfile
from job_hunter.services.scorer import DEFAULT_SALARY_FLOOR_LPA


def test_candidate_skill_ids_dedupes_and_resolves_once_per_skill(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''_candidate_skill_ids uses resolve_many, dedupes, resolves once/skill.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  import job_hunter.graph.nodes as nodes_mod
  import job_hunter.services.skills_taxonomy as taxonomy_mod

  class FakeTaxonomy:
    '''Counts resolve calls; resolve_many dedupes like the real one.'''

    def __init__(self, db_path) -> None:
      '''Accept db path.

      Args:
        db_path: Ignored database path.
      '''
      _ = db_path
      self.resolve_calls: list[str] = []
      self.resolve_many_calls: list[list[str]] = []
      self._ids = {'python': 1, 'sql': 2}

    def resolve(self, name: str):
      '''Record and return the id for a skill name.

      Args:
        name: Raw skill name.

      Returns:
        Skill id or None.
      '''
      self.resolve_calls.append(name)
      return self._ids.get(name.strip().lower())

    def resolve_many(self, names):
      '''Resolve a list to unique ids, once per entry.

      Args:
        names: Raw skill names.

      Returns:
        Unique resolved skill ids.
      '''
      self.resolve_many_calls.append(list(names))
      ids: list[int] = []
      for name in names:
        found = self.resolve(name)
        if found is not None and found not in ids:
          ids.append(found)
      return ids

  created: list[FakeTaxonomy] = []

  def make_fake(db_path) -> FakeTaxonomy:
    '''Build and record a fake taxonomy.

    Args:
      db_path: Ignored database path.

    Returns:
      Fake taxonomy instance.
    '''
    fake = FakeTaxonomy(db_path)
    created.append(fake)
    return fake

  monkeypatch.setattr(taxonomy_mod, 'SkillsTaxonomy', make_fake)
  settings = SimpleNamespace(db_path=tmp_path / 'app.db')
  candidate = CandidateProfile(
    skills=['Python', 'python', 'SQL', 'UnknownSkill'],
  )
  result = nodes_mod._candidate_skill_ids(settings, candidate)
  assert result == [1, 2]
  assert len(created) == 1
  fake = created[0]
  assert fake.resolve_many_calls == [candidate.skills]
  assert fake.resolve_calls == candidate.skills


def test_reset_client_clears_and_instance_assignment_under_lock(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''reset_client clears caches and assigns _instance inside _lock.

  Args:
    monkeypatch: Pytest monkeypatch fixture.
  '''
  monkeypatch.setattr(client_mod, '_instance', object())
  client_mod._clients['k'] = object()
  client_mod.reset_client()
  assert client_mod._instance is None
  assert client_mod._clients == {}
  source = Path(client_mod.__file__).read_text(encoding='utf-8')
  reset_src = source.split('def reset_client')[1]
  lines = reset_src.splitlines()
  lock_idx = next(
    i for i, line in enumerate(lines) if 'with _lock:' in line
  )
  assign_idx = next(
    i for i, line in enumerate(lines) if '_instance = None' in line
  )
  assert assign_idx > lock_idx
  lock_indent = len(lines[lock_idx]) - len(lines[lock_idx].lstrip())
  assign_indent = len(lines[assign_idx]) - len(lines[assign_idx].lstrip())
  assert assign_indent > lock_indent


def test_get_client_instance_assignment_under_lock() -> None:
  '''get_client assigns the _instance alias inside _lock.'''
  source = Path(client_mod.__file__).read_text(encoding='utf-8')
  get_src = source.split('def get_client')[1].split('def reset_client')[0]
  lines = get_src.splitlines()
  # Every _instance assignment in get_client must be nested under a
  # 'with _lock:' block in the same function.
  lock_indents = [
    len(line) - len(line.lstrip())
    for line in lines
    if 'with _lock:' in line
  ]
  assert lock_indents, 'get_client must use the _lock'
  for i, line in enumerate(lines):
    if '_instance = client' in line:
      indent = len(line) - len(line.lstrip())
      assert any(indent > lock_indent for lock_indent in lock_indents)


def test_config_salary_floor_fallback_uses_scorer_constant(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''Config fallback equals the scorer DEFAULT_SALARY_FLOOR_LPA.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('app:\n  host: 127.0.0.1\n', encoding='utf-8')
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  monkeypatch.delenv('SALARY_HARD_FLOOR_LPA', raising=False)
  settings = AppSettings(config_path)
  assert settings.salary_floor_lpa == DEFAULT_SALARY_FLOOR_LPA
  import job_hunter.core.config as config_mod
  source = Path(config_mod.__file__).read_text(encoding='utf-8')
  assert 'DEFAULT_SALARY_FLOOR_LPA' in source
  assert ", 45)" not in source
  assert "'salary_hard_floor_lpa', 45" not in source


def test_connect_closes_conn_on_pragma_failure(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''A PRAGMA failure closes the connection and re-raises.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  class FailingConn:
    '''Connection whose PRAGMA execute always raises OperationalError.'''

    def __init__(self) -> None:
      '''Initialize close tracking.'''
      self.closed = False
      self.row_factory = None

    def execute(self, sql, *args, **kwargs):
      '''Raise on any PRAGMA statement.

      Args:
        sql: SQL text.
        *args: Ignored positional args.
        **kwargs: Ignored keyword args.

      Raises:
        sqlite3.OperationalError: Always, on PRAGMA.
      '''
      _ = (args, kwargs)
      raise sqlite3.OperationalError(f'pragma failed: {sql}')

    def close(self) -> None:
      '''Record that close was called.'''
      self.closed = True

  fake = FailingConn()
  monkeypatch.setattr(db_mod.sqlite3, 'connect', lambda *a, **k: fake)
  with pytest.raises(sqlite3.OperationalError, match='pragma failed'):
    db_mod.connect(tmp_path / 'app.db')
  assert fake.closed is True


def test_connect_closes_conn_on_fk_pragma_failure(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''An FK PRAGMA failure (after WAL) also closes the connection.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  class FkFailingConn:
    '''WAL succeeds; foreign_keys PRAGMA raises.'''

    def __init__(self) -> None:
      '''Initialize close tracking.'''
      self.closed = False
      self.row_factory = None

    def execute(self, sql, *args, **kwargs):
      '''Fail only on the foreign_keys pragma.

      Args:
        sql: SQL text.
        *args: Ignored positional args.
        **kwargs: Ignored keyword args.

      Returns:
        Self for chaining.

      Raises:
        sqlite3.OperationalError: On foreign_keys PRAGMA.
      '''
      _ = (args, kwargs)
      if 'foreign_keys' in sql:
        raise sqlite3.OperationalError('fk pragma failed')
      return self

    def close(self) -> None:
      '''Record that close was called.'''
      self.closed = True

  fake = FkFailingConn()
  monkeypatch.setattr(db_mod.sqlite3, 'connect', lambda *a, **k: fake)
  with pytest.raises(sqlite3.OperationalError, match='fk pragma failed'):
    db_mod.connect(tmp_path / 'app.db')
  assert fake.closed is True


def test_null_provider_model_id_renamed_snake_case() -> None:
  '''The NullProvider helper uses a snake_case private name.'''
  import job_hunter.graph.nodes as nodes_mod
  from job_hunter.services.embedder import NullProvider

  assert hasattr(nodes_mod, '_null_provider_model_id')
  assert callable(nodes_mod._null_provider_model_id)
  assert nodes_mod._null_provider_model_id() == NullProvider.model_id
  assert not hasattr(nodes_mod, 'NullProvider_model_id')
  source = inspect.getsource(nodes_mod.compute_embeddings)
  assert '_null_provider_model_id()' in source
