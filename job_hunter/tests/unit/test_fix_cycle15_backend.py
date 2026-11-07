#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle 15 backend fixes (W1-backend-final).

Covers cooled-exclusion in list_due SQL before LIMIT, empty-string
ATS/board exclusion in list_due and quick_poll, true deferred count
via count_due, and doc-only fixes for mark_checked, bootstrap, and
constants.
'''

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path


def _settings(tmp_path: Path, monkeypatch):
  '''Build migrated settings for service tests.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.

  Returns:
    Migrated AppSettings.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations
  settings = AppSettings(config_path)
  settings.data_dir.mkdir(parents=True, exist_ok=True)
  run_migrations(settings.db_path)
  return settings


def _seed_company_raw(
  db_path: Path, name: str, last_checked: str | None,
  *, ats: str | None = 'lever', board: object = 'auto',
  priority: int = 3, status: str = 'active',
) -> int:
  '''Insert one company row directly.

  Args:
    db_path: Database path.
    name: Company name.
    last_checked: last_checked_at value or None.
    ats: ATS provider or None.
    board: Board ref, None for NULL, or 'auto' for a generated ref.
    priority: Priority value.
    status: Lifecycle status.

  Returns:
    New company id.
  '''
  if board == 'auto':
    board = f'board-{name.lower()}'
  with closing(sqlite3.connect(str(db_path))) as conn:
    cursor = conn.execute(
      'INSERT INTO companies (name, normalized_name, domain, '
      'ats_provider, board_ref, priority, status, last_checked_at) '
      'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
      (
        name, name.lower(), f'{name.lower()}.example.com',
        ats, board, priority, status, last_checked,
      ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def _cool_company(db_path: Path, company_id: int, hours: int = 48) -> None:
  '''Mark one company scope as cooled down.

  Args:
    db_path: Database path.
    company_id: Company id.
    hours: Cooldown length in hours.
  '''
  with closing(sqlite3.connect(str(db_path))) as conn:
    conn.execute(
      'INSERT INTO crawl_state (scope, cooldown_until) '
      'VALUES (?, datetime(\'now\', ?)) '
      'ON CONFLICT(scope) DO UPDATE SET cooldown_until = excluded.cooldown_until',
      (f'company:{company_id}', f'+{hours} hours'),
    )
    conn.commit()


def _expire_company(db_path: Path, company_id: int) -> None:
  '''Mark one company scope with an already-expired cooldown.

  Args:
    db_path: Database path.
    company_id: Company id.
  '''
  with closing(sqlite3.connect(str(db_path))) as conn:
    conn.execute(
      'INSERT INTO crawl_state (scope, cooldown_until) '
      'VALUES (?, ?) '
      'ON CONFLICT(scope) DO UPDATE SET cooldown_until = excluded.cooldown_until',
      (f'company:{company_id}', '2020-01-01 00:00:00'),
    )
    conn.commit()


def _seed_many(db_path: Path, prefix: str, count: int) -> list[int]:
  '''Insert many fresh companies in one transaction.

  Args:
    db_path: Database path.
    prefix: Name prefix.
    count: Number of companies.

  Returns:
    New company ids in insertion order.
  '''
  ids: list[int] = []
  with closing(sqlite3.connect(str(db_path))) as conn:
    for index in range(count):
      name = f'{prefix}{index:03d}'
      cursor = conn.execute(
        'INSERT INTO companies (name, normalized_name, domain, '
        'ats_provider, board_ref, priority, status, last_checked_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (
          name, name.lower(), f'{name.lower()}.example.com',
          'lever', f'board-{name.lower()}', 3, 'active',
          '2026-01-01 00:00:00',
        ),
      )
      ids.append(int(cursor.lastrowid))
    conn.commit()
  return ids


# 1. Cooled exclusion in SQL before LIMIT -----------------------------------


def test_list_due_excludes_cooled_before_limit(
  tmp_path: Path, monkeypatch,
) -> None:
  '''50 cooled oldest + 50 fresh: list_due(50) returns fresh batch only.'''
  settings = _settings(tmp_path, monkeypatch)
  cooled_ids = [
    _seed_company_raw(
      settings.db_path, f'Cooled{i:02d}', '2026-01-01 00:00:00',
    )
    for i in range(50)
  ]
  fresh_ids = [
    _seed_company_raw(
      settings.db_path, f'Fresh{i:02d}', '2026-02-01 00:00:00',
    )
    for i in range(50)
  ]
  for company_id in cooled_ids:
    _cool_company(settings.db_path, company_id)
  from job_hunter.db.repositories.companies import CompaniesRepository
  rows = CompaniesRepository(settings.db_path).list_due(50)
  assert [row['id'] for row in rows] == fresh_ids


def test_list_due_includes_expired_cooldown(
  tmp_path: Path, monkeypatch,
) -> None:
  '''A company whose cooldown_until is in the past is due again.'''
  settings = _settings(tmp_path, monkeypatch)
  company_id = _seed_company_raw(
    settings.db_path, 'ExpiredCo', '2026-01-01 00:00:00',
  )
  _expire_company(settings.db_path, company_id)
  from job_hunter.db.repositories.companies import CompaniesRepository
  rows = CompaniesRepository(settings.db_path).list_due(10)
  assert [row['id'] for row in rows] == [company_id]


# 2. Empty-string ATS/board -------------------------------------------------


def test_list_due_excludes_empty_string_ats_and_board(
  tmp_path: Path, monkeypatch,
) -> None:
  '''list_due excludes ats_provider='' and board_ref='' rows.'''
  settings = _settings(tmp_path, monkeypatch)
  _seed_company_raw(
    settings.db_path, 'EmptyAts', None, ats='', board='b1',
  )
  _seed_company_raw(
    settings.db_path, 'EmptyBoard', None, ats='lever', board='',
  )
  good = _seed_company_raw(
    settings.db_path, 'HasBoth', None, ats='lever', board='b1',
  )
  from job_hunter.db.repositories.companies import CompaniesRepository
  rows = CompaniesRepository(settings.db_path).list_due(10)
  assert [row['id'] for row in rows] == [good]


def test_quick_poll_excludes_empty_string_ats(
  tmp_path: Path, monkeypatch,
) -> None:
  '''quick_poll plan excludes a priority-5 company with ats_provider=''.'''
  settings = _settings(tmp_path, monkeypatch)
  _seed_company_raw(
    settings.db_path, 'EmptyQuick', None, ats='', board='b1',
    priority=5,
  )
  good = _seed_company_raw(
    settings.db_path, 'GoodQuick', None, ats='lever', board='b1',
    priority=5,
  )
  import job_hunter.graph.nodes as nodes_mod
  config = {'configurable': {'settings': settings}}
  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': True}}, config,
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  planned = [int(t['company_id']) for t in update['plan']['targets']]
  assert planned == [good], planned


# 3. True deferred count via count_due ---------------------------------------


def test_count_due_shares_list_due_filters(
  tmp_path: Path, monkeypatch,
) -> None:
  '''count_due counts only active, ATS-bearing, non-cooled companies.'''
  settings = _settings(tmp_path, monkeypatch)
  fresh_ids = [
    _seed_company_raw(
      settings.db_path, f'Fresh{i:02d}', '2026-01-01 00:00:00',
    )
    for i in range(10)
  ]
  cooled_ids = [
    _seed_company_raw(
      settings.db_path, f'Cooled{i:02d}', '2026-01-01 00:00:00',
    )
    for i in range(5)
  ]
  for company_id in cooled_ids:
    _cool_company(settings.db_path, company_id)
  _seed_company_raw(
    settings.db_path, 'EmptyAts', None, ats='', board='b1',
  )
  _seed_company_raw(
    settings.db_path, 'Paused', None, status='paused',
  )
  from job_hunter.db.repositories.companies import CompaniesRepository
  repo = CompaniesRepository(settings.db_path)
  assert repo.count_due() == 10, fresh_ids
  assert len(repo.list_due(500)) == 10


def test_build_plan_deferred_uses_untruncated_count(
  tmp_path: Path, monkeypatch,
) -> None:
  '''600 fresh companies: deferred is 550 (600 - MAX_FANOUT 50).'''
  settings = _settings(tmp_path, monkeypatch)
  _seed_many(settings.db_path, 'Bulk', 600)
  import job_hunter.graph.nodes as nodes_mod
  config = {'configurable': {'settings': settings}}
  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': False}}, config,
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  assert update['stats'].get('deferred_targets') == 550, update['stats']


def test_build_plan_counts_cooled_skips(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Preservation: 50 cooled + 50 fresh plans fresh only, skips=50.'''
  settings = _settings(tmp_path, monkeypatch)
  cooled_ids = [
    _seed_company_raw(
      settings.db_path, f'Cooled{i:02d}', '2026-01-01 00:00:00',
    )
    for i in range(50)
  ]
  fresh_ids = [
    _seed_company_raw(
      settings.db_path, f'Fresh{i:02d}', '2026-02-01 00:00:00',
    )
    for i in range(50)
  ]
  for company_id in cooled_ids:
    _cool_company(settings.db_path, company_id)
  import job_hunter.graph.nodes as nodes_mod
  config = {'configurable': {'settings': settings}}
  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': False}}, config,
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  planned = [int(t['company_id']) for t in update['plan']['targets']]
  assert set(planned) == set(fresh_ids), planned[:10]
  assert update['stats'].get('cooled_skips') == 50, update['stats']
  assert update['stats'].get('deferred_targets') == 0, update['stats']


# 4. mark_checked semantics doc ----------------------------------------------


def test_mark_checked_docstring_rotation_stamp() -> None:
  '''mark_checked documents rotation stamp, not last successful fetch.'''
  from job_hunter.db.repositories.companies import CompaniesRepository
  doc = CompaniesRepository.mark_checked.__doc__ or ''
  assert 'rotation stamp (attempted), not last successful fetch' in doc, doc


# 5. Bootstrap exception doc ---------------------------------------------------


def test_bootstrap_documents_composition_root_exception() -> None:
  '''bootstrap comments the intentional core->services exception.'''
  root = (
    Path(__file__).resolve().parents[2]
    / 'src/job_hunter/job_hunter/core/bootstrap.py'
  )
  text = root.read_text(encoding='utf-8')
  phrase = (
    'composition root \u2014 intentional acyclic exception to '
    'core->services inversion (taxonomy imports only core.db)'
  )
  assert phrase in text, text[:2000]
  comment_at = text.index(phrase)
  import_at = text.index(
    'from job_hunter.services.skills_taxonomy import SkillsTaxonomy',
  )
  assert comment_at < import_at


# 6. Constants docstring softened ----------------------------------------------


def test_constants_softens_never_from_services() -> None:
  '''constants documents the bootstrap composition-root exception.'''
  import job_hunter.core.constants as const
  doc = const.__doc__ or ''
  assert (
    'except core/bootstrap.py composition root (documented)' in doc
  ), doc
  assert 'never from services' not in doc, doc
