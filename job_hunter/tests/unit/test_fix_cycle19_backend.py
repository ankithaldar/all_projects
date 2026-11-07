#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle 19 backend fixes (W1-backend-final).

Covers quick_poll wiring from run kind, the enrich_per_run_cap
attempt counter, and the filtered deferred_targets count.
'''

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path


def _settings(
  tmp_path: Path, monkeypatch, yaml_text: str = 'salary_hard_floor_lpa: 45\n',
):
  '''Build migrated settings for service tests.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.
    yaml_text: app.yaml payload.

  Returns:
    Migrated AppSettings.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(yaml_text, encoding='utf-8')
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
    board: Board ref, None for NULL, or 'auto' for a ref.
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


def _seed_many_ats(
  db_path: Path, prefix: str, count: int, ats: str,
  *, priority: int = 3,
) -> list[int]:
  '''Insert many fresh companies for one provider.

  Args:
    db_path: Database path.
    prefix: Name prefix.
    count: Number of companies.
    ats: ATS provider key.
    priority: Priority value.

  Returns:
    New company ids in insertion order.
  '''
  return [
    _seed_company_raw(
      db_path, f'{prefix}{index:03d}', '2026-01-01 00:00:00',
      ats=ats, priority=priority,
    )
    for index in range(count)
  ]


# 1. quick_poll wiring from run kind ----------------------------------------


def test_refresh_kind_passes_quick_poll_true(
  tmp_path: Path, monkeypatch,
) -> None:
  '''enqueue(kind='refresh') executes with quick_poll=True.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.
  '''
  settings = _settings(tmp_path, monkeypatch)
  pinned = _seed_company_raw(
    settings.db_path, 'PinnedCo', '2026-01-01 00:00:00', priority=5,
  )
  _seed_company_raw(
    settings.db_path, 'PlainCo', '2026-01-01 00:00:00', priority=3,
  )
  captured: dict = {}

  async def fake_discovery(settings_arg, run_id, **kwargs):
    '''Capture discovery kwargs without running the graph.

    Args:
      settings_arg: Settings.
      run_id: Run id.
      **kwargs: Captured kwargs.

    Returns:
      Minimal final state.
    '''
    captured.update(kwargs)
    return {'stats': {}, 'errors': []}

  import job_hunter.graph.discovery_graph as dg
  monkeypatch.setattr(dg, 'run_discovery', fake_discovery)

  from job_hunter.services.run_manager import RunManager

  manager = RunManager(settings)
  run_id = manager.enqueue('refresh', 'test')
  _ = run_id
  result = asyncio.run(manager.execute(None))
  assert result['status'] == 'success', result
  assert captured.get('quick_poll') is True, captured

  # Pinned-only plan: the captured flag must keep priority-5 only.
  import job_hunter.graph.nodes as nodes_mod

  config = {'configurable': {'settings': settings}}
  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': captured['quick_poll']}},
      config,
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  planned = [int(t['company_id']) for t in update['plan']['targets']]
  assert planned == [pinned], planned


def test_discovery_kind_passes_quick_poll_false(
  tmp_path: Path, monkeypatch,
) -> None:
  '''enqueue(kind='discovery') executes with quick_poll=False.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.
  '''
  settings = _settings(tmp_path, monkeypatch)
  captured: dict = {}

  async def fake_discovery(settings_arg, run_id, **kwargs):
    '''Capture discovery kwargs without running the graph.

    Args:
      settings_arg: Settings.
      run_id: Run id.
      **kwargs: Captured kwargs.

    Returns:
      Minimal final state.
    '''
    captured.update(kwargs)
    return {'stats': {}, 'errors': []}

  import job_hunter.graph.discovery_graph as dg
  monkeypatch.setattr(dg, 'run_discovery', fake_discovery)

  from job_hunter.services.run_manager import RunManager

  manager = RunManager(settings)
  manager.enqueue('discovery', 'test')
  result = asyncio.run(manager.execute(None))
  assert result['status'] == 'success', result
  assert captured.get('quick_poll') is False, captured


# 2. enrich_per_run_cap fires -----------------------------------------------


def test_enrich_cap_limits_analyst_calls(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Cap=2 with 4 inserted: analyst called twice, capped==2.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.
  '''
  settings = _settings(
    tmp_path, monkeypatch,
    'salary_hard_floor_lpa: 45\n'
    'discovery:\n'
    '  enrich_per_run_cap: 2\n'
    '  max_llm_calls_per_run: 300\n',
  )
  from job_hunter.core.models import (
    EnrichedJob,
    NormalizedJob,
    RawJobRecord,
  )
  from job_hunter.db.repositories.jobs import JobsRepository
  from job_hunter.graph.nodes import enrich_jds
  from job_hunter.graph.state import initial_state

  jobs_repo = JobsRepository(settings.db_path)
  description = 'python machine learning data pipelines ' * 15
  assert len(description) > 200
  inserted = []
  for index in range(4):
    job_id = jobs_repo.insert({
      'source_key': 'lever',
      'url': f'https://jobs.example.com/c19/{index}',
      'canonical_url': f'https://jobs.example.com/c19/{index}',
      'company_raw_name': 'ExampleCo',
      'title': 'Data Scientist',
      'description_text': description,
      'raw_json': '{}',
      'content_hash': f'c19hash{index}',
    })
    inserted.append(NormalizedJob(
      job_id=job_id,
      raw=RawJobRecord(
        source_key='lever',
        url=f'https://jobs.example.com/c19/{index}',
        title='Data Scientist',
      ),
      canonical_url=f'https://jobs.example.com/c19/{index}',
      content_hash=f'c19hash{index}',
    ))

  calls: list = []

  class FakeAnalyst:
    '''Fake JDAnalyst counting extract calls.'''

    def __init__(self, client) -> None:
      '''Accept client.

      Args:
        client: Fake gateway client.
      '''
      self._client = client

    async def extract(self, title, description_text, session_id):
      '''Return a fixed extraction.

      Args:
        title: Posting title.
        description_text: Description text.
        session_id: Correlation id.

      Returns:
        Fixed enrichment.
      '''
      calls.append(title)
      return EnrichedJob(
        must_have_skills=['Python'],
        salary_max_lpa=60.0,
        confidence=0.9,
      )

  monkeypatch.setattr(
    'job_hunter.llm.client.get_client', lambda settings: object(),
  )
  monkeypatch.setattr(
    'job_hunter.services.jd_analyst.JDAnalyst', FakeAnalyst,
  )

  state = initial_state(run_id=1, triggered_by='test')
  state['inserted'] = inserted
  update = asyncio.run(enrich_jds(
    state, {'configurable': {'settings': settings}},
  ))
  assert 'errors' not in update or not update['errors'], update
  assert len(calls) == 2, calls
  assert update['stats'].get('enrich_capped') == 2, update['stats']
  assert update['stats'].get('enriched', 0) == 2, update['stats']


# 3. deferred from filtered eligible rows ------------------------------------


def test_deferred_disabled_source_filtered(
  tmp_path: Path, monkeypatch,
) -> None:
  '''60 due with 10 disabled-source: deferred is 0, not 10.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.
  '''
  settings = _settings(
    tmp_path, monkeypatch,
    'salary_hard_floor_lpa: 45\n'
    'sources:\n'
    '  lever: {enabled: true, rate_limit_rpm: 30}\n'
    '  greenhouse: {enabled: false, rate_limit_rpm: 30}\n',
  )
  _seed_many_ats(settings.db_path, 'Lever', 50, 'lever')
  _seed_many_ats(settings.db_path, 'Gh', 10, 'greenhouse')
  import job_hunter.graph.nodes as nodes_mod

  config = {'configurable': {'settings': settings}}
  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': False}}, config,
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  assert update['stats'].get('deferred_targets') == 0, update['stats']
  assert len(update['plan']['targets']) == 50, update['plan']


def test_deferred_quick_priority_filtered(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Quick poll: 60 due, 50 pinned: deferred is 0, not 10.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.
  '''
  settings = _settings(tmp_path, monkeypatch)
  _seed_many_ats(settings.db_path, 'Pinned', 50, 'lever', priority=5)
  _seed_many_ats(settings.db_path, 'Plain', 10, 'lever', priority=3)
  import job_hunter.graph.nodes as nodes_mod

  config = {'configurable': {'settings': settings}}
  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': True}}, config,
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  assert update['stats'].get('deferred_targets') == 0, update['stats']
  assert len(update['plan']['targets']) == 50, update['plan']


def test_deferred_filtered_excess_counts_filtered_only(
  tmp_path: Path, monkeypatch,
) -> None:
  '''70 eligible + 10 disabled: deferred is 20, not 30.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.
  '''
  settings = _settings(
    tmp_path, monkeypatch,
    'salary_hard_floor_lpa: 45\n'
    'sources:\n'
    '  lever: {enabled: true, rate_limit_rpm: 30}\n'
    '  greenhouse: {enabled: false, rate_limit_rpm: 30}\n',
  )
  _seed_many_ats(settings.db_path, 'Lever', 70, 'lever')
  _seed_many_ats(settings.db_path, 'Gh', 10, 'greenhouse')
  import job_hunter.graph.nodes as nodes_mod

  config = {'configurable': {'settings': settings}}
  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': False}}, config,
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  assert update['stats'].get('deferred_targets') == 20, update['stats']
  assert len(update['plan']['targets']) == 70, update['plan']
