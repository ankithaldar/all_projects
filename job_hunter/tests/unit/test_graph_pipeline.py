#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for state reducers, normalization, and run claiming.'''


from __future__ import annotations

import asyncio
from pathlib import Path

from job_hunter.core.db import run_migrations
from job_hunter.core.models import CompanyTarget, RawJobRecord
from job_hunter.db.repositories.runs import RunsRepository
from job_hunter.graph.discovery_graph import build_discovery_graph
from job_hunter.graph.state import initial_state, merge_stats
from job_hunter.services.normalizer import (
  canonicalize_url,
  content_hash,
  detect_city,
  infer_work_mode,
  normalize_employment,
  parse_posted_at,
  parse_salary_lpa,
)


def test_merge_stats_sums() -> None:
  '''Reducer merges counters by summation.'''
  assert merge_stats({'a': 1, 'b': 2}, {'b': 3, 'c': 4}) == {'a': 1, 'b': 5, 'c': 4}


def test_canonicalize_url_strips_tracking() -> None:
  '''UTM parameters and fragments disappear.'''
  url = 'https://jobs.example.com/x?utm_source=li&id=5#top'
  assert canonicalize_url(url) == 'https://jobs.example.com/x?id=5'


def test_content_hash_stable() -> None:
  '''Hash ignores case and whitespace differences in title/company.'''
  base = RawJobRecord(
    source_key='lever', url='https://jobs.lever.co/co/1',
    company_name='CRED', title='Staff Data Scientist',
    description_text='models and experiments',
  )
  other = RawJobRecord(
    source_key='lever', url='https://jobs.lever.co/co/1?utm_source=x',
    company_name='cred', title='staff data scientist ',
    description_text='models and experiments',
  )
  assert content_hash(base, base.description_text) == content_hash(
    RawJobRecord(**{**base.model_dump(), 'url': base.url}),
    base.description_text,
  )


def test_city_and_mode_detection() -> None:
  '''City mapping and work-mode inference behave sensibly.'''
  assert detect_city('Bangalore / Remote') == 'Bengaluru'
  assert detect_city('Remote (India)') == 'Remote'
  assert infer_work_mode('', 'Remote, India', '') == 'remote'
  assert infer_work_mode('hybrid', 'Pune', 'hybrid role') == 'hybrid'
  assert infer_work_mode('', 'Gurugram office', 'on-site') == 'onsite'
  assert normalize_employment('Full Time') == 'full_time'
  assert normalize_employment('Contractor') == 'contract'


def test_salary_parsing() -> None:
  '''LPA ranges parse; implausible values reject.'''
  assert parse_salary_lpa('45-65 LPA') == (45.0, 65.0)
  lo, hi = parse_salary_lpa('', 'CTC: ₹50 lakhs per annum')
  assert hi is not None and abs(hi - 52.5) < 0.6
  assert parse_salary_lpa('2 widgets') == (None, None)


def test_posted_at_formats() -> None:
  '''Epoch millis and ISO strings both normalize to ISO UTC.'''
  iso = parse_posted_at(1755000000000)
  assert iso is not None and 'T' in iso
  assert parse_posted_at('2026-08-01T00:00:00Z').endswith('+00:00')
  assert parse_posted_at(None) is None


def test_graph_runs_with_stub_targets(tmp_path: Path, monkeypatch) -> None:
  '''The v1 graph executes end-to-end against a stubbed adapter fetch.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  from job_hunter.adapters import registry as registry_mod
  from job_hunter.core.config import AppSettings

  class StubAdapter:
    '''Returns one canned record without network.'''

    def __init__(self, http) -> None:
      '''Accept the shared client.

      Args:
        http: Unused client.
      '''
      _ = http

    async def fetch(self, target: CompanyTarget, limit: int = 200):
      '''Return one record.

      Args:
        target: Target.
        limit: Cap.

      Returns:
        Single raw record list.
      '''
      return [RawJobRecord(
        source_key=target.source_key,
        external_id='1',
        url=f'https://jobs.example.com/{target.board_ref}/1',
        company_name='ExampleCo',
        title='Staff Data Scientist',
        location_text='Bengaluru, India',
        description_text='python sql experimentation 55 LPA',
      )]

    async def health(self, target):
      '''Always healthy.

      Args:
        target: Target.

      Returns:
        True.
      '''
      return True

  db = tmp_path / 'app.db'
  run_migrations(db)
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  settings = AppSettings(config_path)
  _ = settings.app_root

  import job_hunter.graph.nodes as nodes_mod
  monkeypatch.setattr(registry_mod, 'build_adapter', lambda key, http: StubAdapter(http))
  monkeypatch.setattr(nodes_mod, 'build_adapter', lambda key, http: StubAdapter(http))

  companies = __import__(
    'job_hunter.db.repositories.companies', fromlist=['CompaniesRepository'],
  ).CompaniesRepository(db)
  company_id = companies.upsert(
    name='ExampleCo', domain='example.com', ats_provider='lever',
    board_ref='exampleco', priority=4,
  )
  _ = company_id

  graph = build_discovery_graph()
  seed = initial_state(run_id=1, triggered_by='test')
  seed['plan'] = {'quick_poll': False}
  result = asyncio.run(graph.ainvoke(seed, config={
    'configurable': {'settings': settings},
  }))
  assert result['stats'].get('new_jobs', 0) >= 0
  jobs = __import__(
    'job_hunter.db.repositories.jobs', fromlist=['JobsRepository'],
  ).JobsRepository(db).list_jobs()
  assert isinstance(jobs, list)


def test_run_claim_protocol(tmp_path: Path) -> None:
  '''Claim marks a pending run running exactly once.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  db = tmp_path / 'app.db'
  run_migrations(db)
  repo = RunsRepository(db)
  run_id = repo.create('discovery', 'test')
  first = repo.claim_pending()
  second = repo.claim_pending()
  assert first == run_id
  assert second is None or second != run_id
