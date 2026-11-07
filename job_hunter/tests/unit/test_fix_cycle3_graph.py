#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''RED tests for F3-graph-arch (cycle-3 anti-arch remainder).

Covers: salary_fit zero/negative floor, sources.enabled filtering in
build_plan, shared HttpClient identity in fetch_pair, dict-candidate
normalization in the score path, safe_node CancelledError passthrough,
node_var setting, get_client keying by gateway paths, load_embeddings
job_ids scoping, single-sourced defaults, and vertical_classifier
debug logging instead of a silent pass.
'''

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from job_hunter.services.matcher import salary_fit


def test_salary_fit_zero_floor_neutral() -> None:
  '''Zero floor is neutral 1.0 instead of ZeroDivisionError.'''
  assert salary_fit(10.0, 20.0, 0.0) == 1.0


def test_salary_fit_negative_floor_neutral() -> None:
  '''Negative floor is neutral 1.0 instead of a negative score.'''
  assert salary_fit(10.0, 20.0, -5.0) == 1.0


def test_salary_fit_clamped_to_unit_interval() -> None:
  '''Salary fit never escapes [0, 1].'''
  assert 0.0 <= salary_fit(200.0, 200.0, 45.0) <= 1.0
  assert salary_fit(-50.0, -50.0, 45.0) == 0.0


def _settings(tmp_path: Path, monkeypatch, yaml_text: str):
  '''Build AppSettings rooted at tmp_path with a migrated database.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest monkeypatch fixture.
    yaml_text: app.yaml payload.

  Returns:
    AppSettings instance.
  '''
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations

  config_path = tmp_path / 'app.yaml'
  config_path.write_text(yaml_text, encoding='utf-8')
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  settings = AppSettings(config_path)
  run_migrations(settings.db_path)
  return settings


def test_build_plan_skips_disabled_source(tmp_path, monkeypatch) -> None:
  '''build_plan excludes targets whose source has enabled: false.'''
  from job_hunter.db.repositories.companies import CompaniesRepository
  from job_hunter.graph.nodes import build_plan

  settings = _settings(
    tmp_path, monkeypatch,
    'salary_hard_floor_lpa: 45\n'
    'sources:\n'
    '  lever: {enabled: false, rate_limit_rpm: 30}\n'
    '  greenhouse: {enabled: true, rate_limit_rpm: 30}\n',
  )
  companies = CompaniesRepository(settings.db_path)
  companies.upsert(
    name='LeverCo', domain='lever.example', ats_provider='lever',
    board_ref='leverco', priority=4,
  )
  companies.upsert(
    name='GhCo', domain='gh.example', ats_provider='greenhouse',
    board_ref='ghco', priority=4,
  )
  result = asyncio.run(build_plan(
    {'run_id': 1, 'plan': {'quick_poll': False}},
    {'configurable': {'settings': settings}},
  ))
  assert 'errors' not in result, result.get('errors')
  keys = [t['source_key'] for t in result['plan']['targets']]
  assert 'lever' not in keys
  assert 'greenhouse' in keys


def test_fetch_pair_shares_http_client(tmp_path, monkeypatch) -> None:
  '''Two fetch_pair calls receive the same shared HttpClient instance.'''
  import job_hunter.graph.nodes as nodes_mod

  settings = _settings(
    tmp_path, monkeypatch, 'salary_hard_floor_lpa: 45\n',
  )
  sentinel = SimpleNamespace(close_calls=0)

  async def _close() -> None:
    sentinel.close_calls += 1

  sentinel.close = _close
  getter_calls: list = []
  received: list = []

  def fake_getter():
    getter_calls.append(1)
    return sentinel

  class StubAdapter:
    '''Returns no records.'''

    def __init__(self, http) -> None:
      '''Capture the client.'''
      received.append(http)

    async def fetch(self, target, limit=200):
      '''Return no records.'''
      return []

  monkeypatch.setattr(nodes_mod, 'get_shared_http_client', fake_getter)
  monkeypatch.setattr(nodes_mod, 'build_adapter', lambda key, http: (
    StubAdapter(http)
  ))
  config = {'configurable': {'settings': settings}}
  first = asyncio.run(nodes_mod.fetch_pair(
    {'company_id': 41, 'name': 'A', 'source_key': 'lever',
     'board_ref': 'a'},
    config,
  ))
  second = asyncio.run(nodes_mod.fetch_pair(
    {'company_id': 42, 'name': 'B', 'source_key': 'lever',
     'board_ref': 'b'},
    config,
  ))
  assert not first['errors'] and not second['errors']
  assert len(getter_calls) == 2
  assert received == [sentinel, sentinel]
  assert sentinel.close_calls == 0


def test_score_path_accepts_dict_candidate(tmp_path, monkeypatch) -> None:
  '''score_rank_persist normalizes a dict-shaped candidate.'''
  from job_hunter.core.models import CandidateProfile
  from job_hunter.graph.nodes import score_rank_persist

  settings = _settings(
    tmp_path, monkeypatch, 'salary_hard_floor_lpa: 45\n',
  )
  candidate = CandidateProfile(
    skills=[], target_roles=['Data Scientist'], salary_floor_lpa=45.0,
    experience_years=9.0, employment_types=['full_time'],
  ).model_dump()
  result = asyncio.run(score_rank_persist(
    {'run_id': 1, 'candidate': candidate, 'inserted': [],
     'enriched': {}},
    {'configurable': {'settings': settings}},
  ))
  assert 'errors' not in result, result.get('errors')
  assert result['scored'] == []


def test_load_embeddings_scoped_to_scoring_targets(
  tmp_path, monkeypatch,
) -> None:
  '''load_embeddings receives the scoring-target job ids.'''
  from job_hunter.core.models import (
    CandidateProfile,
    NormalizedJob,
    RawJobRecord,
  )
  from job_hunter.db.repositories.jobs import JobsRepository
  from job_hunter.graph.nodes import score_rank_persist

  settings = _settings(
    tmp_path, monkeypatch, 'salary_hard_floor_lpa: 45\n',
  )
  calls: list = []

  def fake_load(self, model, job_ids=None):
    calls.append({'model': model, 'job_ids': job_ids})
    return {}

  monkeypatch.setattr(JobsRepository, 'load_embeddings', fake_load)
  from job_hunter.services.persist import persist_record

  repo = JobsRepository(settings.db_path)
  assert asyncio.run(persist_record(
    jobs_repo=repo,
    record={
      'source_key': 'lever',
      'url': 'https://j.example/11',
      'company_name': 'X',
      'title': 'Data Scientist',
      'description_text': 'python sql data science role 55 LPA',
    },
  )) == 'inserted'
  real_id = int(repo.list_jobs()[0]['id'])
  from job_hunter.db.repositories.runs import RunsRepository

  run_id = RunsRepository(settings.db_path).create('discovery', 'test')
  job = NormalizedJob(
    job_id=real_id,
    raw=RawJobRecord(
      source_key='lever', url='https://j.example/11', title='Data Scientist',
    ),
    canonical_url='', content_hash='hash-11',
  )
  candidate = CandidateProfile(
    skills=[], target_roles=['Data Scientist'], salary_floor_lpa=45.0,
    experience_years=9.0, employment_types=['full_time'],
  )
  result = asyncio.run(score_rank_persist(
    {'run_id': run_id, 'candidate': candidate, 'inserted': [job],
     'enriched': {}},
    {'configurable': {'settings': settings}},
  ))
  assert 'errors' not in result, result.get('errors')
  assert calls and calls[0]['job_ids'] == [real_id]
  assert calls[0]['model'] == 'BAAI/bge-small-en-v1.5'


def test_safe_node_propagates_cancelled_error() -> None:
  '''A wrapped node raising CancelledError propagates, unconverted.'''
  from job_hunter.graph.nodes import safe_node

  @safe_node
  async def cancelling_node(state, config):
    '''Raise CancelledError.'''
    raise asyncio.CancelledError()

  try:
    asyncio.run(cancelling_node({}, {}))
  except asyncio.CancelledError:
    pass
  else:  # pragma: no cover
    raise AssertionError('CancelledError was converted to NodeError')


def test_safe_node_propagates_keyboard_interrupt() -> None:
  '''A wrapped sync node raising KeyboardInterrupt propagates.'''
  from job_hunter.graph.nodes import safe_node

  @safe_node
  def interrupting_node(state, config):
    '''Raise KeyboardInterrupt.'''
    raise KeyboardInterrupt()

  try:
    interrupting_node({}, {})
  except KeyboardInterrupt:
    pass
  else:  # pragma: no cover
    raise AssertionError('KeyboardInterrupt was swallowed')


def test_safe_node_sets_node_var() -> None:
  '''node_var carries the node name inside the wrapped node.'''
  from job_hunter.core.logging import node_var
  from job_hunter.graph.nodes import safe_node

  seen: list = []

  @safe_node
  def probing_node(state, config):
    '''Record the ambient node name.'''
    seen.append(node_var.get())
    return {'ok': True}

  result = probing_node({}, {})
  assert result == {'ok': True}
  assert seen == ['probing_node']
  assert node_var.get() is None


def test_get_client_keyed_by_gateway_paths(tmp_path, monkeypatch) -> None:
  '''Different gateway paths yield different clients; same yield same.'''
  import job_hunter.llm.client as client_mod

  created: list = []

  class FakeClient:
    '''Stand-in that records construction.'''

    def __init__(self, settings) -> None:
      '''Record settings.'''
      created.append(settings)
      self.settings = settings

  monkeypatch.setattr(client_mod, 'GatewayClient', FakeClient)
  client_mod.reset_client()
  try:
    settings_a = SimpleNamespace(
      gateway_db_path=Path(str(tmp_path / 'a' / 'gateway.db')),
      gateway_cache_path=Path(str(tmp_path / 'a' / 'cache.db')),
    )
    settings_b = SimpleNamespace(
      gateway_db_path=Path(str(tmp_path / 'b' / 'gateway.db')),
      gateway_cache_path=Path(str(tmp_path / 'b' / 'cache.db')),
    )
    first = client_mod.get_client(settings_a)
    again = client_mod.get_client(settings_a)
    other = client_mod.get_client(settings_b)
    assert first is again
    assert other is not first
    assert len(created) == 2
  finally:
    client_mod.reset_client()


def test_defaults_single_sourced_in_scorer() -> None:
  '''DEFAULT_* constants live in scorer with the historical values.'''
  from job_hunter.services import scorer

  assert scorer.DEFAULT_SALARY_FLOOR_LPA == 45.0
  assert scorer.DEFAULT_EMBED_MODEL == 'BAAI/bge-small-en-v1.5'


def test_embedder_default_model_uses_scorer_constant(monkeypatch) -> None:
  '''get_embedder falls back to DEFAULT_EMBED_MODEL.'''
  import job_hunter.services.embedder as emb
  from job_hunter.services.scorer import DEFAULT_EMBED_MODEL

  created: list = []

  class FakeProvider:
    '''Stand-in recording the requested model.'''

    def __init__(self, model) -> None:
      '''Record model.'''
      created.append(model)
      self.model_id = model

    def embed(self, texts):
      '''Return zero vectors.'''
      return [[0.0] for _ in texts]

  monkeypatch.setattr(emb, 'FastembedProvider', FakeProvider)
  emb._EMBEDDER_CACHE.clear()  # pylint: disable=protected-access
  try:
    settings = SimpleNamespace(embeddings={})
    emb.get_embedder(settings)  # pylint: disable=protected-access
  finally:
    emb._EMBEDDER_CACHE.clear()  # pylint: disable=protected-access
  assert created == [DEFAULT_EMBED_MODEL]


def test_vertical_classifier_logs_debug_on_llm_failure(
  tmp_path, monkeypatch, caplog,
) -> None:
  '''LLM classification failure logs at debug with exc_info.'''
  import logging

  import job_hunter.llm.structured as structured_mod
  from job_hunter.services.vertical_classifier import VerticalClassifier

  taxonomy = tmp_path / 'verticals.yaml'
  taxonomy.write_text(
    'verticals:\n  fintech:\n    keywords: [banking]\n'
    'default_confidence:\n  keyword_hit: 0.7\n',
    encoding='utf-8',
  )
  classifier = VerticalClassifier(taxonomy)

  async def failing(client, schema, instruction, text, session_id):
    '''Always raise.'''
    raise RuntimeError('gateway down')

  monkeypatch.setattr(
    structured_mod, 'complete_structured', failing,
  )
  with caplog.at_level(logging.DEBUG):
    result = asyncio.run(classifier.classify(
      'zzz no match here', SimpleNamespace(), 'session-1',
    ))
  assert result == ('unknown', 0.0)
  records = [
    record for record in caplog.records
    if record.name == 'job_hunter.services.vertical_classifier'
    and record.levelno == logging.DEBUG
  ]
  assert records, 'expected a debug log record on LLM failure'
  assert any(record.exc_info is not None for record in records)
