#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle 7: shared quote_identifier dedup and _normalize_inserted guard.'''

from __future__ import annotations

from pathlib import Path


def test_quote_identifier_simple() -> None:
  '''quote_identifier double-quotes a plain identifier.'''
  from job_hunter.core.db import quote_identifier

  assert quote_identifier('col') == '"col"'


def test_quote_identifier_escapes_double_quotes() -> None:
  '''Embedded double quotes are doubled inside the quoting.'''
  from job_hunter.core.db import quote_identifier

  assert quote_identifier('we"ird') == '"we""ird"'


def test_repos_do_not_define_private_quote() -> None:
  '''Jobs and companies repos use the shared helper, no private _quote.'''
  import job_hunter.db.repositories.companies as companies
  import job_hunter.db.repositories.jobs as jobs

  assert not hasattr(jobs, '_quote')
  assert not hasattr(companies, '_quote')


def test_base_docstring_says_api_wraps_valid_domain() -> None:
  '''base.py no longer claims the API keeps its own copy.'''
  base_path = (
    Path(__file__).resolve().parents[2]
    / 'src'
    / 'job_hunter'
    / 'job_hunter'
    / 'adapters'
    / 'base.py'
  )
  text = base_path.read_text(encoding='utf-8')
  assert 'API keeps its own copy' not in text
  assert 'API wraps valid_domain' in text


def test_normalize_empty_dict_skipped() -> None:
  '''[{}] yields no phantom jobs and does not raise.'''
  from job_hunter.graph.state import _normalize_inserted

  assert _normalize_inserted({'inserted': [{}]}) == []


def test_normalize_unrelated_dict_skipped() -> None:
  '''[{'x': 1}] lacking identity is skipped without raising.'''
  from job_hunter.graph.state import _normalize_inserted

  assert _normalize_inserted({'inserted': [{'x': 1}]}) == []


def test_normalize_good_flat_dict_preserved() -> None:
  '''A flat dict with identity still normalizes with fields kept.'''
  from job_hunter.graph.state import _normalize_inserted

  good = {
    'canonical_url': 'https://e.com/1',
    'content_hash': 'h123',
    'title': 'Data Scientist',
    'company_name': 'Acme',
    'source_key': 'lever',
    'url': 'https://e.com/1',
  }
  out = _normalize_inserted({'inserted': [good]})
  assert len(out) == 1
  assert out[0].canonical_url == 'https://e.com/1'
  assert out[0].content_hash == 'h123'
  assert out[0].raw.title == 'Data Scientist'
  assert out[0].raw.company_name == 'Acme'
  assert out[0].raw.url == 'https://e.com/1'
  assert out[0].raw.source_key == 'lever'


def test_normalize_good_nested_dict_preserved() -> None:
  '''A nested dict with identity still validates with fields kept.'''
  from job_hunter.graph.state import _normalize_inserted

  good = {
    'job_id': 7,
    'raw': {
      'source_key': 'lever',
      'url': 'https://e.com/2',
      'title': 'ML Engineer',
      'company_name': 'Acme',
    },
    'canonical_url': 'https://e.com/2',
    'content_hash': 'abc',
  }
  out = _normalize_inserted({'inserted': [good]})
  assert len(out) == 1
  assert out[0].job_id == 7
  assert out[0].raw.title == 'ML Engineer'
  assert out[0].canonical_url == 'https://e.com/2'
  assert out[0].content_hash == 'abc'


def test_normalize_mixed_skips_only_phantom() -> None:
  '''Mixed lists keep good items and skip identity-less ones.'''
  from job_hunter.graph.state import _normalize_inserted

  good = {
    'canonical_url': 'https://e.com/3',
    'content_hash': 'h999',
    'title': 'DS',
    'company_name': 'Acme',
    'source_key': 'lever',
    'url': 'https://e.com/3',
  }
  out = _normalize_inserted({'inserted': [good, {}, {'x': 1}]})
  assert len(out) == 1
  assert out[0].canonical_url == 'https://e.com/3'


def test_normalize_flat_manual_build_copies_salary_and_fields() -> None:
  '''Flat dict needing manual build copies fields and salary list.'''
  from job_hunter.graph.state import _normalize_inserted

  flat = {
    'job_id': '7',
    'canonical_url': 'https://e.com/flat-salary',
    'content_hash': 'h-flat-salary',
    'city': 'Pune',
    'region': 'MH',
    'url': 'https://e.com/flat-salary',
    'title': 'Flat DS',
    'company_name': 'Acme',
    'source_key': 'lever',
    'salary': [60, 80],
  }
  out = _normalize_inserted({'inserted': [flat]})
  assert len(out) == 1
  job = out[0]
  assert job.job_id == 7
  assert job.canonical_url == 'https://e.com/flat-salary'
  assert job.content_hash == 'h-flat-salary'
  assert job.city == 'Pune'
  assert job.region == 'MH'
  assert job.salary_min_lpa == 60.0
  assert job.salary_max_lpa == 80.0
  assert job.raw.title == 'Flat DS'
  assert job.raw.company_name == 'Acme'
  assert job.raw.url == 'https://e.com/flat-salary'
  assert job.raw.source_key == 'lever'
