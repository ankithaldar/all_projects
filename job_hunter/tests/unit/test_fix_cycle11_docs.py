#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-11 pins: docs truthfulness plus bool/payload/sanitize/inbox/curator.'''

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import pytest

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

ROOT = Path(__file__).resolve().parents[2]
TESTS_UNIT = ROOT / 'tests' / 'unit'


def _section(text: str, start: str, end: str) -> str:
  '''Extract a markdown section between two headings.

  Args:
    text: Full document text.
    start: Heading marker for the section start.
    end: Heading marker for the section end.

  Returns:
    Text between the markers.
  '''
  begin = text.index(start)
  finish = text.index(end, begin)
  return text[begin:finish]


def _recount_number() -> int:
  '''Count unit test modules on disk.

  Returns:
    Number of tests/unit/test_*.py files.
  '''
  return len(list(TESTS_UNIT.glob('test_*.py')))


def _settings(tmp_path: Path, monkeypatch):
  '''Build migrated settings for direct service tests.

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


def _make_client(tmp_path: Path, monkeypatch) -> TestClient:
  '''Build an app client against a temporary data directory.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.

  Returns:
    Configured TestClient.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  from job_hunter.api.main import create_app
  app = create_app(str(config_path))
  return TestClient(app, raise_server_exceptions=False)


def _make_curator(settings):
  '''Build a curator with a stub client (no gateway network).

  Args:
    settings: Application settings.

  Returns:
    ProfileCurator instance.
  '''
  from job_hunter.services.profile_curator import ProfileCurator
  return ProfileCurator(settings, client=object())


def _parsed_ok(path) -> dict:
  '''Fake successful parse result for a written file.

  Args:
    path: Written resume file path.

  Returns:
    Parse mapping with ok True.
  '''
  data = Path(path).read_bytes()
  return {
    'text': 'python data scientist experience',
    'pages': 1,
    'sha256': hashlib.sha256(data).hexdigest(),
    'ok': True,
    'truncated': False,
  }


def _extraction():
  '''Build a fake extraction result.

  Returns:
    ProfileExtraction instance.
  '''
  from job_hunter.services.profile_curator import ProfileExtraction
  return ProfileExtraction(
    skills=['Python'],
    summary='Experienced data scientist.',
    current_title='Data Scientist',
    experience_years=5.0,
    seniority_keywords=['senior'],
  )


def _configure_curator_ok(curator):
  '''Stub parser, skills, and extract for a successful ingest.

  Args:
    curator: ProfileCurator instance.
  '''
  curator._parser.parse_file = _parsed_ok
  curator._persist_skills = lambda raw, cid: []

  async def fake_extract(_text):
    '''Return a fixed extraction.

    Args:
      _text: Resume text.

    Returns:
      Fake extraction.
    '''
    return _extraction()

  curator.extract = fake_extract  # type: ignore[method-assign]


# Docs pins ---------------------------------------------------------------


def test_docs_taxonomy_split_14() -> None:
  '''§14 splits None vs non-None non-dict skill specs exactly.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s14 = _section(arch, '## 14.', '## 15.')
  assert 'None spec seeded with empty category/aliases' in s14
  assert (
    'non-None non-dict spec skipped/not counted, category preserved' in s14
  )
  # Retain the log phrase older docs tests pin.
  assert 'skipping non-dict skill spec' in s14


def test_docs_resolve_many_contract_14() -> None:
  '''§14 states the resolve_many contract and its 422 mapping.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s14 = _section(arch, '## 14.', '## 15.')
  assert (
    'resolve_many(non-str elements skipped with warning; '
    'bare str/bytes/non-list raises TypeError -> 422 on '
    'PUT /api/profile/skills)' in s14
  )


def test_docs_exc_file_path_falsy_14_and_api() -> None:
  '''§14 and api resume row state the FK-blocked file_path rule.'''
  phrase = (
    'FK-blocked/retained failure sets exc.file_path falsy '
    'so API does not unlink stored file'
  )
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s14 = _section(arch, '## 14.', '## 15.')
  assert phrase in s14
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  resume_row = next(
    (line for line in api.splitlines() if '/api/profile/resume' in line), '',
  )
  assert resume_row, 'resume row not found'
  assert phrase in resume_row


def test_docs_sanitize_caps_api_error_section() -> None:
  '''api.md error section states sanitize + page/experience caps.'''
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  errors = _section(api, '## Error bodies', '## Auth / token note')
  assert (
    '_sanitize_nonfinite stringifies non-finite floats '
    'in validation errors (422 not 500); jobs page le=10000; '
    'profile experience 0-60' in errors
  )


def test_docs_roadmap_feature07_quarantine() -> None:
  '''roadmap feature_07 qualifies quarantine as counts only.'''
  roadmap = (ROOT / 'docs' / 'roadmap.md').read_text(encoding='utf-8')
  row = next(
    (line for line in roadmap.splitlines() if 'feature_07' in line), '',
  )
  assert row, 'feature_07 row not found'
  assert 'quarantine counts (no writer/review path)' in row
  assert 'quarantine review path' not in row


def test_docs_test_module_counts_match_recount() -> None:
  '''Docs state the live test-module recount (live-list hedge kept).'''
  count = _recount_number()
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  writer = (ROOT / 'agents' / 'test-writer.md').read_text(encoding='utf-8')
  assert f'{count} unit test modules' in arch
  assert f'{count} unit test modules' in writer
  assert 'live list is tests/unit/' in arch
  assert 'live list is every module in `tests/unit/`' in writer


# Settings bool -----------------------------------------------------------


def test_settings_scoring_weights_bool_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''PUT scoring_weights with a bool value -> 422.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.put(
    '/api/settings?key=scoring_weights', json={'w': True},
  )
  assert response.status_code == 422, response.text


def test_settings_salary_floor_bool_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''PUT salary_hard_floor_lpa with True -> 422.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.put(
    '/api/settings?key=salary_hard_floor_lpa', json=True,
  )
  assert response.status_code == 422, response.text


def test_settings_cap_bool_422(tmp_path: Path, monkeypatch) -> None:
  '''PUT either discovery cap with True -> 422.'''
  client = _make_client(tmp_path, monkeypatch)
  for key in (
    'discovery.max_llm_calls_per_run',
    'discovery.enrich_per_run_cap',
  ):
    response = client.put(f'/api/settings?key={key}', json=True)
    assert response.status_code == 422, (key, response.text)


# Taxonomy payload guard --------------------------------------------------


def test_load_seed_non_dict_payload_returns_zero_with_warning(
  tmp_path: Path, monkeypatch, caplog,
) -> None:
  '''load_seed with list/str skills payload -> 0 and a warning.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  with caplog.at_level(logging.WARNING):
    assert taxonomy.load_seed({'skills': ['Python']}) == 0
    assert taxonomy.load_seed({'skills': 'x'}) == 0
  messages = [record.getMessage() for record in caplog.records]
  assert any(
    'non-dict skills payload' in message for message in messages
  ), messages
  assert any(
    record.levelno >= logging.WARNING for record in caplog.records
  ), messages


# _sanitize_nonfinite -----------------------------------------------------


def test_sanitize_nonfinite_direct() -> None:
  '''_sanitize_nonfinite stringifies inf/nan, passes finite through.'''
  from job_hunter.api.main import _sanitize_nonfinite
  value = {
    'a': float('inf'),
    'b': [float('nan'), 1.5, (float('-inf'), 2)],
    'c': 'keep',
    'd': None,
    'e': 3,
  }
  assert _sanitize_nonfinite(value) == {
    'a': 'inf',
    'b': ['nan', 1.5, ['-inf', 2]],
    'c': 'keep',
    'd': None,
    'e': 3,
  }
  assert _sanitize_nonfinite(float('inf')) == 'inf'
  assert _sanitize_nonfinite(1.5) == 1.5
  assert _sanitize_nonfinite('x') == 'x'


# Manual inbox generic exception ------------------------------------------


def test_manual_inbox_generic_valueerror_moves_file(
  tmp_path: Path, monkeypatch,
) -> None:
  '''persist_record ValueError (non-DB) -> count 0, file still moved.

  Actual behaviour: _insert_records swallows non-DB per-record
  errors (continue), returns inserted=0, and scan_inbox treats the
  file as successfully processed and renames it to _processed.
  '''
  import job_hunter.adapters.manual_inbox as inbox_mod
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations

  async def _value_boom(*args, **kwargs):
    raise ValueError('non-db boom')

  monkeypatch.setattr(inbox_mod, 'persist_record', _value_boom)
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  inbox = tmp_path / 'inbox'
  inbox.mkdir(parents=True, exist_ok=True)
  run_migrations(tmp_path / 'app.db')
  source = inbox / 'saved.csv'
  source.write_text(
    'Job Title,Company,Location\nStuff,Acme Corp,Mumbai\n',
    encoding='utf-8',
  )
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  settings = AppSettings(config_path)
  count = asyncio.run(inbox_mod.scan_inbox(settings))
  assert count == 0, count
  assert not source.exists(), 'generic per-record error still moves file'
  processed = inbox / '_processed'
  moved = list(processed.iterdir()) if processed.exists() else []
  assert len(moved) == 1, moved
  assert moved[0].name.endswith('_saved.csv'), moved


# Curator retention branches ----------------------------------------------


def test_curator_extract_failure_temp_unlinked(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Extract raising before save_resume_ex unlinks the temp file.

  Actual behaviour: exc.file_path is the (now dangling) temp path,
  whose file is gone, so the API route's unlink on it is a no-op;
  no resume row is written.
  '''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  _configure_curator_ok(curator)

  async def boom_extract(_text):
    raise RuntimeError('extract boom')

  curator.extract = boom_extract  # type: ignore[method-assign]
  with pytest.raises(RuntimeError, match='extract boom') as excinfo:
    asyncio.run(curator.ingest_upload('x.pdf', b'%PDF-extract-boom'))
  exc = excinfo.value
  temp_path = getattr(exc, 'file_path', None)
  assert temp_path, 'expected the temp path on the exception'
  assert not Path(temp_path).exists(), temp_path
  resumes_dir = settings.data_dir / 'resumes'
  leftovers = list(resumes_dir.iterdir()) if resumes_dir.exists() else []
  assert leftovers == [], leftovers
  from job_hunter.db.repositories.profile import ProfileRepository
  assert ProfileRepository(settings.db_path).get_profile(1) is None


def test_curator_unlink_oserror_swallowed_on_failure(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Unlink raising OSError during failure cleanup does not mask error.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  _configure_curator_ok(curator)

  async def boom_extract(_text):
    raise RuntimeError('extract boom')

  curator.extract = boom_extract  # type: ignore[method-assign]

  def boom_unlink(self, missing_ok=False):
    '''Raise OSError as a locked file would.

    Args:
      self: Path being unlinked.
      missing_ok: Missing-ok flag.

    Raises:
      OSError: Simulated unlink failure.
    '''
    raise OSError('unlink boom')

  monkeypatch.setattr(Path, 'unlink', boom_unlink)
  with pytest.raises(RuntimeError, match='extract boom'):
    asyncio.run(curator.ingest_upload('y.pdf', b'%PDF-extract-boom2'))
