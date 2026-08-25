#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the private-dataset artifact publisher.

Covers credential fallbacks, artifact collection (globs, missing
sources), metadata generation and the create-vs-version decision --
all without touching the network or requiring the ``kaggle`` CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import publish_dataset as pub  # noqa: E402  # pylint: disable=wrong-import-position


class TestCollectArtifacts:
  """Staging mirrors the $WORK layout the restore step expects."""

  def _work(self, tmp_path: Path) -> Path:
    work = tmp_path / 'work'
    (work / 'checkpoints' / 'fold0').mkdir(parents=True)
    (work / 'checkpoints' / 'fold0' / 'last.ckpt').touch()
    (work / 'checkpoints' / 'fold0-best.ckpt').touch()
    (work / 'predictions').mkdir(parents=True)
    (work / 'predictions' / 'exp_oof_fold0.parquet').touch()
    (work / 'train_folds.csv').touch()
    return work

  def test_copies_dirs_files_and_globs(self, tmp_path: Path):
    staging = tmp_path / 'stage'
    staged = pub.collect_artifacts(str(self._work(tmp_path)), staging)
    assert staged == 4  # 2 ckpts + 1 parquet + folds csv
    assert (staging / 'checkpoints' / 'fold0' / 'last.ckpt').exists()
    assert (staging / 'train_folds.csv').exists()

  def test_missing_sources_are_skipped(self, tmp_path: Path):
    staging = tmp_path / 'stage'
    bare = tmp_path / 'bare'
    bare.mkdir()
    (bare / 'train_folds.csv').touch()
    assert pub.collect_artifacts(str(bare), staging) == 1

  def test_empty_work_aborts(self, tmp_path: Path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(SystemExit):
      pub.collect_artifacts(str(empty), tmp_path / 'stage')


class TestSanitizeSlug:
  """Kaggle slugs are [a-z0-9-]; everything else must be normalized."""

  def test_underscore_becomes_dash(self):
    assert (
      pub.sanitize_slug('ah2022_rsna-knee-abnormality-detection')
      == 'ah2022-rsna-knee-abnormality-detection'
    )

  def test_mixed_punctuation_and_case(self):
    assert pub.sanitize_slug('  My__Weird!!Name  ') == 'my-weird-name'

  def test_valid_slug_unchanged(self):
    assert pub.sanitize_slug('already-fine-123') == 'already-fine-123'


class TestWriteMetadata:
  """Metadata follows the official ``datasets init`` + patch flow."""

  @staticmethod
  def _fake_init(template: str | None, code: int = 0):
    """Build a _run_kaggle stub emulating ``datasets init``.

    Args:
        template: JSON text the CLI writes into the target folder
            (None simulates an init failure).
        code: Exit code to report for the init call.
    """

    def fake(args: list[str]) -> subprocess.CompletedProcess:
      if args[1] == 'init':
        if template is None or code != 0:
          return subprocess.CompletedProcess([], 1, '', 'init failed')
        target = Path(args[args.index('-p') + 1])
        (target / 'dataset-metadata.json').write_text(template)
        return subprocess.CompletedProcess([], 0, 'ok', '')
      raise AssertionError(f'unexpected CLI call {args}')

    return fake

  def test_patches_modern_slug_only_template(self, tmp_path, monkeypatch):
    template = (
      '{"title": "INSERT_TITLE_HERE", "id": "INSERT_SLUG_HERE", "licenses": []}'
    )
    monkeypatch.setattr(pub, '_run_kaggle', self._fake_init(template))
    staging = tmp_path
    pub.write_metadata(staging, 'ah2022', 'my-ds')
    metadata = json.loads((staging / 'dataset-metadata.json').read_text())
    assert metadata == {
      'title': 'my-ds',
      'id': 'my-ds',
      'licenses': [{'name': 'CC0-1.0'}],
    }

  def test_preserves_owner_slug_template(self, tmp_path, monkeypatch):
    template = (
      '{"title": "INSERT_TITLE_HERE",'
      ' "id": "INSERT_OWNER/INSERT_SLUG",'
      ' "licenses": [{"name": "CC0-1.0"}]}'
    )
    monkeypatch.setattr(pub, '_run_kaggle', self._fake_init(template))
    pub.write_metadata(tmp_path, 'ah2022', 'my-ds')
    metadata = json.loads((tmp_path / 'dataset-metadata.json').read_text())
    assert metadata['id'] == 'ah2022/my-ds'

  def test_fallback_when_init_fails(self, tmp_path, monkeypatch):
    monkeypatch.setattr(pub, '_run_kaggle', self._fake_init(None))
    pub.write_metadata(tmp_path, 'ah2022', 'my-ds')
    metadata = json.loads((tmp_path / 'dataset-metadata.json').read_text())
    assert metadata['id'] == 'ah2022/my-ds'
    assert metadata['licenses'] == [{'name': 'CC0-1.0'}]


class TestDatasetExists:
  def test_true_on_status_zero(self, monkeypatch):
    done = subprocess.CompletedProcess([], 0, stdout='ok', stderr='')
    monkeypatch.setattr(pub, '_run_kaggle', lambda args: done)
    assert pub.dataset_exists('u', 'n') is True

  def test_false_on_404(self, monkeypatch):
    done = subprocess.CompletedProcess(
      [], 1, stdout='', stderr='404 - Not Found'
    )
    monkeypatch.setattr(pub, '_run_kaggle', lambda args: done)
    assert pub.dataset_exists('u', 'n') is False

  def test_false_on_403_for_missing_datasets(self, monkeypatch):
    # Current kagglesdk hides existence: absent -> 403 Forbidden.
    done = subprocess.CompletedProcess(
      [],
      1,
      stdout='',
      stderr=(
        '403 Client Error: Forbidden for url: '
        'https://api.kaggle.com/v1/datasets.DatasetApiService/'
        'GetDatasetStatus'
      ),
    )
    monkeypatch.setattr(pub, '_run_kaggle', lambda args: done)
    assert pub.dataset_exists('u', 'n') is False

  def test_ambiguous_failure_raises(self, monkeypatch):
    done = subprocess.CompletedProcess(
      [], 1, stdout='', stderr='502 bad gateway'
    )
    monkeypatch.setattr(pub, '_run_kaggle', lambda args: done)
    with pytest.raises(RuntimeError):
      pub.dataset_exists('u', 'n')


class TestVerify:
  """Post-push verification waits out search-index lag."""

  def test_success_first_try(self, monkeypatch):
    monkeypatch.setattr(pub.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(pub, 'dataset_exists', lambda u, n: True)
    pub._verify('u', 'n')  # must not raise

  def test_retries_then_gives_up_quietly(self, monkeypatch):
    sleeps = []
    monkeypatch.setattr(pub.time, 'sleep', sleeps.append)
    monkeypatch.setattr(pub, '_VERIFY_ATTEMPTS', 3)
    monkeypatch.setattr(pub, '_VERIFY_WAIT_SECONDS', 7)
    monkeypatch.setattr(pub, 'dataset_exists', lambda u, n: False)
    pub._verify('u', 'n')  # warns instead of raising
    assert sleeps == [7, 7]


class TestPushDecision:
  """create on first push; version afterwards; lost-race retry."""

  def _stub(self, monkeypatch, expected_cmd: str, responses: dict):
    calls = []
    monkeypatch.setattr(pub.time, 'sleep', lambda seconds: None)

    def fake(args):
      calls.append(args[1])  # args[0] is the group name ('datasets')
      code, out = responses.get(args[1], (1, 'unexpected'))
      return subprocess.CompletedProcess([], code, stdout=out, stderr='')

    monkeypatch.setattr(pub, '_run_kaggle', fake)
    monkeypatch.setattr(
      pub,
      'dataset_exists',
      lambda u, n: expected_cmd == 'version',
    )
    return calls

  def test_first_push_creates(self, tmp_path: Path, monkeypatch):
    calls = self._stub(monkeypatch, 'create', {'create': (0, 'ok')})
    pub.push(tmp_path, 'u', 'n', 'msg', 'zip')
    assert calls == ['create']

  def test_existing_dataset_versions(self, tmp_path: Path, monkeypatch):
    calls = self._stub(monkeypatch, 'version', {'version': (0, 'ok')})
    pub.push(tmp_path, 'u', 'n', 'msg', 'zip')
    assert calls == ['version']

  def test_create_race_retries_as_version(self, tmp_path: Path, monkeypatch):
    calls = []

    def fake(args):
      calls.append(args[1])
      if args[1] == 'create':
        # Probe said "missing" but another worker created it first.
        return subprocess.CompletedProcess([], 1, '', 'already exists')
      return subprocess.CompletedProcess([], 0, 'ok', '')

    monkeypatch.setattr(pub, '_run_kaggle', fake)
    monkeypatch.setattr(pub, 'dataset_exists', lambda u, n: False)
    pub.push(tmp_path, 'u', 'n', 'msg', 'zip')
    assert calls == ['create', 'version']


class TestTransientRetries:
  """504/502/429 gateway blips must be retried, not fatal."""

  @staticmethod
  def _patch_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(pub.time, 'sleep', sleeps.append)
    return sleeps

  def test_504_then_success(self, tmp_path: Path, monkeypatch):
    sleeps = self._patch_sleep(monkeypatch)
    calls = []

    def fake(args):
      calls.append(args[1])
      if len(calls) == 1:
        return subprocess.CompletedProcess(
          [],
          1,
          '',
          '504 Server Error: Gateway Timeout for url: .../CreateDataset',
        )
      return subprocess.CompletedProcess([], 0, 'ok', '')

    monkeypatch.setattr(pub, '_run_kaggle', fake)
    monkeypatch.setattr(pub, 'dataset_exists', lambda u, n: False)
    pub.push(tmp_path, 'u', 'n', 'msg', 'zip')
    assert calls == ['create', 'create']
    assert sleeps and sleeps[0] > 0

  def test_exhausted_transient_raises_without_auth_hint(
    self, tmp_path: Path, monkeypatch
  ):
    self._patch_sleep(monkeypatch)
    monkeypatch.setattr(pub, 'dataset_exists', lambda u, n: False)

    def fake(args):
      return subprocess.CompletedProcess(
        [],
        1,
        '',
        '504 Server Error: Gateway Timeout for url: .../CreateDataset',
      )

    monkeypatch.setattr(pub, '_run_kaggle', fake)
    with pytest.raises(RuntimeError) as err:
      pub.push(tmp_path, 'u', 'n', 'msg', 'zip')
    assert 'hint:' not in str(err.value)
    assert 'after 4 attempts' in str(err.value)


class TestAuthHint:
  """Credential failures carry an actionable hint."""

  def test_hint_appended_on_forbidden(self):
    text = pub._auth_hint('create failed:\n403 Forbidden')
    assert 'hint:' in text and 'KAGGLE_USERNAME' in text

  def test_no_hint_on_gateway_timeout(self):
    text = pub._auth_hint(
      'create failed:\n504 Gateway Timeout (html says Forbidden somewhere)'
    )
    assert 'hint:' not in text

  def test_no_hint_on_other_failures(self):
    assert 'hint:' not in pub._auth_hint('boom\n502 bad gateway')
