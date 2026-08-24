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


class TestWriteMetadata:
  def test_metadata_matches_tutorial_contract(self, tmp_path: Path):
    pub.write_metadata(
      tmp_path, 'ah2022', 'ah2022_rsna-knee-abnormality-detection'
    )
    metadata = json.loads((tmp_path / 'dataset-metadata.json').read_text())
    assert metadata['id'] == ('ah2022/ah2022_rsna-knee-abnormality-detection')
    assert metadata['title'] == 'ah2022_rsna-knee-abnormality-detection'
    assert metadata['licenses'], 'Kaggle requires a license entry'


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

  def test_ambiguous_failure_raises(self, monkeypatch):
    done = subprocess.CompletedProcess([], 1, stdout='', stderr='403 auth')
    monkeypatch.setattr(pub, '_run_kaggle', lambda args: done)
    with pytest.raises(RuntimeError):
      pub.dataset_exists('u', 'n')


class TestPushDecision:
  """create on first push; version afterwards; lost-race retry."""

  def _stub(self, monkeypatch, expected_cmd: str, responses: dict):
    calls = []

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
