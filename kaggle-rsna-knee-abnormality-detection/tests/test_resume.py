#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the Kaggle checkpoint persistence boundary (CLI mocked)."""

import os

import pytest

from knee.helpers.kaggle_io import CredentialResolver, KaggleDatasetClient


class FakeRunner:
  """Scriptable subprocess stand-in returning queued results."""

  def __init__(self, fail_first: int = 0):
    """Configure scripted failures.

    Args:
        fail_first: Number of initial calls that exit non-zero.
    """
    self.fail_first = fail_first
    self.calls: list[list[str]] = []

  def __call__(self, args, capture_output=True, text=True):
    """Record the call and return a fabricated CompletedProcess.

    Args:
        args: Argument vector.
        capture_output: Ignored; present for signature parity.
        text: Ignored; present for signature parity.

    Returns:
        Object mimicking subprocess.CompletedProcess.
    """
    self.calls.append(list(args))
    code = 1 if len(self.calls) <= self.fail_first else 0
    return type('R', (), {'returncode': code, 'stderr': 'boom', 'stdout': ''})


@pytest.fixture()
def env_credentials(monkeypatch):
  """Provide credentials through environment variables."""
  monkeypatch.setenv('KAGGLE_USERNAME', 'tester')
  monkeypatch.setenv('KAGGLE_KEY', 'secret')
  monkeypatch.setenv('HOME', '/tmp/fakehome')


class TestCredentialResolver:
  """Credential resolution paths."""

  def test_env_fallback_exports_kaggle_json(self, env_credentials):
    resolver = CredentialResolver('KAGGLE_USERNAME', 'KAGGLE_KEY',
                                  use_user_secrets=False)
    resolver.apply()
    marker = os.path.expanduser('~/.kaggle/kaggle.json')
    assert os.path.exists(marker)
    assert os.environ['KAGGLE_USERNAME'] == 'tester'

  def test_missing_credentials_raise(self, monkeypatch):
    monkeypatch.delenv('KAGGLE_USERNAME', raising=False)
    monkeypatch.delenv('KAGGLE_KEY', raising=False)
    resolver = CredentialResolver('KAGGLE_USERNAME', 'KAGGLE_KEY',
                                  use_user_secrets=False)
    with pytest.raises(RuntimeError):
      resolver.apply()


class TestKaggleDatasetClient:
  """Retry, create-vs-version, and pull behavior."""

  @pytest.fixture()
  def slug_env(self, monkeypatch, tmp_path):
    """Point HOME into tmp so apply() never touches the real one."""
    monkeypatch.setenv('HOME', str(tmp_path))
    return tmp_path

  def test_push_creates_when_absent(self, slug_env):
    runner = FakeRunner()  # status fails -> treated as absent
    client = KaggleDatasetClient(
        CredentialResolver('KAGGLE_USERNAME', 'KAGGLE_KEY', False),
        runner=runner,
    )
    folder = slug_env / 'payload'
    (folder / 'fold0').mkdir(parents=True)
    (folder / 'fold0' / 'last.ckpt').write_text('weights')
    client.push_version('user/ckpt', str(folder))
    verbs = [c[2] for c in runner.calls]
    assert 'status' in verbs and 'create' in verbs and 'version' not in verbs

  def test_push_versions_when_present(self, slug_env):
    # First call = status success -> dataset exists -> version path taken.
    runner_ok_status = FakeRunner()

    def always_ok(args, capture_output=True, text=True):
      runner_ok_status.calls.append(list(args))
      return type('R', (), {'returncode': 0, 'stderr': '', 'stdout': ''})

    client = KaggleDatasetClient(
        CredentialResolver('KAGGLE_USERNAME', 'KAGGLE_KEY', False),
        runner=always_ok,
    )
    folder = slug_env / 'payload'
    folder.mkdir(parents=True)
    client.push_version('user/ckpt', str(folder))
    assert any(c[2] == 'version' for c in runner_ok_status.calls)

  def test_retries_then_succeeds(self, slug_env):
    runner = FakeRunner(fail_first=2)
    client = KaggleDatasetClient(
        CredentialResolver('KAGGLE_USERNAME', 'KAGGLE_KEY', False),
        runner=runner,
        retries=3,
        backoff_seconds=0.0,
    )
    assert client.dataset_exists('user/x') is True
    assert len(runner.calls) == 3

  def test_pull_returns_false_when_missing(self, slug_env):
    runner = FakeRunner()
    client = KaggleDatasetClient(
        CredentialResolver('KAGGLE_USERNAME', 'KAGGLE_KEY', False),
        runner=runner,
    )
    assert client.pull_latest('user/nope', str(slug_env / 'out')) is False
