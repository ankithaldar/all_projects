#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the Kaggle checkpoint persistence boundary (CLI mocked)."""

# pytest fixture injection triggers redefinition/unused-argument warnings
# that do not apply to test code.
# pylint: disable=redefined-outer-name,unused-argument

import os

import pytest

from knee.helpers.kaggle_io import CredentialResolver, KaggleDatasetClient


class FakeRunner:
  """Scriptable subprocess stand-in failing selected CLI verbs."""

  def __init__(
    self, fail_verbs: set[str] | None = None, max_status_failures: int = 0
  ):
    """Configure scripted behavior.

    Args:
        fail_verbs: Verbs (e.g. 'status') that always exit non-zero.
        max_status_failures: Number of initial 'status' calls that fail
            before succeeding (retry testing).
    """
    self.fail_verbs = fail_verbs or set()
    self.max_status_failures = max_status_failures
    self.status_calls = 0
    self.calls: list[list[str]] = []

  def __call__(self, args, capture_output=True, text=True, check=False):
    """Record the call and fabricate a CompletedProcess-like result.

    Args:
        args: Argument vector.
        capture_output: Ignored; signature parity only.
        text: Ignored; signature parity only.
        check: Ignored; the client inspects returncode itself.

    Returns:
        Object mimicking subprocess.CompletedProcess.
    """
    self.calls.append(list(args))
    verb = args[2] if len(args) > 2 else ''
    if verb in self.fail_verbs:
      return type('R', (), {'returncode': 1, 'stderr': 'boom', 'stdout': ''})
    if verb == 'status' and self.max_status_failures > 0:
      self.status_calls += 1
      if self.status_calls <= self.max_status_failures:
        return type('R', (), {'returncode': 1, 'stderr': 'flaky', 'stdout': ''})
    return type('R', (), {'returncode': 0, 'stderr': '', 'stdout': ''})


@pytest.fixture()
def env_credentials(monkeypatch):
  """Provide credentials through environment variables."""
  monkeypatch.setenv('KAGGLE_USERNAME', 'tester')
  monkeypatch.setenv('KAGGLE_KEY', 'secret')
  monkeypatch.setenv('HOME', '/tmp/fakehome')


class TestCredentialResolver:
  """Credential resolution paths."""

  def test_env_fallback_exports_kaggle_json(self, env_credentials):
    resolver = CredentialResolver(
      'KAGGLE_USERNAME', 'KAGGLE_KEY', use_user_secrets=False
    )
    resolver.apply()
    marker = os.path.expanduser('~/.kaggle/kaggle.json')
    assert os.path.exists(marker)
    assert os.environ['KAGGLE_USERNAME'] == 'tester'

  def test_missing_credentials_raise(self, monkeypatch):
    monkeypatch.delenv('KAGGLE_USERNAME', raising=False)
    monkeypatch.delenv('KAGGLE_KEY', raising=False)
    resolver = CredentialResolver(
      'KAGGLE_USERNAME', 'KAGGLE_KEY', use_user_secrets=False
    )
    with pytest.raises(RuntimeError):
      resolver.apply()


class TestKaggleDatasetClient:
  """Retry, create-vs-version, and pull behavior."""

  @pytest.fixture()
  def slug_env(self, monkeypatch, tmp_path):
    """Point HOME into tmp and provide env credentials for the client."""
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('KAGGLE_USERNAME', 'tester')
    monkeypatch.setenv('KAGGLE_KEY', 'secret')
    return tmp_path

  def test_push_creates_when_absent(self, slug_env):
    runner = FakeRunner(fail_verbs={'status'})  # status fails -> absent
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
    runner = FakeRunner()  # status succeeds -> dataset exists
    client = KaggleDatasetClient(
      CredentialResolver('KAGGLE_USERNAME', 'KAGGLE_KEY', False),
      runner=runner,
    )
    folder = slug_env / 'payload'
    folder.mkdir(parents=True)
    client.push_version('user/ckpt', str(folder))
    verbs = [c[2] for c in runner.calls]
    assert 'version' in verbs and 'create' not in verbs

  def test_retries_then_succeeds(self, slug_env):
    runner = FakeRunner(max_status_failures=2)
    client = KaggleDatasetClient(
      CredentialResolver('KAGGLE_USERNAME', 'KAGGLE_KEY', False),
      runner=runner,
      retries=3,
      backoff_seconds=0.0,
    )
    assert client.dataset_exists('user/x') is True
    assert len(runner.calls) == 3

  def test_pull_returns_false_when_missing(self, slug_env):
    runner = FakeRunner(fail_verbs={'status'})
    client = KaggleDatasetClient(
      CredentialResolver('KAGGLE_USERNAME', 'KAGGLE_KEY', False),
      runner=runner,
    )
    assert client.pull_latest('user/nope', str(slug_env / 'out')) is False
