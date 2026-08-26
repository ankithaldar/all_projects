#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kaggle Dataset push/pull client for session persistence.

Implements the checkpoint-resume protocol from BLUEPRINT Section 6:

* Credentials resolve from Kaggle ``UserSecretsClient`` (secret names come
  from configuration) with a ``KAGGLE_CONFIG_DIR``/``.env`` fallback for
  local development.
* Artifacts live in versioned Kaggle Datasets; every session ends with an
  immutable new version, doubling as rollback history.

All shell interaction goes through :mod:`subprocess` so unit tests can mock
the CLI boundary without network access (see tests/test_resume.py).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Callable

from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

CommandRunner = Callable[..., subprocess.CompletedProcess]

DEFAULT_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10.0


class CredentialResolver:
  """Resolve Kaggle API credentials from secrets or the environment."""

  def __init__(
    self,
    username_key: str,
    token_key: str,
    use_user_secrets: bool = True,
  ) -> None:
    """Store secret names used for resolution.

    Args:
        username_key: Secret/environment name holding KAGGLE_USERNAME.
        token_key: Secret/environment name holding the API token.
        use_user_secrets: Try ``UserSecretsClient`` before os.environ.
    """
    self.username_key = username_key
    self.token_key = token_key
    self.use_user_secrets = use_user_secrets

  def _from_user_secrets(self) -> tuple[str, str] | None:
    """Read credentials from the Kaggle notebook secret store.

    Returns:
        ``(username, token)`` or None when unavailable.

    Raises:
        RuntimeError: When the secret exists but cannot be retrieved.
    """
    if not self.use_user_secrets:
      return None
    try:
      # Kaggle-notebook-only dependency; absent in local/CI environments.
      # pylint: disable=import-outside-toplevel
      from kaggle_secrets import UserSecretsClient
      # pylint: enable=import-outside-toplevel

      secrets = UserSecretsClient()
      return (
        secrets.get_secret(self.username_key),
        secrets.get_secret(self.token_key),
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      _LOGGER.info('UserSecrets unavailable (%s); falling back to env', exc)
      return None

  def apply(self) -> None:
    """Export resolved credentials into process env and kaggle.json.

    Raises:
        RuntimeError: If neither UserSecrets nor environment variables
            provide a complete credential pair.
    """
    credentials = self._from_user_secrets()
    username, token = credentials or (
      os.environ.get(self.username_key),
      os.environ.get(self.token_key),
    )
    if not username or not token:
      raise RuntimeError(
        f'Missing Kaggle credentials ({self.username_key}/{self.token_key})'
      )
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = token
    config_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(config_dir, exist_ok=True)
    marker = os.path.join(config_dir, 'kaggle.json')
    with open(marker, 'w', encoding='utf-8') as handle:
      json.dump({'username': username, 'key': token}, handle)
    os.chmod(marker, 0o600)


class KaggleDatasetClient:
  """Thin, retrying wrapper around the kaggle CLI for dataset versioning."""

  def __init__(
    self,
    credential_resolver: CredentialResolver,
    runner: CommandRunner | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = RETRY_BACKOFF_SECONDS,
  ) -> None:
    """Compose the client.

    Args:
        credential_resolver: Source of API credentials.
        runner: Optional subprocess callable (injection point for tests).
        retries: Attempts per CLI invocation before failing.
        backoff_seconds: Linear backoff between attempts.
    """
    self._credentials = credential_resolver
    self._runner = runner or subprocess.run
    self._retries = retries
    self._backoff_seconds = backoff_seconds

  def _run(self, args: list[str]) -> subprocess.CompletedProcess:
    """Execute one CLI command with retry/backoff.

    Args:
        args: Argument vector following the program name.

    Returns:
        Completed process on success.

    Raises:
        RuntimeError: After exhausting retries with non-zero exit codes.
    """
    self._credentials.apply()
    last_error = 'no attempts made'
    for attempt in range(1, self._retries + 1):
      result = self._runner(args, capture_output=True, text=True, check=False)
      if getattr(result, 'returncode', 1) == 0:
        return result
      last_error = getattr(result, 'stderr', '') or str(result)
      _LOGGER.warning(
        'CLI attempt %d/%d failed: %s',
        attempt,
        self._retries,
        last_error.strip(),
      )
      time.sleep(self._backoff_seconds * attempt)
    raise RuntimeError(
      f'kaggle CLI failed after {self._retries} attempts: {last_error}'
    )

  def dataset_exists(self, slug: str) -> bool:
    """Check whether a dataset slug is visible to the account.

    Args:
        slug: Fully qualified slug such as ``user/dataset-name``.

    Returns:
        True when the dataset status endpoint succeeds.
    """
    try:
      self._run(['kaggle', 'datasets', 'status', slug])
      return True
    except RuntimeError:
      return False

  def create_dataset(self, slug: str, folder: str, title: str) -> None:
    """Create a brand-new dataset from a local folder.

    Args:
        slug: Target slug (owner inferred from credentials).
        folder: Directory whose files become dataset content.
        title: Human-readable dataset title.
    """
    metadata = {
      'title': title,
      'id': slug,
      'licenses': [{'name': 'CC0-1.0'}],
    }
    with tempfile.TemporaryDirectory() as staging:
      staged_folder = os.path.join(staging, 'payload')
      shutil.copytree(folder, staged_folder)
      with open(
        os.path.join(staged_folder, 'dataset-metadata.json'),
        'w',
        encoding='utf-8',
      ) as handle:
        json.dump(metadata, handle)
      self._run(
        [
          'kaggle',
          'datasets',
          'create',
          '-p',
          staged_folder,
          '--dir-mode',
          'zip',
        ]
      )

  def push_version(self, slug: str, folder: str) -> None:
    """Publish folder contents as a new immutable dataset version.

    Creates the dataset first when it does not exist yet.

    Args:
        slug: Target slug; created when absent.
        folder: Local directory to publish.
    """
    if not self.dataset_exists(slug):
      _LOGGER.info('Dataset %s missing; creating instead of versioning', slug)
      self.create_dataset(slug, folder, title=slug.rsplit('/', 1)[-1])
      return
    self._run(
      [
        'kaggle',
        'datasets',
        'version',
        '-p',
        folder,
        '--dir-mode',
        'zip',
        '-m',
        f'auto-version {int(time.time())}',
      ]
    )

  def pull_latest(self, slug: str, dest: str) -> bool:
    """Download and unpack the newest dataset version into dest.

    Args:
        slug: Dataset slug to download.
        dest: Destination directory created when missing.

    Returns:
        True on success, False when the dataset does not exist.
    """
    os.makedirs(dest, exist_ok=True)
    if not self.dataset_exists(slug):
      _LOGGER.info('No remote dataset %s yet; starting fresh', slug)
      return False
    self._run(['kaggle', 'datasets', 'download', slug, '--unzip', '-p', dest])
    return True
