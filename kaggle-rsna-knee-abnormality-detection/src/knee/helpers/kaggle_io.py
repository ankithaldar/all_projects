#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kaggle Dataset push/pull client for session persistence.

Implements the checkpoint-resume protocol from BLUEPRINT Section 6:

* Credentials resolve through the shared chain in ``knee.helpers.secrets``:
  process environment -> project ``.env`` -> Kaggle notebook User Secrets,
  so a notebook works with registered secrets even without a ``.env``.
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

from knee.helpers import secrets as secret_chain
from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

CommandRunner = Callable[..., subprocess.CompletedProcess]

DEFAULT_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10.0


class CredentialResolver:
  """Resolve Kaggle API credentials through the unified secret chain.

  Lookup order per key: process environment -> ``.env`` file -> Kaggle
  notebook User Secrets (when enabled). This means a Kaggle notebook works
  with credentials registered under Add-ons > Secrets whenever no ``.env``
  is available, while local shells and CI keep their env-var workflow.
  """

  def __init__(
    self,
    username_key: str,
    token_key: str,
    use_user_secrets: bool = True,
    env_path: str | None = None,
  ) -> None:
    """Store resolution parameters.

    Args:
        username_key: Secret/environment name holding KAGGLE_USERNAME.
        token_key: Secret/environment name holding the API token.
        use_user_secrets: When False, skip the Kaggle User Secrets backend
            (env/.env only) - used by tests and offline tooling.
        env_path: Optional explicit ``.env`` path forwarded to
            ``knee.helpers.secrets.get_secret``.
    """
    self.username_key = username_key
    self.token_key = token_key
    self.use_user_secrets = use_user_secrets
    self.env_path = env_path

  def _lookup(self, name: str) -> str | None:
    """Resolve one credential name via the shared secrets helper.

    Args:
        name: Secret/environment variable name.

    Returns:
        Value from env/.env/UserSecrets, or None.
    """
    if not self.use_user_secrets:
      secret_chain.load_project_env(self.env_path)
      return os.environ.get(name)
    return secret_chain.get_secret(name, env_path=self.env_path)

  def apply(self) -> None:
    """Export resolved credentials into process env and kaggle.json.

    Raises:
        RuntimeError: If no backend provides a complete credential pair.
    """
    username = self._lookup(self.username_key)
    token = self._lookup(self.token_key)
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

  def _full_slug(self, slug: str) -> str:
    """Qualify a bare dataset slug with the authenticated username.

    The kaggle CLI requires ``<owner>/<dataset>`` for create/status/
    version/download; a bare id crashes ``dataset_create_new`` with
    IndexError while splitting the reference. Configuration therefore
    stores short names (``rsna-knee-mvp-ckpt``) and this client expands
    them using KAGGLE_USERNAME (set by ``apply()`` or the environment).

    Args:
        slug: Bare name or already qualified ``owner/name`` string.

    Returns:
        Qualified slug; input passes through untouched when it already
        contains a slash.

    Raises:
        RuntimeError: When the slug is bare and no username is resolvable.
    """
    if '/' in slug:
      return slug
    username = os.environ.get('KAGGLE_USERNAME')
    if not username:
      username = self._credentials._lookup(  # pylint: disable=protected-access
        self._credentials.username_key  # pylint: disable=protected-access
      )
    if not username:
      raise RuntimeError(
        f'Cannot qualify dataset slug {slug!r}: resolve credentials first'
      )
    return f'{username}/{slug}'

  def dataset_exists(self, slug: str) -> bool:
    """Check whether a dataset slug is visible to the account.

    Args:
        slug: Fully qualified slug such as ``user/dataset-name``.

    Returns:
        True when the dataset status endpoint succeeds.
    """
    try:
      self._run(['kaggle', 'datasets', 'status', self._full_slug(slug)])
      return True
    except RuntimeError:
      return False

  @staticmethod
  def _write_metadata(staged_folder: str, slug: str, title: str) -> None:
    """Write kaggle dataset-metadata.json into a staging directory.

    Args:
        staged_folder: Directory that will be handed to the CLI.
        slug: Qualified ``owner/dataset`` id.
        title: Human-readable dataset title.
    """
    metadata = {
      'title': title,
      'id': slug,
      'licenses': [{'name': 'CC0-1.0'}],
    }
    with open(
      os.path.join(staged_folder, 'dataset-metadata.json'),
      'w',
      encoding='utf-8',
    ) as handle:
      json.dump(metadata, handle)

  def _stage_payload(
    self, slug: str, folder: str, title: str | None = None
  ) -> str:
    """Copy payload into a staging dir carrying valid metadata.

    Both ``create`` and ``version`` subcommands read identity from
    ``dataset-metadata.json`` inside ``-p <dir>``; publishing raw working
    directories without that file fails at the CLI layer.

    Args:
        slug: Dataset id (qualified via :meth:`_full_slug`).
        folder: Local source directory copied verbatim.
        title: Optional dataset title; defaults to the bare slug name.

    Returns:
        Path to the staged directory (caller removes when done).
    """
    staged_folder = os.path.join(
      tempfile.gettempdir(), f'kaggle_stage_{int(time.time() * 1000)}'
    )
    shutil.copytree(folder, staged_folder)
    self._write_metadata(
      staged_folder,
      self._full_slug(slug),
      title=title or slug.rsplit('/', 1)[-1],
    )
    return staged_folder

  def create_dataset(self, slug: str, folder: str, title: str) -> None:
    """Create a brand-new dataset from a local folder.

    Args:
        slug: Target slug (bare names qualified with the account owner).
        folder: Directory whose files become dataset content.
        title: Human-readable dataset title.
    """
    staged = self._stage_payload(slug, folder, title=title)
    try:
      self._run(
        [
          'kaggle',
          'datasets',
          'create',
          '-p',
          staged,
          '--dir-mode',
          'zip',
        ]
      )
    finally:
      shutil.rmtree(staged, ignore_errors=True)

  def push_version_inplace(self, slug: str, folder: str) -> None:
    """Version a directory by writing metadata directly into it.

    :meth:`push_version` stages a full copy of the payload, which is
    impossible for multi-gigabyte volume caches (disk would double).
    Here dataset-metadata.json lives inside ``folder`` itself; the CLI
    then zips the directory in place. The file is left behind, making
    re-pushes idempotent.

    Args:
        slug: Target slug; created when absent.
        folder: Directory holding ONLY files meant for the dataset.
            Callers must have moved/linked their shards into a
            dedicated             staging directory beforehand.
    """
    # Metadata carries the FULL owner/name id, so credentials must be
    # resolved before the first _run() would lazily apply them; cache
    # pushes are otherwise the first client call of a session and
    # crashed with 'Cannot qualify dataset slug'.
    try:
      self._credentials.apply()
    except RuntimeError as exc:
      _LOGGER.warning('credential apply during push failed: %s', exc)
    self._write_metadata(
      folder, self._full_slug(slug), title=slug.rsplit('/', 1)[-1]
    )
    if not self.dataset_exists(slug):
      _LOGGER.info('Dataset %s missing; creating from %s', slug, folder)
      self._run(
        [
          'kaggle',
          'datasets',
          'create',
          '-p',
          folder,
          '--dir-mode',
          'zip',
        ]
      )
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
    staged = self._stage_payload(slug, folder)
    try:
      self._run(
        [
          'kaggle',
          'datasets',
          'version',
          '-p',
          staged,
          '--dir-mode',
          'zip',
          '-m',
          f'auto-version {int(time.time())}',
        ]
      )
    finally:
      shutil.rmtree(staged, ignore_errors=True)

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
    self._run(
      [
        'kaggle',
        'datasets',
        'download',
        self._full_slug(slug),
        '--unzip',
        '-p',
        dest,
      ]
    )
    return True


class ArtifactSync:
  """Keep small stage artifacts persistent across ephemeral containers.

  Kaggle resets the container whenever the accelerator changes or a session
  restarts; only Kaggle Datasets survive. This helper mirrors the data-stage
  outputs (index/labels/folds) into a dedicated dataset so every session
  rehydrates itself, exactly like the checkpoint protocol does for training.

  Policy:

  * ``pull_if_missing`` downloads the newest version when any tracked file
    is absent locally (fresh container) — never overwrites a complete set.
  * ``push`` republishes after a build stage completes; failures are logged,
    never raised, because local copies remain usable within the session.
  """

  def __init__(
    self,
    client: KaggleDatasetClient | None,
    slug: str,
    local_dir: str,
    file_names: list[str],
  ) -> None:
    """Compose the sync helper.

    Args:
        client: Kaggle client (None disables all remote traffic).
        slug: Dataset slug backing the artifacts.
        local_dir: Directory holding the tracked files.
        file_names: File names whose presence defines completeness.
    """
    self.client = client
    self.slug = slug
    self.local_dir = local_dir
    self.file_names = file_names

  def _missing(self) -> list[str]:
    """List tracked files absent from the local directory.

    Returns:
        Names of missing files.
    """
    return [
      name
      for name in self.file_names
      if not os.path.exists(os.path.join(self.local_dir, name))
    ]

  def pull_if_missing(self) -> bool:
    """Restore artifacts from the dataset when locals are incomplete.

    Returns:
        True when a pull happened, False when files were already complete
        or syncing is disabled/unavailable.
    """
    if self.client is None or not self._missing():
      return False
    _LOGGER.info('Restoring %s from %s', self._missing(), self.slug)
    try:
      pulled = self.client.pull_latest(self.slug, self.local_dir)
    except RuntimeError as exc:
      _LOGGER.warning('Artifact restore failed (%s); continuing local', exc)
      return False
    still_missing = self._missing()
    if still_missing:
      _LOGGER.warning(
        'Dataset %s lacks %s; affected stages may need upstream builds',
        self.slug,
        still_missing,
      )
      return False
    return pulled

  def push(self) -> bool:
    """Publish current locals as a new dataset version (best effort).

    Returns:
        True when a new version was pushed, False when syncing is
        disabled, the tracked set is incomplete, or the CLI failed
        (callers may use this to schedule a retry).
    """
    if self.client is None:
      return False
    missing = self._missing()
    if missing:
      _LOGGER.warning('Skipping artifact push; incomplete set: %s', missing)
      return False
    try:
      self.client.push_version(self.slug, self.local_dir)
      _LOGGER.info('Artifacts pushed to %s', self.slug)
      return True
    except RuntimeError as exc:
      _LOGGER.error(
        'Artifact push failed (%s); local copy retained in %s',
        exc,
        self.local_dir,
      )
      return False
