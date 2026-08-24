#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publish pipeline artifacts to ONE private Kaggle dataset.

Kernels are fresh containers, so resumable state (fold checkpoints,
per-fold OOF parquets, folds CSV, weak labels, teacher checkpoints)
must leave the machine before the 12 h wall kills it. This script
mirrors everything resumable from ``$WORK`` into a staging folder and
pushes it as a single PRIVATE dataset (default:
``ah2022_rsna-knee-abnormality-detection``) using the Kaggle API:

    kaggle datasets init     -> patch dataset-metadata.json ->
    kaggle datasets create   (first push)   |   datasets version   (later)

The next kernel attaches that dataset (read-only ``/kaggle/input``
mount) and points ``PREV_OUTPUT`` at it; ``scripts/kaggle_run.sh`` then
copies the state forward automatically.

Setup (once per kernel): Add-ons -> Secrets -> add ``KAGGLE_USERNAME``
and ``KAGGLE_KEY`` (resolution order: env -> .env -> Kaggle Secrets ->
~/.kaggle/kaggle.json).

Usage:
    python scripts/publish_dataset.py \
        --work /kaggle/working \
        --dataset-name ah2022_rsna-knee-abnormality-detection \
        [--message 'folds 0-1 done'] [--dir-mode zip]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from knee.helpers.env import get_secret
from knee.helpers.logging_utils import get_logger

#: Directories copied wholesale from $WORK into the dataset.
ARTIFACT_DIRS = ('checkpoints', 'predictions', 'text_teacher')

#: File patterns copied from $WORK into the dataset (fnmatch-style).
ARTIFACT_FILES = (
  'train_folds.csv',
  'weak_labels.parquet',
  'weak_labels_round*.parquet',
  'submission*.csv',
)

_LICENSE = {'name': 'CC0-1.0'}

#: Post-push visibility retries (search indexing lags after create).
_VERIFY_ATTEMPTS = 4
_VERIFY_WAIT_SECONDS = 5


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with work, dataset_name, message, dir_mode, staging_dir.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--work',
    default='/kaggle/working',
    help='Kernel scratch dir holding pipeline artifacts.',
  )
  parser.add_argument(
    '--dataset-name',
    default='ah2022_rsna-knee-abnormality-detection',
    help='Single private dataset receiving every artifact.',
  )
  parser.add_argument(
    '--message',
    default=None,
    help='Version note; defaults to a UTC timestamp.',
  )
  parser.add_argument(
    '--dir-mode',
    default='zip',
    choices=['skip', 'zip', 'tar'],
    help='How subdirectories (checkpoints/foldN/) are uploaded.',
  )
  parser.add_argument(
    '--staging-dir',
    default=None,
    help='Scratch used to assemble the snapshot; defaults to /kaggle/tmp '
    '(kept OUT of /kaggle/working to save its 30 GB budget).',
  )
  return parser.parse_args()


def resolve_credentials() -> tuple[str, str]:
  """Resolve Kaggle API credentials for the CLI.

  Priority: env/.env/Kaggle Secrets (via ``get_secret``), then a
  classic ``~/.kaggle/kaggle.json`` file. Credentials are exported to
  ``os.environ`` so child ``kaggle`` calls inherit them.

  Returns:
      (username, key) pair.

  Raises:
      SystemExit: When no usable credentials can be found.
  """
  username = get_secret('KAGGLE_USERNAME') or ''
  key = get_secret('KAGGLE_KEY') or ''
  kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
  if (not username or not key) and kaggle_json.exists():
    payload = json.loads(kaggle_json.read_text())
    username = username or payload.get('username', '')
    key = key or payload.get('key', '')
  if not username or not key:
    raise SystemExit(
      'no Kaggle credentials: add KAGGLE_USERNAME / KAGGLE_KEY secrets '
      'or place ~/.kaggle/kaggle.json'
    )
  os.environ['KAGGLE_USERNAME'] = username
  os.environ['KAGGLE_KEY'] = key
  return username, key


def collect_artifacts(work: str, staging: Path) -> int:
  """Copy every resumable artifact into the staging folder.

  Args:
      work: Kernel scratch dir holding the pipeline outputs.
      staging: Destination folder forming the dataset snapshot.

  Returns:
      Number of files staged (directories count as their contents).

  Raises:
      SystemExit: When nothing could be staged.
  """
  log = get_logger('publish_dataset')
  work_path = Path(work)
  staging.mkdir(parents=True, exist_ok=True)
  staged = 0
  for name in ARTIFACT_DIRS:
    source = work_path / name
    if not source.is_dir():
      continue
    target = staging / name
    shutil.copytree(source, target, dirs_exist_ok=True)
    staged += sum(1 for p in target.rglob('*') if p.is_file())
  for pattern in ARTIFACT_FILES:
    for source in sorted(work_path.glob(pattern)):
      if source.is_file():
        shutil.copy2(source, staging / source.name)
        staged += 1
  if not staged:
    raise SystemExit(f'no artifacts found under {work}; nothing to publish')
  log.info('staged %d files -> %s', staged, staging)
  return staged


def write_metadata(staging: Path, username: str, dataset_name: str) -> None:
  """Produce ``dataset-metadata.json`` via the documented init+edit flow.

  Follows the official CLI tutorial exactly:
  1. ``kaggle datasets init -p <staging>`` generates the template,
  2. placeholders (``INSERT_TITLE_HERE`` / ``INSERT_SLUG_HERE``) are
     patched programmatically instead of by hand,
  3. a license is ensured.

  The template's id shape is preserved: current CLIs emit a bare slug
  (the username is attached server-side), legacy ones use
  ``owner/slug``. When init itself fails, a known-good owner/slug file
  is written as fallback so publishing still works on old CLIs.

  Args:
      staging: Folder that will be uploaded.
      username: Kaggle account owning the dataset.
      dataset_name: Dataset slug (single private dataset).
  """
  log = get_logger('publish_dataset')
  meta_path = staging / 'dataset-metadata.json'
  metadata: dict = {}
  done = _run_kaggle(['datasets', 'init', '-p', str(staging)])
  if done.returncode == 0 and meta_path.exists():
    try:
      metadata = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
      metadata = {}
  from_init = bool(metadata)
  # Preserve whatever id form this CLI vintage expects.
  slug_only = '/' not in str(metadata.get('id', 'INSERT_SLUG_HERE'))
  metadata['title'] = dataset_name
  metadata['id'] = (
    dataset_name if slug_only and from_init else f'{username}/{dataset_name}'
  )
  if not metadata.get('licenses'):
    metadata['licenses'] = [dict(_LICENSE)]
  meta_path.write_text(json.dumps(metadata, indent=2))
  if from_init:
    log.info(
      'metadata initialized via kaggle datasets init (id=%s)', metadata['id']
    )
  else:
    log.warning(
      'kaggle datasets init unavailable; wrote fallback metadata (id=%s)',
      metadata['id'],
    )


def _run_kaggle(args: list[str]) -> subprocess.CompletedProcess:
  """Run one ``kaggle`` CLI command, returning the completed process.

  Args:
      args: Argument vector after the ``kaggle`` program name.

  Returns:
      CompletedProcess for inspection by callers.
  """
  return subprocess.run(
    ['kaggle', *args], capture_output=True, text=True, check=False
  )


def dataset_exists(username: str, dataset_name: str) -> bool:
  """Probe whether the dataset has ever been created.

  Args:
      username: Kaggle account owning the dataset.
      dataset_name: Dataset slug to probe.

  Returns:
      True when it exists, False when it is absent OR invisible to the
      credentials (the current kagglesdk backend reports missing
      datasets as 403 Forbidden instead of the legacy 404 -- it does
      not disclose existence). A follow-up ``create`` still surfaces
      genuine credential problems loudly.

  Raises:
      RuntimeError: On ambiguous CLI failures (network, quota).
  """
  done = _run_kaggle(['datasets', 'status', f'{username}/{dataset_name}'])
  if done.returncode == 0:
    return True
  combined = f'{done.stdout}{done.stderr}'.lower()
  if '404' in combined or '403' in combined or 'not found' in combined:
    return False
  raise RuntimeError(f'kaggle datasets status failed:\n{combined.strip()}')


def _auth_hint(combined: str) -> str:
  """Append a credential hint when a failure looks auth-related.

  Args:
      combined: Raw CLI stdout+stderr text.

  Returns:
      The original text, plus a hint line for auth-style failures.
  """
  lowered = combined.lower()
  if any(
    token in lowered for token in ('401', '403', 'unauthorized', 'forbidden')
  ):
    return (
      f'{combined.strip()}\nhint: verify that KAGGLE_USERNAME / '
      'KAGGLE_KEY secrets belong to the account that owns the dataset'
    )
  return combined


def push(
  staging: Path, username: str, dataset_name: str, message: str, dir_mode: str
) -> None:
  """Create the dataset on first push, else publish a new version.

  Follows the documented API flow: patch
  ``dataset-metadata.json`` -> ``datasets create`` initially and
  ``datasets version -m <msg>`` afterwards.

  Args:
      staging: Folder holding the snapshot plus metadata.
      username: Kaggle account owning the dataset.
      dataset_name: Dataset slug.
      message: Human-readable version note.
      dir_mode: Subdirectory handling passed to the CLI.
  """
  log = get_logger('publish_dataset')
  # -t/--keep-tabular: OOF parquets must upload byte-identical (the CLI
  # otherwise converts tabular files to CSV, per datasets.md).
  keep_tabular = ['-t']

  def _push_cmd(action: str) -> list[str]:
    """Assemble the argv for a create/version call.

    Args:
        action: 'create' or 'version'.

    Returns:
        Full argument vector after the program name.
    """
    cmd = ['datasets', action, *keep_tabular]
    if action == 'version':
      cmd += ['-m', message]
    return [*cmd, '-p', str(staging), '-r', dir_mode]

  if dataset_exists(username, dataset_name):
    done = _run_kaggle(_push_cmd('version'))
    action = 'versioned'
  else:
    done = _run_kaggle(_push_cmd('create'))
    action = 'created'
  if done.returncode != 0:
    combined = f'{done.stdout}{done.stderr}'
    # A lost race between probe and push ("already exists") is safe to
    # retry as a version push.
    if action == 'created' and 'already exist' in combined.lower():
      retry = _run_kaggle(_push_cmd('version'))
      if retry.returncode == 0:
        _verify(username, dataset_name)
        log.info('dataset %s/%s versioned', username, dataset_name)
        return
      combined = f'{retry.stdout}{retry.stderr}'
    raise RuntimeError(
      f'kaggle datasets {action} failed:\n{_auth_hint(combined)}'
    )
  _verify(username, dataset_name)
  log.info('dataset %s/%s %s (%s)', username, dataset_name, action, message)


def _verify(username: str, dataset_name: str) -> None:
  """Confirm the pushed dataset is visible (tutorial's verify step).

  Args:
      username: Kaggle account owning the dataset.
      dataset_name: Dataset slug.
  """
  log = get_logger('publish_dataset')
  reference = f'{username}/{dataset_name}'
  for attempt in range(_VERIFY_ATTEMPTS):
    if dataset_exists(username, dataset_name):
      log.info('verified: kaggle datasets status %s is reachable', reference)
      return
    if attempt < _VERIFY_ATTEMPTS - 1:
      # Server-side indexing can lag tens of seconds after create.
      log.info(
        'waiting for %s to appear in search index (%d/%d)...',
        reference,
        attempt + 1,
        _VERIFY_ATTEMPTS,
      )
      time.sleep(_VERIFY_WAIT_SECONDS)
  log.warning(
    'push accepted but %s not yet visible via status; '
    'check kaggle.com -> Your Work -> Datasets in a moment',
    reference,
  )


def main() -> None:
  """Snapshot $WORK artifacts and push them to the private dataset."""

  args = parse_args()
  log = get_logger('publish_dataset')
  username, _ = resolve_credentials()

  staging = (
    Path(
      args.staging_dir
      or (Path('/kaggle/tmp') if Path('/kaggle/tmp').exists() else '')
      or Path(os.environ.get('TMPDIR', '/tmp'))
    )
    / f'dataset_stage_{args.dataset_name}'
  )
  staging.mkdir(parents=True, exist_ok=True)

  collect_artifacts(args.work, staging)
  write_metadata(staging, username, args.dataset_name)
  stamp = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
  message = args.message or f'artifacts {stamp}'
  push(staging, username, args.dataset_name, message, args.dir_mode)
  slug = args.dataset_name.lower()
  log.info(
    'next kernel: Add Data -> Your Datasets -> %s, then '
    'export PREV_OUTPUT=/kaggle/input/%s',
    args.dataset_name,
    slug,
  )


if __name__ == '__main__':
  main()
