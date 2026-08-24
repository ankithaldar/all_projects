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
    choices=['skip', 'zip', 'tar', 'gzip'],
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
  """Write ``dataset-metadata.json`` expected by the Kaggle CLI.

  Mirrors the ``kaggle datasets init`` + manual-edit flow from the
  Kaggle how-to, minus the redundant round trip.

  Args:
      staging: Folder that will be uploaded.
      username: Kaggle account owning the dataset.
      dataset_name: Dataset slug (single private dataset).
  """
  metadata = {
    'title': dataset_name,
    'id': f'{username}/{dataset_name}',
    'licenses': [_LICENSE],
  }
  (staging / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2))


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
      True when it exists, False on a definitive 404.

  Raises:
      RuntimeError: On ambiguous CLI failures (auth, network, quota).
  """
  done = _run_kaggle(['datasets', 'status', f'{username}/{dataset_name}'])
  if done.returncode == 0:
    return True
  combined = f'{done.stdout}{done.stderr}'
  if '404' in combined or 'not found' in combined.lower():
    return False
  raise RuntimeError(f'kaggle datasets status failed:\n{combined.strip()}')


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
  if dataset_exists(username, dataset_name):
    done = _run_kaggle(
      [
        'datasets',
        'version',
        '-m',
        message,
        '-p',
        str(staging),
        '--dir-mode',
        dir_mode,
      ]
    )
    action = 'versioned'
  else:
    done = _run_kaggle(
      ['datasets', 'create', '-p', str(staging), '--dir-mode', dir_mode]
    )
    action = 'created'
  if done.returncode != 0:
    combined = f'{done.stdout}{done.stderr}'
    # A lost race between probe and push ("already exists") is safe to
    # retry as a version push.
    if action == 'created' and 'already exist' in combined.lower():
      retry = _run_kaggle(
        [
          'datasets',
          'version',
          '-m',
          message,
          '-p',
          str(staging),
          '--dir-mode',
          dir_mode,
        ]
      )
      if retry.returncode == 0:
        log.info('dataset %s/%s %s', username, dataset_name, 'versioned')
        return
      combined = f'{retry.stdout}{retry.stderr}'
    raise RuntimeError(f'kaggle datasets {action} failed:\n{combined.strip()}')
  log.info('dataset %s/%s %s (%s)', username, dataset_name, action, message)


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
