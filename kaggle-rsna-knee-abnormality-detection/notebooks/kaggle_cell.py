#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RSNA Knee Abnormality Detection -- Kaggle one-cell bootstrap.

Paste this whole file into the FIRST cell of every Kaggle kernel and
run it. Edit ``REPO_URL`` (and optionally ``GIT_REF``) once; everything
else is automatic:

  1. pulls GITHUB_TOKEN / KAGGLE_USERNAME / KAGGLE_KEY from Kaggle
     Secrets (plain %%bash cells cannot use UserSecretsClient),
  2. clones the pipeline repo into /kaggle/working/repo,
  3. points PREV_OUTPUT at the private artifact dataset when it is
     attached, so finished folds skip and interrupted ones resume,
  4. dispatches ``scripts/kaggle_run.sh $STAGE`` (deps install, GPU
     precision fallback and fold resume are handled inside), streaming
     output into the cell,
  5. with AUTO_PUBLISH=1 pushes every artifact to the single private
     dataset afterwards, ready for the next kernel.

Per-kernel workflow: change ``STAGE`` / ``FOLDS_LIST`` only.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ------------------------------- CONFIG -------------------------------------
REPO_URL = 'https://github.com/ankithaldar/all_projects_02.git'  # <-- EDIT ONCE
GIT_REF = 'competitions/kaggle-rsna-knee-abnormality-detection'
# STAGE: volumes|folds|weak-labels|teacher|student|self-train|infer|
#        blend|publish
STAGE = 'student'
EXP = 'configs/experiment/student_2p5d_effnetv2.yaml'
FOLDS_LIST = None  # '0,1' then '2,3' per kernel; None = config default
TIME_BUDGET_HOURS = 11  # clean stop before Kaggle's hard 12 h kill
AUTO_PUBLISH = 1  # push artifacts to the private dataset post-stage
DATA_ROOT = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
DATASET_NAME = 'ah2022_rsna-knee-abnormality-detection'
# -----------------------------------------------------------------------------


def _secret(name: str) -> str:
  """Fetch a Kaggle Secret, falling back to the process environment.

  Args:
      name: Secret/environment variable name.

  Returns:
      Secret value, or '' when unavailable.
  """
  try:  # pragma: no cover - only importable inside Kaggle kernels
    # pylint: disable=import-outside-toplevel
    from kaggle_secrets import UserSecretsClient

    return UserSecretsClient().get_secret(name) or ''
  except Exception:  # pylint: disable=broad-exception-caught
    return os.environ.get(name, '')


for _key in ('GITHUB_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY'):
  _value = _secret(_key)
  if _value:
    os.environ[_key] = _value

WORK = '/kaggle/working'
REPO_DIR = f'{WORK}/repo'


def _clone(url: str, extra_args: list[str]) -> bool:
  """Clone ``url`` into REPO_DIR, tolerating leftover partial dirs.

  Args:
      url: Resolved remote URL (token-injected for private repos).
      extra_args: Extra git arguments (e.g. branch pinning).

  Returns:
      True on success, False otherwise.
  """
  shutil.rmtree(REPO_DIR, ignore_errors=True)
  cmd = ['git', 'clone', '--depth', '1', *extra_args, url, REPO_DIR]
  print('==>', ' '.join(cmd[:2]), url.split('@')[-1])
  return subprocess.run(cmd, check=False).returncode == 0


_url = REPO_URL
_token = os.environ.get('GITHUB_TOKEN', '')
if _token and _url.startswith('https://'):
  _url = _url.replace('https://', f'https://x-access-token:{_token}@')

if not _clone(_url, ['-b', GIT_REF]):
  print(f'==> branch {GIT_REF!r} not found; trying default branch')
  if not _clone(_url, []):
    raise SystemExit(
      'git clone failed -- check REPO_URL, GITHUB_TOKEN secret and '
      'repository visibility'
    )

# Auto-restore: adopt the artifact dataset when it is attached.
_prev = Path('/kaggle/input') / DATASET_NAME
_env = {
  'DATA_ROOT': DATA_ROOT,
  'WORK': WORK,
  'EXP': EXP,
  'DATASET_NAME': DATASET_NAME,
  'AUTO_PUBLISH': str(AUTO_PUBLISH),
  'TIME_BUDGET_HOURS': str(TIME_BUDGET_HOURS),
}
if FOLDS_LIST:
  _env['FOLDS_LIST'] = FOLDS_LIST
if _prev.is_dir():
  _env['PREV_OUTPUT'] = str(_prev)
  print(f'==> restoring prior state from {_prev}')
else:
  print(f'==> {DATASET_NAME} not attached (fresh start?)')

_process = subprocess.Popen(
  ['bash', f'{REPO_DIR}/kaggle-rsna-knee-abnormality-detection/scripts/kaggle_run.sh', STAGE],
  env={**os.environ, **_env},
  stdout=subprocess.PIPE,
  stderr=subprocess.STDOUT,
  text=True,
  bufsize=1,
)
assert _process.stdout is not None
for _line in _process.stdout:
  print(_line, end='')
_code = _process.wait()
print(f'\n==> stage {STAGE!r} exited with code {_code}')
sys.exit(_code)
