#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Notebook-friendly stage runner wrapping ``kaggle_run.sh``.

One cell is all a Kaggle session needs:

    %run kaggle_cell.py --stage train --fold 0

Responsibilities:

* validate the requested stage before touching the network;
* guarantee artifact/checkpoint directories exist (fresh sessions);
* forward every extra flag verbatim to ``main.py`` via the shell driver;
* print a compact tail of the run for notebook scrollback;
* self-bootstrap from GitHub: when executed outside a git checkout (e.g.
  the file was pasted into a notebook or mounted via a dataset), read the
  repository coordinates from ``repo_meta.json``, clone the upstream at the
  recorded branch into ``KNEE_REPO_DIR`` (default ``/kaggle/working/repo``)
  and re-exec itself from the fresh clone - zero manual code needed.

Repository coordinates are never hand-written: whenever this file runs
inside a checkout it refreshes ``repo_meta.json`` from the local ``.git``
configuration (remote URL + tracked branch), including worktrees where
``.git`` is a pointer file.

Stages mirror kaggle_run.sh: setup | index | labels | folds | cache |
selftest | train | infer | all.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

VALID_STAGES = ('setup', 'index', 'labels', 'folds', 'cache', 'selftest', 'train', 'infer', 'all')
LOG_TAIL_LINES = 30
META_FILENAME = 'repo_meta.json'
DEFAULT_CLONE_DIR = '/kaggle/working/repo'
TOKEN_ENV_VARS = ('GIT_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN')
CLONE_DEPTH = 1


def _parser() -> argparse.ArgumentParser:
  """Build the stage parser.

  Returns:
      ArgumentParser accepting a stage and passthrough flags.
  """
  parser = argparse.ArgumentParser(description='Kaggle cell stage runner')
  parser.add_argument('--stage', required=True, choices=VALID_STAGES)
  parser.add_argument(
    '--experiment',
    default=None,
    help='Override EXPERIMENT env for this invocation',
  )
  known, extra = parser.parse_known_args()
  # Re-attach passthrough flags for the shell driver.
  known.extra = extra
  return known


def _git(args: list[str], cwd: Path | None = None) -> str | None:
  """Run one git command quietly.

  Args:
      args: Arguments following the program name.
      cwd: Working directory for the command.

  Returns:
      Stripped stdout on success, None on any failure.
  """
  try:
    result = subprocess.run(
      ['git', *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
      return None
    return result.stdout.strip() or None
  except OSError:
    return None


def _resolve_git_dirs(dot_git: Path) -> tuple[Path | None, Path | None]:
  """Resolve (per-worktree gitdir, common gitdir) for any repository shape.

  Handles plain repos (``.git`` directory), linked worktrees (``.git`` file
  containing ``gitdir:``), and the ``commondir`` indirection those use.

  Args:
      dot_git: The ``.git`` path of a checkout (file or directory).

  Returns:
      Tuple ``(linked, common)``; either entry may be None when unresolvable.
      ``linked`` holds per-checkout state (HEAD), ``common`` holds shared
      config.
  """
  if dot_git.is_dir():
    return dot_git, dot_git
  try:
    first_line = dot_git.read_text(encoding='utf-8').splitlines()[0]
  except (OSError, IndexError):
    return None, None
  if not first_line.startswith('gitdir:'):
    return None, None
  linked = Path(first_line.split(':', 1)[1].strip())
  if not linked.is_absolute():
    linked = dot_git.parent / linked
  commondir_file = linked / 'commondir'
  if not commondir_file.exists():
    return linked, linked
  try:
    relative = commondir_file.read_text(encoding='utf-8').strip()
  except OSError:
    return linked, None
  common = (linked / relative).resolve()
  if not (common / 'config').exists():
    return linked, None
  return linked, common


def read_repo_meta(root: Path) -> dict | None:
  """Read repository coordinates straight from the ``.git`` folder.

  Resolution order mirrors git itself: prefer the ``git`` binary (handles
  every layout including worktrees), then fall back to parsing
  ``config``/``HEAD`` files manually.

  Args:
      root: Project directory that should contain ``.git``.

  Returns:
      Mapping with ``url``, ``branch`` and ``remote`` keys, or None when
      unavailable or incomplete.
  """
  dot_git = root / '.git'
  url = _git(['config', '--get', 'remote.origin.url'], cwd=root)
  branch = _git(['rev-parse', '--abbrev-ref', 'HEAD'], cwd=root)
  if url is None or branch in (None, '', 'HEAD'):
    linked, common = _resolve_git_dirs(dot_git)
    if linked is None:
      return None
    if url is None and common is not None:
      url = _parse_config_value(common / 'config', 'remote "origin"', 'url')
    if branch in (None, '', 'HEAD'):
      head_ref = _head_ref(linked)
      branch = head_ref.rsplit('refs/heads/', 1)[-1] if head_ref else None
  if not url or not branch:
    return None
  return {'url': url, 'branch': branch, 'remote': 'origin'}


def _head_ref(git_dir: Path) -> str | None:
  """Read the symbolic HEAD ref from a git directory.

  Args:
      git_dir: Per-worktree or common git directory holding ``HEAD``.

  Returns:
      Full ref string such as ``refs/heads/main``.
  """
  head_file = git_dir / 'HEAD'
  if not head_file.exists():
    return None
  try:
    content = head_file.read_text(encoding='utf-8').strip()
  except OSError:
    return None
  if content.startswith('ref:'):
    return content.split(':', 1)[1].strip()
  return None


def _parse_config_value(
  config_path: Path, section: str, key: str
) -> str | None:
  """Extract the last value of ``key`` within ``section`` of a git config.

  Minimal INI scan avoiding a git-binary dependency.

  Args:
      config_path: Path to a ``config`` file.
      section: Section header without brackets, e.g. ``remote "origin"``.
      key: Key name inside the section.

  Returns:
      Trimmed value string or None when absent.
  """
  try:
    lines = config_path.read_text(encoding='utf-8').splitlines()
  except OSError:
    return None
  in_section = False
  value = None
  for line in lines:
    stripped = line.strip()
    if stripped.startswith('['):
      in_section = stripped[1:-1].strip() == section
    elif in_section and stripped.startswith(f'{key} '):
      _, _, raw = stripped.partition('=')
      value = raw.strip()
  return value


def load_or_refresh_meta(root: Path) -> tuple[dict | None, bool]:
  """Return repo coordinates, refreshing them when running in a checkout.

  Inside a real checkout the fresh values are persisted to
  ``repo_meta.json`` (so future Kaggle copies carry current coordinates).
  Outside one, the committed JSON is loaded instead.

  Args:
      root: Directory of this script.

  Returns:
      Tuple ``(meta, in_checkout)``; ``meta`` is None when neither source
      yields usable coordinates.
  """
  live = read_repo_meta(root)
  if live is not None:
    meta_path = root / META_FILENAME
    try:
      meta_path.write_text(
        json.dumps(live, indent=2, sort_keys=True) + '\n', encoding='utf-8'
      )
    except OSError:
      pass
    return live, True
  meta_path = root / META_FILENAME
  if meta_path.exists():
    try:
      stored = json.loads(meta_path.read_text(encoding='utf-8'))
      if stored.get('url') and stored.get('branch'):
        return stored, False
    except (OSError, json.JSONDecodeError):
      return None, False
  return None, False


def _authenticated_url(url: str) -> str:
  """Inject a token into an https remote when one is configured.

  Args:
      url: Original remote URL.

  Returns:
      URL with credentials embedded, or the input unchanged for ssh URLs
      or missing tokens.
  """
  token = next(
    (os.environ[v] for v in TOKEN_ENV_VARS if os.environ.get(v)), None
  )
  if not token or not url.startswith('https://'):
    return url
  return url.replace('https://', f'https://{token}@', 1)


def _project_root_within(clone_dir: Path) -> Path:
  """Locate the knee project directory inside a freshly cloned repository.

  The upstream hosts multiple competitions, so the project may sit at the
  repository root or one level deeper.

  Args:
      clone_dir: Checkout root.

  Returns:
      Directory containing ``kaggle_cell.py``.

  Raises:
      RuntimeError: When no candidate directory carries the marker files.
  """
  candidates = [clone_dir, *clone_dir.glob('*/')]
  for candidate in candidates:
    if (
      (candidate / 'kaggle_cell.py').exists()
      and (candidate / 'kaggle_run.sh').exists()
    ):
      return candidate
  raise RuntimeError(f'Project markers not found under {clone_dir}')


def ensure_clone(meta: dict) -> Path:
  """Clone/refresh the upstream branch into KNEE_REPO_DIR.

  Idempotent: an existing clone on the right branch is reused
  (fast-forward), otherwise a shallow single-branch clone is performed.

  Args:
      meta: Repository coordinates (url/branch).

  Returns:
      Path to the ready project directory inside the checkout.

  Raises:
      RuntimeError: When cloning or branch checkout fails.
  """
  target = Path(os.environ.get('KNEE_REPO_DIR', DEFAULT_CLONE_DIR))
  marker_branch = (
    _git(['rev-parse', '--abbrev-ref', 'HEAD'], cwd=target)
    if target.exists()
    else None
  )
  if marker_branch == meta['branch']:
    _git(['pull', '--ff-only'], cwd=target)
    return _project_root_within(target)
  if target.exists():
    _git(['fetch', 'origin', meta['branch']], cwd=target)
    if _git(['checkout', meta['branch']], cwd=target) is not None:
      return _project_root_within(target)
    branch = meta['branch']
    raise RuntimeError(f'{target} exists but cannot serve branch {branch!r}')
  cloned = _git(
    [
      'clone',
      '--depth',
      str(CLONE_DEPTH),
      '--branch',
      meta['branch'],
      _authenticated_url(meta['url']),
      str(target),
    ]
  )
  del cloned  # output unused; failures surface via the marker search below
  return _project_root_within(target)


def reexec_from_clone(clone_dir: Path, extra: list[str]) -> int:
  """Re-run this same command inside the fresh clone.

  Args:
      clone_dir: Ready checkout path.
      extra: Original passthrough arguments.

  Returns:
      Child process exit code.
  """
  env = os.environ.copy()
  env['PYTHONPATH'] = os.pathsep.join(
    [str(clone_dir / 'src'), env.get('PYTHONPATH', '')]
  ).rstrip(os.pathsep)
  completed = subprocess.run(
    [
      sys.executable,
      str(clone_dir / 'kaggle_cell.py'),
      *sys.argv[1:],
    ],
    cwd=clone_dir,
    env=env,
    check=False,
  )
  del extra
  return completed.returncode


def _ensure_dirs(root: Path) -> None:
  """Create the writable artifact tree expected by main.py.

  Args:
      root: Project root directory.
  """
  for relative in (
    'artifacts',
    'checkpoints',
    'oof',
    'logs',
  ):
    (root / relative).mkdir(parents=True, exist_ok=True)


def _run_shell(stage: str, experiment: str | None, extra: list[str]) -> int:
  """Delegate to kaggle_run.sh with environment forwarded.

  Args:
      stage: Validated stage name.
      experiment: Optional experiment override exported to the child.
      extra: Additional CLI arguments passed through unchanged.

  Returns:
      Child process exit code.

  Raises:
      FileNotFoundError: When kaggle_run.sh is missing from the project.
  """
  root = Path(__file__).resolve().parent
  script = root / 'kaggle_run.sh'
  if not script.exists():
    raise FileNotFoundError(f'Driver missing: {script}')
  env = os.environ.copy()
  if experiment:
    env['EXPERIMENT'] = experiment
  command = ['bash', str(script), stage, *extra]
  completed = subprocess.run(command, cwd=root, env=env, check=False)
  return completed.returncode


def _print_tail(root: Path) -> None:
  """Echo the newest lightning_logs csv tail for quick inspection.

  Args:
      root: Project root directory.
  """
  logs_dir = root / 'logs'
  logs = sorted(logs_dir.rglob('metrics.csv')) if logs_dir.exists() else []
  if not logs:
    return
  latest = logs[-1]
  lines = latest.read_text(encoding='utf-8').splitlines()
  print(f'--- {latest.name} (last {LOG_TAIL_LINES}) ---')
  for line in lines[-LOG_TAIL_LINES:]:
    print(line)


def main() -> None:
  """Entry point: resolve repo, bootstrap if needed, execute stage."""
  args = _parser()
  root = Path(__file__).resolve().parent
  meta, in_checkout = load_or_refresh_meta(root)
  if not in_checkout and meta is not None and not (root / '.git').exists():
    upstream = meta['url']
    branch = meta['branch']
    print(f'[bootstrap] outside checkout; cloning {upstream} @{branch}')
    clone_dir = ensure_clone(meta)
    sys.exit(reexec_from_clone(clone_dir, list(args.extra)))
  if meta is None:
    print(
      'warning: no .git and no repo_meta.json beside this script; '
      'continuing locally',
      file=sys.stderr,
    )
  _ensure_dirs(root)
  code = _run_shell(args.stage, args.experiment, list(args.extra))
  if code == 0:
    _print_tail(root)
  else:
    print(f'Stage {args.stage!r} failed with exit code {code}', file=sys.stderr)
  sys.exit(code)


if __name__ == '__main__':
  main()
