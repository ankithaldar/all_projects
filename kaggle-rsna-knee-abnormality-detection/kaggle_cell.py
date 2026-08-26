#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Notebook-friendly stage runner wrapping ``kaggle_run.sh``.

One cell is all a Kaggle session needs:

    %run kaggle_cell.py --stage train --fold 0

Responsibilities:

* validate the requested stage before touching the network;
* guarantee artifact/checkpoint directories exist (fresh sessions);
* forward every extra flag verbatim to ``main.py`` via the shell driver;
* print a compact tail of the run for notebook scrollback.

Stages mirror kaggle_run.sh: setup | index | labels | folds | train |
infer | all.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

VALID_STAGES = ('setup', 'index', 'labels', 'folds', 'train', 'infer', 'all')
LOG_TAIL_LINES = 30


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
  logs = (
    sorted((root / 'logs').rglob('metrics.csv'))
    if (root / 'logs').exists()
    else []
  )
  if not logs:
    return
  latest = logs[-1]
  lines = latest.read_text(encoding='utf-8').splitlines()
  print(f'--- {latest.name} (last {LOG_TAIL_LINES}) ---')
  for line in lines[-LOG_TAIL_LINES:]:
    print(line)


def main() -> None:
  """Entry point: validate, prepare, execute, report."""
  args = _parser()
  root = Path(__file__).resolve().parent
  _ensure_dirs(root)
  code = _run_shell(args.stage, args.experiment, list(args.extra))
  if code == 0:
    _print_tail(root)
  else:
    print(f'Stage {args.stage!r} failed with exit code {code}', file=sys.stderr)
  sys.exit(code)


if __name__ == '__main__':
  main()
