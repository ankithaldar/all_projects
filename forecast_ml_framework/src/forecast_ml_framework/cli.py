#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''CLI wrapper for forecast-ml.

Usage: forecast-ml train|predict|serve|run --env local

Delegates to ``kedro run`` when kedro is available, otherwise
prints the planned command. Stdlib argparse only.

No auth or Docker is provided (out of scope for Phase 1).
'''

import argparse
import shutil
import subprocess
import sys


def _planned_command(pipeline: str, env: str) -> list[str]:
  '''Build planned kedro command.'''
  return ['kedro', 'run', '--pipeline', pipeline, '--env', env]


def _run_kedro(pipeline: str, env: str) -> int:
  '''Run kedro pipeline or print planned command.'''
  command = _planned_command(pipeline, env)
  if shutil.which('kedro') is None:
    print('planned command: ' + ' '.join(command))
    return 0
  return subprocess.run(command, check=False).returncode


def _build_parser() -> argparse.ArgumentParser:
  '''Build CLI argument parser.'''
  parser = argparse.ArgumentParser(
    prog='forecast-ml',
    description='Train, predict, or serve a forecast model.',
  )
  sub = parser.add_subparsers(dest='command', required=True)
  for name in ('train', 'predict', 'serve', 'run'):
    child = sub.add_parser(name, help=f'Run the {name} pipeline.')
    child.add_argument(
      '--env', default='local', help='Kedro environment.')
    child.add_argument(
      '--artifact', default=None, help='Model artifact path.')
    child.add_argument('--host', default='127.0.0.1')
    child.add_argument('--port', type=int, default=8000)
  return parser


def main(argv: list[str] | None = None) -> int:
  '''Run CLI entry point.

  Args:
    argv: Argument list, defaults to sys.argv.

  Returns:
    Process exit code.
  '''
  args = _build_parser().parse_args(argv)
  return _run_kedro(args.command, args.env)


if __name__ == '__main__':
  sys.exit(main())
