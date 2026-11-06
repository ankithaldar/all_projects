#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''CLI wrapper for forecast-ml.

Usage: forecast-ml train|predict|serve|run --env local

Delegates to ``kedro run`` when kedro is available, otherwise
prints the planned command. Stdlib argparse only.

Optional API-key auth is supported via FORECAST_API_KEY in the
serving app (see README Serving API); no Docker is provided.
'''

import argparse
import shutil
import subprocess
import sys


# CLI command -> Kedro pipeline. ``run`` uses the default
# pipeline (no --pipeline flag); ``serve`` is not a Kedro
# pipeline at all (uvicorn, see _planned_command).
_COMMAND_PIPELINES = {
  'train': 'modeling',
  'predict': 'scoring',
  'run': None,
  'serve': None,
}


def _planned_command(
    command: str, env: str, host: str = '127.0.0.1',
    port: int = 8000) -> list[str]:
  '''Build planned command for a CLI command.

  Args:
    command: CLI command (train/predict/serve/run); any other
      value is treated as a Kedro pipeline name directly.
    env: Kedro environment.
    host: Bind host for ``serve``.
    port: Bind port for ``serve``.

  Returns:
    Planned argv: ``train`` -> modeling pipeline, ``predict``
    -> scoring pipeline, ``run`` -> default pipeline (no
    ``--pipeline``), ``serve`` -> uvicorn serving the FastAPI
    app in ``forecast_ml_framework.serving.app``.
  '''
  if command == 'serve':
    return [
      'uvicorn', 'forecast_ml_framework.serving.app:app',
      '--host', host, '--port', str(port)]
  pipeline = _COMMAND_PIPELINES.get(command, command)
  if pipeline is None:
    return ['kedro', 'run', '--env', env]
  return ['kedro', 'run', '--pipeline', pipeline, '--env', env]


def _run_kedro(
    command: str, env: str, host: str = '127.0.0.1',
    port: int = 8000) -> int:
  '''Run planned command or print it when the binary is absent.'''
  planned = _planned_command(command, env, host=host, port=port)
  if shutil.which(planned[0]) is None:
    print('planned command: ' + ' '.join(planned))
    return 0
  return subprocess.run(planned, check=False).returncode


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
  return _run_kedro(
    args.command, args.env, host=args.host, port=args.port)


if __name__ == '__main__':
  sys.exit(main())
